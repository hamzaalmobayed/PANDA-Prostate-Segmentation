
from __future__ import annotations

"""
Streaming paper-results stage for the final PANDA prostate pipeline.

Validation/Test WSI originals are retained, so this module does NOT need saved Test patches.
It streams 512x512 patches directly from WSI, accumulates:
- per-patch confusion counts
- per-level WSI probability maps
- three-level fusion map
- threshold sweep on Validation
- Test WSI metrics, per-slide stats, provider stats, bootstrap CIs
- figures and qualitative error maps

CAMELYON-specific ITC/micro/macro FROC is intentionally not copied to prostate.
"""

import json, math, gc, os, threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch

import panda_camelyon_style_pipeline as core
from panda_final_pipeline import FinalConfig


THRESHOLDS = np.round(np.arange(0.30, 0.701, 0.01), 2)
BOOTSTRAP_ITERS = 1000


def root(cfg: FinalConfig):
    p=cfg.work/"paper_results"
    for s in ["tables","figures","qualitative","wsi_cache","stats"]:
        (p/s).mkdir(parents=True,exist_ok=True)
    return p


def sdiv(a,b): return float(a)/float(b) if b else 0.0

def metrics(tp,fp,tn,fn):
    return {
        "tp":int(tp),"fp":int(fp),"tn":int(tn),"fn":int(fn),
        "dice":sdiv(2*tp,2*tp+fp+fn),
        "iou":sdiv(tp,tp+fp+fn),
        "precision":sdiv(tp,tp+fp),
        "recall":sdiv(tp,tp+fn),
        "specificity":sdiv(tn,tn+fp),
        "accuracy":sdiv(tp+tn,tp+fp+tn+fn),
    }

def counts(pred, gt, valid=None):
    p=pred.astype(bool); g=gt.astype(bool)
    if valid is not None:
        v=valid.astype(bool); p=p[v]; g=g[v]
    return (int((p&g).sum()),int((p&~g).sum()),int((~p&~g).sum()),int((~p&g).sum()))

def save_json(p,obj):
    p.parent.mkdir(parents=True,exist_ok=True)
    json.dump(obj,open(p,"w",encoding="utf-8"),indent=2,ensure_ascii=False,default=str)


def make_training_curves(cfg):
    lr=json.load(open(cfg.work/"latest_run.json","r",encoding="utf-8"))
    h=pd.read_csv(Path(lr["run_dir"])/"history.csv")
    fig,ax=plt.subplots(figsize=(8,5))
    ax.plot(h.epoch,h.train_loss,label="Training Loss")
    ax.plot(h.epoch,h.val_loss,label="Validation Loss")
    ax2=ax.twinx(); ax2.plot(h.epoch,h.val_dice,label="Validation Dice")
    best=int(h.loc[h.val_dice.idxmax(),"epoch"]); ax.axvline(best,ls="--")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax2.set_ylabel("Dice")
    lines=ax.get_lines()+ax2.get_lines(); ax.legend(lines,[x.get_label() for x in lines],loc="best")
    ax.set_title("Training Dynamics")
    fig.tight_layout(); fig.savefig(root(cfg)/"figures"/"01_training_curves.png",dpi=300,bbox_inches="tight"); plt.close(fig)


def make_pipeline_figure(cfg):
    labels=["PANDA WSI + GT","WSI split","Tissue mask","L0/L1/L2","512 patches",
            "6 buckets","EffNet-B4 + KMeans","Vahadane + Aug","U-Net Eff-B4","WSI results"]
    fig,ax=plt.subplots(figsize=(13,4)); ax.axis("off")
    xs=np.linspace(.04,.96,len(labels))
    for i,(x,l) in enumerate(zip(xs,labels)):
        ax.text(x,.5,l,ha="center",va="center",fontsize=8,bbox=dict(boxstyle="round",fc="white",ec="black"))
        if i<len(labels)-1: ax.annotate("",xy=(xs[i+1]-.03,.5),xytext=(x+.03,.5),arrowprops=dict(arrowstyle="->"))
    ax.set_title("PANDA Prostate Cancer Segmentation Pipeline")
    fig.tight_layout(); fig.savefig(root(cfg)/"figures"/"00_pipeline_overview.png",dpi=300,bbox_inches="tight"); plt.close(fig)


def _tissue_reader(reader, level):
    try:
        return reader.tissue_mask("morphological",resolution=level,units="level")
    except Exception:
        return None


def _valid_coords(cfg: FinalConfig, row: pd.Series, lv: int):
    """
    Stream coordinates from tissue only. Does not save patch files.
    """
    im=core.SlideScaleReader(row.image_path)
    ds=float(cfg.logical_downsamples[lv]); field0=int(cfg.patch_size*ds); stride0=int(cfg.stride*ds)
    thumb,meta=im.thumbnail_at_downsample(max(ds,16.0),cfg.target_max_size)
    tissue,_=core.generate_tissue_mask(thumb,cfg)
    eff=float(meta.get("effective_ds",max(ds,16.0)))
    w0,h0=im.dimensions
    coords=[]
    for y0 in range(0,max(1,h0-field0+1),stride0):
        for x0 in range(0,max(1,w0-field0+1),stride0):
            tf=core.tissue_fraction_for_patch(tissue,x0,y0,ds,cfg.patch_size,eff)
            if tf>=cfg.tissue_min_frac: coords.append((x0,y0))
    im.close()
    return coords


def _normalize(cfg, rgb, proc):
    return proc.transform(rgb) if proc is not None else rgb



def _paper_inference_batch_size(cfg, device):
    """
    Performance-only setting for dense WSI inference.
    Keeps the trained model/data/evaluation unchanged.
    On a 6-GB RTX 3050, batch=8 is a conservative fast default.
    Override with PANDA_WSI_BATCH if desired.
    """
    env = os.environ.get("PANDA_WSI_BATCH", "").strip()
    if env:
        try:
            return max(1, int(env))
        except Exception:
            pass
    if device.type != "cuda":
        return 1
    try:
        total_gb = torch.cuda.get_device_properties(device).total_memory / (1024**3)
        if total_gb >= 10:
            return 24
        if total_gb >= 7:
            return 20
        return 16
    except Exception:
        return 8


def _forward_probability_batch(cfg, model, device, rgbs, coords_norm, batch_size):
    """
    Batched CUDA inference with automatic OOM fallback.
    Returns one float32 probability map per input patch.
    """
    if not rgbs:
        return []
    current_bs = min(int(batch_size), len(rgbs))
    while True:
        try:
            outputs = []
            for start in range(0, len(rgbs), current_bs):
                ims = rgbs[start:start+current_bs]
                cxy = coords_norm[start:start+current_bs]

                arr = np.stack(ims, axis=0)
                x = torch.from_numpy(arr).permute(0,3,1,2).float().div_(255.0)
                if device.type == "cuda":
                    try:
                        x = x.pin_memory()
                    except Exception:
                        pass
                x = x.to(device, non_blocking=(device.type=="cuda"))
                if getattr(cfg, "channels_last", False) and device.type == "cuda":
                    x = x.contiguous(memory_format=torch.channels_last)

                c = torch.as_tensor(cxy, dtype=torch.float32)
                if device.type == "cuda":
                    try:
                        c = c.pin_memory()
                    except Exception:
                        pass
                c = c.to(device, non_blocking=(device.type=="cuda"))

                with torch.amp.autocast("cuda", enabled=cfg.amp and device.type=="cuda"):
                    p = torch.sigmoid(model(x, c))
                p = p[:,0].detach().float().cpu().numpy()
                outputs.extend([p[i] for i in range(p.shape[0])])

                del x, c, p
            return outputs
        except torch.cuda.OutOfMemoryError:
            if device.type != "cuda" or current_bs <= 1:
                raise
            torch.cuda.empty_cache()
            current_bs = max(1, current_bs // 2)
            print(f"[WSI INFERENCE] CUDA OOM -> automatic batch fallback to {current_bs}", flush=True)


# ---------------- V20 lightning WSI execution helpers ----------------

_WSI_STAIN_TLS = threading.local()


def _thread_vahadane(rgb: np.ndarray, target_path: str) -> np.ndarray:
    """
    One Vahadane processor per worker thread.
    Prevents shared mutable normalizer state while allowing CPU stain work
    to run in parallel. Target/reference remains the SAME Train-derived file.
    """
    proc = getattr(_WSI_STAIN_TLS, "proc", None)
    key = getattr(_WSI_STAIN_TLS, "target_path", None)
    if proc is None or key != target_path:
        target = np.array(Image.open(target_path).convert("RGB"), dtype=np.uint8, copy=True)
        proc = core.VahadaneProcessor(target)
        _WSI_STAIN_TLS.proc = proc
        _WSI_STAIN_TLS.target_path = target_path
    return proc.transform(rgb)


def _normalize_many_fast(cfg, rgbs, fallback_proc, workers: int):
    if not rgbs:
        return []
    # Same Vahadane transform; only execution is parallelized.
    if fallback_proc is None:
        return rgbs
    target_path = str(cfg.stain_target_path)
    workers = max(1, int(workers))
    if workers == 1 or len(rgbs) < 2:
        return [fallback_proc.transform(x) for x in rgbs]
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="vahadane") as ex:
        return list(ex.map(lambda x: _thread_vahadane(x, target_path), rgbs))


def _shared_tissue_for_wsi(cfg, im):
    """
    Build the low-resolution tissue mask ONCE per WSI.
    V18/V19 rebuilt the same max(ds,16) thumbnail for L0/L1/L2 separately.
    """
    base_ds = 16.0
    thumb, meta = im.thumbnail_at_downsample(base_ds, cfg.target_max_size)
    tissue, _ = core.generate_tissue_mask(thumb, cfg)
    eff = float(meta.get("effective_ds", base_ds))
    return tissue, eff


def _valid_coords_vectorized(cfg, dimensions, tissue, tissue_eff: float, lv: int):
    """
    Exact regular-grid tissue gating using an integral image.
    Same coordinates and same tissue_min_frac criterion as
    core.tissue_fraction_for_patch, but evaluates the whole grid vectorially.
    """
    w0, h0 = dimensions
    ds = float(cfg.logical_downsamples[lv])
    field0 = int(cfg.patch_size * ds)
    stride0 = int(cfg.stride * ds)

    xs0 = np.arange(0, max(1, w0-field0+1), stride0, dtype=np.int64)
    ys0 = np.arange(0, max(1, h0-field0+1), stride0, dtype=np.int64)
    if len(xs0) == 0 or len(ys0) == 0:
        return []

    # Python round() and np.rint() both use bankers rounding.
    xt = np.rint(xs0 / tissue_eff).astype(np.int64)
    yt = np.rint(ys0 / tissue_eff).astype(np.int64)
    size = max(1, int(round((cfg.patch_size * ds) / tissue_eff)))

    mask = (tissue > 0).astype(np.uint8)
    Ht, Wt = mask.shape
    # Integral image with 1-pixel zero border.
    ii = np.pad(mask.astype(np.uint64), ((1,0),(1,0))).cumsum(0).cumsum(1)

    x1 = np.clip(xt, 0, Wt)
    y1 = np.clip(yt, 0, Ht)
    x2 = np.minimum(Wt, x1 + size)
    y2 = np.minimum(Ht, y1 + size)

    # Broadcast rectangle sums over all Y,X grid locations.
    S = ii[y2[:,None], x2[None,:]] - ii[y1[:,None], x2[None,:]] \
        - ii[y2[:,None], x1[None,:]] + ii[y1[:,None], x1[None,:]]
    A = (y2-y1)[:,None] * (x2-x1)[None,:]
    frac = np.divide(S, A, out=np.zeros_like(S, dtype=np.float64), where=A>0)
    good = frac >= float(cfg.tissue_min_frac)

    gy, gx = np.nonzero(good)
    return [(int(xs0[ix]), int(ys0[iy])) for iy,ix in zip(gy,gx)]


def _wsi_prep_workers():
    env = os.environ.get("PANDA_WSI_PREP_WORKERS", "").strip()
    if env:
        try:
            return max(1, int(env))
        except Exception:
            pass
    # i5-13420H / 16GB RAM: 4 parallel stain workers is aggressive but safe.
    return 4


def _wsi_prep_chunk(infer_bs: int):
    env = os.environ.get("PANDA_WSI_PREP_CHUNK", "").strip()
    if env:
        try:
            return max(infer_bs, int(env))
        except Exception:
            pass
    # Prepare several inference batches together so Vahadane can parallelize.
    return max(infer_bs, infer_bs * 4)

@torch.inference_mode()
def reconstruct_one(cfg: FinalConfig, row: pd.Series, model, device, stain_proc,
                    split_name: str) -> Dict[str,Any]:
    # SAME cache name/schema as V18/V19: completed slides are reused, not rerun.
    cache=root(cfg)/"wsi_cache"/f"{split_name}_{row.image_id}_ds{int(cfg.paper_eval_downsample)}_validsupport_v2.npz"
    if cache.exists():
        with np.load(cache) as z:
            return {k:z[k] for k in z.files}

    im=core.SlideScaleReader(row.image_path)
    mr=core.MaskScaleReader(row.mask_path,im.dimensions)
    w0,h0=im.dimensions
    eds=float(cfg.paper_eval_downsample)
    W=max(1,int(math.ceil(w0/eds)))
    H=max(1,int(math.ceil(h0/eds)))

    gt_sum=np.zeros((H,W),np.float32)
    gt_valid_sum=np.zeros((H,W),np.float32)

    level_maps={}
    level_support={}
    patch_counts={}
    infer_bs=_paper_inference_batch_size(cfg, device)
    prep_workers=_wsi_prep_workers()
    prep_chunk=_wsi_prep_chunk(infer_bs)

    # Major CPU saving: tissue thumbnail/mask once for the whole WSI.
    tissue_shared, tissue_eff = _shared_tissue_for_wsi(cfg, im)

    for lv in cfg.levels:
        sp=np.zeros((H,W),np.float32)
        sw=np.zeros((H,W),np.float32)
        ptp=pfp=ptn=pfn=0
        ds=float(cfg.logical_downsamples[lv])

        # Same regular coordinate grid and tissue rule, evaluated vectorially.
        coords=_valid_coords_vectorized(cfg, im.dimensions, tissue_shared, tissue_eff, lv)

        outer = tqdm(range(0, len(coords), prep_chunk),
                     total=(len(coords)+prep_chunk-1)//prep_chunk,
                     desc=f"{split_name} {row.image_id} L{lv}",
                     leave=False, unit="chunk")

        for st in outer:
            chunk_coords=coords[st:st+prep_chunk]

            raw_rgbs=[]
            metas=[]
            cnorm=[]
            for x0,y0 in chunk_coords:
                rgb,_=im.read_at_downsample(x0,y0,ds,cfg.patch_size)
                raw,_=mr.read_raw(x0,y0,ds,cfg.patch_size)
                gt=core.raw_to_binary_mask(raw,str(row.provider))>0
                valid_patch=core.raw_valid_tissue_mask(raw,str(row.provider))>0
                if not valid_patch.any():
                    continue
                field0=cfg.patch_size*ds
                raw_rgbs.append(np.ascontiguousarray(rgb,dtype=np.uint8))
                metas.append((x0,y0,gt,valid_patch))
                cnorm.append([(x0+field0/2)/max(w0,1),
                              (y0+field0/2)/max(h0,1)])

            if not raw_rgbs:
                continue

            # Same Train-derived Vahadane reference, now parallel across CPU cores.
            rgbs=_normalize_many_fast(cfg,raw_rgbs,stain_proc,prep_workers)

            # CUDA batches; default tries 16 on the 6GB RTX 3050 and automatically
            # falls back on OOM without changing predictions/evaluation.
            probs=_forward_probability_batch(cfg,model,device,rgbs,cnorm,infer_bs)

            for prob,(x0,y0,gt,valid_patch) in zip(probs,metas):
                pc=counts(prob>=0.5,gt,valid_patch)
                ptp+=pc[0]; pfp+=pc[1]; ptn+=pc[2]; pfn+=pc[3]

                ow=max(1,int(round(cfg.patch_size*ds/eds)))
                pr=cv2.resize(prob.astype(np.float32),(ow,ow),interpolation=cv2.INTER_LINEAR)
                gr=cv2.resize(gt.astype(np.uint8),(ow,ow),interpolation=cv2.INTER_NEAREST).astype(np.float32)
                vr=cv2.resize(valid_patch.astype(np.uint8),(ow,ow),interpolation=cv2.INTER_NEAREST).astype(np.float32)

                ox=int(round(x0/eds)); oy=int(round(y0/eds))
                x2=min(W,ox+ow); y2=min(H,oy+ow)
                if x2>ox and y2>oy:
                    hh=y2-oy; ww=x2-ox
                    vloc=vr[:hh,:ww]>0
                    spr=sp[oy:y2,ox:x2]; swr=sw[oy:y2,ox:x2]
                    spr[vloc]+=pr[:hh,:ww][vloc]
                    swr[vloc]+=1
                    gs=gt_sum[oy:y2,ox:x2]; gv=gt_valid_sum[oy:y2,ox:x2]
                    gs[vloc]+=gr[:hh,:ww][vloc]
                    gv[vloc]+=1

            del raw_rgbs,rgbs,metas,cnorm,probs

        p=np.zeros_like(sp)
        support=sw>0
        p[support]=sp[support]/sw[support]
        level_maps[lv]=p
        level_support[lv]=support
        patch_counts[lv]=np.array([ptp,pfp,ptn,pfn],dtype=np.int64)

    valid=gt_valid_sum>0
    gt=np.zeros((H,W),bool)
    gt[valid]=(gt_sum[valid]/gt_valid_sum[valid])>=0.5

    fusion_sum=np.zeros((H,W),np.float32)
    fusion_n=np.zeros((H,W),np.float32)
    for lv in cfg.levels:
        s=level_support[lv]
        fusion_sum[s]+=level_maps[lv][s]
        fusion_n[s]+=1
    fusion=np.zeros((H,W),np.float32)
    fs=fusion_n>0
    fusion[fs]=fusion_sum[fs]/fusion_n[fs]
    eval_valid=valid & fs

    tmp=cache.with_name(cache.name+".tmp")
    with open(tmp,"wb") as fh:
        np.savez(
            fh,
            gt=gt.astype(np.uint8),
            valid=eval_valid.astype(np.uint8),
            annotated_valid=valid.astype(np.uint8),
            fusion_support=fs.astype(np.uint8),
            fusion=fusion.astype(np.float16),
            **{f"prob_L{lv}":level_maps[lv].astype(np.float16) for lv in cfg.levels},
            **{f"support_L{lv}":level_support[lv].astype(np.uint8) for lv in cfg.levels},
            **{f"patch_counts_L{lv}":patch_counts[lv] for lv in cfg.levels}
        )
        fh.flush(); os.fsync(fh.fileno())
    os.replace(tmp,cache)

    im.close(); mr.close()
    return dict(np.load(cache))

def threshold_validation(cfg, val_rows, model, device, stain_proc):
    # Stream threshold counts slide-by-slide. V18 retained all reconstructed
    # validation WSIs in RAM; with ~1.5k WSIs that can force Windows paging and
    # progressively slow the run. This produces the same pooled threshold metrics.
    accum={float(thr):[0,0,0,0] for thr in THRESHOLDS}

    outer=tqdm(total=len(val_rows), desc="VAL full-WSI threshold sweep", unit="WSI")
    for _,r in val_rows.iterrows():
        z=reconstruct_one(cfg,r,model,device,stain_proc,"val")
        fusion=z["fusion"].astype(np.float32)
        gt=z["gt"].astype(bool)
        valid=z["valid"].astype(bool)
        for thr in THRESHOLDS:
            c=counts(fusion>=thr,gt,valid)
            a=accum[float(thr)]
            for i in range(4):
                a[i]+=c[i]
        del z, fusion, gt, valid
        gc.collect()
        outer.update(1)
    outer.close()

    rows=[]
    for thr in THRESHOLDS:
        tp,fp,tn,fn=accum[float(thr)]
        rows.append({"threshold":float(thr),**metrics(tp,fp,tn,fn)})

    df=pd.DataFrame(rows)
    df.to_csv(root(cfg)/"tables"/"threshold_sensitivity_validation.csv",index=False)
    best=df.loc[df.dice.idxmax()].to_dict()
    save_json(root(cfg)/"tables"/"best_threshold.json",best)

    fig,ax=plt.subplots(figsize=(8,5))
    for m in ["dice","precision","recall"]:
        ax.plot(df.threshold,df[m],label=m.capitalize())
    ax.axvline(best["threshold"],ls="--")
    ax.set_xlabel("Threshold"); ax.set_ylabel("Score")
    ax.set_title("Validation Threshold Sensitivity"); ax.legend()
    fig.tight_layout()
    fig.savefig(root(cfg)/"figures"/"02_threshold_sensitivity.png",dpi=300,bbox_inches="tight")
    plt.close(fig)
    return float(best["threshold"])

def confusion_fig(name,m,out):
    mat=np.array([[m["tn"],m["fp"]],[m["fn"],m["tp"]]])
    total=max(1,mat.sum()); fig,ax=plt.subplots(figsize=(4.5,4))
    ax.imshow(mat,cmap="Blues")
    for (i,j),v in np.ndenumerate(mat): ax.text(j,i,f"{int(v):,}\n{100*v/total:.1f}%",ha="center",va="center")
    ax.set_xticks([0,1],["Normal","Tumor"]); ax.set_yticks([0,1],["Normal","Tumor"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(name)
    fig.tight_layout(); fig.savefig(out,dpi=300,bbox_inches="tight"); plt.close(fig)


def bootstrap(ps,key,seed=42):
    rng=np.random.default_rng(seed); vals={m:[] for m in ["dice","iou","precision","recall","specificity","accuracy"]}
    n=len(ps)
    for _ in range(BOOTSTRAP_ITERS):
        b=ps.iloc[rng.integers(0,n,size=n)]
        mm=metrics(int(b[f"{key}_tp"].sum()),int(b[f"{key}_fp"].sum()),int(b[f"{key}_tn"].sum()),int(b[f"{key}_fn"].sum()))
        for m in vals: vals[m].append(mm[m])
    return {m:(float(np.percentile(a,2.5)),float(np.percentile(a,97.5))) for m,a in vals.items()}


def qualitative(cfg,row,z,thr,tag):
    gt=z["gt"].astype(bool); valid=z["valid"].astype(bool); prob=z["fusion"].astype(np.float32); pred=prob>=thr
    sr=core.SlideScaleReader(row.image_path)
    rgb=np.asarray(sr.slide.get_thumbnail((gt.shape[1],gt.shape[0])).convert("RGB").resize((gt.shape[1],gt.shape[0])))
    sr.close()
    gtov=rgb.copy(); gtov[gt&valid]=(0.6*gtov[gt&valid]+0.4*np.array([0,255,0])).astype(np.uint8)
    prov=rgb.copy(); prov[pred&valid]=(0.6*prov[pred&valid]+0.4*np.array([255,0,0])).astype(np.uint8)
    err=np.zeros_like(rgb)+255
    err[(~gt)&(~pred)&valid]=[230,230,230]; err[gt&pred&valid]=[60,170,90]
    err[gt&(~pred)&valid]=[240,190,40]; err[(~gt)&pred&valid]=[210,60,60]
    arr=[("Original H&E",rgb),("Ground Truth",gt*255),("Prediction",pred*255),
         ("GT Overlay",gtov),("Prediction Overlay",prov),("Error Map",err)]
    fig,axs=plt.subplots(2,3,figsize=(12,8))
    for ax,(t,im) in zip(axs.flat,arr): ax.imshow(im,cmap="gray" if im.ndim==2 else None); ax.set_title(t); ax.axis("off")
    fig.suptitle(f"{tag.capitalize()} test WSI: {row.image_id}"); fig.tight_layout()
    fig.savefig(root(cfg)/"qualitative"/f"{tag}_{row.image_id}_six_panel.png",dpi=300,bbox_inches="tight"); plt.close(fig)
    Image.fromarray((np.clip(prob,0,1)*255).astype(np.uint8)).save(root(cfg)/"qualitative"/f"{tag}_{row.image_id}_probability.png")
    Image.fromarray((pred*255).astype(np.uint8)).save(root(cfg)/"qualitative"/f"{tag}_{row.image_id}_mask.png")
    Image.fromarray(prov).save(root(cfg)/"qualitative"/f"{tag}_{row.image_id}_overlay.png")
    Image.fromarray(err).save(root(cfg)/"qualitative"/f"{tag}_{row.image_id}_error_map.png")






def _image_case_dir(root_dir, image_id: str, prefix: str):
    """
    Create a dedicated subfolder per image, e.g.
      .../test_visuals_good_live/good_<image_id>/
      .../test_visuals_bad_review/bad_<image_id>/
    """
    d = Path(root_dir) / f"{prefix}_{str(image_id)}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _good_test_min_dice():
    """
    Save a live preview only if fusion Dice is good enough.
    Default = 0.80, override with environment variable:
        PANDA_GOOD_TEST_MIN_DICE=0.85
    """
    try:
        return float(os.environ.get("PANDA_GOOD_TEST_MIN_DICE", "0.80"))
    except Exception:
        return 0.80


def save_good_test_preview(cfg, row, z, thr, rr):
    """
    During TEST, save a visual preview only for slides whose fusion Dice is
    close enough to the original mask (good result), to avoid generating too
    many images.
    """
    dice = float(rr.get("fusion_dice", 0.0))
    min_dice = _good_test_min_dice()
    if dice < min_dice:
        return False

    outdir = root(cfg) / "test_visuals_good_live"
    outdir.mkdir(parents=True, exist_ok=True)
    case_dir = _image_case_dir(outdir, row.image_id, "good")

    gt = z["gt"].astype(bool)
    valid = z["valid"].astype(bool)
    prob = z["fusion"].astype(np.float32)
    pred = prob >= thr

    sr = core.SlideScaleReader(row.image_path)
    rgb = np.asarray(
        sr.slide.get_thumbnail((gt.shape[1], gt.shape[0])).convert("RGB").resize((gt.shape[1], gt.shape[0]))
    )
    sr.close()

    gtov = rgb.copy()
    gtov[gt & valid] = (0.6 * gtov[gt & valid] + 0.4 * np.array([0, 255, 0])).astype(np.uint8)

    prov = rgb.copy()
    prov[pred & valid] = (0.6 * prov[pred & valid] + 0.4 * np.array([255, 0, 0])).astype(np.uint8)

    err = np.zeros_like(rgb) + 255
    err[(~gt) & (~pred) & valid] = [230, 230, 230]
    err[gt & pred & valid] = [60, 170, 90]
    err[gt & (~pred) & valid] = [240, 190, 40]
    err[(~gt) & pred & valid] = [210, 60, 60]

    arr = [
        ("Original H&E", rgb),
        ("Ground Truth", gt * 255),
        ("Prediction", pred * 255),
        ("Overlay", prov),
        ("Error Map", err),
    ]

    fig, axs = plt.subplots(1, 5, figsize=(18, 4))
    for ax, (title, im) in zip(axs.flat, arr):
        ax.imshow(im, cmap="gray" if getattr(im, "ndim", 3) == 2 else None)
        ax.set_title(title)
        ax.axis("off")

    fig.suptitle(
        f"Good TEST WSI | image_id={row.image_id} | provider={row.provider} | fusion_dice={dice:.4f}"
    )
    fig.tight_layout()

    base = f"good_{str(row.image_id)}_dice_{dice:.4f}"
    fig.savefig(case_dir / f"{base}_preview.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    Image.fromarray((np.clip(prob, 0, 1) * 255).astype(np.uint8)).save(case_dir / f"{base}_probability.png")
    Image.fromarray((pred * 255).astype(np.uint8)).save(case_dir / f"{base}_mask.png")
    Image.fromarray(prov).save(case_dir / f"{base}_overlay.png")
    Image.fromarray(err).save(case_dir / f"{base}_error_map.png")

    return True



def _bad_test_max_dice():
    """
    Save pathology-review bad cases when fusion Dice is BELOW this threshold.
    Default = 0.80, override with:
        PANDA_BAD_TEST_MAX_DICE=0.78
    """
    try:
        return float(os.environ.get("PANDA_BAD_TEST_MAX_DICE", "0.80"))
    except Exception:
        return 0.80


def _bad_test_max_cases():
    """
    Maximum number of poor test WSIs to save for pathology review.
    Default = 30, override with:
        PANDA_BAD_TEST_MAX_CASES=20
    """
    try:
        return max(1, int(os.environ.get("PANDA_BAD_TEST_MAX_CASES", "30")))
    except Exception:
        return 30


def _bad_test_crop_level0_px():
    """
    High-resolution crop size from the original WSI (level-0 pixels).
    Default = 3072 px, override with:
        PANDA_BAD_TEST_CROP_L0=4096
    """
    try:
        return max(512, int(os.environ.get("PANDA_BAD_TEST_CROP_L0", "3072")))
    except Exception:
        return 3072


def _largest_error_component_center(err_mask: np.ndarray):
    """
    Find the center of the largest connected error region on the evaluation grid.
    Fallback to the center of all error pixels if connected-components is empty.
    """
    m = (err_mask.astype(np.uint8) > 0).astype(np.uint8)
    if m.sum() == 0:
        h, w = m.shape
        return w // 2, h // 2

    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        ys, xs = np.nonzero(m)
        return int(np.mean(xs)), int(np.mean(ys))

    areas = stats[1:, cv2.CC_STAT_AREA]
    idx = 1 + int(np.argmax(areas))
    ys, xs = np.nonzero(labels == idx)
    if len(xs) == 0:
        ys, xs = np.nonzero(m)
    return int(np.mean(xs)), int(np.mean(ys))


def _safe_eval_crop(arr: np.ndarray, x0: int, y0: int, size: int, fill=0):
    """
    Crop an evaluation-grid array with zero-padding if the requested crop touches
    image borders.
    """
    if arr.ndim == 2:
        out = np.full((size, size), fill, dtype=arr.dtype)
    else:
        out = np.full((size, size, arr.shape[2]), fill, dtype=arr.dtype)

    H, W = arr.shape[:2]
    x1 = max(0, min(W, x0))
    y1 = max(0, min(H, y0))
    x2 = max(0, min(W, x0 + size))
    y2 = max(0, min(H, y0 + size))
    if x2 <= x1 or y2 <= y1:
        return out

    ox1 = x1 - x0
    oy1 = y1 - y0
    ox2 = ox1 + (x2 - x1)
    oy2 = oy1 + (y2 - y1)
    out[oy1:oy2, ox1:ox2] = arr[y1:y2, x1:x2]
    return out


def save_bad_test_review_pack(cfg, row, z, thr, rr):
    """
    Save a pathology-review package for poor TEST WSIs (Dice < 0.80 by default).
    Generates:
      1) full-slide overview composite
      2) high-resolution original crop + GT + prediction + error map
      3) individual files
    """
    dice = float(rr.get("fusion_dice", 0.0))
    if dice >= _bad_test_max_dice():
        return False

    outdir = root(cfg) / "test_visuals_bad_review"
    outdir.mkdir(parents=True, exist_ok=True)
    case_dir = _image_case_dir(outdir, row.image_id, "bad")

    gt = z["gt"].astype(bool)
    valid = z["valid"].astype(bool)
    prob = z["fusion"].astype(np.float32)
    pred = prob >= thr
    err_mask = (pred != gt) & valid

    # Full-slide overview (for global context).
    sr = core.SlideScaleReader(row.image_path)
    rgb = np.asarray(
        sr.slide.get_thumbnail((gt.shape[1], gt.shape[0])).convert("RGB").resize((gt.shape[1], gt.shape[0]))
    )

    err_rgb = np.zeros_like(rgb) + 255
    err_rgb[(~gt) & (~pred) & valid] = [230, 230, 230]
    err_rgb[gt & pred & valid] = [60, 170, 90]
    err_rgb[gt & (~pred) & valid] = [240, 190, 40]
    err_rgb[(~gt) & pred & valid] = [210, 60, 60]

    base = f"bad_{str(row.image_id)}_dice_{dice:.4f}"
    fig, axs = plt.subplots(1, 4, figsize=(18, 4.5))
    panels = [
        ("Original H&E (overview)", rgb),
        ("Original Mask", gt * 255),
        ("Model Mask", pred * 255),
        ("Error Map", err_rgb),
    ]
    for ax, (title, im) in zip(axs.flat, panels):
        ax.imshow(im, cmap="gray" if getattr(im, "ndim", 3) == 2 else None)
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle(
        f"Poor TEST WSI overview | image_id={row.image_id} | provider={row.provider} | fusion_dice={dice:.4f}"
    )
    fig.tight_layout()
    fig.savefig(case_dir / f"{base}_overview.png", dpi=260, bbox_inches="tight")
    plt.close(fig)

    # High-resolution crop focused on the largest error region.
    cx_eval, cy_eval = _largest_error_component_center(err_mask)
    crop_l0 = _bad_test_crop_level0_px()
    crop_eval = max(32, int(round(crop_l0 / float(cfg.paper_eval_downsample))))

    x0_eval = max(0, cx_eval - crop_eval // 2)
    y0_eval = max(0, cy_eval - crop_eval // 2)

    x0_l0 = int(round(x0_eval * float(cfg.paper_eval_downsample)))
    y0_l0 = int(round(y0_eval * float(cfg.paper_eval_downsample)))
    # Clamp to slide dimensions to keep the level-0 crop valid.
    if hasattr(sr, "dimensions"):
        w0, h0 = sr.dimensions
        x0_l0 = max(0, min(w0 - crop_l0, x0_l0))
        y0_l0 = max(0, min(h0 - crop_l0, y0_l0))
        x0_eval = int(round(x0_l0 / float(cfg.paper_eval_downsample)))
        y0_eval = int(round(y0_l0 / float(cfg.paper_eval_downsample)))

    rgb_crop, _ = sr.read_at_downsample(x0_l0, y0_l0, 1.0, crop_l0)
    sr.close()
    rgb_crop = np.asarray(rgb_crop)

    gt_crop = _safe_eval_crop((gt.astype(np.uint8) * 255), x0_eval, y0_eval, crop_eval, fill=0)
    pred_crop = _safe_eval_crop((pred.astype(np.uint8) * 255), x0_eval, y0_eval, crop_eval, fill=0)
    err_crop = _safe_eval_crop(err_rgb, x0_eval, y0_eval, crop_eval, fill=255)

    gt_up = cv2.resize(gt_crop, (rgb_crop.shape[1], rgb_crop.shape[0]), interpolation=cv2.INTER_NEAREST)
    pred_up = cv2.resize(pred_crop, (rgb_crop.shape[1], rgb_crop.shape[0]), interpolation=cv2.INTER_NEAREST)
    err_up = cv2.resize(err_crop, (rgb_crop.shape[1], rgb_crop.shape[0]), interpolation=cv2.INTER_NEAREST)

    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    crop_panels = [
        ("Original H&E (high-detail crop)", rgb_crop),
        ("Original Mask (crop)", gt_up),
        ("Model Mask (crop)", pred_up),
        ("Error Map (crop)", err_up),
    ]
    for ax, (title, im) in zip(axs.flat, crop_panels):
        ax.imshow(im, cmap="gray" if getattr(im, "ndim", 3) == 2 else None)
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle(
        f"Poor TEST WSI detail crop | image_id={row.image_id} | provider={row.provider} | fusion_dice={dice:.4f}"
    )
    fig.tight_layout()
    fig.savefig(case_dir / f"{base}_detail_crop.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Individual files for manual pathology review.
    Image.fromarray(rgb_crop).save(case_dir / f"{base}_detail_original.png")
    Image.fromarray(gt_up.astype(np.uint8)).save(case_dir / f"{base}_detail_gt_mask.png")
    Image.fromarray(pred_up.astype(np.uint8)).save(case_dir / f"{base}_detail_model_mask.png")
    Image.fromarray(err_up.astype(np.uint8)).save(case_dir / f"{base}_detail_error_map.png")
    Image.fromarray((np.clip(prob, 0, 1) * 255).astype(np.uint8)).save(case_dir / f"{base}_probability.png")

    return True


def save_bad_test_review_selection(cfg, test_df: pd.DataFrame, test_rows: pd.DataFrame,
                                   model, device, stain_proc, thr: float):
    """
    After TEST finishes, save up to N poor test slides (fusion_dice < threshold)
    for pathology review. This does not alter metrics; it only creates visuals.
    """
    outdir = root(cfg) / "test_visuals_bad_review"
    outdir.mkdir(parents=True, exist_ok=True)

    max_dice = _bad_test_max_dice()
    max_cases = _bad_test_max_cases()

    bad = test_df[test_df["fusion_dice"] < max_dice].copy()
    bad = bad.sort_values(["fusion_dice", "image_id"], ascending=[True, True]).head(max_cases).reset_index(drop=True)

    if len(bad) == 0:
        save_json(outdir / "bad_case_summary.json", {
            "selected_cases": 0,
            "max_cases": max_cases,
            "max_dice_threshold": max_dice,
            "note": "No test WSI had fusion_dice below the configured threshold."
        })
        return 0

    manifest_rows = []
    for _, rbad in bad.iterrows():
        rid = str(rbad["image_id"])
        row = test_rows[test_rows.image_id.astype(str) == rid].iloc[0]
        z = reconstruct_one(cfg, row, model, device, stain_proc, "test")  # cache hit
        saved = save_bad_test_review_pack(cfg, row, z, thr, {"fusion_dice": float(rbad["fusion_dice"])})
        if saved:
            print(f"[BAD TEST REVIEW SAVED] image_id={rid} | fusion_dice={float(rbad['fusion_dice']):.4f}", flush=True)
            manifest_rows.append({
                "image_id": rid,
                "provider": str(rbad.get("provider", row.provider)),
                "fusion_dice": float(rbad["fusion_dice"]),
                "overview_png": str(outdir / f"bad_{rid}_dice_{float(rbad['fusion_dice']):.4f}_overview.png"),
                "detail_crop_png": str(outdir / f"bad_{rid}_dice_{float(rbad['fusion_dice']):.4f}_detail_crop.png"),
            })
        del z
        gc.collect()

    if len(manifest_rows):
        pd.DataFrame(manifest_rows).to_csv(outdir / "bad_case_manifest.csv", index=False)

    save_json(outdir / "bad_case_summary.json", {
        "selected_cases": int(len(manifest_rows)),
        "max_cases": int(max_cases),
        "max_dice_threshold": float(max_dice),
        "crop_level0_px": int(_bad_test_crop_level0_px()),
        "folder": str(outdir),
    })
    return int(len(manifest_rows))



def _load_live_bad_manifest(cfg):
    """
    Resume-safe manifest for bad TEST cases saved during TEST.
    """
    outdir = root(cfg) / "test_visuals_bad_review"
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / "bad_case_manifest_live.csv"
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
            if "image_id" in df.columns:
                df["image_id"] = df["image_id"].astype(str)
            return df
        except Exception:
            pass
    return pd.DataFrame(columns=[
        "order_saved", "image_id", "provider", "fusion_dice",
        "overview_png", "detail_crop_png"
    ])


def _save_live_bad_manifest(cfg, df: pd.DataFrame):
    outdir = root(cfg) / "test_visuals_bad_review"
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / "bad_case_manifest_live.csv"
    df.to_csv(csv_path, index=False)
    save_json(outdir / "bad_case_summary_live.json", {
        "saved_cases_so_far": int(len(df)),
        "max_cases": int(_bad_test_max_cases()),
        "max_dice_threshold": float(_bad_test_max_dice()),
        "crop_level0_px": int(_bad_test_crop_level0_px()),
        "folder": str(outdir),
        "mode": "saved during TEST in arrival order",
        "note": "The first poor TEST WSIs encountered (below threshold) are saved immediately."
    })


def save_bad_test_review_live(cfg, row, z, thr, rr, live_manifest: pd.DataFrame):
    """
    During TEST, immediately save the FIRST N poor slides encountered
    (fusion_dice < threshold). Resume-safe: already-saved image_ids are skipped.
    """
    image_id = str(row.image_id)
    dice = float(rr.get("fusion_dice", 0.0))
    max_cases = _bad_test_max_cases()

    if dice >= _bad_test_max_dice():
        return live_manifest, False, "not_bad_enough"

    if len(live_manifest) >= max_cases:
        return live_manifest, False, "quota_reached"

    if len(live_manifest) and image_id in set(live_manifest["image_id"].astype(str).tolist()):
        return live_manifest, False, "already_saved"

    saved = save_bad_test_review_pack(cfg, row, z, thr, {"fusion_dice": dice})
    if not saved:
        return live_manifest, False, "save_failed"

    outdir = root(cfg) / "test_visuals_bad_review"
    base = f"bad_{image_id}_dice_{dice:.4f}"
    new_row = pd.DataFrame([{
        "order_saved": int(len(live_manifest) + 1),
        "image_id": image_id,
        "provider": str(rr.get("provider", row.provider)),
        "fusion_dice": dice,
        "overview_png": str((outdir / f"bad_{image_id}") / f"{base}_overview.png"),
        "detail_crop_png": str((outdir / f"bad_{image_id}") / f"{base}_detail_crop.png"),
    }])
    live_manifest = pd.concat([live_manifest, new_row], ignore_index=True)
    _save_live_bad_manifest(cfg, live_manifest)
    return live_manifest, True, "saved"

# ---------------------------------------------------------------------------
# Reviewer-proof additions: boundary metrics, denominators, statistics,
# reproducibility manifests. These are evaluation/reporting only and do NOT
# alter the trained model, optimizer, loss, sampling, or training data.
# ---------------------------------------------------------------------------

def _binary_boundary(mask: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """1-pixel boundary on the evaluation grid, restricted to valid support."""
    m = (mask.astype(bool) & valid.astype(bool)).astype(np.uint8)
    if not m.any():
        return np.zeros_like(m, dtype=bool)
    kernel = np.ones((3, 3), np.uint8)
    er = cv2.erode(m, kernel, iterations=1)
    return (m.astype(bool) & ~er.astype(bool)) & valid.astype(bool)


def boundary_metrics_pixels(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray,
                            eval_downsample: float, surface_tol_px: float = 2.0) -> Dict[str, Any]:
    """
    Boundary metrics in evaluation-grid PIXELS (plus level-0-equivalent pixels).

    HD95 / ASSD are undefined when exactly one boundary is empty; those cases are
    kept as NaN and separately flagged instead of inventing a distance.
    """
    v = valid.astype(bool)
    pb = _binary_boundary(pred, v)
    gb = _binary_boundary(gt, v)
    npb, ngb = int(pb.sum()), int(gb.sum())
    base = {
        "boundary_pred_pixels": npb,
        "boundary_gt_pixels": ngb,
        "boundary_empty_pred": int(npb == 0),
        "boundary_empty_gt": int(ngb == 0),
        "surface_tolerance_eval_px": float(surface_tol_px),
        "surface_tolerance_level0_px": float(surface_tol_px * eval_downsample),
    }
    if npb == 0 and ngb == 0:
        return {
            **base, "hd95_eval_px": 0.0, "assd_eval_px": 0.0, "surface_dice": 1.0,
            "hd95_level0_px": 0.0, "assd_level0_px": 0.0,
        }
    if npb == 0 or ngb == 0:
        return {
            **base, "hd95_eval_px": float("nan"), "assd_eval_px": float("nan"),
            "surface_dice": 0.0, "hd95_level0_px": float("nan"),
            "assd_level0_px": float("nan"),
        }

    # distanceTransform returns distance to nearest zero pixel. Make target
    # boundary pixels zero and everything else one.
    to_g = cv2.distanceTransform((~gb).astype(np.uint8), cv2.DIST_L2, 5)
    to_p = cv2.distanceTransform((~pb).astype(np.uint8), cv2.DIST_L2, 5)
    d_pg = to_g[pb].astype(np.float64)
    d_gp = to_p[gb].astype(np.float64)
    both = np.concatenate([d_pg, d_gp])
    hd95 = float(np.percentile(both, 95))
    assd = float((d_pg.mean() + d_gp.mean()) / 2.0)
    within_p = float((d_pg <= surface_tol_px).sum())
    within_g = float((d_gp <= surface_tol_px).sum())
    surface_dice = float((within_p + within_g) / max(1.0, npb + ngb))
    return {
        **base,
        "hd95_eval_px": hd95,
        "assd_eval_px": assd,
        "surface_dice": surface_dice,
        "hd95_level0_px": float(hd95 * eval_downsample),
        "assd_level0_px": float(assd * eval_downsample),
    }


def _holm_adjust(pvals):
    """Holm family-wise error correction, no scipy dependency."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    if n == 0:
        return np.asarray([], dtype=float)
    order = np.argsort(p)
    adj = np.empty(n, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, (n - rank) * p[idx])
        running = max(running, val)
        adj[idx] = running
    return adj


def _paired_bootstrap_comparison(a: np.ndarray, b: np.ndarray, seed: int,
                                 iters: int = BOOTSTRAP_ITERS) -> Dict[str, Any]:
    """Two-sided paired slide bootstrap + effect sizes."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    n = len(a)
    if n == 0:
        return {"n_pairs": 0}
    d = a - b
    rng = np.random.default_rng(seed)
    boot = np.empty(iters, dtype=np.float64)
    for i in range(iters):
        idx = rng.integers(0, n, size=n)
        boot[i] = float(d[idx].mean())
    p = float(2 * min((boot <= 0).mean(), (boot >= 0).mean()))
    sd = float(d.std(ddof=1)) if n > 1 else 0.0
    cohen_dz = float(d.mean() / sd) if sd > 0 else (0.0 if d.mean() == 0 else float("inf"))
    # Paired probability of superiority: fraction of slides where A > B,
    # ties contribute half.
    psup = float(((d > 0).sum() + 0.5 * (d == 0).sum()) / n)
    return {
        "n_pairs": int(n),
        "mean_a": float(a.mean()),
        "mean_b": float(b.mean()),
        "mean_difference": float(d.mean()),
        "median_difference": float(np.median(d)),
        "difference_95CI": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        "two_sided_bootstrap_p": min(1.0, p),
        "cohen_dz": cohen_dz,
        "probability_superiority": psup,
        "bootstrap_iterations": int(iters),
    }


def major_wsi_significance(cfg: FinalConfig, ps: pd.DataFrame):
    """Fusion vs each native level with two-sided paired bootstrap + Holm correction."""
    rows = []
    for lv in cfg.levels:
        res = _paired_bootstrap_comparison(
            ps["fusion_dice"].to_numpy(),
            ps[f"L{lv}_dice"].to_numpy(),
            seed=int(cfg.seed) + int(lv) + 100,
        )
        rows.append({"comparison": f"fusion_vs_L{lv}", "metric": "dice", **res})
    df = pd.DataFrame(rows)
    if len(df) and "two_sided_bootstrap_p" in df:
        df["p_holm"] = _holm_adjust(df["two_sided_bootstrap_p"].fillna(1.0).to_numpy())
        df["significant_holm_0.05"] = df["p_holm"] < 0.05
    df.to_csv(root(cfg)/"stats"/"major_wsi_comparisons_two_sided_holm.csv", index=False)
    return df


def boundary_summary_table(cfg: FinalConfig, ps: pd.DataFrame):
    cols = ["fusion_hd95_eval_px", "fusion_assd_eval_px", "fusion_surface_dice",
            "fusion_hd95_level0_px", "fusion_assd_level0_px"]
    rows = []
    for c in cols:
        if c not in ps.columns:
            continue
        x = pd.to_numeric(ps[c], errors="coerce")
        finite = x[np.isfinite(x)]
        rows.append({
            "metric": c,
            "n_defined": int(len(finite)),
            "n_undefined": int(len(x) - len(finite)),
            "mean": float(finite.mean()) if len(finite) else float("nan"),
            "median": float(finite.median()) if len(finite) else float("nan"),
            "sd": float(finite.std(ddof=1)) if len(finite) > 1 else 0.0,
            "min": float(finite.min()) if len(finite) else float("nan"),
            "max": float(finite.max()) if len(finite) else float("nan"),
        })
    df = pd.DataFrame(rows)
    df.to_csv(root(cfg)/"tables"/"boundary_metrics_summary.csv", index=False)
    return df


def provider_significance(cfg: FinalConfig, ps: pd.DataFrame):
    """Descriptive provider contrast. Unpaired bootstrap because providers are different slides."""
    providers = sorted(ps["provider"].dropna().astype(str).unique())
    if len(providers) != 2:
        out = {"status": "not_run", "reason": f"Expected exactly 2 providers, found {providers}"}
        save_json(root(cfg)/"stats"/"provider_difference_bootstrap.json", out)
        return out
    a = ps[ps.provider.astype(str) == providers[0]].fusion_dice.to_numpy(dtype=float)
    b = ps[ps.provider.astype(str) == providers[1]].fusion_dice.to_numpy(dtype=float)
    rng = np.random.default_rng(int(cfg.seed) + 444)
    boot = []
    for _ in range(BOOTSTRAP_ITERS):
        aa = a[rng.integers(0, len(a), size=len(a))]
        bb = b[rng.integers(0, len(b), size=len(b))]
        boot.append(float(aa.mean() - bb.mean()))
    boot = np.asarray(boot)
    p = float(2 * min((boot <= 0).mean(), (boot >= 0).mean()))
    pooled_sd = float(np.sqrt(((len(a)-1)*a.var(ddof=1)+(len(b)-1)*b.var(ddof=1)) /
                              max(1, len(a)+len(b)-2))) if len(a)>1 and len(b)>1 else 0.0
    effect = float((a.mean()-b.mean())/pooled_sd) if pooled_sd > 0 else 0.0
    out = {
        "provider_a": providers[0], "provider_b": providers[1],
        "n_a": int(len(a)), "n_b": int(len(b)),
        "mean_dice_a": float(a.mean()), "mean_dice_b": float(b.mean()),
        "mean_difference_a_minus_b": float(a.mean()-b.mean()),
        "difference_95CI": [float(np.percentile(boot,2.5)), float(np.percentile(boot,97.5))],
        "two_sided_bootstrap_p": min(1.0,p),
        "cohen_d": effect,
        "bootstrap_iterations": BOOTSTRAP_ITERS,
        "interpretation": "Provider comparison is an unpaired robustness analysis, not a randomized causal test."
    }
    save_json(root(cfg)/"stats"/"provider_difference_bootstrap.json", out)
    return out


def write_reproducibility_bundle(cfg: FinalConfig, split: pd.DataFrame, threshold: float, checkpoint: str):
    """Machine-readable and human-readable reproducibility record."""
    import sys, platform
    versions = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "opencv": cv2.__version__,
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device": torch.cuda.get_device_name(torch.cuda.current_device()) if torch.cuda.is_available() else None,
    }
    for modname in ["timm", "sklearn", "tiatoolbox", "segmentation_models_pytorch"]:
        try:
            mod = __import__(modname)
            versions[modname] = getattr(mod, "__version__", "unknown")
        except Exception as e:
            versions[modname] = f"unavailable: {type(e).__name__}"

    counts = {
        "total_rows": int(len(split)),
        "split_counts": {str(k): int(v) for k,v in split["split"].value_counts().to_dict().items()},
        "provider_counts": {str(k): int(v) for k,v in split["provider"].value_counts().to_dict().items()} if "provider" in split else {},
        "provider_by_split": (
            split.groupby(["split","provider"]).size().unstack(fill_value=0).astype(int).to_dict()
            if {"split","provider"}.issubset(split.columns) else {}
        ),
        "source_of_truth": "Counts recomputed from the actual split CSV generated from train.csv; no manuscript hard-coded counts.",
    }
    save_json(root(cfg)/"reproducibility_versions.json", versions)
    save_json(root(cfg)/"tables"/"dataset_counts_verified.json", counts)

    keep_cols = [c for c in ["image_id","split","provider","isup_grade","gleason_score","image_path","mask_path"] if c in split.columns]
    split[keep_cols].to_csv(root(cfg)/"tables"/"split_ids_reproducibility.csv", index=False)

    try:
        cfg_dict = dict(vars(cfg))
        cfg_dict = {k: str(v) if isinstance(v, Path) else v for k,v in cfg_dict.items()}
    except Exception:
        cfg_dict = {"repr": repr(cfg)}
    save_json(root(cfg)/"reproducibility_config.json", {
        "seed": int(cfg.seed),
        "optimal_validation_threshold": float(threshold),
        "checkpoint": str(checkpoint),
        "config": cfg_dict,
        "notes": {
            "threshold_rule": "Selected on Validation only, then frozen for held-out Test.",
            "wsi_rule": "Test WSI metrics come from dense streaming reconstruction of retained original Test WSIs.",
            "fusion_rule": "Support-aware mean: a level contributes only where that level has prediction support.",
            "normalization_rule": "Vahadane target/reference is selected from Train only and the same fixed target is applied to Train/Validation/Test/WSI inference.",
        }
    })

    readme = f"""# PANDA reproducibility record

This directory is generated automatically by the pipeline.

- Random seed: {int(cfg.seed)}
- Best model checkpoint: {checkpoint}
- Validation-selected threshold: {float(threshold):.4f}
- Test set is held out from model/threshold selection.
- WSI results are reconstructed from retained original Test WSIs by dense streaming.
- Fusion is support-aware across L0/L1/L2.
- Vahadane reference is fitted/selected from Train only; the same fixed reference is applied at Validation/Test inference.
- `tables/split_ids_reproducibility.csv` contains exact slide IDs and split membership.
- `tables/dataset_counts_verified.json` contains counts recomputed from the generated split, not hard-coded manuscript numbers.
- `reproducibility_versions.json` records software/CUDA versions.
- `reproducibility_config.json` records the run configuration.
- `tables/per_slide_wsi_metrics.csv` exposes raw slide-level points behind means/SD/CIs.
- Boundary distances are reported in evaluation-grid pixels and level-0-equivalent pixels; no mm conversion is imposed.
- CAMELYON-only ITC/micro/macro/FROC analyses are intentionally not transplanted to prostate biopsy segmentation.
"""
    (root(cfg)/"README_REPRODUCIBILITY.md").write_text(readme, encoding="utf-8")

def paper_all(cfg: FinalConfig):
    r=root(cfg); make_pipeline_figure(cfg); make_training_curves(cfg)
    model,device,ckpt=core.load_best_model(cfg,None)
    if device.type=="cuda":
        torch.backends.cudnn.benchmark=True
        if getattr(cfg, "channels_last", False):
            model=model.to(memory_format=torch.channels_last)
        print(f"[WSI INFERENCE CUDA] ENABLED | {torch.cuda.get_device_name(device)} | AMP={bool(cfg.amp)} | batch={_paper_inference_batch_size(cfg,device)} | prep_workers={_wsi_prep_workers()} | prep_chunk={_wsi_prep_chunk(_paper_inference_batch_size(cfg,device))} | channels_last={bool(getattr(cfg,'channels_last',False))}", flush=True)
    else:
        print("[WSI INFERENCE CUDA] DISABLED -> CPU inference", flush=True)
    selected=pd.read_csv(cfg.selected_train_csv)
    core.auto_choose_stain_target(cfg,selected)
    target=np.array(Image.open(cfg.stain_target_path).convert("RGB"), dtype=np.uint8, copy=True)
    stain=core.VahadaneProcessor(target)

    split=pd.read_csv(cfg.split_csv)
    val=split[split.split=="val"].reset_index(drop=True)
    test=split[split.split=="test"].reset_index(drop=True)

    thr=threshold_validation(cfg,val,model,device,stain)

    per=[]; patch_acc={lv:[0,0,0,0] for lv in cfg.levels}
    live_bad_manifest = _load_live_bad_manifest(cfg)
    print(f"[BAD TEST LIVE] already saved {len(live_bad_manifest)}/{_bad_test_max_cases()} poor TEST cases", flush=True)
    test_outer=tqdm(total=len(test), desc="TEST full-WSI reconstruction", unit="WSI")
    for _,row in test.iterrows():
        z=reconstruct_one(cfg,row,model,device,stain,"test")
        rr={"image_id":str(row.image_id),"provider":str(row.provider)}
        for lv in cfg.levels:
            pc=z[f"patch_counts_L{lv}"].astype(np.int64)
            for i in range(4): patch_acc[lv][i]+=int(pc[i])
            lv_valid = z["annotated_valid"].astype(bool) & z[f"support_L{lv}"].astype(bool)
            lv_pred = z[f"prob_L{lv}"].astype(np.float32)>=thr
            gt_bool = z["gt"].astype(bool)
            c=counts(lv_pred,gt_bool,lv_valid)
            mm=metrics(*c)
            for k,v in mm.items(): rr[f"L{lv}_{k}"]=v
            rr[f"L{lv}_valid_pixels"] = int(lv_valid.sum())
            rr[f"L{lv}_gt_tumor_pixels"] = int((gt_bool & lv_valid).sum())
            rr[f"L{lv}_pred_tumor_pixels"] = int((lv_pred & lv_valid).sum())
            bm = boundary_metrics_pixels(lv_pred, gt_bool, lv_valid, float(cfg.paper_eval_downsample))
            for k,v in bm.items(): rr[f"L{lv}_{k}"] = v

        fusion_valid = z["valid"].astype(bool)
        fusion_pred = z["fusion"].astype(np.float32)>=thr
        gt_bool = z["gt"].astype(bool)
        c=counts(fusion_pred,gt_bool,fusion_valid)
        mm=metrics(*c)
        for k,v in mm.items(): rr[f"fusion_{k}"]=v
        rr["fusion_valid_pixels"] = int(fusion_valid.sum())
        rr["fusion_gt_tumor_pixels"] = int((gt_bool & fusion_valid).sum())
        rr["fusion_pred_tumor_pixels"] = int((fusion_pred & fusion_valid).sum())
        bm = boundary_metrics_pixels(fusion_pred, gt_bool, fusion_valid, float(cfg.paper_eval_downsample))
        for k,v in bm.items(): rr[f"fusion_{k}"] = v

        # Save a live visual only for good TEST results.
        try:
            saved_live = save_good_test_preview(cfg, row, z, thr, rr)
            if saved_live:
                print(f"[GOOD TEST VISUAL SAVED] image_id={row.image_id} | fusion_dice={rr['fusion_dice']:.4f}", flush=True)
        except Exception as e:
            print(f"[GOOD TEST VISUAL SKIP] image_id={row.image_id} | reason={type(e).__name__}: {e}", flush=True)

        # Save the FIRST poor TEST cases immediately during TEST.
        try:
            live_bad_manifest, bad_saved, bad_reason = save_bad_test_review_live(cfg, row, z, thr, rr, live_bad_manifest)
            if bad_saved:
                print(f"[BAD TEST REVIEW SAVED] image_id={row.image_id} | fusion_dice={rr['fusion_dice']:.4f} | count={len(live_bad_manifest)}/{_bad_test_max_cases()}", flush=True)
        except Exception as e:
            print(f"[BAD TEST REVIEW SKIP] image_id={row.image_id} | reason={type(e).__name__}: {e}", flush=True)

        per.append(rr)
        del z
        gc.collect()
        test_outer.update(1)
    test_outer.close()

    ps=pd.DataFrame(per); ps.to_csv(r/"tables"/"per_slide_wsi_metrics.csv",index=False)
    save_json(r/"tables"/"per_slide_summary.json",{
        "n_slides":len(ps),"mean_dice":float(ps.fusion_dice.mean()),
        "sd_dice":float(ps.fusion_dice.std(ddof=1)),"min_dice":float(ps.fusion_dice.min()),
        "max_dice":float(ps.fusion_dice.max()),"median_dice":float(ps.fusion_dice.median()),
        "total_valid_pixels":int(ps.fusion_valid_pixels.sum()),
        "total_gt_tumor_pixels":int(ps.fusion_gt_tumor_pixels.sum()),
        "total_pred_tumor_pixels":int(ps.fusion_pred_tumor_pixels.sum()),
        "pixel_units_note":"Counts are on the WSI evaluation grid; boundary distances are also exported in level-0-equivalent pixels."
    })
    boundary_df = boundary_summary_table(cfg, ps)

    # Global WSI metrics + CIs.
    global_rows=[]; ci_rows=[]
    for key in [f"L{lv}" for lv in cfg.levels]+["fusion"]:
        mm=metrics(int(ps[f"{key}_tp"].sum()),int(ps[f"{key}_fp"].sum()),int(ps[f"{key}_tn"].sum()),int(ps[f"{key}_fn"].sum()))
        global_rows.append({"subset":key,**mm})
        ci=bootstrap(ps,key,cfg.seed)
        for metric_name in ["dice","iou","precision","recall","specificity","accuracy"]:
            ci_rows.append({"subset":key,"metric":metric_name,"value":mm[metric_name],
                            "ci_low":ci[metric_name][0],"ci_high":ci[metric_name][1],
                            "bootstrap_iterations":BOOTSTRAP_ITERS})
    pd.DataFrame(global_rows).to_csv(r/"tables"/"wsi_metrics_per_level_and_fusion.csv",index=False)
    pd.DataFrame(ci_rows).to_csv(r/"tables"/"global_wsi_metrics_95CI.csv",index=False)

    # Patch confusion matrices.
    prow=[]
    for lv in cfg.levels:
        mm=metrics(*patch_acc[lv]); prow.append({"subset":f"L{lv}",**mm})
        confusion_fig(f"Patch Confusion Matrix L{lv}",mm,r/"figures"/f"03_confusion_L{lv}.png")
    pooled=[sum(patch_acc[lv][i] for lv in cfg.levels) for i in range(4)]
    pmm=metrics(*pooled); prow.append({"subset":"all_levels_pooled",**pmm})
    confusion_fig("Patch Confusion Matrix — All Levels Pooled",pmm,r/"figures"/"03_confusion_all_levels.png")
    pd.DataFrame(prow).to_csv(r/"tables"/"patch_metrics_per_level_and_pooled.csv",index=False)

    # Per-slide plot.
    s=ps.sort_values("fusion_dice").reset_index(drop=True)
    fig,ax=plt.subplots(figsize=(11,5)); ax.scatter(np.arange(1,len(s)+1),s.fusion_dice*100,s=22)
    mean=s.fusion_dice.mean()*100; ax.axhline(mean,ls="--"); ax.set_xlabel("Test slide (sorted)")
    ax.set_ylabel("Dice (%)"); ax.set_title("Per-slide WSI Dice")
    fig.tight_layout(); fig.savefig(r/"figures"/"04_per_slide_dice.png",dpi=300,bbox_inches="tight"); plt.close(fig)

    # Provider metrics.
    provider_rows=[]
    for provider,sub in ps.groupby("provider"):
        mm=metrics(int(sub.fusion_tp.sum()),int(sub.fusion_fp.sum()),int(sub.fusion_tn.sum()),int(sub.fusion_fn.sum()))
        ci=bootstrap(sub.reset_index(drop=True),"fusion",cfg.seed)
        provider_rows.append({"provider":provider,"n_slides":len(sub),**mm,
                              "dice_ci_low":ci["dice"][0],"dice_ci_high":ci["dice"][1]})
    pdf=pd.DataFrame(provider_rows); pdf.to_csv(r/"tables"/"provider_metrics_95CI.csv",index=False)
    if len(pdf):
        fig,ax=plt.subplots(figsize=(7,4)); ax.bar(pdf.provider,pdf.dice)
        ax.set_ylabel("Dice"); ax.set_title("Radboud vs Karolinska WSI Performance")
        fig.tight_layout(); fig.savefig(r/"figures"/"05_provider_comparison.png",dpi=300,bbox_inches="tight"); plt.close(fig)

    # Qualitative best/median/worst.
    ids={"worst":str(s.iloc[0].image_id),"median":str(s.iloc[len(s)//2].image_id),"best":str(s.iloc[-1].image_id)}
    for tag,iid in ids.items():
        row=test[test.image_id.astype(str)==str(iid)].iloc[0]
        z=reconstruct_one(cfg,row,model,device,stain,"test")  # cache hit; no inference rerun
        qualitative(cfg,row,z,thr,tag)
        del z
    save_json(r/"qualitative"/"selected_examples.json",ids)

    # Poor TEST pathology-review cases are now saved LIVE during TEST
    # (first N encountered below the configured Dice threshold).

    # Major WSI statistical comparisons: two-sided paired bootstrap, 95% CI,
    # effect sizes, and Holm correction across fusion-vs-level comparisons.
    sig_df = major_wsi_significance(cfg, ps)
    # Preserve the legacy single best-level JSON for backwards compatibility.
    level_means={f"L{lv}":float(ps[f"L{lv}_dice"].mean()) for lv in cfg.levels}
    best=max(level_means,key=level_means.get)
    best_row = sig_df[sig_df.comparison == f"fusion_vs_{best}"].iloc[0].to_dict() if len(sig_df) else {}
    save_json(r/"stats"/"paired_bootstrap_fusion_vs_best_level.json", best_row)
    provider_stat = provider_significance(cfg, ps)

    save_json(r/"tables"/"not_applicable_camelyon_specific_analyses.json",{
        "FROC_ITC_micro_macro":"not copied",
        "reason":"CAMELYON ITC/micrometastasis/macrometastasis definitions are lymph-node-metastasis-specific and are not prostate-biopsy categories."
    })

    write_reproducibility_bundle(cfg, split, thr, ckpt)
    save_json(r/"paper_results_summary.json",{
        "checkpoint":ckpt,"optimal_validation_threshold":thr,"test_slides":len(ps),
        "levels":list(cfg.levels),
        "fusion":"support-aware mean probability of L0/L1/L2 on common WSI grid",
        "wsi_evaluation":"dense streaming reconstruction from retained original Test WSIs",
        "boundary_metrics":["HD95","ASSD","surface Dice"],
        "boundary_units":"evaluation-grid pixels + level-0-equivalent pixels",
        "major_statistics_csv":str(r/"stats"/"major_wsi_comparisons_two_sided_holm.csv"),
        "provider_difference_stats":str(r/"stats"/"provider_difference_bootstrap.json"),
        "raw_per_slide_points":str(r/"tables"/"per_slide_wsi_metrics.csv"),
        "reproducibility_readme":str(r/"README_REPRODUCIBILITY.md")
    })
