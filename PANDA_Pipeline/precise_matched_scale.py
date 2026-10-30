#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PRECISE external full-WSI test at the PANDA pixel size (revision of 30 Oct 2026).

The first PRECISE run (run_external_precise_wsi_test.py) read OME levels 0, 2
and 4 of PRECISE (0.243, 0.97 and 3.9 um/pixel), about twice the PANDA
magnification. This script reads OME levels 1, 3 and 5 instead:

    logical L0 = PRECISE level 1   0.486 um/pixel   (PANDA L0 about 0.45-0.50)
    logical L1 = PRECISE level 3   1.944 um/pixel   (PANDA L1 about 1.8-2.0)
    logical L2 = PRECISE level 5   7.776 um/pixel   (PANDA L2 about 7.2-8.0)
    evaluation grid = PRECISE level 5 (16x below logical L0, 7.776 um/pixel)

Everything else is the first run's protocol: the frozen PANDA checkpoint, the
training-derived Vahadane target, threshold 0.43, 512 x 512 patches, stride
128, the same tissue rules and support-aware fusion, label 1 = tumor,
labels 2 and 7 = negative, labels 0 and 3-6 not scored.

Two changes, both recorded in the outputs:
  * Patch retention uses the H&E tissue rules only. The first run also skipped
    patches without any label 1/2/7 pixel; that rule cannot change a scored
    pixel's prediction, but it left labels 3-6 without predictions. With
    tissue-only retention the model is also applied to those regions (B5), and
    retention uses no reference-mask information (D14).
  * Each patch is predicted twice, with its true normalized coordinates and
    with the constant coordinate (0.5, 0.5) (coordinate reliance check, D8).
    The second map never enters the primary metrics.

Each cache stores the raw PRECISE labels on the evaluation grid (label_eval),
so excluded-label analyses need no further inference.

Usage (through the launcher run_external_precise_matched_scale.py):
    set PANDA_DATA_ROOT=<work root holding runs/, stain_target_train.png, paper_results/tables/best_threshold.json>
    set PRECISE_MATCHED_ROOT=<folder for downloads, caches and results>
    python run_external_precise_matched_scale.py
Optional: PRECISE_DELETE_IMAGE_AFTER=1 deletes each H&E image (not its mask)
after its cache is complete.
"""
import gc
import json
import math
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

import run_external_precise_wsi_test as base
from panda_final_pipeline import FinalConfig
import panda_camelyon_style_pipeline as core

LEVEL_OFFSET = 1
OME_LEVEL_INDEX = {0: 1, 1: 3, 2: 5}
LEVEL_DOWNSAMPLES = {0: 1, 1: 4, 2: 16}       # relative to logical L0 (PRECISE level 1)
EVAL_DS = 16.0                                # relative to logical L0
EVAL_OME_LEVEL = 5
UM_PER_PX_LEVEL0 = 0.243                      # PRECISE README
PATCH, STRIDE = base.PATCH, base.STRIDE
POS, NEG = base.POSITIVE_LABELS, base.NEGATIVE_LABELS
VALID = POS | NEG
CONST_COORD = (0.5, 0.5)

ROOT = Path(os.environ.get("PRECISE_MATCHED_ROOT", str(base.WORK_ROOT / "external_precise_wsi_matched_scale")))
DATA = ROOT / "dataset_he_only"
CACHE = ROOT / "wsi_cache"
RESULTS = ROOT / "results"
# The download helpers of the first script write next to these globals.
base.EXT_ROOT, base.DATA_ROOT, base.STATE_ROOT = ROOT, DATA, ROOT / "state"
base.RESULT_ROOT, base.CACHE_ROOT = RESULTS, CACHE


def cache_path(sid):
    return CACHE / f"{sid}_lvl135_ds16_matched_v1.npz"


# Vahadane normalization in worker processes (the thread version of the first
# script is limited by the GIL). Workers use the light norm_pool module.
_POOL = None


def _quiet():
    import logging
    for name in ("tiatoolbox", ""):
        logging.getLogger(name).setLevel(logging.ERROR)


def normalize_many(rgbs, stain, workers):
    global _POOL
    if workers <= 1:
        return [stain.transform(x) for x in rgbs]
    if _POOL is None:
        from concurrent.futures import ProcessPoolExecutor
        import norm_pool
        _POOL = ProcessPoolExecutor(max_workers=workers, initializer=norm_pool.init,
                                    initargs=(str(FinalConfig().stain_target_path),))
    import norm_pool
    return list(_POOL.map(norm_pool.normalize, rgbs, chunksize=2))


def read_level(pyr, li):
    a = np.asarray(pyr.arr(li))
    if a.ndim == 3 and a.shape[-1] > 3:
        a = a[..., :3]
    return a


@torch.inference_mode()
def predict_two(model, device, rgbs, coords, cfg, bs):
    """Predict each patch with its own coordinates and with CONST_COORD."""
    true_c = base.predict_batch(model, device, rgbs, coords, cfg, bs)
    const_c = base.predict_batch(model, device, rgbs, [CONST_COORD] * len(rgbs), cfg, bs)
    return true_c, const_c


def reconstruct(row, model, device, stain, cfg, thr):
    sid = str(row["image_id"])
    out = cache_path(sid)
    if out.exists():
        with np.load(out) as z:
            return {k: z[k] for k in z.files}
    im, ma = base.OmePyramid(row["image_path"]), base.OmePyramid(row["mask_path"])
    try:
        l0 = im.arr(OME_LEVEL_INDEX[0])
        h0, w0 = int(l0.shape[0]), int(l0.shape[1])
        H, W = math.ceil(h0 / EVAL_DS), math.ceil(w0 / EVAL_DS)

        lab = read_level(ma, EVAL_OME_LEVEL)
        if lab.ndim == 3:
            lab = lab[..., 0]
        lab = lab[:H, :W]
        if lab.shape != (H, W):
            lab = cv2.resize(lab.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
        gt = np.isin(lab, list(POS))
        annotated_valid = np.isin(lab, list(VALID))

        ev = read_level(im, EVAL_OME_LEVEL).astype(np.uint8)
        if ev.ndim == 2:
            ev = np.repeat(ev[..., None], 3, axis=2)
        tissue_eval = base.make_tissue_mask_conservative(ev)
        if tissue_eval.shape != (H, W):
            tissue_eval = cv2.resize(tissue_eval[:H, :W].astype(np.uint8), (W, H),
                                     interpolation=cv2.INTER_NEAREST).astype(bool)

        thumb = im.thumbnail(EVAL_OME_LEVEL, 2048)
        tissue_thumb = base.make_tissue_mask(thumb)
        th, tw = tissue_thumb.shape
        bs = int(os.environ.get("PRECISE_INFER_BATCH", "16"))
        workers = int(os.environ.get("PRECISE_PREP_WORKERS", "4"))
        min_patch_tissue = float(os.environ.get("PRECISE_MIN_PATCH_TISSUE", "0.05"))
        chunk_size = max(bs * 4, 64)

        prob, prob_c, support, pcounts, n_patches = {}, {}, {}, {}, {}
        for lv in (0, 1, 2):
            li, ds = OME_LEVEL_INDEX[lv], LEVEL_DOWNSAMPLES[lv]
            arr = im.arr(li)
            lh, lw = int(arr.shape[0]), int(arr.shape[1])
            sx, sy = lw / max(tw, 1), lh / max(th, 1)
            coords = []
            for y in range(0, max(1, lh - PATCH + 1), STRIDE):
                for x in range(0, max(1, lw - PATCH + 1), STRIDE):
                    tx1, ty1 = int(x / sx), int(y / sy)
                    tx2 = max(tx1 + 1, min(tw, int((x + PATCH) / sx)))
                    ty2 = max(ty1 + 1, min(th, int((y + PATCH) / sy)))
                    frac = float(tissue_thumb[ty1:ty2, tx1:tx2].mean()) if ty2 > ty1 and tx2 > tx1 else 0.0
                    if frac >= float(cfg.tissue_min_frac):
                        coords.append((x, y))
            sp = np.zeros((H, W), np.float32); spc = np.zeros((H, W), np.float32)
            sw = np.zeros((H, W), np.float32)
            ptp = pfp = ptn = pfn = 0
            used = 0
            for st in tqdm(range(0, len(coords), chunk_size), desc=f"{sid} L{lv}", leave=False, unit="chunk"):
                rgbs, cn, meta = [], [], []
                for x, y in coords[st:st + chunk_size]:
                    rgb = im.read_patch(li, x, y, PATCH, True)
                    tissuep = base.make_tissue_mask_conservative(rgb)
                    if float(tissuep.mean()) < min_patch_tissue:
                        continue
                    maskp = ma.read_patch(li, x, y, PATCH, False)
                    cx0, cy0 = (x + PATCH / 2.0) * ds, (y + PATCH / 2.0) * ds
                    rgbs.append(rgb)
                    cn.append((cx0 / max(w0, 1), cy0 / max(h0, 1)))
                    meta.append((x, y, maskp, tissuep))
                if not rgbs:
                    continue
                nrgb = normalize_many(rgbs, stain, workers)
                p_true, p_const = predict_two(model, device, nrgb, cn, cfg, bs)
                used += len(rgbs)
                for pt, pc_, (x, y, maskp, tissuep) in zip(p_true, p_const, meta):
                    tf = tissuep.astype(np.float32)
                    pt = pt.astype(np.float32) * tf
                    pc_ = pc_.astype(np.float32) * tf
                    validp = np.isin(maskp, list(VALID))
                    if validp.any():
                        c = base.counts(pt >= thr, np.isin(maskp, list(POS)), validp)
                        ptp += c[0]; pfp += c[1]; ptn += c[2]; pfn += c[3]
                    x0, y0 = int((x * ds) / EVAL_DS), int((y * ds) / EVAL_DS)
                    x1 = min(W, int(math.ceil(((x + PATCH) * ds) / EVAL_DS)))
                    y1 = min(H, int(math.ceil(((y + PATCH) * ds) / EVAL_DS)))
                    if x1 <= x0 or y1 <= y0:
                        continue
                    sp[y0:y1, x0:x1] += cv2.resize(pt, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
                    spc[y0:y1, x0:x1] += cv2.resize(pc_, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
                    sw[y0:y1, x0:x1] += 1.0
            s = sw > 0
            lp = np.zeros_like(sp); np.divide(sp, sw, out=lp, where=s); lp[~tissue_eval] = 0.0
            lc = np.zeros_like(spc); np.divide(spc, sw, out=lc, where=s); lc[~tissue_eval] = 0.0
            prob[lv], prob_c[lv], support[lv] = lp, lc, s
            pcounts[lv] = np.array([ptp, pfp, ptn, pfn], np.int64)
            n_patches[lv] = np.array([len(coords), used], np.int64)
            print(f"[LEVEL DONE] {sid} L{lv} (OME level {li}) | candidates={len(coords)} used={used}", flush=True)

        def fuse(maps):
            num = np.zeros((H, W), np.float32); den = np.zeros((H, W), np.float32)
            for lv in (0, 1, 2):
                num += maps[lv] * support[lv]; den += support[lv]
            f = np.zeros((H, W), np.float32); np.divide(num, den, out=f, where=den > 0)
            return f, den > 0
        fusion, fsup = fuse(prob)
        fusion_c, _ = fuse(prob_c)
        CACHE.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".npz.tmp")
        with open(tmp, "wb") as fh:
            np.savez_compressed(
                fh, gt=gt.astype(np.uint8), annotated_valid=annotated_valid.astype(np.uint8),
                valid=(annotated_valid & fsup).astype(np.uint8), label_eval=lab.astype(np.uint8),
                tissue_eval=tissue_eval.astype(np.uint8), fusion=fusion.astype(np.float16),
                fusion_support=fsup.astype(np.uint8), fusion_constcoord=fusion_c.astype(np.float16),
                **{f"prob_L{lv}": prob[lv].astype(np.float16) for lv in (0, 1, 2)},
                **{f"prob_constcoord_L{lv}": prob_c[lv].astype(np.float16) for lv in (0, 1, 2)},
                **{f"support_L{lv}": support[lv].astype(np.uint8) for lv in (0, 1, 2)},
                **{f"patch_counts_L{lv}": pcounts[lv] for lv in (0, 1, 2)},
                **{f"n_patches_L{lv}": n_patches[lv] for lv in (0, 1, 2)},
                ome_levels=np.array([OME_LEVEL_INDEX[l] for l in (0, 1, 2)] + [EVAL_OME_LEVEL], np.int64),
                um_per_px_eval=np.array([UM_PER_PX_LEVEL0 * 2 ** EVAL_OME_LEVEL], np.float64))
        os.replace(tmp, out)
        with np.load(out) as z:
            return {k: z[k] for k in z.files}
    finally:
        im.close(); ma.close()


def slide_metrics(z, thr):
    gt = z["gt"].astype(bool)
    te = z["tissue_eval"].astype(bool)
    rr = {}
    for lv in (0, 1, 2):
        valid = z["annotated_valid"].astype(bool) & z[f"support_L{lv}"].astype(bool)
        for tag, key in (("", f"prob_L{lv}"), ("constcoord_", f"prob_constcoord_L{lv}")):
            pred = (z[key].astype(np.float32) >= thr) & te
            for k, v in base.metrics(*base.counts(pred, gt, valid)).items():
                rr[f"{tag}L{lv}_{k}"] = v
    valid = z["valid"].astype(bool)
    for tag, key in (("", "fusion"), ("constcoord_", "fusion_constcoord")):
        pred = (z[key].astype(np.float32) >= thr) & te
        for k, v in base.metrics(*base.counts(pred, gt, valid)).items():
            rr[f"{tag}fusion_{k}"] = v
    pred = (z["fusion"].astype(np.float32) >= thr) & te
    rr["valid_pixels"] = int(valid.sum())
    rr["gt_tumor_pixels"] = int((gt & valid).sum())
    rr["pred_tumor_pixels"] = int((pred & valid).sum())
    av = z["annotated_valid"].astype(bool)
    rr["annotated_valid_pixels"] = int(av.sum())
    rr["fusion_supported_valid_pixels"] = int((av & z["fusion_support"].astype(bool)).sum())
    for lv in (0, 1, 2):
        rr[f"L{lv}_n_candidates"], rr[f"L{lv}_n_patches_used"] = map(int, z[f"n_patches_L{lv}"])
    return rr


def main():
    _quiet()
    for d in (ROOT, DATA, CACHE, RESULTS / "tables"):
        d.mkdir(parents=True, exist_ok=True)
    manifest = base.precise_remote_index()
    cfg = FinalConfig()
    model, device, ck = core.load_best_model(cfg, None)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        model = model.to(memory_format=torch.channels_last)
    target = np.array(Image.open(cfg.stain_target_path).convert("RGB"), dtype=np.uint8, copy=True)
    stain = core.VahadaneProcessor(target)
    thr = base.load_threshold()
    np.random.seed(42)
    print(f"[MODEL] {ck} | threshold {thr} | OME levels {OME_LEVEL_INDEX} | eval level {EVAL_OME_LEVEL}", flush=True)

    prog = RESULTS / "tables" / "per_slide_progress.csv"
    rows = {}
    if prog.exists():
        rows = {str(r["image_id"]): r for r in pd.read_csv(prog).to_dict("records") if isinstance(r.get("image_id"), str)}
    order = manifest.sort_values(["image_size"]).reset_index(drop=True)
    for i, meta in order.iterrows():
        sid = str(meta.image_id)
        if sid in rows and cache_path(sid).exists():
            continue
        t0 = time.time()
        if not cache_path(sid).exists():
            base.ensure_slide_pair_downloaded(meta, i + 1, len(order))
        z = reconstruct(meta, model, device, stain, cfg, thr)
        rr = {"image_id": sid, **slide_metrics(z, thr), "seconds": round(time.time() - t0, 1)}
        rows[sid] = rr
        base.atomic_csv(pd.DataFrame(list(rows.values())).sort_values("image_id"), prog)
        print(f"[SLIDE {i + 1}/{len(order)}] {sid} fusion Dice={rr['fusion_dice']:.4f} "
              f"(const coords {rr['constcoord_fusion_dice']:.4f}) in {rr['seconds']}s", flush=True)
        if os.environ.get("PRECISE_DELETE_IMAGE_AFTER") == "1":
            Path(meta.image_path).unlink(missing_ok=True)
        del z; gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    ps = pd.DataFrame(list(rows.values())).sort_values("image_id").reset_index(drop=True)
    base.atomic_csv(ps, RESULTS / "tables" / "per_slide_wsi_metrics.csv")
    glob_rows = []
    for tag in ("", "constcoord_"):
        for key in ("L0", "L1", "L2", "fusion"):
            k = f"{tag}{key}"
            glob_rows.append({"subset": k, **base.metrics(*(int(ps[f"{k}_{c}"].sum()) for c in ("tp", "fp", "tn", "fn")))})
    base.atomic_csv(pd.DataFrame(glob_rows), RESULTS / "tables" / "external_wsi_metrics_per_level_and_fusion.csv")
    base.atomic_json(RESULTS / "tables" / "run_settings.json", {
        "ome_levels_L0_L1_L2": [OME_LEVEL_INDEX[l] for l in (0, 1, 2)], "eval_ome_level": EVAL_OME_LEVEL,
        "um_per_px": {"L0": UM_PER_PX_LEVEL0 * 2, "L1": UM_PER_PX_LEVEL0 * 8, "L2": UM_PER_PX_LEVEL0 * 32,
                      "evaluation_grid": UM_PER_PX_LEVEL0 * 32},
        "threshold": thr, "checkpoint": str(ck), "patch": PATCH, "stride": STRIDE,
        "retention": "H&E tissue rules only (no reference-mask information)",
        "constant_coordinate": CONST_COORD, "n_wsi": int(len(ps))})
    print("[DONE]", ps.shape)

