#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PANDA B1-B6 follow-up analyses. No training, no overwrite of original results.
Place next to V31 scripts, then Run in your Python IDE. No CLI options required.
Optional --work-root D:/PANDA_PROSTATE --source-dir PATH --skip-inference
"""
import argparse, csv, json, os, sys, re, traceback, hashlib, importlib.util
from pathlib import Path
import base64, zlib, shutil
import numpy as np
import pandas as pd

LEVELS=['L0','L1','L2','fusion']
COUNTS=['tp','fp','tn','fn']

def write_json(p,d):
    p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp'); tmp.write_text(json.dumps(d,indent=2,default=str,ensure_ascii=False),encoding='utf-8');os.replace(tmp,p)
def write_csv(p,df):
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');df.to_csv(tmp,index=False);os.replace(tmp,p)
def dice(tp,fp,fn):
    den=2*tp+fp+fn
    return 2*tp/den if den else 0.
def paired_boot(df,filter_mask,n=10000,seed=42,kind='pooled'):
    a=df.loc[filter_mask].reset_index(drop=True)
    if len(a)<2: raise ValueError('Not enough paired slides')
    arr={k:a[[f'{k}_{c}' for c in COUNTS]].to_numpy(dtype=np.int64) for k in LEVELS}
    scores={k:np.divide(2*v[:,0],2*v[:,0]+v[:,1]+v[:,3],out=np.zeros(len(v)),where=(2*v[:,0]+v[:,1]+v[:,3])>0) for k,v in arr.items()}
    rng=np.random.default_rng(seed); rows=[]
    for lv in LEVELS[:-1]:
        if kind=='pooled':
            v=arr['fusion'];w=arr[lv];obs=dice(v[:,0].sum(),v[:,1].sum(),v[:,3].sum())-dice(w[:,0].sum(),w[:,1].sum(),w[:,3].sum())
        else:obs=float((scores['fusion']-scores[lv]).mean())
        sims=np.empty(n,float)
        for j in range(n):
            ix=rng.integers(len(a),size=len(a))
            if kind=='pooled':
                x=arr['fusion'][ix].sum(axis=0);y=arr[lv][ix].sum(axis=0)
                sims[j]=dice(x[0],x[1],x[3])-dice(y[0],y[1],y[3])
            else:sims[j]=(scores['fusion'][ix]-scores[lv][ix]).mean()
        lo,hi=np.percentile(sims,[2.5,97.5]);p=min(1.,2*min(np.mean(sims<=0),np.mean(sims>=0)))
        rows.append({'comparison':f'fusion_minus_{lv}','unit':'WSI','scope':kind,'n_wsi':len(a),'difference':obs,'ci_low':lo,'ci_high':hi,'bootstrap_two_sided_p_approx':p,'iterations':n,'seed':seed})
    # Holm adjusted bootstrap p for three contrasts (descriptive bootstrap-tail approximation)
    p=np.array([r['bootstrap_two_sided_p_approx'] for r in rows]);order=np.argsort(p);adj=np.empty(3)
    running=0.
    for i,k in enumerate(order):
        running=max(running,min(1.,(3-i)*p[k]));adj[k]=running
    for r,v in zip(rows,adj):r['holm_adjusted_p_approx']=v
    return pd.DataFrame(rows)

def b1_b2_b3(root,n,output):
    src=root/'paper_results'/'tables'/'per_slide_wsi_metrics.csv'
    if not src.exists():raise FileNotFoundError(f'B1/B2 requires original {src}')
    df=pd.read_csv(src)
    required=['image_id']+[f'{k}_{c}' for k in LEVELS for c in COUNTS]
    missing=[c for c in required if c not in df]
    if missing: raise ValueError('Missing columns '+str(missing))
    if df.image_id.astype(str).duplicated().any():raise ValueError('Duplicate image_id; cannot bootstrap')
    df=df.dropna(subset=required).copy()
    if len(df)!=1577:raise ValueError(f'Expected complete 1577 test WSIs; found {len(df)}. Refusing partial scientific results.')
    for k in LEVELS:
        if (df[[f'{k}_{c}' for c in COUNTS]]<0).any().any(): raise ValueError('Negative confusion count')
    out=output/'tables';out.mkdir(parents=True,exist_ok=True)
    write_csv(out/'B1_paired_bootstrap_pooled_dice.csv',paired_boot(df,np.ones(len(df),dtype=bool),n,42,'pooled'))
    # Fixed tumor-bearing set from FUSION annotation; do not select distinct cases for each level.
    positive=df['fusion_tp']+df['fusion_fn']>0
    write_csv(out/'B2_paired_bootstrap_tumor_only_mean_dice.csv',paired_boot(df,positive,n,43,'slide_mean'))
    rows=[]
    for k in LEVELS:
        s=df[f'{k}_tp']*2/(df[f'{k}_tp']*2+df[f'{k}_fp']+df[f'{k}_fn']).replace(0,np.nan)
        for sub,mask in [('all_original_zero_empty',np.ones(len(df),bool)),('fusion_tumor_bearing_only',positive)]:
            scores=s[mask].fillna(0)
            rows.append({'map':k,'subset':sub,'n':len(scores),'mean_dice':scores.mean(),'median_dice':scores.median(),'sd_dice':scores.std(ddof=1),'empty_reference_convention':'both empty -> 0, for original; tumor-bearing only preferred'})
    write_csv(out/'B2_per_slide_dice_summaries.csv',pd.DataFrame(rows))
    neg=df[~positive].copy();fp=neg['fusion_fp'].to_numpy();valid=(neg['fusion_tp']+neg['fusion_fp']+neg['fusion_tn']+neg['fusion_fn']).to_numpy()
    r=[]
    for t in (1,10,100,1000,10000):
        r.append({'minimum_fp_pixels':t,'negative_wsi_count':len(neg),'wsi_at_or_above_cutoff':int((fp>=t).sum()),'fraction':float((fp>=t).mean()),'total_false_positive_pixels':int(fp.sum()),'total_valid_pixels':int(valid.sum()),'pooled_fp_pixel_fraction':float(fp.sum()/max(valid.sum(),1))})
    write_csv(out/'B2_B3_negative_slide_fp_thresholds.csv',pd.DataFrame(r))
    assert len(neg)+int(positive.sum())==1577
    (out/'B3_manuscript_ready.txt').write_text(
        f"Of {len(neg)} reference-negative PANDA test WSIs, {int((fp>0).sum())} ({100*(fp>0).mean():.1f}%) contained at least one false-positive tumor pixel. "
        f"The pooled false-positive area was {int(fp.sum()):,} pixels across {int(valid.sum()):,} annotated-valid pixels. "
        'These are any-pixel detections, not necessarily clinically meaningful tumor foci.\n',encoding='utf-8')
    write_json(out/'B3_abstract_numbers.json',{'negative_slides':len(neg),'any_false_positive_slides':int((fp>0).sum()),'any_fp_fraction':float((fp>0).mean()),'tumor_bearing_slides':int(positive.sum()),'recommendation':'Use any-FP wording: at least one false-positive pixel; distinguish from clinically sized lesion'})
    return {'input':str(src),'test_wsi':len(df),'tumor_bearing':int(positive.sum()),'negative':len(neg),'any_false_positive':int((fp>0).sum())}

def cache_paths(root,split='test'):
    return sorted((root/'paper_results'/'wsi_cache').glob(f'{split}_*_ds16_validsupport_v2.npz'))

def b4(root,output,threshold=0.43):
    paths=cache_paths(root)
    if len(paths)!=1577:raise ValueError(f'B4 requires all 1577 test map caches, found {len(paths)} at paper_results/wsi_cache')
    records=[]
    for i,p in enumerate(paths,1):
        with np.load(p,allow_pickle=False) as z:
            needed=['annotated_valid','gt','fusion_support','fusion']+[f'prob_{x}' for x in LEVELS[:-1]]+[f'support_{x}' for x in LEVELS[:-1]]
            missing=[q for q in needed if q not in z.files]
            if missing:raise ValueError(f'{p.name} missing {missing}')
            valid=z['annotated_valid'].astype(bool);gt=z['gt'].astype(bool)
            # Same fusion support & annotation valid for all maps. Missing level support = negative prediction.
            common=valid & z['fusion_support'].astype(bool)
            for lv in LEVELS:
                prob=z['fusion'] if lv=='fusion' else z[f'prob_{lv}']
                sup=z['fusion_support'].astype(bool) if lv=='fusion' else z[f'support_{lv}'].astype(bool)
                if prob.shape!=gt.shape:raise ValueError('Shape mismatch: '+p.name)
                pred=(prob>=threshold)&sup
                pp=pred[common]; gg=gt[common]
                tp=int((pp&gg).sum());fp=int((pp&~gg).sum());fn=int((~pp&gg).sum());tn=int((~pp&~gg).sum())
                records.append({'cache':p.name,'map':lv,'tp':tp,'fp':fp,'tn':tn,'fn':fn,'dice':dice(tp,fp,fn),'common_valid_pixels':int(common.sum()),'original_level_support_pixels':int((valid&sup).sum())})
        if i%100==0:print('[B4]',i,'/',len(paths),flush=True)
    df=pd.DataFrame(records);out=output/'tables';write_csv(out/'B4_common_support_per_slide.csv',df)
    agg=[]
    for lv,g in df.groupby('map',sort=False):
        c=g[COUNTS].sum();agg.append({'map':lv,'n_wsi':len(g),**{k:int(c[k]) for k in COUNTS},'pooled_dice':dice(c.tp,c.fp,c.fn),'mean_slide_dice_zero_empty':g.dice.mean()})
    write_csv(out/'B4_common_support_pooled.csv',pd.DataFrame(agg))
    return {'test_maps':len(paths),'common_mask':'annotated_valid & fusion_support; unsupported individual-level pixels scored negative'}

def b5(root,source_dir,output,thr,limit=0):
    # Complete rerun of saved network from ORIGINAL PANDA WSI (not cache editing), with zeros instead of coordinates.
    sys.path.insert(0,str(source_dir));import torch
    import panda_stream_paper as paper
    import panda_camelyon_style_pipeline as core
    from panda_final_pipeline import FinalConfig
    from PIL import Image
    cfg=FinalConfig(work_root=str(root))
    if not cfg.split_csv.exists():raise FileNotFoundError('B5: original split CSV missing '+str(cfg.split_csv))
    if not cfg.stain_target_path.exists():raise FileNotFoundError('B5 stain target missing '+str(cfg.stain_target_path))
    tests=pd.read_csv(cfg.split_csv).query('split == "test"').reset_index(drop=True)
    if limit:tests=tests.head(limit)
    if not len(tests):raise ValueError('No test rows')
    model,device,checkpoint=core.load_best_model(cfg,None)
    target=np.asarray(Image.open(cfg.stain_target_path).convert('RGB'),dtype=np.uint8)
    stain=core.VahadaneProcessor(target)
    zero_dir=output/'B5_zero_coordinates_PANDA';zero_dir.mkdir(parents=True,exist_ok=True)
    old_root=paper.root;old_forward=paper._forward_probability_batch
    def special_root(c):
        for s in ['wsi_cache','tables','figures','qualitative','stats']:(zero_dir/s).mkdir(parents=True,exist_ok=True)
        return zero_dir
    def forward_zero(c,m,d,rgbs,cnorm,bs):return old_forward(c,m,d,rgbs,[[0.,0.] for _ in cnorm],bs)
    paper.root=special_root;paper._forward_probability_batch=forward_zero
    rows=[]
    try:
        saved=zero_dir/'tables'/'B5_PANDA_zero_coords_per_slide.csv'
        if saved.exists():rows=pd.read_csv(saved).to_dict('records')
        done={str(r['image_id']) for r in rows}
        for idx,(_,r) in enumerate(tests.iterrows(),1):
            if str(r.image_id) in done:continue
            z=paper.reconstruct_one(cfg,r,model,device,stain,'test')
            rr={'image_id':str(r.image_id)};gt=z['gt'].astype(bool)
            for lv in LEVELS:
                valid=z['valid'].astype(bool) if lv=='fusion' else z['annotated_valid'].astype(bool)&z[f'support_{lv}'].astype(bool)
                prob=z['fusion'] if lv=='fusion' else z[f'prob_{lv}']
                c=paper.counts(prob>=thr,gt,valid)
                for col,v in zip(COUNTS,c):rr[f'{lv}_{col}']=v
                rr[f'{lv}_dice']=dice(c[0],c[1],c[3])
            rows.append(rr);write_csv(saved,pd.DataFrame(rows));print(f'[B5 PANDA] {idx}/{len(tests)} {r.image_id}',flush=True)
    finally:
        paper.root=old_root;paper._forward_probability_batch=old_forward
    if len(rows)!=len(tests):raise RuntimeError('B5 incomplete')
    out=pd.DataFrame(rows);agg=[]
    for lv in LEVELS:
        c=out[[f'{lv}_{k}' for k in COUNTS]].sum();agg.append({'map':lv,'zero_coords_pooled_dice':dice(c[f'{lv}_tp'],c[f'{lv}_fp'],c[f'{lv}_fn']),'n_wsi':len(out)})
    write_csv(zero_dir/'tables'/'B5_PANDA_zero_coords_pooled.csv',pd.DataFrame(agg))
    return {'slides':len(rows),'checkpoint':str(checkpoint),'zero_coordinates_cache':str(zero_dir),'notes':'Re-inferred with frozen model; no training'}

def b6_precise(root,output,threshold=0.43,n=10000):
    cache=root/'external_precise_wsi'/'wsi_cache';files=sorted(cache.glob('*_ds16_precise_v*.npz'))
    by_id={}
    for p in files:
        sid=re.sub(r'_ds16_precise_v\d.*$','',p.name);by_id[sid]=p
    if len(by_id)!=27:raise ValueError(f'PRECISE requires 27 completed caches; found {len(by_id)}')
    # For original mapping gt/valid; other-label analysis must use raw label OME mask, not binary cached GT.
    import scipy.ndimage as ndi
    records=[]
    for sid,p in sorted(by_id.items()):
        with np.load(p,allow_pickle=False) as z:
            for field in ['gt','annotated_valid','fusion','fusion_support']:
                if field not in z:raise ValueError(f'{p.name} missing {field}')
            valid=z['annotated_valid'].astype(bool)&z['fusion_support'].astype(bool)
            gt=z['gt'].astype(bool)&valid;pred=(z['fusion']>=threshold)&valid
            a=pred&valid;b=gt&valid
            tp=int((a&b).sum());fp=int((a&~b).sum());fn=int((~a&b).sum());tn=int((~a&~b&valid).sum())
            # 8-connected binary edge; boundary tolerance 2 eval pixels.
            be=ndi.binary_erosion(b);ae=ndi.binary_erosion(a);edge_b=b&~be;edge_a=a&~ae
            hd=assd=np.nan
            if not edge_b.any() and not edge_a.any():hd=assd=0.
            elif edge_b.any() and edge_a.any():
                da=ndi.distance_transform_edt(~edge_a);db=ndi.distance_transform_edt(~edge_b)
                distances=np.concatenate([da[edge_b],db[edge_a]])
                hd=float(np.percentile(distances,95));assd=float(distances.mean())
            records.append({'image_id':sid,'tp':tp,'fp':fp,'tn':tn,'fn':fn,'dice':dice(tp,fp,fn),'hd95_eval_grid_px':hd,'assd_eval_grid_px':assd,'hd95_defined':np.isfinite(hd),'gt_positive_pixels':int(b.sum())})
    df=pd.DataFrame(records);d=output/'tables';write_csv(d/'B6_PRECISE_boundary_per_slide.csv',df)
    stats={'n_wsi':len(df),'hd95_mean_defined_px':float(df.hd95_eval_grid_px.mean()),'assd_mean_defined_px':float(df.assd_eval_grid_px.mean()),'hd95_defined_n':int(df.hd95_defined.sum()),'one_sided_empty_undefined':int((~df.hd95_defined).sum()),'note':'Evaluation grid pixels (downsample=16); one-sided empty=undefined; both empty=0.'}
    # Save 10k WSI resamples, but warn about 27 WSIs from 25 subjects.
    rng=np.random.default_rng(42);a=df[COUNTS].to_numpy(np.int64);boot=[]
    for _ in range(n):
        c=a[rng.integers(len(a),size=len(a))].sum(axis=0);boot.append(dice(c[0],c[1],c[3]))
    stats.update({'pooled_dice':dice(a[:,0].sum(),a[:,1].sum(),a[:,3].sum()),'bootstrap_ci_95_wsi':np.percentile(boot,[2.5,97.5]).tolist(),'resamples':n,'warning':'WSI bootstrap ignores repeated subject correlation (27 WSIs/25 subjects)'})
    write_json(d/'B6_PRECISE_boundary_and_10kbootstrap.json',stats)
    return stats

def b6_precise_labels(root,output,thr):
    # Need original multi-class masks; do not infer label 3-6 from binary cache.
    try:import tifffile
    except ImportError:raise RuntimeError('tifffile required for full PRECISE label sensitivity')
    cache=root/'external_precise_wsi'/'wsi_cache';dataset=root/'external_precise_wsi'/'dataset_he_only'
    files=list(dataset.rglob('*_h-e_mask.ome.tif'))
    if len(files)!=27:raise FileNotFoundError(f'Need 27 original H&E masks for B6 label sensitivity, found {len(files)}; binary cache cannot recover 3–6')
    results=[]
    for f in sorted(files):
        sid=f.name.replace('_h-e_mask.ome.tif','')
        possible=[cache/f'{sid}_ds16_precise_v2_resume.npz',cache/f'{sid}_ds16_precise_v1.npz']
        cp=next((p for p in possible if p.exists()),None)
        if cp is None:raise FileNotFoundError('missing cache '+sid)
        with tifffile.TiffFile(f) as tif:
            arr=tif.series[0].levels[4].asarray()
        if arr.ndim==3:arr=arr[...,0]
        with np.load(cp) as z:
            pred=z['fusion'].astype(np.float32)>=thr;support=z['fusion_support'].astype(bool)
            if arr.shape!=pred.shape:
                import cv2
                arr=cv2.resize(arr.astype('uint8'),(pred.shape[1],pred.shape[0]),interpolation=cv2.INTER_NEAREST)
            # original positive=1; original ignored labels 3-6 now NEGATIVE for explicit sensitivity only.
            valid=np.isin(arr,[1,2,3,4,5,6,7])&support
            gt=(arr==1);p=pred[valid];g=gt[valid]
            tp=int((p&g).sum());fp=int((p&~g).sum());fn=int((~p&g).sum());tn=int((~p&~g).sum())
            results.append({'image_id':sid,'tp':tp,'fp':fp,'tn':tn,'fn':fn,'dice':dice(tp,fp,fn),'valid_pixels':int(valid.sum())})
    df=pd.DataFrame(results);write_csv(output/'tables'/'B6_PRECISE_labels_3to6_as_negative_sensitivity.csv',df)
    c=df[COUNTS].sum();return {'n':len(df),'pooled_dice':dice(c.tp,c.fp,c.fn),'mapping':'positive 1; negative 2,3,4,5,6,7; background 0 ignored'}

def b5_precise(root,source_dir,output,thr,limit=0):
    # Reuses original inference functions but isolates new NPZ caches from the baseline.
    import torch
    from PIL import Image
    import panda_camelyon_style_pipeline as core
    from panda_final_pipeline import FinalConfig
    spec=importlib.util.spec_from_file_location('precise_followup',str(source_dir/'run_external_precise_wsi_test.py'))
    if not spec or not spec.loader:raise FileNotFoundError('Copy run_external_precise_wsi_test.py beside this runner')
    precise=importlib.util.module_from_spec(spec);spec.loader.exec_module(precise)
    precise.WORK_ROOT=root
    precise.EXT_ROOT=root/'external_precise_wsi';precise.DATA_ROOT=precise.EXT_ROOT/'dataset_he_only'
    precise.CACHE_ROOT=output/'B5_zero_coordinates_PRECISE'/'wsi_cache'
    precise.STATE_ROOT=output/'B5_zero_coordinates_PRECISE'/'state'
    precise.RESULT_ROOT=output/'B5_zero_coordinates_PRECISE'/'results'
    imgs=sorted(precise.DATA_ROOT.rglob('*_h-e.ome.tif'))
    masks={f.name.replace('_h-e_mask.ome.tif',''):f for f in precise.DATA_ROOT.rglob('*_h-e_mask.ome.tif')}
    items=[]
    for img in imgs:
        sid=img.name.replace('_h-e.ome.tif','')
        if sid in masks:items.append({'image_id':sid,'image_path':str(img),'mask_path':str(masks[sid])})
    if not items:raise FileNotFoundError('PRECISE original H&E and mask slides missing; cannot rerun coordinates')
    if limit:items=items[:limit]
    cfg=FinalConfig(work_root=str(root));model,device,checkpoint=core.load_best_model(cfg,None)
    if not cfg.stain_target_path.exists():raise FileNotFoundError(str(cfg.stain_target_path))
    stain=core.VahadaneProcessor(np.asarray(Image.open(cfg.stain_target_path).convert('RGB'),dtype=np.uint8))
    old_predict=precise.predict_batch
    precise.predict_batch=lambda model,device,rgbs,coords_norm,cfg,batch_size:old_predict(model,device,rgbs,[[0.,0.] for _ in coords_norm],cfg,batch_size)
    rows=[];saved=precise.RESULT_ROOT/'tables'/'B5_PRECISE_zero_coords_per_slide.csv'
    if saved.exists():rows=pd.read_csv(saved).to_dict('records')
    done={str(r['image_id']) for r in rows}
    for i,item in enumerate(items,1):
        if item['image_id'] in done:continue
        z=precise.reconstruct_slide(item,model,device,stain,cfg,thr)
        gt=z['gt'].astype(bool);valid=z['valid'].astype(bool)
        pred=(z['fusion'].astype(np.float32)>=thr)&z['tissue_eval'].astype(bool)
        c=precise.counts(pred,gt,valid)
        rows.append({'image_id':item['image_id'],**dict(zip(COUNTS,c)),'dice':dice(c[0],c[1],c[3])})
        write_csv(saved,pd.DataFrame(rows))
        print('[B5 PRECISE]',i,'/',len(items),flush=True)
    if len(rows)!=len(items):raise RuntimeError('PRECISE zero-coordinate incomplete')
    arr=pd.DataFrame(rows)[COUNTS].sum()
    write_json(precise.RESULT_ROOT/'tables'/'B5_PRECISE_zero_coords_pooled.json',{'n_wsi':len(rows),'pooled_dice':dice(arr.tp,arr.fp,arr.fn),'checkpoint':str(checkpoint)})
    return {'n_wsi':len(rows),'pooled_dice':dice(arr.tp,arr.fp,arr.fn),'isolated_cache':str(precise.CACHE_ROOT)}

def b6_sicap(root,source_dir,output,run_inference=False):
    ext=root/'external_sicapv2';dataset=ext/'dataset'
    tests=list(dataset.rglob('Test.xlsx')) if dataset.exists() else []
    if not tests:raise FileNotFoundError('SICAPv2 official Test.xlsx missing from external_sicapv2/dataset; cannot claim official test')
    if not run_inference:raise RuntimeError('Found official Test.xlsx. Add --run-sicap-official to run independent index and inference')
    spec=importlib.util.spec_from_file_location('sicap_followup',str(source_dir/'run_external_sicapv2_test.py'))
    if not spec or not spec.loader:raise FileNotFoundError('Copy run_external_sicapv2_test.py beside this runner')
    sicap=importlib.util.module_from_spec(spec);spec.loader.exec_module(sicap)
    from panda_final_pipeline import FinalConfig
    cfg=FinalConfig(work_root=str(root))
    result_root=ext/'results'/'B_followup_official_test'
    ds=sicap.discover_dataset(dataset)
    if ds.get('test_xlsx') is None:raise RuntimeError('Official Test.xlsx not identified by source script')
    index=result_root/'tables'/'external_test_index.csv'
    df=sicap.build_index(ds,index)
    manifest=json.loads(index.with_name('external_dataset_manifest.json').read_text(encoding='utf-8'))
    if manifest.get('split_source')=='all_images_fallback':raise RuntimeError('Refusing all-images fallback for official test')
    if not len(df):raise RuntimeError('Empty official index')
    sicap.run_inference(df,cfg,result_root,root)
    ps,source=sicap.build_statistics(df,result_root,cfg)
    sicap.write_final_readme(result_root,df)
    # Our fresh 10,000-resample result, independent of the source script's original 1,000-resample export.
    agg=source[COUNTS].to_numpy(dtype=np.int64); rng=np.random.default_rng(42); boots=[]
    for _ in range(10000):
        c=agg[rng.integers(len(agg),size=len(agg))].sum(axis=0)
        boots.append(dice(c[0],c[1],c[3]))
    write_json(result_root/'tables'/'B6_official_source_groups_10000bootstrap.json',{'n_groups':len(source),'n_images':len(ps),'dice_CI_95':np.percentile(boots,[2.5,97.5]).tolist(),'iterations':10000,'warning':'Grouping is inferred by filename parser, and must not be called patient-clustered without verified patient identifiers.'})
    return {'n_test_patches':len(ps),'source_groups':len(source),'original_test_xlsx':str(ds['test_xlsx']),'results':str(result_root),'warning':'Source group IDs inferred from filenames; verify these are real original WSIs before interpreting clustered uncertainty'}


# Embedded copies of the user's exact standalone external validation scripts.
# On first use, recover them under the B-followup output folder (not the project code folder).
_EMBEDDED_SICAP = """c-rNCYj@)|vfy|B3M_kPn(CJ5M?WS}k21$`yAz+dZLh6P?u=WPho&ftX^P~Klx@Z1d-g}{FWg_U^#DNdCCfdTd+t5kCy^x*C;){*p{h_-m}SXAxbAFOEVICMMYu?ltPoxtCxuspNt|zOS(deRQF!sxi;_4H!Cet#u@?<)yeOPfA(4h@5QXq^mSp1M=wScsX3yE$8qdRA%)%%T@HdQ$AjZ;O6s<%aq+aF~Py>6AEo10I6uBuj1D|P-IqAv~M}?RsK`u^C$6H0_g|UDJ7RfZ6twb>o#M%DI`+a;6L|AyZCB$wIs@%p=;!SgrF0Z1{S6veDAGP-}h@nQX67Rjj6SrY87g>-m7v2^0^~dq}Oq_f1b?^?r?e%cPndcWd7lf`2XiQXt%Dn1{We&Z>9woC`=!agU+Z+dZAz-jYNT|A<2eI&OyfDJXu=eYo;Lc)oFUkUMx*E`6gQ*<g6~J&(oR=x!^G-iErl6QZf96RvMUZcL!b{UAlzqy*MIh^q&-af`2Gbx5Z{XL@-rSpdFmHvIT?Y)`Pm*jJ#xOSn@7Bu##x6pA3#fn=m+&eVS5PE_NmwpmMr<!$@qon-QQ+lCJV?C)=|n8N{0q>>T@Z=y#7<A#<RXUGR1aXnKj=J-Llj}2F9{`o>WSMd#AXoN0Lq_lhYRmI=!-mAW_~dE<>E+q*VkEa4ZZ1$>nOSMqQ1CFk|GD(rTBFTw3My-+=`#Xvt<sL8Z5jb3-4&Ge_ukwNa!~(amylI7CAQmO;7kR4cI<Xe3ayQ6y&*>hu8A~pgD<_gmXVxT=6WO{2-2xem*4nKm7Ud{I7bMhyv%IIu9%2oM?uY!!&@khD$<t1>iLRWTta34;-;C9HbY$a)7p`1X<B_Uy&qqVk`d=rn#mY=u#R5u&x%t;ws4UKCHW0kOi?1WI!bmqtqSYd;UCt5{QYLFb7<0=YWhY__&6lWAW`QxLStMl$XtPnb8W7sB(xkY$=|RT|=+)K>f~_S80~`fH(bQrN6>J=TUg2e*Ov+qrSuR%+>cqSIA1Zb3lI70uxt&2b6oQAA}2C3(*x^dHxsm;I}YE{BF^}fH?eR23RT_W`()pgku7FmL$>PUEnVZAX=&cG>HRH#m=zP^rBd$&;|8;KVJ34`_M1?;yBFVbG(G<(Kz#O_SDySxky*SgVm9$N5l+se3nib_}S61f_%iwf&RSDyj%J8LlPIVBY&SRRAKzC8-+3W^Q}rFt`8g{S|zcY7po|6H9v>8{iMV_7ls8Ecnjvz)I0p5(1HbKz#F0xh^+hppB}(646nC<nXiVlF!lWYeF+OC-wN*h065(t{lP^B4Fp_Nl@TLx0_&2Vx#x%f>(k-I_`*H<cy@X|b`Q_bPtT!5uoXnPtk_wIz-(>3-yiQ^9FE<i_wda6_U>-+=9}qF{>svCU`bybou0t!J^k!t|KlN+myEtFe764sy_gq8nh&?PfkS)FLT)(@tWbKA2SvVJwzUlae%k-xhvP$6!TAw*{4qhw=UF(N!|F;CSfiour8mvu^=5BxZ=El|Bcq_&W;?KF<)*lQcBEO2fQM-UdpEM}TbhCp_5^PU`wHylJ}}8CRq%XV>b!$^F@VbgSV%GNrmuHi5d-sp^{*Dm5}1phg(<KQah3(ZpY%rkin$VS6Mqg|8O4VT_CA0&hNrNdLWxi3#{eyglG|K&`nlfgxqj>|a2dkFzG123od{=Q8S9<54<KO_39uU&1&Smy_!P(@V1>}OIZb?E)-jGy!oJy69qYqJ9{O`+&pEK82rB?YwdqS_%MCRyU^C~fm^kk?Zi?HuF$_w<09Mt~N8TgiAecg%b70%f7D05+K7D_Dbl}3sFJOu<iK=Z&(UhCNa$yep<=g}3oerix{4ZyHKxi3Au9rLcD<@bEZm|oyj<@iB1G(%rCqgM{a<Jz>1*_L@Yj5{o(B7_UZ?g5bqca!i3Wp3MZeZbZXL7ka*@B;kCl^Ha2M6N&zlzt+4lKmO5Br~v$L{z0<AWdJ-R@ic=<NLP%>Cu`{3lq@@aFB-_ot`h3y_M=+@mo++uhmO0j4kwX2J!Uw|k2F5Iy6j;=o+Z@QZEz;je=~ECzp=ia-1~{KLoLA1<t(s_Ox9g}#e>n~mGWkamf_NUnYz;>IB1;w#+mar0SxnTDCoKl0HyTL!q*!^YtzU+8C#suYV9`bia0Bz5y82%UG-h{NB-OJO-s%;E|pxF%^3+wcsg?H1+^M1ll&%aOG#W`l369&*|l_X+=Dznv-UN;Y)3FJ@RHM=in2{V*JT@PNIdbTNg#j`sQj#_TPlVg$J9X&^Ia7A^C+T|P;2XO^#G--cH}8F6Cw_$9o^f;56{nmdW3(YTzVyy}vOW8ktU2H&B`7jduko`V0p*q;N(8Bsw-jeG?EK_BT+uk06a2he$BHHVO%tEpH2Bni;a^@|u}F&kvn%$9ZWJ5FP-*QX7pY(v(^iKJ)sy>7VG+LFY<trr%s_a+%|kP1QN0kD{9K13CZ=vPrz1`q6+b$M~|Ui`9uG@ghHu5L0yV$Aml@4*r00U!|>Ckv$a67)mvI1bPZO{PJJzykpQ;;k`52I_dpL_v_+5?;^fA9wHk@Zj|1<nUm8_`d8VFjoXfz4WeuhdKh}ha!xZ*u@I>uJ=yjOrpz;!Y_dh=pKsifjw!2*9ZtDi<xx~A{-U&Ij#%)2<XKRLvare9xMko9hhnEKk{kdZJHtDKP$+xI1ziqjl-A~8xW0YBf)3W;3o8GXoxn#7Z(MIdYc7n0Wgxx>X$&n?L<8k)|EG9)nFR%F3OK@LDa$`;#K9>-mVEqKYmj!_4YOX^hxgr;IM1qW`*;J&tX9M5;i+*)r7VL+D`+YHpnFBVW7n4$}JC+c;CmzGessdRLUfXiOE($fwb>{fME`h9_g#Zub#{diPH@e`!>nGxEZK=Y6Z;VzIf3WK+i!m#MKRp0_n|dE(#`DBoc!_O~zq#Y8D%eOD@)*g4I>xWz!>&3$kpP7PXpQXz+$L!mPn}YE;@o4%JvwddxMRR9Wcvgn@^N$khr`y+rOJBxTp=`k`fEoByheNB2XcxLL#JMH;kVMowjvz^ZlGg5nah#FdDX$)mf1RSL5PdqNQ3gc<Ba*8#AK<I{uv<NdQU6nU+l;^bJ(nJ2IvOMadzwvS+XVp~{bE^$eU_<0Z1fs0Qc4=pac>82}bA}v`9$i6NvksO#sVbCLug%rI@CQtmFRHO)10sAJgK=2&gbzwf*5Y$>!_9pjk<l|aj`fOB?i{ix#G8{+}f_)tjb#F<T%^C_*wSi?zRpBxHh2P6`it;{4Ht>*TBMtL`K;qg>aqYIEr?+OwRmCNel6sx%s9H7k@?6i#+;hJKD;w#X22GOze=F#h<_8B#?^02Rq#=;!SQhFtx6MaZcpZcOYxR20Jh+>N*C4>!z02X=&P2AT6eU+15bb5Am=N%D7bIFW6*ayk(Sb%3{Cln~kXnmSN*Kv+dcTv}$bd~@ptpok0^7B-MPSadFv-F~qj#5#lbIMO*?a)~>m%O*<_%!N0(75gI3p`j!G<2vepsG+C&BFXN(Hg5tU*^{>zJY`GEn7T^-TS#yQ<cxWcT;AidfM{U+?U0qZSGKW<jR8De_V;mU`y^blK2f5iUSR?XxT=r9v;xLmXNh6gUGsw`fVEapJs#TDAzLA%F#36q&M{qG+dj3`&PnnJz$}Hb_9PP|`ExH)M9=d>NzcbBOH$dp3(pMVjq;<;<!X2fo~Hw0J=Mf)+26))O@cvMY{sWm9`l&kO|uOG^!cX^mI)<^Z@Bb1K<yK$|M4gly81rKaeFxhuaafCksl8{PZUUrvrs_fc;Aba6Nl_p)r)UdHRbs{>^N>dg}6m@oDs%ySSmOB_-`P3t83O%zmERx}d)`#(m)P+&{=WmU&ZK_8piz;a-9x=;za4xe%hL$W}|)Gw=G8Dy;uU3h3JG1cO@44}&8k^s8U0{$#}F{yIWmhQE5q9^h2OdJQ-p3e&uX!;BC=p>HFjOl?i2FgBK%h5Po6u|=CUIo4f<24LHynZFTg}0JsCSudD;OPkH77Z*olqgA390_VCcy1uM0x!&nI^%k2NMgJ)O~7wwjRo$$H2yto!_;ph>A7kh2GhSlQ+RAJy06&+A2b^<8h!%4+ekqhSOAvGV1X)1_zLqmpbW*yk!C^81wWDkt1JMU7K|aw9p1U^`%-rQ1XMTJzecTOD6EgkZ-8TO`;D_B>|a3sN88%T7{q30=beBDZ(qL?cW+<!_~2<Ecc|?>XU9|s_I=!-5#WpM7v_s|H1Y+RA!4Alk_$Y@K_R`ktd1B4Uc{qhK1_=8hg=LL7E8=N6Z?GN=0&V1(Vhk*2RvN?6H?d7y&3ZSt3dE107oY4Ur0wkjsf#05O(F69}_v9<k9gOLtNl7mE1W}AkmQQoa$Zn1$XF@k@0(1zBdNkDjflpnH;#oKG6=D3^{bdXjfpMUf~vKy=a(%dMs`uQ`%SvTR$Qg08UAj=hy4BJp)c!&Ujt!1AuEe_2ZJJiIG0Y!boF4r4@=JJPeI)w2Q(IPm@=>_q2=CPv-}?sTsZK+SuCE+161LS115u?$wI6aKavr+wQ?1HxGXcnzMvwC}O7pNU{~u9_XYTkTA0la6Y;(TX`5*4@A0UyWfc&273R{t;Q-$Otd8Y4~hLz1t;)WYD5UzY<bwPBdpOH$<&HZBDpt87;+Czdh2_MiIM!<UWgCOSSo8RfKnoe9wYnJPG9VPv(sxY=HZzKC>kK@1-=sV8YRh)n+;nb0BbByZeZZsRq>g%9w2o<jz3q-%Mgy8R;VZn6Dh&6WpO$dHtfxP+`)V8Jt?K+vmm3@o5U<AcNVAZz@h`C`$~Hc%;77v@vb3bh$~R4zXVgOvw>a*&OS18QF<3^E_&zr=g~D90is?%hNlm((od4&1Dep3N+D0zUbi5?>TTZm?pxx$)pgq4AH!6S>_Y3gz0MlY>P7;7+rK7*JUIP$c6>NC<^B#1@rY@wY4htI^COx;S3PYDYb}vEG`B8DIWA9%-D@wopt%y3cV%_WYmLp~j8>-__R_tbx82xQ26==31{sRlYz3K0uP9`?*c2`al9VLjvn=Ioi1W3(cUPs#4z|--Od;#G16e^-MHj4rHRNtWYiHj{0itG2;%O{8C5w}2h36Q+h6lWRu2}&K67roqKr@M*S=;vK>C4{#1SD&Smo;dUwd-)PezmEwASEoE>nvHO_HM7&jmD0F9S8AsF<&<_)sis}FR9s<fk|kSj>S=)Z!$l^8RPTJKKZ`b>pL-`&we-7;t-XIQQ6b(vP&pCvc5-UK1Q>-??r*M$IydmcoTw3Jlfsad)*BMk9I~gt6>PVE8MMTwL_qxUwK(K?qRh75L%{d7ZPe;{(KpK!7ZJSHtBE&=ESDwBsY$}5ibM|Nd9?*q06IIs=vNu=V3ylpAPv8a-Y`odBBTOha(^uuCcy~iLI9-QC53wm}bTAM&j-3DxMx6eDnN#&7RzTuD<EO){?=a-Dquo>+1yvNh41d8d?mm-J`PJYcr2wD5$C!@}Sm}KB|8^I&<3hl$PPPDq)YQpq<@q(@nbzH+0tSqRMBqda%=79({D^fJ4v4n!?B*ptLp)R}{|ndJj!<Sg%mGE7t!B@}ra)w_3XT$T@NItk|e)*Hof@Q}*w*93?l5Xu$@!U}J;$uzz%nt{0WjZ*n^PBr)y^^x!XB>n8U18rJ#Y`24TU;e6Erxa36}AIGmbG;e31-8#ZEal(gI+c`_(<!KgSV<^ZeLP+mrby%hBX<`=Kw!lQ*m`l&{*%IC6ml$UeKPNxwlq{HZW++YVju>a?7b&gyc?-xe9xQ@I0$uUgc_gN@IKc;#E$b#*;W;f?>ZM~xb>TJYptTz$Oz1;qHxFI=CComC8N3v`P4m}gwc~aQzzzIW2j6LsA1<fUpAXM#6i7IM&E`IS;7NEVq`ws%HOk|%4HT#+D^JapSM$u;-H{vEx*gQok0Rr=vLT-G0CG7@N0hTTvyNd2E28CfST5@7f-IjuR6(4o*qRg;6%;hj5+uc-vk-d%NN+{<fSxa}?Amer5j&pi_vp(AyHgaA`+iY#4o1a^@yyhG#tZEys;%68b2JL!SweYyM5+mFGN#8gvN@PLG)bY?Fp*{&_#q4qRcl_m@_-oq@8cX#cq^T+NETsjBy}&s>Wl8JRH^^$S3W80$zv_TPB49EBC6_y=XB9-84Tk1^T7Y&VtM;AU*%3faTUA@t-%7_#ZnM3$eE0q!LNhA<EQSBsp4)v`kKQAzi{Q;8U7nGxTfMov)*+>$&q?RXfKMiLMMxmwy-5x6ZzSX%d6+PfcoqZA-y2@$bbK2%<T-I1&o11t}LT(&?#sH2-iod8XOFKM&W24MOq=pMh6cx83TWIB)~JENuKB*KYi~W>>vDih$pJ=hsrM^!jLfS$S%cT>f;m(9t;(v^I(uI<CvCB&Ly_xIT>1nRIyVGK{7juvxA+afGUheQ*F(X2(<q`P=31FMi$5ZC}Hm7ZW2u846EW&AEgMxM3|GO7aty~lDUcwHaIXq%DjU8(<{@-r*$Z4n83D$<tUm_G^D^-+POiV^feFY^d`Yr*PaVz8|&Q{*l}1Mm7fQ6j$RM@lF%eJ=|BhYtN90c{gmysbhCL}26#}oSBs^xM5-H(96zb!e7ZQ(3Wm6sCuN`GHV5>r);+a~u0u6}^+=17JTRVaBdoT$WIyHfu7b%89B{UbgXD4{@M%KAN69K`5CbZHg#{}}=45v!wc;cVQaz^3QOTZ=N)-@S1^3`X%?7!m)edS?Kcdnx^U|tu5o<y=z07U^?`^eBELDPa&?w$|f>-WIqqZ%c3(@|M+`}!hAdp^^cAohamOjmkCPz2ffOg6Y2FS>W9}P<uLX3k=kKvTJ_DGN!lg?5xbT-39IxXej1Au^WiplbNE`vNtR4(lzY;nuHd``@>FpiMSZgJZqahaAbmuh@YhS9LL^}_1W`bXmlSx$PBEyD-~waGL}p!#fa)CNbNI-01}C9Rnwa^yA^DmiClKDOR4H)a7DcQYS`7Ey?;TO@iCs;dyD0=Z1CIW1JF7kHwl+gck8pamgToDuLr?KphI{jWY9Zf$I(l#Jx`g2d<!)%N#r0DqFkS;j@S8zideqfjHnjp^z~xN7wie9SLltyzpidmSXcJ*1RAHs7w#)T4!<M^90FHZ(i@p7Zl$tH7hiZ)3VPifX8H-tw65zBTf6-M(3eg(_GR4l8DB95J4Wlf)2cb)*ajKN`E_!iy;=oji685!0#h94Sgl|9`zRI7-S_lAGN+;@<i}vXp#&xc?qaORBL)26&My0yel{-&x*3@k|NipljwqG(*{0N^>mkO1TE<<6R9TIkMfNnqC*oNCCyz0oaAugE>c8kW(}02vg^A4%sU8sS!8nRT?>@G@{CFJ<WvsOdQQh1hHI15OoACW@C^;D&;<j$9@YuH3EqX4-c-T(M$Mq%xiG95;?j8YQ=toQB}p`rkqjM>WiwNT${(AgA5ayTm@)nLzQF^7N&NBo=N`|WpTM^^h)>|Vf$YlSJlPrv}O3orr_UWBiMKYXg3M7m9o_m+GWk?3uRx$d2X?c=3fU%erupeb@9)l<#@AZZ`^FqRkM_+*jVw%;)66rsnC5LHzKW(&f^qm#@T9O*7l&^!qhe6Z8x_MbsG}YgVKPovHjri-^S<r2jj0WAbh>nx{-WAs*pOd{acvgp_N7+P*27&=D*Eagy49Vpg!~4%)zfwh+$S+El2lCYoQ8ttwj#4OItCu7OdZ@wiq55MRVe?8Xs^2&*%~tsSqe#6t*{9=i~8YZDwQH@EI^i==Xs17&_Zt1Rmv}06Bwm18H%iw1vSA8HH^JHcDv<8d_7(sy1mntx=gC9RtDsI<2p!z65psC3sYUF(5qYpruo}DObqMAy*~DQMoEe-7!-%T=8>IY}UJp7)lc&Jh}RnRi`PY8;Zzps^Oor79c8j(%s`->DNI|A9XBgsAWNyUbfkZ{@2Vv`_cOEE(<9(Y&a7<O%D2R6<J<xI$tS0eVz2v89<IX3S3N!N9|9E*H(;sVO9+RE@R8lal39i>$011<Tb^mI20!=^FJVulTD~cIu4M}NiZSD*5eM`2gwq&5S?{F?PSmICG>~qhNgHGKwF@A5_$EssWZnEU3S{AV|JxA0J+BA<p4Dtd+Nw>JHMvaRu3h_&Jt;iB{sB_%jR~>$m@l7XA^Sz;!Cg^QP)+1hqaoG5n7M=)GjH$+6b(t#7a1=t)yW>!jnZREK!Kbm^Q6QWo5lW<{~rcs3Ke`dUEcf{7w;7or)|3!zCsftc$-<)WIe$cP3bso1@qn?Pn~JdanjFlxWaB9rEb<+zf$*XZrhtT27QC&{cvBNTtzo5!=(*P^41_lVpC#C=^O(jUg0aq)0$hF;QXm;vYsNbV|9sGP1{9av!OlrKUjT22l@5Vo*QmFXM#ph0#WMuHg3+{)WP5dpV+hI&-+&c`X%z?nNa|mR=E@R-&ahmnk<-Gy%Z-{B0iU9CUp#*wwAox>3Tt0rdjKklCd_alD*(C-NoobGBdY7^;wjjSG<STG%%|A{hi-Lq2hHT_`Vt1<5kFh+9iz7jN(lcBhi~h#LK_99$J6?V|??1HYMMs0t#;vZ#*ulF9an-Y8R~OWx%!a0B;lW#a%!cFpNCXpv(Amw!lXm*3^Gm+tWtdueCZzuV4>zx3?C4BgMum-v70FYa9!hps*N3;uFf#t(F&$5wIE@~-m{z&Yw|HQnSD>(buv{fc+|Ua$7xmE@3%y_)ZScURW-WUuTBH9+TU!Qyhb*mKAw-8|Wmouz`ZTYIG&f9=Vh?CCSitArkAoC*~);h{D?l<i%uK#-)=n(U~u2x58UHPhOH+FPJwPM$^~XzcEJa1EP+YxN`xE1%GL6;dow1`nNQ*q{>1JnCYGF+6b?*7-hr5BvFzu{D_7YG{W0L$OT`yQedZjGS_8EYz`O-MaiHlQ)bM>#ediS0zqD=Bs~5yWnjeS`SUD2}5)+1@0B@eI*}Z`2(HKr?Yh;+R@4T!+%p&fp~N;YoKfEy&T47{;Yt7C!dA$xYe;F#{36U`|`y!zf^nEq$+0gMc2z<&0CT`r1K9qx$tm3-1Ei?+f~W;jV?XtjlR8$yjiyQu=y_NerfiPn(OXf*&CyGxDv`b%o025$T<Iq@^7i_7?aG>ek{GpB9ymPK(+<T+GUiwYKcpCvO7BD;|QKoQqU|2-b)7{T%JUawC@ymy|z__04gg9sVKtIW%h#vPlr3?4haA=+ZDE+=4!G&+Gpx%S2|O#lwvlu$h0w_jg{MEk1d_Nd#r>q`8tQ?gf=PJGrcs;D9h?xl&L|Kc#wIwIENC%C4i_11JzN)>Ie&Ju>`Qya7fiO4#;T<_O5hC<(Q}$Y$<|^Wr(;YL-f81emkgUo0W3VkfiqZ)!TMdgibCo<@5Hg<0LPn?<8mDZZY|R&IKr?WQ{hj#@4o2NT;rhMwMPRvN;-aLEt;iv|G_c71odr5!)2@I|u0!G^O0GmN6DXZIz>ltX5j}sJ}^|YY;J!-oTg$M3b_zzbKVSK32(UXjiXtq;kn3Pt)W?oCZr&)-;r9UD6G609o}Ee=T9r<p3rdrB@-(XQib@M@y9f>NQj8XKRO~xOLYT=+fRM-ZpxHm{l!QgMLhrVknw%KJtAyMSYLKwiY14g9CLMF*=55!X*}s5{A4Y1;JVYlxE@8UG4FT9~%d<D#MG~6SRnCBNFs31ALX7RoAA`(#C$ytB=AVP9=M}db2_?_ies-zzZ`x+O@Rz-GRQ|K%f<TN>VMk%PKb|;bYFSjEuFY(tG|K2rInC^yTU#ofq3L7e|g9*ri4HCb)Ox{<h=1sXbp+p5ua168iK`s*UvEYGs><@A1<dBQF<~O*SVNIH(pS{v`mJnB)yF9h@BpcD&fLdk%VAvmA-9R*hHHRdBZmV<6^*hho7jO1#3>5aAVNwyi(*?&@~o6+v0mp{&}WtU93FhSOqBtNqS?v%BZ)$c_!Dm$1dq!)l#*5MIxVYNZudT6I<`QHhyQjmfmjQOfO_dfh9#u=nOIKoi^C*`DJ^VQOP0k4IZr87-{ZT3FRu;GtET(COe%$9dXu3G}E)Og)DaI^S`2XDP-8g@@v3>4(^nFS0BkQ)U?jWX`O91Bi>E9Ia`gs8kffVx%0DGSj#5&fNEqi5WkQQpf=U%yR@|FRf&tF-i5PXF}8qk9oH$3TP@{Iq@<M|Gvazqej)r#sPBdlJaV)b}Dbz)C=xBJdq%~o%!N&_-1Ef)Z)umaEHUoCNSR?^QG!#D`NB=-_P=nzw3Kf`uZLha^?T=kB9n381p6;8C}lmNhM@eQ5&79;Ra=wPRD9u7hW9BP>km6e|;jPnhP+itmO%cF-sNwwZ#7DzGE`lRNGRWWrG5_H|UFU>S=@xyAv%xG|6Pr%#_xBtm)1R@)3xNoE)qvgoVG7@k_TO6SXS4V_ViiO2n%3blIHAfX)V6j-x7qjHR0)n+oQZZ_Q*(d=}nb8WkqJK4(I&U<>vrw8TpN(1o9Fbv#4@uB1=PL@nJEL;_EoWDDD$T@TG$Ka^7hmcpY!OoZJR6c0pKOrTXU$pXRww@F12jwyT7IW*RFkX4cm-Cs^U%TPe^h7$+755iUMz}q->t^kp9JOzYiD(?ObBT%!thX%&QGIBTrQ!#tBSBKKsZg-UpY%6985>%M^rmG^cIoJ~O#z4<T+^|3Q;uu7J7X#>e=5ZA1F@Q!i7lyIRw>^>c*QURBpH@|Gk0M~|>L^JMLvi2Qj#89mY0$({$ICKrWz)4ad?Mb_L%PHxD{SlhhwqV*OmW^v0ALA3^9{!dHCKHnne}BSfO1{FGl;V-<9Um^Ldc;fwoW}R%Mn>~?IDf;FPMTcNZgrzSry2zo^Cv}0al1OZmJ-aS`O4vOF~`Lb;dz-Fd5v7k{4j(jp`;o;xhL(E>#r&c^i>;32@f*Iq)@iemj(mkkADme7;kTxHP`|H+xRFz-DcB^T7v4AJH|?{o#E7uNMdV$A_gYyDL?0liK+H2NLJ{;uaNK**6W#p(mHay$NPEvay&!#j10_wZh*5d(jv6ZC}jU1Ri8*5>X`Pi28dnK0J3%4)@OwFUD0W!i`j7F3s@Ly1>n8lSbXev`O#Bs|k$0#`6@hwUA48eI?gH-dG6$`_k0nCXcNGWye?A5uh1l+&h$W@roUa9xt^R)IVT5&O$$@Y)<I4!z_mGA>jO$NTPHTjbqLN^t!}5y|~8Z+@vZRLaAi}Zg_?yA0P>%PadsA;nzVN;DvxLzadaIG)5@pIxL%vWL+gH<Mj<{^zVZi8ikA_3?h^yOJU<mu2(whFis>Clz%V;k%Ec~V4{k;s4)kfra*D}Gwg~xyOjey>Irs`%Xq*=M|)Uu@AaGBlYqb!!h*<cMvt0U9#*+wx=b$csep5NB7J0>)P2sXA*#R4MPYz476;+^F(};&2bRsgyb=Q#q``;`e%W&<bpdWzw(bR2x>L)5%qrK*WGNQ}R_AJIk_Yx)Uy#OQ&_2mEul7(g+I#cHVOIlEt;{Wk<vFBCws0%$u8+b_sfO0xuhjcVOM@nlxa)J1eL;A<29`;4s~9eaw54RtWr4BcI$BzL+Pqa{8)iimF*%42zRj$Ao%Fth)O)C058$S#1O2`|Adr0iv923wDI;=B|L@!lh6$R`=G`FeEVpq0CR@gttNx~0Y~$SWDhRr!)PQEpy_Os#n|0BSt6KKk;K+jZ3%^&z7ZBA~^|fusmQ4J*U>?p!;w<CE@>`H4k`496tL7F%c8E8s&ZYey*F{hKQP}^1y~A(R2Q+A0?qMQR^)W%#JU-DCVZ-<%Zl=>pmAAOI{{$c3Vg<v}KR&D0Y;Er~*8X+}GmcJl%?~Gy&NiCe+l5MR>HkUD=7+P&Z5>%Arvxvm)fZ~E_{TK2|29HzcNS+hfl*MSj^+NYEVHSF$6m8+dtVL2SaiLOeaIgb$no5xU5%77gVI3z<=67uqf5E($qQxFJ|@kAyX*>4zhQ<iiaGo@*w#+=kNE~Ui`C@Ht#ZAdx@!i6LKhq3e8`)FHgBS<wP3!|4b{}%|It)?t78mf`^wnNU&~ow+l!G@<x=%X9DI7eKaq`$?)fiTUkJ?TzJ!Jk&2f+J&0qM|%-XX5@eC@1I70#k$j;JnT#scmdX6J`PNp6`gQI)>b?CgFRqZ=6*Gt_+mB;b%^!?%S#2~QIefho3PSgZ3{t+~RA5V|pPjp(aI?1(${YfpjJR0K&aF?&w&2^GJc{@NRP5+oz5syMlE!;<&Q<v<(^pZKo9#^V@Db;M>^4C_xWA)m#9BJB-YyPVOe6zrJr|3b@uy*+yn7b-YY2+J}&|PDE3A4npt=g!v-d;z4Dzo!))RHyo|BE2v^{1^e@sT{N%<#+r&!%;X$2B~KuiBcFF(%rsJ1`96y@zen+-q`=ELeCUhG%j`T>*;8X=Bul?yHmeVB|(^<r~J8Stg;#6BJ56B=>o%Z*;${_kEd0RvjrwtQg@+<@>TuBX)`J3m+d|jCBMPd&$ZTKywcYBWd1DZNuiHS+dEIIa%P{uy$)e2r8-BrJCiKPyXf0%WxRJEO&s(;}OKtS~6AM48ffP0)riSi>oPU+}W^UPsvDrQqllmh)#!_bbW`~E1Nt?s-~;6?U<5F`d{qJ6x|I?_h0G<XmO;Z@3cVlABDL4M2L?YsT(7@yK>SncG<FP_t46#WTiNXUCLgM>wvbKC)j{n2kSqju_vUEP$SXsdhc|D<+Jw77<dGF*^wM*)8fMCKqL-k8}ngcHX{>v3p{n1jyfcZ67L=SG)k`7otaM>?$F$H)sb>9ORjQFCV6<hNW!Vj0D5c}06hBX5<Z!JdY8k#*qLm#1|TA}BO&z(Fp|}8VM_iGeNABeF7HbC(I*k2Vsb$L3aH$*2O&7t{7`;WTHv~b@JY>5L=dFq%`k`bg{`2$Qt4~ZuoBnUjgF7d<l2SqfJXy8_-<5+h3zCwqEd$<-d15jR;qU0yj=oV(3L5us9&LtnvP#(bvkfuG|Y#rlyHI>+-TR?=@LWy$_aZTiEhOq2((%A!KD9KJ#rK(Yr)TTbt-N@Sj%|ucw5q=sBBFdf4l3Mj!0O|G#RhUS<EK9p|ag&?WVoQAgnqdtTup9o}?-m`pMdcwJsuEugVAOsu591T9d8?Y+lA#k+uE2E?heHuU59nn8Iu1aTugcv_UZqDv!f?@nW%9_x7se?L#|{SgYvN*GWelD#7TmOl}H<Pi~Nnu)&$IVz5bRKYM}pvmRgxL{B>f)-Shi>+IwQYct3-o1n?)B-~&lxD|JfZY{GheAKG^j$0U>i37TtQ{Np;XD$Q0&?~ql5jx31)2&>#iCZDYq;~);L({1jpsmmyZUsm=0@WbO!<KM<ms@BrAQzJM`1N%f>qgPrbZ8UigeM@HwJPW)+GjO_<MF!y47KX#ql-`Q=VJe7O4<$6^t_y==aUDt{obp-K3M(7fmc<eu78hQt<b6_e}%gIuTY!6T7BL~jb4*NO;59N^Tck8#-Og2D1Blpe&oNOy81F~D6GxY3z1-9xG!a7DU1G-ijGuL(4&StcA4|kQhodm8T3t;;qJV97UqUcW`eaK8I+b`Si1_)MSa3Bt9~bThuyg`Hparnm}s}`vWeMvw!JO(dh0Wjqit85o_-V`_K%Oh-#_?iBI4lIZJKsBNIu@V5dPCs8LsV3CviN+f&Oa^<C*w{tT^g!YfQ0>hAxcC^aLj42lyFgc+S6|8!&E@m<29hW;0AE3S%VSiqMzihk@tL+Ds3;$Iv;)KbP>q$oKT>KlxsTZWF~b@qun8h!HO!nFS*ELk?#3>0V%Y4Pr&u$T36>Z%WU<gyeMVJYr2HZtq9vxA0VZW0@4HX;`FFNI!`%$VTVNz0F&WBAW+~9TBqC5T6ZBCcgS$LQZ(skwR<a-jTwJObDn>Sg@b>W;;lbsTq%F2H0%uTU&ALXbDx3C-YY@)Ew{1kU>5e{|SO0|7gxd9^mmkySTpI>x)?Z`2O^Cd@(-XKXZ@9c-SXrgieq^Z^&T(Nfr(?=>djacy1#o?xaWO_NvLtORJ!)`7`<~V)`CilL?yZwLue_efwIwQ_&5siy_QCyM)o*)z&ohIhHwDQZ{}NXG8MmN2~?DOv$gVEC>tKl`YTrml@tyJ$Wea#g1+0WUqF;o=RGQ?Xucz#U6Qxo4<@6VMoiH4wR83!;9|h*dQQ`qZuLQaEQTmkmVR2p}NqgoXjKj(}+WbO7MgtXh?gN7TvAKEky--NZQCX?>M;v1@id4YI<Znf-0@!lgq_K(dS|TKPt3mpa-Tc4Kj>M!h~u!eX(bHEj8YL`zLZ*;@10R$_Eug^iT<}M~;LEt^Lh-NYlkZ&H=l>x414yzxa0o%F_1LEF#}n2A~bR3N*w`-565JThYcl+qAVbSFtg2Xy`l<WUdZ$kim>h84eV*YSA;&De(v54eVaH&A}OIlHfHk+$irPNQfmxrTOKTiL|EBM1J9=*}lZN!_0%+M@^fu90wiz-eEm)N2E<!4SR{*ikXpz{CJQJ4a&`ZVB$LH<z`QZZ4W3WQ2M<x!tsh7Hdra7Q$E<1Fc&sJX(U#f^<p{O#2S^g4|<HhrGIC&qto!QI#{AwE1#~bkrz{YI!$Jy-Cj>|SHLRjB=VYWXQvWaw7&hY-evD@ZF!2!t0QGT{q~KU-kGK85m~;|J7=0DDFqoy6p_+pmLQ!gm1&JCn?bE5F9+;YTcSs6TQW;*+>(@=8)z_}w>MX*xxTrUn5vS?NE+0PMzI+3=<FNoSF@P?>+3AIM$;QFb<_>ILsrEYlR$bmNjS~%oTY#ri0|_C3gglm30p0vg2LnRR)d(sWYZOF60nK@WG;<su#ZHMxDIdVY9LVn>&0b=()BiXV8C{~D7XnCW_;c?3Oj({w)odRs4uR`RJ+8Se$7DMZ<@XPZC31w<gzN%5&28EKn#6tm594}uFV(P7O70VlR2~UYY~@E<DMZn7#-0e;xAqpju(Bhq|1vEwF_KDUNvBt#o9MDBl6Kxv^0HmZ}ZV5p-UE)k(ea=()lcY0nv9<mw6?q7PN!(LWgZR307e<2wK+CIFdUAhEFm<EO7~Io(?h=M&5>Ql-$B|hJCpsAI`(;dHEDiF|BfFIO}wLiJMw&1SA&I86O60qT>c`PG-6vDN&fyBfH*f&PT`ccfWmeaAZ!D?%W#L14XnYZrWnu0hBu_Yb+|>@s}lna6wJ_2F7rmP2h=HW>zm#U>Va(-q14S^<2<=8YM*(Udd!P>Zh{=4cP()W_6Tz(iMI~J4PB6OdIG}OJU{k#KF78pg?ZJW0LaiHyFbBM)^(RV7)ug6!Up<Yx^<<13bYf)jN=UzvT8Mx)0+ptus8&(vh;rH0TNcnpEnp>UDn@S8KxWN<E}$d$sC|ySpBp+8|w%i*^7C8il`L|4RCXhA36(9VV85dNb=DY7P5ncbR)5i!&8i2uuH{bg>8vkSMjFbf}z+enwu&H_`py|NVc&8A+ujv3w+Ak)QyrPJvP9Zse_!Wnr`AO{Y!p*H&`9dGnC3K)I6c(ElC?Bn?Neb^u>jR|$w4G0JBni-21I6=#!8Ca70zZ8X>4lrs%uo};Icl3BFg%hB}EY8dCxew6?s?<$CF>pgDT&<8cQ^0CGkr<;&@Bj0u4_bkZd28ZC_1J-AFMJZ{Xj!fjsr9=iFC9tnYC*DVaiLV4eACxiy^_Q7L&e^eOufQZ(F0s+vX0t#K9YJ8=GAkR1L570$*j%Bjy)jw2j~9w&HgO_S=yzFT&-30(4D-pmCNdxGm9yyG&4KaCoC$Vb6Us#dC>ac_)=5|MW+1}m#9+M^TBwnToO?|n7cAf)5G{{Q1y_3N+#773M)TG4-63M?g-Y5QNI(lOTPXqazNrs^uxX|r(Ay&JNwC@Zq4BovC=KsxzWF*F8SS^d9`3bU?d}mZyIr;7wiS8Q|7;nnzIT}Qgidp{rgj+%T_&tFb=IUHt@9))y^*##YLb&4A?LisZ7Rs{0G^X8ND_M|$o+o+Si_TI%8zM=zEesyDsfY?X5Eaq4Kli&{I?)h*M(?@QJKpD*@DX0u%X4dwEdVSXPJLN>X&^p`2sCY`dKrS=Bk*^&zyi*ORIkGb+633j3eXJQ>8i3Iv9Tb1^5X_L!UoDwfwV}fimwD#mrF6TZ;LBsjDok6|}aOdA6O;VUXL{i+mf^5R^=|r@^n@&2sxa{P}r#;l;U8by7_lTXwdRHI_&y?R6NiTfGM)ipY@z`|YUipcR@hMA4N`yemg%2WWt&wI(gbWdJ)~CP$MDG>tP;M%cmMT7VsLZacFs|9yY(H*fIU&fr_e9ZX*0>T+eIN)D@bxce5J8x31xhc84kO1kB2!pP!G=6o^}`c>}}LybXlMCGG4E~;WWtjU!?`P>?%X*P|M3L`@ou;=zjEah}w=2FDCL!(pM)S3Zg<2x0Xcp@Mh#+WaKNu|lV*D@mpI{|S_W0~W$Et3OJDZJb=Le9-N^Ag4FapvrA4$`Iy{RUcH8}*F<JEOiy{|*<|)lPJJaG8flWxtNgyp#$ab&DdQDa7BrG(|V4%SumHl?{M9a;_CPr;-Lhx}h3T`U*~uGXt<tbsqTa`eYgw%WEOEqZ8fOmBRLqmu--90%oaGbK}NowJmb)Z42~=T?|SZR(Sh~VfE0|6C&%M5)TiM|D_@EWytDNeB-tx(N?I>aM-&py4>~p<QqOhvTIu#ydXQEu5$xi16L9iX=MW-oPj(w@M4rTfOvs@YxCZZe>|k58izk0o{Q4kc&aH*JKo7W4uLlX%>ozn?kkQbA^Db`yxvh;^^KS75yinRcMWrBK=)twODHdO?U%HitlK#>rEVv{oTn1Xd-(3)ht1HXOuS2(_fcBddxP@uUin*!_$b_CsaZ2thLo~@h<oD{SZ9kuX?}>e^!b<1OZNPHA}@+ZsMwYnwnR!B*FdTJ`Gf)w0pd~eNj?vgW%>9_os7zt`lPfhVumuE)7hyk%Ll1t!He=ZRdu?2QWo2n$E>o9I(3x=EF-=N2l$H6Ha-Ik$<6V3kv^aFhJA$f!F)a|pN~(>@^M+-d_F6mmtyL@bfu<>>M53UUa8I@Nw84^Cb%zlcTh#c8SdjXMbXH*iZCrP`mn~uXtvvHtMx|J+H0@1*IMhFH?r2N&RXSiAR*N{HC7qQKl(RiTC;x`W9D}2BPjv+2!9<azAgn8TDAmE#%dt9=$DM6kL-+#u6CeUb7^VO^jBDb@T|`;DMN2*Il}l6?z(&rLR!gM4P5;C;udd#+pDbNKOt@q^csAYtO0ZgkaBEl2KodL`_OsCW@Q<2M+VuW>W_@K92MaL1z#h};4^Wtz{P<{;>ZUlht1RXGM31dFR_rm#CSn)nkCbvA6|t~sBTHF=cto{RPFqP9GL6Uks-yRWGAIOP=Gc$5^s;^C)XV@QU`TRksSQGvsJs!jNN?bQw0WUyMmGfRlH+sa3F;g1wNlT>2m^f*GRHyZiYW~xk0xoFaMN}bJHF8zoi3S)sK*75CetZiFSWpbOpsSe}G-S=J~RbQeMt->BY<O%m$IDqy|+NGkY^-#d5T^-~0Ltu{IruX<8^pBuTw8BwmEvp&2PWIL(WoJy{GIARDe$YU#mnXzQ~|)3<h=-#<Qjzdt@YJt5P#N<R5H{v5w&4+xMNOb>_$adTgDcmTYd(HW_fHH;c`<?-kEm=l-63z@hSo}H@f@<EGHxlG9kbt1YRjt7X7V3-y;G3gL@Vt(JA4HJiR^zrQUe7t`$78eIchbQBs4@U>$<P^thqzTh@W#BNL&Kd?A^PNCzI`duWOTlrPXgX$z^oJ<$fT<2rGUW381iAA$9)s9P)-07(EEir1+7&c`5Kz;_dqO#|8_%P)af@r!tYiJk+s|>Ei3~TcJAhwJQK<6f$(Z6k+1Y%Z0(#)x(t&nRAakI@ZH^*`G3>`HJk7zRIPwU+ZL~7mDqgCHhmcdefRt5XKKgOQN8WuTaKtsHNB<l{!LG!s4&I8brbtR7Se@K{p<c1X5oxIlr42&!EhPuW2>@dIG(P(@zW5vuq({~EdgrBD!AB2-jq>aK6&^3P{#+C5OS|~?+S4Vo;3Fp|X3-WyJHBo*&h0;!MZeqrvkImBZhI3m4Nl0IRWaVmK@dXI0;T2Si@~)%rxqaUb*%BFH9Zh%m@<|F>(2qRIf8>p2X5EC4PY|HkB4RI8BEG>Jic&>yP~=S;71$3XU>d?7?gF1t|);uMVaoMeR*yB8UD)TZUhtPO+lQJ?`&%@z*RQDUg2Q|n<z)^X>u_8jeCB2im4DX>;3REKR!D@y%_J04^>)_mNrm&+^h{`*_Adxq{I#F5#X>x43=4BfdAXknY({><bFCoR<60%DDVRSz6k(-+v+M^okm#RFM%(ensCyz?_kSRU_|BhPeb|~p=GuN)Kk~}s@5tAjP+v5L*Gkp_M~>EE`es_r>+bY)GMoTi*UW5^dpoqwn2*3aw>Up)$o3)iu5a;souI=l{_a#0#&|#I!sDDpjtyIC(=<0!omjD*go^zk#%%(G*;#qb&oM7-8Yf6t=$kEC}J7;P^?SdHq?dH6Y&$>tS#NE#3Cd`aTaEDDLuMOy&tld`@8ZIYf8ll`?5-C>9nOCdM3{1fOE0?a*u?X^5Q)upU5_454dUR2zIr?!vtnS5}Lov5Iw+Yl6bX9mU-Eq*E_E$5-bW9oNYlOgV-}pP%ba5jHFU#$cg3DT6~D*YNG0weU^K{$l{C0;g!6Rd=BkJ!HDkxca^~PBdleWVfH#H<n`s|G<98QUc21USc1w}S;8|eUT7T;-8^9qr!ffKyjrPd>*jSXYv7j`T-U(2UH9^gZbqjhsIV&d*b$f6Yzb;_P|m35V@hGv9?>;+<vh|8IdcnHy}U`PXXr>0^ChY^(?S#g_w0-EV(=El?1@NLov{ckVON1yuK((H?Xv3LW}{VzH?KbJx+cEPs-8H!<{azM)mHl}TI+lTioUr@iH&%JmB4$gLlS$KyVEU{>uy<<W_nUBK>F%-=WjX@P$hCjEAM)#-r(0S0r$2ZLjx7X04U>102u~VIg;+Wo1u!T*MusS;#fr#kv213W;i6$YS?#&Sg<^lspWf=8H%dMRCUIo8ht~JmWS+^YMC%X?a_r}WuvMa)OfxD>n!0~Dqx{sx`MU@chtL0r$3brzoiZ<B|S#2klzMDN}G@)4r+!d(n-_-(^N#D$4GE67~IY@=HBtW^JrRxTJZ`m*16W9TzqSfC;ZII=i8_w0oiVV0c9zX#*h}{-D5NLh^+yb_G}%cD%-&ZWNLU$=uQ!zs+iEsbcELDc=XTx>&$ex)+k~Osw+?8Iro`PSIXXZXI8X+-+^pxp@C4wp;D~1>!NPrx-#Cn&BWO-Z?ZPC)orl+7hTJ)XSa;BJv%x(#9T<?@cjJre1haui?;2{!1V5LTjb#Z35e7HUb$Es-7|5)mQWSnip$rFZoO5G4otJbJn*7m>WEVcdzFPS>KF*Q9wS?W3`=<-O+;i+fxSXU%w8{nkp$Csq~>bV*DV&GCf7o)_gqT4$O~$~r2$u5frNq=*Xf&$t1-B=xT!IvS^ZP0<yRS_sreRbbc-dLi+03ew`5Q}XFSm*yyzd363fPnXn=WkGH@K{k$~)0BR7-q{{xr{8ms"""
_EMBEDDED_PRECISE = """c-rlKYj@j5vfy|AioVE62CyNBq-7^z0^ci3w5=Uk(vjqNG7Jv}Bta4*62JhUBwA6<{)qjB`%AW}x*KS`NXp5~eVp}4L=xy{b#=YEs`{&c+g~N|{>TsZz2KHb>vSFlt*_YLvpqHrCw?&Nu~j<V`wpM9G)-&u?^7=h+y%S1>>r(6_1Sc_SnPeeI$^1oq-+|-ES-DIyZ4h6s<Vs3v*SY+#bJ`VsmBs;w)6t{>W2Y7{ku0#tyb&s_?TV4Ke=LWPfq*ntpAs5c75J!F&3`^2YYZLH<-BaH}YaDTEiRuVCutIoXC&7g&%n8o3Xp}7V9u@l61ZBYKtUk?77Rl(bnb1GwhUuf%^8ube$;n#(v^CcZu)dOsV{*xtlUSWs}fLT6@aBR>xqMo;yhh7}099@W&a%{^|wcBxHX*xnQxk3{w~;TO|mr_t)1K?9vTp9=!R-3c5+GR@Y$1;a#u@vAOfJ)1TS<Z~E-~Lx1o3<n3DqpikIy?8lGk&C*S75~c@X%1+)N8CL6s!LF_^`-dOchx6lpkJiX@$8(k}{E276=y~h=vA0S*m`{T3JjZrSzN#=qf4TG~zKer3S>oQ}2$R(!P4;j7WaTbQ96`XAA-qXLnsxxd<9%4I1B0E05*{Vw7-1L7aN;c(EDO87JUltun|QH*>rL2C?%bWY0W|8yGY@d(rNKTX9&d=(zjdJx!l=E(TL3~qwR8@{n}>@D^QX+cb^V1qTEIA7!Md!1giX8vrh3}mKke+FcJ~0-J~RWE$^2mI#a=M>#G1%ei`KEb^e1fL-Fk}zu!ME)**-hKKb=?iCVScG0$!!#Ilw24Vb-kE{mx_|j6IpVF${|4XtlnD^;ktw7^i#g9lU@ol;Cy%(8c*!MMqZaJA?IMY%73$0<8hEOngAb6uykZW#q<GS`HKHLM=9eQr^^s1ww?zf7h!eth)|~X|pIy{1m_|UcK=Ge-^OW0>Q5XzCCAGX&f$HrhkjY18?T4#oh7EEDl##xNVs1Fi!ocJ5HJY0)D-JcX4vY^aJ>Ma(uLR!St6{29RBUg^fW^Apis0x}N(9OgQubFA^W2k8_#>{v+V?1qSlu#%qBAI4?i&ak;J5_W(-3#e~5Q-Xm0FOAmlQPE43iI08DA{T_v33Jc>#?C8YgE44>325^K6KB6(qDuuGx9DkQpN7%RsaTpD(CC7Tcmtfu2xDTyXr){vqC|dY1B<L%IolKi@kJjz|)lbhyu|J!qftLUV`$@9$_Qqj=!=hG4IcO6=5x`-@4C|Mm1M7AayUBe2>gsqeai@I$Ej?P^y=)Kn2qzXE?tv+Q6yO_UQP|j%AD9K$k}A)!124vHhO>i#z^+_zNbKbsL2Z*@%qx$9gov8SjUPpoda#t<_$CT{0BMg?91c@h_52ky0nqk=*^J!<0BM9MunK_mBq21;aSkR8Sf9WM1HgPBRKQJOCL9C_n*y>DzARj#S4f{Gycrlyk^&`W;gpGLfLy#)tZ@fK!o$VmgU@Hmx1Rz&C<+8FBW&4Ja1L?2ic$#xfinBR-e$Oh?C9zz0@MGxIzMBa59Vud7yD@n4ZCT$<RIX!#2!)z7^@(#P<jFEYO-cCFYsa)$CwYU)jC;Xc{fONM${6VhWJl1hYlr`a1)|SFJD@(_TIf=Ks{aSvmaS}OD5ykk5YDg-oIjJ=hqo>wQ_*D@pujt!D1(A#;Q1RA{OK=M_#O~3+Hpl@_^a*?;}p6vyBnCYivI+F9t4pK|H<ny*tG9-@4yfZ3bgjnIMwm-`ZC9rF=m=P?YZC!!%5hM@5TOk})nqV4}KnfGZ9xxb<PrqJTn=k%$-hdJb9vN$11iU!1GcljHuC-Dz|B$i8A>g!BpM15s!ojfj>%NRrIjKnG!^S@+wlos+Y-{Y&S~;q}pbyYni044)Uyr}N7{_b;#PgY5C~`KPnf^TT83vVVPf0)2Pe*}J3n{i8o$oS&RsJN=*fmp{XUk7s}8!~2TG;bMW*wQiTfmY(1v67*U=AruP}`7K#zUuR?a?e7T)jN<#!P3Q7^>My;l;(V3*i|p5G1Uq948zU<lyJHVYpL_tQh#RCu_yCl8zVJs(yt}|o=mk)N@hXN%q}Ftmu3%YZnd>?3lnWs1{d;e`O2e4fNY@d~otHZH$0=~800f#-5J*g5k+An$tuV1XjtN%crDDx=?d;q+Jbm-=L;vjh@cQKZ?8^CgaeR2)*G#5$G^5pmlFR|Q0!E@oH#1Z8;{+gy+Gnue49<)3H!{NfEdwqBHRx>`uO-U{@z;ZS1<Y|oVMoW}2EF(0#~$p`K79bT<+Y+17E_;UgTs&4=X)n-SJ#K9r$hDtT|B~Qr?3g;_I11x4dK6hQ7pm-2Pu(0fF-v)0wD?+FwwmwP8Wn}{6NwQ#qjIiKj1I4KQy+E9n_1<j2tj7vEe3{R0d@6Rt5xGHcmYNDdfuW`hWGKRy`2pDri|v&Dg2AyyFawr&B`3Egi{nqMvf@O1q!kINsJKmMOMJwJL4-OLyjhw&sl!+~Gm^54XqO9<)2#I#4PW`)O`b?$BVhjMm6Tf=E6hohrd+(L_EQ-*)-VyEr+Ojh|rS3dx8TVG8>j6e|2i^%tqwum6}VGg!j+j9`So7+R7%RP{hv=Pf=x3WKRXD~z&9ImFJ#Ae|Nqt7;+*55Kfpf9;>0AD;sSK03cVhW)H{+uwFO-+ueO*1GI}IKS>YfZxu?%Tsta)#hm$CB6Oq&&1iRFrI-F1AR@B{f9zJkNYGW>`R{f?{Z(*9j%3&z_~mJo@^5#)MM?q_X|nMz$(sRqXRuMTA)A6QhAMirqyY)S__ESp)8)&gwi+yAb`%?bC65pw5~p$Ue^^BELCeA9RmBQD}j15F<h-hqpT}Kq7iVO3*>_E4m2t#d<o1Iet<I7It7MyO3d%-@WaI^h$l9CX!lshWSt&6Fj*JAzj|zeT;TAsCuhg~zlc(8D%GV@2am1(PlwPJHqm)ywOems5YXA-g>!NZ%oZQF+ij#)S18V<nX=_C{B#Y78zC>TS{LV6C)X!GL5GKL`lmFa&SUGW|L(B#6gvL)u?3x;9OqBVwSQ%$nu4v|xqb>5h|JC*DNskY*drtWNf|IHm?ZW~!C`=Ze}W1Olo@|P0@VbyX_D|k(Lf&f4C_Bb`4t6FQX$<Lyt9z^V;gsKKtqOHB4;cG-b9=R8nO7WtmO^pr8@tS5PLDXrtgF|C2qr+R5(I5s5F5%U~!q=!BN4+#6niOjI=`CJ3pO+jO7J76#)KoH_{B;o73X>h~`_=mOj$L8_lSBP5e-=z5zL4nyiDd4n<Hq4Rxbt3f3wB(s83NL7M}>)=!r6!t7Bq>n^Zw)OtQ0o?H*vksF{rhb97iJO|mM#c-u@MNt(+l0qtyWmyfJp4JhBGb>qmUZe}u0k%wt#<dU!rMd!!g`>asP*lnh?2t;b=Xsx8EhQy~2Kp~UCBP4fNQfCuvI4GqPvf`vcNnHdXk;|t5xhez0+@jSOomTeBg1DSS|Nz639vyOU<05AaW4SrIkmgk1~OR3O~$@&zXjb5t!!M5w1J!ap-2b?Kvi6T1kERbvwjlt;rzTN3`A1RK3xJyWT<NV3{oTr(Vp%ZmP4Ye<2?4*gZH=r+Zu>h3FXGgtv;FdKzOi@f$_YJp>aMNj)U7V57q4dCWtqgT4_jArZq#%x#x&dEcX?A3!;w4Mr$<wr6C$YZrI3!y^cyXF1s|90ef1aB?0X?(c0RG=ke(I26Fjd6BTiF|G9`rJW@Ot*j7N6Oq6;Gnn*TRki#tARSkFBnX_%+YV|9)arJiNWJ06Ehg&pIQIarPK$l<-c<a#ICEu|-Z2@Y&&??hA?tuW4+wL<)=Ym2HdIL9>)iXN2@sVZLENT-6ERO=^@w2o3(RKeAI50PkR{JmY&L>NftPX!?u!PT^f!rOtqIZ?UX2VfL;yN*%nzOKqW@dFBG-_Y#G0~5PlY*aoZfQKlFS_>i&wF1l_r9L6uiy8+{?PmSN;9%CJ@V${uwjFsLDu2+JJQ#&_1Iv0uhD+K5C#l96M%QwZr6VK1&sjvnqe)%JE+NlDX>7>N8x|I=k04i>7X{+LIcYKpd-5LniWD@petua6hCp$g4Z|z*)WyLYE$DlEI9_=7@4dcf8wCJNTj0FE`0K4M^dXILxqKZ)$T_5d~r6#&SCm6`$B=@I{{+3q}&MM5arit7%Jn-cpZISlKmPDrF14DY`D2imGcjaorP$~MwvddrA*k|i#^T_uJI^RYPx7CkJ()q-z40qzc*U*6BOQe=}=G@+OT~(7{mdSsU6{mr-AT^W^t=9@JzGZm15@mV3!(Hg>N;D^F#?&(3?4E0rsnSVHXwh3_D{yr?Jt3uPa8snFkc0$<m68*==iv=-m&dAvS^LJ$kTk0Vv9El*(OT%gYFl7FmpkyE^ErviIk#V{8ykh#zG3DZMlg<O}iVDi4>V8r)O=_tf)e3m-24f`D&+WwYONwt$#FEHpR*;(7BFmMuWw)$?&a1wvb{oD(}|4+2P!`ws#%k6*&#VO|@keg@SdFKSedmVThWXq)T>dnOlAFiq%(0~t0OuC>DYT-e@t$c*ju<2a0A)<@xLF`<#+m>Rb*%npI~s0r*Et$}X!$WT21DrG9)Ko=Vh(X8;8p&1ScSDFTv4=C{`P_B*)V!?dXvkKI!YWjv2Oz5aoVgQea0}ZKxM))MDcClIJfegU7tx+4&ZkvXDdXekLgr>ZQ$v~E(@6_<npsDfqIJqQ@0cmwUMi79ONm&ZTv51DS0zaXY!-U^M-ucqKcf@JtW(ts>@ZjF3tuJ~}nt09Hiv3;8ON^C%n(W3~+Y5I&nz#%W`hL(E5|Et4|Lg%7xF2+t2gab6w?2Gg%k81SA3F7Nxbgs%R!iMq7CK$J%1;EegQpfIJV#Jb^Qv5wjRW?v#Nn&{DTqk0R~IO>QjJGCgA<M^+#TgtrYV(aTr6|e5XXP^gEHRzPP=>X?AeQ+)tx@#3Er=&6N@4S<&;3mG748Hvjaz-@4`|B4vp5%$8Baq;$yIt@d${<;T>V@Fxz#6n*P#3v_`-=avM^cIZ}kHRO~_EYtTe)b=}7{vT}<M<Vwe1LXl#XeJ-1buJR3*%PDzvSigqqVt`H6aekH?fdVqo9}3%wZz<SyC~(4@>t1qJ0tL#0ER{x_$P1T;RNFLSC^igOs5Z_wOI`Cy1+|0P3cL%{1xcEYb-4(&Emz7dU8UTkz?8`pO#_Ndrz30)RR`cE53^tcm-68p<;%dIqN^E{(+U#xWGV%ADW*BVgCr-%xjLDMi&7ImZ>OmfM1YnisWk2RlY8-y(4D>;MFF%=_AGK4KwicBLDokOBp?<N96%Xr33(MwxNPLj-CIA5=~?;w9G^X7hnd3#13Cu^7IzHv6XDii$#Q|}X2JgUH}Yip+uz8Kf_N|<bweB6cVh`MFS?R=V;6;2m!z#VBkzwyws{`#$C)>Y;;4|%F!oRz_a@}umbqZ$#+G1(CB^E8Q_xs(M(Fl|?mR?!(W8Pp$Z#(rRNPUIl>Rp8Lc;i-BNs6_z>_KUDSThBfD6QbvWjz)PqgOalHz&p;Df#1prnR9LJ={-DU6Rmj3Q7%1Ps*dG|X*j(z-Ox6kf5aX8@QUt-qWhwzKryAW=cbPs?CaBfR~^0@MQZ#|ugF*`Ii_vJh|(K&wc)407W?z;O!Vhsj2(RDyg@`pFQgiHa2;Cce93Pl#Hw4jmMJ_TsRpp8Q@7fVMfhg6?cQY;mLBQNc%x*{?aeTT-yVMyY0^Z&D6KN#u390Uw-=+k85OU0)*uX9BsL9|+<ffuBDBJpdt`XaS1g*aH%iO;UHWY47}sMBtKAQx?C4G#I1Wfc>e)RGMLKOT|)J21wW<E^h0Ew~r1xIMoR?<O7ay2%4GZjQbtXiuq{<Xe_!Tq^W2L)UUq~(iex9*8uLTKFUqhFhLkJrx+au)43aDRC^29srEL*-P0FCs;o9^H(JtVnddg^G=ol@=g}7Am{|i!qp}!WZD^NQq0riHh^|<ru?OOP#tw^gbpGMP$@TBu4>d|~)#R@9TJA1m=8pv#I<NXrn7<I4yFIG|V)F+Sn>onr2L*KHQkq$9xFlA~e@5#MvQGdq#GjGU;*VTrsREL^fTCm-$8ywc5|69xa-wiaESva=;N#@GJGphy-A!;BGEx99<*wElPqAv-1lGon%^hUqIPx5pwH2_KGje0q5>x@gwA^k%iQLd6y29v<1vACZRz_)VEWQf-)Yjghu?)UB2|&XW42&L3{97MXP8-o?b4wt_jqRz%d)}j2J#e)pTVT0+Hny|y+*?l^oi-LUYTs~D^%T%@M`0Ur%0gj->^|Rt1cvh!A7-gV!)R<y*r2G@-o{ySThYlmHxq3v$~XIEK|BS4k=&ATzKQ4Epj=pxplS$44FOTm<dF+jo2xa?yae<;o1t<|fY=DG47WjOMbH&3YMXY0wTdRF>F50ztv{rbtb_3)Oft4u{)gf^x_-GMt<0SO<Amw!Zd4CX<*bkR5;oR!Dw|o}o<snu>~Zt1qqvGMA{<*wsYGRa;U^NDItuetvcXNIz#$wwCCvn%+G0?^;ZYy8s>8+sSM`dwS3jQ}Id6WxM%`jF)~B)l>YzxNHZ17|8dWFH?L-<&WYO4+Pdih>>On&-u@Z;gTPTXyD7(^zn$>^46XlYGC9n6|R*+-+jGKKep6q+^JJ5OiD4x`NBhj3NDS69Or2n#zj|Km2H{rg>L8lt2WuONwF}#aCKueF<R8dII&R2F<5o*<gi_`w~mnqC;<v=KCfA-d+(2XZ2($>}6GKj0|^NR~~Pmnq<ntXa(r`*^-g6-*UI2$z=^bY>eFCXV^IK+E`&QTQbkI#8n5?TAW&5L`7kUb0SWk`5DjAn4-wJCLfGyxK8mIi=7@X+upSUWG`ouC*t^44LH9rRUfNIaIz&6bVKKwS0xq13L6;W0ttqj@h6fn&*>A1NZG!D)}VnUh;GOZF~ho>FIjZD-am^2P|$CR@nHbIrFd41AI~3*gKZja4)FNW5LnSC{TRo{MM4tkPjZ_)_wN3_MeDa_0s#sW%v~g-6AehjYxV*~dk%hRmlzac>}ah}dU=H)wL_Dk7Txlgz!4pR6KNhH(t?1CpT(rjVx!nYVskj1Cb*!R%C_P%b}f7}mNvh@%}bBh6&8f14jkkxwDOMu3GMW{H++&kb6tn4t_+9MlXAlSs1ihR#@qiUg9+jh01pt(?p4*s*#*ROmN9+EKgBKvmPL3b`W4)Wg8tV5c0xWVodqlQkZZRS%f*2g)=m$4l97Q3%hLTH}SAB<y_YUGRu>;je?IK@JM=j-w~uVk&)S(aTA327ot~KsD^4=%Pt)Q#`nmkxSM!e0d9B^sHv7c;X2gx~ME34BEqDIT23_f)#q8C{@LqyjdVqD10fk4E7^?*=ubjM_FX)Yx^Yzhn9l&O_D4g%8ftvxJ&(`R+^fVq+v{R6J8ei8#j*KwRn@kA07(OQ_GI<>B!I1)WZjUP)4{!CrjfL<H`RDZLq`3%fp{VDh*IpDa}J+CRQ#vd_pBM2|?wTOSX?uVvMZeVE}Xf8?WLf_*^&sMk;5LClSDyt=w*Zh!eATvmp#DPB8leuYx7nyEt4$Y9tAb9&0f@oZ@NAjknHc+tv(=y$6~K3~mGsvW2@2S80OL-O`013*#k!Gyp{$CJ7VKg<gI%juY6I)68aQ)q<9gC|d+#aOgdvEM5_xXsIO@bHo@Z4Gc5*6>BMIQYcQ_3nR%2I2)1#jO}o1CKcg|zpce(sT8I#cN1WUNEAb0Z(1^UBTuVg&bZ<>wO@FJ_J?ZWulNYWLb^}{<PUyI2w&oaoRk(~O#w+(0B4xl+G^;QESxpHymgawA5Tg_qU7BGW*|mX8%mqBr7?12KC=PT95xwuB%rYTCO{QOK9)y@1)yyV&dJYRlB90Z?<hEbk;9Em0f{lVnKtwUu-79|1CV~7cDvXRRiOApQK}whMe_%ceOx_`)H)`Qh&IeS-KLh}AaR*svrt<Wwd=s6TB)$2DUSXQaHX09s59JJL0LPBquIW6RXqxOZ7jaqau$~Lj{_e4-5GdOO8jA07nDG0>;g;P#Li~a5*(@azXFn?zk14%?@tvFNCIlCu)A!Mjq8SsBc&4T!{Vz|5AEGSAetT2?|>TVrP(~Ac#onM0xGW?2<i+x5_8@l1Ld@Yl9Ff}+Imp9)T6~NktN|+?Fan(SYNdSca@8lxgJKSZ-sq>H9=SuP;jH@U3S?&JjO^LQi^f`)M`Et<xMDKvI`T2Iz(UWhZJ)?=*S1@fq}0kXsNIt@F9I|I;3i{Q&epg+wrByN5sBp1TZi8A5jeAL!7QN$&=>LCJ-rf#+C963NGzR91pI6%cqYnqh{ejM=G7zJlaGY%9zw3-<nK*uEnR~YZlK&+yvDyGNrM6jG)Th3G4dCG0O7yUFf?DHPEJT2jgMS89(u$C(ztlK2cpVF_Hy%T*7ILJ6QMDU7XB)cUVwJU=9_J5U6ei6Q3Lrx;1_Wvh~=DJU7+h_rS6&lL*z&WG_tS-usE&HR>AXGu-P8*=zQqsqw7I-iu)=rg6gi#@u<jj??B#0^I^cx-Vav0^6a?3Sj!3;w(}T8cTECKAd*LXwW2;sKYUuhXIuay&Mv4fSB^LL(cE;yNuLOEyvq*zFLj~*I!i8);=(~|2B-m?zRuUD`BUC6}5<3zz(+w58MI<xZBpBn%c1P*qmEkaj{oq)m~Roz46pd?23-n&gabb$>N4pi*w;Meq^0$*W=qRhUY_V-8GpmEOxjf@19XL!!VihbS|Iree1<hxF82l8(TU#yY62)hd|>il*t9ylhrs!i)>B#(L#yRkt24l;Bke_!mN2-Me?fUNJ(~N@df0&WIoAHKXANcY0A<O@=&iSJQTKlR6#I%A(<Hl)mT&V3C4sYj?A%vPbgsn|BBfY^Y+U@g4K5Rp4?9E>mDJ9^AYz{yonlY$F<-_dO!j62pyfDfn!{9UHpAj5`$4<P89^AJ{f4)pta!!zpm@|m;I~v=cmWRUD8H^fH04H*1`0Vo3J0E-Tb|<g)c=R8GD+Ilzlq9JUcmiH)J)7L-HUO_W`4LuFAIcvSyTGWbl$>q{vUFrvUOP{GSH!8Hi=`B1FWFxA(nTSFv%Xk?^WdBiPp3RQ{aO&oq#419~feLi>uHbK;NDH=nOy_gNb4)8FS)_%C=4Lon3Rez?jCW6{Qf+;<*;TvNPU7K&o(s&c_<qe%pfPW`c;uJh6Xln&~eb;qFgyW_klwhFM==bzPqKk_x<i6?cK!Ng=U5Sj=(WoR@Wo5CAO6)_Oj7*GkmH)p7Gr;V=_c~6hNnHd&F0tupwPkt*u`mO$1_DJB7d*eAgi6syG0wGC^c!tV}jZ%|^4&u7X=y(lyt5c~TS@+<jQ6>1KL~-^B{1T*w7bc1>#R6Gd$WWGH9KrSoXY2mG&fl2y;=}pn#e3)I^!%!CvKyFBiun`Kc4&7BTZC6Cz_C?7R|m6pdhspgMO7v)O^{cV{FA-A-a~N#UHj4IgJDr($P`42G_@h+Cuew+=~hV@`Wz5tfxOOz@$y&+_?SzdVFW1W-|j-uhWC&PEmjz(7I<#FcQ?m0GW)=q?)H?H@{wfqZakz1kxLGB?m`Y8C{ehY%|R%bxbY3}l*qpay;EAcvg#NTfZ177KLA}dLGDjkD|x=yBr8V@GhxXRlOke%$z*@#M)~`RPl-J-P#GYxSflYcjd37+JR=;m%TF_f^Z$6kQ<72}AAsTcpzkRMs7UO2<!S6vPU=!PK`B&VHYy5*xa6%|Ij9o=vpPcjgWsD3K+#Zx_{ST|nvmjO_PD%(Oo|SXIYp@KJh>G+Wqg|+As*q7_!c=jKRv&6F5kWBzQ6iOgwVMt(?ccRV`o^uM}f|$#kvi}1Vz}^`@74-pEK|N53VowB#T!IfzMx|)P(hU22n=1hao61Ub*@?nu{vumzXvc#)EM|9BF0CAk+s4{5#LRLDb?KLLxv;t#loyisb5BgMGtvgwJbWl?R5we$9zl_oZPph_5{HAtuPi{eX<+CfUz+14X+?^kYkjvLio?l68)yd{{!{7X*uNh94#SfdfE5X`KY|SMux>Hi$lF&!E;bkObQBuWJa=3fMo-8g0@!DSS9`rQ}}0fE7Sm=pGM-uCP>2VR<3{7C59WR(revs;}sL&ceU(L=7VoC!yCuOnhJzBuX4O4QK!b1)>}|m?D6*hM|GrmlH0^Rn-4L<idts3Exm(?C^ECc~RsmtjbvA1S))mUevLb7X`l3eQGo37yYwJE1ha9oD|CX)$%Gkqe3NahRiGoW1zI4#$kvdkupqq9fxC+4&b~v5ep$4W3WIaw~MfB;>O{1*pIfVG|FPwu0<{^u;feq-cV8!K)XY+Gp;;Lu9tdDq#;FZLnNpm?nZI*GN3QPv&VcI&^IV0jMBX?d8$%=;xD!^<qda~Q*R|NBzd7hYmQQ?n8TC0mN}zSI>OGLS>zTXv_=Fipp+sFNZ}0{rW1}7_n3w=)HoL?Xx>r16<ee|=iG;Q4`tBgCy45(al77nF{ltgNwtb;MR0_gOniu1Ku#^0YyUznfoQ&zA7PPLI0Y0+E7lifr85pme$LgDyl6jg1n$@p@5uG+zJf($o%imor5ou0AhIbYD2M(Br2@6GY{m|rrNVlMR4brpq|S&8my~=T)3Duw^1&NGh+qaz2QdkLOlKye!m77nxJn~Dug+}wAmGD~yc|cCQlyj*tN{HCXq@?UMv1V+KqF>eV9z(21E}~>2&^LD>6nD<WZ?Jw=dgE#0MRZk&^=-(frY1fz*Mbh={79bQLsNey=}fQJ7(9gxQ=1~Rk?Na?#q`}si|?+RO$tY5DLKZL2x`1U1Ezbsy0mycMD4*LDt5riK|tWVI2Ft6~TU70;`PIHBPBGrg3u--a;Dca;1YXa7M6CF+9RnhGE-$5Y-uvzr6ZWXG&4FvsJiCbUqvOFkHH+fF@ps03r+UPq*xmO1p}ea+g3EU`vgmkk0!<I||aliH&q$2h_$=<l9y^hq|c6Z0Qq!wha9V-$D94Y!t&7^vsfxSL_oDlUISP5oNeFj#luOV`0gd<D;YXC@DLlin+tctn*cRKK($Dsfk?#Xya}yGM;)LR~{6RwTnK6Yo3s{gg<<H3o9o4*x&CqxB?%2JU(RS=O2cg7~}{2d#~98)O>8V#Ry=DAbxIF&hxMnImetRB7lE!oBK54QCJ}fmXdLoOnl@}s_~=}@OP@6J}hn@mDa5w`mD+P0hqg~(!Vaf8+rdcuMr;u|ACpF9CD6#$l+9Zl|((HRj7^us}j%ds?X>&$>C;?9?!f!pwptlpP|_*=Y6D0A}Hk)rX!$86=L{x3GFjaO~Lz{A&o#b{xfq%YPc-Bq$q7pbc-wk;PScQV72}%v%HjcCS~Aq=;hbOP!TJjz^kVClO(c}NYR4?Unr=h4K?rV<nD`VsyJmUy1M#&hvEM>NzYp6j8C^nu>1yyQpQX`^hZYYpjRJD{0F=0{{fEm^)76R5BZBMc_E#&RA^a+jGFwkyi9b4IibOus+h}7u)?JxZ}11zd4*2wMwkL+A0%ju*x2_e0SG8$n6vT==#PLI5s2!Eit0F1p-e<80I}&%g%AyqrxbskAo=+`(2__SR(Dy+z41<zz1v>4Oew_SJ=8A0(T0?vs({gB-jTXf3tjTPS~#bSxY+~4d$=<PcjUmkG=P3IzKg~doqxisgF2VupYYJI!6j#^DVL@29<o>?Ab1aQOphi~u?IPf$2<oQJftUD!>*~>+S#_|0XBtg%zovU5%Jrsgu_c2-sSyUw|6T?u;oTzMiI`+6aYm{jJ8gM(lXB~Xw+X!Qlen>Q&ecy$r&ZdFVU+8+EnoZqTq9J&65o90KJ-a-je`P&-7zTQKfIq`7FqRdVIMh>$`fAT=0+M)GfQ2N)VBMhKwk>s7i>X&bJDPm`}Jy4&=AqZ4(C7w>eQhqLfQOmeHkUUB0MGxjqXI2Y7Z<5psaGArTg@9L8Yk-V$|g0`dPO;n2y`Q$0ye#1*yiPni#NUNy;ae6?-yS+R-lAj7FYD$L+NU2yw~ouj&uT?B_o=od<EFrJ!AUbv-j4k)_{viQWJ<M$R3;g7g&;z<%Ecf=IAaO&j9{{b>%nYukyWZa#8{Ii8S+@L~|f^RzrczkaPDUbiiYw;t4-zldN&1ja?1abHy#5JwV6Zi#xsz9z8)l1T(tMSLHl5C}f2i7_MNRv0r&y@2(ap=osAhFkwSwBq(`fn=N2+tP|Wp0dfa__Qi8|E^Ah?mQ9Mq;=wRU_X&U~5w^#$(|yo&fvT1MiN>OhiDNML_Suo4MmPyX_b?fxPgoV5rMI@(MB(sLHr3&Lru)%Chv|b{*au$xr4}YIc+=Ljf3yeJDbw1|)C&AlFs1oCFzQkWQiN2XP<{DL~lCYE0QFi9zA18Ncz<$`d30W91_tZPoK@mSXBdx`_zS#~tbuA7x+oMaKCzN0O9+lU6u#PTQI(zjpGkUA3U7E6QfSI)z__8Y)$iT@)U5s*hv?k0mF^O;}kxrY3^k;QT04d~Qkbo{{xBnyy5XBpxW#B(Ee!t0CEnTssR7+ecUfD>p<R2{9tiD4nhR9`$K7$b!-WF7Zv!taoSUml+ZLl|4ivQbTlOKZq~ed?{-|rvw&?bO%$|-y#q<4cW-!$<^`NVDt_OFjQtD?J@x4&oV9wUM87?M?v`F+)beIQ1Pp%DK$&Cm5@kXS4?t`FS`q;1otAZ$N}GbLI(W;Y34y6{ATLaSfr8W*%nIYsC0NP&Q8<0BurZ}BXGuSKbfXVH=4I46%fgtOrpyAC)e-$m*S%4JJHwqWk&06JN%g?8lx+Z{X|}(M3Mr(;}3ZrxfS94!`wqmB%{R}$dh?~A9j`q@GC$=o`wyZT9Lck6m?O&>SR2pcxg4&Ex)-+kY5g)Y*uy?3&>>2ylid6UxqITAC4*oF$WMOYhh!`ngWr^s?-rH=`VcKgAH&q<PnibPzBi%(enAi@0>NaPW7fN>6GidkY%x&z7@;m855fFaOCMEYKj-qaj2!|ZIj)#=|W$=&x@Is-qR34h#GtTLZ7z*)5Xn`#b!Pg%H0*p71JoCPVC+ly%Co#1*jYt6<vy#-W0o&wa35IM(~u&^xm7S_i5PBStiWuQ0$Yjp-UyAq?QSY1}mlmrhhWo`wd|wXZ^#={?&DBOC;k=_|^Epm2}>j$(>C~7ju>XTM~kr8c{F%G)i3yXX1D^ay;%ssjMfUQp0+ds%;8DtL;JfqL$w67?@V>d`pm8bpW;9mV?(sAyH+s=k>Ne>zz>>Ee}f9+p)Mf&TXiyHocoEOyGEZ%7e{#Ut=39`2ni$gtda(4wO_U`McasQ2SaHsg;~4c}6(tE43xnI-PSkZ}9qGx>fyEIr>Ms8g6a|J}(@MeCl6Pw)TT^Gm3Rhrg21vj|$L499vUA`I&`-*AM&*4?dp#nY!$>%YBwdKqo64B-pnCqi1-NvEqkT1SUDo-XcrY@j_#hb&a|dAD14#`G8wONQ${b%dnK3#PtOt*}toJ##hB+RU~P&n4yu&6nmvsIoTqt)Fb3zaXU1NNT@zi5f7h}C!^?QvsO`gyFeGd(#6r(WHojT(XVn7b!}1Zx+jSF!mom5F^8FCan3RrU5Rwg9%b>^#Xz{#a>RScLtW9m@PIoP!gHBGX^(X<@D~GJn+Ta~trFKmFY3bZK;+N8(0Tm^jfkRB9Q1jE>GIIi6u4x^XefzWaxs@B=Tx!=%a389<0T5?sXyl4oOp+bN9@;vO)y&di<E&*J4WArE{cou!8$xg=f?W14Y=*rxwm>d3v`81?lFF2`E!CXzp=bQ;m&=Z%gx+9r5OVMJyc8A9f3Xfc$~+fERbiNJy{zYTIpgdUD1k;Rz4brQE6{ux(|I_t+@Vz6udAVK@nko2nrs*kH`YAHl!WpbD)ZAwc?znx1r)RcG0T6d3qx5PeJqnbkj>QTL28S2V)!ZP&Z?1zE~XTYg#$Fo8_K<-FG%t<F_GL?Js~RW`)5izLp`g=nN{iw;wq8a!Ya=zA@yzQ0M}>Hfr~yxcNst_ri$C<q8CqFy{BoT)#iLl4;B;9o=Lsrp}RPJCC{9oC$`!8l-zlyTNg-kgI_DqUey?1C)ZN%#&+z_LYh=U-&xBzotR{G^ibI{V`W@nz`FXG)BrczS%esGaHAQ4Z~DN0j)ILx^NMPDJCX$XD()LllQWy=bjWv?J7C2$^^S9yO*+TKe9F+3n@=t^Hj}*q8spdL(Z9cH<Zi6vv*DBFxr-O9`f-$zJ5@~_xJ-VyQkE7raUF%%ytT}1J-lQM;pyTZ*NGDez&#A({Q@V{^^dcEFlwdNfn(HSel9yT|*9{ZMpJ!S7t+AHM++_W++q^zhR}eURc3X<<p2YbDLcvQxOt}C-H_YRKIwX`GRsA;(QyD=!qom9Bu<Bj)zDXOq{{5DNWU#BLZcB3@p9WCE>^v|F@b&9Fq%6TQa}%=v9zXl>|cz-=yj$mnKZsd_=%)b;(UcjDKSzA`Z5x6|&+)iW!HRt+(QA;IpXJ@TM#btlP>}8%{WK>r~T_He06(o!p~oG8gMWIO|$%1C-23b9%P%nkucIREdie(N>d978h23AQcFuE(^TnLd!--HeTI&a^}zkoa>XTtB-xUz}0HM)Qm5bqZBDHlHwk)r1GeL=eo;nCk3vJ5E}=XT!w4SHKsc>4IqJB%O`VsvH?+WcNZi)%tH~0L=>1MgwNV&xCsTL*Tt}#XCH`=_)`37MrFidk_-miz{^95q$@`FZ3KGh*ec+$=Rqav?+Xxv4vonRWE0%z5X6sk8apSZTj+x@$ZfwDsLm~~6fm*1jc0P8+gyfrhYJmj=PrSXytXU3acU#`L?V=^0LCqyNRi_?)E~5mI~1OZ!kt|TgV;iayF0Zk3cqMzk6HX*J6E}pt{P1TL*;+#tm67<74M~~U@J1@=THTMz>|90GKnu5%le-(7PO<#c+g((2?U3ZNE-MapKn#?0jxjTxSVJqqVMbW_N=R0nE>C|I~*)*rvzJWeIoM>%wi8FL6F#k8SGG}=?voZ1WzH%-?1V|GyoW`VO{Rqz5C7(cT}{Qb|iPzDjlocYSxdrF_5o%L8~Z<#zwPDU69V7XZa;+>>?E>;2v{RU`k0^J&fQ@Q!bN5xIrt{H(zL17#vXFazD+t{lqGOUfZh9Ri7TV8QnJpm^LnD`F8P&u64WleLngv*J(vt-rsG0ukU6yF}_4+-%a~dRM~V3dC>A43Hd^7F*_7)#n^IRBdoB!)537D@K~i?SBmc6$WD6@6K!?NZh&vqz>pnfCU<yAU-e8TswA#SY6K7{fw$C6@GZBng{#zF!1ud3z*&Yp)SUT5gWr_HU_7&`GwhP9bry_UJHx?l_*V(p-NpJ)ANP~lU~!&`&vVTz3rBcp&xsn!D93uZDEwDbd)4zk;c#8u4I&G4<0UT=Wy{H}S4(*mzOy|yP!7m<uC&uZZZ;aak*)yJXI1!Bku)1vR$nQ;P1G+s@#5UF@8yldi#`cv`Th}eH1qcPX5x*?4OJXXu`%g|Q})aGrZk6EO|%tv2HYv!YQv?8jXCC@5Tm7lW<wx<2J+1$bC%k(U4yO~+?wpt?<cm#+xU{2e9^>;?;LPekG3VedE!VELTx|YgC|q2pFQkw5ifTaCL43FwApK5u)}Q<75cKV&f_03hs-g*FpYYeQ{#~JUuGV~Ac5kVlz7m;astM`;u6nt<RcNt`l9t$w(j7{zT>G}!FRcu%i<}~hr1r7HQpb+LsVi$^EX8&Za==|gr^X#Sd_|Zla(*s+Q&uTZCc8PAd?+^x0My%a$hOta9nQo_G)CC4U?=xuMHP8l}594zi>1Su>BpQ*^<Mhe|FT*Z+8{QPDyZ8gMpH*_KV|fpL`YU|6kSLx2j)Cz;;{CGzPkjS;#CJivttU|MdaZFO4q%i;~@>2gC?vX0(|R?IBkwVKb$N=t@%A4^P_#Jj{voTz)a=t)Hyi1rKxjY_d46hBTGJnPi5@iI0{<oWkjKmS=-EmpJZX;oK#@$hV<X*)=-^sIVhHAop9*7ql)PI$`Y3FrTS2ODX&1!tpR1+KPgirUVL6;l`lzmPmQJQ=Ck8Y@a)w9+>Q){rxMMIkk8JIH)2rUQTsF|3PR7@Qz22)&!mxee9v6@}N{6=UI47tc!cjiDbXrnxjPcNy|vyAs#8(YLOZx`L&H`0m`M)Tb|*We7G@@IX+(T)apw&N@U<y1~j?>NAPFQMYkaM^k?%CES^a6d}rq=!R56s7-3>7hXVR6eb#+hlS1Fr@65SjZV460r(J3mAN~G@X=O_-)3e{z^y{_&#mGi*yl(#hGr;^LIE&Z9>JQt%F52<%ApJS?D_;-v-=GQO8~rz^ZG0ofI<yDf7j0AgKg<e4(QlxfQtVaR?0nlczb_T4Hg^C8zi*odN<q<evB7SqZN9?)#Npx8pPBA`Vn>VAN>-z25yIx}m<J}jpqq>IZr6Mz<oL+NRH61j*Ukk=!2IZ%iL*8{rwI1$nq9A!VXT=5I(kF*p?pBvp_v?p_?WUPun%senJ{1@zXry=H}%v{7oM3PYr`l_J^=M1rF+X?CiC!4kIXUb0b9c?=^DQ-1|yejyURByB}4uJD<od(P~*G{{t)AlbS<1tg+&IY5AcY1xt8L0C*m59{jq010F?hEw12#D<WEC>p$j_+KNep4;VOHlDXm?|`$P(*vs*^8I+t{pNTCqFUZl3wtNcQdLKWqrk*waKxK5-{P~I^jO8pY3tzTah%BB&NG9TI7;^V1!e0^3d9f;D!$5Zk6*UJ4k-KofAmM=?+S2Q^MQd3=7Enu^lZd-C9a}xP>_hs80jlz4!4=^=>tx;vM#)$KrjX*vAh$ne+@cJ8_bHaU1+-4&#ECQx2lAHrcPZN-aB4_ODlc~uP4=DSAG_uCQ42eSwaz(hbpn|(vz;*~`I+hdHkKnzC1AbeLN!QSr${jPNyhFw;+!15ucfbrK1;pDGdjmzIqPrG(Dpfkjm`wBhFt**Uq=b3M(-Sy(^uQYsG_I~M4=)^0j4!X$Jk$1MYWa(BJb)RY(chbSaiX`)z$CH}7`fSEY|9IQP)HkD8Rp3QUkZ<>+2dC2iLlZ@P~U)jY!q(q%D~bifPwxgosq`yy%3@fuzb)K;homw{#dY#kr$6ahQfQwZ_Tdt5<1o9fJ;q9zW)}>JXY-SHzwI>a0Hg0dP|U$9_0lvejvpry2uyH4f=oK;+}bYczwvfDMFx`hdG;gZ2|u}%$I3WCjr(4KZ~oOZ0*^%-x=x?Nz`E={mZk%Q}*`b>FFNYOs;`Xvj6+P{}0`faej9CvsS3c#cj$Ma(La#lcFeLS|YanIlJxh<H&s;OvWSE3(ZZnPhormnu`R;W3jh$bacjnz{s#g{ED7|{FHkV-UVo$@FqX7llMoIOY4TNZ?THai22Ja1FkRohacF7^W%PxyHZpB2OeC%3{#%&?VjQ3feiNVyO^pGCx*S`9q&`_09e%J(wq38S3qkbHqK8I2IUy)8dfzg)6xz6Davl-jz&ofc<=m(Q?Jj7nAaDFRcakBzymSMlsP+B>v(=}uGcBah)9cXDGl<`3ylCLDL^@%&g{bd2d&&aPI$-V3YAUw0#G~#T|vfEfTPP?etSVN<+)5LuH?s~tlKzt*v$#1-wpz6<N?KBx-rM{=E(hOW!Hk~FINi-NN;%Oq|+DOVjGQpHdS~+{vw*cO@A>*{EjKxvpfuls;8jh<^X;yZKJ_)|EH59RD(H8DM^Epw;V5f;BUxWAhOgOl|}dymi`l$t^Ra$@$pZO@^*Ql{$mhO<>HGv^^me`y#=y)7N&3U8X2B!3nY!p!;`bUiHG)q36pn6$m=RZrWYUywLQss^G9~w)fTn&DwkZ8?zkvjaY6O%9WJ*3X~r(!)F|#!3GYA%7|OJkGt8anjFVeiA@utiUeL&m2!fe9vaAJVxr{V{En^E#&J&|$jnx1ig{)}8o!X&<t`^VtJ#Ui|tyt15Vf11^P84}T0OK-)86&&Gs{`@27@U%>a6g(N`ITdH*|nKQ9X-MH{xMn`kQ^e3BF_xYDoB)(^w2P6wiJg3Nn#EX4FPba9^)=XS<P0)WqlI|=TPK<A7r#vnT0|~)toa_C5K2Z>`*!<k+82Zy-F+h+&biQ5P{P;X^$>kL<Yskk1EBiE^3}F$yF?+nUfC}=a<)qXF02VR+j<{nG%SPK!@;tvM{s?8IOGuIuMB59G1_e(f-+Nc_aCN(!sC?|9R(e|3O`n$2>=jSj9Z+8m5iOx|VrkzR+A*jmK^}WUsa2!ulBOSiqa}a3$w2=kw1=(x`gT0hRy`ctNs?r2#N<{n*Kcf((&hb)R>NxLzCn_{Lx-iU2{@Nx#N{FTa!&7hyB_vtY3RB;L(wvx+;FPf2XkqU#HHzRKrhZ<*O=w0wS1I;bkt(NOx3n^1d*){iwAVw5CX&6z9y<tD<`#=~8_Q>Xr<Ksu;E!{r2er6GO;Wy<I18!~d{C`I;R9V;u21a;<1;fh~2VAh_HP=Y0)9zJWhq||5(1G|ADM7Nhxo7xZy77t_|>W3Q`ICOkl1-AmnMw@XxST-4ScZA3nLgOirASeh;c>@@r7@xnxZ$G4RNK?4Cjku+?uGTDdQndb+E9$W0z7bQ{RG-X)^(_#r%Wc$9$s>b_WZnXnsAAGtiB!3k;Vlo3t)<$Op&=%)l2}8JkUx?XL;+wKc`*#k-~%LP2%QY#xF$yjw<(c`HL;G})O{PHWR0yFCaV{_GK#zP@Tk>#g0HT|o621l8*-Qk4V`2vKF(I7*KC2dH^i>Oj&Qd7`YNVWo10Y-A(W@uaHpi)8+<;1C0?DL9QX4~rkk0g*3imE`|qz3ztWIAk{iIh$x)<cS%D3*&s9D)28CsTw~Onw2@4d}&QBco*7X;7n^6;a*y#qV+%c!_Ch{#l8&#KpCwTN^at&X^9Z$M9$~3z>-t6Rb(-?NM{oK5C#NBC2+gxH-35SLGR2`v7DdY&xz(AGdrlCn3M#{A5;>x6`SMq1t4`@R#y@Dqoxh9%Yce^=}H?)J<u}m5hQ}m(gx_5~$^<7ePGt=C=MK}U-j&?{%<0E+?0}x{2*vZfEsHmf=*~BnC3Lqb7n8#Pc%)J`s>dm;gVt`u~4A@Zu9P|OIKRPL1aiiFvl~+^cRjWR>snAl;+*1B3#}{Q%(v1C}@pQae52~5Zo-LP;RVb@?q_kn=z|Zt=1@;cE2zO2dz0eWRRTe}LLEfUR@=m1_3gI2dQ4Vc8gS8B)9Q$_RR3%EP$1;_1iXMu`G`S1rvuB0DDcPI@a+>HNxRK96XUbA(nVm+O_>tKv@AHA?_v324(uqlJ`R`vIooKzWue_YVqR`T1_I5O2%PBHESMXvr6E)pD>?~pFfzN2YqQ|Kw#HTRvF(6*+4LZXj5^8}rqhNVw<OTjLaA0KN(uI;;^V@P+c#n5sJ9tqgW^V?f1LJX^hgEO`N@D=k@WRgn^QHMJmw&j<^_Xqh3%Wvx@6ZTjX^`dQ)+J{fk=-v#=L7F2tC;K^phNRq>{*DBrCjS$-q{V>lAE2?>j!4(j<&){kuYX$v<2~Vty=p|(NtD(jT$#65US-JMfb`k*Gd=TFGw}3`W0`uJ|PESm5K;uNkdX1RT6nAlWQ1)xbT%kNFs?E3}AyJRWwm>6Q4-i=r?>0bIOqSJv7x@W;cyVLsuuBan9jA*A9=r%Es740P~*e=pTPuPXT!cMV9o&P5d#89{G@9og7PneJ4eBt%|f8IW~9CD0*jPo*A9C`BLB)x;2d56h^L-rCVHhj$|z{o_7p0v6Mcab=qw+vAf@8mEHUK!Vds>)nX$51eIPSwzjvY<-Ng(?}-TMT3?bWNxN{T>O9}OQLM5SRo?Pw9+aY){&lGkce==$-Q(1+(E2e!uE8bXhk`!<f=d-Cvh}4Yq7NH+3cP67dkXweb`#(-#nLWtm6ix3S1z|4%$5T6@eA&Sp-IoC@)_=v;c$nAerX<9%1R557wAs-yu3lI((-HX^|g6#@PlT$>dFgNOJq@asajg}o=E{5`n|a}LE0eo-sqXI<>#)gNv3m~e=jb7>W^;{9Y{ZLmw|RhIX6In;=Vt_+}UQmrC=RSUGT&ku)mdd#JhJfo)Te_{d8W7IoX-DL`-t=4+9ZH_Q8cMeg7o%$&h9&TLIpSt$Zm02}^3Oh?6j43pvCr$}~k>D;!w=5)87Fc{y#V+71gxe<&AOlhrZ;#3~3xKHXMk%M6B6#llQ`3%o7uK~W0mkPn!c)UfFOX~#j|1;-IbvZcdo^lE7BrXJ>V5{~KY&Cl%PRsV7*^ZxmX>XBYPo!&u_tQh&lLQGv$|HZL5eb^grE8YFr{0T?D67Y3P+P}O!2PiZRNWFV!la|TB^#1^+Bz-y"""

def _install_helpers(project_root, code_folder):
    helper_dir=project_root/'paper_results'/'B_followup_analyses'/'_embedded_helpers'
    helper_dir.mkdir(parents=True, exist_ok=True)
    for name,encoded in [('run_external_sicapv2_test.py',_EMBEDDED_SICAP),('run_external_precise_wsi_test.py',_EMBEDDED_PRECISE)]:
        payload=zlib.decompress(base64.b85decode(encoded.encode('ascii')))
        dest=helper_dir/name
        if not dest.exists() or dest.read_bytes()!=payload:
            dest.write_bytes(payload)
    if str(code_folder) not in sys.path:sys.path.insert(0,str(code_folder))
    return helper_dir

def _find_project_root(explicit):
    candidates=[Path(explicit),Path(__file__).resolve().parent,Path(__file__).resolve().parent.parent,Path(r'D:\PANDA_PROSTATE')]
    for candidate in candidates:
        if (candidate/'paper_results'/'tables'/'best_threshold.json').exists():return candidate
    return Path(explicit)

def main():
    ap=argparse.ArgumentParser(description='PANDA paper B1-B6 no retraining, safe original-preserving runner')
    ap.add_argument('--work-root',default=os.environ.get('PANDA_WORK_ROOT',r'D:\PANDA_PROSTATE'))
    ap.add_argument('--source-dir',default=str(Path(__file__).resolve().parent))
    ap.add_argument('--iterations',type=int,default=10000)
    ap.add_argument('--skip-inference',action='store_true',help='Do not run slow B5 inference')
    ap.add_argument('--only',choices=['all','B1B2B3','B4','B5','B6'],default='all')
    ap.add_argument('--limit-b5',type=int,default=0,help='For smoke testing ONLY, not publication')
    ap.add_argument('--skip-sicap-official',action='store_true',help='Only if you want to skip official SICAP rerun')
    args=ap.parse_args()
    root=_find_project_root(args.work_root);code_dir=Path(args.source_dir);source=_install_helpers(root,code_dir) if root.exists() else code_dir
    if not root.is_dir():ap.error('work root not found: '+str(root))
    if args.iterations<100:ap.error('iterations must be >=100')
    output=root/'paper_results'/'B_followup_analyses';output.mkdir(parents=True,exist_ok=True)
    thrfile=root/'paper_results'/'tables'/'best_threshold.json'
    if not thrfile.exists():ap.error('Missing original best_threshold.json. Refusing default 0.5.')
    thr=float(json.loads(thrfile.read_text(encoding='utf-8'))['threshold'])
    if not (0<thr<1):ap.error('bad threshold')
    status={};jobs=[]
    if args.only in ['all','B1B2B3']:jobs.append(('B1_B2_B3',lambda:b1_b2_b3(root,args.iterations,output)))
    if args.only in ['all','B4']:jobs.append(('B4',lambda:b4(root,output,thr)))
    if args.only in ['all','B5'] and not args.skip_inference:
        jobs.append(('B5_PANDA_zero_coord',lambda:b5(root,source,output,thr,args.limit_b5)))
        jobs.append(('B5_PRECISE_zero_coord',lambda:b5_precise(root,source,root/'external_precise_wsi'/'results'/'B_followup_analyses',thr,args.limit_b5)))
    if args.only in ['all','B6']:
        jobs += [('B6_PRECISE_metrics',lambda:b6_precise(root,root/'external_precise_wsi'/'results'/'B_followup_analyses',thr,args.iterations)),('B6_PRECISE_label_sensitivity',lambda:b6_precise_labels(root,root/'external_precise_wsi'/'results'/'B_followup_analyses',thr)),('B6_SICAP_official',lambda:b6_sicap(root,source,output,not args.skip_sicap_official))]
    for key,fn in jobs:
        print('\n===',key,'===',flush=True)
        try:
            detail=fn();status[key]={'status':'COMPLETE','detail':detail};print('COMPLETE',key,flush=True)
        except Exception as exc:
            status[key]={'status':'BLOCKED','error':f'{type(exc).__name__}: {exc}'}
            print('BLOCKED:',key,exc,flush=True);traceback.print_exc()
        write_json(output/'B1_B6_execution_status.json',{'work_root':str(root),'threshold':thr,'bootstrap_iterations':args.iterations,'B7':'EXCLUDED: requires retraining','jobs':status})
    print('\nALL JOBS FINISHED. Report:',output/'B1_B6_execution_status.json')
    return 0 if status and all(x['status']=='COMPLETE' for x in status.values()) else 2
if __name__=='__main__':sys.exit(main())
