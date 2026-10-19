#!/usr/bin/env python3
r"""Fig 9: original four-panel grouped boxplot style, corrected for B2.

Run on Windows with no arguments:
  python GENERATE_GROUPED_PER_SLIDE_DICE_FINAL_PLOS_V2.py

Uses original D:\PANDA_PROSTATE\paper_results\tables\per_slide_wsi_metrics.csv.
Outputs Fig9.tif (PLOS 300 DPI LZW) directly into the existing PLOS_FIXED folder,
and the PNG/PDF/provenance JSON beside it.  No image-generation service or fabricated data.
Optional: --csv PATH --outdir PATH --groups 8
Dependencies: numpy pandas matplotlib pillow
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image

CSV_DEFAULT=Path(r'D:\PANDA_PROSTATE\paper_results\tables\per_slide_wsi_metrics.csv')
OUTPUT_DEFAULT=Path(r'D:\PANDA_PROSTATE\PANDA_Figures_1-17_Editable_Replacement_Templates.png\PLOS_FIXED')
SPECS=[('L0','L0 (1x)'),('L1','L1 (4x downsample)'),('L2','L2 (16x downsample)'),('fusion','Three-level fusion')]

def resolve_csv(user):
    candidates=[Path(user)] if user else []
    candidates.extend([CSV_DEFAULT,Path.cwd()/'per_slide_wsi_metrics.csv',Path(__file__).parent/'per_slide_wsi_metrics.csv'])
    for p in candidates:
        if p.is_file():return p
    raise FileNotFoundError('Original per_slide_wsi_metrics.csv not found: '+str(CSV_DEFAULT))

def draw(csv, out, n_groups):
    df=pd.read_csv(csv)
    needed=['image_id','fusion_tp','fusion_fn']+[f'{k}_{c}' for k,_ in SPECS for c in ('tp','fn','dice')]
    absent=[k for k in needed if k not in df]
    if absent:raise ValueError('Missing original CSV columns: '+', '.join(absent))
    if df['image_id'].duplicated().any():raise ValueError('Duplicate image_id: cannot create a defensible WSI plot')
    fusion_positive=(pd.to_numeric(df.fusion_tp,errors='raise')+pd.to_numeric(df.fusion_fn,errors='raise'))>0
    if len(df)!=1577 or int(fusion_positive.sum())!=1126:
        raise ValueError(f'This is not the expected PANDA test export (total {len(df)}, fusion-positive {fusion_positive.sum()}, expected 1577/1126).')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7,'axes.edgecolor':'#555555','savefig.facecolor':'white'})
    fig,axarr=plt.subplots(2,2,figsize=(7.5,4.667),dpi=300,sharey=True)
    # The entire top row of pixels above the panels is reserved for heading and legend.
    fig.subplots_adjust(left=.068,right=.979,bottom=.110,top=.800,hspace=.49,wspace=.105)
    title='Grouped per-slide PANDA WSI Dice by test-set index'
    fig.suptitle(title,fontsize=11.2,fontweight='bold',y=.974,x=.45)
    fig.text(.45,.928,'Tumour-bearing supports only  |  Threshold = 0.43  |  Benign cases assessed separately',fontsize=7.1,ha='center')
    legend=[Line2D([0],[0],color='#1689e9',lw=1.15,ls='--',label='Overall mean'),
            Line2D([0],[0],color='#e66d43',lw=1.15,ls=':',label='Overall median')]
    fig.legend(handles=legend,loc='upper right',bbox_to_anchor=(.985,.995),
               fontsize=6.8,frameon=True,fancybox=False,facecolor='white',edgecolor='#999999',
               labelspacing=.36,handlelength=2.5,borderpad=.52)
    audit={'source_file':str(csv),'source_sha256':hashlib.sha256(csv.read_bytes()).hexdigest(),
           'total_test_wsi':len(df),'fusion_reference_tumour_wsi':int(fusion_positive.sum()),
           'reference_negative_wsi':int((~fusion_positive).sum()),
           'metric_definition':'Dice displayed only if reference TP+FN > 0 for that level; no automatic zeros',
           'group_order':'original test CSV order after tumour-support filtering',
           'boxplot':'Q1/Q3 box, center median, whiskers 1.5 x IQR, further points outliers',
           'panels':{}}
    for ax,(key,panel_title) in zip(axarr.flat,SPECS):
        tp=pd.to_numeric(df[f'{key}_tp'],errors='raise');fn=pd.to_numeric(df[f'{key}_fn'],errors='raise')
        raw=pd.to_numeric(df[f'{key}_dice'],errors='raise')
        selected=fusion_positive & ((tp+fn)>0)
        vals=raw[selected].to_numpy(float)
        if len(vals)<n_groups or not np.isfinite(vals).all() or np.any((vals<0)|(vals>1)):
            raise ValueError('Invalid original data for '+key)
        groups=np.array_split(vals*100,n_groups)
        bp=ax.boxplot(groups,widths=.59,patch_artist=True,whis=1.5,showfliers=True,
            boxprops=dict(facecolor='#f4f4f4',edgecolor='#222222',linewidth=.7),
            whiskerprops=dict(color='#333333',linewidth=.65),capprops=dict(color='#333333',linewidth=.65),
            medianprops=dict(color='#333333',linewidth=.7),
            flierprops=dict(marker='o',markersize=1.8,markeredgewidth=.5,markerfacecolor='white',markeredgecolor='#333333'))
        mean=float(np.mean(vals));median=float(np.median(vals))
        ax.axhline(mean*100,color='#1689e9',ls='--',lw=1.05,zorder=3)
        ax.axhline(median*100,color='#e66d43',ls=':',lw=1.05,zorder=3)
        ax.set_xlim(.45,n_groups+.55);ax.set_ylim(0,102)
        ax.set_title(panel_title+'  (n='+str(len(vals))+')',fontsize=8.35,fontweight='bold',pad=4)
        ax.set_xticks(range(1,n_groups+1));ax.set_xticklabels([str(i) for i in range(1,n_groups+1)],fontsize=6.4)
        ax.set_yticks([0,20,40,60,80,100]);ax.tick_params(axis='y',labelsize=6.4,length=2)
        ax.set_xlabel('Sequential tumour-bearing WSI group',fontsize=6.7,labelpad=2.5)
        if key in ('L0','L2'):ax.set_ylabel('Per-slide Dice (%)',fontsize=6.8)
        ax.grid(axis='y',alpha=.13,lw=.45)
        # Panel stats displayed in a blank reserved strip just above each panel, never over data.
        ax.text(.985,.965,'Mean %.2f%%\nMedian %.2f%%'%(mean*100,median*100),
                transform=ax.transAxes,ha='right',va='top',fontsize=6.25,
                bbox={'facecolor':'white','edgecolor':'#aaaaaa','linewidth':.35,'pad':1.3},clip_on=True,zorder=7)
        audit['panels'][key]={'n':len(vals),'mean_dice':mean,'median_dice':median,
          'excluded_reference_empty_at_level':int((fusion_positive&~((tp+fn)>0)).sum()),
          'group_sizes':[len(g) for g in groups]}
        print(f'[DATA] {key}: n={len(vals)}, mean={mean:.5f}, median={median:.5f}')
    
    out.mkdir(parents=True,exist_ok=True)
    png=out/'Fig09_per_slide_dice.png';pdf=out/'Fig09_per_slide_dice.pdf';tif=out/'Fig9.tif'
    fig.savefig(png,dpi=300);fig.savefig(pdf);plt.close(fig)
    with Image.open(png) as im:im.convert('RGB').save(tif,compression='tiff_lzw',dpi=(300,300))
    with Image.open(tif) as im:
        w,h=im.size;dpi=im.info.get('dpi',(0,0));compression=im.tag_v2.get(259)
    size=tif.stat().st_size
    if not (789<=w<=2250 and h<=2625 and size<=10_000_000 and compression==5 and all(abs(x-300)<1 for x in dpi)):
        raise RuntimeError(f'PLOS export failed validation: {w}x{h} {size} bytes, {dpi} dpi, compression {compression}')
    audit['outputs']={'TIFF':str(tif),'PNG':str(png),'PDF':str(pdf),'dimensions_px':[w,h],
      'dpi':[float(x) for x in dpi],'compression':'LZW','tiff_bytes':size}
    auditpath=out/'Fig09_SOURCE_AUDIT.json';auditpath.write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(f'[PLOS PASS] {tif} | {w}x{h} | {size:,} bytes | LZW | 300 DPI')
    print(f'[OUTPUT] {png}\n[OUTPUT] {pdf}\n[OUTPUT] {auditpath}')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--csv',type=Path,default=None)
    p.add_argument('--outdir',type=Path,default=OUTPUT_DEFAULT)
    p.add_argument('--groups',type=int,default=8,choices=[8,10])
    a=p.parse_args()
    draw(resolve_csv(a.csv),a.outdir,a.groups)
if __name__=='__main__':main()
