"""PANDA workflow figure generated from real patch and mask files.

Usage:
    python GENERATE_PANDA_WHOLE_SLIDE_SEGMENTATION_WORKFLOW.py
    python GENERATE_PANDA_WHOLE_SLIDE_SEGMENTATION_WORKFLOW.py "D:\\PANDA_PROSTATE\\patches\\train"

Only uses actual files beneath the patch root; no synthetic histology images or masks.
The left pyramid visualizes patches, NOT an actual WSI. Sources are logged to JSON.
"""


def _panda_portable_path(original):
    """Resolve legacy PANDA paths through configurable project roots."""
    import os
    from pathlib import Path

    text = str(original)
    normalized = text.replace("/", "\\")

    roots = [
        (
            "PANDA_ARTICLE_ROOT",
            'D:\\PANDA_PROSTATE_FINAL_ARTICLE_COMPLETE_SAFE_RESUME_V31_KAGGLE_SICAP_DOWNLOAD_FIX\\final_article_pkg'
        ),
        (
            "PANDA_DATA_ROOT",
            r"D:\PANDA_PROSTATE"
        ),
    ]

    for variable, legacy_root in roots:
        legacy = legacy_root.replace("/", "\\").rstrip("\\")
        if (
            normalized.lower() == legacy.lower()
            or normalized.lower().startswith(legacy.lower() + "\\")
        ):
            root = os.environ.get(variable, legacy)
            suffix = normalized[len(legacy):].lstrip("\\")
            return str(Path(root).joinpath(*[
                x for x in suffix.split("\\") if x
            ]))

    return original


from pathlib import Path
import json
import sys
import numpy as np
from PIL import Image, ImageOps
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch

ROOT = Path(_panda_portable_path(r'D:\PANDA_PROSTATE\patches\train'))
NAME = 'Fig_panda_whole_slide_segmentation_workflow'
EXTS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}
NAVY = '#0c1644'
ARROW = '#186bd1'
BORDER = '#91ade3'


def files(folder):
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in EXTS)


def first_patch(root, level, category, preferred=None, exclude=None, index=0):
    exclude = set(str(p) for p in (exclude or []))
    cases = [preferred] if preferred else []
    cases += [p for p in sorted(root.iterdir()) if p.is_dir() and p != preferred]
    found = []
    for case in cases:
        paths = [p for p in files(case / level / category) if str(p) not in exclude]
        found.extend(paths)
    if found:
        idx = max(0, min(index, len(found)-1))
        return found[idx]
    raise FileNotFoundError(f'No {level}/{category} image found in {root}')


def get_mask(root, level, preferred=None):
    cases = [preferred] if preferred else []
    cases += [p for p in sorted(root.iterdir()) if p.is_dir() and p != preferred]
    # Require actual positive pixels in masks, never use validity masks.
    for case in cases:
        for path in files(case / level / 'Tumor_Masks')[:30]:
            try:
                a = np.array(Image.open(path).convert('L'))
                if a.max() > 0:
                    return path
            except Exception:
                continue
    raise FileNotFoundError(f'No non-empty tumor mask for {level}. Looked in Tumor_Masks.')


def fit_patch(path, binary=False, size=300):
    im = Image.open(path)
    if binary:
        im = im.convert('L')
        a = np.asarray(im)
        im = Image.fromarray(((a > 0) * 255).astype('uint8'), 'L').convert('RGB')
        return np.asarray(ImageOps.fit(im, (size, size), method=Image.Resampling.NEAREST))
    return np.asarray(ImageOps.fit(im.convert('RGB'), (size, size), method=Image.Resampling.LANCZOS))


def rbox(ax, x, y, w, h, fc, ec, lw=1.4, rounding=0.012):
    p = FancyBboxPatch((x, y), w, h,
                      boxstyle=f'round,pad=0.003,rounding_size={rounding}',
                      fc=fc, ec=ec, lw=lw, zorder=1)
    ax.add_patch(p)
    return p


def text(ax, x, y, value, fs=11, bold=False, ha='center', **kw):
    ax.text(x, y, value, fontsize=fs, ha=ha, va='center', color=kw.pop('color', NAVY),
            fontweight='bold' if bold else 'normal', zorder=5, **kw)


def arrow(ax, a, b, lw=2.4, head=13, color=ARROW):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle='-|>', mutation_scale=head,
                                 color=color, lw=lw, shrinkA=0, shrinkB=0,
                                 connectionstyle='arc3', zorder=4))


def tile(ax, image, x, y, w, h, edge, lw=2):
    ax.imshow(image, extent=(x, x+w, y, y+h), aspect='auto', interpolation='nearest', zorder=2)
    ax.add_patch(Rectangle((x, y), w, h, fill=False, ec=edge, lw=lw, zorder=3))


def figure(root: Path, output: Path):
    if not root.is_dir():
        raise FileNotFoundError(f'Patch directory not found: {root}')
    preferred = None
    for case in sorted(root.iterdir()):
        if case.is_dir() and all(files(case / l / 'Tumor') for l in ('L0', 'L1', 'L2')):
            preferred = case
            break

    records = {}
    normal = []
    tumor = []
    right_paths = []
    masks = []
    for l in ('L2', 'L1', 'L0'):
        n = first_patch(root, l, 'Normal', preferred)
        t = first_patch(root, l, 'Tumor', preferred)
        normal.append(fit_patch(n)); tumor.append(fit_patch(t))
        right_paths.extend([n, t])
        records[f'normal_{l}'] = str(n)
        records[f'tumor_{l}'] = str(t)

    pyramid = {}
    for l in ('L0', 'L1', 'L2'):
        p = first_patch(root, l, 'Tumor', preferred, exclude=right_paths, index=0)
        pyramid[l] = fit_patch(p, size=400)
        records[f'pyramid_{l}'] = str(p)
    for l in ('L2', 'L1', 'L0', 'L2', 'L1', 'L0'):
        m = get_mask(root, l, preferred)
        masks.append(fit_patch(m, binary=True)); records[f'mask_{l}_{len(masks)}'] = str(m)

    # Axes dimensions fixed so all text/graphics have separate, non-overlapping regions.
    fig = plt.figure(figsize=(16, 9), facecolor='white')
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 16); ax.set_ylim(0, 9); ax.axis('off')
    ax.set_aspect('auto')

    text(ax, .45, 8.7, 'Figure 01  |  PANDA workflow', fs=11, bold=True, ha='left', color='#344861')
    text(ax, 8, 8.1, 'PANDA whole-slide segmentation workflow', fs=23, bold=True)

    # Left: whole-slide biopsy image panel using different real patch files from the right panel
    xL, yL, wL, hL = 0.5, 0.65, 4.5, 6.85
    rbox(ax, xL, yL, wL, hL, '#fcfdff', BORDER)
    text(ax, xL+wL/2, 7.15, 'Whole-Slide Biopsy Image', fs=14, bold=True)
    text(ax, xL+wL/2, 6.89, 'Three-level multiresolution input', fs=9.2)

    for level, yy, hh, ww in [('L2', 5.00, 1.45, 2.45), ('L1', 3.05, 1.6, 3.05), ('L0', 1.15, 1.6, 3.65)]:
        xx = xL+(wL-ww)/2
        tile(ax, pyramid[level], xx, yy, ww, hh, '#a5bedf', lw=1.6)
        # label outside image but within panel, above the image
        label = {'L2':'L2 (16× downsample)', 'L1':'L1 (4× downsample)', 'L0':'L0 (1×)'}[level]
        rbox(ax, xL+.20, yy+hh+.05, 1.7 if level != 'L0' else 1.0, .30, '#e1e4f4', '#e1e4f4', .3, .045)
        text(ax, xL+.20+(1.7 if level != 'L0' else 1.0)/2, yy+hh+.20, label, fs=9.4, bold=True)

    # Middle six workflow blocks; every title stays inside its own box
    mx, mw, mh = 5.50, 4.75, .91
    cy = [6.97, 5.80, 4.63, 3.46, 2.29, 1.12]
    defs = [
        ('Preprocessing', 'Tissue masking, QC, stain normalization', '#f6d2df', '#e85a90', 13.6),
        ('Feature Engineering', 'Multilevel patches and coordinates', '#ffe6a9', '#e8a700', 13.6),
        ('Dataset Construction', '12-stratum structured sampling', '#d2e4fa', '#5da8ff', 13.6),
        ('Deep Learning Segmentation Model', 'EfficientNet-B4 U-Net', '#e2ccfa', '#a05aff', 11.8),
        ('Whole-Slide Reconstruction & Fusion', 'Dense streaming, support-aware fusion', '#dcefd1', '#86c95b', 11.6),
        ('Evaluation', 'Dice, IoU, HD95, ASSD', '#d5cbf7', '#8a68ff', 13.6),
    ]
    for center, (head, sub, bg, edge, headfs) in zip(cy, defs):
        bottom = center-mh/2
        rbox(ax, mx, bottom, mw, mh, bg, edge, 1.65)
        text(ax, mx+mw/2, center+.15, head, fs=headfs, bold=True)
        text(ax, mx+mw/2, center-.21, sub, fs=9.6)
    for a,b in zip(cy,cy[1:]):
        arrow(ax, (mx+mw/2,a-mh/2-.04), (mx+mw/2,b+mh/2+.05), lw=2.9)

    # Right panels -- panels wide enough for 3 tiles and readable captions
    rx, rw = 10.75, 4.75
    rbox(ax,rx,4.60,rw,2.90,'#fcfcfe',BORDER)
    text(ax,rx+rw/2,7.15,'Representative Patch Inputs',fs=13.5,bold=True)
    tilew, tileh = 1.32, .87
    xs=[rx+.25,rx+1.72,rx+3.19]
    for i,x in enumerate(xs):
        tile(ax,normal[i],x,5.99,tilew,tileh,'#10b84c')
        tile(ax,tumor[i],x,4.97,tilew,tileh,'#fa8c12')
    ax.add_patch(Rectangle((rx+.33,4.73),.16,.16,fill=False,ec='#10b84c',lw=2))
    text(ax,rx+.57,4.81,'Normal',fs=9,ha='left')
    ax.add_patch(Rectangle((rx+2.26,4.73),.16,.16,fill=False,ec='#fa8c12',lw=2))
    text(ax,rx+2.5,4.81,'Tumor',fs=9,ha='left')

    rbox(ax,rx,.65,rw,3.52,'#fcfcfe',BORDER)
    text(ax,rx+rw/2,3.88,'Segmentation Mask Patches',fs=13.5,bold=True)
    for i,x in enumerate(xs):
        tile(ax,masks[i],x,2.67,tilew,.93,'#ffffff',lw=1)
        tile(ax,masks[i+3],x,1.54,tilew,.93,'#ffffff',lw=1)
    text(ax,rx+rw/2,1.01,'Binary tumor masks from original patch files',fs=9.5)

    # Connections drawn only through the gutters
    arrow(ax,(xL+wL+.03,cy[0]),(mx-.08,cy[0]),lw=3)
    # Patch inputs feed feature engineering (arrow travels from right to left)
    arrow(ax,(rx-.08,cy[1]),(mx+mw+.06,cy[1]),lw=3)
    # model to mask patch panel
    arrow(ax,(mx+mw+.06,cy[3]),(rx-.08,cy[3]),lw=3)
    # Return loop beneath content, never through blocks
    ax.plot([mx+mw/2,mx+mw/2,2.45,2.45],[.62,.33,.33,.57],color=ARROW,lw=2.7)
    arrow(ax,(2.45,.57),(2.45,.73),lw=2.7)

    output.mkdir(parents=True,exist_ok=True)
    for ext in ('png','pdf','tif'):
        fig.savefig(output/f'{NAME}.{ext}',dpi=300,facecolor='white',bbox_inches=None)
    plt.close(fig)
    (output/f'{NAME}_sources.json').write_text(json.dumps({'root':str(root), 'sources':records,
        'note':'Left pyramid is illustrative patches, not a complete WSI. Mask panels use real patch masks.'},indent=2),encoding='utf-8')
    print(f'Saved figures and provenance JSON to: {output}')


if __name__ == '__main__':
    root = Path(sys.argv[1]) if len(sys.argv)>1 else ROOT
    out = Path(__file__).resolve().parent
    figure(root,out)
