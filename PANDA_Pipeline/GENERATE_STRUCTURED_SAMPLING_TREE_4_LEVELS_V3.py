"""Fig 6: four-tier schematic of the realised PANDA sampling allocation.

Updated per user request:
- Keep overall design/style.
- Final column shows realised combined groups (e.g., PANDA L0 Normal),
  not percentages.
- Remove extra explanatory footer text; keep only one bold line stating
  that each realised group contains 13,333 or 13,334 patches.
- Export PLOS-compliant TIFF (LZW, 300 DPI, <=2250 px width, <=10 MB).

Run:
    python GENERATE_STRUCTURED_SAMPLING_TREE_4_LEVELS_V3.py
Optional:
    python GENERATE_STRUCTURED_SAMPLING_TREE_4_LEVELS_V3.py --outdir "D:\\...\\PLOS_FIXED"
"""
from __future__ import annotations


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


import argparse
from pathlib import Path
from datetime import datetime
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from PIL import Image

DEFAULT_DIR = Path(_panda_portable_path(r'D:\PANDA_PROSTATE\PANDA_Figures_1-17_Editable_Replacement_Templates.png\PLOS_FIXED'))


def box(ax, x, y, w, h, label, fill, edge, size=12, weight='bold'):
    obj = FancyBboxPatch(
        (x, y - h / 2), w, h,
        boxstyle='round,pad=0.006,rounding_size=0.012',
        fc=fill, ec=edge, lw=1.55, zorder=3
    )
    ax.add_patch(obj)
    ax.text(
        x + w / 2, y, label,
        ha='center', va='center',
        fontsize=size, fontweight=weight,
        color='#202735', zorder=4,
        linespacing=1.1,
        multialignment='center'
    )


def arrow(ax, x1, y1, x2, y2):
    a = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle='-|>', mutation_scale=14, lw=1.3,
        fc='#344456', ec='#344456',
        connectionstyle='arc3,rad=0',
        shrinkA=0, shrinkB=0, zorder=2
    )
    ax.add_patch(a)


def draw(outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12.5, 7.75), dpi=300)
    fig.patch.set_facecolor('white')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # Four columns: data -> level -> class -> realised group.
    X = (0.025, 0.283, 0.540, 0.804)
    W = (0.158, 0.154, 0.172, 0.177)
    H = 0.067
    leaf_y = [0.815, 0.710, 0.605, 0.500, 0.395, 0.290]
    level_y = [(leaf_y[i] + leaf_y[i + 1]) / 2 for i in (0, 2, 4)]
    root_y = (level_y[0] + level_y[2]) / 2

    ax.text(
        0.5, 0.967, 'Structured sampling of PANDA training patches',
        ha='center', va='center', fontsize=19,
        fontweight='bold', color='#202735'
    )

    headers = ['Data', 'Pyramid level', 'Reference class', 'Realized group']
    for x, w, lab in zip(X, W, headers):
        ax.text(
            x + w / 2, 0.903, lab,
            ha='center', va='center', fontsize=12.6,
            color='#333b45', fontweight='bold'
        )

    # Root
    box(ax, X[0], root_y, W[0], 0.080, 'PANDA', '#dce7fc', '#6785b7', 14)

    level_labels = ['L0  (1x)', 'L1  (4x)', 'L2  (16x)']
    class_labels = ['Normal', 'Tumor']

    for i, (lvl, y) in enumerate(zip(level_labels, level_y)):
        box(ax, X[1], y, W[1], 0.071, lvl, '#ffefc4', '#cc9f45', 12.5)
        arrow(ax, X[0] + W[0] + 0.006, root_y, X[1] - 0.009, y)

        clean_lvl = lvl.split()[0]  # L0 / L1 / L2
        for j, cls in enumerate(class_labels):
            k = i * 2 + j
            yy = leaf_y[k]
            fc, ec = (('#e4f2dd', '#7caf71') if cls == 'Normal' else ('#ffdfdf', '#d78a8a'))
            box(ax, X[2], yy, W[2], H, cls, fc, ec, 12.5)
            arrow(ax, X[1] + W[1] + 0.006, y, X[2] - 0.009, yy)

            final_label = f'PANDA {clean_lvl}\n{cls}'
            box(ax, X[3], yy, W[3], H, final_label, '#edf0f4', '#94a2b3', 11.8)
            arrow(ax, X[2] + W[2] + 0.006, yy, X[3] - 0.009, yy)

    # Minimal footer: one bold line only
    ax.text(
        0.5, 0.132,
        'Each realized group contains 13,333 or 13,334 patches',
        ha='center', va='center', fontsize=13.0,
        fontweight='bold', color='#293647'
    )

    fig.subplots_adjust(left=0.018, right=0.982, bottom=0.018, top=0.992)

    png = outdir / 'Fig06_sampling.png'
    pdf = outdir / 'Fig06_sampling.pdf'
    tiff = outdir / 'Fig6.tif'

    if tiff.exists():
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        shutil.copy2(tiff, outdir / f'Fig6_before_update_{stamp}.tif')

    fig.savefig(png, dpi=300, facecolor='white')
    fig.savefig(pdf, facecolor='white')
    plt.close(fig)

    with Image.open(png) as im:
        im = im.convert('RGB')
        width = 2250
        height = round(im.height * width / im.width)
        if height > 2625:
            height = 2625
            width = round(im.width * height / im.height)
        im = im.resize((width, height), Image.Resampling.LANCZOS)
        im.save(tiff, format='TIFF', compression='tiff_lzw', dpi=(300, 300))

    with Image.open(tiff) as check:
        assert check.tag_v2.get(259) == 5, 'TIFF compression is not LZW'
        assert check.width <= 2250 and check.height <= 2625, 'TIFF dimensions exceed PLOS limits'
    assert tiff.stat().st_size <= 10_000_000, 'TIFF exceeds 10 MB'

    print('[OK] Wrote:')
    print(' PNG :', png)
    print(' PDF :', pdf)
    print(' TIFF:', tiff)
    print(f' TIFF dimensions: {width} x {height} at 300 DPI, LZW compressed')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--outdir', type=Path, default=DEFAULT_DIR)
    args = p.parse_args()
    draw(args.outdir)
