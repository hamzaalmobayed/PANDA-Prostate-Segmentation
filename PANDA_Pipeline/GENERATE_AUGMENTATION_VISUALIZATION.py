"""Generate the six-panel AUGMENTATION VISUALIZATION from a real PANDA patch.

Run on Windows:
    python GENERATE_AUGMENTATION_VISUALIZATION.py
Or choose a particular patch:
    python GENERATE_AUGMENTATION_VISUALIZATION.py "D:\\PANDA_PROSTATE\\patches\\train\\...\\L0\\Tumor\\patch.png"

Dependencies: pip install numpy pillow matplotlib
Outputs next to this script: PNG (300 dpi), PDF and TIFF (300 dpi).
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

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(_panda_portable_path(r'D:\PANDA_PROSTATE\patches'))
NAME = 'Fig_augmentation_visualization'
SEED = 42
EXTS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}


def choose_patch(root: Path) -> Path:
    if not root.exists():
        raise FileNotFoundError(f'Patch root does not exist: {root}')
    # Prefer one PANDA training patch at L0 (real RGB image, not a mask).
    for split in ('train', 'val', 'test'):
        base = root / split
        if not base.exists():
            continue
        for cls in ('Tumor', 'Normal'):
            hits = sorted(p for p in base.glob(f'*/L0/{cls}/*') if p.is_file() and p.suffix.lower() in EXTS)
            if hits:
                return hits[0]
    raise FileNotFoundError('No images found in <root>/<split>/<WSI_ID>/L0/(Tumor|Normal)/')


def transform(img: Image.Image):
    rgb = img.convert('RGB')
    flip = rgb.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    rotate = rgb.transpose(Image.Transpose.ROTATE_90)
    # Fixed, documented illustrative transforms; no new anatomical content is generated.
    jitter = ImageEnhance.Color(ImageEnhance.Contrast(ImageEnhance.Brightness(rgb).enhance(1.06)).enhance(1.12)).enhance(1.20)
    blur = rgb.filter(ImageFilter.GaussianBlur(radius=2.1))
    arr = np.asarray(rgb, dtype=np.float32)
    rng = np.random.default_rng(SEED)
    noise = np.clip(arr + rng.normal(0, 12.0, size=arr.shape), 0, 255).astype(np.uint8)
    return [rgb, flip, rotate, jitter, blur, Image.fromarray(noise)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('patch', nargs='?', type=Path, help='Optional exact path to a source patch.')
    ap.add_argument('--root', type=Path, default=ROOT, help='Root containing patches/train, val, test.')
    args = ap.parse_args()
    path = args.patch if args.patch else choose_patch(args.root)
    if not path.is_file():
        raise FileNotFoundError(path)
    with Image.open(path) as opened:
        panels = transform(opened)

    titles = ['Original', 'Flip', 'Rotate', 'Color Jitter', 'Blur', 'Noise']
    fig, axes = plt.subplots(2, 3, figsize=(12, 8.4), dpi=300)
    for ax, title, pic in zip(axes.flat, titles, panels):
        ax.imshow(pic)
        ax.set_title(title, fontsize=19, fontweight='bold', pad=9)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('#111111')
            spine.set_linewidth(1.2)
    fig.suptitle('AUGMENTATION VISUALIZATION', fontsize=24, fontweight='bold', y=0.98)
    fig.subplots_adjust(left=0.045, right=0.955, top=0.87, bottom=0.055, wspace=0.095, hspace=0.21)

    outdir = Path(__file__).resolve().parent
    for ext in ('png', 'pdf', 'tif'):
        target = outdir / f'{NAME}.{ext}'
        fig.savefig(target, dpi=300, bbox_inches='tight', facecolor='white')
        print('Saved:', target)
    plt.close(fig)
    print('Original source patch:', path)
    print('Illustrative augmentations: horizontal flip, 90-degree rotation, brightness/contrast/color jitter, Gaussian blur, Gaussian noise.')


if __name__ == '__main__':
    main()
