from pathlib import Path
import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Build a 2x3 composite figure from one review folder.
# It looks for the original image, GT mask, and prediction mask,
# then creates:
# (A) Original
# (B) GT Mask
# (C) Pred Mask
# (D) GT Overlay
# (E) Pred Overlay
# (F) Error Map

IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}


def find_first(folder: Path, patterns):
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    lower_files = [(p, p.name.lower()) for p in files]
    for pat_group in patterns:
        for p, low in lower_files:
            if all(token in low for token in pat_group):
                return p
    return None


def detect_files(folder: Path):
    original = find_first(folder, [
        ('detail', 'original'),
        ('detail', 'ori'),
        ('original',),
    ])
    gt_mask = find_first(folder, [
        ('detail', 'gt', 'mask'),
        ('gt', 'mask'),
        ('detail', 'gt'),
        ('ground', 'truth'),
    ])
    pred_mask = find_first(folder, [
        ('detail', 'pred', 'mask'),
        ('pred', 'mask'),
        ('detail', 'model'),
        ('model',),
    ])
    return original, gt_mask, pred_mask


def open_rgb(path: Path):
    return Image.open(path).convert('RGB')


def resize_to(img: Image.Image, size, is_mask=False):
    resample = Image.Resampling.NEAREST if is_mask else Image.Resampling.BILINEAR
    return img.resize(size, resample)


def to_binary_mask(mask_img: Image.Image):
    gray = mask_img.convert('L')
    arr = np.array(gray)
    return arr > 127


def get_tissue_mask(original_rgb: Image.Image):
    arr = np.array(original_rgb).astype(np.uint8)
    # Nearly-white = background; anything darker somewhere = tissue.
    tissue = np.any(arr < 245, axis=2)
    return tissue


def make_overlay(original_rgb: Image.Image, mask_bool, color=(95, 205, 95), alpha=0.55):
    base = np.array(original_rgb).astype(np.float32)
    out = base.copy()
    color_arr = np.array(color, dtype=np.float32).reshape(1, 1, 3)
    mask3 = mask_bool.astype(np.float32)[..., None]
    # Blend only where mask == 1, keep original elsewhere.
    out = base * (1.0 - alpha * mask3) + color_arr * (alpha * mask3)
    return np.clip(out, 0, 255).astype(np.uint8)


def make_error_map(gt, pred, tissue):
    h, w = gt.shape
    out = np.ones((h, w, 3), dtype=np.uint8) * 255
    out[tissue] = np.array([235, 235, 235], dtype=np.uint8)  # TN shown as light gray tissue

    tp = gt & pred
    fn = gt & (~pred)
    fp = (~gt) & pred & tissue

    out[tp] = np.array([95, 205, 95], dtype=np.uint8)   # green
    out[fn] = np.array([242, 214, 63], dtype=np.uint8)  # yellow
    out[fp] = np.array([234, 56, 41], dtype=np.uint8)   # red
    return out


def plot_panel(ax, img, title):
    if img.ndim == 2:
        ax.imshow(img, cmap='gray', vmin=0, vmax=255)
    else:
        ax.imshow(img)
    ax.set_title(title, fontsize=16, fontweight='bold', pad=8, fontfamily='serif')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.2)
        spine.set_color('black')


def build_figure(folder: Path, output_path: Path = None):
    original_path, gt_path, pred_path = detect_files(folder)

    missing = []
    if original_path is None:
        missing.append('original image')
    if gt_path is None:
        missing.append('GT mask')
    if pred_path is None:
        missing.append('prediction mask')
    if missing:
        raise FileNotFoundError(
            'Could not find required file(s): ' + ', '.join(missing) + '\n'
            'Folder checked: ' + str(folder)
        )

    original = open_rgb(original_path)
    gt_img = resize_to(Image.open(gt_path), original.size, is_mask=True)
    pred_img = resize_to(Image.open(pred_path), original.size, is_mask=True)

    gt = to_binary_mask(gt_img)
    pred = to_binary_mask(pred_img)
    tissue = get_tissue_mask(original)

    gt_overlay = make_overlay(original, gt, color=(95, 205, 95), alpha=0.55)
    pred_overlay = make_overlay(original, pred, color=(95, 205, 95), alpha=0.55)
    error_map = make_error_map(gt, pred, tissue)

    gt_mask_vis = np.where(gt, 255, 0).astype(np.uint8)
    pred_mask_vis = np.where(pred, 255, 0).astype(np.uint8)

    fig, axes = plt.subplots(2, 3, figsize=(14, 9), dpi=300)
    plot_panel(axes[0, 0], np.array(original), '(A) Original')
    plot_panel(axes[0, 1], gt_mask_vis, '(B) GT Mask')
    plot_panel(axes[0, 2], pred_mask_vis, '(C) Pred Mask')
    plot_panel(axes[1, 0], gt_overlay, '(D) GT Overlay')
    plot_panel(axes[1, 1], pred_overlay, '(E) Pred Overlay')
    plot_panel(axes[1, 2], error_map, '(F) Error Map')

    legend_handles = [
        Patch(facecolor=(95/255, 205/255, 95/255), edgecolor='black', label='Correct (Overlap)'),
        Patch(facecolor=(242/255, 214/255, 63/255), edgecolor='black', label='False Negative (Missed)'),
        Patch(facecolor=(234/255, 56/255, 41/255), edgecolor='black', label='False Positive (Extra)'),
        Patch(facecolor=(235/255, 235/255, 235/255), edgecolor='black', label='True Negative (Background)'),
    ]

    fig.legend(
        handles=legend_handles,
        loc='lower center',
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.02),
        prop={'family': 'serif', 'size': 11},
        handlelength=1.8,
        handleheight=1.8,
    )
    fig.subplots_adjust(left=0.04, right=0.98, top=0.93, bottom=0.12, wspace=0.05, hspace=0.15)

    if output_path is None:
        output_path = folder / f'{folder.name}_composite_figure.png'

    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return output_path, original_path, gt_path, pred_path


if __name__ == '__main__':
    if len(sys.argv) >= 2:
        folder = Path(sys.argv[1])
    else:
        folder = Path.cwd()

    if not folder.exists() or not folder.is_dir():
        raise SystemExit(f'Folder not found: {folder}')

    out, orig, gt, pred = build_figure(folder)
    print('Composite figure saved to:')
    print(out)
    print('\nFiles used:')
    print('Original :', orig)
    print('GT mask  :', gt)
    print('Pred mask:', pred)
