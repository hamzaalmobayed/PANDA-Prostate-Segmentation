import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


DEFAULT_ROOT = r"D:\PANDA_PROSTATE\patches\train"
LEVELS = ["L0", "L1", "L2"]
LEVEL_TITLES = {
    "L0": "L0 (1×)",
    "L1": "L1 (4× downsample)",
    "L2": "L2 (16× downsample)",
}
ROW_LABELS = ["Original", "Mask", "Overlay"]
PREFERRED_CLASSES = ["Tumor"]
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def ensure_output_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def list_images(folder: Path) -> List[Path]:
    if not folder.exists() or not folder.is_dir():
        return []
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS])


def stem_no_ext(path: Path) -> str:
    return path.stem


def open_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def open_mask_binary(path: Path) -> np.ndarray:
    arr = np.array(Image.open(path).convert("L"))
    # allow either black/white masks or grayscale masks
    return arr > 127


def resize_mask(mask: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    pil = Image.fromarray((mask.astype(np.uint8) * 255))
    pil = pil.resize(size, Image.NEAREST)
    return np.array(pil) > 127


def make_overlay(image: Image.Image, mask: np.ndarray,
                 overlay_color=(255, 76, 76), alpha=0.58) -> Image.Image:
    base = np.asarray(image, dtype=np.float32)
    if mask.shape != (image.height, image.width):
        mask = resize_mask(mask, (image.width, image.height))
    result = base.copy()
    color = np.asarray(overlay_color, dtype=np.float32)
    result[mask] = base[mask] * (1.0 - alpha) + color * alpha
    return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))


def find_case_dirs(root: Path) -> List[Path]:
    if not root.exists() or not root.is_dir():
        return []
    return sorted([p for p in root.iterdir() if p.is_dir()])


def collect_level_files(level_dir: Path, class_name: str) -> Tuple[Dict[str, Path], Dict[str, Path]]:
    # V31 uses <stem>.jpg for the patch and <stem>__tumor_mask.png
    # for the GT tumor mask. The suffixes MUST be normalized before pairing.
    imgs = {p.stem.casefold(): p for p in list_images(level_dir / class_name)}
    mask_dir = level_dir / f"{class_name}_Masks"
    masks = {}
    for p in list_images(mask_dir):
        key = p.stem.casefold()
        for suffix in ("__tumor_mask", "_tumor_mask", "__mask", "_mask"):
            if key.endswith(suffix):
                key = key[:-len(suffix)]
                break
        masks[key] = p
    return imgs, masks


def find_common_triplet(case_dir: Path) -> Optional[Tuple[str, Dict[str, Tuple[Path, Path]]]]:
    # First preference: matching names for the SAME source WSI across L0,L1,L2.
    # Patches from different levels may have different coordinates/names.
    per_level = {}
    common = None
    for level in LEVELS:
        imgs, masks = collect_level_files(case_dir / level, "Tumor")
        both = set(imgs).intersection(masks)
        if not both:
            return None
        common = both if common is None else common.intersection(both)
        per_level[level] = (imgs, masks)
    if not common:
        return None
    key = sorted(common)[0]
    return "Tumor", {lv: (per_level[lv][0][key], per_level[lv][1][key]) for lv in LEVELS}


def find_fallback_triplet(case_dir: Path) -> Optional[Tuple[str, Dict[str, Tuple[Path, Path]]]]:
    # Different resolution patches often have DIFFERENT filenames.
    # Use one valid image+tumor-mask pair per level from the same WSI.
    mapping = {}
    for level in LEVELS:
        imgs, masks = collect_level_files(case_dir / level, "Tumor")
        matches = sorted(set(imgs).intersection(masks))
        if not matches:
            return None
        # Choose positive masks preferentially; do not alter masks.
        chosen = None
        for key in matches[:25]:
            try:
                if np.asarray(Image.open(masks[key]).convert("L")).max() > 127:
                    chosen = key
                    break
            except (OSError, ValueError):
                continue
        if chosen is None:
            chosen = matches[0]
        mapping[level] = (imgs[chosen], masks[chosen])
    return "Tumor", mapping


def auto_pick_case(root: Path) -> Tuple[Path, str, Dict[str, Tuple[Path, Path]]]:
    if not root.is_dir():
        raise FileNotFoundError(f"Root folder does not exist: {root}")
    cases = find_case_dirs(root)
    print(f"Searching {len(cases)} source WSI folders under {root}", flush=True)
    for case_dir in cases:
        found = find_common_triplet(case_dir)
        if found:
            cls, mapping = found
            return case_dir, cls, mapping
    for case_dir in cases:
        found = find_fallback_triplet(case_dir)
        if found:
            cls, mapping = found
            print("NOTE: L0/L1/L2 examples are from the same WSI but not guaranteed to be spatially registered.")
            return case_dir, cls, mapping

    print("No single WSI contains valid Tumor+Tumor_Masks pairs at all three levels.")
    for case_dir in cases[:8]:
        details = []
        for level in LEVELS:
            imgs, masks = collect_level_files(case_dir / level, "Tumor")
            details.append(f"{level}: images={len(imgs)}, masks={len(masks)}, matched={len(set(imgs)&set(masks))}")
        print(case_dir.name, "|", " | ".join(details))
    raise FileNotFoundError(
        "No WSI has matching Tumor patch/mask pairs at every level. "
        "Check printed diagnostics. V31 masks are named <stem>__tumor_mask.png."
    )


def build_figure(case_dir: Path, class_name: str, mapping: Dict[str, Tuple[Path, Path]], output_path: Path):
    fig, axes = plt.subplots(3, 3, figsize=(12, 8), dpi=200)
    fig.patch.set_facecolor("white")

    for col, level in enumerate(LEVELS):
        img_path, mask_path = mapping[level]
        img = open_rgb(img_path)
        mask = open_mask_binary(mask_path)
        if mask.shape != (img.height, img.width):
            mask = resize_mask(mask, (img.width, img.height))
        overlay = make_overlay(img, mask, overlay_color=(255, 76, 76), alpha=0.58)

        panels = [img, Image.fromarray((mask.astype(np.uint8) * 255)), overlay]
        for row in range(3):
            ax = axes[row, col]
            if row == 1:
                ax.imshow(panels[row], cmap="gray", vmin=0, vmax=255)
            else:
                ax.imshow(panels[row])
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(True)
                sp.set_linewidth(2.0)
                sp.set_color("black")
        axes[0, col].set_title(LEVEL_TITLES[level], fontsize=18, fontweight="bold", pad=12)

    # Right-side vertical row labels
    x_pos = 0.955
    y_positions = [0.76, 0.505, 0.245]
    for label, y in zip(ROW_LABELS, y_positions):
        fig.text(x_pos, y, label, rotation=-90, va="center", ha="center",
                 fontsize=18, fontweight="bold")

    # optional tiny footer with provenance
    fig.text(0.5, 0.012,
             f"Case: {case_dir.name}   |   Source class: {class_name}",
             ha="center", va="bottom", fontsize=8, color="dimgray")

    plt.subplots_adjust(left=0.05, right=0.92, top=0.92, bottom=0.06, wspace=0.04, hspace=0.06)
    fig.savefig(output_path.with_suffix('.png'), dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight', facecolor='white')
    try:
        fig.savefig(output_path.with_suffix('.tif'), dpi=300, bbox_inches='tight', facecolor='white')
    except Exception:
        pass
    plt.close(fig)


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(DEFAULT_ROOT)
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).resolve().parent
    ensure_output_dir(out_dir)

    case_dir, class_name, mapping = auto_pick_case(root)
    output_base = out_dir / 'Fig_multilevel_patch_mask_overlay'
    build_figure(case_dir, class_name, mapping, output_base)

    print('DONE')
    print(f'Root used      : {root}')
    print(f'Case selected  : {case_dir}')
    print(f'Class selected : {class_name}')
    for level in LEVELS:
        img_path, mask_path = mapping[level]
        print(f'{level} image : {img_path}')
        print(f'{level} mask  : {mask_path}')
    print(f'Output PNG     : {output_base.with_suffix(".png")}')
    print(f'Output PDF     : {output_base.with_suffix(".pdf")}')
    print(f'Output TIF     : {output_base.with_suffix(".tif")}')


if __name__ == '__main__':
    main()
