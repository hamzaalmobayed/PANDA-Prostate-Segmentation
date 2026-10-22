

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


import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyArrowPatch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from PIL import Image, ImageOps, ImageEnhance, ImageFilter

# Optional OpenSlide support for large WSI TIFFs
try:
    import openslide  # type: ignore
    HAS_OPENSLIDE = True
except Exception:
    HAS_OPENSLIDE = False

Image.MAX_IMAGE_PIXELS = None


# -----------------------------
# Configuration
# -----------------------------
DEFAULT_PATCH_ROOT = _panda_portable_path(r"D:\PANDA_PROSTATE\patches\train")
DEFAULT_WSI_ROOT = r"D:\prostat\train_images"
DEFAULT_MASK_ROOT = r"D:\prostat\train_label_masks"
DEFAULT_VISUAL_ROOT = _panda_portable_path(r"D:\PANDA_PROSTATE\paper_results\test_visuals_good_live")  # optional direct preview/mask source

PANEL_COLORS = {
    1: "#2F80ED",   # blue
    2: "#22A745",   # green
    3: "#8A2BE2",   # purple
    4: "#2F80ED",   # blue
    5: "#FF7A00",   # orange
    6: "#FF2D8B",   # pink
    7: "#23B5D3",   # cyan
    8: "#2F80ED",   # blue
    9: "#FF7A00",   # orange
    10: "#FF2D8B",  # pink
}


# -----------------------------
# Utility helpers
# -----------------------------
def find_image_files(folder: Path) -> List[Path]:
    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
    if not folder.exists():
        return []
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts])


def pick_first_existing(paths: List[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None


def safe_mkdir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def to_rgb(arr_or_img):
    if isinstance(arr_or_img, Image.Image):
        return arr_or_img.convert("RGB")
    arr = np.asarray(arr_or_img)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    return Image.fromarray(arr.astype(np.uint8)).convert("RGB")


def resize_keep(im: Image.Image, size: Tuple[int, int]) -> Image.Image:
    im = im.copy()
    im.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, (255, 255, 255))
    x = (size[0] - im.width) // 2
    y = (size[1] - im.height) // 2
    canvas.paste(im, (x, y))
    return canvas


def mask_bbox(mask: np.ndarray, pad_ratio: float = 0.06) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask)
    if ys.size == 0 or xs.size == 0:
        return None
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    h, w = mask.shape[:2]
    pad_y = max(8, int((y1 - y0 + 1) * pad_ratio))
    pad_x = max(8, int((x1 - x0 + 1) * pad_ratio))
    x0 = max(0, x0 - pad_x)
    y0 = max(0, y0 - pad_y)
    x1 = min(w, x1 + pad_x + 1)
    y1 = min(h, y1 + pad_y + 1)
    return x0, y0, x1, y1


def crop_to_bbox(img: Image.Image, bbox: Optional[Tuple[int, int, int, int]]) -> Image.Image:
    if bbox is None:
        return img.copy()
    x0, y0, x1, y1 = bbox
    return img.crop((x0, y0, x1, y1))


def crop_to_content(img: Image.Image, white_thresh: int = 245) -> Image.Image:
    arr = np.asarray(img.convert("RGB"))
    gray = arr.mean(axis=2)
    std = arr.std(axis=2)
    fg = (gray < white_thresh) | (std > 6)
    bbox = mask_bbox(fg, pad_ratio=0.08)
    return crop_to_bbox(img, bbox)


def wrap_text(text: str, max_chars: int = 22) -> str:
    words = text.split()
    lines = []
    cur = []
    cur_len = 0
    for w in words:
        add = len(w) + (1 if cur else 0)
        if cur and cur_len + add > max_chars:
            lines.append(" ".join(cur))
            cur = [w]
            cur_len = len(w)
        else:
            cur.append(w)
            cur_len += add
    if cur:
        lines.append(" ".join(cur))
    return "\n".join(lines)


def open_slide_thumbnail(path: Path, size=(900, 900)) -> Image.Image:
    if HAS_OPENSLIDE:
        try:
            slide = openslide.OpenSlide(str(path))
            thumb = slide.get_thumbnail(size)
            return thumb.convert("RGB")
        except Exception:
            pass
    img = Image.open(path)
    img = img.convert("RGB")
    return resize_keep(img, size)


def open_mask_thumbnail(path: Path, size=(900, 900)) -> Image.Image:
    if HAS_OPENSLIDE:
        try:
            slide = openslide.OpenSlide(str(path))
            thumb = slide.get_thumbnail(size)
            return thumb.convert("L")
        except Exception:
            pass
    img = Image.open(path)
    if img.mode != "L":
        arr = np.array(img)
        if arr.ndim == 3:
            arr = arr[..., 0]
        img = Image.fromarray(arr.astype(np.uint8)).convert("L")
    return resize_keep(img.convert("RGB"), size).convert("L")




def compute_preview_foreground_mask(rgb: np.ndarray) -> np.ndarray:
    """Foreground mask for already-prepared preview/overlay images from good_live folders."""
    gray = rgb.mean(axis=2)
    std = rgb.std(axis=2)
    # Keep anything that is not near-white OR has visible color variation.
    fg = (gray < 247) | (std > 4)
    # remove tiny noise by keeping only rows/cols that contain enough foreground
    row_sum = fg.sum(axis=1)
    col_sum = fg.sum(axis=0)
    good_rows = row_sum > max(2, int(0.003 * fg.shape[1]))
    good_cols = col_sum > max(2, int(0.003 * fg.shape[0]))
    if good_rows.any() and good_cols.any():
        fg = fg & good_rows[:, None] & good_cols[None, :]
    return fg

def compute_tissue_mask(rgb: np.ndarray) -> np.ndarray:
    # Tissue = non-white pixels with some chroma
    gray = rgb.mean(axis=2)
    std = rgb.std(axis=2)
    mask = (gray < 238) & (std > 6)
    return mask


def compute_quality_keep_mask(rgb: np.ndarray, tissue_mask: np.ndarray) -> np.ndarray:
    gray = rgb.mean(axis=2)
    sat = rgb.std(axis=2)
    keep = tissue_mask & (gray < 235) & (sat > 8)
    return keep


def make_colored_mask(mask: np.ndarray, fg=(0, 128, 0), bg=(255, 255, 255)) -> np.ndarray:
    out = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    out[:] = bg
    out[mask] = fg
    return out


def make_keep_remove_visual(keep_mask: np.ndarray, tissue_mask: np.ndarray) -> np.ndarray:
    out = np.full((keep_mask.shape[0], keep_mask.shape[1], 3), 255, dtype=np.uint8)
    remove = tissue_mask & (~keep_mask)
    out[remove] = np.array([215, 215, 215], dtype=np.uint8)
    out[keep_mask] = np.array([219, 188, 255], dtype=np.uint8)
    return out


def overlay_mask(base: np.ndarray, mask: np.ndarray, color=(255, 90, 90), alpha=0.45) -> np.ndarray:
    out = base.astype(np.float32).copy()
    c = np.array(color, dtype=np.float32)
    out[mask] = (1 - alpha) * out[mask] + alpha * c
    return np.clip(out, 0, 255).astype(np.uint8)


def draw_grid_info(mask: np.ndarray, n_cols=12, n_rows=22):
    h, w = mask.shape
    xs = np.linspace(0, w - 1, n_cols + 1).astype(int)
    ys = np.linspace(0, h - 1, n_rows + 1).astype(int)
    cells = []
    for r in range(n_rows):
        for c in range(n_cols):
            x0, x1 = xs[c], xs[c + 1]
            y0, y1 = ys[r], ys[r + 1]
            cells.append((x0, y0, x1 - x0, y1 - y0))
    return cells


def classify_patch_cells(tissue_mask: np.ndarray, tumor_mask: np.ndarray, cells) -> List[Dict]:
    items = []
    for (x, y, w, h) in cells:
        m_tissue = tissue_mask[y:y + h, x:x + w]
        m_tumor = tumor_mask[y:y + h, x:x + w]
        tissue_ratio = float(m_tissue.mean()) if m_tissue.size else 0.0
        tumor_ratio = float(m_tumor.mean()) if m_tumor.size else 0.0
        if tissue_ratio < 0.15:
            label = "discard"
        elif 0.0 < tumor_ratio < 0.05:
            label = "borderline"
        elif tumor_ratio >= 0.05:
            label = "tumor"
        else:
            label = "normal"
        items.append({
            "rect": (x, y, w, h),
            "tissue_ratio": tissue_ratio,
            "tumor_ratio": tumor_ratio,
            "label": label,
        })
    return items


def normalize_patch_lab_like(patch: Image.Image) -> Image.Image:
    # Lightweight stain-normalization-like appearance adjustment
    arr = np.asarray(patch.convert("RGB")).astype(np.float32)
    mu = arr.reshape(-1, 3).mean(axis=0)
    sigma = arr.reshape(-1, 3).std(axis=0) + 1e-6
    target_mu = np.array([210, 150, 185], dtype=np.float32)
    target_sigma = np.array([30, 25, 28], dtype=np.float32)
    norm = (arr - mu) / sigma * target_sigma + target_mu
    norm = np.clip(norm, 0, 255).astype(np.uint8)
    img = Image.fromarray(norm)
    img = ImageEnhance.Contrast(img).enhance(1.08)
    img = ImageEnhance.Color(img).enhance(1.03)
    return img


def add_panel_chrome(ax, step_num: int, title: str):
    color = PANEL_COLORS[step_num]
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_facecolor("white")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.add_patch(Rectangle((0.005, 0.005), 0.99, 0.99, fill=False, lw=2.0, ec=color,
                           transform=ax.transAxes, clip_on=False))
    circ = Circle((0.07, 0.955), 0.040, transform=ax.transAxes, fc=color, ec=color, lw=1.2)
    ax.add_patch(circ)
    ax.text(0.07, 0.955, str(step_num), ha="center", va="center", transform=ax.transAxes,
            fontsize=15, fontweight="bold", color="white")
    ax.text(0.53, 0.975, title, ha="center", va="top", transform=ax.transAxes,
            fontsize=13.0, fontweight="bold", color="#222222", wrap=True, linespacing=0.95)


def show_centered_image(ax, img: Image.Image, box=(0.08, 0.18, 0.84, 0.68)):
    iax = ax.inset_axes(box)
    iax.imshow(img)
    iax.axis("off")
    return iax


def put_caption(ax, text: str, y=0.08, size=10.5):
    ax.text(0.5, y, text, ha="center", va="center", transform=ax.transAxes,
            fontsize=size, color="#333333", wrap=True)


def add_bullet_legend(ax, items, loc=(0.10, 0.06), line_h=0.065, box=0.035, fontsize=8.8):
    x, y = loc
    for i, (color, text) in enumerate(items):
        yy = y + i * line_h
        ax.add_patch(Rectangle((x, yy), box, box, transform=ax.transAxes, fc=color, ec="#888", lw=0.8))
        ax.text(x + box + 0.025, yy + box / 2, text, transform=ax.transAxes,
                ha="left", va="center", fontsize=fontsize, color="#333333")


def add_horizontal_legend(ax, items, y=0.05, box=0.04, start_x=0.24, gap=0.22, fontsize=10.0):
    x = start_x
    for color, text in items:
        ax.add_patch(Rectangle((x, y), box, box, transform=ax.transAxes, fc=color, ec="#888", lw=0.8))
        ax.text(x + box + 0.015, y + box / 2, text, transform=ax.transAxes,
                ha="left", va="center", fontsize=fontsize, color="#333333")
        x += gap


def pick_preview_mask_from_visual_folder(visual_dir: Path) -> Tuple[Path, Path]:
    """Pick the already-prepared preview image and binary mask from a good_live folder."""
    if not visual_dir.exists() or not visual_dir.is_dir():
        raise FileNotFoundError(f"Visual folder not found: {visual_dir}")

    files = [p for p in visual_dir.iterdir() if p.is_file()]
    if not files:
        raise FileNotFoundError(f"No files found in visual folder: {visual_dir}")

    def find_one(keywords, avoid=()):
        for p in sorted(files, key=lambda x: x.name.lower()):
            n = p.name.lower()
            if all(k in n for k in keywords) and not any(a in n for a in avoid):
                return p
        return None

    preview = (
        find_one(["overlay"])
        or find_one(["original"])
        or find_one(["overview"], avoid=["mask", "overlay", "prob", "error"])
        or find_one(["preview"])
    )
    mask = (
        find_one(["mask"], avoid=["valid", "label"])
        or find_one(["gt", "mask"])
        or find_one(["binary", "mask"])
    )

    if preview is None or mask is None:
        raise FileNotFoundError(
            f"Could not locate preview and mask files in {visual_dir}. "
            f"Found: {[p.name for p in files]}"
        )
    print(f"[VISUAL PREVIEW] {preview}")
    print(f"[VISUAL MASK]    {mask}")
    return preview, mask


def auto_pick_visual_folder(root: Path) -> Optional[Path]:
    if not root.exists() or not root.is_dir():
        return None
    # Direct folder itself may already contain preview/mask
    try:
        pick_preview_mask_from_visual_folder(root)
        return root
    except Exception:
        pass
    for sub in sorted(root.iterdir(), key=lambda x: x.name.lower()):
        if sub.is_dir():
            try:
                pick_preview_mask_from_visual_folder(sub)
                return sub
            except Exception:
                continue
    return None


def pick_wsi_pair(wsi_root: Path, mask_root: Path) -> Tuple[Path, Path]:
    """Pair PANDA WSI <id>.tiff with mask <id>_mask.tiff safely."""
    import re

    extensions = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
    wsis = sorted((p for p in wsi_root.iterdir()
                   if p.is_file() and p.suffix.lower() in extensions),
                  key=lambda x: x.name.lower())
    masks = sorted((p for p in mask_root.iterdir()
                    if p.is_file() and p.suffix.lower() in extensions),
                   key=lambda x: x.name.lower())
    if not wsis:
        raise FileNotFoundError(f"No WSI images found in: {wsi_root}")
    if not masks:
        raise FileNotFoundError(f"No label masks found in: {mask_root}")

    def wsi_id(name: str) -> str:
        return name.casefold()

    def mask_id(name: str) -> str:
        # PANDA official masks typically use <image_id>_mask.tiff.
        return re.sub(r"(?:_mask|_masks|_label_mask|_label|_gt|_annotation)$", "", name.casefold())

    by_id = {}
    for m in masks:
        by_id.setdefault(mask_id(m.stem), []).append(m)

    for wsi in wsis:
        matches = by_id.get(wsi_id(wsi.stem), [])
        if len(matches) == 1:
            print(f"[MATCH] WSI:  {wsi.name}")
            print(f"[MATCH] MASK: {matches[0].name}")
            return wsi, matches[0]
        if len(matches) > 1:
            raise RuntimeError(f"Ambiguous masks for {wsi.name}: {[p.name for p in matches]}")

    raise FileNotFoundError(
        "No WSI/mask ID matches found. Expected e.g. "
        "train_images\\abc.tiff and train_label_masks\\abc_mask.tiff.\n"
        f"Example WSIs: {[p.name for p in wsis[:4]]}\n"
        f"Example masks: {[p.name for p in masks[:4]]}"
    )


def select_illustration_sources(patch_root: Path, wsi_root: Path, mask_root: Path):
    """Select an actual tumor-bearing WSI plus independently sourced patch images.

    The plot is an illustrative overview, NOT an assertion that every patch comes
    from the depicted WSI. Records exact provenance in a sidecar manifest.
    """
    import re
    ext = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
    masks = {}
    for mp in mask_root.iterdir():
        if mp.is_file() and mp.suffix.lower() in ext:
            mid = re.sub(r"(?:_mask|_masks|_label_mask|_label|_gt|_annotation)$", "", mp.stem.lower())
            masks[mid] = mp

    pair = None
    for wsi in sorted(wsi_root.iterdir()):
        if not wsi.is_file() or wsi.suffix.lower() not in ext:
            continue
        mp = masks.get(wsi.stem.lower())
        if mp is None:
            continue
        try:
            mt = open_mask_thumbnail(mp, (500, 500))
            raw = np.asarray(mt.convert("L"))
            # PANDA label masks: nonzero pixel marks a labeled region.
            if np.count_nonzero(raw) > 100:
                pair = (wsi, mp)
                print(f"[WSI MATCH] {wsi.name} / {mp.name} (nonempty mask)")
                break
        except Exception as e:
            print(f"[SKIP MASK] {mp.name}: {e}")
    if pair is None:
        raise FileNotFoundError("No WSI with a readable, nonempty label mask was found. Check train_images and train_label_masks.")

    # Every panel gets a REAL patch image; sources may differ across rows/levels.
    wanted = [(lev, cls) for lev in ("L0", "L1", "L2") for cls in ("Normal", "Tumor")]
    found = {}
    for case in sorted(patch_root.iterdir()):
        if not case.is_dir():
            continue
        for lev, cls in wanted:
            key = (lev, cls)
            if key not in found:
                images = find_image_files(case / lev / cls)
                if images:
                    found[key] = images[0]
        if len(found) == len(wanted):
            break
    if len(found) < len(wanted):
        missing = [f"{lv}/{cl}" for lv, cl in wanted if (lv, cl) not in found]
        raise FileNotFoundError("Missing actual patch image(s) for: " + ", ".join(missing))
    for (lv, cl), fp in found.items():
        print(f"[PATCH {lv}/{cl}] {fp}")
    return pair[0], pair[1], found


def pick_patch_for_class(level_dir: Path, class_name: str) -> Image.Image:
    folder = level_dir / class_name
    files = find_image_files(folder)
    if not files:
        raise FileNotFoundError(f"No patch images in {folder}")
    return Image.open(files[0]).convert("RGB")


def open_selected_patch(sources: dict, level: str, class_name: str) -> Image.Image:
    path = sources[(level, class_name)]
    return Image.open(path).convert("RGB")


def build_patch_output_canvas(sources: dict, patch_size=(120, 100)) -> Image.Image:
    cols = ["L2", "L1", "L0"]
    label_h = 28
    margin = 14
    gap = 12
    total_w = margin * 2 + len(cols) * patch_size[0] + (len(cols) - 1) * gap
    total_h = 24 + label_h + 2 * patch_size[1] + 3 * gap + 12
    canvas = Image.new("RGB", (total_w, total_h), (255, 255, 255))

    # column headers
    import PIL.ImageDraw as ImageDraw
    import PIL.ImageFont as ImageFont
    draw = ImageDraw.Draw(canvas)
    try:
        font_head = ImageFont.truetype("arial.ttf", 18)
        font_small = ImageFont.truetype("arial.ttf", 16)
    except Exception:
        font_head = ImageFont.load_default()
        font_small = ImageFont.load_default()

    level_titles = {"L2": "L2 (16×)", "L1": "L1 (4×)", "L0": "L0 (1×)"}
    for i, lev in enumerate(cols):
        x = margin + i * (patch_size[0] + gap)
        title = level_titles[lev]
        bbox = draw.textbbox((0, 0), title, font=font_head)
        tw = bbox[2] - bbox[0]
        draw.rounded_rectangle([x + 20, 0, x + patch_size[0] - 20, 24], radius=4,
                               fill=(238, 238, 238), outline=(230, 230, 230))
        draw.text((x + (patch_size[0] - tw) / 2, 2), title, fill=(40, 40, 40), font=font_head)

    row_names = [("Normal (benign)", (230, 240, 225)), ("Tumor (cancer)", (245, 232, 223))]
    for r, (row_label, bg) in enumerate(row_names):
        draw.rounded_rectangle([0, 35 + r * (patch_size[1] + gap + 10), margin - 4,
                                35 + r * (patch_size[1] + gap + 10) + patch_size[1]],
                               radius=4, fill=bg, outline=bg)
        # put text in left outside? handled in axes panel instead

    for r, cls in enumerate(["Normal", "Tumor"]):
        for c, lev in enumerate(cols):
            img = open_selected_patch(sources, lev, cls)
            img = resize_keep(img, patch_size)
            x = margin + c * (patch_size[0] + gap)
            y = 35 + r * (patch_size[1] + gap + 10)
            border = (60, 180, 75) if cls == "Normal" else (255, 122, 0)
            ImageOps.expand(img, border=3, fill=border).save(os.devnull) if False else None
            canvas.paste(img, (x, y))
            draw.rectangle([x, y, x + patch_size[0], y + patch_size[1]], outline=border, width=3)
    return canvas


def select_patch_sources_only(patch_root: Path):
    """Find at least one real patch image for each level/class combination."""
    wanted = [(lev, cls) for lev in ("L0", "L1", "L2") for cls in ("Normal", "Tumor")]
    found = {}
    for case in sorted(patch_root.iterdir(), key=lambda x: x.name.lower()):
        if not case.is_dir():
            continue
        for lev, cls in wanted:
            key = (lev, cls)
            if key in found:
                continue
            images = find_image_files(case / lev / cls)
            if images:
                found[key] = images[0]
        if len(found) == len(wanted):
            break
    if len(found) < len(wanted):
        missing = [f"{lv}/{cl}" for lv, cl in wanted if (lv, cl) not in found]
        raise FileNotFoundError("Missing actual patch image(s) for: " + ", ".join(missing))
    for (lv, cl), fp in found.items():
        print(f"[PATCH {lv}/{cl}] {fp}")
    return found


def build_sampling_diagram_image() -> Image.Image:
    import PIL.ImageDraw as ImageDraw
    import PIL.ImageFont as ImageFont
    W, H = 700, 380
    img = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    try:
        font_title = ImageFont.truetype("arial.ttf", 24)
        font_main = ImageFont.truetype("arial.ttf", 20)
        font_small = ImageFont.truetype("arial.ttf", 18)
    except Exception:
        font_title = ImageFont.load_default()
        font_main = ImageFont.load_default()
        font_small = ImageFont.load_default()

    draw.text((W // 2 - 140, 10), "12 buckets: Provider × Level × Class", fill=(50, 50, 50), font=font_main)

    # small matrix-like diagram
    x0 = 170
    y0 = 80
    col_w = 140
    row_h = 95
    cols = ["L2\n(16×)", "L1\n(4×)", "L0\n(1×)"]
    for i, name in enumerate(cols):
        x = x0 + i * 160
        draw.rounded_rectangle([x, y0, x + col_w, y0 + 50], radius=6, fill=(239, 239, 239), outline=(220, 220, 220))
        # crude center text
        for k, line in enumerate(name.split("\n")):
            bbox = draw.textbbox((0, 0), line, font=font_small)
            tw = bbox[2] - bbox[0]
            draw.text((x + (col_w - tw) / 2, y0 + 6 + 20 * k), line, fill=(30, 30, 30), font=font_small)

    # row labels
    draw.rounded_rectangle([20, 130, 150, 215], radius=6, fill=(232, 240, 227), outline=(232, 240, 227))
    draw.multiline_text((45, 150), "Normal\n(benign)", fill=(35, 35, 35), font=font_main, align="center")
    draw.rounded_rectangle([20, 225, 150, 310], radius=6, fill=(247, 236, 229), outline=(247, 236, 229))
    draw.multiline_text((45, 245), "Tumor\n(cancer)", fill=(35, 35, 35), font=font_main, align="center")

    # cells and provider dots
    for j, cls_color in enumerate([(60, 180, 75), (255, 122, 0)]):
        y = 145 + j * 95
        for i in range(3):
            x = x0 + i * 160
            draw.rounded_rectangle([x, y, x + col_w, y + 70], radius=6, fill=(255, 255, 255), outline=cls_color, width=2)
            draw.text((x + 18, y + 8), "P0", fill=(45, 45, 45), font=font_main)
            draw.text((x + 82, y + 8), "P1", fill=(45, 45, 45), font=font_main)
            for dx in [36, 100]:
                draw.ellipse([x + dx - 5, y + 44 - 5, x + dx + 5, y + 44 + 5], fill=cls_color, outline=cls_color)

    draw.multiline_text((70, 325), "Sample patches from both PANDA providers\n(P0, P1) at three pyramid levels for normal\nand tumor classes (12 buckets in total)",
                        fill=(40, 40, 40), font=font_small, align="center")
    return img


def build_patch_pipeline_figure(wsi_root: Path, mask_root: Path, patch_root: Path, out_dir: Path,
                                visual_dir: Optional[Path] = None,
                                direct_preview: Optional[Path] = None,
                                direct_mask: Optional[Path] = None):
    patch_sources = select_patch_sources_only(patch_root)

    if direct_preview is not None and direct_mask is not None:
        print(f"[FORCED PREVIEW FILE] {direct_preview}")
        print(f"[FORCED MASK FILE]    {direct_mask}")
        wsi_path, mask_path = direct_preview, direct_mask
    elif visual_dir is not None:
        wsi_path, mask_path = pick_preview_mask_from_visual_folder(visual_dir)
    else:
        wsi_path, mask_path = pick_wsi_pair(wsi_root, mask_root)

    wsi_thumb = open_slide_thumbnail(wsi_path, size=(1100, 1100))
    mask_thumb = open_mask_thumbnail(mask_path, size=(1100, 1100))

    wsi_arr = np.asarray(wsi_thumb.convert("RGB"))
    mask_arr = np.asarray(mask_thumb.convert("L"))
    tumor_mask = mask_arr > 0

    using_direct_visual = (direct_preview is not None and direct_mask is not None) or (visual_dir is not None)
    if using_direct_visual:
        tissue_mask = compute_preview_foreground_mask(wsi_arr)
        keep_mask = tissue_mask.copy()
        focus_union = tissue_mask | tumor_mask
        focus_bbox = mask_bbox(focus_union, pad_ratio=0.05)
    else:
        tissue_mask = compute_tissue_mask(wsi_arr)
        keep_mask = compute_quality_keep_mask(wsi_arr, tissue_mask)
        focus_bbox = mask_bbox(tissue_mask | tumor_mask, pad_ratio=0.10)
    wsi_focus = crop_to_bbox(wsi_thumb, focus_bbox)
    tissue_mask_focus = crop_to_bbox(Image.fromarray((tissue_mask.astype(np.uint8) * 255)), focus_bbox)
    keep_mask_focus = crop_to_bbox(Image.fromarray((keep_mask.astype(np.uint8) * 255)), focus_bbox)
    tumor_mask_focus = crop_to_bbox(Image.fromarray((tumor_mask.astype(np.uint8) * 255)), focus_bbox)

    wsi_focus_arr = np.asarray(wsi_focus.convert("RGB"))
    tissue_mask2 = np.asarray(tissue_mask_focus.convert("L")) > 0
    keep_mask2 = np.asarray(keep_mask_focus.convert("L")) > 0
    tumor_mask2 = np.asarray(tumor_mask_focus.convert("L")) > 0
    if using_direct_visual:
        tissue_mask2 = compute_preview_foreground_mask(wsi_focus_arr) | tumor_mask2
        keep_mask2 = tissue_mask2.copy()

    cells = draw_grid_info(tissue_mask2, n_cols=14, n_rows=24)
    cell_info = classify_patch_cells(tissue_mask2, tumor_mask2, cells)

    # Choose a sample patch for step 4/8 from L0 Tumor
    sample_patch = open_selected_patch(patch_sources, "L0", "Tumor")
    sample_patch = resize_keep(sample_patch, (260, 260))
    normalized_patch = normalize_patch_lab_like(sample_patch)

    # Build figure layout
    fig = plt.figure(figsize=(18.6, 12.2), facecolor="white")
    gs = fig.add_gridspec(2, 7, height_ratios=[1.15, 0.88], hspace=0.18, wspace=0.04)

    axes_top = [fig.add_subplot(gs[0, i]) for i in range(7)]
    ax8 = fig.add_subplot(gs[1, 0:2])
    ax9 = fig.add_subplot(gs[1, 2:4])
    ax10 = fig.add_subplot(gs[1, 4:7])

    # Step 1
    ax = axes_top[0]
    add_panel_chrome(ax, 1, "Whole-slide\nbiopsy image")
    show_centered_image(ax, resize_keep(crop_to_content(wsi_focus), (300, 420)), (0.18, 0.18, 0.64, 0.68))
    put_caption(ax, "PANDA prostate\ncore biopsy (WSI)", y=0.10, size=11)

    # Step 2
    ax = axes_top[1]
    add_panel_chrome(ax, 2, "Tissue mask\ngeneration")
    tissue_vis = Image.fromarray(make_colored_mask(tissue_mask2, fg=(11, 130, 42), bg=(255, 255, 255)))
    show_centered_image(ax, resize_keep(tissue_vis, (300, 420)), (0.18, 0.18, 0.64, 0.68))
    put_caption(ax, "Retain tissue regions\n(remove background)", y=0.10, size=11)

    # Step 3
    ax = axes_top[2]
    add_panel_chrome(ax, 3, "Background /\nquality detection")
    quality_vis = Image.fromarray(make_keep_remove_visual(keep_mask2, tissue_mask2))
    show_centered_image(ax, resize_keep(quality_vis, (300, 420)), (0.16, 0.22, 0.68, 0.60))
    add_bullet_legend(ax, [((219/255, 188/255, 1.0), "Keep (tissue)"), ((0.85, 0.85, 0.85), "Remove (background/\nlow quality)")], loc=(0.17, 0.06), line_h=0.08, box=0.05)

    # Step 4
    ax = axes_top[3]
    add_panel_chrome(ax, 4, "Cartesian patch\nextraction")
    iax = show_centered_image(ax, resize_keep(crop_to_content(wsi_focus), (330, 420)), (0.10, 0.20, 0.58, 0.64))
    # draw grid on tissue area
    for info in cell_info:
        x, y, w, h = info["rect"]
        if info["tissue_ratio"] >= 0.15:
            iax.add_patch(Rectangle((x, y), w, h, fill=False, ec="#2F80ED", lw=1.0, alpha=0.9))
    # inset example patch
    inset = ax.inset_axes([0.62, 0.43, 0.27, 0.22])
    inset.imshow(sample_patch)
    inset.set_xticks([])
    inset.set_yticks([])
    for s in inset.spines.values():
        s.set_color("#2F80ED")
        s.set_linewidth(2)
    ax.annotate("", xy=(0.60, 0.53), xycoords=ax.transAxes, xytext=(0.47, 0.50), textcoords=ax.transAxes,
                arrowprops=dict(arrowstyle="-", lw=1.2, color="#333", ls="--"))
    ax.annotate("", xy=(0.60, 0.55), xycoords=ax.transAxes, xytext=(0.47, 0.62), textcoords=ax.transAxes,
                arrowprops=dict(arrowstyle="-", lw=1.2, color="#333", ls="--"))
    ax.text(0.75, 0.36, "512 × 512\npixels\nstride 128", transform=ax.transAxes,
            ha="center", va="center", fontsize=11, color="#333333")
    put_caption(ax, "Extract 512 × 512\npatches on a regular grid\n(stride 128)", y=0.08, size=10.8)

    # Step 5
    ax = axes_top[4]
    add_panel_chrome(ax, 5, "Patch\nfiltering")
    iax = show_centered_image(ax, resize_keep(crop_to_content(wsi_focus), (300, 420)), (0.18, 0.21, 0.64, 0.62))
    for info in cell_info:
        x, y, w, h = info["rect"]
        if info["label"] == "discard":
            iax.add_patch(Rectangle((x, y), w, h, fill=True, fc=(0.85, 0.85, 0.85, 0.35), ec="#999999", lw=0.5))
        else:
            iax.add_patch(Rectangle((x, y), w, h, fill=False, ec="#2AA65A", lw=1.0))
    add_bullet_legend(ax, [((0.38, 0.84, 0.47), "Keep (sufficient tissue)"), ((0.86, 0.86, 0.86), "Discard (blank/low quality/\nartifact)")],
                      loc=(0.11, 0.06), line_h=0.07, box=0.05)

    # Step 6
    ax = axes_top[5]
    add_panel_chrome(ax, 6, "Binary tumor\nmask")
    tumor_vis = Image.fromarray(make_colored_mask(tumor_mask2, fg=(255, 255, 255), bg=(0, 0, 0)))
    show_centered_image(ax, resize_keep(tumor_vis, (300, 420)), (0.16, 0.20, 0.68, 0.66))
    put_caption(ax, "Convert PANDA\nlabel mask to binary\ntumor mask", y=0.10, size=11)

    # Step 7
    ax = axes_top[6]
    add_panel_chrome(ax, 7, "Patch label\nassignment")
    iax = show_centered_image(ax, resize_keep(crop_to_content(wsi_focus), (300, 420)), (0.16, 0.18, 0.64, 0.68))
    colors = {
        "normal": "#5AD66A",
        "borderline": "#BFBFBF",
        "tumor": "#FF7A00",
    }
    for info in cell_info:
        x, y, w, h = info["rect"]
        lab = info["label"]
        if lab == "discard":
            continue
        iax.add_patch(Rectangle((x, y), w, h, fill=True, fc=colors[lab], ec="#444444", lw=0.2, alpha=0.65))
    add_bullet_legend(ax, [((0.35, 0.85, 0.42), "Normal patch\n(tumor area = 0%)"),
                           ((0.80, 0.80, 0.80), "Exclude borderline patch\n(0% < tumor area < 5%)"),
                           ((1.0, 0.48, 0.0), "Tumor patch\n(tumor area ≥ 5%)")],
                      loc=(0.12, 0.01), line_h=0.07, box=0.05)

    # Step 8
    add_panel_chrome(ax8, 8, "Vahadane normalization")
    # labels
    ax8.text(0.27, 0.73, "Before", transform=ax8.transAxes, fontsize=13, fontweight="bold", ha="center")
    ax8.text(0.73, 0.73, "After", transform=ax8.transAxes, fontsize=13, fontweight="bold", ha="center")
    iax1 = ax8.inset_axes([0.05, 0.24, 0.38, 0.42])
    iax2 = ax8.inset_axes([0.57, 0.24, 0.38, 0.42])
    iax1.imshow(sample_patch); iax2.imshow(normalized_patch)
    iax1.axis("off"); iax2.axis("off")
    ax8.annotate("", xy=(0.55, 0.45), xycoords=ax8.transAxes, xytext=(0.44, 0.45), textcoords=ax8.transAxes,
                 arrowprops=dict(arrowstyle="-|>", lw=1.8, color="#333333"))
    put_caption(ax8, "Normalize stain using Vahadane method\nto reduce inter-slide color variation", y=0.10, size=11)

    # Step 9
    add_panel_chrome(ax9, 9, "Structured multilevel sampling")
    diagram9 = build_sampling_diagram_image()
    show_centered_image(ax9, diagram9, (0.06, 0.11, 0.88, 0.75))

    # Step 10
    add_panel_chrome(ax10, 10, "PANDA patch outputs")
    patch_canvas = build_patch_output_canvas(patch_sources)
    show_centered_image(ax10, patch_canvas, (0.12, 0.15, 0.78, 0.72))
    ax10.text(0.06, 0.57, "Normal\n(benign)", transform=ax10.transAxes, ha="center", va="center", fontsize=12, fontweight="bold", color="#333")
    ax10.text(0.06, 0.29, "Tumor\n(cancer)", transform=ax10.transAxes, ha="center", va="center", fontsize=12, fontweight="bold", color="#333")
    add_bullet_legend(ax10, [((0.35, 0.85, 0.42), "Normal patch"), ((1.0, 0.48, 0.0), "Tumor patch")],
                      loc=(0.30, 0.02), line_h=0.001, box=0.05)
    ax10.text(0.5, 0.02, "Note: Borderline patches (0% < tumor area < 5%) are excluded\nfrom the final dataset.",
              transform=ax10.transAxes, ha="center", va="bottom", fontsize=10.5, color="#333333")

    # Figure title
    fig.suptitle("PANDA Patch Preparation Pipeline", fontsize=30, fontweight="bold", y=0.98, family="serif")

    # arrows between top panels
    def add_arrow_between(ax_from, ax_to):
        b1, b2 = ax_from.get_position(), ax_to.get_position()
        x1 = b1.x1 - 0.002
        x2 = b2.x0 + 0.002
        y = (b1.y0 + b1.y1) / 2
        arr = FancyArrowPatch((x1, y), (x2, y), transform=fig.transFigure,
                              arrowstyle="-|>", mutation_scale=16, lw=1.8, color="black")
        fig.add_artist(arr)

    for i in range(6):
        add_arrow_between(axes_top[i], axes_top[i + 1])

    # dashed connector from step 7 to the lower row region
    p7 = axes_top[6].get_position()
    p8 = ax8.get_position(); p9 = ax9.get_position(); p10 = ax10.get_position()
    y_dash = p8.y1 + 0.01
    x_left = p8.x0 + 0.01
    x_right = p10.x1 - 0.01
    fig.lines.append(plt.Line2D([x_left, x_right], [y_dash, y_dash], transform=fig.transFigure, ls="--", lw=1.5, color="#333333"))
    fig.lines.append(plt.Line2D([p7.x0 + p7.width/2, p7.x0 + p7.width/2], [p7.y0, y_dash], transform=fig.transFigure, ls="--", lw=1.5, color="#333333"))
    fig.lines.append(plt.Line2D([p8.x0 + p8.width/2, p8.x0 + p8.width/2], [y_dash, p8.y1], transform=fig.transFigure, ls="--", lw=1.5, color="#333333"))
    fig.lines.append(plt.Line2D([p9.x0 + p9.width/2, p9.x0 + p9.width/2], [y_dash, p9.y1], transform=fig.transFigure, ls="--", lw=1.5, color="#333333"))
    fig.lines.append(plt.Line2D([p10.x0 + p10.width/2, p10.x0 + p10.width/2], [y_dash, p10.y1], transform=fig.transFigure, ls="--", lw=1.5, color="#333333"))

    out_png = out_dir / "Fig_panda_patch_preparation_pipeline.png"
    out_pdf = out_dir / "Fig_panda_patch_preparation_pipeline.pdf"
    out_tif = out_dir / "Fig_panda_patch_preparation_pipeline.tif"
    fig.savefig(out_png, dpi=400, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=400, bbox_inches="tight")
    fig.savefig(out_tif, dpi=400, bbox_inches="tight")
    plt.close(fig)
    import json
    manifest = out_dir / "Fig_panda_patch_preparation_pipeline_sources.json"
    manifest.write_text(json.dumps({
        "warning": "Illustrative workflow. WSI/preview source may differ from patch sources; grid filtering and stain panel are visual illustrations.",
        "wsi_or_preview": str(wsi_path), "label_mask_or_binary_mask": str(mask_path),
        "patches": {f"{lvl}/{cls}": str(path) for (lvl, cls), path in patch_sources.items()}
    }, indent=2), encoding="utf-8")
    print(f"[SOURCES] {manifest}")
    return out_png, out_pdf, out_tif, wsi_path, mask_path, patch_sources


def main():
    patch_root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(DEFAULT_PATCH_ROOT)
    wsi_root = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(DEFAULT_WSI_ROOT)
    mask_root = Path(sys.argv[3]) if len(sys.argv) > 3 else Path(DEFAULT_MASK_ROOT)

    direct_preview = None
    direct_mask = None
    visual_dir = None
    if len(sys.argv) > 5:
        direct_preview = Path(sys.argv[4])
        direct_mask = Path(sys.argv[5])
    elif len(sys.argv) > 4:
        arg4 = Path(sys.argv[4])
        if arg4.is_dir():
            visual_dir = arg4
        elif arg4.is_file() and len(sys.argv) > 5:
            direct_preview = arg4
            direct_mask = Path(sys.argv[5])
        else:
            visual_dir = arg4

    script_dir = Path(__file__).resolve().parent
    out_dir = script_dir
    safe_mkdir(out_dir)

    if not patch_root.exists():
        raise FileNotFoundError(f"Patch root not found: {patch_root}")
    if not wsi_root.exists():
        print(f"[WARN] WSI root not found: {wsi_root}")
    if not mask_root.exists():
        print(f"[WARN] Mask root not found: {mask_root}")

    if direct_preview is not None or direct_mask is not None:
        if direct_preview is None or direct_mask is None:
            raise ValueError("When passing direct image inputs, provide BOTH preview_image_path and mask_image_path.")
        if not direct_preview.exists():
            raise FileNotFoundError(f"Direct preview image not found: {direct_preview}")
        if not direct_mask.exists():
            raise FileNotFoundError(f"Direct mask image not found: {direct_mask}")
        print(f"[USING DIRECT PREVIEW] {direct_preview}")
        print(f"[USING DIRECT MASK]    {direct_mask}")
    else:
        if visual_dir is None:
            visual_dir = auto_pick_visual_folder(Path(DEFAULT_VISUAL_ROOT))
        if visual_dir is not None:
            print(f"[USING VISUAL FOLDER] {visual_dir}")
        else:
            print("[INFO] No direct visual folder found/provided; falling back to WSI + mask pairing.")

    out_png, out_pdf, out_tif, wsi_path, mask_path, patch_sources = build_patch_pipeline_figure(
        wsi_root, mask_root, patch_root, out_dir, visual_dir=visual_dir,
        direct_preview=direct_preview, direct_mask=direct_mask
    )

    print("Generated:")
    print(out_png)
    print(out_pdf)
    print(out_tif)
    print(f"Preview/WSI used: {wsi_path}")
    print(f"Mask used:        {mask_path}")
    print(f"Patch images used: {len(patch_sources)} (see sources JSON)")


if __name__ == "__main__":
    main()
