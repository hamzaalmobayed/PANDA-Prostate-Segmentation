from __future__ import annotations

"""
PANDA prostate cancer segmentation — CAMELYON-style multilevel pipeline
=======================================================================

Goal agreed in chat
-------------------
- One binary segmentation model (tumor vs non-tumor) trained jointly on 4 logical scales.
- 8 training sampling buckets = 4 levels x {normal, tumor}.
- EfficientNet-B4 feature extraction + KMeans diversity selection inside each bucket.
- Equal sampling from the 8 buckets.
- CAMELYON-style settings: patch=512, stride=128, levels=1..4, Vahadane,
  EfficientNet-B4 U-Net, coordinate injection, BCEWithLogits + Dice, AdamW,
  batch=8, AMP/TF32/channels-last, strong paired augmentation.
- Split whole slides BEFORE patch extraction/sampling to prevent data leakage.
- Final test is reported separately for L1/L2/L3/L4 and pooled across all levels.

QC / self-repair philosophy
---------------------------
Each stage writes a PASS/FAIL JSON report. Safe, deterministic repairs are attempted
when possible (e.g. mask filename matching, virtual scale fallback when native levels
are missing, tissue-mask fallback, DataLoader worker fallback). The code NEVER invents
unknown label semantics or silently repairs scientifically ambiguous image-mask alignment;
it stops and tells you exactly what needs attention.

Expected PANDA layout
---------------------
D:/PANDA/
  train.csv
  train_images/<image_id>.tiff
  train_label_masks/<image_id>_mask.tiff

PANDA provider masks are converted to a common binary target:
- Radboud: 0 background/unknown, 1 stroma, 2 benign epithelium, 3/4/5 tumor
- Karolinska: 0 background/unknown, 1 benign, 2 tumor
The model target is binary: 1=tumor, 0=non-tumor/background.

Usage
-----
Edit Config paths near the top OR pass --data-root/--work-root.
Run stages in order:
  python panda_camelyon_style_pipeline.py audit
  python panda_camelyon_style_pipeline.py split
  python panda_camelyon_style_pipeline.py qc-alignment
  python panda_camelyon_style_pipeline.py extract
  python panda_camelyon_style_pipeline.py select
  python panda_camelyon_style_pipeline.py qc-normalization
  python panda_camelyon_style_pipeline.py qc-dataloader
  python panda_camelyon_style_pipeline.py sanity
  python panda_camelyon_style_pipeline.py train
  python panda_camelyon_style_pipeline.py test

Or:
  python panda_camelyon_style_pipeline.py all
"""

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import random
import shutil
import sys
import time
import warnings
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from torchvision import models
from torchvision.transforms import functional as TF

try:
    import openslide
except Exception as e:
    openslide = None
    _OPENSLIDE_IMPORT_ERROR = e
else:
    _OPENSLIDE_IMPORT_ERROR = None

try:
    import timm
except Exception as e:
    timm = None
    _TIMM_IMPORT_ERROR = e
else:
    _TIMM_IMPORT_ERROR = None

try:
    from tiatoolbox.tools.stainnorm import VahadaneNormalizer
except Exception as e:
    VahadaneNormalizer = None
    _TIA_IMPORT_ERROR = e
else:
    _TIA_IMPORT_ERROR = None

warnings.filterwarnings("ignore", category=UserWarning)
Image.MAX_IMAGE_PIXELS = None


# =============================================================================
# 0) CONFIG — inherited from CAMELYON code wherever possible
# =============================================================================

@dataclass
class Config:
    # Paths
    data_root: str = r"D:\PANDA"
    work_root: str = r"D:\PANDA_PROSTATE"

    # CAMELYON data settings
    levels: Tuple[int, ...] = (1, 2, 3, 4)
    patch_size: int = 512
    stride: int = 128
    tissue_level: int = 5
    tissue_min_frac: float = 0.02
    tissue_default_threshold: float = 0.20
    morph_disk_size: int = 2
    min_object_size: int = 500
    tumor_threshold: float = 0.05
    target_max_size: int = 4096

    # Important: PANDA often exposes fewer native TIFF pyramid levels than CAMELYON.
    # To preserve the 4-scale CAMELYON design, logical L1..L4 are target downsamples
    # relative to level 0. Exact native levels are used when available; otherwise the
    # nearest native level is read then resized (safe virtual-level repair).
    logical_downsamples: Dict[int, float] = field(
        default_factory=lambda: {1: 2.0, 2: 4.0, 3: 8.0, 4: 16.0, 5: 32.0}
    )

    # Split — strict slide-level separation
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    seed: int = 42

    # Balanced training selection (PANDA has only one dataset source, so 8 buckets)
    num_samples: int = 80000
    max_k_for_elbow: int = 12
    feat_batch_size: int = 32
    feat_img_size: int = 380
    feature_candidate_cap_per_bucket: int = 30000

    # Extraction safety/performance
    max_candidate_patches_per_slide_level: int = 50000
    save_png_compress: int = 3
    extraction_resume: bool = True

    # CAMELYON strong augmentation settings
    use_strong_stain_aug: bool = True
    aug_p_hflip: float = 0.5
    aug_p_vflip: float = 0.2
    aug_p_rot90: float = 0.5
    aug_p_color: float = 0.8
    aug_p_gamma: float = 0.4
    aug_p_blur: float = 0.25
    aug_p_noise: float = 0.25
    aug_p_jpeg: float = 0.25

    # Vahadane
    normalization_method: str = "Vahadane"
    vahadane_cache_max: int = 2048
    persistent_stain_cache: bool = True
    stain_cache_png_compress: int = 1
    prebuild_stain_cache_before_training: bool = True
    stain_cache_build_workers: int = 4

    # Best CAMELYON model settings
    pretrained: bool = True
    use_coords: bool = True
    coord_embed_dim: int = 32
    dropout_p: float = 0.1
    n_classes: int = 1

    # Training defaults as in the CAMELYON code block
    batch_size: int = 8
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    # Final model: allow enough room to converge; stop on validation Dice if it plateaus.
    num_epochs: int = 30
    early_stopping_patience: int = 7
    amp: bool = True
    channels_last: bool = True
    num_workers: int = 4
    prefetch_factor: int = 4
    persistent_workers: bool = True
    threshold: float = 0.50

    # QC
    qc_slides_per_provider: int = 8
    qc_patches_per_stage: int = 24
    alignment_min_overlap: float = 0.20
    sanity_subset: int = 32
    sanity_epochs: int = 20
    sanity_required_dice: float = 0.80

    @property
    def train_csv(self) -> Path:
        return Path(self.data_root) / "train.csv"

    @property
    def images_dir(self) -> Path:
        return Path(self.data_root) / "train_images"

    @property
    def masks_dir(self) -> Path:
        return Path(self.data_root) / "train_label_masks"

    @property
    def work(self) -> Path:
        return Path(self.work_root)

    @property
    def qc_dir(self) -> Path:
        return self.work / "qc"

    @property
    def manifests_dir(self) -> Path:
        return self.work / "manifests"

    @property
    def splits_dir(self) -> Path:
        return self.work / "splits"

    @property
    def patches_dir(self) -> Path:
        return self.work / "patches"

    @property
    def features_dir(self) -> Path:
        return self.work / "features"

    @property
    def selected_dir(self) -> Path:
        return self.work / "selected"

    @property
    def runs_dir(self) -> Path:
        return self.work / "runs"

    @property
    def stain_target_path(self) -> Path:
        return self.work / "stain_target_train.png"

    @property
    def audit_csv(self) -> Path:
        return self.manifests_dir / "audit.csv"

    @property
    def split_csv(self) -> Path:
        return self.splits_dir / "slides.csv"

    @property
    def patch_manifest(self) -> Path:
        return self.manifests_dir / "patches.csv"

    @property
    def selected_train_csv(self) -> Path:
        return self.selected_dir / "train_selected.csv"

    def ensure_dirs(self):
        for p in [self.work, self.qc_dir, self.manifests_dir, self.splits_dir,
                  self.patches_dir, self.features_dir, self.selected_dir, self.runs_dir]:
            p.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Utilities / QC gates
# =============================================================================

class QCError(RuntimeError):
    pass


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_torch_speed():
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass




def fast_worker_init_fn(worker_id: int):
    """Keep Windows DataLoader workers from oversubscribing CPU threads.

    This is a performance-only setting: it does not change samples, augmentation,
    model architecture, optimizer, loss, batch size, or training schedule.
    """
    try:
        torch.set_num_threads(1)
    except Exception:
        pass
    try:
        cv2.setNumThreads(0)
    except Exception:
        pass

def atomic_json(path: Path, obj: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def qc_report(cfg: Config, step: str, passed: bool, details: Dict[str, Any], repairs: Optional[List[str]] = None):
    payload = {
        "step": step,
        "passed": bool(passed),
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "repairs": repairs or [],
        "details": details,
    }
    atomic_json(cfg.qc_dir / f"{step}.json", payload)
    icon = "✅ PASS" if passed else "❌ FAIL"
    print(f"\n[{step}] {icon}")
    for r in repairs or []:
        print(f"  🔧 auto-repair: {r}")
    if not passed:
        raise QCError(f"QC step '{step}' failed. See {cfg.qc_dir / (step + '.json')}")


def require_packages():
    missing = []
    if openslide is None:
        missing.append(f"openslide-python ({_OPENSLIDE_IMPORT_ERROR})")
    if timm is None:
        missing.append(f"timm ({_TIMM_IMPORT_ERROR})")
    if VahadaneNormalizer is None:
        missing.append(f"tiatoolbox ({_TIA_IMPORT_ERROR})")
    if missing:
        raise RuntimeError("Missing required packages: " + "; ".join(missing))


def sha1_text(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]




def _stain_cache_namespace(cfg: Config, normalization: str) -> Path:
    """Stable, lossless disk cache for stain-normalized RGB patches.

    The namespace includes the stain-target identity so changing the target cannot
    silently reuse stale normalized pixels. PNG is lossless, so cached pixels are
    scientifically identical to the normalization output used on first build.
    """
    target = Path(cfg.stain_target_path)
    try:
        st = target.stat()
        target_sig = sha1_text(f"{target.resolve()}|{st.st_size}|{st.st_mtime_ns}")
    except Exception:
        target_sig = sha1_text(str(target))
    return Path(cfg.work_root) / "cache" / "stain_v10" / normalization.lower() / target_sig


def stain_cache_path(cfg: Config, img_path: str | Path, normalization: str) -> Path:
    src = str(Path(str(img_path)))
    return _stain_cache_namespace(cfg, normalization) / f"{sha1_text(src)}.png"


def load_or_build_stain_cache(cfg: Config, img_path: str | Path, normalization: str, transform_fn) -> Image.Image:
    """Read a lossless normalized patch from disk or build it exactly once.

    Safe for Windows DataLoader workers: writes use a PID-specific temporary file
    and atomic os.replace. If cache writing itself is temporarily impossible, the
    already-normalized in-memory image is returned and training can continue; a
    later access will rebuild the missing cache entry.
    """
    src = Path(str(img_path))
    cp = stain_cache_path(cfg, src, normalization)
    if bool(getattr(cfg, "persistent_stain_cache", True)) and cp.exists():
        try:
            with Image.open(cp) as im:
                out = im.convert("RGB").copy()
            if out.size == (int(cfg.patch_size), int(cfg.patch_size)):
                return out
            try: cp.unlink()
            except Exception: pass
        except Exception:
            try: cp.unlink()
            except Exception: pass

    with Image.open(src) as im:
        rgb = np.array(im.convert("RGB"), dtype=np.uint8, copy=True, order="C")
    arr = np.array(transform_fn(rgb), dtype=np.uint8, copy=True, order="C")
    out = Image.fromarray(arr, mode="RGB")

    if bool(getattr(cfg, "persistent_stain_cache", True)):
        tmp = None
        try:
            cp.parent.mkdir(parents=True, exist_ok=True)
            tmp = cp.with_name(cp.name + f".{os.getpid()}.tmp.png")
            out.save(tmp, format="PNG", compress_level=int(getattr(cfg, "stain_cache_png_compress", 1)))
            os.replace(tmp, cp)
        except Exception:
            if tmp is not None:
                try: tmp.unlink()
                except Exception: pass
    return out


def stain_cache_ready(cfg: Config, img_path: str | Path, normalization: str) -> bool:
    """Cheap cache-hit test used by the hot training path.

    It intentionally avoids opening the original RGB patch when a normalized
    cache file is already present. This removes one unnecessary image read per
    sample from every epoch.
    """
    cp = stain_cache_path(cfg, img_path, normalization)
    try:
        return cp.exists() and cp.stat().st_size > 0
    except OSError:
        return False


def load_stain_cache_only(cfg: Config, img_path: str | Path, normalization: str) -> Image.Image | None:
    """Return a cached normalized image, or None if it is absent/corrupt."""
    cp = stain_cache_path(cfg, img_path, normalization)
    try:
        if not cp.exists() or cp.stat().st_size <= 0:
            return None
        with Image.open(cp) as im:
            out = im.convert("RGB").copy()
        if out.size == (int(cfg.patch_size), int(cfg.patch_size)):
            return out
    except Exception:
        pass
    try:
        cp.unlink()
    except Exception:
        pass
    return None

def safe_rel(path: str | Path, root: str | Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(Path(root).resolve())).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")


# =============================================================================
# PANDA mask semantics
# =============================================================================

EXPECTED_MASK_VALUES = {
    "radboud": {0, 1, 2, 3, 4, 5},
    "karolinska": {0, 1, 2},
}


def canonical_provider(x: str) -> str:
    s = str(x).strip().lower()
    if "radboud" in s:
        return "radboud"
    if "karolinska" in s:
        return "karolinska"
    return s


def raw_to_binary_mask(raw: np.ndarray, provider: str) -> np.ndarray:
    """Convert PANDA raw mask labels to binary tumor mask (uint8 0/255)."""
    provider = canonical_provider(provider)
    if raw.ndim == 3:
        raw = raw[..., 0]
    raw = raw.astype(np.uint8, copy=False)
    unique = set(np.unique(raw).tolist())
    expected = EXPECTED_MASK_VALUES.get(provider)
    if expected is None:
        raise QCError(f"Unknown provider '{provider}'. Refusing to invent label semantics.")
    unexpected = unique - expected
    if unexpected:
        raise QCError(f"Unexpected raw mask values for {provider}: {sorted(unexpected)}; expected subset {sorted(expected)}")
    if provider == "radboud":
        tumor = np.isin(raw, [3, 4, 5])
    else:  # karolinska
        tumor = raw == 2
    return (tumor.astype(np.uint8) * 255)


def raw_valid_tissue_mask(raw: np.ndarray, provider: str) -> np.ndarray:
    if raw.ndim == 3:
        raw = raw[..., 0]
    # Both providers use 0 for background/unknown.
    return (raw > 0).astype(np.uint8)


# =============================================================================
# WSI scale reader — native levels + safe virtual-scale repair
# =============================================================================

class SlideScaleReader:
    """
    Read a patch at a target downsample relative to level 0.
    If target downsample exactly matches a native OpenSlide level, use it.
    Otherwise read from the closest *higher-resolution* native level and resize.
    This safely preserves the four CAMELYON logical scales on PANDA slides that expose
    fewer native pyramid levels.
    """
    def __init__(self, path: str | Path):
        if openslide is None:
            raise RuntimeError("openslide-python is required")
        self.path = str(path)
        self.slide = openslide.OpenSlide(self.path)
        self.level_downsamples = [float(x) for x in self.slide.level_downsamples]
        self.level_dimensions = [tuple(map(int, x)) for x in self.slide.level_dimensions]
        self.dimensions = tuple(map(int, self.slide.dimensions))

    def close(self):
        try:
            self.slide.close()
        except Exception:
            pass

    def choose_native_level(self, target_ds: float) -> Tuple[int, float, bool]:
        arr = np.array(self.level_downsamples, dtype=float)
        exact_idx = int(np.argmin(np.abs(np.log(np.maximum(arr, 1e-6) / target_ds))))
        exact = abs(arr[exact_idx] - target_ds) / max(target_ds, 1e-6) <= 0.05
        if exact:
            return exact_idx, arr[exact_idx], False
        # Prefer a native level with ds <= target_ds (higher/equal resolution), then resize down.
        candidates = np.where(arr <= target_ds)[0]
        if len(candidates):
            idx = int(candidates[-1])
        else:
            idx = 0
        return idx, arr[idx], True

    def read_at_downsample(self, x0: int, y0: int, target_ds: float, out_size: int,
                           resample=Image.Resampling.BILINEAR) -> Tuple[np.ndarray, Dict[str, Any]]:
        native_level, native_ds, virtual = self.choose_native_level(target_ds)
        # physical field width in level-0 pixels
        field0 = float(out_size) * float(target_ds)
        native_size = max(1, int(math.ceil(field0 / native_ds)))
        region = self.slide.read_region((int(x0), int(y0)), native_level, (native_size, native_size)).convert("RGB")
        if native_size != out_size:
            region = region.resize((out_size, out_size), resample=resample)
        return np.asarray(region), {
            "native_level": native_level,
            "native_downsample": native_ds,
            "target_downsample": target_ds,
            "virtual_scale": virtual,
        }

    def thumbnail_at_downsample(self, target_ds: float, max_size: int = 4096,
                                resample=Image.Resampling.BILINEAR) -> Tuple[np.ndarray, Dict[str, Any]]:
        w0, h0 = self.dimensions
        out_w = max(1, int(round(w0 / target_ds)))
        out_h = max(1, int(round(h0 / target_ds)))
        scale = min(1.0, max_size / max(out_w, out_h))
        out_w2 = max(1, int(round(out_w * scale)))
        out_h2 = max(1, int(round(out_h * scale)))
        effective_ds = target_ds / scale
        native_level, native_ds, virtual = self.choose_native_level(effective_ds)
        # use get_thumbnail to avoid giant intermediary
        thumb = self.slide.get_thumbnail((out_w2, out_h2)).convert("RGB")
        return np.asarray(thumb), {
            "native_level": native_level,
            "native_downsample": native_ds,
            "target_downsample": effective_ds,
            "virtual_scale": virtual,
            "effective_ds": effective_ds,
        }


class MaskScaleReader:
    """Mask reader with automatic coordinate scaling to image level-0 dimensions."""
    def __init__(self, mask_path: str | Path, image_dims0: Tuple[int, int]):
        self.reader = SlideScaleReader(mask_path)
        self.image_dims0 = tuple(map(int, image_dims0))
        self.mask_dims0 = self.reader.dimensions
        self.rx = self.mask_dims0[0] / max(self.image_dims0[0], 1)
        self.ry = self.mask_dims0[1] / max(self.image_dims0[1], 1)

    def close(self):
        self.reader.close()

    def read_raw(self, x0_img: int, y0_img: int, target_ds_img: float, out_size: int) -> Tuple[np.ndarray, Dict[str, Any]]:
        x0m = int(round(x0_img * self.rx))
        y0m = int(round(y0_img * self.ry))
        # target downsample in mask coordinate system so same physical region is returned
        # image field0 = out_size*target_ds_img, mask field0 = image field0 * ratio.
        target_ds_mask = target_ds_img * (self.rx + self.ry) / 2.0
        arr, meta = self.reader.read_at_downsample(
            x0m, y0m, target_ds_mask, out_size, resample=Image.Resampling.NEAREST
        )
        raw = arr[..., 0].astype(np.uint8)
        meta.update({"mask_rx": self.rx, "mask_ry": self.ry})
        return raw, meta


# =============================================================================
# Tissue mask (CAMELYON-style HSV/Otsu/morphology + safe fallback)
# =============================================================================


def generate_tissue_mask(rgb: np.ndarray, cfg: Config) -> Tuple[np.ndarray, Dict[str, Any]]:
    repairs = []
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError("Tissue mask expects RGB image")
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    sat = hsv[..., 1]
    # Otsu on saturation is robust across H&E slides.
    try:
        otsu, _ = cv2.threshold(sat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        threshold = max(int(otsu), int(round(cfg.tissue_default_threshold * 255)))
    except Exception:
        threshold = int(round(cfg.tissue_default_threshold * 255))
    mask = sat > threshold

    k = max(1, int(cfg.morph_disk_size))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
    m = (mask.astype(np.uint8) * 255)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)

    # Remove tiny components.
    n, labels, stats, _ = cv2.connectedComponentsWithStats((m > 0).astype(np.uint8), 8)
    clean = np.zeros_like(m, dtype=np.uint8)
    for i in range(1, n):
        if int(stats[i, cv2.CC_STAT_AREA]) >= int(cfg.min_object_size):
            clean[labels == i] = 255

    frac = float((clean > 0).mean())
    # Safe repair: if saturation segmentation yields no tissue, fall back to non-white RGB.
    if frac < 0.0005:
        fallback = (rgb.mean(axis=2) < 240) & (rgb.min(axis=2) < 245)
        clean = (fallback.astype(np.uint8) * 255)
        repairs.append("Otsu/saturation tissue mask was nearly empty; used conservative non-white RGB fallback")
        frac = float((clean > 0).mean())

    return clean, {"tissue_fraction": frac, "repairs": repairs, "sat_threshold": threshold}


def tissue_fraction_for_patch(tissue_mask: np.ndarray, x0: int, y0: int, target_ds: float,
                              patch_size: int, tissue_effective_ds: float) -> float:
    # Convert level-0 physical coordinates into tissue thumbnail coordinates.
    x = int(round(x0 / tissue_effective_ds))
    y = int(round(y0 / tissue_effective_ds))
    size = max(1, int(round((patch_size * target_ds) / tissue_effective_ds)))
    x2 = min(tissue_mask.shape[1], x + size)
    y2 = min(tissue_mask.shape[0], y + size)
    if x >= tissue_mask.shape[1] or y >= tissue_mask.shape[0] or x2 <= x or y2 <= y:
        return 0.0
    return float((tissue_mask[y:y2, x:x2] > 0).mean())


# =============================================================================
# STEP 1 — Audit
# =============================================================================


def build_case_index(cfg: Config) -> Dict[str, Dict[str, str]]:
    image_idx: Dict[str, str] = {}
    mask_idx: Dict[str, str] = {}
    for p in cfg.images_dir.glob("*.tif*"):
        image_idx[p.stem.lower()] = str(p)
    for p in cfg.masks_dir.glob("*.tif*"):
        stem = p.stem.lower()
        if stem.endswith("_mask"):
            stem = stem[:-5]
        mask_idx[stem] = str(p)
    return {"images": image_idx, "masks": mask_idx}


def step_audit(cfg: Config):
    cfg.ensure_dirs()
    require_packages()
    repairs: List[str] = []
    if not cfg.train_csv.exists():
        qc_report(cfg, "01_audit", False, {"error": f"Missing {cfg.train_csv}"})
    if not cfg.images_dir.exists() or not cfg.masks_dir.exists():
        qc_report(cfg, "01_audit", False, {"error": "train_images or train_label_masks folder missing"})

    df = pd.read_csv(cfg.train_csv)
    required = {"image_id", "data_provider"}
    missing_cols = required - set(df.columns)
    if missing_cols:
        qc_report(cfg, "01_audit", False, {"error": f"Missing train.csv columns: {sorted(missing_cols)}"})

    idx = build_case_index(cfg)
    rows = []
    invalid = 0
    provider_values: Dict[str, set] = defaultdict(set)
    virtual_counts = Counter()

    for _, r in tqdm(df.iterrows(), total=len(df), desc="AUDIT"):
        image_id = str(r["image_id"])
        key = image_id.lower()
        provider = canonical_provider(r["data_provider"])
        image_path = idx["images"].get(key)
        mask_path = idx["masks"].get(key)
        rec: Dict[str, Any] = {
            "image_id": image_id,
            "provider": provider,
            "isup_grade": r.get("isup_grade", ""),
            "gleason_score": r.get("gleason_score", ""),
            "image_path": image_path or "",
            "mask_path": mask_path or "",
            "image_exists": bool(image_path),
            "mask_exists": bool(mask_path),
            "valid": False,
            "error": "",
        }
        if not image_path or not mask_path:
            invalid += 1
            rec["error"] = "missing image or mask"
            rows.append(rec)
            continue
        try:
            im = SlideScaleReader(image_path)
            ma = SlideScaleReader(mask_path)
            rec["image_dims0"] = json.dumps(im.dimensions)
            rec["mask_dims0"] = json.dumps(ma.dimensions)
            rec["image_level_count"] = len(im.level_dimensions)
            rec["mask_level_count"] = len(ma.level_dimensions)
            rec["image_downsamples"] = json.dumps([round(x, 4) for x in im.level_downsamples])
            rec["mask_downsamples"] = json.dumps([round(x, 4) for x in ma.level_downsamples])
            for lv in cfg.levels:
                _, _, virtual = im.choose_native_level(cfg.logical_downsamples[lv])
                virtual_counts[f"L{lv}_virtual"] += int(virtual)

            # Read a small center mask patch to inspect values; full mask scan would be expensive.
            cx = max(0, im.dimensions[0] // 2 - int(256 * cfg.logical_downsamples[2]))
            cy = max(0, im.dimensions[1] // 2 - int(256 * cfg.logical_downsamples[2]))
            mr = MaskScaleReader(mask_path, im.dimensions)
            raw, _ = mr.read_raw(cx, cy, cfg.logical_downsamples[2], 512)
            provider_values[provider].update(np.unique(raw).tolist())
            mr.close()
            rec["valid"] = True
            im.close(); ma.close()
        except Exception as e:
            invalid += 1
            rec["error"] = repr(e)
        rows.append(rec)

    out = pd.DataFrame(rows)
    out.to_csv(cfg.audit_csv, index=False)
    valid_df = out[out["valid"] == True]
    if len(valid_df) == 0:
        qc_report(cfg, "01_audit", False, {"valid": 0, "invalid": invalid})

    # Semantics gate: only known providers.
    unknown_providers = sorted(set(valid_df["provider"]) - set(EXPECTED_MASK_VALUES))
    if unknown_providers:
        qc_report(cfg, "01_audit", False, {"unknown_providers": unknown_providers})

    # sampled values should be within documented sets; if center patch has only 0 that's fine.
    bad_vals = {}
    for prov, vals in provider_values.items():
        unexpected = set(vals) - EXPECTED_MASK_VALUES.get(prov, set())
        if unexpected:
            bad_vals[prov] = sorted(unexpected)
    if bad_vals:
        qc_report(cfg, "01_audit", False, {"unexpected_mask_values": bad_vals})

    # Virtual scale use is an expected auto-repair for PANDA TIFF pyramids.
    for k, v in virtual_counts.items():
        if v:
            repairs.append(f"{k}: {v} slides do not have an exact native level; nearest higher-resolution native level will be resized to the requested CAMELYON scale")

    qc_report(cfg, "01_audit", True, {
        "rows_in_train_csv": len(df),
        "valid_image_mask_pairs": int(len(valid_df)),
        "invalid_pairs": int(invalid),
        "providers": valid_df["provider"].value_counts().to_dict(),
        "sampled_raw_mask_values": {k: sorted(map(int, v)) for k, v in provider_values.items()},
        "virtual_scale_counts": dict(virtual_counts),
        "audit_csv": str(cfg.audit_csv),
    }, repairs)


# =============================================================================
# STEP 2 — Slide-level split BEFORE patch extraction
# =============================================================================


def step_split(cfg: Config):
    if not cfg.audit_csv.exists():
        step_audit(cfg)
    df = pd.read_csv(cfg.audit_csv)
    df = df[df["valid"] == True].copy()
    if len(df) < 10:
        qc_report(cfg, "02_split", False, {"error": "Too few valid slides"})

    # Stratify by provider + ISUP where possible; merge rare strata safely.
    isup = df.get("isup_grade", pd.Series(["NA"] * len(df))).astype(str)
    strat = df["provider"].astype(str) + "__" + isup
    counts = strat.value_counts()
    rare = set(counts[counts < 3].index)
    strat2 = [s if s not in rare else s.split("__")[0] + "__RARE" for s in strat]
    strat2 = pd.Series(strat2, index=df.index)
    # If rare merging still too small, provider-only fallback.
    if (strat2.value_counts() < 2).any():
        strat2 = df["provider"].astype(str)

    sss1 = StratifiedShuffleSplit(n_splits=1, train_size=cfg.train_ratio, random_state=cfg.seed)
    tr_idx, rest_idx = next(sss1.split(df, strat2))
    train = df.iloc[tr_idx].copy()
    rest = df.iloc[rest_idx].copy()
    rest_strat = strat2.iloc[rest_idx]
    val_rel = cfg.val_ratio / (cfg.val_ratio + cfg.test_ratio)
    try:
        sss2 = StratifiedShuffleSplit(n_splits=1, train_size=val_rel, random_state=cfg.seed + 1)
        va_i, te_i = next(sss2.split(rest, rest_strat))
    except ValueError:
        rng = np.random.default_rng(cfg.seed + 1)
        order = rng.permutation(len(rest))
        cut = int(round(len(rest) * val_rel))
        va_i, te_i = order[:cut], order[cut:]

    val = rest.iloc[va_i].copy(); test = rest.iloc[te_i].copy()
    train["split"] = "train"; val["split"] = "val"; test["split"] = "test"
    out = pd.concat([train, val, test], ignore_index=True)
    cfg.splits_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(cfg.split_csv, index=False)

    sets = {s: set(out[out.split == s].image_id.astype(str)) for s in ["train", "val", "test"]}
    overlaps = {
        "train_val": len(sets["train"] & sets["val"]),
        "train_test": len(sets["train"] & sets["test"]),
        "val_test": len(sets["val"] & sets["test"]),
    }
    passed = all(v == 0 for v in overlaps.values()) and len(out) == len(df)
    details = {
        "counts": out["split"].value_counts().to_dict(),
        "providers_by_split": out.groupby(["split", "provider"]).size().unstack(fill_value=0).to_dict(),
        "overlaps": overlaps,
        "split_csv": str(cfg.split_csv),
    }
    qc_report(cfg, "02_split", passed, details)


# =============================================================================
# STEP 3 — Image/mask alignment + label conversion visual QC
# =============================================================================


def save_overlay(rgb: np.ndarray, mask255: np.ndarray, path: Path, alpha: float = 0.35):
    out = rgb.astype(np.float32).copy()
    red = np.zeros_like(out); red[..., 0] = 255
    m = mask255 > 0
    out[m] = (1 - alpha) * out[m] + alpha * red[m]
    Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)).save(path)


def make_triptych(rgb: np.ndarray, raw: np.ndarray, binmask: np.ndarray, out_path: Path):
    h, w = rgb.shape[:2]
    raw_vis = np.zeros((h, w, 3), dtype=np.uint8)
    palette = {
        0: (255, 255, 255), 1: (180, 180, 180), 2: (100, 180, 100),
        3: (255, 230, 0), 4: (255, 140, 0), 5: (220, 0, 0)
    }
    for v in np.unique(raw):
        raw_vis[raw == v] = palette.get(int(v), (0, 0, 0))
    overlay = rgb.astype(np.float32).copy()
    m = binmask > 0
    overlay[m] = overlay[m] * 0.6 + np.array([255, 0, 0], dtype=np.float32) * 0.4
    canvas = np.concatenate([rgb, raw_vis, overlay.astype(np.uint8)], axis=1)
    Image.fromarray(canvas).save(out_path)


def step_qc_alignment(cfg: Config):
    if not cfg.split_csv.exists():
        step_split(cfg)
    df = pd.read_csv(cfg.split_csv)
    qdir = cfg.qc_dir / "alignment_samples"
    qdir.mkdir(parents=True, exist_ok=True)
    rows = []
    repairs = []
    rng = np.random.default_rng(cfg.seed)

    chosen = []
    for prov in ["radboud", "karolinska"]:
        sub = df[df.provider == prov]
        if len(sub):
            idx = rng.choice(len(sub), size=min(cfg.qc_slides_per_provider, len(sub)), replace=False)
            chosen.extend([sub.iloc[int(i)] for i in idx])

    for r in chosen:
        im = SlideScaleReader(r.image_path)
        mr = MaskScaleReader(r.mask_path, im.dimensions)
        # Build candidates from actual tissue locations, not only the slide center.
        ds = cfg.logical_downsamples[2]
        field = int(cfg.patch_size * ds)
        low_rgb, low_meta = im.thumbnail_at_downsample(cfg.logical_downsamples.get(cfg.tissue_level, 32.0), cfg.target_max_size)
        low_tissue, _ = generate_tissue_mask(low_rgb, cfg)
        eff_ds = float(low_meta.get("effective_ds", cfg.logical_downsamples.get(cfg.tissue_level, 32.0)))
        ys, xs = np.where(low_tissue > 0)
        candidates = []
        if len(xs):
            # deterministic spread through tissue pixels; convert thumbnail coords to level-0.
            take = np.linspace(0, len(xs)-1, min(30, len(xs))).astype(int)
            for ii in take:
                cx0 = int(xs[ii] * eff_ds); cy0 = int(ys[ii] * eff_ds)
                candidates.append((max(0, min(im.dimensions[0]-field, cx0-field//2)),
                                   max(0, min(im.dimensions[1]-field, cy0-field//2))))
        if not candidates:
            candidates = [(max(0, im.dimensions[0]//2-field//2), max(0, im.dimensions[1]//2-field//2))]
        best = None
        for x0, y0 in candidates:
            rgb, _ = im.read_at_downsample(x0, y0, ds, cfg.patch_size)
            raw, meta = mr.read_raw(x0, y0, ds, cfg.patch_size)
            try:
                binm = raw_to_binary_mask(raw, r.provider)
            except QCError:
                im.close(); mr.close(); raise
            img_tissue, _ = generate_tissue_mask(rgb, cfg)
            valid = raw_valid_tissue_mask(raw, r.provider)
            union = ((img_tissue > 0) | (valid > 0)).sum()
            inter = ((img_tissue > 0) & (valid > 0)).sum()
            overlap = float(inter / max(union, 1))
            tissue_frac = float((img_tissue > 0).mean())
            valid_frac = float((valid > 0).mean())
            score = overlap + 0.1 * min(tissue_frac, valid_frac)
            if best is None or score > best[0]:
                best = (score, x0, y0, rgb, raw, binm, overlap, tissue_frac, valid_frac, meta)
        assert best is not None
        _, x0, y0, rgb, raw, binm, overlap, tissue_frac, valid_frac, meta = best
        out = qdir / f"{r.image_id}_{r.provider}.png"
        make_triptych(rgb, raw, binm, out)
        rows.append({
            "image_id": r.image_id, "provider": r.provider,
            "overlap_jaccard": overlap, "image_tissue_frac": tissue_frac,
            "mask_valid_frac": valid_frac, "mask_rx": meta["mask_rx"], "mask_ry": meta["mask_ry"],
            "sample_png": str(out),
        })
        im.close(); mr.close()

    rdf = pd.DataFrame(rows)
    rdf.to_csv(cfg.qc_dir / "alignment_report.csv", index=False)
    # Alignment overlap is only meaningful on annotated regions; low values can occur in unknown mask areas.
    # Gate on dimensional mapping sanity plus at least some successful overlaps.
    ratio_bad = ((rdf.mask_rx < 0.95) | (rdf.mask_rx > 1.05) | (rdf.mask_ry < 0.95) | (rdf.mask_ry > 1.05)).sum()
    if ratio_bad:
        repairs.append(f"{int(ratio_bad)} sampled masks had level-0 dimensions different from image; coordinate scale ratios are applied automatically")
    strong = int((rdf.overlap_jaccard >= cfg.alignment_min_overlap).sum())
    passed = len(rdf) > 0 and strong >= max(1, len(rdf)//4)
    qc_report(cfg, "03_alignment", passed, {
        "samples": len(rdf),
        "samples_with_overlap_above_threshold": strong,
        "mean_overlap_jaccard": float(rdf.overlap_jaccard.mean()) if len(rdf) else 0.0,
        "visual_dir": str(qdir),
        "note": "Visual triptych = RGB | raw provider mask | binary tumor overlay. Review these before extraction.",
    }, repairs)


# =============================================================================
# STEP 4 — Patch extraction: 4 levels, tissue filter, 8 bucket metadata
# =============================================================================


def patch_output_paths(cfg: Config, split: str, level: int, category: str, image_id: str, x0: int, y0: int) -> Tuple[Path, Path]:
    stem = f"{image_id}_L{level}_{x0}_{y0}"
    img = cfg.patches_dir / split / f"level_{level}" / f"{category}_patches" / f"{stem}.png"
    msk = cfg.patches_dir / split / f"level_{level}" / f"{category}_masks" / f"{stem}.png"
    return img, msk


def extract_slide_level(cfg: Config, r: pd.Series, level: int) -> List[Dict[str, Any]]:
    ds = float(cfg.logical_downsamples[level])
    image_id = str(r.image_id)
    split = str(r.split)
    provider = str(r.provider)
    im = SlideScaleReader(r.image_path)
    mr = MaskScaleReader(r.mask_path, im.dimensions)

    # tissue thumbnail at logical tissue level (virtual if needed)
    tissue_ds = float(cfg.logical_downsamples.get(cfg.tissue_level, 32.0))
    thumb, tmeta = im.thumbnail_at_downsample(tissue_ds, cfg.target_max_size)
    tissue_mask, tinfo = generate_tissue_mask(thumb, cfg)
    tissue_effective_ds = float(tmeta.get("effective_ds", tissue_ds))

    # CAMELYON stride=128 at the target level => stride0 = 128*downsample.
    stride0 = max(1, int(round(cfg.stride * ds)))
    field0 = max(1, int(round(cfg.patch_size * ds)))
    w0, h0 = im.dimensions

    coords = []
    for y0 in range(0, max(1, h0 - field0 + 1), stride0):
        for x0 in range(0, max(1, w0 - field0 + 1), stride0):
            tf = tissue_fraction_for_patch(tissue_mask, x0, y0, ds, cfg.patch_size, tissue_effective_ds)
            if tf >= cfg.tissue_min_frac:
                coords.append((x0, y0, tf))

    # deterministic cap only protects pathological slides from exploding storage.
    if len(coords) > cfg.max_candidate_patches_per_slide_level:
        rng = np.random.default_rng(cfg.seed + int(sha1_text(image_id + str(level)), 16) % 1_000_000)
        idx = rng.choice(len(coords), cfg.max_candidate_patches_per_slide_level, replace=False)
        coords = [coords[int(i)] for i in sorted(idx)]

    records = []
    for x0, y0, tf in coords:
        img_path, mask_path = patch_output_paths(cfg, split, level, "tmp", image_id, x0, y0)
        # We don't know category until mask read; compute final paths after.
        rgb, smeta = im.read_at_downsample(x0, y0, ds, cfg.patch_size)
        raw, mmeta = mr.read_raw(x0, y0, ds, cfg.patch_size)
        binm = raw_to_binary_mask(raw, provider)
        tumor_fraction = float((binm > 0).mean())
        category = "tumor" if tumor_fraction >= cfg.tumor_threshold else "normal"
        img_path, mask_path = patch_output_paths(cfg, split, level, category, image_id, x0, y0)
        img_path.parent.mkdir(parents=True, exist_ok=True)
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        if not (cfg.extraction_resume and img_path.exists() and mask_path.exists()):
            ok1 = cv2.imwrite(str(img_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, cfg.save_png_compress])
            ok2 = cv2.imwrite(str(mask_path), binm, [cv2.IMWRITE_PNG_COMPRESSION, cfg.save_png_compress])
            if not (ok1 and ok2):
                raise IOError(f"Failed writing patch {img_path}")
        # normalized coordinates within WSI, better bounded than filename-only formula
        cx0 = x0 + field0 / 2.0; cy0 = y0 + field0 / 2.0
        records.append({
            "image_id": image_id, "provider": provider, "split": split,
            "level": level, "label": int(category == "tumor"), "category": category,
            "x0": int(x0), "y0": int(y0), "coord_x": float(cx0 / max(w0, 1)), "coord_y": float(cy0 / max(h0, 1)),
            "target_downsample": ds, "native_level": smeta["native_level"], "virtual_scale": bool(smeta["virtual_scale"]),
            "tissue_fraction": float(tf), "tumor_fraction": tumor_fraction,
            "img": str(img_path), "mask": str(mask_path),
        })

    im.close(); mr.close()
    return records


def verify_patch_manifest(cfg: Config, df: pd.DataFrame) -> Tuple[bool, Dict[str, Any]]:
    errors = []
    sample = df.sample(min(len(df), cfg.qc_patches_per_stage), random_state=cfg.seed) if len(df) else df
    for _, r in sample.iterrows():
        if not Path(r.img).exists() or not Path(r["mask"]).exists():
            errors.append(f"missing:{r.img}"); continue
        im = cv2.imread(str(r.img), cv2.IMREAD_COLOR)
        ma = cv2.imread(str(r["mask"]), cv2.IMREAD_GRAYSCALE)
        if im is None or ma is None:
            errors.append(f"unreadable:{r.img}"); continue
        if im.shape[:2] != (cfg.patch_size, cfg.patch_size) or ma.shape != (cfg.patch_size, cfg.patch_size):
            errors.append(f"shape:{r.img}:{im.shape}:{ma.shape}")
        vals = set(np.unique(ma).tolist())
        if not vals.issubset({0, 255}):
            errors.append(f"mask_values:{r['mask']}:{vals}")
        tf_re = float(((cv2.cvtColor(im, cv2.COLOR_BGR2RGB).mean(axis=2) < 248)).mean())
        if tf_re < 0.001:
            errors.append(f"blank_patch:{r.img}")
    overlaps = {}
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        sa = set(df[df.split == a].image_id.astype(str)); sb = set(df[df.split == b].image_id.astype(str))
        overlaps[f"{a}_{b}"] = len(sa & sb)
    passed = not errors and all(v == 0 for v in overlaps.values())
    return passed, {"sample_errors": errors[:50], "slide_overlap": overlaps}


def step_extract(cfg: Config):
    if not (cfg.qc_dir / "03_alignment.json").exists():
        step_qc_alignment(cfg)
    split_df = pd.read_csv(cfg.split_csv)
    all_records: List[Dict[str, Any]] = []
    # resume at slide-level manifest shards
    shard_dir = cfg.manifests_dir / "extract_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    total_jobs = len(split_df) * len(cfg.levels)
    with tqdm(total=total_jobs, desc="EXTRACT") as bar:
        for _, r in split_df.iterrows():
            for lv in cfg.levels:
                shard = shard_dir / f"{r.image_id}_L{lv}.csv"
                try:
                    if cfg.extraction_resume and shard.exists():
                        sdf = pd.read_csv(shard)
                        recs = sdf.to_dict("records")
                    else:
                        recs = extract_slide_level(cfg, r, lv)
                        pd.DataFrame(recs).to_csv(shard, index=False)
                    all_records.extend(recs)
                except Exception as e:
                    atomic_json(cfg.qc_dir / "extract_last_error.json", {
                        "image_id": str(r.image_id), "level": lv, "error": repr(e)
                    })
                    raise
                finally:
                    bar.update(1)

    manifest = pd.DataFrame(all_records)
    if len(manifest) == 0:
        qc_report(cfg, "04_extract", False, {"error": "No patches extracted"})
    manifest.to_csv(cfg.patch_manifest, index=False)
    passed, extra = verify_patch_manifest(cfg, manifest)

    bucket = manifest.groupby(["split", "level", "category"]).size().reset_index(name="n")
    bucket.to_csv(cfg.qc_dir / "bucket_counts_before_selection.csv", index=False)

    # QC montage
    qdir = cfg.qc_dir / "patch_samples"
    qdir.mkdir(parents=True, exist_ok=True)
    sample = manifest.sample(min(cfg.qc_patches_per_stage, len(manifest)), random_state=cfg.seed)
    thumbs = []
    for _, rr in sample.iterrows():
        rgb = cv2.cvtColor(cv2.imread(rr.img), cv2.COLOR_BGR2RGB)
        ma = cv2.imread(rr["mask"], cv2.IMREAD_GRAYSCALE)
        ov = rgb.copy(); ov[ma > 0] = (0.6 * ov[ma > 0] + 0.4 * np.array([255, 0, 0])).astype(np.uint8)
        thumbs.append(ov)
    if thumbs:
        cols = 4; rows = math.ceil(len(thumbs)/cols)
        canvas = Image.new("RGB", (cols*cfg.patch_size, rows*cfg.patch_size), "white")
        for i, a in enumerate(thumbs):
            canvas.paste(Image.fromarray(a), ((i%cols)*cfg.patch_size, (i//cols)*cfg.patch_size))
        canvas.thumbnail((2048, 2048))
        canvas.save(qdir / "accepted_patch_overlay_grid.jpg", quality=90)

    qc_report(cfg, "04_extract", passed, {
        "total_patches": int(len(manifest)),
        "counts_by_split_level_category": bucket.to_dict("records"),
        "manifest": str(cfg.patch_manifest),
        **extra,
    })


# =============================================================================
# STEP 5 — EfficientNet-B4 features + KMeans within 8 TRAIN buckets
# =============================================================================


def get_effb4_feature_extractor(cfg: Config, device: torch.device):
    weights = models.EfficientNet_B4_Weights.DEFAULT
    model = models.efficientnet_b4(weights=weights)
    model.eval().to(device)
    features = model.features
    avgpool = model.avgpool
    mean = weights.transforms().mean
    std = weights.transforms().std
    preprocess = T.Compose([
        T.Resize((cfg.feat_img_size, cfg.feat_img_size)),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])
    return features, avgpool, preprocess


@torch.inference_mode()
def extract_features_for_df(cfg: Config, df: pd.DataFrame, out_npz: Path) -> Tuple[np.ndarray, pd.DataFrame]:
    # Candidate cap is applied per bucket before feature extraction to keep this feasible.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    features_net, avgpool, preprocess = get_effb4_feature_extractor(cfg, device)
    all_feats = []
    kept_rows = []
    bs = cfg.feat_batch_size
    for start in tqdm(range(0, len(df), bs), desc=f"EFFB4 FEATURES {out_npz.stem}"):
        batch_df = df.iloc[start:start+bs]
        imgs = []; meta_rows = []
        for _, r in batch_df.iterrows():
            try:
                imgs.append(preprocess(Image.open(r.img).convert("RGB")))
                meta_rows.append(r)
            except Exception:
                pass
        if not imgs:
            continue
        x = torch.stack(imgs).to(device, non_blocking=True)
        f = features_net(x); f = avgpool(f); f = torch.flatten(f, 1)
        all_feats.append(f.cpu().numpy().astype(np.float32))
        kept_rows.extend(meta_rows)
    if not all_feats:
        raise QCError("No EfficientNet features extracted")
    X = np.concatenate(all_feats, axis=0)
    kdf = pd.DataFrame(kept_rows).reset_index(drop=True)
    np.savez_compressed(out_npz, X=X)
    kdf.to_csv(out_npz.with_suffix(".csv"), index=False)
    return X, kdf


def choose_k_elbow(X: np.ndarray, max_k: int, seed: int) -> int:
    n = len(X)
    if n < 4:
        return 1
    max_k = min(max_k, n)
    if max_k <= 2:
        return max_k
    # fit on a bounded subset to avoid huge elbow cost
    rng = np.random.default_rng(seed)
    subset = X if n <= 5000 else X[rng.choice(n, 5000, replace=False)]
    ks = list(range(2, max_k+1))
    inertias = []
    for k in ks:
        km = KMeans(n_clusters=k, random_state=seed, n_init=10)
        km.fit(subset)
        inertias.append(float(km.inertia_))
    # maximum perpendicular distance from line joining first-last elbow curve
    x = np.array(ks, dtype=float); y = np.array(inertias, dtype=float)
    xN = (x-x.min()) / max(x.max()-x.min(), 1e-9)
    yN = (y-y.min()) / max(y.max()-y.min(), 1e-9)
    p1 = np.array([xN[0], yN[0]]); p2 = np.array([xN[-1], yN[-1]])
    line = p2-p1; den = np.linalg.norm(line)
    if den < 1e-9:
        return min(4, max_k)
    d = []
    for xx, yy in zip(xN, yN):
        p = np.array([xx, yy])
        d.append(abs(np.cross(line, p-p1))/den)
    return int(ks[int(np.argmax(d))])


def representative_select(X: np.ndarray, df: pd.DataFrame, target_n: int, max_k: int, seed: int) -> pd.DataFrame:
    if len(df) <= target_n:
        return df.copy()
    k = choose_k_elbow(X, max_k, seed)
    km = KMeans(n_clusters=k, random_state=seed, n_init=10)
    labels = km.fit_predict(X)
    centers = km.cluster_centers_
    # Allocate quota proportional to cluster size but at least one, then adjust.
    counts = np.bincount(labels, minlength=k)
    quotas = np.maximum(1, np.floor(target_n * counts / counts.sum()).astype(int))
    while quotas.sum() > target_n:
        j = int(np.argmax(quotas))
        if quotas[j] > 1: quotas[j] -= 1
        else: break
    while quotas.sum() < target_n:
        room = counts - quotas
        j = int(np.argmax(room))
        if room[j] <= 0: break
        quotas[j] += 1
    chosen = []
    for c in range(k):
        idx = np.where(labels == c)[0]
        if len(idx) == 0: continue
        dist = np.linalg.norm(X[idx] - centers[c][None, :], axis=1)
        order = idx[np.argsort(dist)]
        # representative points around centroid; spread through ranked list if quota > 1
        q = min(int(quotas[c]), len(order))
        if q == 1:
            pick = [order[0]]
        else:
            pos = np.linspace(0, min(len(order)-1, max(q*3, q)-1), q).astype(int)
            pick = order[pos].tolist()
        chosen.extend(pick)
    if len(chosen) < target_n:
        remain = np.setdiff1d(np.arange(len(df)), np.array(chosen, dtype=int))
        rng = np.random.default_rng(seed)
        add = rng.choice(remain, min(target_n-len(chosen), len(remain)), replace=False)
        chosen.extend(add.tolist())
    return df.iloc[chosen[:target_n]].copy()


def step_select(cfg: Config):
    if not cfg.patch_manifest.exists():
        step_extract(cfg)
    df = pd.read_csv(cfg.patch_manifest)
    train = df[df.split == "train"].copy()
    if len(train) == 0:
        qc_report(cfg, "05_select", False, {"error": "No train patches"})

    # Exactly 8 buckets expected: level x tumor/normal.
    bucket_counts = train.groupby(["level", "category"]).size().to_dict()
    expected = [(lv, cat) for lv in cfg.levels for cat in ["normal", "tumor"]]
    missing = [b for b in expected if bucket_counts.get(b, 0) == 0]
    if missing:
        qc_report(cfg, "05_select", False, {"missing_training_buckets": missing, "counts": {str(k): int(v) for k,v in bucket_counts.items()}})

    per_bucket_requested = cfg.num_samples // len(expected)
    per_bucket = min(per_bucket_requested, min(int(bucket_counts[b]) for b in expected))
    repairs = []
    if per_bucket < per_bucket_requested:
        repairs.append(f"At least one bucket has < {per_bucket_requested} patches; reduced ALL 8 buckets equally to {per_bucket} to preserve balance without duplicating patches")

    selected = []
    selection_info = []
    for lv, cat in expected:
        bdf = train[(train.level == lv) & (train.category == cat)].copy()
        # deterministic candidate cap before expensive features
        if len(bdf) > cfg.feature_candidate_cap_per_bucket:
            bdf = bdf.sample(cfg.feature_candidate_cap_per_bucket, random_state=cfg.seed + lv + (100 if cat=="tumor" else 0))
            repairs.append(f"Capped feature candidates for L{lv}-{cat} at {cfg.feature_candidate_cap_per_bucket} before KMeans")
        fbase = cfg.features_dir / f"L{lv}_{cat}_effb4"
        npz = fbase.with_suffix(".npz")
        csvp = fbase.with_suffix(".csv")
        if npz.exists() and csvp.exists():
            X = np.load(npz)["X"]
            fdf = pd.read_csv(csvp)
        else:
            X, fdf = extract_features_for_df(cfg, bdf, npz)
        chosen = representative_select(X, fdf, per_bucket, cfg.max_k_for_elbow, cfg.seed + lv)
        selected.append(chosen)
        selection_info.append({"level": lv, "category": cat, "available": len(train[(train.level==lv)&(train.category==cat)]),
                               "feature_candidates": len(fdf), "selected": len(chosen)})

    out = pd.concat(selected, ignore_index=True)
    out = out.sample(frac=1.0, random_state=cfg.seed).reset_index(drop=True)
    cfg.selected_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(cfg.selected_train_csv, index=False)

    counts_after = out.groupby(["level", "category"]).size().to_dict()
    balanced = len(set(counts_after.values())) == 1 and len(counts_after) == 8
    # Leakage re-check selected train vs val/test slides from manifest.
    valtest_ids = set(df[df.split.isin(["val", "test"])].image_id.astype(str))
    leak = len(set(out.image_id.astype(str)) & valtest_ids)
    passed = balanced and leak == 0
    qc_report(cfg, "05_select", passed, {
        "requested_total": cfg.num_samples,
        "actual_total": len(out),
        "per_bucket": per_bucket,
        "counts_after": {str(k): int(v) for k,v in counts_after.items()},
        "selection": selection_info,
        "train_to_valtest_slide_leakage": leak,
        "selected_train_csv": str(cfg.selected_train_csv),
    }, repairs)


# =============================================================================
# STEP 6 — Vahadane target selection + QC
# =============================================================================


def auto_choose_stain_target(cfg: Config, train_df: pd.DataFrame) -> Path:
    # Select a train-only patch near median RGB statistics among high-tissue normal/tumor mix.
    cand = train_df[train_df.tissue_fraction >= max(cfg.tissue_min_frac, 0.5)].copy()
    if len(cand) < 20:
        cand = train_df.copy()
    cand = cand.sample(min(200, len(cand)), random_state=cfg.seed)
    stats = []
    for _, r in cand.iterrows():
        arr = cv2.cvtColor(cv2.imread(r.img), cv2.COLOR_BGR2RGB)
        if arr is None: continue
        # ignore near-white pixels
        pix = arr.reshape(-1,3)
        pix = pix[pix.mean(axis=1) < 240]
        if len(pix) < 100: continue
        stats.append((r.img, *pix.mean(axis=0).tolist()))
    if not stats:
        raise QCError("Could not auto-select a Vahadane target from training patches")
    s = np.array([[x[1],x[2],x[3]] for x in stats])
    med = np.median(s, axis=0)
    d = np.linalg.norm(s-med, axis=1)
    chosen = stats[int(np.argmin(d))][0]
    shutil.copy2(chosen, cfg.stain_target_path)
    return cfg.stain_target_path


class VahadaneProcessor:
    def __init__(self, target_rgb: np.ndarray):
        if VahadaneNormalizer is None:
            raise RuntimeError("TIAToolbox VahadaneNormalizer unavailable")
        self.norm = VahadaneNormalizer()
        # TIAToolbox stain-normalization internals may modify RGB buffers in-place.
        # PIL-backed np.asarray arrays can be read-only, so always pass writable C copies.
        target_w = np.array(target_rgb, dtype=np.uint8, copy=True, order="C")
        self.norm.fit(target_w)

    def transform(self, rgb: np.ndarray) -> np.ndarray:
        rgb_w = np.array(rgb, dtype=np.uint8, copy=True, order="C")
        out = self.norm.transform(rgb_w)
        return np.array(out, dtype=np.uint8, copy=True)


def step_qc_normalization(cfg: Config):
    if not cfg.selected_train_csv.exists():
        step_select(cfg)
    train = pd.read_csv(cfg.selected_train_csv)
    repairs = []
    if not cfg.stain_target_path.exists():
        auto_choose_stain_target(cfg, train)
        repairs.append("Automatically selected a representative Vahadane target from TRAIN only")
    target = np.array(Image.open(cfg.stain_target_path).convert("RGB"))
    proc = VahadaneProcessor(target)

    sample = train.sample(min(cfg.qc_patches_per_stage, len(train)), random_state=cfg.seed)
    qdir = cfg.qc_dir / "normalization_samples"; qdir.mkdir(parents=True, exist_ok=True)
    failures = 0; changes = []
    for i, (_, r) in enumerate(sample.iterrows()):
        rgb = np.array(Image.open(r.img).convert("RGB"))
        try:
            norm = proc.transform(rgb)
            if norm.shape != rgb.shape or not np.isfinite(norm).all():
                raise ValueError("bad normalized output")
        except Exception:
            failures += 1
            continue
        changes.append(float(np.abs(norm.astype(float)-rgb.astype(float)).mean()))
        Image.fromarray(np.concatenate([rgb, norm], axis=1)).save(qdir / f"{i:03d}.jpg", quality=90)
    passed = failures == 0 and len(changes) > 0
    qc_report(cfg, "06_normalization", passed, {
        "target": str(cfg.stain_target_path), "samples": len(sample), "failures": failures,
        "mean_absolute_color_change": float(np.mean(changes)) if changes else None,
        "visual_dir": str(qdir),
    }, repairs)


# =============================================================================
# Dataset + augmentation + model
# =============================================================================

class PairAugStainHeavy:
    def __init__(self, cfg: Config, seed: int):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.color = T.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.05)
        self.blur = T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.2))
    def r(self): return float(self.rng.random())
    def __call__(self, img: Image.Image, mask: Image.Image):
        c = self.cfg
        if self.r() < c.aug_p_hflip: img, mask = TF.hflip(img), TF.hflip(mask)
        if self.r() < c.aug_p_vflip: img, mask = TF.vflip(img), TF.vflip(mask)
        if self.r() < c.aug_p_rot90:
            k = int(self.rng.integers(0,4))
            if k: img, mask = img.rotate(90*k), mask.rotate(90*k)
        if self.r() < c.aug_p_color: img = self.color(img)
        if self.r() < c.aug_p_gamma: img = TF.adjust_gamma(img, float(self.rng.uniform(0.7,1.5)))
        if self.r() < c.aug_p_blur: img = self.blur(img)
        if self.r() < c.aug_p_noise:
            a = np.asarray(img).astype(np.float32); sigma=float(self.rng.uniform(3,12))
            a=np.clip(a+self.rng.normal(0,sigma,a.shape),0,255).astype(np.uint8); img=Image.fromarray(a)
        if self.r() < c.aug_p_jpeg:
            from io import BytesIO
            b=BytesIO(); img.save(b,"JPEG",quality=int(self.rng.integers(35,90))); b.seek(0); img=Image.open(b).convert("RGB")
        return img, mask


class PandaPatchDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, cfg: Config, split: str, augment: bool, normalize: bool):
        self.df = frame.reset_index(drop=True)
        self.cfg = cfg; self.split = split; self.augment = augment; self.normalize = normalize
        self.pair_aug = PairAugStainHeavy(cfg, cfg.seed) if augment else None
        self._norm = None
        self._norm_cache = {}
    def __len__(self): return len(self.df)
    def _get_norm(self):
        if self._norm is None:
            target=np.array(Image.open(self.cfg.stain_target_path).convert("RGB")); self._norm=VahadaneProcessor(target)
        return self._norm
    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        # Use bracket access for manifest columns; Series.mask is a pandas method.
        # For normalized training/validation, DO NOT open the original RGB patch
        # on a cache hit. The old path opened original+cache every epoch, doubling
        # RGB disk I/O even though Vahadane had already been cached.
        img = None
        tumor_mask = np.asarray(Image.open(r["mask"]).convert("L")) > 127

        # PANDA label 0 means background/unknown, so it must NOT be trained as benign tissue.
        # valid_mask is saved beside every patch.  If an old manifest lacks it, fail loudly.
        if "valid_mask" not in self.df.columns or pd.isna(r.get("valid_mask", np.nan)):
            raise QCError(f"Missing valid_mask for patch {r['img']}; refusing to treat background/unknown as Normal.")
        valid_path = Path(str(r["valid_mask"]))
        if not valid_path.exists():
            raise QCError(f"Missing valid tissue mask file: {valid_path}")
        valid_mask = np.asarray(Image.open(valid_path).convert("L")) > 127

        # One synchronized ternary mask for augmentation:
        #   0   = valid benign
        #   128 = ignore/background/unknown
        #   255 = tumor
        ternary = np.full(tumor_mask.shape, 128, dtype=np.uint8)
        ternary[valid_mask & ~tumor_mask] = 0
        ternary[valid_mask & tumor_mask] = 255
        mask = Image.fromarray(ternary, mode="L")

        if self.normalize:
            key = str(r["img"])
            if key in self._norm_cache:
                img = self._norm_cache[key].copy()
            else:
                # Fast path: hit persistent cache without opening the original patch
                # and without constructing VahadaneProcessor. _get_norm() is called
                # ONLY on an actual cache miss.
                img = load_stain_cache_only(self.cfg, key, "vahadane")
                if img is None:
                    img = load_or_build_stain_cache(
                        self.cfg, key, "vahadane", self._get_norm().transform
                    )
                if len(self._norm_cache) < self.cfg.vahadane_cache_max:
                    self._norm_cache[key] = img.copy()
        else:
            with Image.open(r["img"]) as _im:
                img = _im.convert("RGB").copy()

        if self.pair_aug is not None:
            img, mask = self.pair_aug(img, mask)

        x = torch.from_numpy(np.asarray(img).copy()).permute(2,0,1).float()/255.0
        ma = np.asarray(mask)
        y_np = np.full(ma.shape, -1.0, dtype=np.float32)
        y_np[ma < 64] = 0.0
        y_np[ma > 192] = 1.0
        y = torch.from_numpy(y_np).unsqueeze(0)

        coords = torch.tensor([float(r.coord_x), float(r.coord_y)], dtype=torch.float32)
        level = torch.tensor(int(r.level), dtype=torch.long)
        return x, y, level, coords, str(r.image_id)


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        layers=[nn.Conv2d(in_ch,out_ch,3,padding=1,bias=False),nn.BatchNorm2d(out_ch),nn.ReLU(inplace=True),
                nn.Conv2d(out_ch,out_ch,3,padding=1,bias=False),nn.BatchNorm2d(out_ch),nn.ReLU(inplace=True)]
        if dropout>0: layers.append(nn.Dropout2d(dropout))
        self.net=nn.Sequential(*layers)
    def forward(self,x): return self.net(x)


class UNetEfficientNetB4(nn.Module):
    """CAMELYON-style EfficientNet-B4 U-Net with coordinate embedding at bottleneck."""
    def __init__(self, n_classes=1, dropout_p=0.1, use_coords=True, pretrained=True, coord_embed_dim=32):
        super().__init__()
        if timm is None: raise RuntimeError("timm required")
        self.use_coords = use_coords
        self.encoder = timm.create_model("efficientnet_b4", pretrained=pretrained, features_only=True, out_indices=(0,1,2,3,4))
        c0,c1,c2,c3,c4 = self.encoder.feature_info.channels()
        if use_coords:
            self.coord_fc=nn.Sequential(nn.Linear(2,64),nn.ReLU(inplace=True),nn.Linear(64,coord_embed_dim),nn.ReLU(inplace=True))
        else:
            self.coord_fc=None; coord_embed_dim=0
        self.dec3=DoubleConv(c4+coord_embed_dim+c3,256,dropout_p)
        self.dec2=DoubleConv(256+c2,128,dropout_p)
        self.dec1=DoubleConv(128+c1,64,dropout_p)
        self.dec0=DoubleConv(64+c0,32,dropout_p)
        self.final_conv=nn.Conv2d(32,n_classes,1)
    def forward(self,x,coords=None):
        x0,x1,x2,x3,x4=self.encoder(x)
        bottleneck=x4
        if self.use_coords:
            if coords is None: coords=torch.zeros((x.shape[0],2),device=x.device,dtype=x.dtype)
            ce=self.coord_fc(coords).unsqueeze(-1).unsqueeze(-1).expand(-1,-1,bottleneck.shape[2],bottleneck.shape[3])
            bottleneck=torch.cat([bottleneck,ce],dim=1)
        d3=F.interpolate(bottleneck,size=x3.shape[2:],mode="bilinear",align_corners=False); d3=self.dec3(torch.cat([d3,x3],1))
        d2=F.interpolate(d3,size=x2.shape[2:],mode="bilinear",align_corners=False); d2=self.dec2(torch.cat([d2,x2],1))
        d1=F.interpolate(d2,size=x1.shape[2:],mode="bilinear",align_corners=False); d1=self.dec1(torch.cat([d1,x1],1))
        d0=F.interpolate(d1,size=x0.shape[2:],mode="bilinear",align_corners=False); d0=self.dec0(torch.cat([d0,x0],1))
        logits=self.final_conv(d0)
        # EfficientNet first stage may be lower resolution than input; return input-sized logits.
        if logits.shape[2:] != x.shape[2:]: logits=F.interpolate(logits,size=x.shape[2:],mode="bilinear",align_corners=False)
        return logits


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, probs, targets):
        valid = targets >= 0.0
        t = torch.clamp(targets, 0.0, 1.0)
        p = probs * valid.float()
        t = t * valid.float()
        p = p.reshape(p.size(0), -1)
        t = t.reshape(t.size(0), -1)
        valid_flat = valid.reshape(valid.size(0), -1)
        inter = (p * t).sum(1)
        den = p.sum(1) + t.sum(1)
        dice = (2 * inter + self.smooth) / (den + self.smooth)
        has_valid = valid_flat.any(1)
        if has_valid.any():
            return 1 - dice[has_valid].mean()
        return probs.sum() * 0.0


class CombinedLoss(nn.Module):
    """BCE + Dice computed ONLY on annotated/valid PANDA tissue pixels."""
    def __init__(self, bce_weight=1.0, dice_weight=1.0):
        super().__init__()
        self.bw = bce_weight
        self.dw = dice_weight
        self.dice = DiceLoss()

    def forward(self, logits, targets):
        valid = targets >= 0.0
        if not valid.any():
            return logits.sum() * 0.0
        t = torch.clamp(targets, 0.0, 1.0)
        bce_map = F.binary_cross_entropy_with_logits(logits, t, reduction="none")
        bce = bce_map[valid].mean()
        dice = self.dice(torch.sigmoid(logits), targets)
        return self.bw * bce + self.dw * dice


# =============================================================================
# Metrics — background/unknown ignored
# =============================================================================

class PixelMeter:
    def __init__(self):
        self.tp=self.fp=self.tn=self.fn=0

    def update(self, logits, y, threshold=0.5):
        valid = y >= 0.0
        p = (torch.sigmoid(logits) >= threshold)
        t = (y >= 0.5)
        p = p[valid]
        t = t[valid]
        self.tp += int((p & t).sum().item())
        self.fp += int((p & ~t).sum().item())
        self.tn += int((~p & ~t).sum().item())
        self.fn += int((~p & t).sum().item())

    def metrics(self):
        tp,fp,tn,fn=map(float,[self.tp,self.fp,self.tn,self.fn]); eps=1e-9
        dice=(2*tp)/(2*tp+fp+fn+eps); iou=tp/(tp+fp+fn+eps)
        precision=tp/(tp+fp+eps); recall=tp/(tp+fn+eps)
        specificity=tn/(tn+fp+eps); acc=(tp+tn)/(tp+tn+fp+fn+eps)
        return {"tp":int(tp),"fp":int(fp),"tn":int(tn),"fn":int(fn),
                "dice":dice,"iou":iou,"precision":precision,"recall":recall,
                "specificity":specificity,"accuracy":acc}


def make_loader(ds: Dataset, cfg: Config, shuffle: bool) -> DataLoader:
    kwargs=dict(batch_size=cfg.batch_size,shuffle=shuffle,num_workers=cfg.num_workers,
                pin_memory=torch.cuda.is_available(),drop_last=False, worker_init_fn=fast_worker_init_fn)
    if cfg.num_workers>0:
        kwargs.update(persistent_workers=cfg.persistent_workers,prefetch_factor=cfg.prefetch_factor)
    try:
        dl=DataLoader(ds,**kwargs)
        _=next(iter(dl))
        return dl
    except Exception as e:
        print(f"DataLoader worker mode failed ({e}); auto-repair -> num_workers=0")
        kwargs.pop("persistent_workers",None); kwargs.pop("prefetch_factor",None); kwargs["num_workers"]=0
        return DataLoader(ds,**kwargs)


# =============================================================================
# STEP 7 — DataLoader QC + augmentation alignment
# =============================================================================


def build_frames(cfg: Config):
    selected = pd.read_csv(cfg.selected_train_csv)
    allp = pd.read_csv(cfg.patch_manifest)
    val = allp[allp.split=="val"].copy(); test=allp[allp.split=="test"].copy()
    return selected, val, test


def step_qc_dataloader(cfg: Config):
    if not (cfg.qc_dir/"06_normalization.json").exists(): step_qc_normalization(cfg)
    tr,va,te=build_frames(cfg)
    ds=PandaPatchDataset(tr.head(max(cfg.batch_size*2,16)),cfg,"train",augment=True,normalize=True)
    dl=make_loader(ds,cfg,shuffle=False)
    x,y,levels,coords,ids=next(iter(dl))
    vals=set(torch.unique(y).cpu().numpy().tolist())
    valid_pixels = int((y >= 0).sum().item())
    ignored_pixels = int((y < 0).sum().item())
    passed=(x.ndim==4 and x.shape[1]==3 and y.ndim==4 and y.shape[1]==1 and vals.issubset({-1.0,0.0,1.0}) and valid_pixels>0 and torch.isfinite(x).all() and torch.isfinite(y).all() and coords.shape[1]==2)
    # visual batch
    qdir=cfg.qc_dir/"dataloader_samples"; qdir.mkdir(parents=True,exist_ok=True)
    ims=[]
    for i in range(min(len(x),8)):
        a=(x[i].permute(1,2,0).cpu().numpy()*255).clip(0,255).astype(np.uint8)
        yi=y[i,0].cpu().numpy(); m=(yi>0.5); ign=(yi<0)
        a[m]=(0.6*a[m]+0.4*np.array([255,0,0])).astype(np.uint8)
        a[ign]=(0.75*a[ign]+0.25*np.array([128,128,128])).astype(np.uint8); ims.append(a)
    if ims:
        canvas=np.concatenate(ims,axis=1); Image.fromarray(canvas).save(qdir/"batch_overlay.jpg")
    qc_report(cfg,"07_dataloader",bool(passed),{
        "image_shape":list(x.shape),"mask_shape":list(y.shape),"mask_values":sorted(vals),"valid_pixels":valid_pixels,"ignored_pixels":ignored_pixels,"levels":sorted(set(levels.tolist())),
        "coords_min":coords.min(0).values.tolist(),"coords_max":coords.max(0).values.tolist(),"visual":str(qdir/"batch_overlay.jpg")
    })


# =============================================================================
# STEP 8 — Sanity overfit QC (with bounded auto-repair retry)
# =============================================================================


def dice_from_logits(logits,y,thr=0.5):
    valid=(y>=0); p=(torch.sigmoid(logits)>=thr).float(); t=torch.clamp(y,0,1)
    p=p[valid]; t=t[valid]
    if p.numel()==0: return 0.0
    inter=(p*t).sum(); return float((2*inter+1)/(p.sum()+t.sum()+1))


def run_sanity_once(cfg: Config, frame: pd.DataFrame, normalize: bool, augment: bool) -> float:
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds=PandaPatchDataset(frame,cfg,"train",augment=augment,normalize=normalize)
    _nw = int(getattr(cfg, "num_workers", 4))
    _kw = dict(batch_size=min(cfg.batch_size,len(ds)), shuffle=True, num_workers=_nw,
               pin_memory=torch.cuda.is_available(), worker_init_fn=fast_worker_init_fn)
    if _nw > 0:
        _kw.update(persistent_workers=bool(getattr(cfg, "persistent_workers", True)),
                   prefetch_factor=int(getattr(cfg, "prefetch_factor", 2)))
    dl=DataLoader(ds, **_kw)
    model=UNetEfficientNetB4(1,cfg.dropout_p,cfg.use_coords,cfg.pretrained,cfg.coord_embed_dim).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=cfg.learning_rate,weight_decay=cfg.weight_decay)
    lossfn=CombinedLoss(); scaler=torch.amp.GradScaler("cuda",enabled=cfg.amp and device.type=="cuda")
    best=0.0
    model.train()
    for ep in range(cfg.sanity_epochs):
        meter=PixelMeter()
        for x,y,lv,coords,ids in dl:
            x=x.to(device); y=y.to(device); coords=coords.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda",enabled=cfg.amp and device.type=="cuda"):
                z=model(x,coords); loss=lossfn(z,y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); meter.update(z.detach(),y,cfg.threshold)
        best=max(best,meter.metrics()["dice"])
        if best>=cfg.sanity_required_dice: break
    del model; gc.collect();
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return best


def step_sanity(cfg: Config):
    if not (cfg.qc_dir/"07_dataloader.json").exists(): step_qc_dataloader(cfg)
    tr=pd.read_csv(cfg.selected_train_csv)
    # Ensure small set contains both categories and multiple levels.
    parts=[]
    per=max(1,cfg.sanity_subset//8)
    for lv in cfg.levels:
        for cat in ["normal","tumor"]:
            b=tr[(tr.level==lv)&(tr.category==cat)]
            if len(b): parts.append(b.sample(min(per,len(b)),random_state=cfg.seed+lv))
    small=pd.concat(parts,ignore_index=True).head(cfg.sanity_subset)
    repairs=[]
    score=run_sanity_once(cfg,small,normalize=True,augment=False)
    if score<cfg.sanity_required_dice:
        # Safe diagnostic repair: retry without Vahadane. If this fixes it, normalization is suspect.
        score2=run_sanity_once(cfg,small,normalize=False,augment=False)
        repairs.append(f"Initial sanity Dice {score:.4f} below target; automatically retried without Vahadane for diagnosis -> {score2:.4f}")
        score=max(score,score2)
    passed=score>=cfg.sanity_required_dice
    qc_report(cfg,"08_sanity",passed,{"best_small_set_dice":score,"required":cfg.sanity_required_dice,"n":len(small)},repairs)


# =============================================================================
# Training
# =============================================================================


def evaluate_loader(model, dl, device, cfg: Config) -> Dict[str,Any]:
    model.eval(); meter=PixelMeter(); lossfn=CombinedLoss(); loss_sum=0.; n=0
    with torch.inference_mode():
        for x,y,lv,coords,ids in dl:
            x=x.to(device,non_blocking=True); y=y.to(device,non_blocking=True); coords=coords.to(device,non_blocking=True)
            if cfg.channels_last and device.type=="cuda": x=x.contiguous(memory_format=torch.channels_last)
            with torch.amp.autocast("cuda",enabled=cfg.amp and device.type=="cuda"):
                z=model(x,coords); loss=lossfn(z,y)
            loss_sum+=float(loss.item())*len(x); n+=len(x); meter.update(z,y,cfg.threshold)
    m=meter.metrics(); m["loss"]=loss_sum/max(n,1); m["n_patches"]=n; return m


def train_model(cfg: Config, run_name: Optional[str]=None) -> Path:
    tr,va,te=build_frames(cfg)
    train_ds=PandaPatchDataset(tr,cfg,"train",augment=cfg.use_strong_stain_aug,normalize=True)
    val_ds=PandaPatchDataset(va,cfg,"val",augment=False,normalize=True)
    train_dl=make_loader(train_ds,cfg,shuffle=True); val_dl=make_loader(val_ds,cfg,shuffle=False)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=UNetEfficientNetB4(1,cfg.dropout_p,cfg.use_coords,cfg.pretrained,cfg.coord_embed_dim).to(device)
    if cfg.channels_last and device.type=="cuda": model=model.to(memory_format=torch.channels_last)
    opt=torch.optim.AdamW(model.parameters(),lr=cfg.learning_rate,weight_decay=cfg.weight_decay)
    lossfn=CombinedLoss(); scaler=torch.amp.GradScaler("cuda",enabled=cfg.amp and device.type=="cuda")
    run_name=run_name or time.strftime("panda_multilevel_%Y%m%d_%H%M%S")
    out=cfg.runs_dir/run_name; out.mkdir(parents=True,exist_ok=True)
    atomic_json(out/"config.json",asdict(cfg))
    best=-1.; bad=0; history=[]
    for ep in range(1,cfg.num_epochs+1):
        model.train(); meter=PixelMeter(); loss_sum=0.; n=0
        bar=tqdm(train_dl,desc=f"TRAIN epoch {ep}/{cfg.num_epochs}")
        for x,y,lv,coords,ids in bar:
            x=x.to(device,non_blocking=True); y=y.to(device,non_blocking=True); coords=coords.to(device,non_blocking=True)
            if cfg.channels_last and device.type=="cuda": x=x.contiguous(memory_format=torch.channels_last)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda",enabled=cfg.amp and device.type=="cuda"):
                z=model(x,coords); loss=lossfn(z,y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            loss_sum+=float(loss.item())*len(x); n+=len(x); meter.update(z.detach(),y,cfg.threshold)
            bar.set_postfix(loss=f"{loss.item():.4f}")
        tm=meter.metrics(); tm["loss"]=loss_sum/max(n,1); tm["n_patches"]=n
        vm=evaluate_loader(model,val_dl,device,cfg)
        row={"epoch":ep,**{f"train_{k}":v for k,v in tm.items()},**{f"val_{k}":v for k,v in vm.items()}}
        history.append(row); pd.DataFrame(history).to_csv(out/"history.csv",index=False)
        state={"epoch":ep,"model":model.state_dict(),"optimizer":opt.state_dict(),"val":vm,"config":asdict(cfg)}
        torch.save(state,out/"last.pt")
        print(f"Epoch {ep}: train Dice={tm['dice']:.4f} | val Dice={vm['dice']:.4f}")
        if vm["dice"]>best:
            best=vm["dice"]; bad=0; torch.save(state,out/"best.pt")
        else:
            bad+=1
            if bad>=cfg.early_stopping_patience:
                print("Early stopping")
                break
    atomic_json(cfg.work/"latest_run.json",{"run_dir":str(out),"best_val_dice":best})
    return out


def step_train(cfg: Config):
    if not (cfg.qc_dir/"08_sanity.json").exists(): step_sanity(cfg)
    run=train_model(cfg)
    qc_report(cfg,"09_train",(run/"best.pt").exists(),{"run_dir":str(run),"best_checkpoint":str(run/"best.pt")})


# =============================================================================
# Final test: L1, L2, L3, L4 separately + pooled ALL levels
# =============================================================================


def load_best_model(cfg: Config, checkpoint: Optional[str]=None):
    if checkpoint is None:
        latest=json.load(open(cfg.work/"latest_run.json","r",encoding="utf-8")); checkpoint=str(Path(latest["run_dir"])/"best.pt")
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=UNetEfficientNetB4(1,cfg.dropout_p,cfg.use_coords,False,cfg.coord_embed_dim).to(device)
    ck=torch.load(checkpoint,map_location=device); model.load_state_dict(ck["model"]); model.eval()
    return model,device,checkpoint


def test_frame(cfg: Config, model, device, frame: pd.DataFrame) -> Dict[str,Any]:
    ds=PandaPatchDataset(frame,cfg,"test",augment=False,normalize=True)
    dl=make_loader(ds,cfg,shuffle=False)
    return evaluate_loader(model,dl,device,cfg)


def step_test(cfg: Config, checkpoint: Optional[str]=None):
    if not cfg.patch_manifest.exists(): raise QCError("Patch manifest missing")
    model,device,checkpoint=load_best_model(cfg,checkpoint)
    df=pd.read_csv(cfg.patch_manifest); test=df[df.split=="test"].copy()
    results={}
    for lv in cfg.levels:
        sub=test[test.level==lv].copy()
        if len(sub)==0: results[f"level_{lv}"]={"error":"no test patches"}; continue
        results[f"level_{lv}"]=test_frame(cfg,model,device,sub)
    results["all_levels_pooled"]=test_frame(cfg,model,device,test)
    out=Path(checkpoint).parent/"test_per_level_and_pooled.json"; atomic_json(out,results)
    table=[]
    for name,m in results.items():
        if "error" in m: continue
        table.append({"subset":name,**m})
    pd.DataFrame(table).to_csv(Path(checkpoint).parent/"test_per_level_and_pooled.csv",index=False)
    passed=all(f"level_{lv}" in results and "error" not in results[f"level_{lv}"] for lv in cfg.levels)
    qc_report(cfg,"10_test",passed,{"checkpoint":checkpoint,"results":results,"csv":str(Path(checkpoint).parent/"test_per_level_and_pooled.csv")})


# =============================================================================
# Optional full-WSI inference with one shared model over all 4 scales
# =============================================================================


def infer_wsi(cfg: Config, wsi_path: str, output_dir: str, checkpoint: Optional[str]=None):
    model,device,checkpoint=load_best_model(cfg,checkpoint)
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    im=SlideScaleReader(wsi_path)
    # Low-res output grid at logical L4 resolution to keep memory bounded.
    out_ds=cfg.logical_downsamples[4]
    W=max(1,int(math.ceil(im.dimensions[0]/out_ds))); H=max(1,int(math.ceil(im.dimensions[1]/out_ds)))
    sum_prob=np.zeros((H,W),np.float32); sum_w=np.zeros((H,W),np.float32)
    target=np.array(Image.open(cfg.stain_target_path).convert("RGB")); norm=VahadaneProcessor(target)
    for lv in cfg.levels:
        ds=cfg.logical_downsamples[lv]; field0=int(cfg.patch_size*ds); stride0=int(cfg.stride*ds)
        for y0 in tqdm(range(0,max(1,im.dimensions[1]-field0+1),stride0),desc=f"INFER L{lv}"):
            for x0 in range(0,max(1,im.dimensions[0]-field0+1),stride0):
                rgb,_=im.read_at_downsample(x0,y0,ds,cfg.patch_size)
                # cheap tissue gate at patch level
                tm,_=generate_tissue_mask(rgb,cfg)
                if float((tm>0).mean())<cfg.tissue_min_frac: continue
                rgb=norm.transform(rgb)
                x=torch.from_numpy(rgb.copy()).permute(2,0,1).float().unsqueeze(0).to(device)/255.
                coords=torch.tensor([[(x0+field0/2)/im.dimensions[0],(y0+field0/2)/im.dimensions[1]]],dtype=torch.float32,device=device)
                with torch.inference_mode(), torch.amp.autocast("cuda",enabled=cfg.amp and device.type=="cuda"):
                    prob=torch.sigmoid(model(x,coords))[0,0].cpu().numpy()
                # resize patch probability to L4 output physical footprint
                ow=max(1,int(round(field0/out_ds))); oh=ow
                pr=cv2.resize(prob,(ow,oh),interpolation=cv2.INTER_LINEAR)
                ox=int(round(x0/out_ds)); oy=int(round(y0/out_ds)); x2=min(W,ox+ow); y2=min(H,oy+oh)
                if x2>ox and y2>oy:
                    sum_prob[oy:y2,ox:x2]+=pr[:y2-oy,:x2-ox]; sum_w[oy:y2,ox:x2]+=1
    avg=sum_prob/np.maximum(sum_w,1e-6); mask=(avg>=cfg.threshold).astype(np.uint8)*255
    cv2.imwrite(str(out/"tumor_probability.png"),(np.clip(avg,0,1)*255).astype(np.uint8))
    cv2.imwrite(str(out/"tumor_mask.png"),mask)
    thumb=im.slide.get_thumbnail((W,H)).convert("RGB").resize((W,H)); rgb=np.array(thumb)
    ov=rgb.copy(); m=mask>0; ov[m]=(0.6*ov[m]+0.4*np.array([255,0,0])).astype(np.uint8); Image.fromarray(ov).save(out/"overlay.png")
    im.close()
    atomic_json(out/"summary.json",{"checkpoint":checkpoint,"levels":list(cfg.levels),"threshold":cfg.threshold,"tumor_fraction_output_grid":float((mask>0).mean())})
    print(f"Saved inference outputs to {out}")


# =============================================================================
# Orchestrator
# =============================================================================

STEPS = {
    "audit": step_audit,
    "split": step_split,
    "qc-alignment": step_qc_alignment,
    "extract": step_extract,
    "select": step_select,
    "qc-normalization": step_qc_normalization,
    "qc-dataloader": step_qc_dataloader,
    "sanity": step_sanity,
    "train": step_train,
    "test": step_test,
}


def print_plan(cfg: Config):
    print("\n=== PANDA CAMELYON-STYLE PLAN ===")
    print("1. audit             -> WSI/mask existence, levels, provider mask values")
    print("2. split             -> 70/15/15 slide-level; leakage must be zero")
    print("3. qc-alignment      -> RGB/raw mask/binary tumor overlay")
    print("4. extract           -> 512x512, stride 128, logical L1..L4, tissue filtering")
    print("5. select            -> 8 buckets, EfficientNet-B4 features, KMeans, equal sampling")
    print("6. qc-normalization  -> Vahadane target from TRAIN only + before/after checks")
    print("7. qc-dataloader     -> tensors/masks/coords/augmentation alignment")
    print("8. sanity            -> small-set overfit gate")
    print("9. train             -> ONE UNet-EfficientNetB4 on mixed 4-level dataset")
    print("10. test             -> same model on L1/L2/L3/L4 separately + all levels pooled")
    print("================================\n")


def parse_args():
    p=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("command",choices=list(STEPS)+["all","plan","infer-wsi"])
    p.add_argument("--data-root",default=None)
    p.add_argument("--work-root",default=None)
    p.add_argument("--checkpoint",default=None)
    p.add_argument("--wsi",default=None)
    p.add_argument("--output",default=None)
    return p.parse_args()


def main():
    a=parse_args(); cfg=Config()
    if a.data_root: cfg.data_root=a.data_root
    if a.work_root: cfg.work_root=a.work_root
    cfg.ensure_dirs(); seed_everything(cfg.seed); setup_torch_speed(); print_plan(cfg)
    atomic_json(cfg.work/"effective_config.json",asdict(cfg))
    if a.command=="plan": return
    if a.command=="all":
        for name in ["audit","split","qc-alignment","extract","select","qc-normalization","qc-dataloader","sanity","train","test"]:
            print(f"\n\n######## {name.upper()} ########")
            if name=="test": step_test(cfg,a.checkpoint)
            else: STEPS[name](cfg)
        return
    if a.command=="infer-wsi":
        if not a.wsi or not a.output: raise SystemExit("infer-wsi requires --wsi and --output")
        infer_wsi(cfg,a.wsi,a.output,a.checkpoint); return
    if a.command=="test": step_test(cfg,a.checkpoint)
    else: STEPS[a.command](cfg)


if __name__=="__main__":
    main()
