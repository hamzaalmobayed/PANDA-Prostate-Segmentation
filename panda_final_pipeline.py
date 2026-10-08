
from __future__ import annotations

"""
PANDA prostate segmentation — FINAL ONE-COMMAND PIPELINE

Design locked for this project
------------------------------
- Input root defaults to D:\prostat
- Work root defaults to D:\PANDA_PROSTATE
- Mapping is by train.csv image_id, NEVER by file counts/order.
- WSI without a matching *_mask.tiff is excluded from segmentation and reported.
- Strict slide-level 70/15/15 split BEFORE patch extraction.
- PANDA native levels: L0, L1, L2.
- 512x512 patches, stride 128.
- Training candidate patches are physically saved in:
    patches/train/<image_id>/L0|L1|L2/Tumor|Normal/
  with image JPEG + binary mask PNG.
- Validation and Test also use small fixed saved classified patch subsets so all three split folders share the same structure.
- Train / Validation / Test all receive the same folder hierarchy for saved classified patches.
- Test WSI evaluation is still streamed densely from retained original WSIs so WSI reconstruction is complete.
- Train WSIs are deleted ONLY after:
    extraction complete + 6 buckets sufficient for final 80k + patch QC PASS.
  Validation/Test WSIs are retained.
- Train masks are retained.
- 80,000 final selected training patches, balanced 40k Tumor / 40k Normal
  and as evenly as possible across 3 levels.
- EfficientNet-B4 features + KMeans within each of 6 buckets.
- TIAToolbox-compatible CAMELYON components are retained:
    VahadaneNormalizer, WSIReader/tissue_mask where available,
    PairAugStainHeavy, U-Net + EfficientNet-B4, BCE+Dice, AdamW, AMP.
- Pilot ablations run before unselected candidate patches are pruned.
- Resume/self-healing state file allows the SAME command to resume.

Run:
    python run_full_panda_paper.py

Optional override:
    python run_full_panda_paper.py --data-root "D:\prostat" --work-root "D:\PANDA_PROSTATE"
"""

import gc
import json
import math
import os
import random
import shutil
import time
import traceback
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Tuple, List, Any, Optional

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch

import panda_camelyon_style_pipeline as core
import panda_paper_suite_legacy as legacy_paper

try:
    from tiatoolbox.wsicore.wsireader import WSIReader
except Exception:
    WSIReader = None


@dataclass
class FinalConfig(core.Config):
    data_root: str = r"D:\prostat"
    work_root: str = r"D:\PANDA_PROSTATE"

    # PANDA native pyramid levels.
    levels: Tuple[int, ...] = (0, 1, 2)
    logical_downsamples: Dict[int, float] = field(
        default_factory=lambda: {0: 1.0, 1: 4.0, 2: 16.0, 3: 64.0}
    )
    tissue_level: int = 2

    # Same CAMELYON patch geometry.
    patch_size: int = 512
    stride: int = 128
    tumor_threshold: float = 0.05
    # Pure Normal: zero tumor pixels in annotated tissue.
    # Borderline patches with 0 < tumor_fraction < tumor_threshold are excluded from the six sampling buckets.
    exclude_mixed_borderline: bool = True
    min_valid_annotated_fraction: float = 0.02

    # Final requested training set.
    num_samples: int = 80000

    # Disk-aware candidate extraction.
    # First save 4 patches per class / slide / level.
    # If a global bucket is short, safely expand 8 -> 16 -> 32 BEFORE deleting train WSIs.
    train_candidate_caps: Tuple[int, ...] = (4, 8, 16, 32)
    val_saved_cap_per_class_slide_level: int = 2
    test_saved_cap_per_class_slide_level: int = 2
    coordinate_pool_cap_per_class_slide_level: int = 32
    patch_jpeg_quality: int = 92

    # Fine-grained durable resume controls.
    coordinate_checkpoint_every_positions: int = 5000
    # Heavy optimizer/model checkpoint. Lower = less redo after a full PC power loss,
    # but more SSD writes. Validation itself checkpoints lightweight metrics every batch.
    train_checkpoint_every_batches: int = 100
    storage_retry_seconds: int = 10

    # KMeans feature candidate cap per 6-way bucket.
    feature_candidate_cap_per_bucket: int = 25000

    # Never delete masks automatically.
    delete_train_wsi_after_verified_extraction: bool = True
    keep_all_masks: bool = True

    # Paper WSI evaluation grid (baseline downsample).
    paper_eval_downsample: float = 16.0

    @property
    def candidate_manifest(self) -> Path:
        return self.manifests_dir / "candidate_patches_train_val.csv"

    @property
    def coordinate_pool_dir(self) -> Path:
        return self.manifests_dir / "coordinate_pools"

    @property
    def excluded_missing_masks_csv(self) -> Path:
        return self.manifests_dir / "excluded_missing_masks.csv"

    @property
    def deleted_train_wsi_csv(self) -> Path:
        return self.manifests_dir / "deleted_train_wsi.csv"

    @property
    def final_training_manifest(self) -> Path:
        return self.manifests_dir / "final_train_val_manifest.csv"

    def ensure_dirs(self):
        super().ensure_dirs()
        self.coordinate_pool_dir.mkdir(parents=True, exist_ok=True)


def bucket_targets(cfg: FinalConfig) -> Dict[Tuple[int, str], int]:
    """Exactly 80,000 total; exactly 40,000 normal + 40,000 tumor."""
    out = {}
    for cat in ("normal", "tumor"):
        total = cfg.num_samples // 2
        base = total // len(cfg.levels)
        rem = total % len(cfg.levels)
        for i, lv in enumerate(cfg.levels):
            out[(lv, cat)] = base + (1 if i < rem else 0)
    assert sum(out.values()) == cfg.num_samples
    return out


def _class_name(cat: str) -> str:
    return "Tumor" if cat.lower() == "tumor" else "Normal"


def patch_paths(cfg: FinalConfig, split: str, image_id: str, level: int,
                category: str, x0: int, y0: int) -> Tuple[Path, Path]:
    """
    Reconstruction-safe naming.

    x0/y0 are ALWAYS baseline (level-0) WSI coordinates, not local patch coordinates.
    Filename carries split, slide id, logical level, baseline coordinate, target downsample,
    patch size, stride, and class.  Tumor masks are stored in a separate sibling folder.
    """
    cls = _class_name(category)
    level_dir = cfg.patches_dir / split / str(image_id) / f"L{level}"
    img_dir = level_dir / cls
    mask_dir = level_dir / f"{cls}_Masks"

    ds = float(cfg.logical_downsamples[level])
    ds_token = str(int(ds)) if float(ds).is_integer() else str(ds).replace(".", "p")
    stem = (
        f"{image_id}"
        f"__split-{split}"
        f"__L{level}"
        f"__x0-{int(x0)}"
        f"__y0-{int(y0)}"
        f"__ds-{ds_token}"
        f"__ps-{int(cfg.patch_size)}"
        f"__stride-{int(cfg.stride)}"
        f"__class-{cls}"
    )
    return img_dir / f"{stem}.jpg", mask_dir / f"{stem}__tumor_mask.png"


def valid_mask_path(mask_path: Path) -> Path:
    """
    Put the PANDA annotated-tissue/ignore mask in its own sibling folder.
    Example:
      L0/Normal_Masks/<stem>__tumor_mask.png
      L0/Normal_Valid_Masks/<stem>__valid_mask.png
    """
    mask_path = Path(mask_path)
    parent_name = mask_path.parent.name
    if parent_name.endswith("_Masks"):
        cls = parent_name[:-6]
        valid_dir = mask_path.parent.parent / f"{cls}_Valid_Masks"
    else:
        valid_dir = mask_path.parent / "Valid_Masks"
    name = mask_path.name.replace("__tumor_mask.png", "__valid_mask.png")
    return valid_dir / name


def ensure_full_patch_tree(cfg: FinalConfig, split_df: pd.DataFrame):
    """
    Pre-create the complete folder tree for every split / image / level.

    Each level has:
      Normal/
      Normal_Masks/
      Normal_Valid_Masks/
      Tumor/
      Tumor_Masks/
      Tumor_Valid_Masks/
    """
    for _, r in split_df.iterrows():
        for lv in cfg.levels:
            level_dir = cfg.patches_dir / str(r.split) / str(r.image_id) / f"L{lv}"
            for cls in ("Normal", "Tumor"):
                for sub in (cls, f"{cls}_Masks", f"{cls}_Valid_Masks"):
                    (level_dir / sub).mkdir(parents=True, exist_ok=True)


def _tia_tissue_mask_thumbnail(image_path: str, level: int) -> Optional[np.ndarray]:
    """Use the same TIAToolbox WSIReader/tissue_mask family used by CAMELYON when available."""
    if WSIReader is None:
        return None
    try:
        reader = WSIReader.open(input_img=image_path)
        mask_reader = reader.tissue_mask("morphological", resolution=level, units="level")
        m = mask_reader.slide_thumbnail(resolution=level, units="level")
        if m.ndim == 3:
            m = m[..., 0]
        return (m > 0).astype(np.uint8)
    except Exception:
        return None


def _lowres_masks(cfg: FinalConfig, r: pd.Series, level: int):
    """
    Return tissue and tumor thumbnails plus effective baseline scaling.
    TIAToolbox morphological tissue mask is preferred; CAMELYON-style fallback is used.
    """
    im = core.SlideScaleReader(r.image_path)
    ds = float(cfg.logical_downsamples[level])
    # Keep low-resolution thumbnails bounded.
    thumb, meta = im.thumbnail_at_downsample(max(ds, 16.0), cfg.target_max_size)
    eff_ds = float(meta.get("effective_ds", max(ds, 16.0)))

    tia_mask = _tia_tissue_mask_thumbnail(str(r.image_path), min(level, 2))
    if tia_mask is not None:
        tissue = cv2.resize(tia_mask.astype(np.uint8),
                            (thumb.shape[1], thumb.shape[0]),
                            interpolation=cv2.INTER_NEAREST)
    else:
        tissue, _ = core.generate_tissue_mask(thumb, cfg)

    # Provider label mask thumbnail.
    mr = core.MaskScaleReader(r.mask_path, im.dimensions)
    mthumb = mr.reader.slide.get_thumbnail((thumb.shape[1], thumb.shape[0])).convert("RGB")
    raw = np.asarray(mthumb)
    tumor = core.raw_to_binary_mask(raw, str(r.provider))
    valid = core.raw_valid_tissue_mask(raw, str(r.provider))
    if tumor.shape != tissue.shape:
        tumor = cv2.resize(tumor, (tissue.shape[1], tissue.shape[0]), interpolation=cv2.INTER_NEAREST)
        valid = cv2.resize(valid, (tissue.shape[1], tissue.shape[0]), interpolation=cv2.INTER_NEAREST)
    im.close(); mr.close()
    return tissue.astype(np.uint8), (tumor > 0).astype(np.uint8), valid.astype(np.uint8), eff_ds


def _fraction_lowres(mask: np.ndarray, x0: int, y0: int, field0: int, eff_ds: float) -> float:
    x1 = max(0, int(math.floor(x0 / eff_ds)))
    y1 = max(0, int(math.floor(y0 / eff_ds)))
    x2 = min(mask.shape[1], int(math.ceil((x0 + field0) / eff_ds)))
    y2 = min(mask.shape[0], int(math.ceil((y0 + field0) / eff_ds)))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return float(mask[y1:y2, x1:x2].mean())


def build_coordinate_pool(cfg: FinalConfig, r: pd.Series, level: int) -> pd.DataFrame:
    """
    Scan tissue grid but retain only a bounded coordinate pool for each class.
    No patch image is written in this phase.
    """
    out_csv = cfg.coordinate_pool_dir / f"{r.image_id}_L{level}.csv"
    if out_csv.exists():
        return pd.read_csv(out_csv)

    im = core.SlideScaleReader(r.image_path)
    w0, h0 = im.dimensions
    ds = float(cfg.logical_downsamples[level])
    field0 = int(round(cfg.patch_size * ds))
    stride0 = int(round(cfg.stride * ds))
    tissue, tumor_lr, valid_lr, eff_ds = _lowres_masks(cfg, r, level)

    pools = {"normal": [], "tumor": []}
    rng = np.random.default_rng(cfg.seed + abs(hash((str(r.image_id), level))) % 1_000_000)
    seen = {"normal": 0, "tumor": 0}
    cap = cfg.coordinate_pool_cap_per_class_slide_level

    # Reservoir sampling: memory and disk stay bounded even if a slide contains millions of grid positions.
    for y0 in range(0, max(1, h0 - field0 + 1), stride0):
        for x0 in range(0, max(1, w0 - field0 + 1), stride0):
            tissue_frac = _fraction_lowres(tissue, x0, y0, field0, eff_ds)
            if tissue_frac < cfg.tissue_min_frac:
                continue
            valid_frac = _fraction_lowres(valid_lr, x0, y0, field0, eff_ds)
            # PANDA 0 is background/unknown: candidates need annotated tissue.
            if valid_frac < cfg.min_valid_annotated_fraction:
                continue
            tumor_frac_lr = _fraction_lowres(tumor_lr, x0, y0, field0, eff_ds)
            if tumor_frac_lr >= cfg.tumor_threshold:
                cat = "tumor"
            elif tumor_frac_lr <= 0.0:
                cat = "normal"
            else:
                # Mixed/borderline patch: not part of the six balanced sampling buckets.
                continue
            seen[cat] += 1
            item = {
                "image_id": str(r.image_id), "provider": str(r.provider), "split": str(r.split),
                "level": int(level), "x0": int(x0), "y0": int(y0),
                "tissue_fraction_lr": float(tissue_frac),
                "tumor_fraction_lr": float(tumor_frac_lr),
            }
            if len(pools[cat]) < cap:
                pools[cat].append(item)
            else:
                j = int(rng.integers(0, seen[cat]))
                if j < cap:
                    pools[cat][j] = item
    im.close()

    df = pd.DataFrame(pools["normal"] + pools["tumor"])
    if len(df):
        df["category_lr"] = np.where(df.tumor_fraction_lr >= cfg.tumor_threshold, "tumor", "normal")
    df.to_csv(out_csv, index=False)
    return df


def materialize_from_pool(cfg: FinalConfig, r: pd.Series, level: int,
                          desired_per_class: int) -> List[Dict[str, Any]]:
    pool = build_coordinate_pool(cfg, r, level)
    if len(pool) == 0:
        return []

    im = core.SlideScaleReader(r.image_path)
    mr = core.MaskScaleReader(r.mask_path, im.dimensions)
    ds = float(cfg.logical_downsamples[level])
    w0, h0 = im.dimensions
    field0 = int(round(cfg.patch_size * ds))
    records = []

    for cat in ("normal", "tumor"):
        sub = pool[pool.category_lr == cat].copy()
        if len(sub) == 0:
            continue
        sub = sub.sort_values(["tissue_fraction_lr", "y0", "x0"],
                              ascending=[False, True, True]).head(desired_per_class * 3)

        accepted_for_cat = 0
        for _, c in sub.iterrows():
            if accepted_for_cat >= desired_per_class:
                break

            x0, y0 = int(c.x0), int(c.y0)
            rgb, smeta = im.read_at_downsample(x0, y0, ds, cfg.patch_size)
            raw, _ = mr.read_raw(x0, y0, ds, cfg.patch_size)

            binm = core.raw_to_binary_mask(raw, str(r.provider))
            validm = core.raw_valid_tissue_mask(raw, str(r.provider))
            valid_fraction = float((validm > 0).mean())
            if valid_fraction < cfg.min_valid_annotated_fraction:
                continue

            tumor_fraction = float(((binm > 0) & (validm > 0)).sum() / max(1, (validm > 0).sum()))

            if tumor_fraction >= cfg.tumor_threshold:
                exact_cat = "tumor"
            elif tumor_fraction <= 0.0:
                exact_cat = "normal"
            else:
                # Scientifically ambiguous for the 6-bucket balancing scheme.
                # Keep it out of Normal and Tumor buckets.
                continue

            # Candidate may have changed class when exact high-resolution GT is read.
            if exact_cat != cat:
                continue

            ip, mp = patch_paths(cfg, str(r.split), str(r.image_id), level, exact_cat, x0, y0)
            vp = valid_mask_path(mp)
            ip.parent.mkdir(parents=True, exist_ok=True)
            mp.parent.mkdir(parents=True, exist_ok=True)
            vp.parent.mkdir(parents=True, exist_ok=True)

            if not ip.exists():
                Image.fromarray(rgb).save(ip, "JPEG", quality=cfg.patch_jpeg_quality, subsampling=0)
            if not mp.exists():
                Image.fromarray(binm.astype(np.uint8)).save(mp, "PNG", compress_level=3)
            if not vp.exists():
                Image.fromarray((validm.astype(np.uint8) * 255)).save(vp, "PNG", compress_level=3)

            cx0 = x0 + field0 / 2.0
            cy0 = y0 + field0 / 2.0
            records.append({
                "image_id": str(r.image_id), "provider": str(r.provider), "split": str(r.split),
                "level": int(level), "label": int(exact_cat == "tumor"), "category": exact_cat,
                "x0": x0, "y0": y0,
                "coord_x": float(cx0 / max(w0, 1)), "coord_y": float(cy0 / max(h0, 1)),
                "target_downsample": ds,
                "native_level": int(smeta["native_level"]),
                "native_downsample": float(smeta["native_downsample"]),
                "virtual_scale": bool(smeta["virtual_scale"]),
                "tissue_fraction": float(c.tissue_fraction_lr),
                "valid_annotated_fraction": valid_fraction,
                "tumor_fraction": tumor_fraction,
                "img": str(ip), "mask": str(mp), "valid_mask": str(vp),
            })
            accepted_for_cat += 1

    im.close(); mr.close()
    return records


def _dedupe_manifest(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) == 0:
        return df
    return df.drop_duplicates(["split","image_id","level","x0","y0"], keep="last").reset_index(drop=True)


def _verify_saved_patches(cfg: FinalConfig, df: pd.DataFrame) -> Tuple[bool, Dict[str, Any]]:
    errs = []
    if len(df) == 0:
        return False, {"errors": ["empty manifest"]}
    sample = df.sample(min(100, len(df)), random_state=cfg.seed)
    for _, r in sample.iterrows():
        ip, mp = Path(r["img"]), Path(r["mask"])
        vp = Path(str(r["valid_mask"])) if "valid_mask" in r.index else valid_mask_path(mp)
        if not ip.exists() or not mp.exists() or not vp.exists():
            errs.append(f"missing:{ip}|{mp}|{vp}")
            continue
        im = cv2.imread(str(ip), cv2.IMREAD_COLOR)
        ma = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        va = cv2.imread(str(vp), cv2.IMREAD_GRAYSCALE)
        if im is None or ma is None or va is None:
            errs.append(f"unreadable:{ip}|{mp}|{vp}")
            continue
        if im.shape[:2] != (cfg.patch_size, cfg.patch_size):
            errs.append(f"image_shape:{ip}:{im.shape}")
        if ma.shape != (cfg.patch_size, cfg.patch_size) or va.shape != (cfg.patch_size, cfg.patch_size):
            errs.append(f"mask_shape:{mp}:{ma.shape}|valid:{va.shape}")
        if not set(np.unique(ma).tolist()).issubset({0,255}):
            errs.append(f"mask_values:{mp}:{np.unique(ma).tolist()}")
        if not set(np.unique(va).tolist()).issubset({0,255}):
            errs.append(f"valid_mask_values:{vp}:{np.unique(va).tolist()}")
        valid = va > 127
        tf = float(((ma > 127) & valid).sum() / max(1, valid.sum()))
        if str(r.category).lower() == "normal" and tf != 0.0:
            errs.append(f"normal_contains_tumor:{ip}:{tf}")
        if str(r.category).lower() == "tumor" and tf < cfg.tumor_threshold:
            errs.append(f"tumor_below_threshold:{ip}:{tf}")
    return not errs, {"sample_errors": errs[:50]}


def step_audit_final(cfg: FinalConfig):
    """
    Core audit already maps using train.csv IDs. Add explicit missing-mask export.
    """
    core.step_audit(cfg)
    audit = pd.read_csv(cfg.audit_csv)
    missing = audit[~audit["mask_exists"].astype(bool)].copy() if "mask_exists" in audit.columns else audit.iloc[0:0]
    missing.to_csv(cfg.excluded_missing_masks_csv, index=False)

    # Segmentation cohort = only WSI+mask valid pairs.
    if "valid_for_segmentation" in audit.columns:
        valid_n = int(audit.valid_for_segmentation.astype(bool).sum())
    else:
        valid_n = int((audit.image_exists.astype(bool) & audit.mask_exists.astype(bool)).sum())
    # Scientific gate: verify the actual WSI pyramid rather than blindly assuming 1/4/16.
    scale_rows = []
    bad_scale = []
    for _, rr in audit[audit.get("valid", False) == True].iterrows():
        try:
            downs = json.loads(rr.image_downsamples) if isinstance(rr.image_downsamples, str) else []
        except Exception:
            downs = []
        rec = {"image_id": str(rr.image_id), "level_count": int(rr.get("image_level_count", 0)),
               "downsamples": json.dumps(downs)}
        scale_rows.append(rec)
        if len(downs) < 3:
            bad_scale.append({"image_id": str(rr.image_id), "reason": "fewer_than_3_native_levels", "downsamples": downs})
        else:
            expected = [1.0, 4.0, 16.0]
            rel_err = [abs(float(downs[i])-expected[i])/expected[i] for i in range(3)]
            if any(e > 0.25 for e in rel_err):
                bad_scale.append({"image_id": str(rr.image_id), "reason": "unexpected_native_downsample",
                                  "downsamples": downs[:3], "relative_error": rel_err})
    pd.DataFrame(scale_rows).to_csv(cfg.qc_dir / "01c_native_pyramid_audit.csv", index=False)
    if bad_scale:
        core.qc_report(cfg, "01c_native_pyramid", False, {
            "bad_count": len(bad_scale), "examples": bad_scale[:50],
            "rule": "Require >=3 native levels approximately 1x/4x/16x before using L0/L1/L2."
        })
    else:
        core.qc_report(cfg, "01c_native_pyramid", True, {
            "checked_slides": len(scale_rows),
            "rule": ">=3 native levels; first three downsamples within 25% of 1x/4x/16x."
        })

    core.atomic_json(cfg.qc_dir / "01b_mapping_summary.json", {
        "train_csv_rows": int(len(audit)),
        "valid_segmentation_pairs": valid_n,
        "missing_masks": int(len(missing)),
        "missing_masks_csv": str(cfg.excluded_missing_masks_csv),
        "mapping_rule": "train.csv image_id -> train_images/<id>.tiff and train_label_masks/<id>_mask.tiff",
        "native_pyramid_qc": str(cfg.qc_dir / "01c_native_pyramid.json"),
    })


def step_split_final(cfg: FinalConfig):
    core.step_split(cfg)


def step_alignment_final(cfg: FinalConfig):
    core.step_qc_alignment(cfg)


def _materialize_split(cfg: FinalConfig, split_df: pd.DataFrame, split: str, cap: int) -> pd.DataFrame:
    """
    Materialize a split and print a human-readable result after EVERY WSI.

    Example:
      WSI 12/7358 | image_id=abc...
        L0 Normal=4 Tumor=3
        L1 Normal=4 Tumor=1
        L2 Normal=4 Tumor=0
        TOTAL SAVED=16
        EMPTY REASON L2 Tumor: no eligible tumor candidates
    """
    rows = []
    sdf = split_df[split_df.split == split].reset_index(drop=True)

    log_dir = cfg.work / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_csv = log_dir / f"extraction_{split}_cap{cap}.csv"

    old_log = pd.read_csv(log_csv) if log_csv.exists() else pd.DataFrame()
    completed_ids = set(old_log.image_id.astype(str)) if len(old_log) and "image_id" in old_log.columns else set()
    log_rows = old_log.to_dict("records") if len(old_log) else []

    bar = tqdm(total=len(sdf), desc=f"PATCHES {split} cap={cap}")

    # If previous log exists, reflect those WSI as already completed in the progress bar.
    if completed_ids:
        bar.update(sum(str(x) in completed_ids for x in sdf.image_id.astype(str)))

    for idx, r in sdf.iterrows():
        image_id = str(r.image_id)

        if image_id in completed_ids:
            # Reuse existing manifest shards through resumable materializer only when needed by caller.
            continue

        level_summary = {}
        wsi_records = []

        # One OpenSlide image+mask session + one shared low-res mask set for all 3 levels.
        try:
            import panda_resilience as _resilience
            _resilience.begin_wsi_session(cfg, r)

            for lv in cfg.levels:
                recs = materialize_from_pool(cfg, r, lv, cap)
                wsi_records.extend(recs)

                n_normal = sum(1 for x in recs if str(x.get("category","")).lower() == "normal")
                n_tumor = sum(1 for x in recs if str(x.get("category","")).lower() == "tumor")

                reason_path = cfg.manifests_dir / "materialize_shards" / f"{split}_{image_id}_L{lv}_cap{cap}.reasons.json"
                reasons = {}
                if reason_path.exists():
                    try:
                        reasons = json.load(open(reason_path, "r", encoding="utf-8"))
                    except Exception:
                        reasons = {}

                level_summary[lv] = {
                    "normal": int(n_normal),
                    "tumor": int(n_tumor),
                    "reason_if_zero": reasons.get("reason_if_zero"),
                    "coordinate_stats": reasons.get("coordinate_stats", {}),
                    "exact_rejection_counts": reasons.get("exact_rejection_counts", {}),
                }
        finally:
            try:
                import panda_resilience as _resilience
                _resilience.end_wsi_session()
            except Exception:
                pass

        rows.extend(wsi_records)

        total_saved = sum(v["normal"] + v["tumor"] for v in level_summary.values())

        # Console log: image name + exact output.
        tqdm.write("")
        tqdm.write(f"[WSI DONE] {idx+1}/{len(sdf)} | split={split} | image_id={image_id}")
        for lv in cfg.levels:
            s = level_summary[lv]
            tqdm.write(
                f"  L{lv}: Normal={s['normal']} | Tumor={s['tumor']} | Total={s['normal']+s['tumor']}"
            )

            # Explain empty class folders specifically.
            cs = s.get("coordinate_stats") or {}
            er = s.get("exact_rejection_counts") or {}

            if s["normal"] == 0:
                normal_eligible = int(cs.get("eligible_normal_before_cap", 0))
                if normal_eligible == 0:
                    reason = (
                        f"no pure-Normal candidates; "
                        f"low_tissue={cs.get('rejected_low_tissue',0)}, "
                        f"low_valid={cs.get('rejected_low_valid_annotation',0)}, "
                        f"mixed={cs.get('rejected_mixed_borderline',0)}"
                    )
                else:
                    reason = (
                        f"Normal candidates existed ({normal_eligible}) but none passed exact saving; "
                        f"exact_low_valid={er.get('rejected_exact_low_valid_annotation',0)}, "
                        f"exact_mixed={er.get('rejected_exact_mixed_borderline',0)}, "
                        f"class_changed={er.get('rejected_exact_class_changed',0)}"
                    )
                tqdm.write(f"       Normal EMPTY -> {reason}")

            if s["tumor"] == 0:
                tumor_eligible = int(cs.get("eligible_tumor_before_cap", 0))
                if tumor_eligible == 0:
                    reason = (
                        f"no Tumor candidates >= threshold; "
                        f"low_tissue={cs.get('rejected_low_tissue',0)}, "
                        f"low_valid={cs.get('rejected_low_valid_annotation',0)}, "
                        f"mixed={cs.get('rejected_mixed_borderline',0)}"
                    )
                else:
                    reason = (
                        f"Tumor candidates existed ({tumor_eligible}) but none passed exact saving; "
                        f"exact_low_valid={er.get('rejected_exact_low_valid_annotation',0)}, "
                        f"exact_mixed={er.get('rejected_exact_mixed_borderline',0)}, "
                        f"class_changed={er.get('rejected_exact_class_changed',0)}"
                    )
                tqdm.write(f"       Tumor EMPTY  -> {reason}")

        if total_saved == 0:
            tqdm.write("  >>> NO PATCHES SAVED FROM THIS WSI")
        else:
            tqdm.write(f"  >>> TOTAL PATCHES SAVED FROM WSI = {total_saved}")

        # Durable one-row WSI log.
        row_log = {
            "image_id": image_id,
            "provider": str(r.provider),
            "split": split,
            "cap": int(cap),
            "L0_normal": level_summary.get(0, {}).get("normal", 0),
            "L0_tumor": level_summary.get(0, {}).get("tumor", 0),
            "L1_normal": level_summary.get(1, {}).get("normal", 0),
            "L1_tumor": level_summary.get(1, {}).get("tumor", 0),
            "L2_normal": level_summary.get(2, {}).get("normal", 0),
            "L2_tumor": level_summary.get(2, {}).get("tumor", 0),
            "total_saved": int(total_saved),
        }

        # Add concise reason columns for each empty bucket.
        for lv in cfg.levels:
            s = level_summary[lv]
            cs = s.get("coordinate_stats") or {}
            er = s.get("exact_rejection_counts") or {}
            row_log[f"L{lv}_normal_candidate_count"] = int(cs.get("eligible_normal_before_cap", 0))
            row_log[f"L{lv}_tumor_candidate_count"] = int(cs.get("eligible_tumor_before_cap", 0))
            row_log[f"L{lv}_low_tissue_rejected"] = int(cs.get("rejected_low_tissue", 0))
            row_log[f"L{lv}_low_valid_rejected"] = int(cs.get("rejected_low_valid_annotation", 0))
            row_log[f"L{lv}_mixed_rejected"] = int(cs.get("rejected_mixed_borderline", 0))
            row_log[f"L{lv}_exact_low_valid_rejected"] = int(er.get("rejected_exact_low_valid_annotation", 0))
            row_log[f"L{lv}_exact_mixed_rejected"] = int(er.get("rejected_exact_mixed_borderline", 0))
            row_log[f"L{lv}_class_changed_rejected"] = int(er.get("rejected_exact_class_changed", 0))

        log_rows.append(row_log)

        tmp = log_csv.with_name(log_csv.name + ".tmp")
        pd.DataFrame(log_rows).to_csv(tmp, index=False)
        os.replace(tmp, log_csv)

        completed_ids.add(image_id)
        bar.update(1)

    bar.close()

    # Return records from current processing plus all durable shards for this split/cap,
    # so resuming does not lose earlier WSI rows.
    shard_dir = cfg.manifests_dir / "materialize_shards"
    all_frames = []
    for sh in shard_dir.glob(f"{split}_*_cap{cap}.csv"):
        try:
            d = pd.read_csv(sh)
            if len(d):
                all_frames.append(d)
        except Exception:
            pass
    if all_frames:
        return pd.concat(all_frames, ignore_index=True)
    return pd.DataFrame(rows)


def step_extract_final(cfg: FinalConfig):
    """
    Save bounded classified patches for ALL three splits using identical structure:

      patches/train/<image_id>/L0|L1|L2/Tumor|Normal/
      patches/val/<image_id>/L0|L1|L2/Tumor|Normal/
      patches/test/<image_id>/L0|L1|L2/Tumor|Normal/

    Train automatically expands until all six Train buckets can supply the final 80,000.
    Validation/Test keep a bounded QC/evaluation sample on disk.
    Dense Test WSI reconstruction still streams from retained original Test WSIs.
    """
    split_df = pd.read_csv(cfg.split_csv)
    ensure_full_patch_tree(cfg, split_df)
    targets = bucket_targets(cfg)
    previous = pd.read_csv(cfg.candidate_manifest) if cfg.candidate_manifest.exists() else pd.DataFrame()

    chosen_cap = None
    for cap in cfg.train_candidate_caps:
        tr = _materialize_split(cfg, split_df, "train", cap)
        va = _materialize_split(cfg, split_df, "val", cfg.val_saved_cap_per_class_slide_level)
        te = _materialize_split(cfg, split_df, "test", cfg.test_saved_cap_per_class_slide_level)

        all_df = _dedupe_manifest(pd.concat([previous, tr, va, te], ignore_index=True))

        tmp_manifest = cfg.candidate_manifest.with_name(cfg.candidate_manifest.name + ".tmp")
        all_df.to_csv(tmp_manifest, index=False)
        os.replace(tmp_manifest, cfg.candidate_manifest)
        previous = all_df

        counts = all_df[all_df.split=="train"].groupby(["level","category"]).size().to_dict()
        sufficient = all(int(counts.get(b, 0)) >= int(targets[b]) for b in targets)
        if sufficient:
            chosen_cap = cap
            break

        print("[AUTO-REPAIR] Some Train buckets are short. Expanding Train extraction before deleting Train WSIs.")

    train_counts = previous[previous.split=="train"].groupby(["level","category"]).size().to_dict()
    val_counts = previous[previous.split=="val"].groupby(["level","category"]).size().to_dict()
    test_counts = previous[previous.split=="test"].groupby(["level","category"]).size().to_dict()

    missing = {
        str(k): {"available": int(train_counts.get(k,0)), "required": int(v)}
        for k,v in targets.items()
        if int(train_counts.get(k,0)) < int(v)
    }

    passed_files, details = _verify_saved_patches(cfg, previous)

    # Every split should have at least some saved patches as proof the hierarchy is populated.
    split_nonempty = all(int((previous.split == s).sum()) > 0 for s in ("train","val","test"))
    passed = passed_files and not missing and split_nonempty

    core.qc_report(cfg, "04_extract_final", passed, {
        "chosen_train_cap_per_class_slide_level": chosen_cap,
        "train_bucket_counts": {str(k): int(v) for k,v in train_counts.items()},
        "val_bucket_counts": {str(k): int(v) for k,v in val_counts.items()},
        "test_bucket_counts": {str(k): int(v) for k,v in test_counts.items()},
        "final_train_bucket_targets": {str(k): int(v) for k,v in targets.items()},
        "deficient_train_buckets": missing,
        "saved_train_patches": int((previous.split=="train").sum()),
        "saved_val_patches": int((previous.split=="val").sum()),
        "saved_test_patches": int((previous.split=="test").sum()),
        "candidate_manifest": str(cfg.candidate_manifest),
        "folder_layout": "patches/<train|val|test>/<image_id>/L0|L1|L2/{Normal,Normal_Masks,Normal_Valid_Masks,Tumor,Tumor_Masks,Tumor_Valid_Masks}/",
        "note": "Saved Test patches are a bounded classified subset. Full WSI Test reconstruction streams densely from the retained original Test WSI.",
        **details,
    })


def step_delete_train_wsi(cfg: FinalConfig):
    """
    Destructive stage, but only after candidate patch extraction passed and all 6 buckets
    can supply the final requested 80k. Validation/Test WSI and ALL masks remain.
    """
    q = cfg.qc_dir / "04_extract_final.json"
    if not q.exists() or not json.load(open(q, "r", encoding="utf-8")).get("passed", False):
        raise core.QCError("Refusing to delete Train WSIs before verified extraction PASS.")

    split_df = pd.read_csv(cfg.split_csv)
    train = split_df[split_df.split=="train"].copy()
    log_rows = []
    for _, r in tqdm(train.iterrows(), total=len(train), desc="DELETE VERIFIED TRAIN WSI"):
        p = Path(r.image_path)
        rec = {"image_id": str(r.image_id), "image_path": str(p), "existed_before": p.exists(), "deleted": False}
        if cfg.delete_train_wsi_after_verified_extraction and p.exists():
            p.unlink()
            rec["deleted"] = not p.exists()
        log_rows.append(rec)
    pd.DataFrame(log_rows).to_csv(cfg.deleted_train_wsi_csv, index=False)

    # Critical assertions: no validation/test WSI was touched; no masks were touched.
    untouched_missing = []
    for _, r in split_df[split_df.split.isin(["val","test"])].iterrows():
        if not Path(r.image_path).exists():
            untouched_missing.append(str(r.image_id))
    mask_missing = [str(r.image_id) for _,r in split_df.iterrows() if not Path(r.mask_path).exists()]
    passed = (not untouched_missing) and (not mask_missing)
    core.qc_report(cfg, "04b_delete_train_wsi", passed, {
        "train_wsi_rows": len(log_rows),
        "deleted_count": int(sum(bool(x["deleted"]) for x in log_rows)),
        "validation_test_missing_after_delete": untouched_missing[:50],
        "mask_missing_after_delete": mask_missing[:50],
        "deletion_log": str(cfg.deleted_train_wsi_csv),
        "note": "Only TRAIN image TIFFs are deleted. Validation/Test images and all label masks are retained.",
    })


def step_select_final(cfg: FinalConfig):
    df = pd.read_csv(cfg.candidate_manifest)
    train = df[df.split=="train"].copy()
    targets = bucket_targets(cfg)
    selected = []
    info = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for lv in cfg.levels:
        for cat in ("normal","tumor"):
            bdf = train[(train.level==lv)&(train.category==cat)].copy()
            target = int(targets[(lv,cat)])
            if len(bdf) < target:
                raise core.QCError(f"Bucket L{lv}-{cat} has {len(bdf)} < required {target}")

            # Limit expensive EfficientNet feature extraction while keeping > target.
            feat_cap = max(target, min(cfg.feature_candidate_cap_per_bucket, len(bdf)))
            if len(bdf) > feat_cap:
                bdf = bdf.sample(feat_cap, random_state=cfg.seed + lv + (100 if cat=="tumor" else 0))

            fbase = cfg.features_dir / f"PANDA_L{lv}_{cat}_EffB4"
            npz = fbase.with_suffix(".npz")
            csvp = fbase.with_suffix(".csv")
            if npz.exists() and csvp.exists():
                X = np.load(npz)["X"]
                fdf = pd.read_csv(csvp)
            else:
                X, fdf = core.extract_features_for_df(cfg, bdf, npz)

            chosen = core.representative_select(X, fdf, target, cfg.max_k_for_elbow, cfg.seed + lv)
            selected.append(chosen)
            info.append({"level":lv,"category":cat,"available":len(train[(train.level==lv)&(train.category==cat)]),
                         "feature_candidates":len(fdf),"selected":len(chosen),"target":target})

    out = pd.concat(selected, ignore_index=True)
    out = out.sample(frac=1, random_state=cfg.seed).reset_index(drop=True)
    tmp_sel = cfg.selected_train_csv.with_name(cfg.selected_train_csv.name + ".tmp")
    out.to_csv(tmp_sel, index=False)
    os.replace(tmp_sel, cfg.selected_train_csv)

    counts = out.groupby(["level","category"]).size().to_dict()
    exact = len(out)==cfg.num_samples and all(int(counts.get(k,0))==int(v) for k,v in targets.items())
    core.qc_report(cfg, "05_select_final", exact, {
        "requested_total": cfg.num_samples,
        "actual_total": len(out),
        "targets": {str(k):int(v) for k,v in targets.items()},
        "counts": {str(k):int(v) for k,v in counts.items()},
        "selection": info,
        "selected_train_csv": str(cfg.selected_train_csv),
    })


def step_normalization_final(cfg: FinalConfig):
    # Make core training functions see the candidate manifest.
    shutil.copy2(cfg.candidate_manifest, cfg.patch_manifest)
    core.step_qc_normalization(cfg)


def step_dataloader_final(cfg: FinalConfig):
    core.step_qc_dataloader(cfg)


def step_sanity_final(cfg: FinalConfig):
    core.step_sanity(cfg)


def step_pilot_experiments(cfg: FinalConfig):
    """
    Pilot experiments run after the final model/results in V11, while all bounded
    training candidates still exist. They are article ablations and do not configure
    or tune the final model. Completed experiments are reused by the resilience layer.
    """
    # Legacy suite is level-agnostic in most pilot routines and reads cfg.levels.
    legacy_paper.pilot_experiments(cfg, include_encoder=True, include_provider_holdout=True)
    p = cfg.work / "paper_results" / "tables" / "pilot_ablation_results.csv"
    if not p.exists():
        raise RuntimeError("Pilot ablation output missing")


def step_prune_unselected(cfg: FinalConfig):
    """
    V11 intentionally delays pruning until AFTER the resumed pilot experiments.
    This preserves non-selected candidate patches required by random/balanced ablations.
    Then keep only the final 80k TRAIN patches + validation/test bounded patches.
    """
    cand = pd.read_csv(cfg.candidate_manifest)
    sel = pd.read_csv(cfg.selected_train_csv)
    keep_img = set(sel.img.astype(str))
    keep_mask = set(sel["mask"].astype(str))

    removed_img = removed_mask = removed_valid = 0
    train_cand = cand[cand.split=="train"]
    for _, r in tqdm(train_cand.iterrows(), total=len(train_cand), desc="PRUNE UNSELECTED TRAIN PATCHES"):
        if str(r.img) not in keep_img:
            p = Path(r.img)
            if p.exists():
                p.unlink(); removed_img += 1
        if str(r["mask"]) not in keep_mask:
            p = Path(r["mask"])
            if p.exists():
                p.unlink(); removed_mask += 1
            vp = Path(str(r["valid_mask"])) if "valid_mask" in r.index else valid_mask_path(p)
            if vp.exists():
                vp.unlink(); removed_valid += 1

    # Final manifest = exactly selected Train + retained Validation.
    val = cand[cand.split=="val"].copy()
    test = cand[cand.split=="test"].copy()
    final = pd.concat([sel, val, test], ignore_index=True)
    final.to_csv(cfg.final_training_manifest, index=False)
    shutil.copy2(cfg.final_training_manifest, cfg.patch_manifest)

    missing_selected = [p for p in sel.img.astype(str) if not Path(p).exists()]
    passed = len(missing_selected)==0 and len(sel)==cfg.num_samples
    core.qc_report(cfg, "08b_prune", passed, {
        "removed_unselected_images": removed_img,
        "removed_unselected_masks": removed_mask,
        "removed_unselected_valid_masks": removed_valid,
        "kept_selected_train_patches": len(sel),
        "kept_validation_patches": len(val),
        "kept_test_patches": len(test),
        "missing_selected_after_prune": missing_selected[:20],
        "final_training_manifest": str(cfg.final_training_manifest),
    })


def step_train_final(cfg: FinalConfig):
    core.step_train(cfg)


def list_final_outputs(cfg: FinalConfig) -> Dict[str, Any]:
    return {
        "selected_train": str(cfg.selected_train_csv),
        "final_train_val_manifest": str(cfg.final_training_manifest),
        "deleted_train_wsi_log": str(cfg.deleted_train_wsi_csv),
        "missing_mask_report": str(cfg.excluded_missing_masks_csv),
        "paper_results": str(cfg.work / "paper_results"),
        "state": str(cfg.work / "pipeline_state.json"),
    }
