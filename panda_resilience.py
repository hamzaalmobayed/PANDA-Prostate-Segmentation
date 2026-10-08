
from __future__ import annotations

"""
Crash/SSD-disconnect resilience layer for the PANDA prostate pipeline.

Goals
-----
1. Never trust an incomplete file: all state writes are atomic (temp -> replace).
2. Every expensive stage has durable sub-stage progress.
3. If an external SSD disappears while Python is still running, wait for it and retry
   the SAME unit of work instead of advancing or corrupting state.
4. If the process/PC restarts, resume from the last durable checkpoint.
5. Never delete a Train WSI until extraction QC has passed and enough candidates exist.
6. Validation/Test WSI and every label mask are never automatically deleted.

Important physical limitation
-----------------------------
A power cut can happen between two durable writes. It is impossible to preserve an
in-flight GPU kernel or a half-written optimizer update without writing the whole model
state after every batch (which would cause extreme SSD writes). Therefore the main
training loop uses periodic heavy checkpoints plus:
- exact phase-boundary checkpoints,
- exact Validation metric progress every batch,
- immediate in-RAM recovery on a temporary SSD disconnect when Python stays alive.

The checkpoint interval is configurable in FinalConfig.
"""

import os
import gc
import io
import json
import math
import time
import random
import hashlib
import shutil
import traceback
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, Sampler, Dataset

import panda_camelyon_style_pipeline as core
import panda_final_pipeline as final


# -----------------------------------------------------------------------------
# Atomic / storage helpers
# -----------------------------------------------------------------------------

def atomic_json(path: Path, obj: Dict[str, Any]):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_csv(path: Path, df: pd.DataFrame):
    """Atomic CSV write that is Windows-safe."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    # Keep the descriptor writable while flushing/fsyncing.
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        df.to_csv(f, index=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_torch_save(path: Path, obj: Dict[str, Any]):
    """Atomic torch checkpoint write that is Windows-safe.

    The existing checkpoint is never overwritten in-place.  If an external drive
    disappears while torch.save is writing, only ``*.tmp`` can be damaged; the
    last completed checkpoint remains intact and is used after reconnect/restart.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    # A previous interrupted save may have left a partial temporary archive.
    # Opening with wb already truncates it, but unlinking first also handles stale
    # Windows file metadata after an SSD reconnect.
    try:
        if tmp.exists():
            tmp.unlink()
    except OSError:
        # Let the real write below raise; retry_same_unit will wait/retry storage errors.
        pass
    with open(tmp, "wb") as f:
        torch.save(obj, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def storage_roots(cfg) -> List[Path]:
    roots = [Path(cfg.work_root), Path(cfg.data_root)]
    # Keep unique drive/root locations.
    out = []
    seen = set()
    for p in roots:
        key = str(p)
        if key not in seen:
            out.append(p); seen.add(key)
    return out


def storage_ready(cfg) -> bool:
    # Work root may not exist on first run, so parent/drive is enough.
    for p in storage_roots(cfg):
        if p.exists():
            continue
        parent = p.parent
        if parent.exists():
            continue
        return False
    return True


def wait_for_storage(cfg, reason: str = "", poll_seconds: int = 10):
    """
    Wait forever for a temporarily missing external drive. This is deliberate:
    reconnecting the SSD lets the same Python process continue.
    """
    announced = False
    while not storage_ready(cfg):
        if not announced:
            print(f"\n[STORAGE WAIT] Storage unavailable. {reason}")
            print(f"[STORAGE WAIT] Reconnect the SSD/drive. Retrying every {poll_seconds}s; no stage is marked PASS.")
            announced = True
        time.sleep(poll_seconds)
    if announced:
        print("[STORAGE RESTORED] Required storage paths are visible again. Resuming the same unit of work.")


def is_openslide_transient_open_error(exc: BaseException) -> bool:
    """
    OpenSlide uses the same exception text both for a genuinely unsupported/corrupt TIFF
    and for a file that temporarily disappeared because an external drive disconnected.
    """
    m = repr(exc).lower()
    return (
        "openslideunsupportedformaterror" in m
        or "unsupported or missing image file" in m
        or ("openslideerror" in m and "missing" in m)
    )



def wait_for_storage_stable(cfg, reason: str = "", poll_seconds: int = 10, stable_checks: int = 3):
    """Wait indefinitely until data/work storage is visible and stable again."""
    announced = False
    good = 0
    while True:
        roots = [Path(str(cfg.data_root)), Path(str(cfg.work_root))]
        missing = [str(p) for p in roots if not (p.exists() or p.parent.exists())]

        if not missing:
            good += 1
            if good >= stable_checks:
                if announced:
                    print("[SSD RESTORED] Storage is stable again. Resuming the same stage/sub-step.")
                return
        else:
            good = 0
            if not announced:
                print("")
                print(f"[SSD WAIT - GLOBAL] {reason}")
                print("[SSD WAIT - GLOBAL] Required storage disappeared.")
                for x in missing:
                    print("   ", x)
                print(f"[SSD WAIT - GLOBAL] Reconnect the SSD. Retrying every {poll_seconds}s.")
                announced = True

        time.sleep(poll_seconds)


def storage_is_currently_missing(cfg) -> bool:
    roots = [Path(str(cfg.data_root)), Path(str(cfg.work_root))]
    return any(not (p.exists() or p.parent.exists()) for p in roots)

def is_torch_serialization_io_error(exc: BaseException) -> bool:
    """Recognize PyTorch zip-writer failures caused by a disappearing/intermittent drive.

    torch.save sometimes converts the original OSError into RuntimeError while the
    zip writer is closing, e.g. ``unexpected pos 64 vs 0``.  Those RuntimeErrors
    are storage failures even though they are not subclasses of OSError.
    """
    m = repr(exc).lower()
    return any(k in m for k in [
        "unexpected pos",
        "inline_container.cc",
        "pytorchstreamwriter",
        "failed writing file",
        "file write failed",
        "write_end_of_file",
        "invalid argument",
        "errno 22",
        "winerror 3",
    ])


def looks_like_io_error(exc: BaseException) -> bool:
    m = repr(exc).lower()
    if is_torch_serialization_io_error(exc):
        return True
    keys = [
        "input/output", "i/o error", "device not ready", "no such file or directory",
        "winerror 21", "winerror 1117", "winerror 53", "winerror 64", "winerror 1167",
        "invalid handle", "broken pipe", "transport endpoint", "network name is no longer available",
        "cannot open", "failed to open", "permissionerror", "oserror", "file not found", "path not found", "disk", "drive", "device", "failed to read", "failed to write", "read error", "write error", "truncated", "unable to open file", "failed opening file", "no such device"
    ]
    if getattr(exc, "errno", None) == 9 or "bad file descriptor" in m:
        return False
    return isinstance(exc, (OSError, IOError, FileNotFoundError, PermissionError)) or any(k in m for k in keys)


def wait_for_paths(cfg, paths, unit_name: str, poll_seconds: int = 10):
    """
    Wait until the exact WSI/mask files needed by the current unit are visible again.
    """
    paths = [Path(str(x)) for x in paths if x]
    announced = False
    while True:
        missing = [str(x) for x in paths if not x.exists()]
        if not missing:
            if announced:
                print(f"[FILES RESTORED] {unit_name}: required files are visible again.")
            return
        if not announced:
            print("")
            print(f"[SSD/FILE WAIT] {unit_name}")
            print("[SSD/FILE WAIT] Required file(s) temporarily unavailable:")
            for m in missing[:10]:
                print("   ", m)
            print(f"[SSD/FILE WAIT] Reconnect the SSD. Retrying every {poll_seconds}s.")
            announced = True
        time.sleep(poll_seconds)


def retry_same_unit(cfg, fn, unit_name: str, required_paths=None,
                    openslide_retries_when_present: int = 5):
    """
    Retry the SAME small unit.

    - Normal storage errors: wait for storage and retry.
    - OpenSlide "unsupported or missing image file":
        * if the exact file disappeared -> wait for reconnect indefinitely;
        * if the file is present -> retry a few times for a transient handle failure;
        * if it still fails while present -> raise, because it may truly be corrupt.
    """
    transient_open_attempts = 0
    while True:
        if required_paths:
            wait_for_paths(cfg, required_paths, unit_name)
        try:
            return fn()
        except Exception as e:
            if is_openslide_transient_open_error(e):
                missing = []
                if required_paths:
                    missing = [str(Path(str(x))) for x in required_paths if not Path(str(x)).exists()]
                if missing:
                    print(f"[SSD DISCONNECT DETECTED] {unit_name}: OpenSlide lost access to a required file.")
                    wait_for_paths(cfg, required_paths, unit_name)
                    transient_open_attempts = 0
                    time.sleep(1)
                    continue

                transient_open_attempts += 1
                if transient_open_attempts <= openslide_retries_when_present:
                    print(
                        f"[OPENSLIDE TRANSIENT RETRY] {unit_name}: "
                        f"file is present but OpenSlide could not open it "
                        f"({transient_open_attempts}/{openslide_retries_when_present})."
                    )
                    time.sleep(2)
                    continue

                raise

            if not looks_like_io_error(e):
                raise
            print(f"[I/O INTERRUPTION] {unit_name}: {e}")
            # Require several consecutive healthy checks before retrying a checkpoint
            # write. This avoids immediately retrying while Windows is still remounting
            # an external SSD.
            wait_for_storage_stable(cfg, unit_name)
            if required_paths:
                wait_for_paths(cfg, required_paths, unit_name)
            time.sleep(1)



# -----------------------------------------------------------------------------
# One-open-per-WSI shared extraction session
# -----------------------------------------------------------------------------

_ACTIVE_WSI_ID = None
_ACTIVE_IMAGE_READER = None
_ACTIVE_MASK_READER = None
_ACTIVE_LOWRES = None


def _close_active_session():
    global _ACTIVE_WSI_ID, _ACTIVE_IMAGE_READER, _ACTIVE_MASK_READER, _ACTIVE_LOWRES
    try:
        if _ACTIVE_IMAGE_READER is not None:
            _ACTIVE_IMAGE_READER.close()
    except Exception:
        pass
    try:
        if _ACTIVE_MASK_READER is not None:
            _ACTIVE_MASK_READER.close()
    except Exception:
        pass
    _ACTIVE_WSI_ID = None
    _ACTIVE_IMAGE_READER = None
    _ACTIVE_MASK_READER = None
    _ACTIVE_LOWRES = None


def begin_wsi_session(cfg, r):
    """
    Open the current WSI and its GT mask ONCE and build the shared low-resolution
    Tissue/Tumor/Valid masks ONCE. L0/L1/L2 reuse the same session.
    """
    global _ACTIVE_WSI_ID, _ACTIVE_IMAGE_READER, _ACTIVE_MASK_READER, _ACTIVE_LOWRES

    image_id = str(r.image_id)
    if _ACTIVE_WSI_ID == image_id and _ACTIVE_IMAGE_READER is not None and _ACTIVE_MASK_READER is not None:
        return

    _close_active_session()

    def _open():
        im = core.SlideScaleReader(r.image_path)
        mr = core.MaskScaleReader(r.mask_path, im.dimensions)
        return im, mr

    im, mr = retry_same_unit(
        cfg, _open, f"open shared WSI session {image_id}",
        required_paths=[r.image_path, r.mask_path]
    )

    # L0/L1/L2 all use a low-resolution mask at >=16x downsample in this pipeline,
    # therefore one shared thumbnail is enough for all three levels.
    def _make_lowres():
        thumb, meta = im.thumbnail_at_downsample(16.0, cfg.target_max_size)
        eff_ds = float(meta.get("effective_ds", 16.0))
        tissue, _ = core.generate_tissue_mask(thumb, cfg)

        mthumb = mr.reader.slide.get_thumbnail((thumb.shape[1], thumb.shape[0])).convert("RGB")
        raw = np.asarray(mthumb)
        tumor = core.raw_to_binary_mask(raw, str(r.provider))
        valid = core.raw_valid_tissue_mask(raw, str(r.provider))

        if tumor.shape != tissue.shape:
            tumor = cv2.resize(
                tumor, (tissue.shape[1], tissue.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )
            valid = cv2.resize(
                valid, (tissue.shape[1], tissue.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )

        return (
            tissue.astype(np.uint8),
            (tumor > 0).astype(np.uint8),
            (valid > 0).astype(np.uint8),
            eff_ds,
            im.dimensions,
        )

    lowres = retry_same_unit(
        cfg, _make_lowres, f"shared lowres masks {image_id}",
        required_paths=[r.image_path, r.mask_path]
    )

    _ACTIVE_WSI_ID = image_id
    _ACTIVE_IMAGE_READER = im
    _ACTIVE_MASK_READER = mr
    _ACTIVE_LOWRES = lowres


def end_wsi_session():
    _close_active_session()


def _get_active_session(cfg, r):
    if _ACTIVE_WSI_ID != str(r.image_id) or _ACTIVE_IMAGE_READER is None or _ACTIVE_MASK_READER is None:
        begin_wsi_session(cfg, r)
    return _ACTIVE_IMAGE_READER, _ACTIVE_MASK_READER, _ACTIVE_LOWRES


def _reopen_active_session(cfg, r):
    _close_active_session()
    begin_wsi_session(cfg, r)
    return _ACTIVE_IMAGE_READER, _ACTIVE_MASK_READER, _ACTIVE_LOWRES


def _integral_image_u64(mask: np.ndarray) -> np.ndarray:
    a = np.asarray(mask, dtype=np.uint64)
    ii = np.zeros((a.shape[0] + 1, a.shape[1] + 1), dtype=np.uint64)
    ii[1:, 1:] = a.cumsum(axis=0).cumsum(axis=1)
    return ii


def _rect_fraction_vectorized(ii: np.ndarray, x1, y1, x2, y2) -> np.ndarray:
    x1 = np.asarray(x1, dtype=np.int64)
    y1 = np.asarray(y1, dtype=np.int64)
    x2 = np.asarray(x2, dtype=np.int64)
    y2 = np.asarray(y2, dtype=np.int64)
    area = np.maximum(1, (x2 - x1) * (y2 - y1)).astype(np.float64)
    sums = (
        ii[y2, x2]
        - ii[y1, x2]
        - ii[y2, x1]
        + ii[y1, x1]
    ).astype(np.float64)
    return sums / area


# -----------------------------------------------------------------------------
# Resumable coordinate scanning
# -----------------------------------------------------------------------------

def _score(image_id: str, level: int, x0: int, y0: int, category: str, seed: int) -> int:
    b = f"{seed}|{image_id}|{level}|{category}|{x0}|{y0}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(b, digest_size=8).digest(), "big")


def _keep_smallest(pool: List[Dict[str, Any]], item: Dict[str, Any], cap: int):
    """Deterministic bounded sampling: retain the cap smallest hash scores."""
    if len(pool) < cap:
        pool.append(item)
        return
    worst_i = max(range(len(pool)), key=lambda i: int(pool[i]["_score"]))
    if int(item["_score"]) < int(pool[worst_i]["_score"]):
        pool[worst_i] = item


def resumable_build_coordinate_pool(cfg: final.FinalConfig, r: pd.Series, level: int) -> pd.DataFrame:
    """
    ULTRAFAST + backward-compatible coordinate search.

    Scientific design is unchanged:
      - exact same WSI split
      - L0/L1/L2
      - 512x512
      - stride 128
      - same tissue/valid/tumor thresholds
      - same Normal/Tumor six buckets

    Speed changes only:
      - reuse ONE shared low-res Tissue/Tumor/Valid mask set for L0/L1/L2
      - evaluate the same regular grid by vectorized integral-image rectangle sums
      - reuse completed coordinate CSVs from the old run
    """
    out_csv = cfg.coordinate_pool_dir / f"{r.image_id}_L{level}.csv"
    progress = cfg.coordinate_pool_dir / f"{r.image_id}_L{level}.progress.json"
    stats_json = cfg.coordinate_pool_dir / f"{r.image_id}_L{level}.stats.json"

    if out_csv.exists():
        df = pd.read_csv(out_csv)
        if not stats_json.exists():
            atomic_json(stats_json, {
                "image_id": str(r.image_id),
                "split": str(r.split),
                "level": int(level),
                "status": "reused_existing_coordinate_pool",
                "normal_candidates_after_cap": int((df.get("category_lr", pd.Series(dtype=str)) == "normal").sum()) if len(df) else 0,
                "tumor_candidates_after_cap": int((df.get("category_lr", pd.Series(dtype=str)) == "tumor").sum()) if len(df) else 0,
            })
        return df

    im, mr, lowres = _get_active_session(cfg, r)
    tissue, tumor_lr, valid_lr, eff_ds, dims = lowres
    w0, h0 = dims

    ds = float(cfg.logical_downsamples[level])
    field0 = int(round(cfg.patch_size * ds))
    stride0 = int(round(cfg.stride * ds))

    xs0 = np.arange(0, max(1, w0 - field0 + 1), stride0, dtype=np.int64)
    ys0 = np.arange(0, max(1, h0 - field0 + 1), stride0, dtype=np.int64)
    xx0, yy0 = np.meshgrid(xs0, ys0)
    x0 = xx0.ravel()
    y0 = yy0.ravel()
    total = int(len(x0))

    H, W = tissue.shape[:2]
    x1 = np.clip(np.floor(x0 / eff_ds).astype(np.int64), 0, W)
    y1 = np.clip(np.floor(y0 / eff_ds).astype(np.int64), 0, H)
    x2 = np.clip(np.ceil((x0 + field0) / eff_ds).astype(np.int64), 0, W)
    y2 = np.clip(np.ceil((y0 + field0) / eff_ds).astype(np.int64), 0, H)

    tissue_frac = _rect_fraction_vectorized(
        _integral_image_u64(tissue > 0), x1, y1, x2, y2
    )
    valid_frac = _rect_fraction_vectorized(
        _integral_image_u64(valid_lr > 0), x1, y1, x2, y2
    )
    tumor_frac = _rect_fraction_vectorized(
        _integral_image_u64(tumor_lr > 0), x1, y1, x2, y2
    )

    tissue_ok = tissue_frac >= float(cfg.tissue_min_frac)
    valid_ok = valid_frac >= float(cfg.min_valid_annotated_fraction)
    eligible = tissue_ok & valid_ok

    tumor_ok = eligible & (tumor_frac >= float(cfg.tumor_threshold))
    normal_ok = eligible & (tumor_frac <= 0.0)
    mixed = eligible & (~tumor_ok) & (~normal_ok)

    cap = int(cfg.coordinate_pool_cap_per_class_slide_level)

    def choose(mask: np.ndarray, cat: str) -> list:
        inds = np.flatnonzero(mask)
        if len(inds) == 0:
            return []

        # Stable deterministic selection, so execution order does not change the pool.
        scored = [
            (
                int(_score(str(r.image_id), level, int(x0[i]), int(y0[i]), cat, cfg.seed)),
                int(i)
            )
            for i in inds.tolist()
        ]
        scored.sort(key=lambda z: z[0])
        chosen = [i for _, i in scored[:cap]]

        return [{
            "image_id": str(r.image_id),
            "provider": str(r.provider),
            "split": str(r.split),
            "level": int(level),
            "x0": int(x0[i]),
            "y0": int(y0[i]),
            "tissue_fraction_lr": float(tissue_frac[i]),
            "tumor_fraction_lr": float(tumor_frac[i]),
            "category_lr": cat,
        } for i in chosen]

    df = pd.DataFrame(choose(normal_ok, "normal") + choose(tumor_ok, "tumor"))

    retry_same_unit(
        cfg,
        lambda: atomic_csv(out_csv, df),
        f"save coordinate pool {r.image_id} L{level}"
    )

    stats = {
        "image_id": str(r.image_id),
        "provider": str(r.provider),
        "split": str(r.split),
        "level": int(level),
        "target_downsample": ds,
        "patch_size": int(cfg.patch_size),
        "stride": int(cfg.stride),
        "grid_positions_total": total,
        "grid_positions_processed": total,
        "rejected_low_tissue": int((~tissue_ok).sum()),
        "rejected_low_valid_annotation": int((tissue_ok & ~valid_ok).sum()),
        "rejected_mixed_borderline": int(mixed.sum()),
        "eligible_normal_before_cap": int(normal_ok.sum()),
        "eligible_tumor_before_cap": int(tumor_ok.sum()),
        "normal_candidates_after_cap": int((df.category_lr == "normal").sum()) if len(df) else 0,
        "tumor_candidates_after_cap": int((df.category_lr == "tumor").sum()) if len(df) else 0,
        "candidate_cap_per_class_slide_level": cap,
        "scan_engine": "shared_lowres_plus_vectorized_integral_exact_grid",
        "shared_wsi_session": True,
        "backward_compatible_resume": True,
    }

    retry_same_unit(
        cfg,
        lambda: atomic_json(stats_json, stats),
        f"save coordinate stats {r.image_id} L{level}"
    )

    # Old partial per-position progress is superseded only for this unfinished level.
    try:
        progress.unlink(missing_ok=True)
    except Exception:
        pass

    return df


# -----------------------------------------------------------------------------
# Resumable patch materialization
# -----------------------------------------------------------------------------

def _record_from_saved(cfg, r, level, cat_hint, c, ip: Path, mp: Path, w0, h0, ds):
    ma = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
    if ma is None:
        raise OSError(f"Cannot read saved mask {mp}")
    exact_tumor = float((ma > 0).mean())
    exact_cat = "tumor" if exact_tumor >= cfg.tumor_threshold else "normal"
    field0 = cfg.patch_size * ds
    return {
        "image_id": str(r.image_id), "provider": str(r.provider), "split": str(r.split),
        "level": int(level), "label": int(exact_cat == "tumor"), "category": exact_cat,
        "x0": int(c.x0), "y0": int(c.y0),
        "coord_x": float((int(c.x0) + field0/2) / max(w0,1)),
        "coord_y": float((int(c.y0) + field0/2) / max(h0,1)),
        "target_downsample": float(ds), "native_level": -1, "virtual_scale": False,
        "tissue_fraction": float(c.tissue_fraction_lr), "tumor_fraction": exact_tumor,
        "img": str(ip), "mask": str(mp),
    }


def resumable_materialize_from_pool(cfg: final.FinalConfig, r: pd.Series, level: int,
                                     desired_per_class: int) -> List[Dict[str, Any]]:
    """
    Materialize accepted patches and write an exact reason summary for this WSI/level.
    """
    pool = resumable_build_coordinate_pool(cfg, r, level)

    shard_dir = cfg.manifests_dir / "materialize_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard = shard_dir / f"{r.split}_{r.image_id}_L{level}_cap{desired_per_class}.csv"
    reason_json = shard_dir / f"{r.split}_{r.image_id}_L{level}_cap{desired_per_class}.reasons.json"

    if len(pool) == 0:
        stats_path = cfg.coordinate_pool_dir / f"{r.image_id}_L{level}.stats.json"
        coord_stats = {}
        if stats_path.exists():
            try:
                coord_stats = json.load(open(stats_path, "r", encoding="utf-8"))
            except Exception:
                pass
        summary = {
            "image_id": str(r.image_id), "split": str(r.split), "level": int(level),
            "desired_per_class": int(desired_per_class),
            "saved_normal": 0, "saved_tumor": 0, "saved_total": 0,
            "reason_if_zero": "no_coordinate_candidates_passed_preselection",
            "coordinate_stats": coord_stats,
        }
        atomic_json(reason_json, summary)
        return []

    done_df = pd.read_csv(shard) if shard.exists() else pd.DataFrame()
    done_keys = set()
    records = []
    if len(done_df):
        records = done_df.to_dict("records")
        done_keys = {(int(x["x0"]), int(x["y0"])) for x in records}

    exact_reasons = {
        "rejected_exact_low_valid_annotation": 0,
        "rejected_exact_mixed_borderline": 0,
        "rejected_exact_class_changed": 0,
        "reused_already_saved": len(done_keys),
    }

    im, mr, lowres = _get_active_session(cfg, r)
    w0, h0 = im.dimensions
    ds = float(cfg.logical_downsamples[level])

    try:
        for cat in ("normal", "tumor"):
            sub = pool[pool.category_lr == cat].copy()
            if len(sub) == 0:
                continue
            sub = sub.sort_values(
                ["tissue_fraction_lr", "y0", "x0"],
                ascending=[False, True, True]
            ).head(max(desired_per_class * 3, desired_per_class))

            accepted_for_cat = sum(1 for x in records if str(x.get("category","")).lower() == cat)

            for _, c in sub.iterrows():
                if accepted_for_cat >= desired_per_class:
                    break

                key = (int(c.x0), int(c.y0))
                if key in done_keys:
                    continue
                x0, y0 = key

                while True:
                    try:
                        rgb, smeta = im.read_at_downsample(x0, y0, ds, cfg.patch_size)
                        raw, _ = mr.read_raw(x0, y0, ds, cfg.patch_size)
                        break
                    except Exception as e:
                        if not (looks_like_io_error(e) or is_openslide_transient_open_error(e)):
                            raise
                        print(f"[I/O/OPENSLIDE INTERRUPTION] patch {r.image_id} L{level} x{x0} y{y0}: {e}")
                        wait_for_paths(
                            cfg, [r.image_path, r.mask_path],
                            f"patch {r.image_id} L{level}"
                        )
                        im, mr, lowres = _reopen_active_session(cfg, r)
                        w0, h0 = im.dimensions

                binm = core.raw_to_binary_mask(raw, str(r.provider))
                validm = core.raw_valid_tissue_mask(raw, str(r.provider))
                valid_fraction = float((validm > 0).mean())

                if valid_fraction < cfg.min_valid_annotated_fraction:
                    exact_reasons["rejected_exact_low_valid_annotation"] += 1
                    continue

                exact_tumor = float(
                    ((binm > 0) & (validm > 0)).sum() / max(1, (validm > 0).sum())
                )

                if exact_tumor >= cfg.tumor_threshold:
                    exact_cat = "tumor"
                elif exact_tumor <= 0.0:
                    exact_cat = "normal"
                else:
                    exact_reasons["rejected_exact_mixed_borderline"] += 1
                    continue

                if exact_cat != cat:
                    exact_reasons["rejected_exact_class_changed"] += 1
                    continue

                ip, mp = final.patch_paths(
                    cfg, str(r.split), str(r.image_id), level, exact_cat, x0, y0
                )
                vp = final.valid_mask_path(mp)
                ip.parent.mkdir(parents=True, exist_ok=True)
                mp.parent.mkdir(parents=True, exist_ok=True)
                vp.parent.mkdir(parents=True, exist_ok=True)

                def write_patch():
                    if not ip.exists():
                        tmp = ip.with_name(ip.name + ".tmp")
                        Image.fromarray(rgb).save(
                            tmp, "JPEG", quality=cfg.patch_jpeg_quality, subsampling=0
                        )
                        os.replace(tmp, ip)
                    if not mp.exists():
                        tmpm = mp.with_name(mp.name + ".tmp")
                        Image.fromarray(binm.astype(np.uint8)).save(
                            tmpm, "PNG", compress_level=3
                        )
                        os.replace(tmpm, mp)
                    if not vp.exists():
                        tmpv = vp.with_name(vp.name + ".tmp")
                        Image.fromarray((validm.astype(np.uint8) * 255)).save(
                            tmpv, "PNG", compress_level=3
                        )
                        os.replace(tmpv, vp)

                retry_same_unit(cfg, write_patch, f"write patch {ip.name}")

                field0 = cfg.patch_size * ds
                rec = {
                    "image_id": str(r.image_id), "provider": str(r.provider),
                    "split": str(r.split), "level": int(level),
                    "label": int(exact_cat == "tumor"), "category": exact_cat,
                    "x0": x0, "y0": y0,
                    "coord_x": float((x0 + field0/2) / max(w0,1)),
                    "coord_y": float((y0 + field0/2) / max(h0,1)),
                    "target_downsample": ds,
                    "native_level": int(smeta["native_level"]),
                    "native_downsample": float(smeta["native_downsample"]),
                    "virtual_scale": bool(smeta["virtual_scale"]),
                    "tissue_fraction": float(c.tissue_fraction_lr),
                    "valid_annotated_fraction": valid_fraction,
                    "tumor_fraction": exact_tumor,
                    "img": str(ip), "mask": str(mp), "valid_mask": str(vp),
                }
                records.append(rec)
                done_keys.add(key)
                accepted_for_cat += 1

                retry_same_unit(
                    cfg,
                    lambda: atomic_csv(shard, pd.DataFrame(records)),
                    f"save patch shard {r.image_id} L{level}"
                )
    finally:
        # Do NOT close here. L0/L1/L2 share the same OpenSlide readers.
        pass

    saved_normal = sum(1 for x in records if str(x.get("category","")).lower() == "normal")
    saved_tumor = sum(1 for x in records if str(x.get("category","")).lower() == "tumor")

    coord_stats = {}
    stats_path = cfg.coordinate_pool_dir / f"{r.image_id}_L{level}.stats.json"
    if stats_path.exists():
        try:
            coord_stats = json.load(open(stats_path, "r", encoding="utf-8"))
        except Exception:
            pass

    reason_if_zero = None
    if saved_normal + saved_tumor == 0:
        if not len(pool):
            reason_if_zero = "no_coordinate_candidates"
        elif exact_reasons["rejected_exact_low_valid_annotation"] > 0:
            reason_if_zero = "all_exact_candidates_failed_valid_annotation_check"
        elif exact_reasons["rejected_exact_mixed_borderline"] > 0:
            reason_if_zero = "all_exact_candidates_were_mixed_borderline"
        elif exact_reasons["rejected_exact_class_changed"] > 0:
            reason_if_zero = "all_exact_candidates_changed_class_at_full_resolution"
        else:
            reason_if_zero = "no_patch_passed_final_acceptance"

    summary = {
        "image_id": str(r.image_id), "provider": str(r.provider),
        "split": str(r.split), "level": int(level),
        "desired_per_class": int(desired_per_class),
        "saved_normal": int(saved_normal),
        "saved_tumor": int(saved_tumor),
        "saved_total": int(saved_normal + saved_tumor),
        "reason_if_zero": reason_if_zero,
        "exact_rejection_counts": exact_reasons,
        "coordinate_stats": coord_stats,
    }
    atomic_json(reason_json, summary)

    return records


# -----------------------------------------------------------------------------
# Resumable EfficientNet-B4 features: each batch is a durable shard
# -----------------------------------------------------------------------------

@torch.inference_mode()
def resumable_extract_features_for_df(cfg, df: pd.DataFrame, out_npz: Path):
    out_npz = Path(out_npz)
    out_csv = out_npz.with_suffix(".csv")
    if out_npz.exists() and out_csv.exists():
        return np.load(out_npz)["X"], pd.read_csv(out_csv)

    shard_dir = out_npz.parent / (out_npz.stem + "_feature_shards")
    shard_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    features_net, avgpool, preprocess = core.get_effb4_feature_extractor(cfg, device)
    bs = int(cfg.feat_batch_size)

    for start in tqdm(range(0, len(df), bs), desc=f"EFFB4 RESUME {out_npz.stem}"):
        stop = min(len(df), start + bs)
        fsh = shard_dir / f"{start:08d}_{stop:08d}.npy"
        csh = shard_dir / f"{start:08d}_{stop:08d}.csv"
        if fsh.exists() and csh.exists():
            continue

        batch_df = df.iloc[start:stop].copy()

        while True:
            try:
                imgs = []
                rows = []
                for _, rr in batch_df.iterrows():
                    imgs.append(preprocess(Image.open(rr.img).convert("RGB")))
                    rows.append(rr)
                x = torch.stack(imgs).to(device, non_blocking=True)
                f = features_net(x); f = avgpool(f); f = torch.flatten(f, 1)
                arr = f.detach().cpu().numpy().astype(np.float32)
                kdf = pd.DataFrame(rows).reset_index(drop=True)

                tmpn = fsh.with_name(fsh.name + ".tmp")
                with open(tmpn, "wb") as fh:
                    np.save(fh, arr)
                    fh.flush(); os.fsync(fh.fileno())
                os.replace(tmpn, fsh)
                atomic_csv(csh, kdf)
                break
            except Exception as e:
                if not looks_like_io_error(e):
                    raise
                print(f"[I/O INTERRUPTION] EfficientNet feature batch {start}:{stop}: {e}")
                wait_for_storage(cfg, f"feature batch {out_npz.stem} {start}:{stop}")
                time.sleep(1)

    arrays = []
    frames = []
    for start in range(0, len(df), bs):
        stop = min(len(df), start + bs)
        arrays.append(np.load(shard_dir / f"{start:08d}_{stop:08d}.npy"))
        frames.append(pd.read_csv(shard_dir / f"{start:08d}_{stop:08d}.csv"))
    X = np.concatenate(arrays, axis=0) if arrays else np.empty((0,0), np.float32)
    kdf = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if len(X) == 0:
        raise core.QCError("No EfficientNet features extracted")

    tmp = out_npz.with_name(out_npz.name + ".tmp")
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, X=X)
        fh.flush(); os.fsync(fh.fileno())
    os.replace(tmp, out_npz)
    atomic_csv(out_csv, kdf)
    return X, kdf


# -----------------------------------------------------------------------------
# Fully phased main training resume
# -----------------------------------------------------------------------------

class FixedEpochSampler(Sampler):
    def __init__(self, n: int, seed: int, epoch: int, start_sample: int = 0, shuffle: bool = True):
        self.n = int(n); self.start = int(start_sample)
        if shuffle:
            g = torch.Generator()
            g.manual_seed(int(seed) + int(epoch) * 100003)
            self.indices = torch.randperm(self.n, generator=g).tolist()
        else:
            self.indices = list(range(self.n))

    def __iter__(self):
        return iter(self.indices[self.start:])

    def __len__(self):
        return max(0, self.n - self.start)


def _rng_state():
    d = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        d["torch_cuda"] = torch.cuda.get_rng_state_all()
    return d


def _restore_rng(d):
    if not d:
        return
    try: random.setstate(d["python"])
    except Exception: pass
    try: np.random.set_state(d["numpy"])
    except Exception: pass
    try: torch.set_rng_state(d["torch_cpu"])
    except Exception: pass
    if torch.cuda.is_available() and "torch_cuda" in d:
        try: torch.cuda.set_rng_state_all(d["torch_cuda"])
        except Exception: pass


def _meter_to_dict(m):
    return {"tp":int(m.tp),"fp":int(m.fp),"tn":int(m.tn),"fn":int(m.fn)}

def _meter_from_dict(d):
    m = core.PixelMeter()
    if d:
        m.tp=int(d.get("tp",0)); m.fp=int(d.get("fp",0)); m.tn=int(d.get("tn",0)); m.fn=int(d.get("fn",0))
    return m


def _make_fixed_loader(ds, cfg, epoch, next_batch, shuffle):
    start_sample = int(next_batch) * int(cfg.batch_size)
    sampler = FixedEpochSampler(len(ds), cfg.seed, epoch, start_sample=start_sample, shuffle=shuffle)
    # FixedEpochSampler still controls the exact sample order/resume offset; workers only parallelize loading.
    nw = int(getattr(cfg, "num_workers", 4))
    kw = dict(
        batch_size=cfg.batch_size, sampler=sampler, shuffle=False,
        num_workers=nw, pin_memory=torch.cuda.is_available(), drop_last=False, worker_init_fn=core.fast_worker_init_fn,
    )
    if nw > 0:
        kw.update(
            persistent_workers=bool(getattr(cfg, "persistent_workers", True)),
            prefetch_factor=int(getattr(cfg, "prefetch_factor", 4)),
        )
    return DataLoader(ds, **kw)


def _training_state(model, opt, scaler, epoch, phase, next_batch, best, bad, history,
                    meter=None, loss_sum=0.0, n=0):
    return {
        "epoch": int(epoch), "phase": str(phase), "next_batch": int(next_batch),
        "model": model.state_dict(), "optimizer": opt.state_dict(), "scaler": scaler.state_dict(),
        "best": float(best), "bad": int(bad), "history": history,
        "meter": _meter_to_dict(meter) if meter is not None else None,
        "loss_sum": float(loss_sum), "n": int(n), "rng": _rng_state(),
    }



class _VahadaneCacheBuildDataset(Dataset):
    """Pre-build Vahadane cache once, before the expensive training loop.

    This changes no pixels, labels, augmentation, model, optimizer, or sampling.
    It only moves deterministic Vahadane preprocessing out of __getitem__ during
    the epochs so workers spend their time feeding the GPU instead of normalizing.
    """
    def __init__(self, frame, cfg):
        self.paths = frame["img"].astype(str).tolist()
        self.cfg = cfg
        self._norm = None

    def __len__(self):
        return len(self.paths)

    def _get_norm(self):
        if self._norm is None:
            with Image.open(self.cfg.stain_target_path) as im:
                target = np.array(im.convert("RGB"), dtype=np.uint8, copy=True, order="C")
            self._norm = core.VahadaneProcessor(target)
        return self._norm

    def __getitem__(self, idx):
        src = self.paths[idx]
        cp = core.stain_cache_path(self.cfg, src, "vahadane")
        try:
            if cp.exists() and cp.stat().st_size > 0:
                return 1  # cache hit
        except OSError:
            pass
        # Construct the normalizer only on a real miss. This also prevents the
        # TIAToolbox Vahadane warning from repeating on pure cache-hit epochs.
        core.load_or_build_stain_cache(self.cfg, src, "vahadane", self._get_norm().transform)
        return 0  # newly built


def _prebuild_final_vahadane_cache(cfg, tr, va):
    if not bool(getattr(cfg, "prebuild_stain_cache_before_training", True)):
        return

    ns = core._stain_cache_namespace(cfg, "vahadane")
    marker = ns / "FINAL_TRAIN_VAL_CACHE_READY.json"
    expected_train = int(len(tr)); expected_val = int(len(va))

    # Once the full cache has been built successfully, future restarts skip the
    # 80k-file scan entirely. The marker lives inside the target-specific cache
    # namespace, so a new stain target automatically gets a different marker.
    try:
        if marker.exists():
            meta = json.load(open(marker, "r", encoding="utf-8"))
            if int(meta.get("train", -1)) == expected_train and int(meta.get("val", -1)) == expected_val:
                print(f"[VAHADANE CACHE READY] train={expected_train} | val={expected_val} | prebuild skipped")
                return
    except Exception:
        pass

    allf = pd.concat([tr[["img"]], va[["img"]]], ignore_index=True).drop_duplicates("img")
    ds = _VahadaneCacheBuildDataset(allf, cfg)
    nw = max(1, min(int(getattr(cfg, "stain_cache_build_workers", 4)), int(getattr(cfg, "num_workers", 4))))
    kw = dict(batch_size=32, shuffle=False, num_workers=nw, pin_memory=False,
              worker_init_fn=core.fast_worker_init_fn)
    if nw > 0:
        kw.update(persistent_workers=True, prefetch_factor=2)
    dl = DataLoader(ds, **kw)

    print(f"[VAHADANE CACHE PREBUILD] checking/building {len(ds)} unique Train+Val patches with {nw} workers")
    hits = built = done = 0
    bar = tqdm(total=len(ds), desc="VAHADANE CACHE")
    try:
        for flags in dl:
            vals = flags.tolist() if hasattr(flags, "tolist") else list(flags)
            hits += sum(int(v) == 1 for v in vals)
            built += sum(int(v) == 0 for v in vals)
            done += len(vals)
            bar.update(len(vals))
            bar.set_postfix(hit=hits, built=built)
    finally:
        bar.close()

    ns.mkdir(parents=True, exist_ok=True)
    payload = {
        "train": expected_train,
        "val": expected_val,
        "unique": int(len(ds)),
        "cache_hits": int(hits),
        "newly_built": int(built),
        "complete": True,
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    core.atomic_json(marker, payload)
    print(f"[VAHADANE CACHE COMPLETE] unique={len(ds)} | reused={hits} | built={built}")

def resumable_train_model(cfg, run_name: Optional[str] = None) -> Path:
    tr, va, te = core.build_frames(cfg)
    # Performance-only optimization: finish deterministic Vahadane preprocessing
    # before epoch timing begins. Existing cache entries are reused; interruption
    # is naturally resumable because completed PNGs remain on disk.
    _prebuild_final_vahadane_cache(cfg, tr, va)
    train_ds = core.PandaPatchDataset(tr, cfg, "train", augment=cfg.use_strong_stain_aug, normalize=True)
    val_ds = core.PandaPatchDataset(va, cfg, "val", augment=False, normalize=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        # Throughput-only CUDA kernel settings; architecture/hyperparameters stay unchanged.
        torch.backends.cudnn.benchmark = True
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    model = core.UNetEfficientNetB4(1, cfg.dropout_p, cfg.use_coords, cfg.pretrained, cfg.coord_embed_dim).to(device)
    if cfg.channels_last and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    lossfn = core.CombinedLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and device.type == "cuda")

    # Stable run directory is essential for restart-after-power-loss.
    run_name = run_name or "final_main_training"
    out = cfg.runs_dir / run_name
    out.mkdir(parents=True, exist_ok=True)
    core.atomic_json(out / "config.json", {**core.asdict(cfg), "resumable_training": True})

    resume_pt = out / "resume_training.pt"
    train_phase_pt = out / "train_phase_complete.pt"
    val_progress_json = out / "val_progress.json"

    best = -1.0
    bad = 0
    history = []
    epoch = 1
    phase = "train"
    next_batch = 0
    meter = core.PixelMeter()
    loss_sum = 0.0
    nseen = 0

    wait_for_storage_stable(cfg, "checking final-training resume checkpoint")
    if resume_pt.exists():
        ck = retry_same_unit(
            cfg,
            lambda: torch.load(resume_pt, map_location=device),
            "load final-training resume checkpoint",
            required_paths=[resume_pt],
        )
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        try: scaler.load_state_dict(ck.get("scaler", {}))
        except Exception: pass
        epoch = int(ck.get("epoch", 1))
        phase = str(ck.get("phase", "train"))
        next_batch = int(ck.get("next_batch", 0))
        best = float(ck.get("best", -1.0))
        bad = int(ck.get("bad", 0))
        history = ck.get("history", [])
        meter = _meter_from_dict(ck.get("meter"))
        loss_sum = float(ck.get("loss_sum", 0.0))
        nseen = int(ck.get("n", 0))
        _restore_rng(ck.get("rng"))
        print(f"[TRAIN RESUME] epoch={epoch}, phase={phase}, next_batch={next_batch}")

    checkpoint_every = int(getattr(cfg, "train_checkpoint_every_batches", 100))

    while epoch <= cfg.num_epochs:
        if phase == "train":
            dl = _make_fixed_loader(train_ds, cfg, epoch, next_batch, shuffle=True)
            model.train()
            processed_batch = next_batch
            try:
                bar = tqdm(dl, desc=f"TRAIN epoch {epoch}/{cfg.num_epochs} resume@{next_batch}")
                for local_i, (x,y,lv,coords,ids) in enumerate(bar):
                    batch_idx = next_batch + local_i
                    x=x.to(device,non_blocking=True); y=y.to(device,non_blocking=True); coords=coords.to(device,non_blocking=True)
                    if cfg.channels_last and device.type=="cuda":
                        x=x.contiguous(memory_format=torch.channels_last)
                    opt.zero_grad(set_to_none=True)
                    with torch.amp.autocast("cuda",enabled=cfg.amp and device.type=="cuda"):
                        z=model(x,coords); loss=lossfn(z,y)
                    scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
                    loss_sum += float(loss.item()) * len(x); nseen += len(x)
                    meter.update(z.detach(),y,cfg.threshold)
                    processed_batch = batch_idx + 1
                    bar.set_postfix(loss=f"{loss.item():.4f}")

                    if processed_batch % checkpoint_every == 0:
                        st = _training_state(model,opt,scaler,epoch,"train",processed_batch,best,bad,history,
                                             meter,loss_sum,nseen)
                        retry_same_unit(cfg, lambda st=st: atomic_torch_save(resume_pt, st),
                                        f"training checkpoint epoch {epoch} batch {processed_batch}")
            except Exception as e:
                if not looks_like_io_error(e):
                    raise
                # If Python is alive, the model/optimizer updates already made are still in RAM.
                # Wait for SSD, then save the current in-RAM state so none of those completed batches are lost.
                print(f"[TRAIN I/O INTERRUPTION] epoch {epoch}, completed batch {processed_batch}: {e}")
                wait_for_storage(cfg, f"training epoch {epoch}")
                st = _training_state(model,opt,scaler,epoch,"train",processed_batch,best,bad,history,
                                     meter,loss_sum,nseen)
                retry_same_unit(cfg, lambda: atomic_torch_save(resume_pt, st),
                                f"post-reconnect training checkpoint epoch {epoch}")
                next_batch = processed_batch
                continue

            # Exact phase boundary: Training part of epoch is durably complete BEFORE Validation starts.
            tm = meter.metrics(); tm["loss"] = loss_sum/max(nseen,1); tm["n_patches"]=nseen
            phase_state = _training_state(model,opt,scaler,epoch,"val",0,best,bad,history,None,0.0,0)
            phase_state["train_metrics"] = tm
            retry_same_unit(cfg, lambda: atomic_torch_save(train_phase_pt, phase_state),
                            f"train-phase complete epoch {epoch}")
            retry_same_unit(cfg, lambda: atomic_torch_save(resume_pt, phase_state),
                            f"resume phase switch epoch {epoch}")
            phase = "val"; next_batch = 0
            meter = core.PixelMeter(); loss_sum=0.0; nseen=0

        if phase == "val":
            # Always reload the exact model frozen at the train/val boundary.
            phase_ck = retry_same_unit(
                cfg,
                lambda: torch.load(train_phase_pt, map_location=device),
                f"load train/val boundary checkpoint epoch {epoch}",
                required_paths=[train_phase_pt],
            )
            model.load_state_dict(phase_ck["model"])
            opt.load_state_dict(phase_ck["optimizer"])
            try: scaler.load_state_dict(phase_ck.get("scaler", {}))
            except Exception: pass
            tm = phase_ck["train_metrics"]

            vmeter = core.PixelMeter()
            vloss = 0.0; vn = 0; vnext = 0
            if val_progress_json.exists():
                try:
                    vp = json.load(open(val_progress_json,"r",encoding="utf-8"))
                    if int(vp.get("epoch",-1)) == epoch:
                        vnext = int(vp.get("next_batch",0))
                        vmeter = _meter_from_dict(vp.get("meter"))
                        vloss = float(vp.get("loss_sum",0.0)); vn=int(vp.get("n",0))
                        print(f"[VAL RESUME] epoch={epoch}, next_batch={vnext}")
                except Exception:
                    pass

            val_dl = _make_fixed_loader(val_ds, cfg, epoch, vnext, shuffle=False)
            model.eval()
            processed = vnext
            try:
                with torch.inference_mode():
                    for local_i,(x,y,lv,coords,ids) in enumerate(tqdm(val_dl,desc=f"VAL epoch {epoch} resume@{vnext}")):
                        batch_idx = vnext + local_i
                        x=x.to(device,non_blocking=True); y=y.to(device,non_blocking=True); coords=coords.to(device,non_blocking=True)
                        if cfg.channels_last and device.type=="cuda":
                            x=x.contiguous(memory_format=torch.channels_last)
                        with torch.amp.autocast("cuda",enabled=cfg.amp and device.type=="cuda"):
                            z=model(x,coords); loss=lossfn(z,y)
                        vloss += float(loss.item())*len(x); vn += len(x); vmeter.update(z,y,cfg.threshold)
                        processed = batch_idx + 1
                        # Validation progress is lightweight, so save after EVERY batch.
                        atomic_json(val_progress_json,{
                            "epoch":epoch,"next_batch":processed,"meter":_meter_to_dict(vmeter),
                            "loss_sum":vloss,"n":vn
                        })
            except Exception as e:
                if not looks_like_io_error(e):
                    raise
                print(f"[VAL I/O INTERRUPTION] epoch {epoch}, completed batch {processed}: {e}")
                wait_for_storage(cfg, f"validation epoch {epoch}")
                atomic_json(val_progress_json,{
                    "epoch":epoch,"next_batch":processed,"meter":_meter_to_dict(vmeter),
                    "loss_sum":vloss,"n":vn
                })
                phase="val"; next_batch=0
                continue

            vm = vmeter.metrics(); vm["loss"]=vloss/max(vn,1); vm["n_patches"]=vn
            row={"epoch":epoch,**{f"train_{k}":v for k,v in tm.items()},**{f"val_{k}":v for k,v in vm.items()}}
            # Avoid duplicate epoch row after a crash exactly at commit.
            history=[h for h in history if int(h.get("epoch",-1)) != epoch]
            history.append(row)
            retry_same_unit(cfg, lambda: atomic_csv(out/"history.csv",pd.DataFrame(history)),
                            f"history epoch {epoch}")

            state={"epoch":epoch,"model":model.state_dict(),"optimizer":opt.state_dict(),
                   "scaler":scaler.state_dict(),"val":vm,"config":core.asdict(cfg),
                   "best":best,"bad":bad,"history":history}
            retry_same_unit(cfg, lambda: atomic_torch_save(out/"last.pt",state), f"last.pt epoch {epoch}")

            print(f"Epoch {epoch}: train Dice={tm['dice']:.4f} | val Dice={vm['dice']:.4f}")
            if vm["dice"] > best:
                best=float(vm["dice"]); bad=0
                state["best"]=best; state["bad"]=bad
                retry_same_unit(cfg, lambda: atomic_torch_save(out/"best.pt",state), f"best.pt epoch {epoch}")
            else:
                bad += 1

            # Commit the completed epoch and point resume to the NEXT epoch.
            epoch_done = _training_state(model,opt,scaler,epoch+1,"train",0,best,bad,history,
                                         core.PixelMeter(),0.0,0)
            retry_same_unit(cfg, lambda: atomic_torch_save(resume_pt,epoch_done),
                            f"epoch {epoch} commit")
            try: val_progress_json.unlink(missing_ok=True)
            except Exception: pass

            core.atomic_json(cfg.work/"latest_run.json",{"run_dir":str(out),"best_val_dice":best})
            if bad >= cfg.early_stopping_patience:
                print("Early stopping")
                break

            epoch += 1; phase="train"; next_batch=0
            meter=core.PixelMeter(); loss_sum=0.0; nseen=0

    core.atomic_json(cfg.work/"latest_run.json",{"run_dir":str(out),"best_val_dice":best})
    return out


# -----------------------------------------------------------------------------
# Pilot experiment restart: completed experiment directories are reused
# -----------------------------------------------------------------------------

def install_legacy_pilot_resume(legacy):
    original = legacy.run_pilot_training

    def wrapped(cfg, name, train_frame, val_frame, normalization="vahadane",
                use_coords=True, encoder="efficientnet_b4", epochs=None):
        out = legacy.paper_root(cfg)/"experiments"/name
        summary = out/"summary.json"
        if summary.exists():
            try:
                print(f"[PILOT RESUME] Reusing completed experiment: {name}")
                return json.load(open(summary,"r",encoding="utf-8"))
            except Exception:
                pass
        kwargs = {}
        if epochs is not None:
            kwargs["epochs"] = epochs
        return original(cfg,name,train_frame,val_frame,normalization,use_coords,encoder,**kwargs)

    legacy.run_pilot_training = wrapped


# -----------------------------------------------------------------------------
# Installer
# -----------------------------------------------------------------------------

def install(legacy_module=None):
    final.build_coordinate_pool = resumable_build_coordinate_pool
    final.materialize_from_pool = resumable_materialize_from_pool
    core.extract_features_for_df = resumable_extract_features_for_df
    core.train_model = resumable_train_model
    if legacy_module is not None:
        install_legacy_pilot_resume(legacy_module)
