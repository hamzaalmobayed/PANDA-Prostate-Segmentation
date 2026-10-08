#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
External PRECISE full-WSI test for the existing PANDA prostate segmentation project.

ADD THIS FILE NEXT TO:
  run_full_panda_paper.py
  panda_final_pipeline.py
  panda_camelyon_style_pipeline.py
  panda_stream_paper.py

RUN:
  python run_external_precise_wsi_test.py

What it does
------------
1) Reads the public PRECISE Zenodo ZIP remotely using HTTP Range requests.
2) Downloads ONLY H&E OME-TIFF WSIs + their H&E masks (not IHC).
3) STREAM MODE: for each slide, download/reuse its H&E+mask, test it immediately, save results/visual, then move to the next slide.
4) Loads the existing PANDA best model and TRAIN-derived Vahadane target.
5) Uses the PANDA validation-selected threshold if available.
6) Runs dense L0/L1/L2-equivalent inference using PRECISE pyramid levels
   1x / 4x / 16x, 512 patch, stride 128, one shared model.
7) Support-aware fusion over available levels.
8) Evaluates only directly comparable PRECISE labels by default:
      Tumor (1) = positive
      Benign gland (2) + Stroma (7) = negative
      Background (0), Artifact (3), HGPIN (4), IDC-P (5), AIP (6) = ignored
   This avoids silently changing the PANDA binary target definition.
9) Saves per-slide metrics, global metrics, bootstrap CI, patch-level pooled
   counts, per-level metrics, and a combined visual for every WSI.
10) Applies stronger H&E-only HSV+brightness tissue-constrained background suppression.\n11) Crash/SSD-safe resume:
   - download: completed files are reused; .part files are preserved and continued
   - inference: completed slides are cached and skipped
   - inference checkpoints: each level is checkpointed periodically by chunk, so an
     SSD disconnect/restart resumes from the latest saved chunk instead of restarting
     the whole WSI
   - completed levels are reused when the next level is interrupted
   - metrics/progress CSV and JSON state files are written atomically
   - reporting/visuals already generated are reused

Important
---------
Zenodo publishes PRECISE as one 55.6-GB data.zip. This script DOES NOT download
the whole archive. It requests only H&E members from the remote ZIP.

Expected H&E-only payload from the Zenodo archive preview:
  ~28.0 GB H&E OME-TIFFs
  ~0.25 GB H&E masks
  ~28.2 GB total, plus results/cache overhead.

Environment overrides
---------------------
PRECISE_MAX_SLIDES=10       # optional compact test; default 0 = all 27
PRECISE_INFER_BATCH=16
PRECISE_PREP_WORKERS=4
PRECISE_DOWNLOAD_RETRIES=20
PRECISE_CHECKPOINT_EVERY_CHUNKS=10  # rolling inference checkpoint cadence
"""

import os
import sys
import gc
import json
import math
import time
import shutil
import subprocess
import traceback
import zipfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, List, Tuple

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

def _ensure(pkg, import_name=None):
    name = import_name or pkg
    try:
        return __import__(name)
    except Exception:
        print(f"[AUTO-INSTALL] {pkg}", flush=True)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])
        return __import__(name)

# Remote ZIP selective downloader.
try:
    from remotezip import RemoteZip
except Exception:
    _ensure("remotezip")
    from remotezip import RemoteZip

try:
    import tifffile
except Exception:
    _ensure("tifffile")
    import tifffile

try:
    import zarr
except Exception:
    _ensure("zarr")
    import zarr

try:
    import imagecodecs  # noqa: F401
except Exception:
    _ensure("imagecodecs")
    import imagecodecs  # noqa: F401

import numpy as np
import pandas as pd
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch

from panda_final_pipeline import FinalConfig
import panda_camelyon_style_pipeline as core
import panda_stream_paper as paper

ZENODO_RECORD = "20721779"
REMOTE_ZIP_URL = f"https://zenodo.org/records/{ZENODO_RECORD}/files/data.zip?download=1"
WORK_ROOT = Path(r"D:\PANDA_PROSTATE")
EXT_ROOT = WORK_ROOT / "external_precise_wsi"
DATA_ROOT = EXT_ROOT / "dataset_he_only"
RESULT_ROOT = EXT_ROOT / "results"
CACHE_ROOT = EXT_ROOT / "wsi_cache"
STATE_ROOT = EXT_ROOT / "state"

PATCH = 512
STRIDE = 128
LEVEL_DOWNSAMPLES = {0: 1, 1: 4, 2: 16}
OME_LEVEL_INDEX = {0: 0, 1: 2, 2: 4}
EVAL_DS = 16.0
BOOTSTRAP_ITERS = 1000

# Strict comparability mapping.
POSITIVE_LABELS = {1}
NEGATIVE_LABELS = {2, 7}
VALID_LABELS = POSITIVE_LABELS | NEGATIVE_LABELS

def _wait_parent_ready(path: Path, seconds=5):
    """Wait until the drive containing *path* is writable again."""
    while True:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            probe = path.parent / ".resume_write_probe.tmp"
            with open(probe, "wb") as f:
                f.write(b"ok")
                f.flush()
                os.fsync(f.fileno())
            probe.unlink(missing_ok=True)
            return
        except Exception:
            print(f"[SSD WAIT] Cannot write {path.parent}. Waiting for reconnect...", flush=True)
            time.sleep(seconds)


def atomic_json(path: Path, obj: Any):
    """Atomic + reconnect-safe JSON write."""
    while True:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + ".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(obj, f, indent=2, ensure_ascii=False, default=str)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
            return
        except Exception as e:
            print(f"[STATE WRITE RETRY] {type(e).__name__}: {e}", flush=True)
            _wait_parent_ready(path)


def atomic_csv(df: pd.DataFrame, path: Path):
    """Atomic + reconnect-safe CSV write."""
    while True:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + ".tmp")
            df.to_csv(tmp, index=False)
            # Force bytes to disk before replacing the committed copy.
            with open(tmp, "rb+") as f:
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
            return
        except Exception as e:
            print(f"[CSV WRITE RETRY] {type(e).__name__}: {e}", flush=True)
            _wait_parent_ready(path)


def drive_ready():
    try:
        WORK_ROOT.mkdir(parents=True, exist_ok=True)
        probe = WORK_ROOT / ".drive_probe.tmp"
        with open(probe, "wb") as f:
            f.write(b"ok")
            f.flush()
            os.fsync(f.fileno())
        probe.unlink(missing_ok=True)
        return True
    except Exception:
        return False


def wait_for_drive(seconds=10):
    print("[SSD WAIT] D: unavailable. Waiting automatically...", flush=True)
    while not drive_ready():
        time.sleep(seconds)
    print("[SSD RECONNECTED] Continuing automatically.", flush=True)


def state(stage, **extra):
    atomic_json(
        STATE_ROOT / "external_precise_state.json",
        {"stage": stage, "time": time.strftime("%Y-%m-%d %H:%M:%S"), **extra},
    )


def _wanted_member(name: str) -> bool:
    n = name.replace("\\", "/").lower()
    return "/wsi_h-e/" in n and n.endswith(".ome.tif")

def _is_mask(name: str) -> bool:
    return "_mask.ome.tif" in name.lower()

def _slide_id_from_name(name: str) -> str:
    b = Path(name).name
    return b.replace("_h-e_mask.ome.tif", "").replace("_h-e.ome.tif", "")


def precise_remote_index() -> pd.DataFrame:
    """
    Read only the PRECISE remote ZIP index. No WSI is downloaded here.
    The pipeline later works slide-by-slide:
    download/reuse current WSI+mask -> test -> save -> next.
    """
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    STATE_ROOT.mkdir(parents=True, exist_ok=True)

    state("before_remote_zip_index", url=REMOTE_ZIP_URL)
    retries = int(os.environ.get("PRECISE_DOWNLOAD_RETRIES", "20"))

    infos = None
    last = None
    for attempt in range(1, retries + 1):
        try:
            with RemoteZip(REMOTE_ZIP_URL) as rz:
                infos = [x for x in rz.infolist() if _wanted_member(x.filename)]
            break
        except Exception as e:
            last = e
            print(f"[REMOTE ZIP INDEX RETRY {attempt}/{retries}] {type(e).__name__}: {e}", flush=True)
            if not drive_ready():
                wait_for_drive()
            time.sleep(min(30, 3 * attempt))
    if infos is None:
        raise RuntimeError(f"Could not read PRECISE remote ZIP index: {last}")

    by_id: Dict[str, Dict[str, Any]] = {}
    for info in infos:
        sid = _slide_id_from_name(info.filename)
        by_id.setdefault(sid, {})
        by_id[sid]["mask" if _is_mask(info.filename) else "image"] = info

    pairs = [(sid, d["image"], d["mask"]) for sid, d in sorted(by_id.items())
             if "image" in d and "mask" in d]

    max_slides = int(os.environ.get("PRECISE_MAX_SLIDES", "0"))
    if max_slides > 0:
        pairs = sorted(pairs, key=lambda x: x[1].file_size + x[2].file_size)[:max_slides]
        pairs = sorted(pairs, key=lambda x: x[0])

    total_bytes = sum(im.file_size + ma.file_size for _, im, ma in pairs)
    print(
        f"[PRECISE SELECT] {len(pairs)} H&E WSIs + masks | "
        f"total selected payload={total_bytes/1024**3:.2f} GB | STREAM MODE",
        flush=True,
    )
    state("remote_zip_index_complete", n_pairs=len(pairs), selected_bytes=total_bytes)

    rows = []
    for sid, im_info, ma_info in pairs:
        out_slide_dir = DATA_ROOT / sid
        local_im = out_slide_dir / Path(im_info.filename).name
        local_ma = out_slide_dir / Path(ma_info.filename).name
        rows.append({
            "image_id": sid,
            "image_member": im_info.filename,
            "mask_member": ma_info.filename,
            "image_size": int(im_info.file_size),
            "mask_size": int(ma_info.file_size),
            "image_path": str(local_im),
            "mask_path": str(local_ma),
        })

    df = pd.DataFrame(rows)
    atomic_csv(df, EXT_ROOT / "precise_he_manifest.csv")
    return df


def _download_one_member(info_name: str, expected_size: int, dst: Path, idx: int, total: int):
    """
    Ensure one ZIP member is present locally.

    Resume behavior
    ---------------
    * A complete destination is reused.
    * An interrupted ``.part`` file is NEVER deleted merely because a retry starts.
    * On retry/restart the remote member is reopened, the already committed number
      of uncompressed bytes is skipped, and writing continues in append mode.
    * If the SSD disappears during the copy, the current .part file is kept and the
      function waits for D: to reconnect before continuing.

    Note: ZIP compression means the remote stream may still need to read/decompress
    bytes up to the resume offset, but already-written local bytes are preserved and
    are not rewritten from zero.
    """
    retries = int(os.environ.get("PRECISE_DOWNLOAD_RETRIES", "20"))
    dst.parent.mkdir(parents=True, exist_ok=True)

    # Completed file => reuse.
    if dst.exists():
        try:
            if dst.stat().st_size == int(expected_size):
                print(f"[DOWNLOAD REUSE {idx}/{total}] {dst.name}", flush=True)
                return
        except OSError:
            wait_for_drive()

    part = dst.with_suffix(dst.suffix + ".part")

    # If a stale/oversized partial exists, only then reset it.
    if part.exists():
        try:
            if part.stat().st_size > int(expected_size):
                print(f"[DOWNLOAD PART RESET] {part.name} larger than expected", flush=True)
                part.unlink(missing_ok=True)
        except OSError:
            wait_for_drive()

    attempt = 0
    while True:
        attempt += 1
        try:
            resume_from = int(part.stat().st_size) if part.exists() else 0
            if resume_from == int(expected_size):
                os.replace(part, dst)
                print(f"[DOWNLOAD COMMIT {idx}/{total}] {dst.name}", flush=True)
                return

            print(
                f"[DOWNLOAD {idx}/{total}] {info_name} | "
                f"{expected_size/1024**2:.1f} MB | resume={resume_from/1024**2:.1f} MB",
                flush=True,
            )

            with RemoteZip(REMOTE_ZIP_URL) as rz:
                info = rz.getinfo(info_name)
                with rz.open(info) as src:
                    # Reopen the member and advance to the last durable local byte.
                    remaining_skip = resume_from
                    skip_bar = None
                    if remaining_skip > 0:
                        skip_bar = tqdm(
                            total=resume_from,
                            unit="B", unit_scale=True, unit_divisor=1024,
                            desc=f"resume-skip {Path(info_name).name}",
                            leave=False,
                        )
                    while remaining_skip > 0:
                        chunk = src.read(min(8 * 1024 * 1024, remaining_skip))
                        if not chunk:
                            raise IOError(
                                f"Remote member ended while seeking resume offset "
                                f"{resume_from} bytes"
                            )
                        remaining_skip -= len(chunk)
                        if skip_bar is not None:
                            skip_bar.update(len(chunk))
                    if skip_bar is not None:
                        skip_bar.close()

                    mode = "ab" if resume_from > 0 else "wb"
                    with open(part, mode) as f:
                        bar = tqdm(
                            total=int(expected_size),
                            initial=resume_from,
                            unit="B",
                            unit_scale=True,
                            unit_divisor=1024,
                            desc=Path(info_name).name,
                        )
                        since_sync = 0
                        sync_every = int(os.environ.get(
                            "PRECISE_DOWNLOAD_FSYNC_BYTES",
                            str(64 * 1024 * 1024),
                        ))
                        while True:
                            chunk = src.read(8 * 1024 * 1024)
                            if not chunk:
                                break
                            f.write(chunk)
                            bar.update(len(chunk))
                            since_sync += len(chunk)
                            if since_sync >= sync_every:
                                f.flush()
                                os.fsync(f.fileno())
                                since_sync = 0
                        f.flush()
                        os.fsync(f.fileno())
                        bar.close()

            got = part.stat().st_size
            if got != int(expected_size):
                raise IOError(f"size mismatch {got} != {expected_size}")

            os.replace(part, dst)
            print(f"[DOWNLOAD COMPLETE {idx}/{total}] {dst.name}", flush=True)
            return

        except KeyboardInterrupt:
            print(f"[DOWNLOAD STOPPED] partial kept: {part}", flush=True)
            raise
        except Exception as e:
            print(
                f"[DOWNLOAD RESUME RETRY {attempt}] {type(e).__name__}: {e}",
                flush=True,
            )
            # DO NOT delete .part. Wait for SSD if necessary, then retry from it.
            if not drive_ready():
                wait_for_drive()
            # Keep retrying beyond PRECISE_DOWNLOAD_RETRIES when a partial file exists;
            # a transient disconnect should never force a restart from zero.
            if attempt >= retries and not part.exists():
                raise RuntimeError(f"Failed downloading {info_name}") from e
            time.sleep(min(30, 3 * max(1, attempt)))


def ensure_slide_pair_downloaded(row, idx: int, total: int) -> pd.Series:
    """Download/reuse only the current WSI and mask, then return local paths."""
    sid = str(row["image_id"])
    image_path = Path(row["image_path"])
    mask_path = Path(row["mask_path"])

    state("before_slide_download", image_id=sid, index=idx, total=total)

    _download_one_member(
        str(row["image_member"]), int(row["image_size"]),
        image_path, idx, total
    )
    _download_one_member(
        str(row["mask_member"]), int(row["mask_size"]),
        mask_path, idx, total
    )

    state("slide_download_complete", image_id=sid, index=idx, total=total)
    return pd.Series({
        "image_id": sid,
        "image_path": str(image_path),
        "mask_path": str(mask_path),
    })


class OmePyramid:
    def __init__(self, path: str):
        self.path = str(path)
        self.tf = tifffile.TiffFile(self.path)
        self.series = self.tf.series[0]
        self.levels = list(self.series.levels)
        if len(self.levels) < 5:
            raise RuntimeError(f"Expected >=5 OME pyramid levels, got {len(self.levels)}: {path}")
        self._stores = {}
        self._arrays = {}

    def arr(self, level_index: int):
        """
        Return the actual Zarr ARRAY for this TIFF pyramid level.

        With some tifffile/zarr versions, `level.aszarr()` opens directly as an
        Array. With others it opens as a Group containing the array (often key
        "0"). Handle both layouts so PRECISE OME-TIFFs work across installed
        zarr/tifffile versions.
        """
        if level_index not in self._arrays:
            store = self.levels[level_index].aszarr()
            self._stores[level_index] = store
            root = zarr.open(store, mode="r")

            if hasattr(root, "shape"):
                arr = root
            else:
                # zarr.Group case: find the first real array recursively.
                arr = None

                # Fast/common case: key "0".
                try:
                    cand = root["0"]
                    if hasattr(cand, "shape"):
                        arr = cand
                except Exception:
                    pass

                # General recursive fallback for different OME/Zarr layouts.
                if arr is None:
                    def _find_array(group):
                        try:
                            for key in group.array_keys():
                                obj = group[key]
                                if hasattr(obj, "shape"):
                                    return obj
                        except Exception:
                            pass
                        try:
                            for key in group.group_keys():
                                found = _find_array(group[key])
                                if found is not None:
                                    return found
                        except Exception:
                            pass
                        return None

                    arr = _find_array(root)

                if arr is None:
                    try:
                        keys = list(root.keys())
                    except Exception:
                        keys = []
                    raise RuntimeError(
                        f"Could not locate an array inside OME-Zarr level "
                        f"{level_index}. Root keys={keys}"
                    )

            self._arrays[level_index] = arr
            try:
                print(
                    f"[OME LEVEL READY] {Path(self.path).name} | "
                    f"level_index={level_index} | shape={tuple(arr.shape)} | dtype={arr.dtype}",
                    flush=True,
                )
            except Exception:
                pass

        return self._arrays[level_index]

    @property
    def shape0(self):
        a = self.arr(0)
        return int(a.shape[0]), int(a.shape[1])

    def read_patch(self, level_index: int, x: int, y: int, size: int, rgb=True):
        a = self.arr(level_index)
        h, w = int(a.shape[0]), int(a.shape[1])
        x2, y2 = min(w, x + size), min(h, y + size)
        out = np.asarray(a[y:y2, x:x2])
        if rgb:
            if out.ndim == 2:
                out = np.repeat(out[..., None], 3, axis=2)
            if out.shape[-1] > 3:
                out = out[..., :3]
            pad = np.full((size, size, 3), 255, dtype=np.uint8)
        else:
            if out.ndim == 3:
                out = out[..., 0]
            pad = np.zeros((size, size), dtype=np.uint8)
        pad[:out.shape[0], :out.shape[1]] = out
        return pad

    def thumbnail(self, level_index=4, max_size=2048):
        a = np.asarray(self.arr(level_index))
        if a.ndim == 2:
            a = np.repeat(a[..., None], 3, axis=2)
        if a.shape[-1] > 3:
            a = a[..., :3]
        h, w = a.shape[:2]
        scale = min(1.0, max_size / max(h, w))
        if scale < 1:
            a = cv2.resize(a, (max(1, int(w*scale)), max(1, int(h*scale))), interpolation=cv2.INTER_AREA)
        return a.astype(np.uint8)

    def close(self):
        for s in self._stores.values():
            try:
                s.close()
            except Exception:
                pass
        self.tf.close()

def load_threshold() -> float:
    candidates = [
        WORK_ROOT / "paper_results" / "tables" / "best_threshold.json",
        WORK_ROOT / "paper-results" / "tables" / "best_threshold.json",
    ]
    for p in candidates:
        if p.exists():
            try:
                d = json.load(open(p, "r", encoding="utf-8"))
                thr = float(d["threshold"])
                print(f"[THRESHOLD] PANDA validation-selected threshold={thr:.4f} from {p}", flush=True)
                return thr
            except Exception:
                pass
    print("[THRESHOLD WARNING] best_threshold.json not found -> using 0.5", flush=True)
    return 0.5

def metrics(tp, fp, tn, fn):
    eps = 1e-9
    return {
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "dice": float(2*tp/(2*tp+fp+fn+eps)),
        "iou": float(tp/(tp+fp+fn+eps)),
        "precision": float(tp/(tp+fp+eps)),
        "recall": float(tp/(tp+fn+eps)),
        "specificity": float(tn/(tn+fp+eps)),
        "accuracy": float((tp+tn)/(tp+tn+fp+fn+eps)),
    }

def counts(pred, gt, valid):
    p = pred.astype(bool)[valid]
    g = gt.astype(bool)[valid]
    return int((p&g).sum()), int((p&~g).sum()), int((~p&~g).sum()), int((~p&g).sum())

def make_tissue_mask(rgb):
    gray = rgb.mean(axis=2)
    tissue = (gray < 245).astype(np.uint8)
    k = np.ones((3,3), np.uint8)
    tissue = cv2.morphologyEx(tissue, cv2.MORPH_CLOSE, k, iterations=1)
    return tissue.astype(bool)


def make_tissue_mask_conservative(rgb):
    """
    Strong GT-independent H&E tissue mask for background suppression.

    Uses brightness + color saturation rather than grayscale alone:
      - excludes near-white / low-saturation background
      - keeps pale real tissue through a darker-pixel fallback
      - removes tiny isolated artifacts
      - closes small holes/gaps
      - dilates slightly so true tissue boundaries are not clipped

    Multiple disconnected tissue fragments are preserved.
    """
    rgb = np.asarray(rgb)
    if rgb.ndim == 2:
        rgb = np.repeat(rgb[..., None], 3, axis=2)
    if rgb.shape[-1] > 3:
        rgb = rgb[..., :3]
    rgb = rgb.astype(np.uint8, copy=False)

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    sat = hsv[..., 1].astype(np.uint8)
    val = hsv[..., 2].astype(np.uint8)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    # Main tissue rule:
    #   colored H&E pixels that are not near-white
    # OR genuinely darker tissue even if weakly saturated.
    tissue = (((val < 247) & (sat >= 14)) | (gray < 225)).astype(np.uint8)

    # Remove tiny specks/artifacts but preserve multiple biopsy fragments.
    h, w = tissue.shape
    min_area = max(16, int(round(h * w * 0.00002)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(tissue, connectivity=8)
    clean = np.zeros_like(tissue)
    for lab in range(1, n):
        if int(stats[lab, cv2.CC_STAT_AREA]) >= min_area:
            clean[labels == lab] = 1

    k3 = np.ones((3, 3), np.uint8)
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, k3, iterations=2)
    clean = cv2.morphologyEx(clean, cv2.MORPH_OPEN, k3, iterations=1)
    clean = cv2.dilate(clean, k3, iterations=1)

    return clean.astype(bool)

def normalize_many(rgbs, proc, workers):
    if not rgbs:
        return []
    if workers <= 1:
        return [proc.transform(x) for x in rgbs]
    # Separate processor per worker thread because stain normalizers can hold mutable state.
    target_path = str(FinalConfig().stain_target_path)
    import threading
    tls = threading.local()
    def one(rgb):
        p = getattr(tls, "proc", None)
        if p is None:
            target = np.array(Image.open(target_path).convert("RGB"), dtype=np.uint8, copy=True)
            p = core.VahadaneProcessor(target)
            tls.proc = p
        return p.transform(rgb)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(one, rgbs))

@torch.inference_mode()
def predict_batch(model, device, rgbs, coords_norm, cfg, batch_size):
    outputs = []
    cur = batch_size
    i = 0
    while i < len(rgbs):
        bs = min(cur, len(rgbs)-i)
        try:
            arr = np.stack(rgbs[i:i+bs], axis=0)
            x = torch.from_numpy(arr).permute(0,3,1,2).float().div_(255.0)
            c = torch.tensor(coords_norm[i:i+bs], dtype=torch.float32)
            if device.type == "cuda":
                x = x.pin_memory()
                c = c.pin_memory()
            x = x.to(device, non_blocking=device.type=="cuda")
            c = c.to(device, non_blocking=device.type=="cuda")
            if device.type == "cuda":
                x = x.contiguous(memory_format=torch.channels_last)
            with torch.amp.autocast("cuda", enabled=bool(cfg.amp and device.type=="cuda")):
                p = torch.sigmoid(model(x, c))[:,0]
            outputs.extend(p.float().cpu().numpy())
            i += bs
        except torch.cuda.OutOfMemoryError:
            if cur <= 1:
                raise
            torch.cuda.empty_cache()
            cur = max(1, cur//2)
            print(f"[CUDA OOM] batch fallback -> {cur}", flush=True)
    return outputs

def _level_checkpoint_path(sid: str, lv: int) -> Path:
    return CACHE_ROOT / "partial" / sid / f"L{lv}_checkpoint.npz"


def _save_level_checkpoint(
    sid: str,
    lv: int,
    next_chunk_start: int,
    sp: np.ndarray,
    sw: np.ndarray,
    patch_counts: Tuple[int, int, int, int],
):
    """Atomically save one resumable inference checkpoint."""
    path = _level_checkpoint_path(sid, lv)
    while True:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".npz.tmp")
            with open(tmp, "wb") as f:
                np.savez(
                    f,
                    next_chunk_start=np.array([int(next_chunk_start)], dtype=np.int64),
                    sp=sp.astype(np.float32, copy=False),
                    sw=sw.astype(np.float32, copy=False),
                    patch_counts=np.asarray(patch_counts, dtype=np.int64),
                )
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
            return
        except Exception as e:
            print(
                f"[INFER CHECKPOINT WRITE RETRY] {sid} L{lv}: "
                f"{type(e).__name__}: {e}",
                flush=True,
            )
            if not drive_ready():
                wait_for_drive()
            else:
                time.sleep(3)


def _load_level_checkpoint(sid: str, lv: int, shape: Tuple[int, int]):
    """Load a valid level checkpoint; return None if absent/corrupt/incompatible."""
    path = _level_checkpoint_path(sid, lv)
    if not path.exists():
        return None
    try:
        with np.load(path) as z:
            sp = z["sp"].astype(np.float32)
            sw = z["sw"].astype(np.float32)
            next_chunk_start = int(z["next_chunk_start"][0])
            pc = z["patch_counts"].astype(np.int64)
        if sp.shape != tuple(shape) or sw.shape != tuple(shape) or pc.size != 4:
            raise ValueError(
                f"checkpoint shape mismatch sp={sp.shape}, sw={sw.shape}, expected={shape}"
            )
        print(
            f"[INFER RESUME] {sid} L{lv} | next patch-index={next_chunk_start}",
            flush=True,
        )
        return next_chunk_start, sp, sw, tuple(map(int, pc.tolist()))
    except Exception as e:
        print(
            f"[CHECKPOINT INVALID] {sid} L{lv}: {type(e).__name__}: {e} | "
            f"restarting only this level",
            flush=True,
        )
        try:
            path.rename(path.with_suffix(path.suffix + ".corrupt"))
        except Exception:
            pass
        return None


def _completed_level_path(sid: str, lv: int) -> Path:
    return CACHE_ROOT / "partial" / sid / f"L{lv}_complete.npz"


def _save_completed_level(
    sid: str,
    lv: int,
    prob: np.ndarray,
    support: np.ndarray,
    patch_counts: np.ndarray,
):
    """Commit a fully completed level atomically so later levels can resume independently."""
    path = _completed_level_path(sid, lv)
    while True:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".npz.tmp")
            with open(tmp, "wb") as f:
                np.savez_compressed(
                    f,
                    prob=prob.astype(np.float16),
                    support=support.astype(np.uint8),
                    patch_counts=patch_counts.astype(np.int64),
                )
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
            # Once level is safely committed, the rolling checkpoint is unnecessary.
            _level_checkpoint_path(sid, lv).unlink(missing_ok=True)
            return
        except Exception as e:
            print(
                f"[LEVEL COMMIT RETRY] {sid} L{lv}: {type(e).__name__}: {e}",
                flush=True,
            )
            if not drive_ready():
                wait_for_drive()
            else:
                time.sleep(3)


def _load_completed_level(sid: str, lv: int, shape: Tuple[int, int]):
    path = _completed_level_path(sid, lv)
    if not path.exists():
        return None
    try:
        with np.load(path) as z:
            prob = z["prob"].astype(np.float32)
            support = z["support"].astype(bool)
            pc = z["patch_counts"].astype(np.int64)
        if prob.shape != tuple(shape) or support.shape != tuple(shape) or pc.size != 4:
            raise ValueError("completed level cache incompatible")
        print(f"[LEVEL REUSE] {sid} L{lv} complete", flush=True)
        return prob, support, pc
    except Exception as e:
        print(
            f"[LEVEL CACHE INVALID] {sid} L{lv}: {type(e).__name__}: {e}",
            flush=True,
        )
        try:
            path.rename(path.with_suffix(path.suffix + ".corrupt"))
        except Exception:
            pass
        return None



def complete_cache_candidates(sid: str):
    """Return complete WSI caches in preferred order (new resume-safe, then legacy v1)."""
    sid = str(sid)
    return [
        CACHE_ROOT / f"{sid}_ds16_precise_v2_resume.npz",
        CACHE_ROOT / f"{sid}_ds16_precise_v1.npz",
    ]


def find_complete_cache(sid: str):
    """Find a valid COMPLETE WSI cache, including caches produced by the old script."""
    required = {
        "gt", "annotated_valid", "valid", "fusion", "fusion_support",
        "prob_L0", "prob_L1", "prob_L2",
        "support_L0", "support_L1", "support_L2",
        "patch_counts_L0", "patch_counts_L1", "patch_counts_L2",
    }
    for p in complete_cache_candidates(sid):
        if not p.exists():
            continue
        try:
            with np.load(p) as z:
                keys = set(z.files)
                if not required.issubset(keys):
                    missing = sorted(required - keys)
                    print(f"[COMPLETE CACHE IGNORE] {sid} | {p.name} missing={missing}", flush=True)
                    continue
                # Force a few arrays to be read so a truncated/corrupt npz is not trusted.
                _ = z["fusion"].shape
                _ = z["gt"].shape
                _ = z["prob_L2"].shape
            return p
        except Exception as e:
            print(f"[COMPLETE CACHE INVALID] {sid} | {p.name}: {type(e).__name__}: {e}", flush=True)
    return None

def reconstruct_slide(row, model, device, stain, cfg, thr):
    sid = str(row["image_id"])
    cache = CACHE_ROOT / f"{sid}_ds16_precise_v2_resume.npz"

    # Fully completed WSI => reuse EITHER the new cache OR the old v1 cache.
    # This is the compatibility fix that prevents already-finished slides from
    # being recomputed after switching to the resume-safe script.
    completed = find_complete_cache(sid)
    if completed is not None:
        print(f"[WSI CACHE REUSE] {sid} | {completed.name}", flush=True)
        with np.load(completed) as z:
            return {k: z[k] for k in z.files}

    CACHE_ROOT.mkdir(parents=True, exist_ok=True)

    while True:
        im = ma = None
        try:
            im = OmePyramid(row["image_path"])
            ma = OmePyramid(row["mask_path"])
            h0, w0 = im.shape0
            H = math.ceil(h0 / EVAL_DS)
            W = math.ceil(w0 / EVAL_DS)

            gt_raw = np.asarray(ma.arr(4))
            if gt_raw.ndim == 3:
                gt_raw = gt_raw[..., 0]
            gt_raw = gt_raw[:H, :W]
            if gt_raw.shape != (H, W):
                gt_raw = cv2.resize(
                    gt_raw.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST
                )

            gt = np.isin(gt_raw, list(POSITIVE_LABELS))
            annotated_valid = np.isin(gt_raw, list(VALID_LABELS))

            tissue_rgb_eval = np.asarray(im.arr(4))
            if tissue_rgb_eval.ndim == 2:
                tissue_rgb_eval = np.repeat(tissue_rgb_eval[..., None], 3, axis=2)
            if tissue_rgb_eval.shape[-1] > 3:
                tissue_rgb_eval = tissue_rgb_eval[..., :3]
            tissue_eval = make_tissue_mask_conservative(
                tissue_rgb_eval.astype(np.uint8)
            )
            if tissue_eval.shape != (H, W):
                tissue_eval = cv2.resize(
                    tissue_eval.astype(np.uint8),
                    (W, H),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)

            level_prob = {}
            level_support = {}
            level_patch_counts = {}

            batch_size = int(os.environ.get("PRECISE_INFER_BATCH", "16"))
            workers = int(os.environ.get("PRECISE_PREP_WORKERS", "4"))
            checkpoint_every = max(
                1, int(os.environ.get("PRECISE_CHECKPOINT_EVERY_CHUNKS", "10"))
            )
            chunk_size = max(batch_size * 4, batch_size)

            for lv in (0, 1, 2):
                # Reuse a level that was fully committed before a disconnect.
                completed = _load_completed_level(sid, lv, (H, W))
                if completed is not None:
                    level_prob[lv], level_support[lv], level_patch_counts[lv] = completed
                    continue

                li = OME_LEVEL_INDEX[lv]
                ds = LEVEL_DOWNSAMPLES[lv]
                arr = im.arr(li)
                lh, lw = int(arr.shape[0]), int(arr.shape[1])

                thumb = im.thumbnail(min(4, len(im.levels) - 1), 2048)
                tissue_thumb = make_tissue_mask(thumb)
                th, tw = tissue_thumb.shape
                scale_x = lw / max(tw, 1)
                scale_y = lh / max(th, 1)

                # Deterministic coordinate list. Rebuilt quickly after restart.
                coords = []
                for y in range(0, max(1, lh - PATCH + 1), STRIDE):
                    for x in range(0, max(1, lw - PATCH + 1), STRIDE):
                        tx1, ty1 = int(x / scale_x), int(y / scale_y)
                        tx2, ty2 = int((x + PATCH) / scale_x), int(
                            (y + PATCH) / scale_y
                        )
                        tx2 = max(tx1 + 1, min(tw, tx2))
                        ty2 = max(ty1 + 1, min(th, ty2))
                        frac = (
                            float(tissue_thumb[ty1:ty2, tx1:tx2].mean())
                            if ty2 > ty1 and tx2 > tx1
                            else 0.0
                        )
                        if frac >= float(cfg.tissue_min_frac):
                            coords.append((x, y))

                # Resume the rolling accumulators for THIS level.
                ck = _load_level_checkpoint(sid, lv, (H, W))
                if ck is None:
                    next_start = 0
                    sp = np.zeros((H, W), np.float32)
                    sw = np.zeros((H, W), np.float32)
                    patch_tp = patch_fp = patch_tn = patch_fn = 0
                else:
                    next_start, sp, sw, pcs = ck
                    patch_tp, patch_fp, patch_tn, patch_fn = pcs
                    # Protect against a stale checkpoint from a different coordinate list.
                    if next_start < 0 or next_start > len(coords):
                        print(
                            f"[CHECKPOINT RANGE INVALID] {sid} L{lv}: "
                            f"{next_start}>{len(coords)}; restarting only level",
                            flush=True,
                        )
                        next_start = 0
                        sp.fill(0)
                        sw.fill(0)
                        patch_tp = patch_fp = patch_tn = patch_fn = 0

                starts = list(range(next_start, len(coords), chunk_size))
                pbar = tqdm(
                    starts,
                    desc=f"PRECISE {sid} L{lv}",
                    leave=False,
                    unit="chunk",
                )

                chunks_since_ckpt = 0
                for st in pbar:
                    chunk = coords[st : st + chunk_size]
                    raw_rgbs, cn, meta = [], [], []

                    for x, y in chunk:
                        rgb = im.read_patch(li, x, y, PATCH, True)
                        maskp = ma.read_patch(li, x, y, PATCH, False)
                        validp = np.isin(maskp, list(VALID_LABELS))
                        if not validp.any():
                            continue
                        gtp = np.isin(maskp, list(POSITIVE_LABELS))
                        cx0 = (x + PATCH / 2.0) * ds
                        cy0 = (y + PATCH / 2.0) * ds
                        tissuep = make_tissue_mask_conservative(rgb)
                        min_patch_tissue = float(
                            os.environ.get("PRECISE_MIN_PATCH_TISSUE", "0.05")
                        )
                        if float(tissuep.mean()) < min_patch_tissue:
                            continue
                        raw_rgbs.append(rgb)
                        cn.append((cx0 / max(w0, 1), cy0 / max(h0, 1)))
                        meta.append((x, y, gtp, validp, tissuep))

                    if raw_rgbs:
                        nrgb = normalize_many(raw_rgbs, stain, workers)
                        probs = predict_batch(
                            model, device, nrgb, cn, cfg, batch_size
                        )
                        for prob, (x, y, gtp, validp, tissuep) in zip(probs, meta):
                            prob = (
                                prob.astype(np.float32, copy=False)
                                * tissuep.astype(np.float32)
                            )
                            pp = prob >= thr
                            c = counts(pp, gtp, validp)
                            patch_tp += c[0]
                            patch_fp += c[1]
                            patch_tn += c[2]
                            patch_fn += c[3]

                            x0 = int((x * ds) / EVAL_DS)
                            y0 = int((y * ds) / EVAL_DS)
                            x1 = int(
                                math.ceil(((x + PATCH) * ds) / EVAL_DS)
                            )
                            y1 = int(
                                math.ceil(((y + PATCH) * ds) / EVAL_DS)
                            )
                            x1, y1 = min(W, x1), min(H, y1)
                            if x1 <= x0 or y1 <= y0:
                                continue
                            pr = cv2.resize(
                                prob.astype(np.float32),
                                (x1 - x0, y1 - y0),
                                interpolation=cv2.INTER_LINEAR,
                            )
                            sp[y0:y1, x0:x1] += pr
                            sw[y0:y1, x0:x1] += 1.0

                    next_after = min(len(coords), st + chunk_size)
                    chunks_since_ckpt += 1

                    # Rolling checkpoint: at most checkpoint_every chunks are lost.
                    if (
                        chunks_since_ckpt >= checkpoint_every
                        or next_after >= len(coords)
                    ):
                        _save_level_checkpoint(
                            sid,
                            lv,
                            next_after,
                            sp,
                            sw,
                            (patch_tp, patch_fp, patch_tn, patch_fn),
                        )
                        chunks_since_ckpt = 0
                        state(
                            "inference_checkpoint",
                            image_id=sid,
                            level=lv,
                            next_patch_index=next_after,
                            total_patch_coords=len(coords),
                        )

                support = sw > 0
                lp = np.zeros_like(sp)
                np.divide(sp, sw, out=lp, where=support)
                lp[~tissue_eval] = 0.0
                pc = np.array(
                    [patch_tp, patch_fp, patch_tn, patch_fn], dtype=np.int64
                )

                _save_completed_level(sid, lv, lp, support, pc)
                level_prob[lv] = lp
                level_support[lv] = support
                level_patch_counts[lv] = pc
                state("level_complete", image_id=sid, level=lv)

            sum_prob = np.zeros((H, W), np.float32)
            n_support = np.zeros((H, W), np.float32)
            for lv in (0, 1, 2):
                sum_prob += level_prob[lv] * level_support[lv]
                n_support += level_support[lv].astype(np.float32)
            fusion_support = n_support > 0
            fusion = np.zeros((H, W), np.float32)
            np.divide(sum_prob, n_support, out=fusion, where=fusion_support)
            valid = annotated_valid & fusion_support

            tmp = cache.with_suffix(".npz.tmp")
            while True:
                try:
                    with open(tmp, "wb") as f:
                        np.savez(
                            f,
                            gt=gt.astype(np.uint8),
                            annotated_valid=annotated_valid.astype(np.uint8),
                            valid=valid.astype(np.uint8),
                            fusion=fusion.astype(np.float16),
                            fusion_support=fusion_support.astype(np.uint8),
                            tissue_eval=tissue_eval.astype(np.uint8),
                            **{
                                f"prob_L{lv}": level_prob[lv].astype(np.float16)
                                for lv in (0, 1, 2)
                            },
                            **{
                                f"support_L{lv}": level_support[lv].astype(np.uint8)
                                for lv in (0, 1, 2)
                            },
                            **{
                                f"patch_counts_L{lv}": level_patch_counts[lv]
                                for lv in (0, 1, 2)
                            },
                        )
                        f.flush()
                        os.fsync(f.fileno())
                    os.replace(tmp, cache)
                    break
                except Exception as e:
                    print(
                        f"[WSI CACHE WRITE RETRY] {sid}: {type(e).__name__}: {e}",
                        flush=True,
                    )
                    if not drive_ready():
                        wait_for_drive()
                    else:
                        time.sleep(3)

            im.close()
            ma.close()
            with np.load(cache) as z:
                return {k: z[k] for k in z.files}

        except KeyboardInterrupt:
            if im is not None:
                try:
                    im.close()
                except Exception:
                    pass
            if ma is not None:
                try:
                    ma.close()
                except Exception:
                    pass
            print(
                f"[INFERENCE STOPPED] {sid}: latest checkpoint preserved.",
                flush=True,
            )
            raise
        except Exception as e:
            if im is not None:
                try:
                    im.close()
                except Exception:
                    pass
            if ma is not None:
                try:
                    ma.close()
                except Exception:
                    pass
            print(
                f"[INFERENCE RETRY] {sid}: {type(e).__name__}: {e}",
                flush=True,
            )
            traceback.print_exc()
            if not drive_ready():
                wait_for_drive()
            else:
                time.sleep(5)
            print(
                f"[INFERENCE RESUME] {sid}: reopening files and continuing from latest checkpoint.",
                flush=True,
            )


def save_slide_visual(row, z, rr, thr):
    sid = row["image_id"]
    outdir = RESULT_ROOT / "visuals_all_wsi" / sid
    outdir.mkdir(parents=True, exist_ok=True)
    combined = outdir / "combined_original_gt_model_error.png"
    if combined.exists():
        return

    im = OmePyramid(row["image_path"])
    rgb = im.thumbnail(4, 4096)
    im.close()

    gt=z["gt"].astype(bool); valid=z["valid"].astype(bool)
    tissue_eval = z["tissue_eval"].astype(bool) if "tissue_eval" in z else np.ones_like(z["fusion"], dtype=bool)
    pred=(z["fusion"].astype(np.float32)>=thr) & tissue_eval

    # Resize all evaluation maps to the thumbnail size with nearest neighbor.
    h,w=rgb.shape[:2]
    gt_r=cv2.resize((gt*255).astype(np.uint8),(w,h),interpolation=cv2.INTER_NEAREST)
    pred_r=cv2.resize((pred*255).astype(np.uint8),(w,h),interpolation=cv2.INTER_NEAREST)
    valid_r=cv2.resize((valid*255).astype(np.uint8),(w,h),interpolation=cv2.INTER_NEAREST)>0
    gtb=gt_r>0; prb=pred_r>0

    err=np.zeros_like(rgb)+255
    err[(~gtb)&(~prb)&valid_r]=[230,230,230]
    err[gtb&prb&valid_r]=[60,170,90]
    err[gtb&(~prb)&valid_r]=[240,190,40]
    err[(~gtb)&prb&valid_r]=[210,60,60]

    fig,axs=plt.subplots(1,4,figsize=(22,6))
    panels=[("Original H&E",rgb),("PRECISE GT Tumor",gt_r),("PANDA Model",pred_r),("Error Map",err)]
    for ax,(title,p) in zip(axs,panels):
        ax.imshow(p,cmap="gray" if p.ndim==2 else None); ax.set_title(title); ax.axis("off")
    fig.suptitle(
        f"{sid} | Dice={rr['fusion_dice']:.4f}  IoU={rr['fusion_iou']:.4f}  "
        f"Precision={rr['fusion_precision']:.4f}  Recall={rr['fusion_recall']:.4f}  "
        f"Specificity={rr['fusion_specificity']:.4f}  Accuracy={rr['fusion_accuracy']:.4f}\n"
        f"TP={rr['fusion_tp']:,} FP={rr['fusion_fp']:,} TN={rr['fusion_tn']:,} FN={rr['fusion_fn']:,} | threshold={thr:.2f}"
    )
    fig.tight_layout()
    fig.savefig(combined,dpi=250,bbox_inches="tight")
    plt.close(fig)
    atomic_json(outdir/"metrics.json", rr)

def bootstrap_ci(df, seed=42):
    rng=np.random.default_rng(seed)
    keys=["dice","iou","precision","recall","specificity","accuracy"]
    vals={k:[] for k in keys}
    n=len(df)
    if n==0:
        return {}
    for _ in range(BOOTSTRAP_ITERS):
        b=df.iloc[rng.integers(0,n,size=n)]
        c=metrics(int(b.fusion_tp.sum()),int(b.fusion_fp.sum()),
                  int(b.fusion_tn.sum()),int(b.fusion_fn.sum()))
        for k in keys: vals[k].append(c[k])
    return {k:{"low":float(np.percentile(v,2.5)),"high":float(np.percentile(v,97.5))}
            for k,v in vals.items()}

def main():
    for d in [EXT_ROOT,DATA_ROOT,RESULT_ROOT,CACHE_ROOT,STATE_ROOT]:
        d.mkdir(parents=True,exist_ok=True)

    print("="*78)
    print("PRECISE EXTERNAL FULL-WSI TEST — H&E ONLY")
    print("RESUME COMPAT: legacy v1 completed slides + v2 chunk/level resume")
    print(f"Work: {EXT_ROOT}")
    print("Only H&E WSIs + H&E masks are downloaded; IHC is skipped.")
    print("="*78)

    # STREAM MODE: only read the remote index here.
    # Each slide is downloaded/reused immediately before its own test.
    try:
        manifest = precise_remote_index()
    except Exception:
        state("remote_index_error", traceback=traceback.format_exc())
        raise

    state("before_model_load")
    cfg=FinalConfig()
    model,device,checkpoint=core.load_best_model(cfg,None)
    if device.type=="cuda":
        torch.backends.cudnn.benchmark=True
        try:
            torch.backends.cuda.matmul.allow_tf32=True
            torch.backends.cudnn.allow_tf32=True
        except Exception:
            pass
        model=model.to(memory_format=torch.channels_last)
    print(f"[MODEL] {checkpoint}")
    print(f"[DEVICE] {device} | {torch.cuda.get_device_name(device) if device.type=='cuda' else 'CPU'}")

    if not cfg.stain_target_path.exists():
        raise FileNotFoundError(f"TRAIN-derived Vahadane target missing: {cfg.stain_target_path}")
    target=np.array(Image.open(cfg.stain_target_path).convert("RGB"),dtype=np.uint8,copy=True)
    stain=core.VahadaneProcessor(target)
    thr=load_threshold()

    rows=[]
    progress_csv=RESULT_ROOT/"per_slide_metrics_progress.csv"
    done={}
    if progress_csv.exists():
        old=pd.read_csv(progress_csv)
        done={str(r.image_id):r.to_dict() for _,r in old.iterrows()}
        rows=list(done.values())

    state("before_external_test", total_slides=len(manifest), already_done=len(done))
    total_slides = len(manifest)

    for i, row_meta in manifest.iterrows():
        sid = str(row_meta.image_id)
        cache_path = find_complete_cache(sid)

        # Already fully tested -> skip everything, including download.
        # IMPORTANT: legacy *_ds16_precise_v1.npz is accepted too.
        if sid in done and cache_path is not None:
            print(
                f"[TEST REUSE {i+1}/{total_slides}] {sid} | already complete | {cache_path.name}",
                flush=True,
            )
            continue

        while True:
            try:
                # A) Download/reuse only this slide + mask.
                row = ensure_slide_pair_downloaded(row_meta, i+1, total_slides)

                # B) Immediately test it.
                print(f"[TEST {i+1}/{total_slides}] {sid} | starting now", flush=True)
                state("before_slide_test", image_id=sid, index=i+1, total=total_slides)

                z = reconstruct_slide(row, model, device, stain, cfg, thr)
                rr = {"image_id": sid}
                gt = z["gt"].astype(bool)

                tissue_eval = z["tissue_eval"].astype(bool) if "tissue_eval" in z else np.ones_like(z["fusion"], dtype=bool)

                for lv in (0,1,2):
                    valid = z["annotated_valid"].astype(bool) & z[f"support_L{lv}"].astype(bool)
                    pred = (z[f"prob_L{lv}"].astype(np.float32) >= thr) & tissue_eval
                    mm = metrics(*counts(pred, gt, valid))
                    for k, v in mm.items():
                        rr[f"L{lv}_{k}"] = v

                valid = z["valid"].astype(bool)
                pred = (z["fusion"].astype(np.float32) >= thr) & tissue_eval
                mm = metrics(*counts(pred, gt, valid))
                for k, v in mm.items():
                    rr[f"fusion_{k}"] = v

                rr["valid_pixels"] = int(valid.sum())
                rr["gt_tumor_pixels"] = int((gt & valid).sum())
                rr["pred_tumor_pixels"] = int((pred & valid).sum())

                # C) Commit metrics + visual before moving on.
                rows = [r for r in rows if str(r.get("image_id")) != sid] + [rr]
                atomic_csv(pd.DataFrame(rows), progress_csv)
                save_slide_visual(row, z, rr, thr)

                state(
                    "slide_complete",
                    completed=len(rows),
                    total=total_slides,
                    last_image_id=sid,
                    fusion_dice=float(rr["fusion_dice"]),
                )
                print(
                    f"[SLIDE COMPLETE {i+1}/{total_slides}] {sid} | "
                    f"Dice={rr['fusion_dice']:.4f} | moving to next",
                    flush=True,
                )

                del z
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                break

            except KeyboardInterrupt:
                raise
            except Exception as e:
                print(f"[SLIDE RETRY] {sid}: {type(e).__name__}: {e}", flush=True)
                if not drive_ready():
                    wait_for_drive()
                    continue
                traceback.print_exc()
                time.sleep(10)

    ps=pd.DataFrame(rows).sort_values("image_id").reset_index(drop=True)
    tables=RESULT_ROOT/"tables"; figs=RESULT_ROOT/"figures"
    tables.mkdir(parents=True,exist_ok=True); figs.mkdir(parents=True,exist_ok=True)
    atomic_csv(ps, tables/"per_slide_wsi_metrics.csv")

    # Global WSI metrics per level + fusion.
    global_rows=[]
    for key in ["L0","L1","L2","fusion"]:
        mm=metrics(int(ps[f"{key}_tp"].sum()),int(ps[f"{key}_fp"].sum()),
                   int(ps[f"{key}_tn"].sum()),int(ps[f"{key}_fn"].sum()))
        global_rows.append({"subset":key,**mm})
    atomic_csv(pd.DataFrame(global_rows), tables/"external_wsi_metrics_per_level_and_fusion.csv")

    fusion_global=[r for r in global_rows if r["subset"]=="fusion"][0]
    ci=bootstrap_ci(ps)
    atomic_json(tables/"external_global_metrics.json",{
        **fusion_global,
        "n_wsi":int(len(ps)),
        "threshold":float(thr),
        "bootstrap_iterations":BOOTSTRAP_ITERS,
        "bootstrap_95CI":ci,
        "dataset":"PRECISE H&E external center",
        "label_mapping":{
            "tumor_positive":[1],
            "negative_benign_stroma":[2,7],
            "ignored_non_equivalent_or_unknown":[0,3,4,5,6]
        }
    })

    # Patch-level pooled metrics from counts accumulated during dense WSI inference.
    patch_rows=[]
    for lv in (0,1,2):
        a=np.zeros(4,dtype=np.int64)
        for _,r in manifest.iterrows():
            c=find_complete_cache(str(r.image_id))
            if c is None:
                continue
            with np.load(c) as z:
                a += z[f"patch_counts_L{lv}"].astype(np.int64)
        patch_rows.append({"subset":f"L{lv}",**metrics(*map(int,a))})
    pooled=np.sum([[r["tp"],r["fp"],r["tn"],r["fn"]] for r in patch_rows],axis=0)
    patch_rows.append({"subset":"all_levels_pooled",**metrics(*map(int,pooled))})
    atomic_csv(pd.DataFrame(patch_rows), tables/"external_patch_metrics_per_level_and_pooled.csv")

    # Basic paper figures.
    s=ps.sort_values("fusion_dice").reset_index(drop=True)
    fig,ax=plt.subplots(figsize=(10,5))
    ax.scatter(np.arange(1,len(s)+1),s.fusion_dice*100,s=28)
    ax.axhline(s.fusion_dice.mean()*100,ls="--")
    ax.set_xlabel("External PRECISE WSI (sorted)")
    ax.set_ylabel("Fusion Dice (%)")
    ax.set_title("External-center PRECISE WSI Dice")
    fig.tight_layout(); fig.savefig(figs/"per_slide_external_dice.png",dpi=300,bbox_inches="tight"); plt.close(fig)

    gm=fusion_global
    mat=np.array([[gm["tn"],gm["fp"]],[gm["fn"],gm["tp"]]])
    fig,ax=plt.subplots(figsize=(5,4.5))
    ax.imshow(mat,cmap="Blues")
    for (y,x),v in np.ndenumerate(mat):
        ax.text(x,y,f"{int(v):,}",ha="center",va="center")
    ax.set_xticks([0,1],["Normal","Tumor"]); ax.set_yticks([0,1],["Normal","Tumor"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Ground truth")
    ax.set_title("PRECISE External WSI Confusion Matrix")
    fig.tight_layout(); fig.savefig(figs/"external_confusion_matrix.png",dpi=300,bbox_inches="tight"); plt.close(fig)

    state("complete",n_wsi=len(ps),global_metrics=fusion_global)
    print("\n[EXTERNAL PRECISE COMPLETE]")
    print(json.dumps(fusion_global,indent=2))
    print(f"Results: {RESULT_ROOT}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[STOPPED BY USER] Resume is preserved.")
        raise
    except Exception as e:
        try:
            state("fatal_error",error=f"{type(e).__name__}: {e}",traceback=traceback.format_exc())
        except Exception:
            pass
        print(f"\n[EXTERNAL PRECISE ERROR] {type(e).__name__}: {e}")
        traceback.print_exc()
        raise
