from __future__ import annotations

"""
Standalone external-validation pipeline for SICAPv2.

This file is intentionally separate from run_full_panda_paper.py. It does NOT
train or modify the PANDA model. It:
  1) downloads public SICAPv2 v2 from Mendeley Data with resumable HTTP Range;
  2) extracts with per-file resume;
  3) uses the official SICAPv2 Test partition when available;
  4) loads the already-trained PANDA best.pt and fixed PANDA validation threshold;
  5) applies the same fixed TRAIN-derived Vahadane target and coordinate-aware model;
  6) evaluates binary tumor (any SICAP Gleason-pattern mask pixel > 0) vs non-tumor
     on valid tissue;
  7) writes per-patch/image, source-WSI aggregate, global, bootstrap, boundary,
     confusion-matrix and qualitative outputs;
  8) creates one lossless high-resolution combined PNG LIVE for EVERY external test image.

Resume is idempotent at every phase. A .part file is used for download, extraction
skips already-complete members, inference skips per-image caches, and visual/stat
stages skip/rebuild from durable outputs.
"""

import argparse
import subprocess
import sys
import hashlib
import json
import math
import os
import re
import shutil
import sys
import threading
import time
import traceback
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

import torch

import panda_camelyon_style_pipeline as core
import panda_stream_paper as paper
from panda_final_pipeline import FinalConfig

try:
    import requests
except Exception as e:
    requests = None
    _REQUESTS_IMPORT_ERROR = e
else:
    _REQUESTS_IMPORT_ERROR = None


DATASET_ID = "9xxm58dvs3"
DATASET_VERSION = 2
DATASET_NAME = "SICAPv2"
DATASET_PAGE = "https://data.mendeley.com/datasets/9xxm58dvs3/2"
KAGGLE_DATASET_HANDLE = "shridharspol/sicapv2"
KAGGLE_DATASET_PAGE = "https://www.kaggle.com/datasets/shridharspol/sicapv2"
# Mendeley API download endpoints now require authenticated API access and can
# return HTTP 403 for anonymous scripts. Prefer public S3 cache archives first.
# The v2 S3 URL follows Mendeley's public cache naming convention; if unavailable,
# automatically fall back to the documented public v1 archive, which is still the
# SICAPv2 dataset and contains the image/mask/test-partition structure needed here.
SICAP_PUBLIC_URLS = [
    ("mendeley_s3_v2", "https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/9xxm58dvs3-2.zip"),
    ("mendeley_s3_v1", "https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/9xxm58dvs3-1.zip"),
]
ZIP_API_URL = SICAP_PUBLIC_URLS[0][1]
LICENSE = "CC BY 4.0"
DEFAULT_BATCH = 16
DEFAULT_PREP_WORKERS = 6
BOOTSTRAP_ITERS = 1000


def _now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _atomic_json(path: Path, obj: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _drive_ready(path: Path) -> bool:
    try:
        anchor = Path(path.anchor or str(path))
        if not anchor.exists():
            return False
        next(iter(anchor.iterdir()), None)
        return True
    except Exception:
        return False


def _wait_storage(path: Path, seconds: int = 10):
    print(f"[SSD WAIT] Storage unavailable for {path}. Retrying every {seconds}s...", flush=True)
    while not _drive_ready(path):
        time.sleep(seconds)
    print("[SSD RECONNECTED] Storage is readable again. Continuing automatically.", flush=True)


def _storage_like(exc: BaseException) -> bool:
    s = f"{type(exc).__name__}: {exc}".lower()
    needles = [
        "errno 22", "invalid argument", "device is not ready", "input/output error",
        "i/o error", "bad file descriptor", "winerror 3", "winerror 21",
        "winerror 53", "winerror 64", "cannot find the path", "no such file",
        "unexpected pos", "inline_container.cc", "file not found",
    ]
    return any(x in s for x in needles)


def _retry_io(work_root: Path, fn, *, label: str, sleep_s: int = 10):
    while True:
        try:
            return fn()
        except KeyboardInterrupt:
            raise
        except Exception as e:
            if _storage_like(e) or not _drive_ready(work_root):
                print(f"[{label}] storage interruption: {type(e).__name__}: {e}", flush=True)
                _wait_storage(work_root, sleep_s)
                continue
            raise


def _local_resume_dir() -> Path:
    base = os.environ.get("LOCALAPPDATA") or str(Path.home())
    p = Path(base) / "PANDA_SICAPV2_RESUME"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _state_paths(ext_root: Path) -> Tuple[Path, Path]:
    return ext_root / "resume_state.json", _local_resume_dir() / "resume_state.json"


def save_state(ext_root: Path, phase: str, **extra):
    obj = {"dataset": DATASET_NAME, "phase": phase, "updated_at": _now(), **extra}
    for p in _state_paths(ext_root):
        try:
            _atomic_json(p, obj)
        except Exception:
            pass


def _sha_key(name: str) -> str:
    return hashlib.sha1(name.encode("utf-8", errors="ignore")).hexdigest()[:20]


def download_resumable(url: str, out_zip: Path, work_root: Path):
    """
    Resumable SICAPv2 download with automatic public-source failover.

    Priority:
      1) user --dataset-url, if explicitly different from the default;
      2) Mendeley public S3 cache v2 candidate;
      3) documented Mendeley public S3 cache v1 archive.

    Anonymous Mendeley API 401/403 is treated as an access-policy failure, not a
    transient network error, so the code immediately tries the next public URL.
    """
    if requests is None:
        raise RuntimeError(f"requests is required for automatic download: {_REQUESTS_IMPORT_ERROR}")

    out_zip.parent.mkdir(parents=True, exist_ok=True)
    part = out_zip.with_suffix(out_zip.suffix + ".part")
    source_state = out_zip.with_suffix(out_zip.suffix + ".source.json")

    if out_zip.exists() and zipfile.is_zipfile(out_zip):
        print(f"[DOWNLOAD REUSE] {out_zip}", flush=True)
        return
    if out_zip.exists():
        out_zip.unlink(missing_ok=True)

    candidates = []
    if url and url != ZIP_API_URL:
        candidates.append(("user_url", url))
    for name, candidate_url in SICAP_PUBLIC_URLS:
        if candidate_url not in [u for _, u in candidates]:
            candidates.append((name, candidate_url))

    # Legacy API is kept only as a last-resort attempt because anonymous access
    # may now return 403. It will not loop forever on 401/403.
    legacy_api = f"https://api.data.mendeley.com/datasets/{DATASET_ID}/zip/file_downloaded?version={DATASET_VERSION}"
    if legacy_api not in [u for _, u in candidates]:
        candidates.append(("mendeley_api_legacy", legacy_api))

    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) PANDA-SICAPv2/2.0",
        "Accept": "*/*",
        "Referer": DATASET_PAGE,
    })

    last_error = None
    for source_name, source_url in candidates:
        # A partial file from another source cannot safely be appended to this one.
        # Keep resume only when the source matches the previous source marker.
        previous_source = None
        try:
            if source_state.exists():
                previous_source = json.loads(source_state.read_text(encoding="utf-8")).get("url")
        except Exception:
            previous_source = None
        if part.exists() and previous_source and previous_source != source_url:
            part.unlink(missing_ok=True)

        try:
            _atomic_json(source_state, {"name": source_name, "url": source_url, "updated_at": _now()})
        except Exception:
            pass

        print(f"[DOWNLOAD SOURCE] {source_name}", flush=True)
        while True:
            try:
                start_byte = part.stat().st_size if part.exists() else 0
                headers = {"Range": f"bytes={start_byte}-"} if start_byte > 0 else {}
                print(f"[DOWNLOAD] SICAPv2 | source={source_name} | resume_byte={start_byte:,}", flush=True)

                with session.get(
                    source_url,
                    headers=headers,
                    stream=True,
                    timeout=(30, 180),
                    allow_redirects=True,
                ) as r:
                    # Authentication/access-policy errors: switch source immediately.
                    if r.status_code in (401, 403):
                        raise PermissionError(
                            f"HTTP {r.status_code} from {source_name}; anonymous access blocked"
                        )
                    # Missing public-cache candidate: switch source immediately.
                    if r.status_code == 404:
                        raise FileNotFoundError(f"HTTP 404 from {source_name}")
                    if r.status_code == 416 and start_byte > 0:
                        os.replace(part, out_zip)
                        if zipfile.is_zipfile(out_zip):
                            print(f"[DOWNLOAD COMPLETE] {out_zip} | source={source_name}", flush=True)
                            return
                        out_zip.unlink(missing_ok=True)
                        continue

                    r.raise_for_status()

                    mode = "ab"
                    if start_byte > 0 and r.status_code != 206:
                        print("[DOWNLOAD] Server ignored Range; restarting this source safely.", flush=True)
                        part.unlink(missing_ok=True)
                        start_byte = 0
                        mode = "wb"
                    elif start_byte == 0:
                        mode = "wb"

                    total = None
                    if r.headers.get("Content-Range"):
                        m = re.search(r"/(\d+)$", r.headers["Content-Range"])
                        if m:
                            total = int(m.group(1))
                    elif r.headers.get("Content-Length"):
                        total = start_byte + int(r.headers["Content-Length"])

                    with open(part, mode) as f, tqdm(
                        total=total,
                        initial=start_byte,
                        unit="B", unit_scale=True, unit_divisor=1024,
                        desc=f"SICAPv2 download ({source_name})",
                    ) as bar:
                        since_flush = 0
                        for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
                            if not chunk:
                                continue
                            f.write(chunk)
                            bar.update(len(chunk))
                            since_flush += len(chunk)
                            if since_flush >= 64 * 1024 * 1024:
                                f.flush()
                                os.fsync(f.fileno())
                                since_flush = 0
                        f.flush()
                        os.fsync(f.fileno())

                os.replace(part, out_zip)
                if not zipfile.is_zipfile(out_zip):
                    raise RuntimeError(
                        f"Downloaded file from {source_name} is not a valid ZIP."
                    )

                try:
                    _atomic_json(source_state, {
                        "name": source_name,
                        "url": source_url,
                        "completed_at": _now(),
                        "zip": str(out_zip),
                        "size": out_zip.stat().st_size,
                    })
                except Exception:
                    pass

                print(f"[DOWNLOAD COMPLETE] {out_zip} | source={source_name}", flush=True)
                return

            except KeyboardInterrupt:
                raise
            except (PermissionError, FileNotFoundError) as e:
                last_error = e
                print(f"[DOWNLOAD SOURCE FAILED] {source_name}: {e}", flush=True)
                part.unlink(missing_ok=True)
                break
            except Exception as e:
                last_error = e
                print(f"[DOWNLOAD RETRY] {source_name}: {type(e).__name__}: {e}", flush=True)
                if not _drive_ready(work_root):
                    _wait_storage(work_root)
                    continue

                # Network/server transient error: retry current public source a few
                # times is useful, but don't trap the user forever. Track attempts
                # in-memory and then fail over.
                retry_key = f"_retry_{source_name}"
                n = getattr(download_resumable, retry_key, 0) + 1
                setattr(download_resumable, retry_key, n)
                if n >= 3:
                    print(f"[DOWNLOAD FAILOVER] {source_name} failed {n} times; trying next source.", flush=True)
                    part.unlink(missing_ok=True)
                    break
                time.sleep(10)

    raise RuntimeError(
        "All automatic SICAPv2 download sources failed. "
        f"Last error: {type(last_error).__name__}: {last_error}. "
        f"Official page: {DATASET_PAGE}"
    )



def _ensure_kagglehub():
    """
    Import kagglehub, installing it automatically if missing.
    Public Kaggle datasets can be downloaded without authentication unless the
    specific resource requires consent.
    """
    try:
        import kagglehub
        return kagglehub
    except Exception:
        print("[KAGGLE] kagglehub not installed; installing automatically...", flush=True)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "kagglehub"])
        import kagglehub
        return kagglehub


def download_kaggle_dataset_resumable(extract_root: Path, work_root: Path):
    """
    Download SICAPv2 from the public Kaggle mirror using KaggleHub's own cache.

    IMPORTANT:
    - Do NOT use output_dir=... because KaggleHub refuses a non-empty directory.
    - Put KAGGLEHUB_CACHE on D: so the large dataset is not cached on C:.
    - Re-running is cache/resume safe.
    - Once KaggleHub returns a resolved dataset folder, copy/link the discovered
      dataset contents into extract_root without deleting existing completed files.
    """
    marker = extract_root / ".kaggle_download_complete"

    if marker.exists():
        print(f"[KAGGLE REUSE] {extract_root}", flush=True)
        return True

    # Reuse already-complete extracted data if present.
    if extract_root.exists():
        try:
            _ = discover_dataset(extract_root)
            marker.write_text(f"complete {_now()}\n", encoding="utf-8")
            print(f"[KAGGLE REUSE] Existing SICAPv2 dataset discovered at {extract_root}", flush=True)
            return True
        except Exception:
            pass

    extract_root.mkdir(parents=True, exist_ok=True)

    # Keep KaggleHub cache on the external/project drive, not C:.
    kaggle_cache_root = work_root / "external_sicapv2" / "kagglehub_cache"
    kaggle_cache_root.mkdir(parents=True, exist_ok=True)
    os.environ["KAGGLEHUB_CACHE"] = str(kaggle_cache_root)
    print(f"[KAGGLE CACHE] {kaggle_cache_root}", flush=True)

    kagglehub = _ensure_kagglehub()

    def _copy_tree_resume(src_root: Path, dst_root: Path):
        """
        Resume-safe copy: existing files with identical byte size are reused.
        Copy through .part then os.replace for crash safety.
        """
        files = [p for p in src_root.rglob("*") if p.is_file()]
        bar = tqdm(files, desc="SICAPv2 copy to dataset", unit="file")
        for src in bar:
            rel = src.relative_to(src_root)
            dst = dst_root / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            try:
                if dst.exists() and dst.stat().st_size == src.stat().st_size:
                    continue
            except OSError as e:
                if _storage_like(e) or not _drive_ready(work_root):
                    _wait_storage(work_root)
                else:
                    raise

            tmp = dst.with_suffix(dst.suffix + ".part")
            while True:
                try:
                    shutil.copy2(src, tmp)
                    os.replace(tmp, dst)
                    break
                except Exception as e:
                    if _storage_like(e) or not _drive_ready(work_root):
                        _wait_storage(work_root)
                        continue
                    raise

    while True:
        try:
            print(f"[KAGGLE DOWNLOAD] {KAGGLE_DATASET_HANDLE}", flush=True)
            print(f"[KAGGLE PAGE] {KAGGLE_DATASET_PAGE}", flush=True)

            # No output_dir argument here. KaggleHub manages the cache/resume.
            resolved = Path(kagglehub.dataset_download(KAGGLE_DATASET_HANDLE))
            print(f"[KAGGLE CACHE READY] {resolved}", flush=True)

            # Sometimes the returned directory itself is already the dataset root;
            # otherwise discover_dataset can locate the nested SICAP structure.
            try:
                _ = discover_dataset(resolved)
                source_root = resolved
            except Exception:
                # If returned cache root contains one nested dataset directory,
                # discover the nearest useful child by scanning.
                source_root = resolved

            _copy_tree_resume(source_root, extract_root)

            # Verify before committing completion.
            _ = discover_dataset(extract_root)
            marker.write_text(f"complete {_now()}\n", encoding="utf-8")
            print(f"[KAGGLE DATASET READY] {extract_root}", flush=True)
            return True

        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f"[KAGGLE DOWNLOAD RETRY] {type(e).__name__}: {e}", flush=True)
            if _storage_like(e) or not _drive_ready(work_root):
                _wait_storage(work_root)
                continue

            retry_n = getattr(download_kaggle_dataset_resumable, "_retry_n", 0) + 1
            setattr(download_kaggle_dataset_resumable, "_retry_n", retry_n)
            if retry_n < 4:
                time.sleep(10)
                continue

            print("[KAGGLE DOWNLOAD FAILED] Falling back to Mendeley sources.", flush=True)
            return False


def extract_resumable(zip_path: Path, extract_root: Path, work_root: Path):
    marker = extract_root / ".extract_complete"
    if marker.exists():
        print(f"[EXTRACT REUSE] {extract_root}", flush=True)
        return
    extract_root.mkdir(parents=True, exist_ok=True)

    while True:
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                infos = zf.infolist()
                bar = tqdm(infos, desc="SICAPv2 extract", unit="file")
                for info in bar:
                    dest = extract_root / info.filename
                    if info.is_dir():
                        dest.mkdir(parents=True, exist_ok=True)
                        continue
                    # Exact-size file means this member already completed earlier.
                    if dest.exists() and dest.stat().st_size == info.file_size:
                        continue
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    tmp = dest.with_suffix(dest.suffix + ".part")
                    tmp.unlink(missing_ok=True)
                    try:
                        with zf.open(info, "r") as src, open(tmp, "wb") as dst:
                            shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
                            dst.flush(); os.fsync(dst.fileno())
                        os.replace(tmp, dest)
                    except Exception:
                        tmp.unlink(missing_ok=True)
                        raise
            marker.write_text(f"complete {_now()}\n", encoding="utf-8")
            print(f"[EXTRACT COMPLETE] {extract_root}", flush=True)
            return
        except KeyboardInterrupt:
            raise
        except Exception as e:
            if _storage_like(e) or not _drive_ready(work_root):
                print(f"[EXTRACT RETRY] {type(e).__name__}: {e}", flush=True)
                _wait_storage(work_root)
                continue
            raise


def _find_best_named_dir(root: Path, dirname: str) -> Path:
    cands = [p for p in root.rglob("*") if p.is_dir() and p.name.lower() == dirname.lower()]
    if not cands:
        raise FileNotFoundError(f"Could not find SICAPv2 '{dirname}' directory below {root}")
    def score(p: Path):
        try:
            return sum(1 for _ in p.glob("*.jpg")) + sum(1 for _ in p.glob("*.png"))
        except Exception:
            return 0
    return max(cands, key=score)


def discover_dataset(extract_root: Path) -> Dict[str, Path]:
    images = _find_best_named_dir(extract_root, "images")
    masks = _find_best_named_dir(extract_root, "masks")
    tests = [p for p in extract_root.rglob("*.xlsx") if p.name.lower() == "test.xlsx"]
    test_xlsx = tests[0] if tests else None
    return {"images": images, "masks": masks, "test_xlsx": test_xlsx}


def _infer_filename_column(df: pd.DataFrame, available_names: set[str]) -> str | None:
    preferred = ["image_name", "image", "filename", "file_name", "name"]
    lower = {str(c).lower(): c for c in df.columns}
    for p in preferred:
        if p in lower:
            return lower[p]
    best_col, best_hits = None, -1
    for c in df.columns:
        vals = df[c].astype(str).head(300)
        hits = sum((Path(v).name in available_names) or (Path(v).stem + ".jpg" in available_names) for v in vals)
        if hits > best_hits:
            best_col, best_hits = c, hits
    return best_col if best_hits > 0 else None


def _parse_group_and_xy(stem: str) -> Tuple[str, int | None, int | None]:
    m = re.match(r"^(.*?)(?:_\d+_\d+)?_xini_(-?\d+)_yini_(-?\d+)$", stem, flags=re.I)
    if m:
        return m.group(1), int(m.group(2)), int(m.group(3))
    m2 = re.search(r"_xini_(-?\d+)_yini_(-?\d+)", stem, flags=re.I)
    if m2:
        group = stem[:m2.start()]
        return group, int(m2.group(1)), int(m2.group(2))
    # Conservative fallback: group by leading specimen identifier.
    return stem.split("_Region_")[0] if "_Region_" in stem else stem, None, None


def build_index(ds: Dict[str, Path], out_csv: Path) -> pd.DataFrame:
    if out_csv.exists():
        try:
            df = pd.read_csv(out_csv)
            required = {"image_name", "image_path", "mask_path", "source_wsi"}
            if required.issubset(df.columns) and len(df):
                print(f"[INDEX REUSE] n={len(df)} | {out_csv}", flush=True)
                return df
        except Exception:
            pass

    image_files = sorted([*ds["images"].glob("*.jpg"), *ds["images"].glob("*.jpeg"), *ds["images"].glob("*.png")])
    image_map = {p.name: p for p in image_files}
    mask_files = sorted([*ds["masks"].glob("*.jpg"), *ds["masks"].glob("*.jpeg"), *ds["masks"].glob("*.png")])
    mask_map = {p.name: p for p in mask_files}
    mask_stem_map = {p.stem: p for p in mask_files}

    selected_names: List[str]
    split_source = "all_images_fallback"
    tx = ds.get("test_xlsx")
    if tx is not None and Path(tx).exists():
        test_df = pd.read_excel(tx)
        col = _infer_filename_column(test_df, set(image_map))
        if col is None:
            raise RuntimeError(f"Could not infer image-name column from official Test.xlsx columns={list(test_df.columns)}")
        selected_names = []
        for raw in test_df[col].astype(str):
            n = Path(raw).name
            if n in image_map:
                selected_names.append(n)
            elif (Path(n).stem + ".jpg") in image_map:
                selected_names.append(Path(n).stem + ".jpg")
            elif Path(n).stem in {p.stem for p in image_files}:
                # extension-agnostic resolution
                found = next((p.name for p in image_files if p.stem == Path(n).stem), None)
                if found:
                    selected_names.append(found)
        selected_names = list(dict.fromkeys(selected_names))
        split_source = str(tx)
    else:
        selected_names = [p.name for p in image_files]

    rows = []
    missing_masks = []
    for name in selected_names:
        ip = image_map.get(name)
        if ip is None:
            continue
        mp = mask_map.get(name) or mask_stem_map.get(ip.stem)
        if mp is None:
            missing_masks.append(name)
            continue
        group, x, y = _parse_group_and_xy(ip.stem)
        rows.append({
            "image_name": name,
            "image_id": ip.stem,
            "image_path": str(ip),
            "mask_path": str(mp),
            "source_wsi": group,
            "xini": x,
            "yini": y,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("SICAPv2 test index is empty after image/mask pairing.")

    # Coordinate embedding uses relative center within each original SICAP source WSI.
    df["coord_x"] = 0.5
    df["coord_y"] = 0.5
    for group, idx in df.groupby("source_wsi").groups.items():
        sub = df.loc[idx]
        if sub.xini.notna().all() and sub.yini.notna().all():
            xmin, xmax = float(sub.xini.min()), float(sub.xini.max())
            ymin, ymax = float(sub.yini.min()), float(sub.yini.max())
            width = max(512.0, xmax - xmin + 512.0)
            height = max(512.0, ymax - ymin + 512.0)
            df.loc[idx, "coord_x"] = ((sub.xini.astype(float) - xmin + 256.0) / width).clip(0, 1)
            df.loc[idx, "coord_y"] = ((sub.yini.astype(float) - ymin + 256.0) / height).clip(0, 1)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    meta = {
        "dataset": DATASET_NAME,
        "version": DATASET_VERSION,
        "dataset_page": DATASET_PAGE,
        "license": LICENSE,
        "split_source": split_source,
        "paired_test_images": int(len(df)),
        "source_wsi_groups": int(df.source_wsi.nunique()),
        "missing_masks_count": int(len(missing_masks)),
        "missing_masks_examples": missing_masks[:50],
        "binary_mapping": "SICAP mask pixel > 0 => tumor; 0 => non-tumor, evaluated on H&E-derived valid tissue",
    }
    _atomic_json(out_csv.with_name("external_dataset_manifest.json"), meta)
    print(f"[INDEX COMPLETE] test_images={len(df)} | source_wsi={df.source_wsi.nunique()}", flush=True)
    return df


def load_threshold(work_root: Path) -> Tuple[float, str]:
    p = work_root / "paper_results" / "tables" / "best_threshold.json"
    if p.exists():
        obj = json.load(open(p, "r", encoding="utf-8"))
        if "threshold" in obj:
            return float(obj["threshold"]), str(p)
    return 0.5, "fallback_0.5_best_threshold_json_missing"


def load_model_and_norm(cfg: FinalConfig):
    model, device, checkpoint = core.load_best_model(cfg, None)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass
        model = model.to(memory_format=torch.channels_last)
    if not Path(cfg.stain_target_path).exists():
        raise FileNotFoundError(f"TRAIN-derived stain target missing: {cfg.stain_target_path}")
    target = np.array(Image.open(cfg.stain_target_path).convert("RGB"), dtype=np.uint8, copy=True)
    return model, device, checkpoint, target


_thread_local = threading.local()


def _thread_norm(target: np.ndarray):
    n = getattr(_thread_local, "vah", None)
    if n is None:
        _thread_local.vah = core.VahadaneProcessor(target)
        n = _thread_local.vah
    return n


def _load_mask_binary(path: str, shape_hw: Tuple[int, int]) -> np.ndarray:
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(path)
    h, w = shape_hw
    if m.shape[:2] != (h, w):
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
    return m > 0


def prepare_sample(row: Dict[str, Any], cfg: FinalConfig, target: np.ndarray):
    rgb = np.asarray(Image.open(row["image_path"]).convert("RGB"), dtype=np.uint8)
    # SICAPv2 is 512x512. Preserve original for metrics; resize model input only if needed.
    h, w = rgb.shape[:2]
    gt = _load_mask_binary(row["mask_path"], (h, w))
    tissue, _ = core.generate_tissue_mask(rgb, cfg)
    valid = tissue.astype(bool)
    # Defensive fallback if morphology returns no tissue on a real patch.
    if valid.mean() < 0.01:
        valid = (rgb.mean(axis=2) < 245)
    if valid.mean() < 0.01:
        valid = np.ones((h, w), dtype=bool)

    model_rgb = rgb
    if (h, w) != (cfg.patch_size, cfg.patch_size):
        model_rgb = cv2.resize(rgb, (cfg.patch_size, cfg.patch_size), interpolation=cv2.INTER_AREA)
    norm = _thread_norm(target).transform(model_rgb)
    x = np.ascontiguousarray(norm.transpose(2, 0, 1), dtype=np.float32) / 255.0
    coord = np.array([float(row.get("coord_x", 0.5)), float(row.get("coord_y", 0.5))], dtype=np.float32)
    return x, coord, rgb, gt, valid


def _cache_path(results_root: Path, image_name: str) -> Path:
    return results_root / "cache" / f"{_sha_key(image_name)}.json"


def _pred_path(results_root: Path, image_name: str) -> Path:
    return results_root / "pred_masks" / f"{_sha_key(image_name)}.png"


def _error_path(results_root: Path, image_name: str) -> Path:
    return results_root / "error_maps" / f"{_sha_key(image_name)}.png"


def _counts(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray):
    return paper.counts(pred, gt, valid)


def _metrics_from_counts(c):
    return paper.metrics(*c)


def _error_rgb(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> np.ndarray:
    h, w = gt.shape
    err = np.zeros((h, w, 3), dtype=np.uint8) + 255
    err[(~gt) & (~pred) & valid] = [230, 230, 230]  # TN
    err[gt & pred & valid] = [60, 170, 90]           # TP
    err[gt & (~pred) & valid] = [240, 190, 40]       # FN
    err[(~gt) & pred & valid] = [210, 60, 60]        # FP
    return err


def run_inference(df: pd.DataFrame, cfg: FinalConfig, results_root: Path, work_root: Path):
    cache_dir = results_root / "cache"; cache_dir.mkdir(parents=True, exist_ok=True)
    (results_root / "pred_masks").mkdir(parents=True, exist_ok=True)
    (results_root / "error_maps").mkdir(parents=True, exist_ok=True)

    model, device, checkpoint, target = load_model_and_norm(cfg)
    threshold, threshold_source = load_threshold(work_root)
    batch_size = max(1, int(os.environ.get("PANDA_EXTERNAL_BATCH", str(DEFAULT_BATCH))))
    prep_workers = max(1, int(os.environ.get("PANDA_EXTERNAL_PREP_WORKERS", str(DEFAULT_PREP_WORKERS))))
    if device.type != "cuda":
        batch_size = min(batch_size, 4)

    print(
        f"[EXTERNAL CUDA] device={device} | batch={batch_size} | prep_workers={prep_workers} | "
        f"AMP={bool(cfg.amp and device.type=='cuda')} | threshold={threshold:.4f}", flush=True
    )
    print(f"[EXTERNAL MODEL] checkpoint={checkpoint}", flush=True)
    print(f"[EXTERNAL THRESHOLD] source={threshold_source}", flush=True)

    missing = []
    reused_visuals = 0
    for _, r in df.iterrows():
        cp = _cache_path(results_root, str(r.image_name))
        pp = _pred_path(results_root, str(r.image_name))
        if cp.exists() and pp.exists():
            while True:
                try:
                    rec_done = json.load(open(cp, "r", encoding="utf-8"))
                    build_one_combined_visual(rec_done, results_root)
                    reused_visuals += 1
                    break
                except Exception as e:
                    if _storage_like(e) or not _drive_ready(work_root):
                        _wait_storage(work_root)
                        continue
                    raise
            continue
        missing.append(r.to_dict())

    print(
        f"[EXTERNAL RESUME] completed={len(df)-len(missing)}/{len(df)} | "
        f"remaining={len(missing)} | visuals_ready={reused_visuals}",
        flush=True
    )
    save_state(results_root.parent, "inference", completed=len(df)-len(missing), total=len(df))

    pbar = tqdm(total=len(missing), desc="SICAPv2 external TEST", unit="image")
    i = 0
    with ThreadPoolExecutor(max_workers=prep_workers) as ex:
        while i < len(missing):
            rows = missing[i:i+batch_size]
            try:
                prepared = list(ex.map(lambda rr: prepare_sample(rr, cfg, target), rows))
                xb = torch.from_numpy(np.stack([p[0] for p in prepared], axis=0))
                cb = torch.from_numpy(np.stack([p[1] for p in prepared], axis=0))
                if device.type == "cuda":
                    xb = xb.pin_memory().to(device, non_blocking=True).to(memory_format=torch.channels_last)
                    cb = cb.pin_memory().to(device, non_blocking=True)
                else:
                    xb = xb.to(device); cb = cb.to(device)
                with torch.inference_mode(), torch.amp.autocast("cuda", enabled=cfg.amp and device.type == "cuda"):
                    logits = model(xb, cb)
                    probs = torch.sigmoid(logits).float().cpu().numpy()[:, 0]

                for rr, prep, prob in zip(rows, prepared, probs):
                    _, _, rgb, gt, valid = prep
                    if prob.shape != gt.shape:
                        prob = cv2.resize(prob.astype(np.float32), (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)
                    pred = prob >= threshold
                    c = _counts(pred, gt, valid)
                    mm = _metrics_from_counts(c)
                    bm = paper.boundary_metrics_pixels(pred, gt, valid, eval_downsample=1.0)
                    rec = {
                        "image_name": rr["image_name"],
                        "image_id": rr["image_id"],
                        "source_wsi": rr["source_wsi"],
                        "image_path": rr["image_path"],
                        "mask_path": rr["mask_path"],
                        "coord_x": float(rr.get("coord_x", 0.5)),
                        "coord_y": float(rr.get("coord_y", 0.5)),
                        "threshold": float(threshold),
                        "checkpoint": str(checkpoint),
                        "valid_pixels": int(valid.sum()),
                        "gt_tumor_pixels": int((gt & valid).sum()),
                        "pred_tumor_pixels": int((pred & valid).sum()),
                        **mm,
                        **bm,
                    }
                    pp = _pred_path(results_root, rr["image_name"])
                    ep = _error_path(results_root, rr["image_name"])
                    Image.fromarray((pred.astype(np.uint8) * 255)).save(pp, format="PNG")
                    Image.fromarray(_error_rgb(pred, gt, valid)).save(ep, format="PNG")
                    _atomic_json(_cache_path(results_root, rr["image_name"]), rec)

                    # Create the pathology-review composite immediately for this
                    # completed image rather than waiting for the whole test.
                    while True:
                        try:
                            vis_path = build_one_combined_visual(rec, results_root)
                            print(
                                f"[EXTERNAL VISUAL SAVED] {rr['image_name']} -> {vis_path}",
                                flush=True
                            )
                            break
                        except Exception as e:
                            if _storage_like(e) or not _drive_ready(work_root):
                                _wait_storage(work_root)
                                continue
                            raise

                    pbar.update(1)
                i += len(rows)
                save_state(results_root.parent, "inference", completed=len(df)-len(missing)+i, total=len(df))
                del xb, cb, logits, probs
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            except KeyboardInterrupt:
                raise
            except RuntimeError as e:
                if "out of memory" in str(e).lower() and batch_size > 1:
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    batch_size = max(1, batch_size // 2)
                    print(f"[CUDA OOM FALLBACK] new_batch={batch_size}", flush=True)
                    continue
                if _storage_like(e) or not _drive_ready(work_root):
                    _wait_storage(work_root)
                    continue
                raise
            except Exception as e:
                if _storage_like(e) or not _drive_ready(work_root):
                    _wait_storage(work_root)
                    continue
                # Worker failures caused by a temporary disk outage can surface as generic exceptions.
                print(f"[EXTERNAL BATCH RETRY] {type(e).__name__}: {e}", flush=True)
                time.sleep(3)
                if not _drive_ready(work_root):
                    _wait_storage(work_root)
                else:
                    # Fail on genuine scientific/data errors rather than silently skipping.
                    raise
    pbar.close()
    save_state(results_root.parent, "inference_complete", total=len(df))


def collect_cache_rows(df: pd.DataFrame, results_root: Path) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        p = _cache_path(results_root, str(r.image_name))
        if not p.exists():
            raise RuntimeError(f"Inference cache missing for {r.image_name}: {p}")
        rows.append(json.load(open(p, "r", encoding="utf-8")))
    return pd.DataFrame(rows)


def _bootstrap_group_global(group_df: pd.DataFrame, seed: int = 42, n: int = BOOTSTRAP_ITERS):
    rng = np.random.default_rng(seed)
    arr = group_df[["tp", "fp", "tn", "fn"]].to_numpy(np.int64)
    vals = {m: [] for m in ["dice", "iou", "precision", "recall", "specificity", "accuracy"]}
    if len(arr) == 0:
        return {m: [float("nan"), float("nan")] for m in vals}
    for _ in range(n):
        idx = rng.integers(0, len(arr), size=len(arr))
        c = arr[idx].sum(axis=0)
        mm = _metrics_from_counts(tuple(int(x) for x in c))
        for m in vals:
            vals[m].append(mm[m])
    return {m: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for m, v in vals.items()}


def build_statistics(df: pd.DataFrame, results_root: Path, cfg: FinalConfig):
    tables = results_root / "tables"; figures = results_root / "figures"
    tables.mkdir(parents=True, exist_ok=True); figures.mkdir(parents=True, exist_ok=True)
    ps = collect_cache_rows(df, results_root)
    ps.to_csv(tables / "per_external_image_metrics.csv", index=False)

    # Global patch/pixel pooled result.
    c = tuple(int(ps[k].sum()) for k in ["tp", "fp", "tn", "fn"])
    global_mm = _metrics_from_counts(c)
    global_obj = {
        "dataset": DATASET_NAME,
        "version": DATASET_VERSION,
        "n_test_images": int(len(ps)),
        "n_source_wsi": int(ps.source_wsi.nunique()),
        "metric_scope": "SICAPv2 official-test 512x512 patches; pixels pooled over valid tissue",
        **global_mm,
        "mean_image_dice": float(ps.dice.mean()),
        "median_image_dice": float(ps.dice.median()),
        "sd_image_dice": float(ps.dice.std(ddof=1)) if len(ps) > 1 else 0.0,
        "min_image_dice": float(ps.dice.min()),
        "max_image_dice": float(ps.dice.max()),
        "mean_hd95_px": float(ps.hd95_eval_px.dropna().mean()) if "hd95_eval_px" in ps else float("nan"),
        "mean_assd_px": float(ps.assd_eval_px.dropna().mean()) if "assd_eval_px" in ps else float("nan"),
        "mean_surface_dice": float(ps.surface_dice.dropna().mean()) if "surface_dice" in ps else float("nan"),
    }
    _atomic_json(tables / "global_external_metrics.json", global_obj)

    # Aggregate counts by original SICAP source WSI. This avoids pretending that the public
    # 512x512 tiles are native pyramid WSIs while still giving a true patient/source-image level.
    agg = ps.groupby("source_wsi", as_index=False)[["tp", "fp", "tn", "fn", "valid_pixels", "gt_tumor_pixels", "pred_tumor_pixels"]].sum()
    metric_rows = []
    for _, r in agg.iterrows():
        mm = _metrics_from_counts((int(r.tp), int(r.fp), int(r.tn), int(r.fn)))
        metric_rows.append({**r.to_dict(), **mm})
    source = pd.DataFrame(metric_rows)
    source.to_csv(tables / "per_source_wsi_aggregated_metrics.csv", index=False)

    ci = _bootstrap_group_global(source, seed=cfg.seed, n=BOOTSTRAP_ITERS)
    ci_rows = []
    for m, bounds in ci.items():
        ci_rows.append({"metric": m, "value": global_mm[m], "ci_low": bounds[0], "ci_high": bounds[1], "bootstrap_iterations": BOOTSTRAP_ITERS, "resampling_unit": "source_wsi"})
    pd.DataFrame(ci_rows).to_csv(tables / "global_external_metrics_95CI_source_wsi_bootstrap.csv", index=False)

    # Confusion matrix.
    cm = np.array([[global_mm["tn"], global_mm["fp"]], [global_mm["fn"], global_mm["tp"]]], dtype=np.int64)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(cm)
    ax.set_xticks([0, 1], ["Pred non-tumor", "Pred tumor"])
    ax.set_yticks([0, 1], ["GT non-tumor", "GT tumor"])
    for (yy, xx), val in np.ndenumerate(cm):
        ax.text(xx, yy, f"{val:,}", ha="center", va="center")
    ax.set_title("SICAPv2 External Validation — Pixel Confusion Matrix")
    fig.tight_layout(); fig.savefig(figures / "external_confusion_matrix.png", dpi=300, bbox_inches="tight"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(ps.dice.to_numpy(float), bins=30)
    ax.set_xlabel("Dice"); ax.set_ylabel("External test images"); ax.set_title("SICAPv2 Per-image Dice Distribution")
    fig.tight_layout(); fig.savefig(figures / "per_image_dice_distribution.png", dpi=300, bbox_inches="tight"); plt.close(fig)

    # Sorted per-source WSI Dice.
    ss = source.sort_values("dice").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(np.arange(1, len(ss)+1), ss.dice.to_numpy(float), s=20)
    ax.axhline(float(ss.dice.mean()), ls="--")
    ax.set_xlabel("SICAPv2 source WSI (sorted)"); ax.set_ylabel("Dice"); ax.set_title("External Source-WSI Aggregated Dice")
    fig.tight_layout(); fig.savefig(figures / "source_wsi_dice.png", dpi=300, bbox_inches="tight"); plt.close(fig)

    _atomic_json(tables / "external_test_summary.json", {
        **global_obj,
        "bootstrap_95CI": ci,
        "threshold_source": load_threshold(Path(cfg.work))[1],
        "checkpoint": json.load(open(Path(cfg.work)/"latest_run.json", "r", encoding="utf-8")) if (Path(cfg.work)/"latest_run.json").exists() else None,
        "note": "No threshold/model selection was performed on SICAPv2; PANDA-trained best checkpoint and PANDA-validation threshold were frozen before external evaluation.",
    })
    save_state(results_root.parent, "statistics_complete", images=len(ps), source_wsi=len(source))
    return ps, source


def _font(size=24):
    candidates = [
        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\calibri.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for p in candidates:
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, size=size)
            except Exception:
                pass
    return ImageFont.load_default()


def _safe_component(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(s))[:160]


def build_one_combined_visual(rec: Dict[str, Any], results_root: Path):
    image_name = rec["image_name"]
    source_wsi = _safe_component(rec["source_wsi"])
    stem = _safe_component(Path(image_name).stem)
    outdir = results_root / "visuals_all_images" / source_wsi / stem
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / "combined_original_gt_model_error.png"
    if out.exists():
        return out

    rgb = Image.open(rec["image_path"]).convert("RGB")
    w, h = rgb.size
    gt = _load_mask_binary(rec["mask_path"], (h, w))
    pred_img = Image.open(_pred_path(results_root, image_name)).convert("L")
    if pred_img.size != (w, h):
        pred_img = pred_img.resize((w, h), Image.Resampling.NEAREST)
    pred = np.asarray(pred_img) > 0

    # Recreate valid tissue identically for the displayed error map.
    rgb_np = np.asarray(rgb, dtype=np.uint8)
    cfg = FinalConfig()
    tissue, _ = core.generate_tissue_mask(rgb_np, cfg)
    valid = tissue.astype(bool)
    if valid.mean() < 0.01:
        valid = (rgb_np.mean(axis=2) < 245)
    if valid.mean() < 0.01:
        valid = np.ones((h, w), dtype=bool)
    err = Image.fromarray(_error_rgb(pred, gt, valid))

    gt_img = Image.fromarray((gt.astype(np.uint8) * 255), mode="L").convert("RGB")
    model_img = Image.fromarray((pred.astype(np.uint8) * 255), mode="L").convert("RGB")

    # Pixel-for-pixel panel sizes: original H&E is NEVER downsampled for the composite.
    header_h = 130
    label_h = 40
    canvas = Image.new("RGB", (w * 4, header_h + label_h + h), "white")
    draw = ImageDraw.Draw(canvas)
    f1 = _font(25); f2 = _font(20)
    line1 = f"SICAPv2 external test | {image_name} | source WSI: {rec['source_wsi']}"
    line2 = (
        f"Dice {rec['dice']:.4f} | IoU {rec['iou']:.4f} | Precision {rec['precision']:.4f} | "
        f"Recall {rec['recall']:.4f} | Specificity {rec['specificity']:.4f} | Accuracy {rec['accuracy']:.4f}"
    )
    line3 = f"TP {int(rec['tp']):,} | FP {int(rec['fp']):,} | TN {int(rec['tn']):,} | FN {int(rec['fn']):,} | threshold {float(rec['threshold']):.3f}"
    draw.text((20, 10), line1, fill="black", font=f1)
    draw.text((20, 50), line2, fill="black", font=f2)
    draw.text((20, 85), line3, fill="black", font=f2)

    labels = ["Original H&E", "Original mask (GT)", "Model mask", "Error map"]
    panels = [rgb, gt_img, model_img, err]
    y0 = header_h + label_h
    for j, (lab, panel) in enumerate(zip(labels, panels)):
        draw.text((j*w + 12, header_h + 7), lab, fill="black", font=f2)
        canvas.paste(panel, (j*w, y0))

    canvas.save(out, format="PNG", optimize=False)
    # Small metadata sidecar for pathology review / reproducibility.
    _atomic_json(outdir / "metrics.json", rec)
    return out


def build_all_visuals(ps: pd.DataFrame, results_root: Path, work_root: Path):
    total = len(ps)
    done = 0
    bar = tqdm(total=total, desc="External combined visuals", unit="image")
    for rec in ps.to_dict("records"):
        while True:
            try:
                build_one_combined_visual(rec, results_root)
                break
            except Exception as e:
                if _storage_like(e) or not _drive_ready(work_root):
                    _wait_storage(work_root)
                    continue
                raise
        done += 1; bar.update(1)
        if done % 25 == 0:
            save_state(results_root.parent, "visuals", completed=done, total=total)
    bar.close()
    (results_root / "visuals_all_images" / ".complete").write_text(f"complete {_now()}\n", encoding="utf-8")
    save_state(results_root.parent, "visuals_complete", completed=total, total=total)


def write_final_readme(results_root: Path, index_df: pd.DataFrame):
    text = f"""SICAPv2 EXTERNAL VALIDATION — COMPLETE\n\nDataset: {DATASET_NAME} v{DATASET_VERSION}\nPublic source: {DATASET_PAGE}\nLicense: {LICENSE}\nOfficial external test images evaluated: {len(index_df)}\nSource-WSI groups: {index_df.source_wsi.nunique()}\n\nIMPORTANT SCIENTIFIC NOTE\nSICAPv2 is distributed as 512x512 10x histology patches with per-pixel Gleason-pattern masks.\nThe PANDA model is applied without retraining. The frozen PANDA best checkpoint, frozen PANDA\nvalidation threshold, fixed TRAIN-derived Vahadane target, and coordinate-aware network are reused.\nAny nonzero SICAP Gleason-pattern mask pixel is mapped to binary tumor. Metrics are restricted\nto H&E-derived valid tissue. No SICAPv2 labels are used to tune threshold or model weights.\n\nOUTPUTS\n- tables/per_external_image_metrics.csv\n- tables/per_source_wsi_aggregated_metrics.csv\n- tables/global_external_metrics.json\n- tables/global_external_metrics_95CI_source_wsi_bootstrap.csv\n- figures/external_confusion_matrix.png\n- figures/per_image_dice_distribution.png\n- figures/source_wsi_dice.png\n- visuals_all_images/<source_wsi>/<image_id>/combined_original_gt_model_error.png\n\nThe combined PNG preserves the original 512x512 H&E panel pixel-for-pixel and uses lossless PNG.\n"""
    (results_root / "README_EXTERNAL_RESULTS.txt").write_text(text, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-root", default=os.environ.get("PANDA_WORK_ROOT", r"D:\PANDA_PROSTATE"))
    ap.add_argument("--data-root", default=os.environ.get("PANDA_DATA_ROOT", r"D:\prostat"))
    ap.add_argument("--dataset-url", default=ZIP_API_URL)
    args = ap.parse_args()

    work_root = Path(args.work_root)
    cfg = FinalConfig(data_root=args.data_root, work_root=str(work_root))
    ext_root = work_root / "external_sicapv2"
    download_dir = ext_root / "download"
    extract_root = ext_root / "dataset"
    results_root = ext_root / "results"
    for p in [ext_root, download_dir, extract_root, results_root]:
        _retry_io(work_root, lambda p=p: p.mkdir(parents=True, exist_ok=True), label="INIT")

    zip_path = download_dir / "SICAPv2.zip"
    print("[SICAP DOWNLOAD] Kaggle public mirror first using D: cache; Mendeley only as fallback.", flush=True)

    # Phase 1+2 — preferred source: public Kaggle mirror. This bypasses the
    # current anonymous Mendeley 403 problem and downloads/extracts directly.
    save_state(ext_root, "before_download", preferred_source="kaggle", kaggle_handle=KAGGLE_DATASET_HANDLE)
    got_kaggle = download_kaggle_dataset_resumable(extract_root, work_root)

    if got_kaggle:
        save_state(ext_root, "after_download", source="kaggle", dataset_root=str(extract_root))
        save_state(ext_root, "after_extract", source="kaggle")
    else:
        # Fallback retained for future Mendeley access recovery.
        save_state(ext_root, "before_mendeley_fallback")
        download_resumable(args.dataset_url, zip_path, work_root)
        save_state(ext_root, "after_download", source="mendeley", zip=str(zip_path), size=zip_path.stat().st_size)
        save_state(ext_root, "before_extract")
        extract_resumable(zip_path, extract_root, work_root)
        save_state(ext_root, "after_extract", source="mendeley")

    # Phase 3 — index official SICAPv2 Test partition.
    ds = discover_dataset(extract_root)
    index_csv = results_root / "tables" / "external_test_index.csv"
    df = build_index(ds, index_csv)
    save_state(ext_root, "after_index", test_images=len(df), source_wsi=df.source_wsi.nunique())

    # Phase 4 — model inference with durable per-image cache.
    run_inference(df, cfg, results_root, work_root)

    # Phase 5 — statistics; rebuildable from per-image durable caches.
    save_state(ext_root, "before_statistics")
    ps, source = build_statistics(df, results_root, cfg)

    # Phase 6 — completeness sweep only. Combined visuals are generated live
    # during inference; this fills anything missing after a crash/restart.
    save_state(ext_root, "before_visuals")
    build_all_visuals(ps, results_root, work_root)

    write_final_readme(results_root, df)
    save_state(ext_root, "COMPLETE", test_images=len(df), source_wsi=len(source), results=str(results_root))
    print("\n===============================================================", flush=True)
    print("SICAPv2 EXTERNAL VALIDATION COMPLETE", flush=True)
    print(f"Results: {results_root}", flush=True)
    print("===============================================================", flush=True)


if __name__ == "__main__":
    while True:
        try:
            main()
            break
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print("\n[EXTERNAL PIPELINE ERROR]", type(e).__name__, str(e), flush=True)
            traceback.print_exc()
            # Storage errors are always self-healed. Other errors fail so scientific/data
            # mismatches are not silently guessed; the external watchdog restarts only after
            # storage becomes available.
            work_guess = Path(os.environ.get("PANDA_WORK_ROOT", r"D:\PANDA_PROSTATE"))
            if _storage_like(e) or not _drive_ready(work_guess):
                _wait_storage(work_guess)
                print("[EXTERNAL RESUME] Restarting same phase automatically...", flush=True)
                continue
            raise
