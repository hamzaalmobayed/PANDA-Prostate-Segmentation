"""Shared output and provenance logging for the figure scripts.

save_figure() writes a figure as PNG and PLOS TIFF (RGB, LZW, 300 dpi, at most
2250 px wide) and a JSON log next to it, or in PANDA_PROVENANCE_DIR when that
variable is set. The log is the only place where case identifiers, patch
coordinates, file names and generation settings are kept; the figures and the
manuscript do not carry them.

Each log records:
  figure, outputs (file, SHA-256, pixel size), script (file, SHA-256, command),
  generation time (UTC), software versions, every source file (path relative
  to PANDA_DATA_ROOT and SHA-256), a per-panel description (case, provider,
  split, level, level-0 coordinates, image/mask files), the parameters used and
  the values printed in the figure.

provenance_index.py builds the audit tables from these logs and re-checks every
hash (--verify).
"""
import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

MAX_W = 2250


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rel(path, work):
    """Path relative to the work root, or tagged with its dataset folder when outside it."""
    p = Path(path).resolve()
    try:
        return p.relative_to(Path(work).resolve()).as_posix()
    except ValueError:
        return "<outside PANDA_DATA_ROOT>/" + "/".join(p.parts[-4:])


def source(path, work, **extra):
    rec = {"path": rel(path, work), "sha256": sha(path)}
    rec.update(extra)
    return rec


def environment():
    env = {"python": platform.python_version(), "platform": platform.platform(),
           "numpy": np.__version__, "matplotlib": matplotlib.__version__, "pillow": Image.__version__}
    for mod in ("pandas", "torch", "torchvision", "timm", "cv2", "tiatoolbox"):
        if mod in sys.modules:
            env[mod] = getattr(sys.modules[mod], "__version__", "?")
    return env


def _plain(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, Path):
        return o.as_posix()
    raise TypeError(type(o))


def plos_export(png, tif):
    im = Image.open(png).convert("RGB")
    if im.width > MAX_W:
        im = im.resize((MAX_W, round(im.height * MAX_W / im.width)), Image.LANCZOS)
    im.save(png, dpi=(300, 300))
    im.save(tif, compression="tiff_lzw", dpi=(300, 300))
    return im.size


def prov_dir(out_dir):
    d = Path(os.environ.get("PANDA_PROVENANCE_DIR", out_dir))
    d.mkdir(parents=True, exist_ok=True)
    return d


def patch_record(row, work, **extra):
    """Describe one stored PANDA patch (a row of manifests/patches.csv or selected/train_selected.csv)."""
    rec = {"slide_id": row.image_id, "provider": row.provider, "split": row.split, "level": int(row.level),
           "category": row.category, "x0_level0": int(row.x0), "y0_level0": int(row.y0),
           "field_level0_px": 512 * int(round(float(row.target_downsample))),
           "tissue_fraction": round(float(row.tissue_fraction), 6),
           "tumor_fraction_of_valid": round(float(row.tumor_fraction), 6),
           "image": source(row.img, work), "tumor_mask": source(row["mask"], work)}
    if isinstance(row.get("valid_mask", None), str) and Path(row.valid_mask).exists():
        rec["valid_mask"] = source(row.valid_mask, work)
    rec.update(extra)
    return rec


def stain_cache_file(work, img_path):
    """Vahadane-normalized copy of a patch written by the training pipeline, or None.

    Same key as stain_cache_path() in panda_camelyon_style_pipeline.py: the first
    12 hex digits of SHA-1 of the patch path as stored in the manifest, under
    cache/stain_v10/vahadane/<target signature>/.
    """
    key = hashlib.sha1(str(Path(str(img_path))).encode("utf-8")).hexdigest()[:12] + ".png"
    hits = sorted((Path(work) / "cache" / "stain_v10" / "vahadane").glob(f"*/{key}"))
    return hits[0] if hits else None


def locate_review_crop(work, slide_id, gt_png, model_png, thr=0.43, ds=16):
    """Level-0 field of a saved review detail crop, found in the slide's WSI cache.

    The detail masks are the 16x-grid masks upsampled by 16; their block averages
    are matched (squared difference) against the cached reference and fused
    prediction. Returns the box only for an exact, unique match.
    """
    import cv2
    g = np.asarray(Image.open(gt_png).convert("L")) > 127
    m = np.asarray(Image.open(model_png).convert("L")) > 127
    cache = Path(work) / "paper_results" / "wsi_cache" / f"test_{slide_id}_ds16_validsupport_v2.npz"
    z = np.load(cache)
    valid = z["valid"].astype(bool)
    img = np.dstack([z["gt"].astype(bool) & valid, (z["fusion"].astype(np.float32) >= thr) & valid]).astype(np.float32)
    h, w = g.shape[0] // ds, g.shape[1] // ds
    tpl = np.dstack([cv2.resize(a.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA) for a in (g, m)])
    ph, pw = max(0, h - img.shape[0]), max(0, w - img.shape[1])
    img = np.pad(img, ((0, ph), (0, pw), (0, 0)))
    r = cv2.matchTemplate(img, tpl, cv2.TM_SQDIFF)
    zero = np.argwhere(r < 1e-6)
    assert len(zero) == 1, f"crop of {slide_id} not uniquely located ({len(zero)} exact positions)"
    row, col = (int(v) for v in zero[0])
    return {"wsi_cache": source(cache, work), "grid": f"{ds}x",
            "crop_rows_cols_on_grid": [row, row + h, col, col + w],
            "x0_level0": col * ds, "y0_level0": row * ds, "width_level0_px": w * ds, "height_level0_px": h * ds,
            "match": "exact (squared difference 0, unique)"}


def save_figure(fig, out_dir, png_name, tif_name, record, work, script, extra_files=()):
    """Save fig and its provenance log. record holds figure-specific fields."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    png, tif = out_dir / png_name, out_dir / tif_name
    fig.savefig(png, dpi=300, facecolor="white")
    plt.close(fig)
    size = plos_export(png, tif)
    outputs = {"png": {"file": png.name, "sha256": sha(png), "size_px": list(size)},
               "tif": {"file": tif.name, "sha256": sha(tif), "size_px": list(size)}}
    pdir = prov_dir(out_dir)
    for f in extra_files:  # intermediate arrays saved for exact re-use (e.g. a normalized patch)
        outputs[Path(f).stem] = {"file": Path(f).name, "sha256": sha(f)}
    script = Path(script)
    log = {"figure": record.pop("figure"),
           "outputs": outputs,
           "script": {"file": script.name, "sha256": sha(script),
                      "helper": {"file": Path(__file__).name, "sha256": sha(__file__)},
                      "command": f"python {script.name} <output_dir> <PANDA_DATA_ROOT>"},
           "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "environment": environment(),
           "paths_relative_to": "PANDA_DATA_ROOT"}
    log.update(record)
    (pdir / (Path(png_name).stem + "_sources.json")).write_text(json.dumps(log, indent=2, default=_plain), encoding="utf-8")
    print("saved", png.name, size)
    return log
