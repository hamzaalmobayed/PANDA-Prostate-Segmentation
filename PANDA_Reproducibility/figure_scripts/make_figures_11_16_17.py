"""Regenerate Figs 11, 16 and 17 of the PANDA manuscript from the saved evaluation outputs.

Fig 11: one held-out PANDA test WSI of the pathologist-guided review
    (paper_results/test_visuals_bad_review/). The earlier figure was the
    composite that GENERATE_BAD_REVIEW_COMPOSITE.py built from this case's saved
    detail images (matched by image correlation 0.999; next case 0.69). It is
    redrawn from the same detail images: H&E, reference and model masks, the
    saved error map, and overlays made as in that composite (green, alpha 0.55).
    The level-0 field of the crop is located exactly in the WSI cache.
Fig 16: four SICAPv2 images of the executed fallback index: source image and
    mask (binary: mask > 0), prediction and error map written by the external
    evaluation.
Fig 17: two PRECISE WSIs. Reference, prediction and error map come from the
    PRECISE caches with the evaluation's own rule; the H&E thumbnail is taken
    from the saved combined visual of each WSI.
Images keep their aspect ratio. Error-map colours are those of the evaluation
code: overlap green, false negative yellow, false positive red, true-negative
tissue grey. Case identifiers and file names are written to the JSON log of
each figure, not to the figure (provenance_log.py).

Usage:
    python make_figures_11_16_17.py <output_dir> [work_root]
"""
import glob
import hashlib
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from PIL import Image

from provenance_log import locate_review_crop, save_figure, source

OUT = Path(sys.argv[1])
if len(sys.argv) < 3 and "PANDA_DATA_ROOT" not in os.environ:
    sys.exit("give work_root as the second argument or set PANDA_DATA_ROOT")
WORK = Path(sys.argv[2] if len(sys.argv) > 2 else os.environ["PANDA_DATA_ROOT"])
THR = 0.43
NAVY = "#0c1644"
TN, TP, FN, FP = (230, 230, 230), (60, 170, 90), (240, 190, 40), (210, 60, 60)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7})

# Fig 11: review case shown in the earlier figure (identified by matching its panels)
FIG11_WSI = "66cad79a848b2b9476a2517a80314327"
# Fig 16: SICAPv2 images of the executed fallback index
FIG16 = [("A", "1.jpg"), ("B", "15.jpg"), ("C", "13221.jpg"), ("D", "2.jpg")]
# Fig 17: PRECISE WSIs
FIG17 = [("A", "sub-07_ses-01"), ("B", "sub-09_ses-01")]


def error_map(gt, pred, valid):
    err = np.full(gt.shape + (3,), 255, np.uint8)
    err[~gt & ~pred & valid] = TN
    err[gt & pred & valid] = TP
    err[gt & ~pred & valid] = FN
    err[~gt & pred & valid] = FP
    return err


def show(ax, img, title=None, gray=False):
    ax.imshow(img, cmap="gray" if gray else None, vmin=0, vmax=255 if gray else None,
              interpolation="nearest" if gray else "antialiased")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.5); s.set_edgecolor("#888")
    if title:
        ax.set_title(title, fontsize=6.8, pad=2)


def legend(fig, y, x=0.17, step=0.17, colors=(TP, FN, FP, TN)):
    fig.text(0.03, y + 0.006, "Error map:", fontsize=6.4, va="center", fontweight="bold")
    for col, lab in zip(colors, ("overlap", "false negative", "false positive", "true-negative tissue")):
        fig.add_artist(Rectangle((x, y), 0.014, 0.012, transform=fig.transFigure, fc=np.array(col) / 255, ec="#666", lw=0.4))
        fig.text(x + 0.02, y + 0.006, lab, fontsize=6.4, va="center")
        x += step


def save(fig, png, tif, record):
    save_figure(fig, OUT, png, tif, record, WORK, __file__)


def bbox(mask, margin=0.03):
    ys, xs = np.where(mask)
    h, w = mask.shape
    my, mx = int(margin * h), int(margin * w)
    return max(0, ys.min() - my), min(h, ys.max() + 1 + my), max(0, xs.min() - mx), min(w, xs.max() + 1 + mx)


# ---------------------------------------------------------------- Fig 11
REVIEW = WORK / "paper_results" / "test_visuals_bad_review"
D11 = REVIEW / f"bad_{FIG11_WSI}"
f11 = {tag: Path(glob.glob(str(D11 / f"*_detail_{tag}.png"))[0]) for tag in ("original", "gt_mask", "model_mask", "error_map")}
he = np.asarray(Image.open(f11["original"]).convert("RGB"))
gt = np.asarray(Image.open(f11["gt_mask"]).convert("L").resize(he.shape[1::-1], Image.NEAREST)) > 127
pred = np.asarray(Image.open(f11["model_mask"]).convert("L").resize(he.shape[1::-1], Image.NEAREST)) > 127
err = np.asarray(Image.open(f11["error_map"]).convert("RGB"))
# the saved review error map uses the evaluation palette of this script
used = {tuple(c) for c in np.unique(err.reshape(-1, 3), axis=0)}
assert used <= {TP, FN, FP, TN, (255, 255, 255)}, used


def overlay(img, mask, color=(95, 205, 95), alpha=0.55):
    out = img.astype(np.float32)
    out[mask] = out[mask] * (1 - alpha) + np.array(color, np.float32) * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


small = lambda a, mask=False: np.asarray(Image.fromarray(a).resize((1024, 1024), Image.NEAREST if mask else Image.LANCZOS))
panels = [(small(he), "(A) H&E"), (small((gt * 255).astype(np.uint8), True), "(B) GT mask"),
          (small((pred * 255).astype(np.uint8), True), "(C) Predicted mask"), (small(overlay(he, gt)), "(D) GT overlay"),
          (small(overlay(he, pred)), "(E) Prediction overlay"), (small(err, True), "(F) Error map")]
m11 = pd.read_csv(REVIEW / "bad_case_manifest_live.csv").set_index("image_id").loc[FIG11_WSI]
per = pd.read_csv(WORK / "paper_results" / "tables" / "per_slide_wsi_metrics.csv").set_index("image_id").loc[FIG11_WSI]
assert abs(per.fusion_dice - m11.fusion_dice) < 1e-6
fig = plt.figure(figsize=(7.5, 5.15))
for i, (img, lab) in enumerate(panels):
    ax = fig.add_axes([0.03 + (i % 3) * 0.322, 0.535 - (i // 3) * 0.455, 0.295, 0.405])
    show(ax, img, None, gray=img.ndim == 2)
    ax.set_title(lab, fontsize=8, fontweight="bold", pad=3)
legend(fig, 0.025)
save(fig, "Fig11_qualitative_good.png", "Fig11.tif", {
    "figure": "Fig 11",
    "panels": {"A-F": {"dataset": "PANDA", "split": "test", "slide_id": FIG11_WSI, "provider": m11.provider,
                       "stored_fused_dice": round(float(per.fusion_dice), 4), "screening_order": int(m11.order_saved),
                       "review_manifest": source(REVIEW / "bad_case_manifest_live.csv", WORK),
                       "per_slide_metrics": source(WORK / "paper_results" / "tables" / "per_slide_wsi_metrics.csv", WORK),
                       "detail_crop": locate_review_crop(WORK, FIG11_WSI, f11["gt_mask"], f11["model_mask"]),
                       "files": {k: source(v, WORK) for k, v in f11.items()}}},
    "parameters": {"overlay": "green (95, 205, 95), alpha 0.55 on mask pixels (GENERATE_BAD_REVIEW_COMPOSITE.py)",
                   "error_map": "saved review error map (evaluation palette; the earlier composite recoloured it)",
                   "display_size_px": 1024,
                   "identification": "earlier Fig 11 = composite_figure.png of this case; panel A correlation 0.9987, next case 0.6868"}})

# ---------------------------------------------------------------- Fig 16
SIC = WORK / "external_sicapv2" / "results"
sic = pd.read_csv(SIC / "tables" / "per_external_image_metrics.csv").set_index("image_name")
fig = plt.figure(figsize=(7.5, 4.0))
fig.text(0.5, 0.975, "SICAPv2 external stress-test examples", ha="center", va="top", fontsize=10, fontweight="bold", color=NAVY)
boxes = [(0.02, 0.50, 0.475, 0.40), (0.505, 0.50, 0.475, 0.40), (0.02, 0.075, 0.475, 0.40), (0.505, 0.075, 0.475, 0.40)]
src16 = {}
dice16 = {}
for (letter, name), (x0, y0, w, h) in zip(FIG16, boxes):
    r = sic.loc[name]
    key = hashlib.sha1(name.encode("utf-8")).hexdigest()[:20]
    files = {"he": Path(r.image_path), "gt": Path(r.mask_path), "model": SIC / "pred_masks" / f"{key}.png",
             "error": SIC / "error_maps" / f"{key}.png"}
    he = np.asarray(Image.open(files["he"]).convert("RGB"))
    size = he.shape[1], he.shape[0]
    gt = (np.asarray(Image.open(files["gt"]).convert("L").resize(size, Image.NEAREST)) > 0) * 255
    model = (np.asarray(Image.open(files["model"]).convert("L").resize(size, Image.NEAREST)) > 0) * 255
    err = np.asarray(Image.open(files["error"]).convert("RGB"))
    fig.add_artist(Rectangle((x0, y0), w, h, transform=fig.transFigure, fill=False, ec="#333", lw=0.8))
    fig.text(x0 + 0.008, y0 + h - 0.012, f"{letter}", fontsize=8.5, fontweight="bold", va="top", color=NAVY)
    pw = (w - 0.03) / 4
    for i, (img, lab) in enumerate(zip((he, gt, model, err), ("H&E", "Reference mask", "Model", "Error map"))):
        ax = fig.add_axes([x0 + 0.015 + i * pw + 0.003, y0 + h * 0.13, pw - 0.006, h * 0.70])
        ax.set_aspect("equal")
        show(ax, img, lab, gray=img.ndim == 2)
    fig.text(x0 + w / 2, y0 + h * 0.055, f"Dice {r.dice:.4f}", ha="center", fontsize=6.4, color="#444")
    src16[letter] = {"dataset": "SICAPv2", "image_name": name, "source_wsi": str(r.source_wsi),
                     "stored_dice": round(float(r.dice), 4), "precision": round(float(r.precision), 4),
                     "recall": round(float(r.recall), 4), "files": {k: source(v, WORK) for k, v in files.items()}}
legend(fig, 0.02)
save(fig, "Fig16_sicap_external.png", "Fig16.tif", {
    "figure": "Fig 16", "panels": src16,
    "parameters": {"index": "all-images fallback index of the external SICAPv2 evaluation (18,765 paired files)",
                   "metrics_table": source(SIC / "tables" / "per_external_image_metrics.csv", WORK),
                   "masks": "binary (> 0), nearest-neighbour resize to the image size",
                   "prediction_file_key": "first 20 hex digits of SHA-1 of the image name"}})

# ---------------------------------------------------------------- Fig 17
PRE = WORK / "external_precise_wsi"


def combined_he(path, shape):
    """H&E panel of a saved combined visual, located from its two black mask panels."""
    a = np.asarray(Image.open(path).convert("RGB"))
    black = a.max(2) < 8
    cols = black.mean(0) > 0.5 * black.mean(0).max()
    runs = np.flatnonzero(np.diff(np.r_[0, cols.astype(int), 0]))
    (g0, g1), (m0, _) = runs[0:2], runs[2:4]
    rows = np.flatnonzero(black[:, g0:g1].mean(1) > 0.5)
    y0, y1 = rows.min(), rows.max() + 1
    d = m0 - g0
    he = a[y0:y1, g0 - d:g1 - d]
    return np.asarray(Image.fromarray(he).resize((shape[1], shape[0]), Image.LANCZOS))


rows17 = []
src17 = {}
for letter, sid in FIG17:
    cache = PRE / "wsi_cache" / f"{sid}_ds16_precise_v1.npz"
    if not cache.exists():
        cache = next((PRE / "wsi_cache").glob(f"{sid}_*.npz"))
    z = np.load(cache)
    gt, valid = z["gt"].astype(bool), z["valid"].astype(bool)
    tissue = z["tissue_eval"].astype(bool) if "tissue_eval" in z.files else np.ones_like(gt)
    pred = (z["fusion"].astype(np.float32) >= THR) & tissue
    vis = PRE / "results" / "visuals_all_wsi" / sid
    he = combined_he(vis / "combined_original_gt_model_error.png", gt.shape)
    y0, y1, x0, x1 = bbox(valid | tissue)
    panels = [he[y0:y1, x0:x1], (gt * 255)[y0:y1, x0:x1], (pred * 255)[y0:y1, x0:x1], error_map(gt, pred, valid)[y0:y1, x0:x1]]
    m = json.loads((vis / "metrics.json").read_text())
    rows17.append((letter, sid, panels, m))
    src17[letter] = {"dataset": "PRECISE", "wsi": sid, "fusion_dice": round(m["fusion_dice"], 4),
                     "fusion_precision": round(m["fusion_precision"], 4), "fusion_recall": round(m["fusion_recall"], 4),
                     "crop_rows_cols_on_16x_grid": [int(y0), int(y1), int(x0), int(x1)],
                     "files": {"wsi_cache": source(cache, WORK),
                               "combined_visual_for_he": source(vis / "combined_original_gt_model_error.png", WORK),
                               "metrics": source(vis / "metrics.json", WORK)}}

# layout in inches: each row's panel size follows its WSI's aspect ratio
W_IN, GAP_IN, MAX_H_IN = 7.5, 0.08, 3.5
TOP_IN, TITLE_IN, LABEL_IN, LEGEND_IN = 0.45, 0.22, 0.3, 0.3
sizes = []
for _, _, panels, _ in rows17:
    h, w = panels[0].shape[:2]
    pw = min((W_IN - 0.4 - 3 * GAP_IN) / 4, MAX_H_IN * w / h)
    sizes.append((pw, pw * h / w))
H_IN = TOP_IN + sum(ph + TITLE_IN + LABEL_IN for _, ph in sizes) + LEGEND_IN
fig = plt.figure(figsize=(W_IN, H_IN))
fig.text(0.5, 1 - 0.12 / H_IN, "PRECISE external full-WSI examples", ha="center", va="top", fontsize=10,
         fontweight="bold", color=NAVY)
y = H_IN - TOP_IN
for (letter, sid, panels, m), (pw, ph) in zip(rows17, sizes):
    y -= TITLE_IN + ph
    x = (W_IN - (4 * pw + 3 * GAP_IN)) / 2
    fig.text(0.2 / W_IN, (y + ph + TITLE_IN) / H_IN, letter, fontsize=9, fontweight="bold", color=NAVY, va="top")
    for img, lab in zip(panels, ("H&E", "Reference tumor", "Model", "Error map")):
        show(fig.add_axes([x / W_IN, y / H_IN, pw / W_IN, ph / H_IN]), img, lab, gray=img.ndim == 2)
        x += pw + GAP_IN
    y -= LABEL_IN
    fig.text(0.5, (y + LABEL_IN * 0.4) / H_IN, f"Dice {m['fusion_dice']:.4f} · precision "
             f"{m['fusion_precision']:.4f} · recall {m['fusion_recall']:.4f}", ha="center", fontsize=6.4, color="#444")
legend(fig, 0.1 / H_IN)
save(fig, "Fig17_precise_external.png", "Fig17.tif", {
    "figure": "Fig 17", "panels": src17,
    "parameters": {"prediction": f"fused probability >= {THR} inside the evaluation tissue mask", "grid": "16x",
                   "he_thumbnail": "H&E panel of the saved combined visual, resized to the cache grid",
                   "crop": "bounding box of valid or tissue pixels plus a 3% margin"}})
