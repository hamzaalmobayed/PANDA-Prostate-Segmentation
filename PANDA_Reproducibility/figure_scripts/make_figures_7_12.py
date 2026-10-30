"""Figs 7 and 12 of the PANDA manuscript (numbering of the revision of 30 Oct 2026).

Fig 7: one held-out PANDA test WSI of the pathologist-guided review
    (paper_results/test_visuals_bad_review/). The figure before 28 Oct 2026 was
    the composite that GENERATE_BAD_REVIEW_COMPOSITE.py built from this case's
    saved detail images (matched by image correlation 0.999; next case 0.69). It
    is redrawn from the same detail images: H&E, reference and model masks, the
    saved error map, and overlays made as in that composite (green, alpha 0.55).
    The level-0 field of the crop is located exactly in the WSI cache.
Fig 12: two PRECISE WSIs. Reference, prediction and error map come from the
    PRECISE caches with the evaluation's own rule; the H&E thumbnail is taken
    from the saved combined visual of each WSI.
Images keep their aspect ratio. Error-map colours are those of the evaluation
code: overlap green, false negative yellow, false positive red, true-negative
tissue grey. Drawn with SciencePlots and the PLOS font settings
(provenance_log.plos_style). Case identifiers and file names are written to
the JSON log of each figure, not to the figure (provenance_log.py).

Usage:
    python make_figures_7_12.py <output_dir> [work_root]
"""
import glob
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from PIL import Image

from provenance_log import locate_review_crop, plos_style, save_figure, source

OUT = Path(sys.argv[1])
if len(sys.argv) < 3 and "PANDA_DATA_ROOT" not in os.environ:
    sys.exit("give work_root as the second argument or set PANDA_DATA_ROOT")
WORK = Path(sys.argv[2] if len(sys.argv) > 2 else os.environ["PANDA_DATA_ROOT"])
THR = 0.43
TN, TP, FN, FP = (230, 230, 230), (60, 170, 90), (240, 190, 40), (210, 60, 60)
plos_style()

# Fig 7: review case shown in the earlier figure (identified by matching its panels)
FIG7_WSI = "66cad79a848b2b9476a2517a80314327"
# Fig 12: PRECISE WSIs
FIG12 = [("A", "sub-07_ses-01"), ("B", "sub-09_ses-01")]
LEGEND = [Patch(fc=np.array(c) / 255, ec="0.4", lw=0.4, label=l) for c, l in
          ((TP, "Overlap"), (FN, "False negative"), (FP, "False positive"), (TN, "True-negative tissue"))]


def error_map(gt, pred, valid):
    err = np.full(gt.shape + (3,), 255, np.uint8)
    err[~gt & ~pred & valid] = TN
    err[gt & pred & valid] = TP
    err[gt & ~pred & valid] = FN
    err[~gt & pred & valid] = FP
    return err


def show(ax, img, gray=False):
    ax.imshow(img, cmap="gray" if gray else None, vmin=0, vmax=255 if gray else None,
              interpolation="nearest" if gray else "antialiased")
    ax.set_xticks([]); ax.set_yticks([]); ax.minorticks_off()
    for s in ax.spines.values():
        s.set_linewidth(0.5); s.set_edgecolor("0.55")


def letter(ax, s, x=-0.03, y=1.0):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=10, fontweight="bold", ha="right", va="top")


def save(fig, png, tif, record):
    save_figure(fig, OUT, png, tif, record, WORK, __file__)


def bbox(mask, margin=0.03):
    ys, xs = np.where(mask)
    h, w = mask.shape
    my, mx = int(margin * h), int(margin * w)
    return max(0, ys.min() - my), min(h, ys.max() + 1 + my), max(0, xs.min() - mx), min(w, xs.max() + 1 + mx)


# ---------------------------------------------------------------- Fig 7
REVIEW = WORK / "paper_results" / "test_visuals_bad_review"
D7 = REVIEW / f"bad_{FIG7_WSI}"
f7 = {tag: Path(glob.glob(str(D7 / f"*_detail_{tag}.png"))[0]) for tag in ("original", "gt_mask", "model_mask", "error_map")}
he = np.asarray(Image.open(f7["original"]).convert("RGB"))
gt = np.asarray(Image.open(f7["gt_mask"]).convert("L").resize(he.shape[1::-1], Image.NEAREST)) > 127
pred = np.asarray(Image.open(f7["model_mask"]).convert("L").resize(he.shape[1::-1], Image.NEAREST)) > 127
err = np.asarray(Image.open(f7["error_map"]).convert("RGB"))
# the saved review error map uses the evaluation palette of this script
used = {tuple(c) for c in np.unique(err.reshape(-1, 3), axis=0)}
assert used <= {TP, FN, FP, TN, (255, 255, 255)}, used


def overlay(img, mask, color=(95, 205, 95), alpha=0.55):
    out = img.astype(np.float32)
    out[mask] = out[mask] * (1 - alpha) + np.array(color, np.float32) * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


small = lambda a, mask=False: np.asarray(Image.fromarray(a).resize((1024, 1024), Image.NEAREST if mask else Image.LANCZOS))
panels = [(small(he), "H&E"), (small((gt * 255).astype(np.uint8), True), "Reference"),
          (small((pred * 255).astype(np.uint8), True), "Prediction"), (small(overlay(he, gt)), "Reference overlay"),
          (small(overlay(he, pred)), "Prediction overlay"), (small(err, True), "Error map")]
m7 = pd.read_csv(REVIEW / "bad_case_manifest_live.csv").set_index("image_id").loc[FIG7_WSI]
per = pd.read_csv(WORK / "paper_results" / "tables" / "per_slide_wsi_metrics.csv").set_index("image_id").loc[FIG7_WSI]
assert abs(per.fusion_dice - m7.fusion_dice) < 1e-6
fig, axs = plt.subplots(2, 3, figsize=(7.5, 5.6))
for ax, (img, lab), let in zip(axs.flat, panels, "ABCDEF"):
    show(ax, img, gray=img.ndim == 2)
    ax.set_xlabel(lab)
    letter(ax, let)
fig.legend(handles=LEGEND, loc="outside lower center", ncol=4, frameon=False, title="Error map", title_fontsize=8)
save(fig, "Fig07_qualitative_good.png", "Fig7.tif", {
    "figure": "Fig 7",
    "panels": {"A-F": {"dataset": "PANDA", "split": "test", "slide_id": FIG7_WSI, "provider": m7.provider,
                       "stored_fused_dice": round(float(per.fusion_dice), 4), "screening_order": int(m7.order_saved),
                       "review_manifest": source(REVIEW / "bad_case_manifest_live.csv", WORK),
                       "per_slide_metrics": source(WORK / "paper_results" / "tables" / "per_slide_wsi_metrics.csv", WORK),
                       "detail_crop": locate_review_crop(WORK, FIG7_WSI, f7["gt_mask"], f7["model_mask"]),
                       "files": {k: source(v, WORK) for k, v in f7.items()}}},
    "parameters": {"overlay": "green (95, 205, 95), alpha 0.55 on mask pixels (GENERATE_BAD_REVIEW_COMPOSITE.py)",
                   "error_map": "saved review error map (evaluation palette; the earlier composite recoloured it)",
                   "display_size_px": 1024,
                   "identification": "earlier Fig 11 = composite_figure.png of this case; panel A correlation 0.9987, next case 0.6868"}})

# ---------------------------------------------------------------- Fig 12
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


rows12 = []
src12 = {}
for let, sid in FIG12:
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
    rows12.append((let, sid, panels, m))
    src12[let] = {"dataset": "PRECISE", "wsi": sid, "fusion_dice": round(m["fusion_dice"], 4),
                  "fusion_precision": round(m["fusion_precision"], 4), "fusion_recall": round(m["fusion_recall"], 4),
                  "crop_rows_cols_on_16x_grid": [int(y0), int(y1), int(x0), int(x1)],
                  "files": {"wsi_cache": source(cache, WORK),
                            "combined_visual_for_he": source(vis / "combined_original_gt_model_error.png", WORK),
                            "metrics": source(vis / "metrics.json", WORK)}}

# layout in inches (no layout engine): one row per WSI, panel height from the WSI's aspect ratio
W_IN, PW, GAP = 7.5, 1.6, 0.08
TOP, HEAD, METRICS, ROW_GAP, LEGEND_H = 0.1, 0.25, 0.32, 0.12, 0.62
sizes = [PW * r[2][0].shape[0] / r[2][0].shape[1] for r in rows12]
H_IN = TOP + HEAD + sum(ph + METRICS for ph in sizes) + ROW_GAP * (len(rows12) - 1) + LEGEND_H
assert H_IN <= 8.75, H_IN  # PLOS maximum figure height
fig = plt.figure(figsize=(W_IN, H_IN), layout="none")
x_left = (W_IN - (4 * PW + 3 * GAP)) / 2
y = H_IN - TOP - HEAD
for i, ((let, sid, panels, m), ph) in enumerate(zip(rows12, sizes)):
    y -= ph
    for j, (img, lab) in enumerate(zip(panels, ("H&E", "Reference", "Prediction", "Error map"))):
        ax = fig.add_axes([(x_left + j * (PW + GAP)) / W_IN, y / H_IN, PW / W_IN, ph / H_IN])
        show(ax, img, gray=img.ndim == 2)
        if i == 0:
            ax.text(0.5, 1.02, lab, transform=ax.transAxes, ha="center", va="bottom")
        if j == 0:
            letter(ax, let, x=-0.04)
    fig.text(0.5, (y - 0.08) / H_IN, f"Dice {m['fusion_dice']:.4f}, precision {m['fusion_precision']:.4f}, "
             f"recall {m['fusion_recall']:.4f}", ha="center", va="top")
    y -= METRICS + ROW_GAP
fig.legend(handles=LEGEND, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=4, frameon=False, title="Error map",
           title_fontsize=8)
save(fig, "Fig12_precise_external.png", "Fig12.tif", {
    "figure": "Fig 12", "panels": src12,
    "parameters": {"prediction": f"fused probability >= {THR} inside the evaluation tissue mask", "grid": "16x",
                   "he_thumbnail": "H&E panel of the saved combined visual, resized to the cache grid",
                   "crop": "bounding box of valid or tissue pixels plus a 3% margin"}})
