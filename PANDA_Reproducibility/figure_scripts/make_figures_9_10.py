"""Figs 9 and 10 of the PANDA manuscript (numbering of the revision of 30 Oct 2026).

PANDA panels use the detail images saved during the test run for each
reviewed WSI (paper_results/test_visuals_bad_review/bad_<id>/*_detail_*.png).
Fig 10D shows the SICAPv2 tile of the earlier figure (0.jpg of the renamed
Kaggle copy, byte-identical to the official release tile named in TILE) with
the release's lossless reference mask and the prediction and error map written
by analysis_scripts/revision_2026-10-30/sicap_figure_tile.py
(external_sicapv2_official/results/figure_tile/). Case labels and findings are
the pathologist's; "GT" is written "Reference", and the labels of 9B and of
the Fig 10 heading were reworded (D1, D4). Drawn with SciencePlots and the
PLOS font settings (provenance_log.plos_style). Case identifiers, providers,
stored Dice values and the level-0 field of each detail crop (located exactly
in the slide's WSI cache) are written to the JSON log of each figure, not to
the figure (provenance_log.py).

Usage:
    python make_figures_9_10.py <output_dir> [work_root]

work_root holds paper_results/, external_sicapv2/results/tables/ and
external_sicapv2_official/results/figure_tile/. If omitted, PANDA_DATA_ROOT is
used.
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
REVIEW = WORK / "paper_results" / "test_visuals_bad_review"
TILE_DIR = WORK / "external_sicapv2_official" / "results" / "figure_tile"
TILE = "16B0001851_Block_Region_1_0_0_xini_6803_yini_59786"
plos_style()
# colours of the saved evaluation error maps (checked for every panel in case_panels)
COLOURS = [((60, 170, 90), "Overlap"), ((240, 190, 40), "False negative"), ((210, 60, 60), "False positive"),
           ((230, 230, 230), "True-negative tissue")]
PALETTE = {c for c, _ in COLOURS} | {(255, 255, 255)}
LEGEND = [Patch(fc=np.array(c) / 255, ec="0.4", lw=0.4, label=l) for c, l in COLOURS]
COLUMNS = ("H&E", "Reference", "Prediction", "Error map")

# The name of case 9C ("crowded cells") is to be replaced by the standard entity
# name that S.E.D. gives (manuscript, Results, pathologist-guided review).
FIG9 = [("A", "Benign mimic", "Benign glands mistaken as tumor.", "15bf6b34c8ce10b0eddf687e3a670e4d"),
        ("B", "Possible benign mimic", "Dense benign tissue resembles carcinoma.", "13813facc985c2ea007fdf764ef98a5a"),
        ("C", "Crowded cells", "Crowded cells interpreted as tumor.", "b49d54b4f7e73808440c36fa240614c0")]
FIG10 = [("A", "Broader reference than prediction", "Reference includes additional benign tissue.",
          "9848ee7cc45f7cd373b155191085f06a"),
         ("B", "Possible stain-associated discrepancy", "Annotation may extend into non-tumorous tissue.",
          "cf1c7e7e632db44c5ca6de473da247e6"),
         ("C", "Connected-foci convention", "Reference joins separate tumor foci.", "cfbe34e8826d4e2bb02061d9ef056e83"),
         ("D", "Luminal spaces", "Reference includes internal luminal spaces.", "SICAPv2")]


def load(p, size=520, mask=False):
    im = Image.open(p)
    if mask:
        a = np.asarray(im.convert("L")) > 0
        return np.asarray(Image.fromarray((a * 255).astype(np.uint8)).resize((size, size), Image.NEAREST))
    return np.asarray(im.convert("RGB").resize((size, size), Image.LANCZOS))


man = pd.read_csv(REVIEW / "bad_case_manifest_live.csv").set_index("image_id")


def case_panels(cid):
    """Return (four images, log record of the case)."""
    if cid == "SICAPv2":
        files = {"he": TILE_DIR / f"{TILE}.jpg", "gt": TILE_DIR / f"{TILE}.png",
                 "model": TILE_DIR / f"{TILE}_pred_mask.png", "error": TILE_DIR / f"{TILE}_error_map.png"}
        info = json.loads((TILE_DIR / f"{TILE}.json").read_text())
        first = WORK / "external_sicapv2" / "results" / "tables" / "per_external_image_metrics.csv"
        rec = {"dataset": "SICAPv2 official release v2", "tile": f"{TILE}.jpg",
               "earlier_figure": "0.jpg of the renamed Kaggle copy (first run); same H&E file (SHA-256)",
               "first_run_dice": round(float(pd.read_csv(first).set_index("image_name").loc["0.jpg", "dice"]), 4),
               "first_run_metrics_table": source(first, WORK),
               "redrawn": {"coordinates": info["coordinates"], "threshold": info["threshold"], "counts": info["counts"],
                           "counts_official_run": info["counts_official_run_per_tile_metrics_const"],
                           "dice": info["dice"], "tile_record": source(TILE_DIR / f"{TILE}.json", WORK)}}
    else:
        d = REVIEW / f"bad_{cid}"
        pick = lambda tag: Path(glob.glob(str(d / f"*_detail_{tag}.png"))[0])
        files = {"he": pick("original"), "gt": pick("gt_mask"), "model": pick("model_mask"), "error": pick("error_map")}
        m = man.loc[cid]
        rec = {"dataset": "PANDA", "split": "test", "slide_id": cid, "provider": m.provider,
               "stored_fused_dice": round(float(m.fusion_dice), 4), "screening_order": int(m.order_saved),
               "review_manifest": source(REVIEW / "bad_case_manifest_live.csv", WORK),
               "detail_crop": locate_review_crop(WORK, cid, files["gt"], files["model"])}
    rec["files"] = {k: source(v, WORK) for k, v in files.items()}
    imgs = [load(files["he"]), load(files["gt"], mask=True), load(files["model"], mask=True), load(files["error"])]
    used = {tuple(c) for c in np.unique(np.asarray(Image.open(files["error"]).convert("RGB")).reshape(-1, 3), axis=0)}
    assert used <= PALETTE, (cid, used - PALETTE)
    return imgs, rec


def draw_case(fig, x0, y_top, pw, gap, H, case, column_labels):
    """Draw one case with its top-left corner at (x0, y_top) inches; return (log record, bottom y)."""
    letter, title, finding, cid = case
    fig.text(x0 / 7.5, y_top / H, letter, fontsize=10, fontweight="bold", va="top")
    fig.text((x0 + 0.22) / 7.5, (y_top - 0.02) / H, title, fontweight="bold", va="top")
    y = y_top - 0.25 - (0.2 if column_labels else 0) - pw
    imgs, rec = case_panels(cid)
    for i, (img, lab) in enumerate(zip(imgs, COLUMNS)):
        ax = fig.add_axes([(x0 + i * (pw + gap)) / 7.5, y / H, pw / 7.5, pw / H])
        ax.imshow(img, cmap="gray" if img.ndim == 2 else None, vmin=0, vmax=255,
                  interpolation="nearest" if img.ndim == 2 else "antialiased")
        ax.set_xticks([]); ax.set_yticks([]); ax.minorticks_off()
        for s in ax.spines.values():
            s.set_linewidth(0.5); s.set_edgecolor("0.55")
        if column_labels:
            ax.text(0.5, 1.03, lab, transform=ax.transAxes, ha="center", va="bottom")
    width = 4 * pw + 3 * gap
    fig.text((x0 + width / 2) / 7.5, (y - 0.06) / H, finding, ha="center", va="top")
    rec.update({"category": title, "pathologist_finding": finding})
    return {letter: rec}, y - 0.3


def save(fig, png, tif, figure, panels):
    fig.legend(handles=LEGEND, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=4, frameon=False,
               title="Error map", title_fontsize=8)
    save_figure(fig, OUT, png, tif, {
        "figure": figure, "panels": panels,
        "parameters": {"panel_size_px": 520, "masks": "binary (> 0), nearest-neighbour resize",
                       "error_map": "saved evaluation error map, colours as in the legend",
                       "labels": "case categories and findings from the pathologist review; GT written as Reference; "
                                 "9B 'Hypercellular benign mimic' -> 'Possible benign mimic' (D4)"}}, WORK, __file__)


# Fig 9: three stacked cases (layout in inches, no layout engine)
PW, GAP = 1.72, 0.1
H = 0.1 + 3 * (0.25 + PW + 0.3) + 0.2 + 2 * 0.15 + 0.6
fig = plt.figure(figsize=(7.5, H), layout="none")
x0 = (7.5 - (4 * PW + 3 * GAP)) / 2
y = H - 0.1
src = {}
for k, case in enumerate(FIG9):
    rec, y = draw_case(fig, x0, y, PW, GAP, H, case, column_labels=k == 0)
    src.update(rec)
    y -= 0.15
save(fig, "Fig09_model_related_errors.png", "Fig9.tif", "Fig 9", src)

# Fig 10: 2 x 2 cases
PW, GAP = 0.83, 0.06
H = 0.1 + 2 * (0.25 + PW + 0.3) + 0.2 + 0.2 + 0.6
fig = plt.figure(figsize=(7.5, H), layout="none")
width = 4 * PW + 3 * GAP
xs = (0.3, 7.5 - 0.15 - width)
src = {}
y_row = H - 0.1
for r in range(2):
    for c in range(2):
        rec, y_end = draw_case(fig, xs[c], y_row, PW, GAP, H, FIG10[2 * r + c], column_labels=r == 0)
        src.update(rec)
    y_row = y_end - 0.2
save(fig, "Fig10_annotation_discrepancies.png", "Fig10.tif", "Fig 10", src)
