"""Figs 9 and 10 of the PANDA manuscript (numbering of the revision of 30 Oct 2026).

Both figures use the detail images saved during the test run for each
reviewed WSI (paper_results/test_visuals_bad_review/bad_<id>/*_detail_*.png).
Case labels and findings are the pathologist's (S.E.D.); "GT" is written
"Reference". Labels of 31 Oct 2026: 9A "mistaken for tumor"; 9B finding as in
the text (densely packed cells in benign-appearing tissue); 9C "Crowded
hyperplastic glands" (S.E.D.: closely packed hyperplastic glands with reduced
stroma, with mild nuclear enlargement and hyperchromasia of the epithelium,
were predicted as tumor). The SICAPv2 panel D of Fig 10
was removed: on the official release mask the tile did not show the reported
finding. Drawn with SciencePlots and the PLOS font settings
(provenance_log.plos_style). Case identifiers, providers, stored Dice values
and the level-0 field of each detail crop (located exactly in the slide's WSI
cache) are written to the JSON log of each figure, not to the figure
(provenance_log.py).

Usage:
    python make_figures_9_10.py <output_dir> [work_root]

work_root holds paper_results/. If omitted, PANDA_DATA_ROOT is used.
"""
import glob
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
plos_style()
# colours of the saved evaluation error maps (checked for every panel in case_panels); white = not scored
COLOURS = [((60, 170, 90), "Overlap"), ((240, 190, 40), "False negative"), ((210, 60, 60), "False positive"),
           ((230, 230, 230), "True-negative tissue")]
PALETTE = {c for c, _ in COLOURS} | {(255, 255, 255)}
LEGEND = [Patch(fc=np.array(c) / 255, ec="0.4", lw=0.4, label=l) for c, l in COLOURS]
COLUMNS = ("H&E", "Reference", "Prediction", "Error map")

FIG9 = [("A", "Benign mimic", "Benign glands mistaken for tumor.", "15bf6b34c8ce10b0eddf687e3a670e4d"),
        ("B", "Possible benign mimic", "Densely packed small cells in benign-appearing tissue resemble carcinoma.",
         "13813facc985c2ea007fdf764ef98a5a"),
        ("C", "Crowded hyperplastic glands", "Closely packed hyperplastic glands with reduced stroma predicted as tumor.",
         "b49d54b4f7e73808440c36fa240614c0")]
FIG10 = [("A", "Broader reference than prediction", "Reference includes additional benign tissue.",
          "9848ee7cc45f7cd373b155191085f06a"),
         ("B", "Possible stain-associated discrepancy", "Annotation may extend into non-tumorous tissue.",
          "cf1c7e7e632db44c5ca6de473da247e6"),
         ("C", "Connected-foci convention", "Reference joins separate tumor foci.", "cfbe34e8826d4e2bb02061d9ef056e83")]


def load(p, size=520, mask=False):
    im = Image.open(p)
    if mask:
        a = np.asarray(im.convert("L")) > 0
        return np.asarray(Image.fromarray((a * 255).astype(np.uint8)).resize((size, size), Image.NEAREST))
    return np.asarray(im.convert("RGB").resize((size, size), Image.LANCZOS))


man = pd.read_csv(REVIEW / "bad_case_manifest_live.csv").set_index("image_id")


def case_panels(cid):
    """Return (four images, log record of the case)."""
    d = REVIEW / f"bad_{cid}"
    pick = lambda tag: Path(glob.glob(str(d / f"*_detail_{tag}.png"))[0])
    files = {"he": pick("original"), "gt": pick("gt_mask"), "model": pick("model_mask"), "error": pick("error_map")}
    m = man.loc[cid]
    rec = {"dataset": "PANDA", "split": "test", "slide_id": cid, "provider": m.provider,
           "stored_fused_dice": round(float(m.fusion_dice), 4), "screening_order": int(m.order_saved),
           "review_manifest": source(REVIEW / "bad_case_manifest_live.csv", WORK),
           "detail_crop": locate_review_crop(WORK, cid, files["gt"], files["model"]),
           "files": {k: source(v, WORK) for k, v in files.items()}}
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


def stacked(cases, png, tif, figure):
    """Three cases stacked (layout in inches, no layout engine)."""
    pw, gap = 1.72, 0.1
    H = 0.1 + len(cases) * (0.25 + pw + 0.3) + 0.2 + (len(cases) - 1) * 0.15 + 0.6
    fig = plt.figure(figsize=(7.5, H), layout="none")
    x0 = (7.5 - (4 * pw + 3 * gap)) / 2
    y = H - 0.1
    src = {}
    for k, case in enumerate(cases):
        rec, y = draw_case(fig, x0, y, pw, gap, H, case, column_labels=k == 0)
        src.update(rec)
        y -= 0.15
    fig.legend(handles=LEGEND, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=4, frameon=False,
               title="Error map (white: not scored)", title_fontsize=8)
    save_figure(fig, OUT, png, tif, {
        "figure": figure, "panels": src,
        "parameters": {"panel_size_px": 520, "masks": "binary (> 0), nearest-neighbour resize",
                       "error_map": "saved evaluation error map, colours as in the legend; white = outside the scored region",
                       "labels": "case categories and findings from the pathologist review (S.E.D.); GT written as Reference"}},
        WORK, __file__)


stacked(FIG9, "Fig09_model_related_errors.png", "Fig9.tif", "Fig 9")
stacked(FIG10, "Fig10_annotation_discrepancies.png", "Fig10.tif", "Fig 10")
