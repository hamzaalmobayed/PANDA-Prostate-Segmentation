"""Regenerate Figs 14 and 15 from the saved review outputs.

PANDA panels use the detail images saved during the test run for each
reviewed WSI (paper_results/test_visuals_bad_review/bad_<id>/*_detail_*.png).
Fig 15D uses the SICAPv2 image, reference mask, prediction and error map
written by the external evaluation. Case labels and interpretations are the
pathologist's, unchanged from the earlier figure version. Case identifiers,
providers, stored Dice values and the level-0 field of each detail crop
(located exactly in the slide's WSI cache) are written to the JSON log of each
figure, not to the figure (provenance_log.py).

Usage:
    python make_figures_14_15.py <output_dir> [work_root]

work_root holds paper_results/ and external_sicapv2/results/. If omitted,
PANDA_DATA_ROOT is used.
"""
import glob
import hashlib
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
REVIEW = WORK / "paper_results" / "test_visuals_bad_review"
SICAP = WORK / "external_sicapv2" / "results"
NAVY = "#0c1644"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7})
# colours of the saved evaluation error maps (checked for every panel in case_panels)
LEGEND = [((60, 170, 90), "overlap"), ((240, 190, 40), "false negative"), ((210, 60, 60), "false positive"),
          ((230, 230, 230), "true-negative tissue")]
PALETTE = {c for c, _ in LEGEND} | {(255, 255, 255)}

FIG14 = [("A", "Benign mimic", "Benign glands mistaken as tumor.", "15bf6b34c8ce10b0eddf687e3a670e4d"),
         ("B", "Hypercellular benign mimic", "Dense benign tissue resembles carcinoma.", "13813facc985c2ea007fdf764ef98a5a"),
         ("C", "Crowded cells", "Crowded cells interpreted as tumor.", "b49d54b4f7e73808440c36fa240614c0")]
FIG15 = [("A", "Broader GT than prediction", "GT includes additional benign tissue.", "9848ee7cc45f7cd373b155191085f06a"),
         ("B", "Possible stain-associated discrepancy", "Annotation may extend into non-tumorous tissue.", "cf1c7e7e632db44c5ca6de473da247e6"),
         ("C", "Connected-foci convention", "GT joins separate tumor foci.", "cfbe34e8826d4e2bb02061d9ef056e83"),
         ("D", "Luminal spaces", "GT includes internal luminal spaces.", "SICAPv2:0.jpg")]


def load(p, size=520, mask=False):
    im = Image.open(p)
    if mask:
        a = np.asarray(im.convert("L")) > 0
        return np.asarray(Image.fromarray((a * 255).astype(np.uint8)).resize((size, size), Image.NEAREST))
    return np.asarray(im.convert("RGB").resize((size, size), Image.LANCZOS))


man = pd.read_csv(REVIEW / "bad_case_manifest_live.csv").set_index("image_id")
sic = pd.read_csv(SICAP / "tables" / "per_external_image_metrics.csv").set_index("image_name")


def case_panels(cid):
    """Return (four images, log record of the case)."""
    if cid.startswith("SICAPv2:"):
        name = cid.split(":", 1)[1]
        r = sic.loc[name]
        key = hashlib.sha1(name.encode("utf-8")).hexdigest()[:20]
        files = {"he": Path(r.image_path), "gt": Path(r.mask_path),
                 "model": SICAP / "pred_masks" / f"{key}.png", "error": SICAP / "error_maps" / f"{key}.png"}
        rec = {"dataset": "SICAPv2", "image_name": name, "source_wsi": str(r.source_wsi), "stored_dice": round(float(r.dice), 4),
               "index": "all-images fallback index of the external evaluation",
               "metrics_table": source(SICAP / "tables" / "per_external_image_metrics.csv", WORK)}
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


def draw_case(fig, box, letter, title, finding, cid, img_frac=0.60, img_bottom=0.15):
    x0, y0, w, h = box
    fig.add_artist(Rectangle((x0, y0), w, h, transform=fig.transFigure, fill=False, ec="#333", lw=0.8))
    fig.text(x0 + 0.008, y0 + h - 0.012, f"{letter}. {title}", fontsize=8.2, fontweight="bold", va="top", color=NAVY)
    imgs, rec = case_panels(cid)
    pw = (w - 0.03) / 4
    ph = h * img_frac
    for i, (img, lab) in enumerate(zip(imgs, ("H&E", "GT mask", "Model", "Error map"))):
        ax = fig.add_axes([x0 + 0.015 + i * pw + 0.004, y0 + h * img_bottom, pw - 0.008, ph])
        ax.imshow(img, cmap="gray" if img.ndim == 2 else None, vmin=0, vmax=255, interpolation="nearest" if img.ndim == 2 else "antialiased")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_linewidth(0.5); s.set_edgecolor("#888")
        ax.set_title(lab, fontsize=6.6, pad=2)
    fig.text(x0 + w / 2, y0 + h * (img_bottom - 0.09), finding, ha="center", fontsize=7.2, fontweight="bold", color="#222")
    rec.update({"category": title, "pathologist_finding": finding})
    return {letter: rec}


def legend(fig, y):
    x = 0.17
    for col, lab in LEGEND:
        fig.add_artist(Rectangle((x, y), 0.014, 0.012, transform=fig.transFigure, fc=np.array(col) / 255, ec="#666", lw=0.4))
        fig.text(x + 0.02, y + 0.006, lab, fontsize=6.4, va="center")
        x += 0.17
    fig.text(0.03, y + 0.006, "Error map:", fontsize=6.4, va="center", fontweight="bold")


def save(fig, png, tif, figure, panels):
    save_figure(fig, OUT, png, tif, {
        "figure": figure, "panels": panels,
        "parameters": {"panel_size_px": 520, "masks": "binary (> 0), nearest-neighbour resize",
                       "error_map": "saved evaluation error map, colours as in the legend",
                       "labels": "case categories and findings from the pathologist review, unchanged"}}, WORK, __file__)


# Fig 14: three stacked cases
fig = plt.figure(figsize=(7.5, 6.6))
fig.text(0.5, 0.975, "Pathologist-identified cases interpreted as likely model-related errors", ha="center", va="top",
         fontsize=10, fontweight="bold", color=NAVY)
src = {}
for k, case in enumerate(FIG14):
    src.update(draw_case(fig, (0.02, 0.652 - k * 0.305, 0.96, 0.285), *case))
legend(fig, 0.014)
save(fig, "Fig14_model_related_errors.png", "Fig14.tif", "Fig 14", src)

# Fig 15: 2 x 2 cases
fig = plt.figure(figsize=(7.5, 4.15))
fig.text(0.5, 0.98, "Pathologist-identified cases interpreted as likely annotation-related discrepancies", ha="center",
         va="top", fontsize=10, fontweight="bold", color=NAVY)
src = {}
boxes = [(0.02, 0.495, 0.475, 0.42), (0.505, 0.495, 0.475, 0.42), (0.02, 0.065, 0.475, 0.42), (0.505, 0.065, 0.475, 0.42)]
for box, case in zip(boxes, FIG15):
    src.update(draw_case(fig, box, *case, img_frac=0.58, img_bottom=0.22))
legend(fig, 0.018)
save(fig, "Fig15_annotation_discrepancies.png", "Fig15.tif", "Fig 15", src)
