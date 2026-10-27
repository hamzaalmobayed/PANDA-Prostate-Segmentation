"""Regenerate Figs 14 and 15 from the saved review outputs.

PANDA panels use the detail images saved during the test run for each
reviewed WSI (paper_results/test_visuals_bad_review/bad_<id>/*_detail_*.png).
Fig 15D uses the SICAPv2 image, reference mask, prediction and error map
written by the external evaluation. Case labels and interpretations are the
pathologist's, unchanged from the earlier figure version.

Usage:
    python make_figures_14_15.py <output_dir> [work_root]

work_root holds paper_results/ and external_sicapv2/results/. If omitted,
PANDA_DATA_ROOT is used.
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

OUT = Path(sys.argv[1])
if len(sys.argv) < 3 and "PANDA_DATA_ROOT" not in os.environ:
    sys.exit("give work_root as the second argument or set PANDA_DATA_ROOT")
WORK = Path(sys.argv[2] if len(sys.argv) > 2 else os.environ["PANDA_DATA_ROOT"])
REVIEW = WORK / "paper_results" / "test_visuals_bad_review"
SICAP = WORK / "external_sicapv2" / "results"
NAVY = "#0c1644"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7})
LEGEND = [((95, 205, 95), "overlap"), ((242, 214, 63), "false negative"), ((234, 56, 41), "false positive"),
          ((235, 235, 235), "true-negative tissue")]

FIG14 = [("A", "Benign mimic", "Benign glands mistaken as tumor.", "15bf6b34c8ce10b0eddf687e3a670e4d"),
         ("B", "Hypercellular benign mimic", "Dense benign tissue resembles carcinoma.", "13813facc985c2ea007fdf764ef98a5a"),
         ("C", "Crowded cells", "Crowded cells interpreted as tumor.", "b49d54b4f7e73808440c36fa240614c0")]
FIG15 = [("A", "Broader GT than prediction", "GT includes additional benign tissue.", "9848ee7cc45f7cd373b155191085f06a"),
         ("B", "Possible stain-associated discrepancy", "Annotation may extend into non-tumorous tissue.", "cf1c7e7e632db44c5ca6de473da247e6"),
         ("C", "Connected-foci convention", "GT joins separate tumor foci.", "cfbe34e8826d4e2bb02061d9ef056e83"),
         ("D", "Luminal spaces", "GT includes internal luminal spaces.", "SICAPv2:0.jpg")]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load(p, size=520, mask=False):
    im = Image.open(p)
    if mask:
        a = np.asarray(im.convert("L")) > 0
        return np.asarray(Image.fromarray((a * 255).astype(np.uint8)).resize((size, size), Image.NEAREST))
    return np.asarray(im.convert("RGB").resize((size, size), Image.LANCZOS))


man = pd.read_csv(REVIEW / "bad_case_manifest_live.csv").set_index("image_id")
sic = pd.read_csv(SICAP / "tables" / "per_external_image_metrics.csv").set_index("image_name")


def case_panels(cid):
    """Return (four images, id line, source files)."""
    if cid.startswith("SICAPv2:"):
        name = cid.split(":", 1)[1]
        r = sic.loc[name]
        key = hashlib.sha1(name.encode("utf-8")).hexdigest()[:20]
        files = {"he": Path(r.image_path), "gt": Path(r.mask_path),
                 "model": SICAP / "pred_masks" / f"{key}.png", "error": SICAP / "error_maps" / f"{key}.png"}
        line = f"SICAPv2 image {name} (executed fallback index) · stored Dice {r.dice:.4f}"
    else:
        d = REVIEW / f"bad_{cid}"
        pick = lambda tag: Path(glob.glob(str(d / f"*_detail_{tag}.png"))[0])
        files = {"he": pick("original"), "gt": pick("gt_mask"), "model": pick("model_mask"), "error": pick("error_map")}
        m = man.loc[cid]
        line = f"PANDA WSI {cid} · {m.provider.capitalize()} · stored Dice {m.fusion_dice:.4f}"
    imgs = [load(files["he"]), load(files["gt"], mask=True), load(files["model"], mask=True), load(files["error"])]
    return imgs, line, files


def draw_case(fig, box, letter, title, finding, cid, img_frac=0.60, id_fs=5.4):
    x0, y0, w, h = box
    fig.add_artist(Rectangle((x0, y0), w, h, transform=fig.transFigure, fill=False, ec="#333", lw=0.8))
    fig.text(x0 + 0.008, y0 + h - 0.012, f"{letter}. {title}", fontsize=8.2, fontweight="bold", va="top", color=NAVY)
    imgs, line, files = case_panels(cid)
    pw = (w - 0.03) / 4
    ph = h * img_frac
    for i, (img, lab) in enumerate(zip(imgs, ("H&E", "GT mask", "Model", "Error map"))):
        ax = fig.add_axes([x0 + 0.015 + i * pw + 0.004, y0 + h * 0.20, pw - 0.008, ph])
        ax.imshow(img, cmap="gray" if img.ndim == 2 else None, vmin=0, vmax=255, interpolation="nearest" if img.ndim == 2 else "antialiased")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_linewidth(0.5); s.set_edgecolor("#888")
        ax.set_title(lab, fontsize=6.6, pad=2)
    fig.text(x0 + w / 2, y0 + h * 0.115, finding, ha="center", fontsize=7.2, fontweight="bold", color="#222")
    fig.text(x0 + w / 2, y0 + h * 0.045, line, ha="center", fontsize=id_fs, color="#444")
    return {f"{letter}_{k}": v for k, v in files.items()}


def legend(fig, y):
    x = 0.17
    for col, lab in LEGEND:
        fig.add_artist(Rectangle((x, y), 0.014, 0.012, transform=fig.transFigure, fc=np.array(col) / 255, ec="#666", lw=0.4))
        fig.text(x + 0.02, y + 0.006, lab, fontsize=6.4, va="center")
        x += 0.17
    fig.text(0.03, y + 0.006, "Error map:", fontsize=6.4, va="center", fontweight="bold")


def save(fig, png, tif, sources):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / png, dpi=300, facecolor="white")
    plt.close(fig)
    im = Image.open(OUT / png).convert("RGB")
    im.save(OUT / tif, compression="tiff_lzw", dpi=(300, 300))
    def rel(p):
        # SICAPv2 source images may live outside the work root (dataset download cache)
        p = Path(p).resolve()
        try:
            return p.relative_to(WORK.resolve()).as_posix()
        except ValueError:
            return "<SICAPv2 dataset>/" + "/".join(p.parts[-2:])
    rec = {"script": Path(__file__).name, "paths_relative_to": "PANDA_DATA_ROOT",
           "sources": {k: {"path": rel(v), "sha256": sha(v)} for k, v in sources.items()}}
    (OUT / (Path(png).stem + "_sources.json")).write_text(json.dumps(rec, indent=2), encoding="utf-8")
    print("saved", png, im.size)


# Fig 14: three stacked cases
fig = plt.figure(figsize=(7.5, 6.6))
fig.text(0.5, 0.975, "Pathologist-identified cases interpreted as likely model-related errors", ha="center", va="top",
         fontsize=10, fontweight="bold", color=NAVY)
src = {}
for k, case in enumerate(FIG14):
    src.update(draw_case(fig, (0.02, 0.652 - k * 0.305, 0.96, 0.285), *case))
legend(fig, 0.014)
save(fig, "Fig14_model_related_errors.png", "Fig14.tif", src)

# Fig 15: 2 x 2 cases
fig = plt.figure(figsize=(7.5, 4.15))
fig.text(0.5, 0.98, "Pathologist-identified cases interpreted as likely annotation-related discrepancies", ha="center",
         va="top", fontsize=10, fontweight="bold", color=NAVY)
src = {}
boxes = [(0.02, 0.495, 0.475, 0.42), (0.505, 0.495, 0.475, 0.42), (0.02, 0.065, 0.475, 0.42), (0.505, 0.065, 0.475, 0.42)]
for box, case in zip(boxes, FIG15):
    src.update(draw_case(fig, box, *case, img_frac=0.58, id_fs=4.9))
legend(fig, 0.018)
save(fig, "Fig15_annotation_discrepancies.png", "Fig15.tif", src)
