"""Figures of S1 Appendix (revision of 30 Oct 2026): training augmentations and patch-level confusion matrices.

S1_augmentation.png (Fig 5 before the revision): the stored L0 tumor patch of
training WSI 0018ae58b01bdadc8e347995b69f99aa at level-0 x0 = 1024, y0 = 9344
(found by matching the earlier "Original" panel against all 109,060 stored L0
patches, correlation 0.977, next patch 0.20), shown as the model receives it in
training: the Vahadane-normalized copy from the pipeline's persistent cache.
Each panel applies one transform of PairAugStainHeavy
(panda_camelyon_style_pipeline.py) with the same operation and its strongest
configured setting, so that the effect is visible:
  horizontal flip, 90-degree rotation (PIL rotate, as in the pipeline),
  ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.05) at
  the upper bound of each factor, gamma 1.5 (range 0.7-1.5), Gaussian blur
  kernel 3 with sigma 1.2 (range 0.1-1.2), Gaussian noise sigma 12 (range
  3-12, seed 42), JPEG quality 35 (range 35-89).
S1_confusion.png (Fig 10 before the revision): patch-level confusion matrices
from paper_results/tables/patch_metrics_per_level_and_pooled.csv (the earlier
script typed the counts in; they are identical to this export).

Drawn with SciencePlots and the PLOS font settings (provenance_log.plos_style).
Settings and sources are written to the JSON log of each figure
(provenance_log.py), not the figure.

Usage:
    python make_S1_figures.py <output_dir> [work_root]
"""
import io
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torchvision
import torchvision.transforms.functional as TF
from matplotlib.colors import ListedColormap
from PIL import Image

from provenance_log import patch_record, plos_style, save_figure, source, stain_cache_file

OUT = Path(sys.argv[1])
if len(sys.argv) < 3 and "PANDA_DATA_ROOT" not in os.environ:
    sys.exit("give work_root as the second argument or set PANDA_DATA_ROOT")
WORK = Path(sys.argv[2] if len(sys.argv) > 2 else os.environ["PANDA_DATA_ROOT"])
SLIDE, LEVEL, X0, Y0 = "0018ae58b01bdadc8e347995b69f99aa", 0, 1024, 9344
SEED = 42
plos_style()


# ---------------------------------------------------------------- augmentations
def augmentation():
    sel = pd.read_csv(WORK / "selected" / "train_selected.csv")
    row = sel[(sel.image_id == SLIDE) & (sel.level == LEVEL) & (sel.x0 == X0) & (sel.y0 == Y0)]
    assert len(row) == 1, "source patch not in the selected training set"
    row = row.iloc[0]
    cache = stain_cache_file(WORK, row.img)
    assert cache is not None, "training-time Vahadane copy not found"
    img = Image.open(cache).convert("RGB")

    rng = np.random.default_rng(SEED)
    settings = {"horizontal_flip": "TF.hflip", "rotation": "PIL Image.rotate(90)",
                "color_jitter": {"brightness": 1.25, "contrast": 1.25, "saturation": 1.25, "hue": 0.05,
                                 "order": "brightness, contrast, saturation, hue"},
                "gamma": 1.5, "gaussian_blur": {"kernel_size": 3, "sigma": 1.2},
                "gaussian_noise": {"sigma": 12.0, "numpy_seed": SEED}, "jpeg_quality": 35}
    cj = TF.adjust_hue(TF.adjust_saturation(TF.adjust_contrast(TF.adjust_brightness(img, 1.25), 1.25), 1.25), 0.05)
    a = np.asarray(img).astype(np.float32)
    noise = Image.fromarray(np.clip(a + rng.normal(0, 12.0, a.shape), 0, 255).astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=35)
    buf.seek(0)
    jpeg = Image.open(buf).convert("RGB")
    panels = [(img, "Normalized patch"), (TF.hflip(img), "Horizontal flip"), (img.rotate(90), "Rotation (90°)"),
              (cj, "Color jitter"), (TF.adjust_gamma(img, 1.5), "Gamma"),
              (TF.gaussian_blur(img, kernel_size=3, sigma=1.2), "Gaussian blur"), (noise, "Gaussian noise"),
              (jpeg, "JPEG compression")]

    fig, axes = plt.subplots(2, 4, figsize=(7.5, 4.3))
    for ax, (pic, lab), let in zip(axes.flat, panels, "ABCDEFGH"):
        ax.imshow(pic)
        ax.set_xticks([]); ax.set_yticks([]); ax.minorticks_off()
        ax.set_xlabel(lab)
        ax.text(-0.03, 1.0, let, transform=ax.transAxes, fontsize=10, fontweight="bold", ha="right", va="top")
    save_figure(fig, OUT, "S1_augmentation.png", "S1_augmentation.tif", {
        "figure": "S1 Appendix, augmentation figure",
        "panels": {"all": patch_record(row, WORK, shown="Vahadane-normalized training input and one transform per panel",
                                       normalized_image=source(cache, WORK))},
        "parameters": {"transforms": settings, "pipeline_class": "PairAugStainHeavy in panda_camelyon_style_pipeline.py",
                       "configured_ranges": {"color_jitter": "brightness/contrast/saturation 0.25, hue 0.05",
                                             "gamma": [0.7, 1.5], "blur_sigma": [0.1, 1.2], "noise_sigma": [3, 12],
                                             "jpeg_quality": [35, 89]},
                       "panel_values": "upper bound of each configured range",
                       "torchvision": torchvision.__version__,
                       "identification": "earlier Fig 5 'Original' panel matched this stored patch (correlation 0.977; next 0.20)"}},
        WORK, __file__)


# ---------------------------------------------------------------- confusion matrices
def confusion():
    csv = WORK / "paper_results" / "tables" / "patch_metrics_per_level_and_pooled.csv"
    t = pd.read_csv(csv).set_index("subset")
    specs = [("A", "all_levels_pooled", 1), ("B", "L0", 1), ("C", "L1", 1), ("D", "L2", 2)]
    pooled = t.loc[["L0", "L1", "L2"], ["tp", "fp", "tn", "fn"]].sum()
    assert (pooled == t.loc["all_levels_pooled", ["tp", "fp", "tn", "fn"]]).all()
    cmap = ListedColormap(["#174e82", "#e6f1fa", "#e8f2fb", "#c1dcf6"])
    fig, axes = plt.subplots(2, 2, figsize=(7.5, 6.2))
    shown = {}
    for ax, (panel, key, dec) in zip(axes.flat, specs):
        r = t.loc[key]
        counts = np.array([[r.tn, r.fp], [r.fn, r.tp]], dtype=np.int64)
        ax.imshow(np.array([[0, 1], [2, 3]]), cmap=cmap, vmin=0, vmax=3, interpolation="nearest")
        total = int(counts.sum())
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{int(counts[i, j]):,}\n{100 * counts[i, j] / total:.{dec}f}%", ha="center", va="center",
                        color="white" if (i, j) == (0, 0) else "black", linespacing=1.35)
        ax.set_xticks([0, 1], ["Normal", "Tumor"]); ax.set_yticks([0, 1], ["Normal", "Tumor"])
        ax.minorticks_off(); ax.tick_params(length=0)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Reference")
        ax.text(-0.22, 1.0, panel, transform=ax.transAxes, fontsize=10, fontweight="bold", ha="right", va="top")
        shown[key] = {"tn": int(r.tn), "fp": int(r.fp), "fn": int(r.fn), "tp": int(r.tp)}
    save_figure(fig, OUT, "S1_confusion.png", "S1_confusion.tif", {
        "figure": "S1 Appendix, confusion-matrix figure",
        "sources": {"patch_metrics": source(csv, WORK)},
        "values_shown": shown,
        "parameters": {"panels": "A pooled across levels, B L0, C L1, D L2",
                       "cell_percent": "share of the panel's four counts; 2 decimals for L2, 1 elsewhere",
                       "pooled": "sum of the L0, L1 and L2 counts (checked)"}}, WORK, __file__)


augmentation()
confusion()
