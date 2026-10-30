"""Fig 3 of the PANDA manuscript: nested L0/L1/L2 views of one WSI.

The three columns are stored PANDA training patches of the same WSI whose
level-0 fields are nested (L0 inside L1 inside L2). Rows show the patch, its
stored reference tumor mask and the overlay. A box in the L1 and L2 panels
marks the field of the next finer level. The L2 patch lies on the slide edge;
image, mask and overlay are cropped to the same in-slide columns. The slide,
the patch coordinates and the files used are written to the JSON log, not to
the figure (provenance_log.py). Drawn with SciencePlots and the PLOS font
settings (provenance_log.plos_style).

Usage:
    python make_figure_3.py <output_dir> [work_root]
"""
import os
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from PIL import Image

from provenance_log import patch_record, plos_style, save_figure

OUT = Path(sys.argv[1])
if len(sys.argv) < 3 and "PANDA_DATA_ROOT" not in os.environ:
    sys.exit("give work_root as the second argument or set PANDA_DATA_ROOT")
WORK = Path(sys.argv[2] if len(sys.argv) > 2 else os.environ["PANDA_DATA_ROOT"])
SLIDE = "0018ae58b01bdadc8e347995b69f99aa"
DS = {0: 1, 1: 4, 2: 16}
FIELD = {l: 512 * DS[l] for l in DS}          # level-0 extent of a 512-px patch
BOX = "#ffd400"
plos_style()


def rgb(path):
    return np.asarray(Image.open(path).convert("RGB"))


def binary(path):
    return np.asarray(Image.open(path).convert("L")) > 127


def seam_pixels(mask_path):
    t = binary(mask_path).astype(np.uint8)
    return int(((cv2.morphologyEx(t, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)) > 0) & (t == 0)).sum())


def inside(a, b):
    """Level-0 field of patch a lies inside the field of patch b."""
    return (b.x0 <= a.x0 and a.x0 + FIELD[a.level] <= b.x0 + FIELD[b.level]
            and b.y0 <= a.y0 and a.y0 + FIELD[a.level] <= b.y0 + FIELD[b.level])


rows = pd.concat([c[c.image_id == SLIDE] for c in pd.read_csv(WORK / "manifests" / "patches.csv", chunksize=200_000)])
assert (rows.split == "train").all()
provider = rows.provider.iloc[0]
tum = rows[rows.category == "tumor"].copy()
tum["seam"] = [seam_pixels(p) for p in tum["mask"]]

def margin(a, b):
    """Smallest gap, in pixels of b's level, between a's field and the edge of b's in-slide area."""
    d = DS[b.level]
    gaps = [a.x0 - b.x0, a.y0 - b.y0, b.x0 + FIELD[b.level] - a.x0 - FIELD[a.level],
            b.y0 + FIELD[b.level] - a.y0 - FIELD[a.level]]
    if b.x0 == 0:  # L2 patch on the slide edge: right side is cropped, use vertical gaps only
        gaps = gaps[1::2]
    return min(gaps) / d


# nested tumor triples; prefer boxes away from the panel edges, then masks with fewer seams
cand = [(a, b, c) for _, a in tum[tum.level == 0].iterrows() for _, b in tum[tum.level == 1].iterrows()
        if inside(a, b) for _, c in tum[tum.level == 2].iterrows() if inside(b, c)]
assert cand, "no nested L0/L1/L2 tumor patches"
p0, p1, p2 = min(cand, key=lambda t: (-min(margin(t[0], t[1]), margin(t[1], t[2])),
                                      t[0].seam + t[1].seam + t[2].seam))
pick = {0: p0, 1: p1, 2: p2}

fig_cols = []
for l in (0, 1, 2):
    img, msk = rgb(pick[l].img), binary(pick[l]["mask"])
    ov = img.astype(float)
    ov[msk] = 0.55 * ov[msk] + 0.45 * np.array([230, 30, 50])
    cols = np.where(img.max(axis=(0, 2)) > 0)[0]  # drop columns outside the slide (stored as black)
    sl = slice(0, cols.max() + 1)
    fig_cols.append((img[:, sl], msk[:, sl], ov.astype(np.uint8)[:, sl]))

widths = [c[0].shape[1] for c in fig_cols]
fig = plt.figure(figsize=(7.5, 6.6), layout="none")
gs = fig.add_gridspec(3, 4, width_ratios=widths + [70], wspace=0.06, hspace=0.06,
                      left=0.02, right=0.99, top=0.95, bottom=0.01)
titles = {0: "L0 (1×)", 1: "L1 (4× downsample)", 2: "L2 (16× downsample)"}
for j, l in enumerate((0, 1, 2)):
    for i, im in enumerate(fig_cols[j]):
        ax = fig.add_subplot(gs[i, j])
        ax.imshow(im, cmap="gray" if i == 1 else None, vmin=0, vmax=1 if i == 1 else None, interpolation="nearest" if i == 1 else "antialiased")
        ax.set_xticks([]); ax.set_yticks([]); ax.minorticks_off()
        for s in ax.spines.values():
            s.set_linewidth(0.8)
        if i == 0:
            ax.text(0.5, 1.02, titles[l], transform=ax.transAxes, ha="center", va="bottom", fontweight="bold")
        if l > 0:  # field of the next finer level
            f = pick[l - 1]
            ax.add_patch(Rectangle(((f.x0 - pick[l].x0) / DS[l] - 0.5, (f.y0 - pick[l].y0) / DS[l] - 0.5),
                                   FIELD[l - 1] / DS[l], FIELD[l - 1] / DS[l], fill=False, ec=BOX, lw=1.4))
for i, lab in enumerate(("Image", "Reference mask", "Overlay")):
    ax = fig.add_subplot(gs[i, 3]); ax.axis("off")
    ax.text(0.35, 0.5, lab, rotation=-90, fontweight="bold", ha="center", va="center", transform=ax.transAxes)

boxes = {f"L{l}": {"box_px_in_panel": [round((pick[l - 1].x0 - pick[l].x0) / DS[l], 2), round((pick[l - 1].y0 - pick[l].y0) / DS[l], 2),
                                       FIELD[l - 1] / DS[l], FIELD[l - 1] / DS[l]], "marks_field_of": f"L{l - 1}"} for l in (1, 2)}
save_figure(fig, OUT, "Fig03_multiresolution.png", "Fig3.tif", {
    "figure": "Fig 3",
    "slide": {"slide_id": SLIDE, "provider": provider, "split": "train"},
    "panels": {f"column_L{l}": patch_record(pick[l], WORK, shown="image, reference tumor mask, overlay",
                                            in_slide_columns=int(fig_cols[j][0].shape[1]), **boxes.get(f"L{l}", {}))
               for j, l in enumerate((0, 1, 2))},
    "parameters": {"patch_choice": "nested L0 in L1 in L2 tumor patches; boxes farthest from the panel edges, then fewest mask seams",
                   "overlay": "0.55 x image + 0.45 x (230, 30, 50) on tumor pixels"}}, WORK, __file__)
print("patches:", [(l, int(pick[l].x0), int(pick[l].y0)) for l in (0, 1, 2)])
