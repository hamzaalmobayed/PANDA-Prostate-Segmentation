"""Figs 1, 2 and 4 of the PANDA manuscript (numbering of the revision of 30 Oct 2026).

Every image panel is read from the stored PANDA training patches, their
reference masks, the training-derived Vahadane target, or the final model
checkpoint. Nothing is drawn by hand or by an image generator. Each figure is
drawn with SciencePlots and the PLOS font settings (provenance_log.plos_style)
and written as PNG and as PLOS TIFF (RGB, LZW, 300 dpi, 7.5 in wide). Case
identifiers, patch coordinates and file names are kept out of the figures;
they are written, with the SHA-256 of every source file, to the JSON log of
each figure (provenance_log.py).

Fig 1  workflow with patches of one training WSI.
Fig 2  patch preparation on that WSI. Panels F-G of the earlier figure
       (candidate-grid and training-bucket counts) are now a table of the
       manuscript (fill_manuscript_numbers.py, TABLE_PREP).
Fig 4  network diagram (Fig 7 before the revision).
The earlier Fig 4 (Vahadane normalization) was removed; its patch is still
Fig 2E, and the target is shown in S1 Appendix.

The normalized L0 patch of Fig 2E is the copy the training pipeline wrote to
its persistent Vahadane cache (cache/stain_v10/vahadane/), i.e. the image the
model was trained on. The Fig 4 input patch is not in that cache; it is
normalized here with the same target and a fixed NumPy seed, because the
TIAToolbox dictionary learning draws from NumPy's global random state.

Usage:
    python make_figures_1_2_4.py <output_dir> [work_root]

work_root is the pipeline work directory (patches/, selected/, manifests/,
runs/, cache/, stain_target_train.png). If omitted, PANDA_DATA_ROOT is used.
"""
import json
import os
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
from PIL import Image
from tiatoolbox.tools.stainnorm import VahadaneNormalizer

from provenance_log import patch_record, plos_style, prov_dir, save_figure, source, stain_cache_file

OUT = Path(sys.argv[1])
if len(sys.argv) < 3 and "PANDA_DATA_ROOT" not in os.environ:
    sys.exit("give work_root as the second argument or set PANDA_DATA_ROOT")
WORK = Path(sys.argv[2] if len(sys.argv) > 2 else os.environ["PANDA_DATA_ROOT"])
SLIDE = "362d30ace92ce17128d176137362b8c2"
LEVELS = (0, 1, 2)
DS = {0: 1, 1: 4, 2: 16}
NAVY, ARROW = "#0c1644", "#186bd1"
GREEN, ORANGE = "#10a64a", "#f07f0c"
VAHADANE_SEED = 42
plos_style()


# ---------------------------------------------------------------- helpers
def rgb(path):
    return np.asarray(Image.open(path).convert("RGB"))


def binary(path):
    return np.asarray(Image.open(path).convert("L")) > 127


def save(fig, png_name, tif_name, record, extra_files=()):
    return save_figure(fig, OUT, png_name, tif_name, record, WORK, __file__, extra_files)


def prec(row, **extra):
    return patch_record(row, WORK, **extra)


def tissue_mask(img, thr=0.20, disk=2, min_obj=500):
    """Same rule as generate_tissue_mask() in panda_camelyon_style_pipeline.py."""
    sat = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)[..., 1]
    otsu, _ = cv2.threshold(sat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = ((sat > max(int(otsu), round(thr * 255))) * 255).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * disk + 1, 2 * disk + 1))
    m = cv2.morphologyEx(cv2.morphologyEx(m, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k)
    n, lab, st, _ = cv2.connectedComponentsWithStats((m > 0).astype(np.uint8), 8)
    keep = np.zeros(m.shape, bool)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] >= min_obj:
            keep |= lab == i
    return keep


class Vahadane:
    """Same wrapper as VahadaneProcessor in the pipeline (writable C copies)."""
    def __init__(self, target_path):
        self.norm = VahadaneNormalizer()
        np.random.seed(VAHADANE_SEED)
        self.norm.fit(np.array(rgb(target_path), dtype=np.uint8, copy=True, order="C"))

    def __call__(self, img):
        np.random.seed(VAHADANE_SEED)
        return np.array(self.norm.transform(np.array(img, dtype=np.uint8, copy=True, order="C")), dtype=np.uint8)


class DoubleConv(nn.Module):
    def __init__(self, i, o, p):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                                 nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                                 nn.Dropout2d(p))

    def forward(self, x):
        return self.net(x)


class UNetEfficientNetB4(nn.Module):
    """Copy of the training architecture (panda_camelyon_style_pipeline.py)."""
    def __init__(self, dropout_p=0.1, coord_embed_dim=32):
        super().__init__()
        self.encoder = timm.create_model("efficientnet_b4", pretrained=False, features_only=True, out_indices=(0, 1, 2, 3, 4))
        c0, c1, c2, c3, c4 = self.encoder.feature_info.channels()
        self.channels = (c0, c1, c2, c3, c4)
        self.coord_fc = nn.Sequential(nn.Linear(2, 64), nn.ReLU(inplace=True), nn.Linear(64, coord_embed_dim), nn.ReLU(inplace=True))
        self.dec3 = DoubleConv(c4 + coord_embed_dim + c3, 256, dropout_p)
        self.dec2 = DoubleConv(256 + c2, 128, dropout_p)
        self.dec1 = DoubleConv(128 + c1, 64, dropout_p)
        self.dec0 = DoubleConv(64 + c0, 32, dropout_p)
        self.final_conv = nn.Conv2d(32, 1, 1)

    def forward(self, x, coords):
        x0, x1, x2, x3, x4 = self.encoder(x)
        ce = self.coord_fc(coords)[..., None, None].expand(-1, -1, x4.shape[2], x4.shape[3])
        b = torch.cat([x4, ce], 1)
        d = self.dec3(torch.cat([F.interpolate(b, size=x3.shape[2:], mode="bilinear", align_corners=False), x3], 1))
        d = self.dec2(torch.cat([F.interpolate(d, size=x2.shape[2:], mode="bilinear", align_corners=False), x2], 1))
        d = self.dec1(torch.cat([F.interpolate(d, size=x1.shape[2:], mode="bilinear", align_corners=False), x1], 1))
        d = self.dec0(torch.cat([F.interpolate(d, size=x0.shape[2:], mode="bilinear", align_corners=False), x0], 1))
        return F.interpolate(self.final_conv(d), size=x.shape[2:], mode="bilinear", align_corners=False)


def canvas(h):
    """Figure of 7.5 x h inches with one invisible axes in inch coordinates (no layout engine)."""
    fig = plt.figure(figsize=(7.5, h), layout="none")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 7.5); ax.set_ylim(0, h); ax.axis("off")
    return fig, ax


def image_axes(fig, h, x, y, w, hh, img, edge=None, lw=0.6, gray=False):
    """Image in its own axes at (x, y, w, hh) inches; returns the axes."""
    a = fig.add_axes([x / 7.5, y / h, w / 7.5, hh / h])
    a.imshow(img, cmap="gray" if gray else None, vmin=0, vmax=1 if gray else None,
             interpolation="nearest" if gray else "antialiased", aspect="auto")
    a.set_xticks([]); a.set_yticks([]); a.minorticks_off()
    for s in a.spines.values():
        s.set_edgecolor(edge or "#9aa7b8"); s.set_linewidth(lw)
    return a


def arrow(ax, a, b, color=ARROW, lw=1.1, ls="-", scale=8):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=scale, color=color, lw=lw, linestyle=ls,
                                 shrinkA=0, shrinkB=0))


def rbox(ax, x, y, w, h, fc, ec, lw=1.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.0,rounding_size=0.06", fc=fc, ec=ec, lw=lw))


# ---------------------------------------------------------------- data
sel = pd.read_csv(WORK / "selected" / "train_selected.csv")
rows = sel[sel.image_id == SLIDE].copy()
assert len(rows), "slide not in selected manifest"
assert (rows.split == "train").all()
provider = rows.provider.iloc[0]
# every stored (extracted) patch of the slide, selected or not
all_rows = pd.concat([c[c.image_id == SLIDE] for c in pd.read_csv(WORK / "manifests" / "patches.csv", chunksize=200_000)])
assert (all_rows.split == "train").all() and len(all_rows) >= len(rows)


def seam_pixels(mask_path):
    """1-pixel gaps inside a stored tumor mask (closed by a 3x3 closing)."""
    t = binary(mask_path).astype(np.uint8)
    return int(((cv2.morphologyEx(t, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)) > 0) & (t == 0)).sum())


def pick(level, category, rank=0, src=None):
    src = rows if src is None else src
    r = src[(src.level == level) & (src.category == category)].copy()
    assert len(r) > rank, (level, category)
    # prefer masks without rasterization seams, then the most tissue
    r["seam"] = [seam_pixels(p) if category == "tumor" else 0 for p in r["mask"]]
    r = r.sort_values(["seam", "tissue_fraction", "tumor_fraction"], ascending=[True, False, False])
    return r.iloc[rank]


target_path = WORK / "stain_target_train.png"
vahadane = Vahadane(target_path)

# L2 mosaic of every stored L2 patch of the slide (image, tumor mask, valid mask)
l2_imgs = sorted((WORK / "patches" / "train" / SLIDE / "L2").glob("*/*.jpg"))
coords = []
for f in l2_imgs:
    parts = f.stem.split("__")
    coords.append((int([p for p in parts if p.startswith("x0-")][0][3:]), int([p for p in parts if p.startswith("y0-")][0][3:])))
W = (max(c[0] for c in coords) + 512 * 16) // 16
H = (max(c[1] for c in coords) + 512 * 16) // 16
mos = np.full((H, W, 3), 255, np.uint8)
mos_tum = np.zeros((H, W), bool)
mos_val = np.zeros((H, W), bool)
mos_cov = np.zeros((H, W), bool)
for f, (x, y) in zip(l2_imgs, coords):
    cls = f.parent.name                      # Normal | Tumor
    stem = f.stem
    mask_f = f.parent.parent / f"{cls}_Masks" / f"{stem}__tumor_mask.png"
    val_f = f.parent.parent / f"{cls}_Valid_Masks" / f"{stem}__valid_mask.png"
    ys, xs = slice(y // 16, y // 16 + 512), slice(x // 16, x // 16 + 512)
    mos[ys, xs] = rgb(f)
    mos_tum[ys, xs] |= binary(mask_f)
    mos_val[ys, xs] |= binary(val_f)
    mos_cov[ys, xs] = True
# crop to the covered tissue with a margin
yy, xx = np.where(mos_cov & tissue_mask(mos))
pad = 40
y0, y1 = max(0, yy.min() - pad), min(H, yy.max() + pad)
x0, x1 = max(0, xx.min() - pad), min(W, xx.max() + pad)
mos, mos_tum, mos_val, mos_cov = (a[y0:y1, x0:x1] for a in (mos, mos_tum, mos_val, mos_cov))
tis = tissue_mask(mos) & mos_cov

# Fig 2E patch: training-time Vahadane copy of the selected L0 tumor patch
src_e = pick(0, "tumor")
raw_e = rgb(src_e.img)
cache_e = stain_cache_file(WORK, src_e.img)
assert cache_e is not None, "training-time Vahadane copy of the Fig 2E patch not found"
norm_e = rgb(cache_e)
fresh_diff_e = float(np.abs(vahadane(raw_e).astype(int) - norm_e.astype(int)).mean())  # check only

# ---------------------------------------------------------------- Fig 4
src4 = pick(1, "tumor", src=all_rows)
x4 = vahadane(rgb(src4.img))
gt4 = binary(src4["mask"])
valid4 = binary(src4["valid_mask"])
PROV = prov_dir(OUT)
Image.fromarray(x4).save(PROV / "Fig04_input_normalized.png")
model = UNetEfficientNetB4()
ck = torch.load(WORK / "runs" / "final_main_training" / "best.pt", map_location="cpu", weights_only=False)
model.load_state_dict(ck["model"])
model.eval()
with torch.no_grad():
    t = torch.from_numpy(x4.copy()).permute(2, 0, 1).float()[None] / 255.0
    c = torch.tensor([[float(src4.coord_x), float(src4.coord_y)]], dtype=torch.float32)
    prob4 = torch.sigmoid(model(t, c))[0, 0].numpy()
ch = model.channels  # (24, 32, 56, 160, 448) for EfficientNet-B4
np.save(PROV / "Fig04_probability.npy", prob4.astype(np.float32))

HF = 5.55
fig, ax = canvas(HF)
# input
image_axes(fig, HF, 0.2, 3.95, 1.0, 1.0, x4, edge="#555")
ax.text(0.7, 5.02, "Input patch", ha="center", va="bottom", fontweight="bold", color=NAVY)
ax.text(0.7, 3.87, "3 @ 512×512\nVahadane-normalized", ha="center", va="top")
# encoder (x, w, y, h) and decoder blocks as a staircase in which each block lies below the previous one;
# labels sit on the inner side of each block, above its skip connection
enc = [(1.55, 0.18, 3.95, 1.10), (1.98, 0.22, 3.05, 0.80), (2.41, 0.26, 2.30, 0.62), (2.84, 0.30, 1.70, 0.48),
       (3.30, 0.34, 1.20, 0.38)]
dec = [(4.62, 0.30, 1.70, 0.48), (5.05, 0.26, 2.30, 0.62), (5.46, 0.22, 3.05, 0.80), (5.85, 0.18, 3.95, 1.10)]
enc_lab = [(f"E{i + 1}", f"{ch[i]} @ {512 // 2 ** (i + 1)}²") for i in range(5)]
dec_lab = [("D4", "256 @ 32²"), ("D3", "128 @ 64²"), ("D2", "64 @ 128²"), ("D1", "32 @ 256²")]
mid = lambda b: b[2] + b[3] / 2
for k, ((x, w, y, h), (name, size)) in enumerate(zip(enc, enc_lab)):
    rbox(ax, x, y, w, h, "#ffe6a9", "#d49a00")
    if k < 4:
        ax.text(x + w + 0.05, mid((x, w, y, h)) + 0.04, f"{name}\n{size}", ha="left", va="bottom", color=NAVY)
    else:
        ax.text(x + w / 2, y - 0.05, f"{name}\n{size}", ha="center", va="top", color=NAVY)
for (x, w, y, h), (name, size) in zip(dec, dec_lab):
    rbox(ax, x, y, w, h, "#d2e4fa", "#3d8fe0")
    ax.text(x - 0.05, mid((x, w, y, h)) + 0.04, f"{name}\n{size}", ha="right", va="bottom", color=NAVY)
arrow(ax, (1.22, mid(enc[0])), (1.53, mid(enc[0])))
for a, b in zip(enc, enc[1:]):  # bottom of one block to the top of the next
    arrow(ax, (a[0] + a[1] / 2, a[2]), (b[0] + b[1] / 2, b[2] + b[3] + 0.02))
for a, b in zip(dec, dec[1:]):  # top of one block to the bottom of the next
    arrow(ax, (a[0] + a[1] / 2, a[2] + a[3]), (b[0] + b[1] / 2, b[2] - 0.02))
# skip connections E1-D1 ... E4-D4
for e, d in zip(enc[:4], dec[::-1]):
    arrow(ax, (e[0] + e[1] + 0.03, mid(e)), (d[0] - 0.03, mid(d)), color="#7a8798", lw=0.8, ls=(0, (3, 2)))
ax.text(3.75, 3.15, "skip connections (concatenation)", ha="center", va="center", color="#5b6676", style="italic")
# coordinate branch, concatenation and bottleneck
cx, cy = 3.85, mid(enc[4])
ax.add_patch(Circle((cx, cy), 0.09, fc="white", ec=NAVY, lw=0.9))
ax.text(cx, cy, "C", ha="center", va="center", fontweight="bold")
arrow(ax, (enc[4][0] + enc[4][1], cy), (cx - 0.1, cy))
bott = (4.05, 0.36, 1.20, 0.38)  # x, w, y, h
rbox(ax, bott[0], bott[2], bott[1], bott[3], "#d9c2f7", "#7a3fe0")
arrow(ax, (cx + 0.1, cy), (bott[0] - 0.02, cy))
ax.text(bott[0] + bott[1] + 0.06, bott[2] - 0.02, f"Bottleneck\n{ch[4]} + 32 = {ch[4] + 32} @ 16²", ha="left", va="top",
        color=NAVY)
arrow(ax, (bott[0] + bott[1], bott[2] + bott[3]), (dec[0][0] + dec[0][1] / 2, dec[0][2] - 0.02))
rbox(ax, 2.4, 0.15, 3.7, 0.42, "#e2ccfa", "#8a4ee6")
ax.text(4.25, 0.36, "WSI coordinates (x, y) in [0, 1] → MLP 2→64→32 → 32 @ 16²", ha="center", va="center", color=NAVY)
arrow(ax, (cx, 0.57), (cx, cy - 0.1), color="#8a4ee6")
# head and outputs
rbox(ax, 6.1, mid(dec[3]) - 0.23, 0.32, 0.46, "#dcefd1", "#4f9d2f")
ax.text(6.26, mid(dec[3]), "1×1\nconv", ha="center", va="center")
arrow(ax, (dec[3][0] + dec[3][1], mid(dec[3])), (6.08, mid(dec[3])))
prob_im = image_axes(fig, HF, 6.52, 4.15, 0.9, 0.9, prob4, edge="#555").images[0]
prob_im.set_cmap("magma"); prob_im.set_clim(0, 1)
ax.text(6.97, 5.1, "Tumor probability", ha="center", va="bottom", fontweight="bold", color=NAVY)
cbar = fig.colorbar(prob_im, cax=fig.add_axes([6.52 / 7.5, 4.0 / HF, 0.9 / 7.5, 0.07 / HF]), orientation="horizontal",
                    ticks=[0, 0.5, 1])
cbar.ax.minorticks_off(); cbar.ax.tick_params(length=2, pad=1)
arrow(ax, (6.42, mid(dec[3])), (6.5, mid(dec[3])))
# reference as in Fig 2C: white tumor, gray annotated non-tumor tissue, black not annotated (label 0)
ref4 = np.zeros(gt4.shape + (3,), np.uint8)
ref4[valid4] = (110, 110, 110)
ref4[gt4 & valid4] = (255, 255, 255)
image_axes(fig, HF, 6.52, 2.55, 0.9, 0.9, ref4, edge="#555")
ax.text(6.97, 3.52, "Reference mask", ha="center", va="bottom", fontweight="bold", color=NAVY)
ax.text(6.97, 2.48, "same patch", ha="center", va="top")
ax.text(0.2, 2.4, "Encoder", fontweight="bold", color="#8a6100", va="top")
ax.text(0.2, 2.22, "ImageNet-pretrained\nEfficientNet-B4", va="top", color="#8a6100")
ax.text(6.25, 2.1, "Decoder", fontweight="bold", color="#1d5fa8", va="top")
ax.text(6.25, 1.92, "bilinear ×2,\nskip concatenation,\nDoubleConv", va="top", color="#1d5fa8")
ax.text(0.2, 0.62, "DoubleConv = 2 × (3×3 conv,\nbatch norm, ReLU) + dropout 0.1", va="top")
save(fig, "Fig04_architecture.png", "Fig4.tif", {
    "figure": "Fig 4",
    "panels": {"input_patch": prec(src4, shown="Vahadane-normalized model input",
                                   normalized_input="Fig04_input_normalized.png (this folder)"),
               "tumor_probability": {"shown": "sigmoid output of the selected checkpoint for the input patch",
                                     "checkpoint": source(WORK / "runs" / "final_main_training" / "best.pt", WORK,
                                                          epoch=ck.get("epoch") if isinstance(ck, dict) else None),
                                     "coords_input": [float(src4.coord_x), float(src4.coord_y)],
                                     "probability_range": [round(float(prob4.min()), 6), round(float(prob4.max()), 6)]},
               "reference_mask": {"tumor_mask": source(src4["mask"], WORK), "valid_mask": source(src4["valid_mask"], WORK),
                                  "shown": "white tumor, gray annotated non-tumor tissue, black not annotated"},
               "probability_colour_scale": "magma, 0-1, colour bar below the map"},
    "parameters": {"patch_choice": "stored L1 tumor patch of the slide (any stored patch); fewest mask seams, then most tissue",
                   "vahadane_target": source(target_path, WORK), "vahadane_numpy_seed": VAHADANE_SEED,
                   "encoder_channels": list(ch)}},
    extra_files=(PROV / "Fig04_input_normalized.png", PROV / "Fig04_probability.npy"))

# ---------------------------------------------------------------- Fig 1
NAME_L = {0: "L0 (1×)", 1: "L1 (4× downsample)", 2: "L2 (16× downsample)"}
right = {(l, c): pick(l, c, src=all_rows) for l in LEVELS for c in ("normal", "tumor")}
pyr = {l: pick(l, "tumor", 1, src=all_rows) for l in LEVELS}
sel_right = {(l, c): pick(l, c) for l in LEVELS for c in ("normal", "tumor")}

HF = 5.6
fig, ax = canvas(HF)
# left: multiresolution views
rbox(ax, 0.08, 0.12, 2.0, 5.38, "#fcfdff", "#91ade3")
ax.text(1.08, 5.38, "Multiresolution views", ha="center", va="top", fontweight="bold", color=NAVY)
ax.text(1.08, 5.18, "tumor patches of one training WSI", ha="center", va="top")
for l, yy in ((2, 3.42), (1, 1.88), (0, 0.34)):
    image_axes(fig, HF, 0.505, yy, 1.15, 1.15, rgb(pyr[l].img), edge="#a5bedf")
    ax.text(0.2, yy + 1.2, NAME_L[l], ha="left", va="bottom", fontweight="bold", color=NAVY)
# middle: workflow steps
mx, mw, mh = 2.4, 2.6, 0.62
steps = [("Preprocessing", "Tissue masking, QC, stain normalization", "#f6d2df", "#e85a90"),
         ("Feature engineering", "Multilevel patches and coordinates", "#ffe6a9", "#e8a700"),
         ("Dataset construction", "Six level × class sampling strata", "#d2e4fa", "#5da8ff"),
         ("Segmentation model", "EfficientNet-B4 U-Net", "#e2ccfa", "#a05aff"),
         ("Whole-slide reconstruction and fusion", "Dense streaming, support-aware fusion", "#dcefd1", "#86c95b"),
         ("Evaluation", "Dice, IoU, HD95, ASSD", "#d5cbf7", "#8a68ff")]
cy = [5.17 - k * (mh + 0.22) for k in range(6)]
for c, (head, sub, bg, edge) in zip(cy, steps):
    rbox(ax, mx, c - mh / 2, mw, mh, bg, edge, 1.1)
    ax.text(mx + mw / 2, c + 0.12, head, ha="center", va="center", fontweight="bold", color=NAVY)
    ax.text(mx + mw / 2, c - 0.13, sub, ha="center", va="center")
for a, b in zip(cy, cy[1:]):
    arrow(ax, (mx + mw / 2, a - mh / 2 - 0.02), (mx + mw / 2, b + mh / 2 + 0.03), lw=1.4, scale=9)
# right: representative inputs and their reference masks
rx, rw = 5.3, 2.12
ts, tg = 0.6, 0.1
xs = [rx + (rw - (3 * ts + 2 * tg)) / 2 + i * (ts + tg) for i in range(3)]
rbox(ax, rx, 3.2, rw, 2.3, "#fcfcfe", "#91ade3")
ax.text(rx + rw / 2, 5.38, "Representative patch inputs", ha="center", va="top", fontweight="bold", color=NAVY)
rbox(ax, rx, 0.12, rw, 2.85, "#fcfcfe", "#91ade3")
ax.text(rx + rw / 2, 2.85, "Reference masks", ha="center", va="top", fontweight="bold", color=NAVY)
ax.text(rx + rw / 2, 2.65, "of the six patches above", ha="center", va="top")
for i, l in enumerate((2, 1, 0)):
    ax.text(xs[i] + ts / 2, 5.13, f"L{l}", ha="center", va="top", fontweight="bold", color=NAVY)
    image_axes(fig, HF, xs[i], 4.21, ts, ts, rgb(right[(l, "normal")].img), edge=GREEN, lw=1.6)
    image_axes(fig, HF, xs[i], 3.53, ts, ts, rgb(right[(l, "tumor")].img), edge=ORANGE, lw=1.6)
    image_axes(fig, HF, xs[i], 1.72, ts, ts, binary(right[(l, "normal")]["mask"]).astype(float), edge=GREEN, lw=1.6, gray=True)
    image_axes(fig, HF, xs[i], 1.04, ts, ts, binary(right[(l, "tumor")]["mask"]).astype(float), edge=ORANGE, lw=1.6, gray=True)
for (lx, col, lab) in ((rx + 0.3, GREEN, "Normal"), (rx + 1.15, ORANGE, "Tumor")):
    ax.add_patch(Rectangle((lx, 3.3), 0.13, 0.13, fill=False, ec=col, lw=1.4))
    ax.text(lx + 0.2, 3.365, lab, ha="left", va="center")
ax.text(rx + rw / 2, 0.75, "white = tumor", ha="center", va="center")
arrow(ax, (2.1, cy[0]), (mx - 0.04, cy[0]), lw=1.4, scale=9)
arrow(ax, (rx - 0.02, cy[1]), (mx + mw + 0.04, cy[1]), lw=1.4, scale=9)
arrow(ax, (mx + mw + 0.02, cy[3]), (rx - 0.04, cy[3]), lw=1.4, scale=9)
panels1 = {f"left_L{l}": prec(pyr[l], shown=f"multiresolution view {NAME_L[l]}") for l in LEVELS}
panels1.update({f"right_{c}_L{l}": prec(right[(l, c)], shown=f"representative {c} input at L{l} and its reference mask")
                for l in LEVELS for c in ("normal", "tumor")})
save(fig, "Fig01_pipeline.png", "Fig1.tif", {
    "figure": "Fig 1", "panels": panels1,
    "parameters": {"patch_choice": "stored patches of the slide; right column: fewest mask seams, then most tissue; "
                                   "left column: second-ranked tumor patch per level"}})

# ---------------------------------------------------------------- Fig 2
pool_dir = WORK / "manifests" / "coordinate_pools"
pool1 = pd.read_csv(pool_dir / f"{SLIDE}_L1.csv")

mh_, mw_ = mos.shape[:2]
pw = 1.55                                  # width of panels A-D (in)
ph = min(3.4, pw * mh_ / mw_)              # height of panels A-D
ES = 3.0                                   # size of the two panel-E images (in)
HF = 0.12 + ph + 0.45 + ES + 0.45 + 1.12 + 0.35
fig, ax = canvas(HF)
gx = [0.3 + i * (pw + 0.3) for i in range(4)]
ytop = HF - 0.12 - ph
ref = np.zeros(mos.shape[:2] + (3,), np.uint8)
ref[mos_val] = (110, 110, 110)
ref[mos_tum & mos_val] = (255, 255, 255)
panA = image_axes(fig, HF, gx[0], ytop, pw, ph, mos)
panB = image_axes(fig, HF, gx[1], ytop, pw, ph, tis.astype(float), gray=True)
panC = image_axes(fig, HF, gx[2], ytop, pw, ph, ref)
panD = image_axes(fig, HF, gx[3], ytop, pw, ph, mos)
for a in (panA, panB, panC, panD):
    a.set_aspect("equal")
for _, r in pool1.iterrows():
    px, py = r.x0 / 16 - x0, r.y0 / 16 - y0
    col = ORANGE if str(r.category_lr).lower().startswith("tum") else GREEN
    panD.add_patch(Rectangle((px, py), 512 * 4 / 16, 512 * 4 / 16, fill=False, ec=col, lw=0.7))
for a, lab, let in ((panA, "L2 view (16×)", "A"), (panB, "Tissue mask", "B"), (panC, "Reference mask", "C"),
                    (panD, "L1 candidates by class", "D")):
    a.set_xlabel(lab)
    a.text(-0.04, 1.0, let, transform=a.transAxes, fontsize=10, fontweight="bold", ha="right", va="top")
# E: before | after normalization
ey = ytop - 0.45 - ES
ex = (7.5 - (2 * ES + 0.4)) / 2
for k, (img, lab) in enumerate(((raw_e, "Before normalization"), (norm_e, "After Vahadane normalization"))):
    a = image_axes(fig, HF, ex + k * (ES + 0.4), ey, ES, ES, img)
    a.set_xlabel(lab)
    if k == 0:
        a.text(-0.04, 1.0, "E", transform=a.transAxes, fontsize=10, fontweight="bold", ha="right", va="top")
# F: selected training patches
fy = ey - 0.45 - 1.12
fx = [0.22 + i * (1.12 + 0.096) for i in range(6)]
for i, (l, c) in enumerate([(0, "normal"), (1, "normal"), (2, "normal"), (0, "tumor"), (1, "tumor"), (2, "tumor")]):
    a = image_axes(fig, HF, fx[i], fy, 1.12, 1.12, rgb(sel_right[(l, c)].img), edge=GREEN if c == "normal" else ORANGE, lw=1.6)
    a.set_xlabel(f"L{l} {c}")
    if i == 0:
        a.text(-0.06, 1.0, "F", transform=a.transAxes, fontsize=10, fontweight="bold", ha="right", va="top")
l2_files = []
for f, (x, y) in zip(l2_imgs, coords):
    cls = f.parent.name
    l2_files.append({"x0_level0": x, "y0_level0": y, "class": cls, "image": source(f, WORK),
                     "tumor_mask": source(f.parent.parent / f"{cls}_Masks" / f"{f.stem}__tumor_mask.png", WORK),
                     "valid_mask": source(f.parent.parent / f"{cls}_Valid_Masks" / f"{f.stem}__valid_mask.png", WORK)})
save(fig, "Fig02_preprocessing.png", "Fig2.tif", {
    "figure": "Fig 2",
    "slide": {"slide_id": SLIDE, "provider": provider, "split": "train"},
    "panels": {"A_B_C": {"shown": "L2 mosaic of every stored L2 patch of the slide, its tissue mask and reference mask",
                         "mosaic_crop_px_on_16x_grid": {"rows": [int(y0), int(y1)], "cols": [int(x0), int(x1)]},
                         "l2_patches": l2_files},
               "D": {"shown": "L1 candidates of the coordinate pool outlined by class",
                     "coordinate_pool": source(pool_dir / f"{SLIDE}_L1.csv", WORK), "n_candidates": int(len(pool1))},
               "E": {"shown": "L0 tumor patch before and after Vahadane normalization",
                     "before": prec(src_e), "after": source(cache_e, WORK),
                     "check_mean_abs_diff_vs_renormalization_8bit": round(fresh_diff_e, 3)},
               "F": {f"{c}_L{l}": prec(sel_right[(l, c)], shown="selected training patch")
                     for l in LEVELS for c in ("normal", "tumor")}},
    "moved_to_table": {"former_F_G": "candidate-grid counts of this WSI and training-bucket counts: manuscript table "
                                     "tab:prep (fill_manuscript_numbers.py, TABLE_PREP), from "
                                     "manifests/coordinate_pools/<slide>_L*.stats.json and selected/train_selected.csv"},
    "parameters": {"tissue_mask": "Otsu on HSV saturation, threshold >= 0.20, disk 2, min object 500 (pipeline rule)",
                   "patch_choice": "selected patches; fewest mask seams, then most tissue"}})

print("Fig 2E patch:", src_e.img, "| Fig 4 patch:", src4.img, "| prob range", float(prob4.min()), float(prob4.max()))
