"""Regenerate Figs 1, 2, 4 and 7 of the PANDA manuscript from study data.

Every image panel is read from the stored PANDA training patches, their
reference masks, the training-derived Vahadane target, or the final model
checkpoint. Nothing is drawn by hand or by an image generator. Each figure
is written as PNG and as PLOS TIFF (RGB, LZW, 300 dpi, 7.5 in wide) together
with a JSON file that lists every source file with its SHA-256.

Usage:
    python make_figures_1_2_4_7.py <output_dir> [work_root]

work_root is the pipeline work directory (patches/, selected/, manifests/,
runs/, stain_target_train.png). If omitted, PANDA_DATA_ROOT is used.
"""
import hashlib
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
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from PIL import Image
from tiatoolbox.tools.stainnorm import VahadaneNormalizer

OUT = Path(sys.argv[1])
if len(sys.argv) < 3 and "PANDA_DATA_ROOT" not in os.environ:
    sys.exit("give work_root as the second argument or set PANDA_DATA_ROOT")
WORK = Path(sys.argv[2] if len(sys.argv) > 2 else os.environ["PANDA_DATA_ROOT"])
SLIDE = "362d30ace92ce17128d176137362b8c2"
LEVELS = (0, 1, 2)
DS = {0: 1, 1: 4, 2: 16}
NAVY, ARROW = "#0c1644", "#186bd1"
GREEN, ORANGE = "#10a64a", "#f07f0c"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7})


# ---------------------------------------------------------------- helpers
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rgb(path):
    return np.asarray(Image.open(path).convert("RGB"))


def binary(path):
    return np.asarray(Image.open(path).convert("L")) > 127


def save(fig, png_name, tif_name, sources):
    OUT.mkdir(parents=True, exist_ok=True)
    png = OUT / png_name
    fig.savefig(png, dpi=300, facecolor="white")
    plt.close(fig)
    im = Image.open(png).convert("RGB")
    if im.width > 2250:
        im = im.resize((2250, round(im.height * 2250 / im.width)), Image.LANCZOS)
        im.save(png, dpi=(300, 300))
    im.save(OUT / tif_name, compression="tiff_lzw", dpi=(300, 300))
    rec = {"script": Path(__file__).name, "slide_id": SLIDE, "paths_relative_to": "PANDA_DATA_ROOT",
           "sources": {k: {"path": Path(v).resolve().relative_to(WORK.resolve()).as_posix(), "sha256": sha(v)}
                       for k, v in sources.items()}}
    (OUT / (Path(png_name).stem + "_sources.json")).write_text(json.dumps(rec, indent=2), encoding="utf-8")
    print("saved", png_name, im.size)


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
        self.norm.fit(np.array(rgb(target_path), dtype=np.uint8, copy=True, order="C"))

    def __call__(self, img):
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


def show(ax, img, title=None, edge=None, cmap=None, fs=7):
    ax.imshow(img, cmap=cmap, interpolation="nearest" if cmap == "gray" else "antialiased")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor(edge or "#9aa7b8"); s.set_linewidth(1.6 if edge else 0.6)
    if title:
        ax.set_title(title, fontsize=fs, pad=2)


def panel_letter(ax, letter, x=-0.02, y=1.02):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="right")


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

# ---------------------------------------------------------------- Fig 4
src4 = pick(0, "tumor")
raw4 = rgb(src4.img)
norm4 = vahadane(raw4)
fig, axs = plt.subplots(1, 3, figsize=(7.5, 2.85), gridspec_kw={"width_ratios": [1, 1, 1]})
show(axs[0], raw4, "Source patch (before normalization)")
show(axs[1], norm4, "After Vahadane normalization")
tgt = rgb(target_path)
show(axs[2], tgt, "Training-derived target")
for ax, l in zip(axs, "ABC"):
    panel_letter(ax, l)
fig.text(0.5, 0.075, f"PANDA training WSI {SLIDE} ({provider.capitalize()}), L0 patch at level-0 x = {int(src4.x0):,}, y = {int(src4.y0):,}",
         ha="center", fontsize=6.2, color="#333333")
fig.text(0.5, 0.03, "Target: stain_target_train.png, selected from training patches only and reused unchanged for all inference",
         ha="center", fontsize=6.2, color="#333333")
fig.subplots_adjust(left=0.03, right=0.98, top=0.9, bottom=0.14, wspace=0.08)
save(fig, "Fig04_vahadane.png", "Fig4.tif", {"source_patch": src4.img, "vahadane_target": target_path})
Image.fromarray(norm4).save(OUT / "Fig04_normalized_patch.png")

# ---------------------------------------------------------------- Fig 7
src7 = pick(1, "tumor", src=all_rows)
x7 = vahadane(rgb(src7.img))
gt7 = binary(src7["mask"])
model = UNetEfficientNetB4()
ck = torch.load(WORK / "runs" / "final_main_training" / "best.pt", map_location="cpu", weights_only=False)
model.load_state_dict(ck["model"])
model.eval()
with torch.no_grad():
    t = torch.from_numpy(x7.copy()).permute(2, 0, 1).float()[None] / 255.0
    c = torch.tensor([[float(src7.coord_x), float(src7.coord_y)]], dtype=torch.float32)
    prob7 = torch.sigmoid(model(t, c))[0, 0].numpy()
ch = model.channels  # (24, 32, 56, 160, 448) for EfficientNet-B4

fig = plt.figure(figsize=(7.5, 4.15))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 55); ax.axis("off")


def block(x, y, w, h, fc, ec, label, sub, fs=6.3):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15,rounding_size=0.6", fc=fc, ec=ec, lw=1.0))
    ax.text(x + w / 2, y + h + 0.9, label, ha="center", va="bottom", fontsize=fs, fontweight="bold", color=NAVY)
    ax.text(x + w / 2, y - 0.8, sub, ha="center", va="top", fontsize=5.8, color="#333333")


def arr(a, b, color=ARROW, lw=1.1, style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=7, color=color, lw=lw, linestyle=ls, shrinkA=0, shrinkB=0))


ax.text(50, 53.3, "EfficientNet-B4 U-Net with coordinate embedding (executed architecture)", ha="center", fontsize=8.5, fontweight="bold", color=NAVY)
ax.imshow(x7, extent=(1, 12, 33, 44), zorder=2); ax.add_patch(Rectangle((1, 33), 11, 11, fill=False, ec="#555", lw=0.6))
ax.text(6.5, 45.2, "Input patch", ha="center", fontsize=6.3, fontweight="bold", color=NAVY)
ax.text(6.5, 31.8, "3 @ 512×512\n(Vahadane-normalized)", ha="center", va="top", fontsize=5.6)
enc = [(16, 34, 2.4, 10, f"E1", f"{ch[0]} @ 256²"), (22, 27, 2.8, 8, "E2", f"{ch[1]} @ 128²"),
       (28, 20.5, 3.2, 6.2, "E3", f"{ch[2]} @ 64²"), (34, 14.5, 3.8, 4.8, "E4", f"{ch[3]} @ 32²"),
       (41, 9.5, 4.6, 3.6, "E5", f"{ch[4]} @ 16²")]
for x, y, w, h, l, s in enc:
    block(x, y, w, h, "#ffe6a9", "#d49a00", l, s)
ax.text(29, 47.5, "Encoder: ImageNet-pretrained EfficientNet-B4", ha="center", fontsize=6.5, color="#8a6100", fontweight="bold")
arr((12.3, 38.5), (15.8, 38.5))
for (xa, ya, wa, ha_, *_), (xb, yb, wb, hb, *_) in zip(enc, enc[1:]):
    arr((xa + wa / 2, ya - 2.6), (xb + 0.2, yb + hb / 2))
# coordinate branch
ax.add_patch(FancyBboxPatch((38, 0.8), 46, 3.6, boxstyle="round,pad=0.15,rounding_size=0.6", fc="#e2ccfa", ec="#8a4ee6", lw=1.0))
ax.text(61, 2.6, "WSI coordinates (x, y) in [0, 1] → MLP 2→64→32 → broadcast 32 @ 16²", ha="center", va="center", fontsize=5.8, color=NAVY)
ax.add_patch(plt.Circle((50.5, 11.3), 1.1, fc="white", ec=NAVY, lw=0.9)); ax.text(50.5, 11.3, "C", ha="center", va="center", fontsize=6, fontweight="bold")
arr((45.9, 11.3), (49.3, 11.3)); arr((50.5, 4.6), (50.5, 10.1), color="#8a4ee6")
block(53, 9.5, 4.8, 3.6, "#d9c2f7", "#7a3fe0", "Bottleneck", f"{ch[4]} + 32 = {ch[4] + 32} @ 16²")
arr((51.7, 11.3), (52.8, 11.3))
dec = [(60, 14.5, 3.8, 4.8, "D4", "256 @ 32²"), (66, 20.5, 3.2, 6.2, "D3", "128 @ 64²"),
       (72, 27, 2.8, 8, "D2", "64 @ 128²"), (78, 34, 2.4, 10, "D1", "32 @ 256²")]
for x, y, w, h, l, s in dec:
    block(x, y, w, h, "#d2e4fa", "#3d8fe0", l, s)
ax.text(71, 47.5, "Decoder: bilinear ×2, concatenate skip, DoubleConv", ha="center", fontsize=6.5, color="#1d5fa8", fontweight="bold")
arr((58.1, 11.3), (60.2, 16.9))
for (xa, ya, wa, ha_, *_), (xb, yb, wb, hb, *_) in zip(dec, dec[1:]):
    arr((xa + wa / 2 + 0.6, ya + ha_ + 2.8), (xb + 0.2, yb + hb / 2))
# skip connections E4->D4, E3->D3, E2->D2, E1->D1
for (xe, ye, we, he, *_), (xd, yd, wd, hd, *_) in zip(enc[3::-1], dec):
    arr((xe + we + 0.4, ye + he / 2), (xd - 0.4, yd + hd / 2), color="#7a8798", lw=0.8, ls=(0, (3, 2)))
ax.text(46, 27.6, "skip connections (concatenation)", ha="center", fontsize=5.6, color="#5b6676", style="italic")
# head and outputs
ax.add_patch(FancyBboxPatch((82.6, 36.8), 3.6, 4.4, boxstyle="round,pad=0.15,rounding_size=0.6", fc="#dcefd1", ec="#4f9d2f", lw=1.0))
ax.text(84.4, 39.0, "1×1\nconv\n×2 up", ha="center", va="center", fontsize=5.2)
arr((80.8, 39.0), (82.4, 39.0))
ax.imshow(prob7, extent=(87.3, 97.3, 33.5, 43.5), cmap="magma", vmin=0, vmax=1, zorder=2)
ax.add_patch(Rectangle((87.3, 33.5), 10, 10, fill=False, ec="#555", lw=0.6))
ax.text(92.3, 44.7, "Tumor probability", ha="center", fontsize=6.3, fontweight="bold", color=NAVY)
ax.text(92.3, 32.3, "1 @ 512×512 (sigmoid)", ha="center", va="top", fontsize=5.6)
arr((86.4, 39.0), (87.1, 39.0))
ax.imshow(gt7, extent=(87.3, 97.3, 16, 26), cmap="gray", vmin=0, vmax=1, interpolation="nearest", zorder=2)
ax.add_patch(Rectangle((87.3, 16), 10, 10, fill=False, ec="#555", lw=0.6))
ax.text(92.3, 27.2, "Reference mask", ha="center", fontsize=6.3, fontweight="bold", color=NAVY)
ax.text(92.3, 14.8, "PANDA label, same patch", ha="center", va="top", fontsize=5.6)
ax.text(1, 7.2, "DoubleConv = 2 × (3×3 conv, batch norm, ReLU)\n+ dropout 0.1. Probability map: selected checkpoint\n"
        "(epoch 11) applied to the L1 training patch shown.", fontsize=5.6, va="top", color="#333333")
save(fig, "Fig07_architecture.png", "Fig7.tif", {"input_patch": src7.img, "reference_mask": src7["mask"],
                                                 "checkpoint": WORK / "runs" / "final_main_training" / "best.pt",
                                                 "vahadane_target": target_path})

# ---------------------------------------------------------------- Fig 1
NAME_L = {0: "L0 (1×)", 1: "L1 (4× downsample)", 2: "L2 (16× downsample)"}
right = {(l, c): pick(l, c, src=all_rows) for l in LEVELS for c in ("normal", "tumor")}
pyr = {l: pick(l, "tumor", 1, src=all_rows) for l in LEVELS}
sel_right = {(l, c): pick(l, c) for l in LEVELS for c in ("normal", "tumor")}

fig = plt.figure(figsize=(7.5, 4.25))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 16); ax.set_ylim(0, 9); ax.axis("off")


def rbox(x, y, w, h, fc, ec, lw=1.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.003,rounding_size=0.012", fc=fc, ec=ec, lw=lw))


def txt(x, y, s, fs=7, bold=False, ha="center", color=NAVY):
    ax.text(x, y, s, fontsize=fs, ha=ha, va="center", fontweight="bold" if bold else "normal", color=color)


def tile(img, x, y, w, h, edge, lw=1.2, cmap=None):
    ax.imshow(img, extent=(x, x + w, y, y + h), aspect="auto", cmap=cmap, vmin=0, vmax=1 if cmap else None, zorder=2)
    ax.add_patch(Rectangle((x, y), w, h, fill=False, ec=edge, lw=lw, zorder=3))


def bigarrow(a, b):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=9, color=ARROW, lw=1.6, shrinkA=0, shrinkB=0))


txt(8, 8.45, "PANDA whole-slide segmentation workflow", fs=12, bold=True)
xL, yL, wL, hL = 0.35, 0.75, 4.6, 7.15
rbox(xL, yL, wL, hL, "#fcfdff", "#91ade3")
txt(xL + wL / 2, 7.55, "Multiresolution patch views", fs=8.5, bold=True)
txt(xL + wL / 2, 7.2, "L2, L1 and L0 tumor patches of one training WSI", fs=5.8)
for l, yy in ((2, 4.95), (1, 2.9), (0, 0.95)):
    tile(rgb(pyr[l].img), xL + (wL - 1.75) / 2, yy, 1.75, 1.75, "#a5bedf")
    txt(xL + 0.15, yy + 1.92, NAME_L[l], fs=6.2, bold=True, ha="left")
mx, mw, mh = 5.55, 4.6, 0.9
cy = [7.05, 5.85, 4.65, 3.45, 2.25, 1.05]
steps = [("Preprocessing", "Tissue masking, QC, stain normalization", "#f6d2df", "#e85a90"),
         ("Feature engineering", "Multilevel patches and coordinates", "#ffe6a9", "#e8a700"),
         ("Dataset construction", "6-stratum structured sampling (level × class)", "#d2e4fa", "#5da8ff"),
         ("Deep learning segmentation model", "EfficientNet-B4 U-Net", "#e2ccfa", "#a05aff"),
         ("Whole-slide reconstruction & fusion", "Dense streaming, support-aware fusion", "#dcefd1", "#86c95b"),
         ("Evaluation", "Dice, IoU, HD95, ASSD", "#d5cbf7", "#8a68ff")]
for c, (head, sub, bg, edge) in zip(cy, steps):
    rbox(mx, c - mh / 2, mw, mh, bg, edge, 1.1)
    txt(mx + mw / 2, c + 0.16, head, fs=7.6, bold=True)
    txt(mx + mw / 2, c - 0.2, sub, fs=6.0)
for a, b in zip(cy, cy[1:]):
    bigarrow((mx + mw / 2, a - mh / 2 - 0.03), (mx + mw / 2, b + mh / 2 + 0.04))
rx, rw = 10.75, 4.9
rbox(rx, 4.55, rw, 3.35, "#fcfcfe", "#91ade3")
txt(rx + rw / 2, 7.6, "Representative patch inputs", fs=8.5, bold=True)
xs = [rx + 0.3, rx + 1.85, rx + 3.4]
for i, l in enumerate((2, 1, 0)):
    txt(xs[i] + 0.6, 7.25, NAME_L[l].split(" ")[0], fs=6.2, bold=True)
    tile(rgb(right[(l, "normal")].img), xs[i], 6.0, 1.2, 1.05, GREEN, 1.6)
    tile(rgb(right[(l, "tumor")].img), xs[i], 4.85, 1.2, 1.05, ORANGE, 1.6)
ax.add_patch(Rectangle((rx + 0.35, 4.64), 0.13, 0.13, fill=False, ec=GREEN, lw=1.4)); txt(rx + 0.55, 4.705, "Normal", fs=6, ha="left")
ax.add_patch(Rectangle((rx + 1.9, 4.64), 0.13, 0.13, fill=False, ec=ORANGE, lw=1.4)); txt(rx + 2.1, 4.705, "Tumor", fs=6, ha="left")
rbox(rx, 0.75, rw, 3.5, "#fcfcfe", "#91ade3")
txt(rx + rw / 2, 3.95, "Reference masks", fs=8.5, bold=True)
txt(rx + rw / 2, 3.65, "of the six patches above", fs=5.8)
for i, l in enumerate((2, 1, 0)):
    tile(binary(right[(l, "normal")]["mask"]).astype(float), xs[i], 2.45, 1.2, 1.05, GREEN, 1.6, cmap="gray")
    tile(binary(right[(l, "tumor")]["mask"]).astype(float), xs[i], 1.3, 1.2, 1.05, ORANGE, 1.6, cmap="gray")
txt(rx + rw / 2, 1.0, "Binary PANDA tumor masks (white = tumor)", fs=5.8)
bigarrow((xL + wL + 0.03, cy[0]), (mx - 0.06, cy[0]))
bigarrow((rx - 0.06, cy[1]), (mx + mw + 0.05, cy[1]))
bigarrow((mx + mw + 0.05, cy[3]), (rx - 0.06, cy[3]))
ax.plot([mx + mw / 2, mx + mw / 2, 2.65, 2.65], [0.6, 0.32, 0.32, 0.62], color=ARROW, lw=1.5)
bigarrow((2.65, 0.6), (2.65, 0.74))
src1 = {f"pyramid_L{l}": pyr[l].img for l in LEVELS}
src1.update({f"{c}_L{l}_image": right[(l, c)].img for l in LEVELS for c in ("normal", "tumor")})
src1.update({f"{c}_L{l}_mask": right[(l, c)]["mask"] for l in LEVELS for c in ("normal", "tumor")})
save(fig, "Fig01_pipeline.png", "Fig1.tif", src1)

# ---------------------------------------------------------------- Fig 2
pool_dir = WORK / "manifests" / "coordinate_pools"
stats = {l: json.loads((pool_dir / f"{SLIDE}_L{l}.stats.json").read_text()) for l in LEVELS}
pool1 = pd.read_csv(pool_dir / f"{SLIDE}_L1.csv")
buckets = sel.groupby(["level", "category", "provider"]).size()

fig = plt.figure(figsize=(7.5, 7.3))
gs = fig.add_gridspec(3, 4, height_ratios=[2.75, 1.75, 1.25], hspace=0.2, wspace=0.12,
                      left=0.02, right=0.985, top=0.95, bottom=0.015)
fig.suptitle(f"PANDA patch preparation on training WSI {SLIDE[:12]}… ({provider.capitalize()})", fontsize=9, fontweight="bold", color=NAVY)


def ltitle(ax, s):
    ax.set_title(s, fontsize=7, pad=3, loc="left")


axA, axB, axC, axD = (fig.add_subplot(gs[0, i]) for i in range(4))
show(axA, mos); ltitle(axA, "A  L2 view (16×)")
show(axB, tis.astype(float), cmap="gray"); ltitle(axB, "B  Tissue mask")
ref = np.zeros(mos.shape[:2] + (3,), np.uint8)
ref[mos_val] = (110, 110, 110)
ref[mos_tum & mos_val] = (255, 255, 255)
show(axC, ref); ltitle(axC, "C  Reference mask")
show(axD, mos); ltitle(axD, "D  L1 candidates by class")
for _, r in pool1.iterrows():
    px, py = r.x0 / 16 - x0, r.y0 / 16 - y0
    col = ORANGE if str(r.category_lr).lower().startswith("tum") else GREEN
    axD.add_patch(Rectangle((px, py), 512 * 4 / 16, 512 * 4 / 16, fill=False, ec=col, lw=0.7))

axE = fig.add_subplot(gs[1, 0:2]); axE.axis("off")
ltitle(axE, "E  Vahadane normalization of an L0 tumor patch (before | after)")
axE.imshow(np.concatenate([raw4, np.full((512, 24, 3), 255, np.uint8), norm4], 1))

axF = fig.add_subplot(gs[1, 2]); axF.axis("off")
ltitle(axF, "F  Candidate grid, this WSI")
lines = ["Level            L0     L1     L2"]
for key, lab in (("grid_positions_total", "Grid positions"), ("rejected_low_tissue", "Low tissue"),
                 ("rejected_low_valid_annotation", "Low valid annot."), ("rejected_mixed_borderline", "Tumor 0–5%"),
                 ("eligible_normal_before_cap", "Normal (0%)"), ("eligible_tumor_before_cap", "Tumor (≥5%)")):
    lines.append(f"{lab:<16}" + "".join(f"{stats[l].get(key, 0):>7,}" for l in LEVELS))
axF.text(0.0, 0.92, "\n".join(lines), family="DejaVu Sans Mono", fontsize=5.6, va="top", transform=axF.transAxes)
axF.text(0.0, 0.12, "512×512 patches, stride 128\nat each logical level", fontsize=5.8, transform=axF.transAxes, color="#333")

axG = fig.add_subplot(gs[1, 3]); axG.axis("off")
ltitle(axG, "G  Six training buckets")
g_lines = ["Bucket      Karol.  Radb.   Total"]
for l in LEVELS:
    for c in ("normal", "tumor"):
        k_ = int(buckets.get((l, c, "karolinska"), 0)); r_ = int(buckets.get((l, c, "radboud"), 0))
        g_lines.append(f"L{l} {c:<7}{k_:>7,}{r_:>7,}{k_ + r_:>8,}")
g_lines.append(f"{'Total':<11}{int(buckets.xs('karolinska', level='provider').sum()):>7,}"
               f"{int(buckets.xs('radboud', level='provider').sum()):>7,}{int(buckets.sum()):>8,}")
axG.text(0.0, 0.92, "\n".join(g_lines), family="DejaVu Sans Mono", fontsize=5.6, va="top", transform=axG.transAxes)
axG.text(0.0, 0.12, "80,000 selected patches;\nall training WSIs", fontsize=5.8, transform=axG.transAxes, color="#333")

sub = gs[2, :].subgridspec(1, 6, wspace=0.08)
for i, (l, c) in enumerate([(0, "normal"), (1, "normal"), (2, "normal"), (0, "tumor"), (1, "tumor"), (2, "tumor")]):
    a = fig.add_subplot(sub[0, i])
    show(a, rgb(sel_right[(l, c)].img), edge=GREEN if c == "normal" else ORANGE)
    ltitle(a, ("H  " if i == 0 else "") + f"L{l} {c}")
src2 = {"vahadane_source_L0": src4.img, "vahadane_target": target_path, "selected_manifest": WORK / "selected" / "train_selected.csv"}
src2.update({f"l2_patch_{i:02d}": f for i, f in enumerate(l2_imgs)})
src2.update({f"stats_L{l}": pool_dir / f"{SLIDE}_L{l}.stats.json" for l in LEVELS})
src2["pool_L1"] = pool_dir / f"{SLIDE}_L1.csv"
src2.update({f"output_{c}_L{l}": sel_right[(l, c)].img for l in LEVELS for c in ("normal", "tumor")})
save(fig, "Fig02_preprocessing.png", "Fig2.tif", src2)

print("bucket counts:\n", buckets.unstack("provider"))
print("Fig4 patch:", src4.img, "| Fig7 patch:", src7.img, "| prob range", float(prob7.min()), float(prob7.max()))
