"""Regenerate Fig 5 of the PANDA manuscript: training augmentations on one stored patch.

The source is the stored L0 tumor patch used in the earlier figure (training
WSI 0018ae58b01bdadc8e347995b69f99aa, level-0 x0 = 1024, y0 = 9344; found by
matching the earlier "Original" panel against all 109,060 stored L0 patches,
correlation 0.977, next patch 0.20). It is shown as the model receives it in
training: the Vahadane-normalized copy from the pipeline's persistent cache.

Each panel applies one transform of PairAugStainHeavy
(panda_camelyon_style_pipeline.py) with the same operation and its strongest
configured setting, so that the effect is visible:
  horizontal flip, 90-degree rotation (PIL rotate, as in the pipeline),
  ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.05) at
  the upper bound of each factor, gamma 1.5 (range 0.7-1.5), Gaussian blur
  kernel 3 with sigma 1.2 (range 0.1-1.2), Gaussian noise sigma 12 (range
  3-12, seed 42), JPEG quality 35 (range 35-89).
The settings are written to the JSON log (provenance_log.py), not the figure.

Usage:
    python make_figure_5.py <output_dir> [work_root]
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
from PIL import Image

from provenance_log import patch_record, save_figure, source, stain_cache_file

OUT = Path(sys.argv[1])
if len(sys.argv) < 3 and "PANDA_DATA_ROOT" not in os.environ:
    sys.exit("give work_root as the second argument or set PANDA_DATA_ROOT")
WORK = Path(sys.argv[2] if len(sys.argv) > 2 else os.environ["PANDA_DATA_ROOT"])
SLIDE, LEVEL, X0, Y0 = "0018ae58b01bdadc8e347995b69f99aa", 0, 1024, 9344
SEED = 42
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7})

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

fig, axes = plt.subplots(2, 4, figsize=(7.5, 4.0))
for ax, (pic, title) in zip(axes.flat, panels):
    ax.imshow(pic)
    ax.set_title(title, fontsize=7.5, fontweight="bold", pad=3)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("#111111"); sp.set_linewidth(0.6)
fig.subplots_adjust(left=0.015, right=0.985, top=0.94, bottom=0.015, wspace=0.06, hspace=0.12)
save_figure(fig, OUT, "Fig05_augmentation.png", "Fig5.tif", {
    "figure": "Fig 5",
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
