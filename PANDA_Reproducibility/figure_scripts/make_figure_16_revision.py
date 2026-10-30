"""Fig 16 of the revised manuscript (30 Oct 2026): SICAPv2 official test-partition examples.

Four patches of the official patient-level test partition with reference tumor
on at least 10% of their valid tissue, chosen at the 20th, 40th, 60th and 80th
percentiles of per-patch Dice in external_sicapv2_official/results/tables/
per_tile_metrics.csv (constant coordinate). Each patch is predicted again
with the frozen checkpoint, the same preparation (norm_pool.prepare_sicap),
coordinate (0.5, 0.5) and threshold 0.43; the Dice printed is that of the
redrawn prediction, and the log records it next to the stored value.

Usage: python make_figure_16_revision.py <output_dir> <PANDA_Reproducibility> <SICAPv2 folder> <work_root_with_checkpoint>
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT, WORK, SICAP, CKROOT = (Path(a) for a in sys.argv[1:5])
sys.path.insert(0, str(WORK.resolve().parent / "PANDA_Pipeline"))
import os
os.environ.setdefault("PANDA_DATA_ROOT", str(CKROOT))
import torch
import norm_pool
import panda_camelyon_style_pipeline as core
from panda_final_pipeline import FinalConfig
from provenance_log import plos_style, save_figure, source

plos_style()
SRC = WORK / "external_sicapv2_official" / "results" / "tables" / "per_tile_metrics.csv"
t = pd.read_csv(SRC)
t = t[t.test_partition].copy()
tot = t[["const_tp", "const_fp", "const_tn", "const_fn"]].sum(axis=1)
t["ref_share"] = (t.const_tp + t.const_fn) / tot
t["dice"] = 2 * t.const_tp / (2 * t.const_tp + t.const_fp + t.const_fn)
cand = t[t.ref_share >= 0.10].sort_values("dice").reset_index(drop=True)
picks = [cand.iloc[int(round(q * (len(cand) - 1)))] for q in (0.2, 0.4, 0.6, 0.8)]

cfg = FinalConfig()
model, device, ck = core.load_best_model(cfg, None)
norm_pool.init(str(cfg.stain_target_path))
thr = 0.43
fig, axs = plt.subplots(4, 4, figsize=(7.5, 8.1))
panels, values = {}, {}
for i, r in enumerate(picks):
    img = SICAP / "images" / r.image_name
    msk = SICAP / "masks" / r.image_name.replace(".jpg", ".png")
    norm, gt, valid = norm_pool.prepare_sicap((str(img), str(msk)))
    x = torch.from_numpy(norm[None]).permute(0, 3, 1, 2).float().div_(255.0).to(device)
    c = torch.tensor([[0.5, 0.5]], dtype=torch.float32, device=device)
    with torch.inference_mode(), torch.amp.autocast("cuda", enabled=device.type == "cuda"):
        prob = torch.sigmoid(model(x, c))[0, 0].float().cpu().numpy()
    pred = prob >= thr
    tp = int((pred & gt & valid).sum()); fp = int((pred & ~gt & valid).sum()); fn = int((~pred & gt & valid).sum())
    d = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    rgb = np.asarray(plt.imread(img))
    err = np.full(gt.shape + (3,), 255, np.uint8)
    err[(~gt) & (~pred) & valid] = [230, 230, 230]; err[gt & pred & valid] = [60, 170, 90]
    err[gt & (~pred) & valid] = [240, 190, 40]; err[(~gt) & pred & valid] = [210, 60, 60]
    for j, (im, cmap) in enumerate(((rgb, None), (gt, "gray"), (pred & valid, "gray"), (err, None))):
        axs[i, j].imshow(im, cmap=cmap, vmin=0, vmax=1 if cmap else None); axs[i, j].set_xticks([]); axs[i, j].set_yticks([])
    axs[i, 0].set_ylabel(f"{'ABCD'[i]}   Dice {d:.4f}", rotation=0, ha="right", va="center", labelpad=6)
    panels["ABCD"[i]] = {"dataset": "SICAPv2 official release v2", "partition": "test", "image": source(img, SICAP),
                         "mask": source(msk, SICAP), "stored_dice": float(r.dice), "redrawn_dice": d,
                         "reference_share_of_valid": float(r.ref_share)}
for j, name in enumerate(("H&E", "Reference", "Prediction", "Error map")):
    axs[0, j].set_title(name)
from matplotlib.patches import Patch
fig.legend(handles=[Patch(color=np.array(c) / 255, label=l) for c, l in (([60, 170, 90], "TP"), ([240, 190, 40], "FN"), ([210, 60, 60], "FP"), ([230, 230, 230], "TN"))],
           loc="outside lower center", ncol=4, frameon=False)
record = {"figure": "Fig 16", "sources": [source(SRC, WORK)], "panels": panels,
          "parameters": {"threshold": thr, "coordinate": [0.5, 0.5], "selection": "test partition, reference >= 10% of valid, Dice percentiles 20/40/60/80"},
          "checkpoint": str(ck)}
save_figure(fig, OUT, "Fig16_sicap_external.png", "Fig16.tif", record, WORK, __file__)
