"""Fig 6 of the revised manuscript (30 Oct 2026): per-slide Dice of tumor-bearing PANDA test WSIs.

(A) Per-slide Dice of L0, L1, L2 and fusion on the common scoring region
    (annotated pixels with fused support; unsupported single-level pixels count
    as negative), one box per map.
(B) Fusion against L1 for each WSI, coloured by provider.

Source: revision_2026-10-30/tables/cache_per_slide_stats_test.csv (cache_stats.py).
Drawn with SciencePlots and the PLOS font settings (provenance_log.plos_style).

Usage: python make_figure_6.py <output_dir> <PANDA_Reproducibility>
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from provenance_log import plos_style, save_figure, source

OUT, WORK = Path(sys.argv[1]), Path(sys.argv[2])
plos_style()
SRC = WORK / "revision_2026-10-30" / "tables" / "cache_per_slide_stats_test.csv"
d = pd.read_csv(SRC)
d = d[d.reference_tumor_pixels_common > 0].copy()


def dice(m):
    tp, fp, fn = d[f"{m}_common_tp"], d[f"{m}_common_fp"], d[f"{m}_common_fn"]
    return (2 * tp / (2 * tp + fp + fn)).to_numpy()


maps = ["L0", "L1", "L2", "fusion"]
vals = {m: dice(m) for m in maps}
fig, (a, b) = plt.subplots(1, 2, figsize=(7.5, 3.2), gridspec_kw={"width_ratios": [1.15, 1]})
bp = a.boxplot([vals[m] for m in maps], tick_labels=["L0", "L1", "L2", "Fusion"], widths=0.55, showfliers=True,
               flierprops={"marker": ".", "markersize": 2, "alpha": 0.4}, medianprops={"color": "black"})
a.set_ylabel("Per-slide Dice")
a.set_ylim(-0.02, 1.02)
a.text(-0.16, 1.0, "A", transform=a.transAxes, fontsize=10, fontweight="bold", va="top")
colors = {"karolinska": "#1f77b4", "radboud": "#d62728"}
for prov, g in d.groupby("provider"):
    sel = d.provider == prov
    b.scatter(vals["L1"][sel.to_numpy()], vals["fusion"][sel.to_numpy()], s=3, alpha=0.5, color=colors[prov],
              label=f"{prov.capitalize()} ($n$={int(sel.sum())})", rasterized=True)
b.plot([0, 1], [0, 1], color="0.4", lw=0.8, ls="--")
b.set_xlabel("L1 per-slide Dice"); b.set_ylabel("Fusion per-slide Dice")
b.set_xlim(-0.02, 1.02); b.set_ylim(-0.02, 1.02); b.set_aspect("equal")
b.legend(loc="lower right", frameon=False, markerscale=3)
b.text(-0.22, 1.0, "B", transform=b.transAxes, fontsize=10, fontweight="bold", va="top")
record = {"figure": "Fig 6", "sources": [source(SRC, WORK)],
          "parameters": {"threshold": 0.43, "scoring_region": "annotated & fused support; unsupported single-level pixels negative"},
          "values": {f"median_{m}": float(np.median(vals[m])) for m in maps} | {"n_wsi": int(len(d))}}
save_figure(fig, OUT, "Fig06_per_slide_dice.png", "Fig6.tif", record, WORK, __file__)
