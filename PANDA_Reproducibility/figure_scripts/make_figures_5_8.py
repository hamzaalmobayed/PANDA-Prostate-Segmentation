"""Figs 5 and 8 of the PANDA manuscript (numbering of the revision of 30 Oct 2026).

Fig 5  training dynamics from runs/final_main_training/history.csv (the figure
       before 28 Oct 2026 used values typed into
       GENERATE_TRAINING_DYNAMICS_CURVES.py that did not match history.csv at
       every epoch).
Fig 8  validation threshold sweep from
       paper_results/tables/threshold_sensitivity_validation.csv and the selected
       threshold in best_threshold.json. The earlier "optimal threshold region"
       band (0.40-0.45) had no stated rule and is not drawn.

Both are drawn with SciencePlots and the PLOS font settings
(provenance_log.plos_style). Each figure gets a JSON log with its source files,
SHA-256 and the values it shows (provenance_log.py).

Usage:
    python make_figures_5_8.py <output_dir> [work_root]
"""
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from provenance_log import plos_style, save_figure, source

OUT = Path(sys.argv[1])
if len(sys.argv) < 3 and "PANDA_DATA_ROOT" not in os.environ:
    sys.exit("give work_root as the second argument or set PANDA_DATA_ROOT")
WORK = Path(sys.argv[2] if len(sys.argv) > 2 else os.environ["PANDA_DATA_ROOT"])
TABLES = WORK / "paper_results" / "tables"
plos_style()


def save(fig, png, tif, record):
    save_figure(fig, OUT, png, tif, record, WORK, __file__)


# ---------------------------------------------------------------- Fig 5
def fig5():
    hist_path = WORK / "runs" / "final_main_training" / "history.csv"
    h = pd.read_csv(hist_path)
    ep = h.epoch.to_numpy()
    best = h.loc[h.val_dice.idxmax()]
    mloss = h.loc[h.val_loss.idxmin()]
    fig, ax1 = plt.subplots(figsize=(7.5, 3.6))
    ax2 = ax1.twinx()
    ax1.tick_params(axis="y", which="both", right=False)
    l1, = ax1.plot(ep, h.train_loss, color="#0C5DA5", marker="o", ms=3, label="Training loss")
    l2, = ax1.plot(ep, h.val_loss, color="#FF9500", marker="s", ms=3, ls="--", label="Validation loss")
    l3, = ax2.plot(ep, h.val_dice, color="#00B945", marker="^", ms=3.5, label="Validation Dice")
    ax1.set_xlim(0.5, len(ep) + 0.5); ax1.set_xticks(ep); ax1.minorticks_off()
    ax1.set_xlabel("Epoch")
    ax1.set_ylim(0.4, 1.3); ax1.set_yticks(np.arange(0.4, 1.21, 0.2))
    ax1.set_ylabel("Loss")
    ax2.set_ylim(0.78, 0.85); ax2.set_yticks(np.arange(0.78, 0.8401, 0.01))
    ax2.set_ylabel("Validation Dice")
    ax1.axvline(best.epoch, color="#FF9500", ls=":", lw=0.9)
    ax1.axvline(mloss.epoch, color="#845B97", ls=":", lw=0.9)
    # labels beside their marker line, right-aligned for the earlier epoch so that they cannot overlap
    first, second = sorted([best.epoch, mloss.epoch])
    for e, txt in ((best.epoch, f"Highest validation Dice\nepoch {int(best.epoch)}, Dice {best.val_dice:.4f}"),
                   (mloss.epoch, f"Lowest validation loss\nepoch {int(mloss.epoch)}, loss {mloss.val_loss:.4f}")):
        left = e == first and first != second
        ax1.annotate(txt, xy=(e, 1.28), xytext=(-4 if left else 4, 0), textcoords="offset points",
                     ha="right" if left else "left", va="top",
                     bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.5", lw=0.5))
    ax1.legend(handles=[l1, l2, l3], loc="upper left", frameon=False)
    save(fig, "Fig05_training_dynamics.png", "Fig5.tif", {
        "figure": "Fig 5",
        "sources": {"training_history": source(hist_path, WORK),
                    "run_config": source(WORK / "runs" / "final_main_training" / "config.json", WORK)},
        "values_shown": {"epochs": ep.tolist(), "train_loss": h.train_loss.round(6).tolist(),
                         "val_loss": h.val_loss.round(6).tolist(), "val_dice": h.val_dice.round(6).tolist(),
                         "best_dice": {"epoch": int(best.epoch), "val_dice": round(float(best.val_dice), 8)},
                         "min_val_loss": {"epoch": int(mloss.epoch), "val_loss": round(float(mloss.val_loss), 6)}},
        "parameters": {"val_dice": "training-time patch-level validation Dice (checkpoint selection), threshold 0.50"}})


# ---------------------------------------------------------------- Fig 8
def fig8():
    csv = TABLES / "threshold_sensitivity_validation.csv"
    best_path = TABLES / "best_threshold.json"
    t = pd.read_csv(csv)
    best = json.loads(best_path.read_text())
    thr = float(best["threshold"])
    sel = t.loc[(t.threshold - thr).abs().idxmin()]
    assert abs(sel.threshold - thr) < 1e-9 and t.dice.idxmax() == sel.name
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    colors = {"dice": "#0C5DA5", "precision": "#FF9500", "recall": "#00B945"}
    for k, lab in (("dice", "Dice"), ("precision", "Precision"), ("recall", "Recall")):
        ax.plot(t.threshold, t[k], color=colors[k], marker="o", ms=3, label=lab, zorder=3)
    ax.axvline(thr, color="0.3", ls="--", lw=0.9, label=f"Selected threshold ({thr:.2f})", zorder=2)
    # label positions chosen in empty areas of the plot
    offsets = {"recall": (0.04, 0.04), "precision": (-0.08, -0.09), "dice": (0.04, -0.09)}
    for k, (dx, dy) in offsets.items():
        ax.annotate(f"{sel[k]:.4f}", xy=(thr, sel[k]), xytext=(thr + dx, sel[k] + dy), color=colors[k],
                    va="center", bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=colors[k], lw=0.6),
                    arrowprops=dict(arrowstyle="-", color=colors[k], lw=0.6))
    ax.set_xlabel("Threshold"); ax.set_ylabel("Score")
    ax.set_xlim(0.285, 0.715); ax.set_ylim(0.60, 0.98)
    ax.set_xticks([0.30, 0.35, thr, 0.50, 0.55, 0.60, 0.65, 0.70])
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    save(fig, "Fig08_threshold_sweep.png", "Fig8.tif", {
        "figure": "Fig 8",
        "sources": {"threshold_sweep": source(csv, WORK), "best_threshold": source(best_path, WORK)},
        "values_shown": {"selected_threshold": thr, "dice": round(float(sel.dice), 6),
                         "precision": round(float(sel.precision), 6), "recall": round(float(sel.recall), 6),
                         "thresholds": t.threshold.round(2).tolist()},
        "parameters": {"map": "fused reconstruction, validation WSIs", "selection": "maximum validation Dice"}})


for f in (fig5, fig8):
    f()
