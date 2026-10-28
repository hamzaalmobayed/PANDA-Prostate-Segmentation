"""Regenerate Figs 6, 8, 9, 10 and 12 of the PANDA manuscript from the saved study exports.

Fig 6  sampling schematic; the group sizes are counted in selected/train_selected.csv.
Fig 8  training dynamics from runs/final_main_training/history.csv (the earlier
       figure used values typed into GENERATE_TRAINING_DYNAMICS_CURVES.py that
       did not match history.csv at every epoch).
Fig 9  grouped per-slide Dice from paper_results/tables/per_slide_wsi_metrics.csv,
       as in GENERATE_GROUPED_PER_SLIDE_DICE_FINAL_PLOS_V2.py.
Fig 10 patch-level confusion matrices from
       paper_results/tables/patch_metrics_per_level_and_pooled.csv (the earlier
       script typed the counts in; they are identical to this export).
Fig 12 validation threshold sweep from
       paper_results/tables/threshold_sensitivity_validation.csv and the selected
       threshold in best_threshold.json. The earlier "optimal threshold region"
       band (0.40-0.45) had no stated rule and is not drawn.

Each figure gets a JSON log with its source files, SHA-256 and the values it
shows (provenance_log.py).

Usage:
    python make_figures_6_8_9_10_12.py <output_dir> [work_root]
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
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from provenance_log import save_figure, source

OUT = Path(sys.argv[1])
if len(sys.argv) < 3 and "PANDA_DATA_ROOT" not in os.environ:
    sys.exit("give work_root as the second argument or set PANDA_DATA_ROOT")
WORK = Path(sys.argv[2] if len(sys.argv) > 2 else os.environ["PANDA_DATA_ROOT"])
TABLES = WORK / "paper_results" / "tables"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7})


def save(fig, png, tif, record):
    save_figure(fig, OUT, png, tif, record, WORK, __file__)


# ---------------------------------------------------------------- Fig 6
def fig6():
    sel_path = WORK / "selected" / "train_selected.csv"
    sel = pd.read_csv(sel_path, usecols=["image_id", "level", "category", "provider"])
    counts = sel.groupby(["level", "category"]).size()
    assert len(sel) == 80000 and len(counts) == 6
    lo, hi = int(counts.min()), int(counts.max())

    fig, ax = plt.subplots(figsize=(7.5, 4.65))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    def box(x, y, w, h, label, fill, edge, size):
        ax.add_patch(FancyBboxPatch((x, y - h / 2), w, h, boxstyle="round,pad=0.006,rounding_size=0.012",
                                    fc=fill, ec=edge, lw=0.95, zorder=3))
        ax.text(x + w / 2, y, label, ha="center", va="center", fontsize=size, fontweight="bold", color="#202735",
                zorder=4, linespacing=1.1, multialignment="center")

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=8, lw=0.8,
                                     fc="#344456", ec="#344456", shrinkA=0, shrinkB=0, zorder=2))

    X, W, H = (0.025, 0.283, 0.540, 0.804), (0.158, 0.154, 0.172, 0.177), 0.067
    leaf_y = [0.815, 0.710, 0.605, 0.500, 0.395, 0.290]
    level_y = [(leaf_y[i] + leaf_y[i + 1]) / 2 for i in (0, 2, 4)]
    root_y = (level_y[0] + level_y[2]) / 2
    ax.text(0.5, 0.967, "Structured sampling of PANDA training patches", ha="center", va="center", fontsize=11.5,
            fontweight="bold", color="#202735")
    for x, w, lab in zip(X, W, ("Data", "Pyramid level", "Reference class", "Realized group")):
        ax.text(x + w / 2, 0.903, lab, ha="center", va="center", fontsize=7.6, color="#333b45", fontweight="bold")
    box(X[0], root_y, W[0], 0.080, "PANDA", "#dce7fc", "#6785b7", 8.5)
    for i, (lvl, y) in enumerate(zip(("L0  (1x)", "L1  (4x)", "L2  (16x)"), level_y)):
        box(X[1], y, W[1], 0.071, lvl, "#ffefc4", "#cc9f45", 7.6)
        arrow(X[0] + W[0] + 0.006, root_y, X[1] - 0.009, y)
        for j, cls in enumerate(("Normal", "Tumor")):
            yy = leaf_y[i * 2 + j]
            fc, ec = ("#e4f2dd", "#7caf71") if cls == "Normal" else ("#ffdfdf", "#d78a8a")
            box(X[2], yy, W[2], H, cls, fc, ec, 7.6)
            arrow(X[1] + W[1] + 0.006, y, X[2] - 0.009, yy)
            box(X[3], yy, W[3], H, f"PANDA {lvl.split()[0]}\n{cls}", "#edf0f4", "#94a2b3", 7.1)
            arrow(X[2] + W[2] + 0.006, yy, X[3] - 0.009, yy)
    ax.text(0.5, 0.132, f"Each realized group contains {lo:,} or {hi:,} patches", ha="center", va="center",
            fontsize=7.9, fontweight="bold", color="#293647")
    fig.subplots_adjust(left=0.018, right=0.982, bottom=0.018, top=0.992)
    save(fig, "Fig06_sampling.png", "Fig6.tif", {
        "figure": "Fig 6",
        "sources": {"selected_manifest": source(sel_path, WORK)},
        "values_shown": {"group_sizes": {f"L{l}_{c}": int(n) for (l, c), n in counts.items()},
                         "provider_counts": {f"L{l}_{c}_{p}": int(n) for (l, c, p), n in
                                             sel.groupby(["level", "category", "provider"]).size().items()},
                         "contributing_training_wsis": int(sel.image_id.nunique())},
        "parameters": {"layout": "schematic of GENERATE_STRUCTURED_SAMPLING_TREE_4_LEVELS_V3.py at 7.5 in width"}})


# ---------------------------------------------------------------- Fig 8
def fig8():
    hist_path = WORK / "runs" / "final_main_training" / "history.csv"
    h = pd.read_csv(hist_path)
    ep = h.epoch.to_numpy()
    best = h.loc[h.val_dice.idxmax()]
    mloss = h.loc[h.val_loss.idxmin()]
    fig, ax1 = plt.subplots(figsize=(7.5, 4.3))
    ax2 = ax1.twinx()
    l1, = ax1.plot(ep, h.train_loss, color="#1148ff", marker="o", ms=3.4, lw=1.0, label="Training loss")
    l2, = ax1.plot(ep, h.val_loss, color="#ff7a00", marker="s", ms=3.2, lw=1.0, ls=(0, (6, 3)), label="Validation loss")
    l3, = ax2.plot(ep, h.val_dice, color="#0a7f13", marker="^", ms=3.8, lw=1.0, label="Validation Dice")
    ax1.set_xlim(0.5, len(ep) + 0.5); ax1.set_xticks(ep)
    ax1.set_xlabel("Epoch", fontsize=8, fontweight="bold")
    ax1.set_ylim(0.4, 1.2); ax1.set_yticks(np.arange(0.4, 1.21, 0.2))
    ax1.set_ylabel("Loss", color="#1148ff", fontsize=8.5, fontweight="bold")
    ax1.tick_params(axis="y", labelcolor="#1148ff")
    ax2.set_ylim(0.78, 0.84); ax2.set_yticks(np.arange(0.78, 0.8401, 0.01))
    ax2.set_ylabel("Validation Dice", color="#0a7f13", fontsize=8.5, fontweight="bold")
    ax2.tick_params(axis="y", labelcolor="#0a7f13")
    for y in np.linspace(0.4, 1.2, 9):
        ax1.axhline(y, color="#bfbfbf", ls="--", lw=0.5, alpha=0.6, zorder=0)
    ax1.axvline(best.epoch, color="#ff8c00", ls="--", lw=0.9)
    ax1.axvline(mloss.epoch, color="#b000ff", ls="--", lw=0.9)
    # boxes sit beside their marker line (right-aligned for the earlier epoch) so they cannot overlap
    first, second = sorted([best.epoch, mloss.epoch])
    for e, txt, fc, ec in ((best.epoch, f"Best model\nEpoch {int(best.epoch)}\nDice {best.val_dice:.4f}", "#fff4ea", "#ff8c00"),
                           (mloss.epoch, f"Min. validation loss\nEpoch {int(mloss.epoch)}\nLoss {mloss.val_loss:.4f}", "#f8ebff", "#b000ff")):
        left = e == first and first != second
        ax1.annotate(txt, xy=(e, 1.2), xycoords="data", xytext=(-3 if left else 3, 4), textcoords="offset points",
                     ha="right" if left else "left", va="bottom",
                     fontsize=6.6, fontweight="bold", bbox=dict(boxstyle="round,pad=0.3", fc=fc, ec=ec, lw=0.8),
                     annotation_clip=False)
    fig.legend(handles=[l1, l2, l3], loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=3, frameon=False, fontsize=7.4)
    fig.subplots_adjust(left=0.08, right=0.91, bottom=0.11, top=0.79)
    save(fig, "Fig08_training_dynamics.png", "Fig8.tif", {
        "figure": "Fig 8",
        "sources": {"training_history": source(hist_path, WORK),
                    "run_config": source(WORK / "runs" / "final_main_training" / "config.json", WORK)},
        "values_shown": {"epochs": ep.tolist(), "train_loss": h.train_loss.round(6).tolist(),
                         "val_loss": h.val_loss.round(6).tolist(), "val_dice": h.val_dice.round(6).tolist(),
                         "best_dice": {"epoch": int(best.epoch), "val_dice": round(float(best.val_dice), 8)},
                         "min_val_loss": {"epoch": int(mloss.epoch), "val_loss": round(float(mloss.val_loss), 6)}},
        "parameters": {"val_dice": "training-time patch-level validation Dice (checkpoint selection), threshold 0.50"}})


# ---------------------------------------------------------------- Fig 9
def fig9(n_groups=8):
    csv = TABLES / "per_slide_wsi_metrics.csv"
    df = pd.read_csv(csv)
    assert not df.image_id.duplicated().any()
    fusion_positive = (df.fusion_tp + df.fusion_fn) > 0
    assert len(df) == 1577 and int(fusion_positive.sum()) == 1126
    specs = [("L0", "L0 (1x)"), ("L1", "L1 (4x downsample)"), ("L2", "L2 (16x downsample)"), ("fusion", "Three-level fusion")]
    fig, axarr = plt.subplots(2, 2, figsize=(7.5, 4.5), sharey=True)
    fig.subplots_adjust(left=0.068, right=0.979, bottom=0.105, top=0.855, hspace=0.49, wspace=0.105)
    fig.suptitle("Per-slide Dice of tumor-bearing PANDA test WSIs", fontsize=10.5, fontweight="bold", y=0.975, x=0.45)
    fig.legend(handles=[Line2D([0], [0], color="#1689e9", lw=1.15, ls="--", label="Overall mean"),
                        Line2D([0], [0], color="#e66d43", lw=1.15, ls=":", label="Overall median")],
               loc="upper right", bbox_to_anchor=(0.985, 0.995), fontsize=6.8, frameon=True, fancybox=False,
               facecolor="white", edgecolor="#999999", labelspacing=0.36, handlelength=2.5, borderpad=0.52)
    shown = {}
    for ax, (key, title) in zip(axarr.flat, specs):
        tp, fn = df[f"{key}_tp"], df[f"{key}_fn"]
        selected = fusion_positive & ((tp + fn) > 0)
        vals = df.loc[selected, f"{key}_dice"].to_numpy(float)
        assert np.isfinite(vals).all() and ((vals >= 0) & (vals <= 1)).all()
        groups = np.array_split(vals * 100, n_groups)
        ax.boxplot(groups, widths=0.59, patch_artist=True, whis=1.5, showfliers=True,
                   boxprops=dict(facecolor="#f4f4f4", edgecolor="#222222", linewidth=0.7),
                   whiskerprops=dict(color="#333333", linewidth=0.65), capprops=dict(color="#333333", linewidth=0.65),
                   medianprops=dict(color="#333333", linewidth=0.7),
                   flierprops=dict(marker="o", markersize=1.8, markeredgewidth=0.5, markerfacecolor="white", markeredgecolor="#333333"))
        mean, median = float(np.mean(vals)), float(np.median(vals))
        ax.axhline(mean * 100, color="#1689e9", ls="--", lw=1.05, zorder=3)
        ax.axhline(median * 100, color="#e66d43", ls=":", lw=1.05, zorder=3)
        ax.set_xlim(0.45, n_groups + 0.55); ax.set_ylim(0, 102)
        ax.set_title(f"{title}  (n={len(vals):,})", fontsize=8.35, fontweight="bold", pad=4)
        ax.set_xticks(range(1, n_groups + 1)); ax.set_xticklabels([str(i) for i in range(1, n_groups + 1)], fontsize=6.4)
        ax.set_yticks([0, 20, 40, 60, 80, 100]); ax.tick_params(axis="y", labelsize=6.4, length=2)
        ax.set_xlabel("Consecutive WSI group (test-set order)", fontsize=6.7, labelpad=2.5)
        if key in ("L0", "L2"):
            ax.set_ylabel("Per-slide Dice (%)", fontsize=6.8)
        ax.grid(axis="y", alpha=0.13, lw=0.45)
        ax.text(0.985, 0.965, f"Mean {mean * 100:.2f}%\nMedian {median * 100:.2f}%", transform=ax.transAxes, ha="right",
                va="top", fontsize=6.25, bbox={"facecolor": "white", "edgecolor": "#aaaaaa", "linewidth": 0.35, "pad": 1.3},
                zorder=7)
        shown[key] = {"n": int(len(vals)), "mean_dice": round(mean, 6), "median_dice": round(median, 6),
                      "group_sizes": [len(g) for g in groups],
                      "excluded_reference_empty_at_level": int((fusion_positive & ~((tp + fn) > 0)).sum())}
    save(fig, "Fig09_per_slide_dice.png", "Fig9.tif", {
        "figure": "Fig 9",
        "sources": {"per_slide_metrics": source(csv, WORK)},
        "values_shown": shown,
        "parameters": {"wsis": "fused-reference-tumor WSIs with reference tumor at the level (TP + FN > 0); no zero-coded Dice",
                       "grouping": f"{n_groups} consecutive groups in the order of the test CSV (np.array_split)",
                       "boxplot": "box Q1-Q3, line median, whiskers 1.5 x IQR, points beyond the whiskers",
                       "threshold": 0.43}})


# ---------------------------------------------------------------- Fig 10
def fig10():
    csv = TABLES / "patch_metrics_per_level_and_pooled.csv"
    t = pd.read_csv(csv).set_index("subset")
    specs = [("(A)", "Pooled across all levels", "all_levels_pooled", 1), ("(B)", "L0 (1×)", "L0", 1),
             ("(C)", "L1 (4× downsample)", "L1", 1), ("(D)", "L2 (16× downsample)", "L2", 2)]
    pooled = t.loc[["L0", "L1", "L2"], ["tp", "fp", "tn", "fn"]].sum()
    assert (pooled == t.loc["all_levels_pooled", ["tp", "fp", "tn", "fn"]]).all()
    cmap = ListedColormap(["#174e82", "#e6f1fa", "#e8f2fb", "#c1dcf6"])
    fig, axes = plt.subplots(2, 2, figsize=(7.5, 5.6))
    fig.suptitle("Per-level and pooled diagnostic confusion matrices", fontsize=10, fontweight="bold", y=0.985)
    shown = {}
    for ax, (panel, title, key, dec) in zip(axes.flat, specs):
        r = t.loc[key]
        counts = np.array([[r.tn, r.fp], [r.fn, r.tp]], dtype=np.int64)
        ax.imshow(np.array([[0, 1], [2, 3]]), cmap=cmap, vmin=0, vmax=3, interpolation="nearest")
        ax.set_title(title, fontsize=7.6, fontweight="bold", pad=5)
        total = int(counts.sum())
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{int(counts[i, j]):,}\n{100 * counts[i, j] / total:.{dec}f}%", ha="center", va="center",
                        color="white" if (i, j) == (0, 0) else "black", fontsize=6.2, linespacing=1.35)
        ax.set_xticks([0, 1], ["Normal", "Tumor"], fontsize=6.4); ax.set_yticks([0, 1], ["Normal", "Tumor"], fontsize=6.4)
        ax.set_xlabel("Predicted", fontsize=6.8, labelpad=2); ax.set_ylabel("Actual", fontsize=6.8, labelpad=3)
        for sp in ax.spines.values():
            sp.set_linewidth(0.8)
        ax.text(-0.25, 1.035, panel, transform=ax.transAxes, ha="right", va="bottom", fontsize=9, fontweight="bold")
        shown[key] = {"tn": int(r.tn), "fp": int(r.fp), "fn": int(r.fn), "tp": int(r.tp)}
    fig.subplots_adjust(left=0.1, right=0.98, bottom=0.08, top=0.9, wspace=0.3, hspace=0.38)
    save(fig, "Fig10_confusion.png", "Fig10.tif", {
        "figure": "Fig 10",
        "sources": {"patch_metrics": source(csv, WORK)},
        "values_shown": shown,
        "parameters": {"cell_percent": "share of the panel's four counts; 2 decimals for L2, 1 elsewhere",
                       "pooled": "sum of the L0, L1 and L2 counts (checked)"}})


# ---------------------------------------------------------------- Fig 12
def fig12():
    csv = TABLES / "threshold_sensitivity_validation.csv"
    best_path = TABLES / "best_threshold.json"
    t = pd.read_csv(csv)
    best = json.loads(best_path.read_text())
    thr = float(best["threshold"])
    sel = t.loc[(t.threshold - thr).abs().idxmin()]
    assert abs(sel.threshold - thr) < 1e-9 and t.dice.idxmax() == sel.name
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    colors = {"dice": "#1f77b4", "precision": "#ff7f0e", "recall": "#2ca02c"}
    for k, lab in (("dice", "Dice"), ("precision", "Precision"), ("recall", "Recall")):
        ax.plot(t.threshold, t[k], color=colors[k], marker="o", ms=3.0, lw=1.3, label=lab, zorder=3)
    ax.axvline(thr, color="#1f77b4", ls="--", lw=1.0, label=f"Selected threshold ({thr:.2f})", zorder=2)
    offsets = {"recall": (0.045, 0.055), "precision": (0.045, 0.012), "dice": (0.035, -0.05)}
    for k, (dx, dy) in offsets.items():
        ax.annotate(f"{sel[k]:.5f}", xy=(thr, sel[k]), xytext=(thr + dx, sel[k] + dy), color=colors[k], fontsize=7,
                    fontweight="bold", bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=colors[k], lw=0.7),
                    arrowprops=dict(arrowstyle="-", color=colors[k], lw=0.7))
    ax.set_title("Validation threshold sensitivity", fontsize=9.5)
    ax.set_xlabel("Threshold", fontsize=8); ax.set_ylabel("Score", fontsize=8)
    ax.set_xlim(0.285, 0.715); ax.set_ylim(0.60, 0.98)
    ax.set_xticks([0.30, 0.35, thr, 0.50, 0.55, 0.60, 0.65, 0.70])
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=6.8, frameon=True)
    fig.subplots_adjust(left=0.09, right=0.75, top=0.92, bottom=0.11)
    save(fig, "Fig12_threshold_sweep.png", "Fig12.tif", {
        "figure": "Fig 12",
        "sources": {"threshold_sweep": source(csv, WORK), "best_threshold": source(best_path, WORK)},
        "values_shown": {"selected_threshold": thr, "dice": round(float(sel.dice), 6),
                         "precision": round(float(sel.precision), 6), "recall": round(float(sel.recall), 6),
                         "thresholds": t.threshold.round(2).tolist()},
        "parameters": {"map": "fused reconstruction, validation WSIs", "selection": "maximum validation Dice"}})


for f in (fig6, fig8, fig9, fig10, fig12):
    f()
