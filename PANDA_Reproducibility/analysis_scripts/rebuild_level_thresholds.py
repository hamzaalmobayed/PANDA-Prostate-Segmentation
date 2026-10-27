"""Rebuild the level-specific threshold analysis (Table 9) from the saved WSI caches.

Each cache (paper_results/wsi_cache/<split>_<image_id>_ds16_validsupport_v2.npz)
holds the float16 maps that the original evaluation read back before scoring.
Scoring follows panda_stream_paper.py: maps are cast to float32 before
thresholding; a single level is scored on annotated_valid & support_L<l>, the
fusion map on valid. The grid is the original 0.30-0.70 in steps of 0.01.

Two checks against archived outputs are printed: the fused validation sweep
(threshold_sensitivity_validation.csv) and the per-slide test counts at the
common threshold (per_slide_wsi_metrics.csv).
"""
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd

THRESHOLDS = np.round(np.arange(0.30, 0.701, 0.01), 2)
COMMON = 0.43
MAPS = ("L0", "L1", "L2", "fusion")


def slide_counts(path):
    """tp, fp, tn, fn for every map and threshold of one WSI: array (map, threshold, 4)."""
    z = np.load(path)
    gt = z["gt"].astype(bool)
    thr = THRESHOLDS.astype(np.float32)
    out = []
    for m in MAPS:
        if m == "fusion":
            valid, prob = z["valid"].astype(bool), z["fusion"]
        else:
            valid = z["annotated_valid"].astype(bool) & z[f"support_{m}"].astype(bool)
            prob = z[f"prob_{m}"]
        p = prob.astype(np.float32)
        pos, neg = np.sort(p[valid & gt]), np.sort(p[valid & ~gt])
        tp = len(pos) - np.searchsorted(pos, thr, "left")
        fp = len(neg) - np.searchsorted(neg, thr, "left")
        out.append(np.stack([tp, fp, len(neg) - fp, len(pos) - tp], 1))
    return np.stack(out).astype(np.int64)


def scores(c):
    tp, fp, tn, fn = (int(v) for v in c)
    div = lambda a, b: a / b if b else 0.0
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "dice": div(2 * tp, 2 * tp + fp + fn),
            "iou": div(tp, tp + fp + fn), "precision": div(tp, tp + fp), "recall": div(tp, tp + fn)}


def main():
    if len(sys.argv) < 3 and "PANDA_DATA_ROOT" not in os.environ:
        sys.exit("usage: rebuild_level_thresholds.py <output_dir> [work_root]  (or set PANDA_DATA_ROOT)")
    work = Path(sys.argv[2] if len(sys.argv) > 2 else os.environ["PANDA_DATA_ROOT"])
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    tables = work / "paper_results" / "tables"
    split = pd.read_csv(tables / "split_ids_reproducibility.csv")

    counts = {}
    for s in ("val", "test"):
        ids = split.image_id[split.split == s].tolist()
        paths = [work / "paper_results" / "wsi_cache" / f"{s}_{i}_ds16_validsupport_v2.npz" for i in ids]
        missing = [p.name for p in paths if not p.exists()]
        if missing:
            sys.exit(f"{len(missing)} {s} caches missing, e.g. {missing[0]}")
        with ProcessPoolExecutor(max_workers=6) as ex:
            counts[s] = (ids, np.stack(list(ex.map(slide_counts, paths, chunksize=8))))
        print(f"{s}: {len(ids)} WSIs scored", flush=True)

    # pooled validation sweep for each map
    val = counts["val"][1].sum(0)
    sweep = pd.DataFrame([{"map": m, "threshold": float(t), **scores(val[i, j])}
                          for i, m in enumerate(MAPS) for j, t in enumerate(THRESHOLDS)])
    sweep.to_csv(out / "level_specific_threshold_sweep_validation.csv", index=False)

    # Table 9: validation-selected threshold per map, applied unchanged to test
    test = counts["test"][1].sum(0)
    j43 = int(np.argmin(abs(THRESHOLDS - COMMON)))
    rows = []
    for i, m in enumerate(MAPS):
        best = sweep[sweep["map"] == m].reset_index(drop=True)["dice"].idxmax()
        at_best, at_common = scores(test[i, best]), scores(test[i, j43])
        rows.append({"map": m, "val_threshold": float(THRESHOLDS[best]), "val_dice": sweep[sweep["map"] == m]["dice"].iloc[best],
                     **{f"test_{k}": v for k, v in at_best.items()},
                     "test_dice_at_0.43": at_common["dice"],
                     "test_dice_change_pp": 100 * (at_best["dice"] - at_common["dice"])})
    table9 = pd.DataFrame(rows)
    table9.to_csv(out / "level_specific_thresholds_table9.csv", index=False)
    print(table9[["map", "val_threshold", "val_dice", "test_dice", "test_iou", "test_precision",
                  "test_recall", "test_dice_change_pp"]].round(4).to_string(index=False))

    # check 1: fused validation sweep against the archived export
    arch = pd.read_csv(tables / "threshold_sensitivity_validation.csv")
    mine = sweep[sweep["map"] == "fusion"].reset_index(drop=True)
    same = all((arch[k].to_numpy() == mine[k].to_numpy()).all() for k in ("tp", "fp", "tn", "fn"))
    print("fused validation sweep identical to threshold_sensitivity_validation.csv:", same)

    # check 2: per-slide test counts at 0.43 against per_slide_wsi_metrics.csv
    per = pd.read_csv(tables / "per_slide_wsi_metrics.csv").set_index("image_id").loc[counts["test"][0]]
    for i, m in enumerate(MAPS):
        mine = counts["test"][1][:, i, j43, :]
        arch = per[[f"{m}_{k}" for k in ("tp", "fp", "tn", "fn")]].to_numpy()
        print(f"test {m} at 0.43: {int((mine == arch).all(1).sum())}/{len(mine)} WSIs identical to per_slide_wsi_metrics.csv")


if __name__ == "__main__":
    main()
