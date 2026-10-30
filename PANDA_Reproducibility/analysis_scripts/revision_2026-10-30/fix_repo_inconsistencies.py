"""Correct the repository inconsistencies listed in PANDA_number_check.pdf, section 4.

1. paper_results/tables/pilot_leave_one_provider_out.csv gave, for each column,
   the best value over all epochs. It is rebuilt from each run's history.csv at
   the selected epoch (highest validation Dice), which is what the manuscript
   reports (Table 11). The old file is kept as *_best_per_column_OLD.csv.
2. manifests/excluded_missing_masks.csv lists the 100 images without a label
   mask in the PANDA release. manifests/excluded_pairs_all.csv now lists all
   104 excluded images with the reason, checked against the Kaggle archive:
   the other 4 have a mask in the release, but their image file was missing
   from the local copy of the data used by the pipeline.
3. external_precise_wsi/results/tables/per_slide_wsi_metrics.csv ended with an
   empty row, and external_global_metrics.json counted 28 WSIs. The empty row
   is removed and the JSON is rebuilt with the same bootstrap as
   run_external_precise_wsi_test.py (default_rng(42), 1,000 resamples,
   percentile) on the 27 WSIs. This reproduces the manuscript's 0.5140-0.7021;
   the old JSON (0.5149-0.6983, 28 rows) is kept as *_28rows_OLD.json.
4. external_sicapv2/results/tables/global_external_metrics.json called the
   evaluated set "official-test"; the manifest says all_images_fallback. The
   scope label is corrected; the numbers are unchanged.

Usage: python fix_repo_inconsistencies.py <PANDA_Reproducibility> <prostate-cancer-grade-assessment.zip> <log.json>
"""
import json
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


def metrics(tp, fp, tn, fn):
    eps = 1e-9
    return {"tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
            "dice": float(2 * tp / (2 * tp + fp + fn + eps)), "iou": float(tp / (tp + fp + fn + eps)),
            "precision": float(tp / (tp + fp + eps)), "recall": float(tp / (tp + fn + eps)),
            "specificity": float(tn / (tn + fp + eps)), "accuracy": float((tp + tn) / (tp + tn + fp + fn + eps))}


def main():
    root, kaggle_zip, log_path = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    log = {}

    # 1. provider-transfer summary at the selected epoch
    t = root / "paper_results/tables/pilot_leave_one_provider_out.csv"
    old = pd.read_csv(t)
    backup = t.with_name("pilot_leave_one_provider_out_best_per_column_OLD.csv")
    if not backup.exists():
        shutil.copy2(t, backup)
        old = pd.read_csv(backup)
    rows = []
    for _, r in old.iterrows():
        h = pd.read_csv(root / "paper_results/experiments" / r.experiment / "history.csv")
        b = h.loc[h.val_dice.idxmax()]
        rows.append({**{k: r[k] for k in ("experiment", "n_train", "n_val", "normalization", "use_coords", "encoder", "held_out_provider")},
                     "selected_epoch": int(b.epoch), "val_dice": b.val_dice, "val_iou": b.val_iou,
                     "val_precision": b.val_precision, "val_recall": b.val_recall,
                     "note": "target-provider patches drawn from the training partition; values at the selected epoch"})
    pd.DataFrame(rows).to_csv(t, index=False)
    log["pilot_leave_one_provider_out"] = rows

    # 2. all excluded image-mask pairs with the reason
    a = pd.read_csv(root / "manifests/audit.csv")
    bad = a[a.valid != True].copy()
    names = set(zipfile.ZipFile(kaggle_zip).namelist())
    bad["image_in_release"] = [f"train_images/{i}.tiff" in names for i in bad.image_id]
    bad["mask_in_release"] = [f"train_label_masks/{i}_mask.tiff" in names for i in bad.image_id]
    bad["reason"] = np.where(~bad.mask_in_release, "no label mask in the PANDA release",
                             "image file missing from the local data copy; image and mask are in the PANDA release")
    bad[["image_id", "provider", "isup_grade", "gleason_score", "image_in_release", "mask_in_release", "reason"]].to_csv(
        root / "manifests/excluded_pairs_all.csv", index=False)
    log["excluded_pairs"] = bad.groupby(["provider", "reason"]).size().reset_index(name="n").to_dict("records")

    # 3. PRECISE per-slide table and interval on 27 WSIs
    p = root / "external_precise_wsi/results/tables/per_slide_wsi_metrics.csv"
    ps = pd.read_csv(p)
    ps = ps.dropna(subset=["image_id"]).reset_index(drop=True)
    ps.to_csv(p, index=False)
    g = root / "external_precise_wsi/results/tables/external_global_metrics.json"
    gold = json.load(open(g))
    gb = g.with_name("external_global_metrics_28rows_OLD.json")
    if not gb.exists():
        shutil.copy2(g, gb)
    rng = np.random.default_rng(42)
    keys = ["dice", "iou", "precision", "recall", "specificity", "accuracy"]
    vals = {k: [] for k in keys}
    for _ in range(1000):
        b = ps.iloc[rng.integers(0, len(ps), size=len(ps))]
        c = metrics(int(b.fusion_tp.sum()), int(b.fusion_fp.sum()), int(b.fusion_tn.sum()), int(b.fusion_fn.sum()))
        for k in keys:
            vals[k].append(c[k])
    gnew = dict(gold)
    gnew["n_wsi"] = int(len(ps))
    gnew["bootstrap_95CI"] = {k: {"low": float(np.percentile(v, 2.5)), "high": float(np.percentile(v, 97.5))} for k, v in vals.items()}
    gnew["bootstrap_note"] = ("Rebuilt on 30 Oct 2026 on the 27 WSIs (the earlier file resampled 28 rows, including an "
                              "empty trailing row); default_rng(42), 1,000 resamples, percentile, as in run_external_precise_wsi_test.py.")
    json.dump(gnew, open(g, "w"), indent=2)
    log["precise_ci"] = {"old": gold["bootstrap_95CI"]["dice"], "new": gnew["bootstrap_95CI"]["dice"], "n_wsi": int(len(ps))}

    # 4. SICAPv2 scope label
    s = root / "external_sicapv2/results/tables/global_external_metrics.json"
    sj = json.load(open(s))
    log["sicap_scope_old"] = sj.get("metric_scope")
    sj["metric_scope"] = ("all 18,765 images of the Train folder of a renamed Kaggle copy of SICAPv2 "
                          "(split_source all_images_fallback); not the official test partition; pixels pooled over valid tissue")
    json.dump(sj, open(s, "w"), indent=2)
    log["sicap_scope_new"] = sj["metric_scope"]

    json.dump(log, open(log_path, "w"), indent=2, default=str)
    print(json.dumps(log, indent=1, default=str)[:3000])


if __name__ == "__main__":
    main()
