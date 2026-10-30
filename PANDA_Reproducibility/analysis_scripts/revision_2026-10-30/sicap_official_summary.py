"""Summary tables of the SICAPv2 official-release evaluation (sicapv2_official.py).

Inputs (tables written by the evaluation): official_index.csv, per_tile_metrics.csv,
per_region_dedup_counts.csv. Writes to <output_dir>:

  SICAP_official_summary.csv   pooled metrics per patch set (test partition, listed, unlisted, all),
                               per patch and deduplicated, with 95% CIs by patient and by WSI
                               (10,000 resamples, percentile)
  SICAP_official_coords.csv    constant versus region-derived coordinates
  SICAP_official_facts.json    counts and checks quoted in the text (including whether any listed
                               non-cancerous patch comes from a slide with a cancer grade)

Usage: python sicap_official_summary.py <results_tables_dir> <SICAPv2 folder> <output_dir>
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

B = int(os.environ.get("SICAP_B", "10000"))
C4 = ("tp", "fp", "tn", "fn")


def m(c):
    tp, fp, tn, fn = (float(x) for x in c)
    d = lambda a, b: a / b if b else float("nan")
    return {"dice": d(2 * tp, 2 * tp + fp + fn), "iou": d(tp, tp + fp + fn), "precision": d(tp, tp + fp),
            "recall": d(tp, tp + fn), "specificity": d(tn, tn + fp), "accuracy": d(tp + tn, tp + tn + fp + fn)}


def dice(c):
    return 2 * c[..., 0] / np.maximum(2 * c[..., 0] + c[..., 1] + c[..., 3], 1)


def ci(counts_by_unit, seed):
    rng = np.random.default_rng(seed)
    n = len(counts_by_unit)
    v = np.array([dice(counts_by_unit[rng.integers(0, n, n)].sum(0)) for _ in range(B)])
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def main():
    T, root, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    idx = pd.read_csv(T / "official_index.csv")
    tile = pd.read_csv(T / "per_tile_metrics.csv")
    reg = pd.read_csv(T / "per_region_dedup_counts.csv")
    tile["unlisted"] = ~tile.partition_listed
    sets = {"test_partition": tile.test_partition, "partition_listed": tile.partition_listed,
            "unlisted": tile.unlisted, "all": np.ones(len(tile), bool)}
    rows = []
    for name, sel in sets.items():
        t = tile[sel]
        for mode in ("const", "region"):
            cols = [f"{mode}_{c}" for c in C4]
            tot = t[cols].sum().to_numpy()
            r = {"set": name, "unit_scored": "patch", "coords": mode, "patches": int(len(t)), "wsis": int(t.slide.nunique()),
                 "patients": int(t.patient_id.nunique()), **m(tot)}
            for unit, key, seed in (("patient", "patient_id", 42), ("wsi", "slide", 43)):
                g = t.groupby(key)[cols].sum().to_numpy(np.int64)
                r[f"ci_{unit}_low"], r[f"ci_{unit}_high"] = ci(g, seed) if mode == "const" else (np.nan, np.nan)
            rows.append(r)
        if name in ("test_partition", "partition_listed", "all"):
            for mode in ("const", "region"):
                d = reg[(reg.subset == name) & (reg["mode"] == mode)]
                tot = d[list(C4)].sum().to_numpy()
                r = {"set": name, "unit_scored": "deduplicated pixel", "coords": mode, "patches": int(d.n_tiles.sum()),
                     "wsis": int(d.slide.nunique()), "patients": int(d.patient_id.nunique()), **m(tot)}
                for unit, key, seed in (("patient", "patient_id", 42), ("wsi", "slide", 43)):
                    g = d.groupby(key)[list(C4)].sum().to_numpy(np.int64)
                    r[f"ci_{unit}_low"], r[f"ci_{unit}_high"] = ci(g, seed) if mode == "const" else (np.nan, np.nan)
                rows.append(r)
    s = pd.DataFrame(rows)
    s.to_csv(out / "SICAP_official_summary.csv", index=False)

    labels = pd.read_excel(root / "wsi_labels.xlsx").rename(columns={"slide_id": "slide"})
    part = pd.concat([pd.read_excel(root / "partition/Test/Test.xlsx"), pd.read_excel(root / "partition/Test/Train.xlsx")])
    part["slide"] = part.image_name.str.split("_").str[0]
    part = part.merge(labels, on="slide", how="left")
    nc_from_cancer = int(((part.NC == 1) & (part.Gleason_primary > 0)).sum())
    unl = tile[tile.unlisted].merge(labels, on="slide", how="left")
    fp_all = tile.const_fp.sum()
    facts = {
        "tiles_all": int(len(idx)), "slides_all": int(idx.slide.nunique()), "patients_all": int(idx.patient_id.nunique()),
        "tiles_test": int(idx.test_partition.sum()), "slides_test": int(idx[idx.test_partition].slide.nunique()),
        "patients_test": int(idx[idx.test_partition].patient_id.nunique()),
        "tiles_listed": int(idx.partition_listed.sum()), "tiles_unlisted": int((~idx.partition_listed).sum()),
        "listed_noncancerous_from_slides_with_cancer_grade": nc_from_cancer,
        "unlisted_from_slides_with_cancer_grade": int((unl.Gleason_primary > 0).sum()),
        "unlisted_total": int(len(unl)),
        "unlisted_share_of_fp_pixels": float(tile[tile.unlisted].const_fp.sum() / fp_all) if fp_all else float("nan"),
        "unlisted_share_of_valid_pixels": float(tile[tile.unlisted][[f"const_{c}" for c in C4]].sum().sum() /
                                                tile[[f"const_{c}" for c in C4]].sum().sum()),
        "boundary_test_partition": {k: float(tile[tile.test_partition][f"const_{k}"].mean()) for k in ("hd95_eval_px", "assd_eval_px", "surface_dice")},
        "resamples": B}
    json.dump(facts, open(out / "SICAP_official_facts.json", "w"), indent=2)
    print(s.round(4).to_string(index=False)); print(json.dumps(facts, indent=1))


if __name__ == "__main__":
    main()
