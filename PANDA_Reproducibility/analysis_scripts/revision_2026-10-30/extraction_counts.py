"""Stored (extracted) training patches: totals, the per-WSI limit and the counts of the Fig 2 WSI.

Candidate selection has four stages (panda_final_pipeline.py):
  eligible  grid positions that pass the tissue, annotation and class rules
            (coordinate_pools/<WSI>_L<l>.stats.json: eligible_*_before_cap);
  pooled    at most coordinate_pool_cap_per_class_slide_level (32) per WSI,
            level and class, kept by seeded reservoir sampling, i.e. a uniform
            random sample of the eligible positions (build_coordinate_pool);
  stored    pooled candidates read at full resolution in order of decreasing
            tissue fraction, whose class is confirmed on the full-resolution
            mask, written to patches/ (materialize_from_pool; manifests/patches.csv);
  selected  the 80,000 patches chosen from the stored ones by K-means
            (selected/train_selected.csv; qc/05_select_final.json "available"
            equals the stored count of each bucket).
This script reads manifests/patches.csv of the pipeline work folder (not in
the repository, 205 MB) and writes the counts the manuscript uses.

Usage: python extraction_counts.py <work_root> <PANDA_Reproducibility>
Output: revision_2026-10-30/tables/Z2_extraction_counts.json
"""
import json
import sys
from pathlib import Path

import pandas as pd

WORK, REP = Path(sys.argv[1]), Path(sys.argv[2])
FIG2_WSI = "362d30ace92ce17128d176137362b8c2"
p = pd.read_csv(WORK / "manifests" / "patches.csv", usecols=["image_id", "split", "level", "category"])
t = p[p.split == "train"]
per = t.groupby(["image_id", "level", "category"]).size()
q = json.loads((REP / "qc" / "05_select_final.json").read_text())
avail = {(s["level"], s["category"]): s["available"] for s in q["details"]["selection"]}
totals = {f"L{l}_{c}": int(n) for (l, c), n in t.groupby(["level", "category"]).size().items()}
assert all(totals[f"L{l}_{c}"] == avail[(l, c)] for (l, c) in avail), "stored counts differ from 05_select_final.json"
fig2 = t[t.image_id == FIG2_WSI].groupby(["level", "category"]).size()
out = {"stored_train_patches_by_bucket": totals,
       "stored_per_wsi_level_class_max": int(per.max()),
       "stored_per_wsi_level_class_median": float(per.median()),
       "wsi_level_class_combinations": int(len(per)),
       "fig2_wsi_stored": {f"L{l}_{c}": int(n) for (l, c), n in fig2.items()},
       "source": "manifests/patches.csv of the pipeline work folder; totals equal qc/05_select_final.json 'available'"}
dst = REP / "revision_2026-10-30" / "tables" / "Z2_extraction_counts.json"
dst.write_text(json.dumps(out, indent=1), encoding="utf-8")
print(json.dumps(out, indent=1))
