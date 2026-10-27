"""Share of stored test tumor patches in which every annotated-valid pixel is tumor.

tumor_fraction in manifests/patches.csv is tumor pixels / annotated-valid
pixels of the patch. A value of 1 means the reference labels the whole
annotated area of the patch as tumor, stroma included.
"""
import os
import sys
from pathlib import Path
import pandas as pd

if len(sys.argv) < 3 and "PANDA_DATA_ROOT" not in os.environ:
    sys.exit("usage: provider_tumor_definition.py <output_dir> [work_root]  (or set PANDA_DATA_ROOT)")
WORK = Path(sys.argv[2] if len(sys.argv) > 2 else os.environ["PANDA_DATA_ROOT"])
OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)

cols = ["provider", "split", "level", "category", "tumor_fraction"]
p = pd.concat(c[(c.split == "test") & (c.category == "tumor")]
              for c in pd.read_csv(WORK / "manifests" / "patches.csv", usecols=cols, chunksize=200_000))
p["all_tumor"] = p.tumor_fraction >= 1.0
t = p.groupby(["provider", "level"]).all_tumor.agg(tumor_patches="size", all_tumor="sum").reset_index()
t["share_all_tumor"] = (t.all_tumor / t.tumor_patches).round(4)
t.to_csv(OUT / "provider_tumor_definition_test_patches.csv", index=False)
print(t.to_string(index=False))
print(p.groupby("provider").all_tumor.agg(["sum", "size"]).to_string())
