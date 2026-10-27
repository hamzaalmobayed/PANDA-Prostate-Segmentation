"""Rebuild coverage_analysis/ from the saved WSI caches.

Coverage of a WSI = annotated-valid pixels that also have fused support,
divided by all annotated-valid pixels, on the 16x evaluation grid. The
per-level columns use support_L<l> in place of fusion_support.

PANDA: every .npz in paper_results/wsi_cache (validation and test, including
the 64 validation caches saved as _v3). PRECISE: every .npz in
external_precise_wsi/wsi_cache (sub-24 and sub-25 are saved as _v2_resume).
"""
import json
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd

COHORTS = {"PANDA": ("paper_results/wsi_cache", "*.npz"),
           "PRECISE": ("external_precise_wsi/wsi_cache", "*.npz")}


def one(path, cohort):
    z = np.load(path)
    av = z["annotated_valid"].astype(bool)
    fs = z["fusion_support"].astype(bool)
    n_av, n_sup = int(av.sum()), int((av & fs).sum())
    frac = lambda s: s / n_av if n_av else np.nan   # WSIs without annotated-valid pixels stay empty
    r = {"cohort": cohort, "file": path.name, "annotated_valid_pixels": n_av,
         "fusion_support_pixels": int(fs.sum()), "supported_valid_pixels": n_sup,
         "stored_valid_pixels": int(z["valid"].astype(bool).sum()),
         "coverage_fraction": frac(n_sup), "coverage_percent": 100 * frac(n_sup)}
    for lv in ("L0", "L1", "L2"):
        s = int((av & z[f"support_{lv}"].astype(bool)).sum())
        r[f"{lv}_supported_valid_pixels"] = s
        r[f"{lv}_coverage_fraction"] = frac(s)
    return r


def summary(df, cohort, n_total):
    pct = df.coverage_percent
    av = int(df.annotated_valid_pixels.sum())
    s = {"cohort": cohort, "n_files_total": n_total, "n_files_analyzed": len(df),
         "pooled_annotated_valid_pixels": av,
         "pooled_supported_valid_pixels": int(df.supported_valid_pixels.sum()),
         "pooled_coverage_fraction": df.supported_valid_pixels.sum() / av,
         "pooled_coverage_percent": 100 * df.supported_valid_pixels.sum() / av,
         "mean_per_wsi_coverage_percent": pct.mean(), "median_per_wsi_coverage_percent": pct.median(),
         "std_per_wsi_coverage_percent": pct.std(), "min_per_wsi_coverage_percent": pct.min(),
         "max_per_wsi_coverage_percent": pct.max()}
    for cut in (100, 99, 95, 90):
        s[f"n_coverage_below_{cut}"] = int((pct < cut).sum())
    for lv in ("L0", "L1", "L2"):
        s[f"{lv}_pooled_supported_valid_pixels"] = int(df[f"{lv}_supported_valid_pixels"].sum())
        s[f"{lv}_pooled_coverage_percent"] = 100 * df[f"{lv}_supported_valid_pixels"].sum() / av
    return {k: (float(v) if isinstance(v, np.floating) else v) for k, v in s.items()}


def main():
    if len(sys.argv) < 3 and "PANDA_DATA_ROOT" not in os.environ:
        sys.exit("usage: coverage_from_caches.py <output_dir> [work_root]  (or set PANDA_DATA_ROOT)")
    work = Path(sys.argv[2] if len(sys.argv) > 2 else os.environ["PANDA_DATA_ROOT"])
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    allsum = {}
    for cohort, (folder, pattern) in COHORTS.items():
        files = sorted((work / folder).glob(pattern))
        df = pd.DataFrame([one(f, cohort) for f in files])
        df.to_csv(out / f"{cohort}_coverage_per_wsi.csv", index=False)
        allsum[cohort] = summary(df, cohort, len(files))
        json.dump(allsum[cohort], open(out / f"{cohort}_coverage_summary.json", "w"), indent=2)
        test = df[df.file.str.startswith("test_")] if cohort == "PANDA" else df
        print(f"{cohort}: {len(df)} caches; {'test ' if cohort == 'PANDA' else ''}"
              f"{int(test.supported_valid_pixels.sum()):,} of {int(test.annotated_valid_pixels.sum()):,} annotated-valid pixels supported")
    json.dump(allsum, open(out / "coverage_summary_all.json", "w"), indent=2)


if __name__ == "__main__":
    main()
