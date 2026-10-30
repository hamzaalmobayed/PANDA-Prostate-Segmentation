"""Analyses of the PRECISE run reported in the manuscript (OME levels 0, 2 and 4).

Uses the saved PRECISE WSI caches (precise_wsi_cache.zip of the release
panda-wsi-cache-complement-20261027) and the 27 H&E label masks of the PRECISE
release. No inference is run. Writes to <output_dir>:

  PRECISE_labels.csv          every PRECISE label: use, pixels on the evaluation grid (mask level 4,
                              3.888 um per pixel), share of annotated pixels, share inside fused
                              support, and share of the supported pixels predicted as tumor
  PRECISE_ci.csv              95% CIs of fused and single-level pooled Dice by subject and by WSI
                              (10,000 resamples, percentile)
  PRECISE_paired.csv          fusion minus L1 pooled Dice, resampling subjects
  PRECISE_negative_wsi.csv    false-positive components and area on the tumor-free WSIs
  PRECISE_summary.json

Note: the first run retained only patches containing a label 1, 2 or 7 pixel, so
pixels of the unscored labels 3-6 have predictions only where such patches
overlap them; the share inside fused support is given for that reason.

Usage: python precise_first_run_analyses.py <precise_wsi_cache.zip> <mask_dir> <per_slide_wsi_metrics.csv> <output_dir>
"""
import io
import json
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import tifffile

THR = np.float32(0.43)
B = 10_000
UM = 0.243 * 16
LABELS = {0: ("Background", "not annotated tissue"), 1: ("Tumor", "tumor"), 2: ("Benign gland", "negative"),
          3: ("Artifact", "not scored"), 4: ("High-grade prostatic intraepithelial neoplasia", "not scored"),
          5: ("Intraductal carcinoma", "not scored"), 6: ("Atypical intraductal proliferation", "not scored"),
          7: ("Stroma", "negative")}


def dice(c):
    return 2 * c[..., 0] / np.maximum(2 * c[..., 0] + c[..., 1] + c[..., 3], 1)


def mask_level4(path, shape):
    with tifffile.TiffFile(path) as tf:
        a = tf.series[0].levels[4].asarray()
    if a.ndim == 3:
        a = a[..., 0]
    # Same as reconstruct_slide() of run_external_precise_wsi_test.py: crop, then nearest-neighbour resize.
    H, W = shape
    a = a[:H, :W]
    if a.shape != (H, W):
        a = cv2.resize(a.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
    return a


def main():
    zpath, mdir, ps_csv, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
    out.mkdir(parents=True, exist_ok=True)
    z = zipfile.ZipFile(zpath)
    lab_rows, neg_rows, cov = [], [], []
    for name in sorted(z.namelist()):
        sid = Path(name).name.split("_ds16")[0]
        c = np.load(io.BytesIO(z.read(name)))
        gt = c["gt"].astype(bool); valid = c["valid"].astype(bool)
        fsup = c["fusion_support"].astype(bool)
        te = c["tissue_eval"].astype(bool) if "tissue_eval" in c.files else np.ones_like(gt)
        pred = (c["fusion"].astype(np.float32) >= THR) & te
        lab = mask_level4(mdir / f"{sid}_h-e_mask.ome.tif", gt.shape)
        assert ((lab == 1) == gt)[valid].all(), f"label 1 does not reproduce the cached reference for {sid}"
        for L in LABELS:
            sel = lab == L
            lab_rows.append({"image_id": sid, "label": L, "pixels": int(sel.sum()), "supported": int((sel & fsup).sum()),
                             "pred_tumor": int((sel & fsup & pred).sum())})
        if not (gt & valid).any():
            fp = pred & valid & ~gt
            n, _, st, _ = cv2.connectedComponentsWithStats(fp.astype(np.uint8), connectivity=8)
            areas = st[1:, cv2.CC_STAT_AREA] if n > 1 else np.array([], int)
            neg_rows.append({"image_id": sid, "valid_pixels": int(valid.sum()), "fp_pixels": int(fp.sum()),
                             "fp_area_mm2": float(fp.sum() * (UM / 1000) ** 2), "fp_components": int(n - 1),
                             "largest_component_mm2": float(areas.max() * (UM / 1000) ** 2) if len(areas) else 0.0})
    lr = pd.DataFrame(lab_rows).groupby("label")[["pixels", "supported", "pred_tumor"]].sum().reset_index()
    annotated = lr[lr.label > 0].pixels.sum()
    lr["class"] = lr.label.map(lambda L: LABELS[L][0]); lr["use"] = lr.label.map(lambda L: LABELS[L][1])
    lr["share_of_annotated"] = np.where(lr.label > 0, lr.pixels / annotated, np.nan)
    lr["share_in_support"] = lr.supported / lr.pixels.replace(0, np.nan)
    lr["predicted_tumor_share"] = lr.pred_tumor / lr.supported.replace(0, np.nan)
    lr.to_csv(out / "PRECISE_labels.csv", index=False)
    neg = pd.DataFrame(neg_rows); neg.to_csv(out / "PRECISE_negative_wsi.csv", index=False)

    ps = pd.read_csv(ps_csv).dropna(subset=["image_id"]).sort_values("image_id").reset_index(drop=True)
    ps["subject"] = ps.image_id.str.slice(0, 6)
    C = {k: ps[[f"{k}_{x}" for x in ("tp", "fp", "tn", "fn")]].to_numpy(np.int64) for k in ("L0", "L1", "L2", "fusion")}
    rows = []
    for unit, u, seed in (("subject", ps.subject.to_numpy(), 42), ("WSI", ps.image_id.to_numpy(), 43)):
        rng = np.random.default_rng(seed); uniq = np.unique(u); groups = [np.flatnonzero(u == s) for s in uniq]
        sims = {k: np.empty(B) for k in C}; diff = np.empty(B)
        for j in range(B):
            ix = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
            for k in C:
                sims[k][j] = dice(C[k][ix].sum(0))
            diff[j] = sims["fusion"][j] - sims["L1"][j]
        for k in C:
            rows.append({"map": k, "unit": unit, "n_units": len(uniq), "dice": float(dice(C[k].sum(0))),
                         "ci_low": float(np.percentile(sims[k], 2.5)), "ci_high": float(np.percentile(sims[k], 97.5)),
                         "resamples": B, "seed": seed})
        if unit == "subject":
            pr = {"contrast": "fusion-L1 pooled Dice", "difference": float(dice(C["fusion"].sum(0)) - dice(C["L1"].sum(0))),
                  "ci_low": float(np.percentile(diff, 2.5)), "ci_high": float(np.percentile(diff, 97.5)),
                  "unit": "subject", "resamples": B, "seed": seed}
    pd.DataFrame(rows).to_csv(out / "PRECISE_ci.csv", index=False)
    pd.DataFrame([pr]).to_csv(out / "PRECISE_paired.csv", index=False)
    summ = {"n_wsi": int(len(ps)), "n_subjects": int(ps.subject.nunique()), "neg_n": int(len(neg)),
            "neg_fp_area_mm2_total": float(neg.fp_area_mm2.sum()), "neg_components_total": int(neg.fp_components.sum()),
            "neg_largest_component_mm2": float(neg.largest_component_mm2.max()),
            "neg_components_median": float(neg.fp_components.median()), "um_per_eval_px": UM}
    json.dump(summ, open(out / "PRECISE_summary.json", "w"), indent=2)
    print(lr.round(4).to_string(index=False)); print(pd.DataFrame(rows).round(4).to_string(index=False)); print(pr); print(summ)


if __name__ == "__main__":
    main()
