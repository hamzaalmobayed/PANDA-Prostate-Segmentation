"""Summary tables of the PRECISE run at the PANDA pixel size (precise_matched_scale.py).

Reads the matched-scale caches (one .npz per WSI, with label_eval) and the
first run's per-slide table, and writes to <output_dir>:

  PRECISE_matched_metrics.csv     pooled metrics of L0, L1, L2, fusion and the constant-coordinate maps
  PRECISE_matched_ci.csv          95% CIs (10,000 resamples) by subject and by WSI, for both runs
  PRECISE_matched_paired.csv      fusion minus L1 (pooled Dice), resampling subjects
  PRECISE_matched_per_slide.csv   per-WSI metrics and false-positive components
  PRECISE_labels.csv              every PRECISE label, its use, pixel share and predicted-tumor share
  PRECISE_matched_summary.json    the numbers quoted in the text

Areas use 7.776 um per evaluation pixel (0.243 um at level 0, level 5 grid).

Usage: python precise_matched_summary.py <matched_root> <first_run_per_slide_csv> <participants.csv> <output_dir>
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

THR = np.float32(0.43)
B = int(__import__("os").environ.get("PRECISE_B", "10000"))
UM = 0.243 * 32
LABELS = {0: ("Background", "not annotated tissue"), 1: ("Tumor", "tumor"), 2: ("Benign gland", "negative"),
          3: ("Artifact", "not scored"), 4: ("High-grade prostatic intraepithelial neoplasia", "not scored"),
          5: ("Intraductal carcinoma", "not scored"), 6: ("Atypical intraductal proliferation", "not scored"),
          7: ("Stroma", "negative")}


def m(tp, fp, tn, fn):
    d = lambda a, b: a / b if b else float("nan")
    return {"tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn), "dice": d(2 * tp, 2 * tp + fp + fn),
            "iou": d(tp, tp + fp + fn), "precision": d(tp, tp + fp), "recall": d(tp, tp + fn),
            "specificity": d(tn, tn + fp), "accuracy": d(tp + tn, tp + tn + fp + fn)}


def counts(pred, gt, region):
    p, g = pred[region], gt[region]
    return [int((p & g).sum()), int((p & ~g).sum()), int((~p & ~g).sum()), int((~p & g).sum())]


def dice_of(c):
    return 2 * c[..., 0] / np.maximum(2 * c[..., 0] + c[..., 1] + c[..., 3], 1)


def boot(c_by_unit, units, seed):
    """c_by_unit: dict name -> (n_wsi, 4); units: subject per WSI (or WSI index)."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(units); groups = [np.flatnonzero(units == u) for u in uniq]
    out = {k: np.empty(B) for k in c_by_unit}
    for j in range(B):
        ix = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        for k, c in c_by_unit.items():
            out[k][j] = dice_of(c[ix].sum(0))
    return out


def main():
    root, first_csv, part_csv, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
    out.mkdir(parents=True, exist_ok=True)
    caches = sorted((root / "wsi_cache").glob("*_lvl135_ds16_matched_v1.npz"))
    part = pd.read_csv(part_csv)
    subj = dict(zip(part.IMAGE_NAME, part.IMAGE_NAME.str.slice(0, 6)))
    rows, lab_rows = [], []
    keys = ["L0", "L1", "L2", "fusion", "cc_L0", "cc_L1", "cc_L2", "cc_fusion"]
    for p in caches:
        sid = p.name.split("_lvl")[0]
        z = np.load(p)
        gt = z["gt"].astype(bool); av = z["annotated_valid"].astype(bool); te = z["tissue_eval"].astype(bool)
        valid = z["valid"].astype(bool); lab = z["label_eval"]; fsup = z["fusion_support"].astype(bool)
        r = {"image_id": sid, "subject": subj.get(sid, sid[:6]), "valid_pixels": int(valid.sum()),
             "annotated_valid_pixels": int(av.sum()), "supported_valid_pixels": int((av & fsup).sum())}
        for k in keys:
            cc = k.startswith("cc_"); base = k[3:] if cc else k
            if base == "fusion":
                prob, region = z["fusion_constcoord" if cc else "fusion"], valid
            else:
                prob = z[f"prob_constcoord_{base}" if cc else f"prob_{base}"]
                region = av & z[f"support_{base}"].astype(bool)
            pred = (prob.astype(np.float32) >= THR) & te
            for name, v in zip(("tp", "fp", "tn", "fn"), counts(pred, gt, region)):
                r[f"{k}_{name}"] = v
        fpred = (z["fusion"].astype(np.float32) >= THR) & te
        fp_mask = fpred & valid & ~gt
        n, _, st, _ = cv2.connectedComponentsWithStats(fp_mask.astype(np.uint8), connectivity=8)
        areas = st[1:, cv2.CC_STAT_AREA] if n > 1 else np.array([], int)
        r["fp_components"] = int(n - 1)
        r["fp_area_mm2"] = float(fp_mask.sum() * (UM / 1000) ** 2)
        r["largest_fp_component_mm2"] = float(areas.max() * (UM / 1000) ** 2) if len(areas) else 0.0
        for L in LABELS:
            sel = (lab == L) & fsup
            lab_rows.append({"image_id": sid, "label": L, "pixels": int((lab == L).sum()), "supported": int(sel.sum()),
                             "pred_tumor": int((sel & fpred).sum())})
        rows.append(r)
    ps = pd.DataFrame(rows)
    ps["fusion_dice"] = dice_of(ps[[f"fusion_{c}" for c in ("tp", "fp", "tn", "fn")]].to_numpy())
    ps["tumor_bearing"] = (ps.fusion_tp + ps.fusion_fn) > 0
    ps.to_csv(out / "PRECISE_matched_per_slide.csv", index=False)

    C = {k: ps[[f"{k}_{c}" for c in ("tp", "fp", "tn", "fn")]].to_numpy(np.int64) for k in keys}
    met = pd.DataFrame([{"map": k, **m(*C[k].sum(0))} for k in keys])
    met.to_csv(out / "PRECISE_matched_metrics.csv", index=False)

    first = pd.read_csv(first_csv).dropna(subset=["image_id"]).sort_values("image_id").reset_index(drop=True)
    first["subject"] = first.image_id.str.slice(0, 6)
    F = {k: first[[f"{k}_{c}" for c in ("tp", "fp", "tn", "fn")]].to_numpy(np.int64) for k in ("L0", "L1", "L2", "fusion")}
    ci = []
    for run, CC, df in (("matched", {k: C[k] for k in ("L0", "L1", "L2", "fusion", "cc_fusion")}, ps), ("first run", F, first)):
        for unit, u, seed in (("subject", df.subject.to_numpy(), 42), ("WSI", np.arange(len(df)), 43)):
            bs = boot(CC, u, seed)
            for k, v in bs.items():
                ci.append({"run": run, "map": k, "unit": unit, "dice": float(dice_of(CC[k].sum(0))),
                           "ci_low": float(np.percentile(v, 2.5)), "ci_high": float(np.percentile(v, 97.5)),
                           "resamples": B, "seed": seed, "n_units": int(len(np.unique(u)))})
    ci = pd.DataFrame(ci); ci.to_csv(out / "PRECISE_matched_ci.csv", index=False)

    rng = np.random.default_rng(44)
    subjects = ps.subject.to_numpy(); uniq = np.unique(subjects); groups = [np.flatnonzero(subjects == s) for s in uniq]
    d = np.empty(B)
    for j in range(B):
        ix = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        d[j] = dice_of(C["fusion"][ix].sum(0)) - dice_of(C["L1"][ix].sum(0))
    pr = {"contrast": "fusion-L1 pooled Dice", "difference": float(dice_of(C["fusion"].sum(0)) - dice_of(C["L1"].sum(0))),
          "ci_low": float(np.percentile(d, 2.5)), "ci_high": float(np.percentile(d, 97.5)), "unit": "subject",
          "resamples": B, "seed": 44}
    pd.DataFrame([pr]).to_csv(out / "PRECISE_matched_paired.csv", index=False)

    lr = pd.DataFrame(lab_rows).groupby("label")[["pixels", "supported", "pred_tumor"]].sum().reset_index()
    annotated = lr[lr.label > 0].pixels.sum()
    lr["class"] = lr.label.map(lambda L: LABELS[L][0]); lr["use"] = lr.label.map(lambda L: LABELS[L][1])
    lr["share_of_annotated"] = np.where(lr.label > 0, lr.pixels / annotated, np.nan)
    lr["predicted_tumor_share"] = lr.pred_tumor / lr.supported.replace(0, np.nan)
    lr.to_csv(out / "PRECISE_labels.csv", index=False)

    tb, neg = ps[ps.tumor_bearing], ps[~ps.tumor_bearing]
    summ = {"n_wsi": int(len(ps)), "n_subjects": int(ps.subject.nunique()), "n_tumor_bearing": int(len(tb)),
            "coverage": float(ps.supported_valid_pixels.sum() / ps.annotated_valid_pixels.sum()),
            "per_slide_mean_tumor_bearing": float(tb.fusion_dice.mean()), "per_slide_median_tumor_bearing": float(tb.fusion_dice.median()),
            "per_slide_sd_tumor_bearing": float(tb.fusion_dice.std(ddof=1)),
            "per_slide_min": float(tb.fusion_dice.min()), "per_slide_max": float(tb.fusion_dice.max()),
            "neg_n": int(len(neg)), "neg_with_fp": int((neg.fusion_fp > 0).sum()), "neg_fp_area_mm2": float(neg.fp_area_mm2.sum()),
            "neg_fp_share": float(neg.fusion_fp.sum() / neg.valid_pixels.sum()) if len(neg) else float("nan"),
            "neg_components": int(neg.fp_components.sum()), "neg_largest_component_mm2": float(neg.largest_fp_component_mm2.max()) if len(neg) else 0.0,
            "um_per_eval_px": UM}
    json.dump(summ, open(out / "PRECISE_matched_summary.json", "w"), indent=2)
    print(met.round(4).to_string(index=False)); print(json.dumps(summ, indent=1)); print(lr.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
