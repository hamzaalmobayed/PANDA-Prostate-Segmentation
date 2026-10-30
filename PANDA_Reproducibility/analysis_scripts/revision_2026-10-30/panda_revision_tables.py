"""Revision tables for the PANDA test and validation splits (30 Oct 2026).

Inputs (all in the repository):
  revision_2026-10-30/tables/cache_per_slide_stats_{val,test}.csv   (cache_stats.py)
  revision_2026-10-30/tables/panda_test_raw_mask_per_slide.csv     (panda_raw_mask_analyses.py)
  paper_results/tables/per_slide_wsi_metrics.csv, split_ids_reproducibility.csv
  PANDA train.csv (slide-level ISUP grade), given as the second argument

Every bootstrap resamples WSIs with numpy.random.default_rng(seed), uses the
percentile interval and 10,000 resamples unless stated. The approximate
two-sided p is min(1, 2 * min(P*(d <= 0), P*(d >= 0))) over the bootstrap
replicates d, as in PANDA_B_ANALYSIS_ONE_CLICK.py; when no replicate falls in
the tail it is reported as a bound, p < 2/B, and Holm's step-down multipliers
(3, 2, 1, with monotonicity) are applied to the bounds as well.

Outputs (revision_2026-10-30/tables/):
  A1_provider_level_metrics.csv       level and fusion metrics by provider (own support, 0.43)
  A1_provider_paired_contrasts.csv    fusion minus level by provider (common support)
  B4_paired_contrasts_common_support.csv   revised Table 5 (one support rule for both estimands)
  B4_paired_contrasts_original_rule.csv    Table 5 recomputed under the original rules (check)
  B4_per_slide_rule_sensitivity.csv        per-slide contrast when single-level slides are dropped, not zero-coded
  B4_level_support_coverage.csv            share of annotated-valid and reference-tumor pixels inside each support
  A4_validation_paired_contrasts.csv       fusion minus L1 on the validation split (and test, same code)
  B6_size_strata.csv, B6_foci_detection.csv, B6_complete_misses.csv, B6_borderline_rule.csv
  D3_negative_slide_fp_by_provider.csv, D3_negative_slides_by_grade.csv
  D5_boundary_nonempty.csv
  D6_primary_ci_10000.csv
  B2_design_effect.csv
  A1_raw_label_prediction.csv, D2_pixel_size.csv, D13_pen_ink.csv   (if the raw-mask table exists)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

B = 10_000
MAPS = ("L0", "L1", "L2", "fusion")
LEVELS = ("L0", "L1", "L2")
NOMINAL_UM = {"radboud": 0.48, "karolinska": 0.452}  # PANDA design document (Zenodo 3715938)


def dice(tp, fp, fn):
    d = 2 * tp + fp + fn
    return np.divide(2 * tp, d, out=np.zeros_like(d, dtype=float), where=d > 0) if np.ndim(d) else (2 * tp / d if d else 0.0)


def metrics(c):
    tp, fp, tn, fn = (float(x) for x in c)
    div = lambda a, b: a / b if b else float("nan")
    return {"dice": div(2 * tp, 2 * tp + fp + fn), "iou": div(tp, tp + fp + fn), "precision": div(tp, tp + fp),
            "recall": div(tp, tp + fn), "specificity": div(tn, tn + fp)}


def counts(df, prefix):
    return df[[f"{prefix}_{k}" for k in ("tp", "fp", "tn", "fn")]].to_numpy(np.int64)


def boot_idx(n, seed, strata=None):
    rng = np.random.default_rng(seed)
    if strata is None:
        for _ in range(B):
            yield rng.integers(0, n, n)
    else:
        groups = [np.flatnonzero(strata == s) for s in pd.unique(strata)]
        for _ in range(B):
            yield np.concatenate([g[rng.integers(0, len(g), len(g))] for g in groups])


def p_two_sided(sims):
    p = min(1.0, 2 * min(np.mean(sims <= 0), np.mean(sims >= 0)))
    return (p, False) if p > 0 else (2 / len(sims), True)


def holm(ps):
    """Holm step-down for (p, is_bound) pairs; bounds stay bounds."""
    order = np.argsort([p for p, _ in ps]); out = [None] * len(ps); run = 0.0; bound_run = False
    for i, k in enumerate(order):
        p, b = ps[k]
        adj = min(1.0, (len(ps) - i) * p)
        if adj >= run:
            run, bound_run = adj, b
        out[k] = (run, bound_run)
    return out


def fmt_p(p, bound):
    return f"<{p:.4f}" if bound else f"{p:.4f}"


def paired(cnt, kind, seed, strata=None, levels=LEVELS, label=""):
    """cnt: dict map -> (n, 4) counts. kind: pooled or slide."""
    n = len(next(iter(cnt.values())))
    if kind == "pooled":
        stat = lambda c, ix: dice(*(c[ix].sum(0)[[0, 1, 3]]))
    else:
        sc = {m: dice(c[:, 0], c[:, 1], c[:, 3]) for m, c in cnt.items()}
        stat = None
    obs, sims = {}, {lv: np.empty(B) for lv in levels}
    for lv in levels:
        if kind == "pooled":
            obs[lv] = stat(cnt["fusion"], slice(None)) - stat(cnt[lv], slice(None))
        else:
            obs[lv] = float((sc["fusion"] - sc[lv]).mean())
    for j, ix in enumerate(boot_idx(n, seed, strata)):
        for lv in levels:
            if kind == "pooled":
                sims[lv][j] = stat(cnt["fusion"], ix) - stat(cnt[lv], ix)
            else:
                sims[lv][j] = (sc["fusion"][ix] - sc[lv][ix]).mean()
    ps = [p_two_sided(sims[lv]) for lv in levels]
    adj = holm(ps) if len(levels) > 1 else ps
    rows = []
    for lv, (p, pb), (a, ab) in zip(levels, ps, adj):
        lo, hi = np.percentile(sims[lv], [2.5, 97.5])
        rows.append({"analysis": label, "estimand": kind, "contrast": f"fusion-{lv}", "n_wsi": n,
                     "difference": obs[lv], "ci_low": lo, "ci_high": hi, "boot_se": sims[lv].std(ddof=1),
                     "p_approx": fmt_p(p, pb), "p_holm": fmt_p(a, ab), "resamples": B, "seed": seed,
                     "strata": "provider" if strata is not None else "none"})
    return rows


def main():
    rev = Path(sys.argv[1]); train_csv = Path(sys.argv[2])
    T = rev / "tables"; repo_tables = rev.parent / "paper_results" / "tables"
    test = pd.read_csv(T / "cache_per_slide_stats_test.csv")
    val = pd.read_csv(T / "cache_per_slide_stats_val.csv")
    assert len(test) == 1577 and len(val) == 1577
    raw_path = T / "panda_test_raw_mask_per_slide.csv"
    raw = pd.read_csv(raw_path) if raw_path.exists() else None
    um = test.provider.map(NOMINAL_UM)
    if raw is not None:
        um = test[["image_id"]].merge(raw[["image_id", "um_per_px_image"]], on="image_id", how="left").um_per_px_image.to_numpy()
    test["um_eval_px"] = np.asarray(um, float) * 16
    test["px_area_mm2"] = (test.um_eval_px / 1000) ** 2

    # ---------- A1: provider x level ----------
    rows = []
    for prov, d in [("pooled", test)] + [(p, g) for p, g in test.groupby("provider")]:
        tb = d.reference_tumor_pixels_common > 0
        for m in MAPS:
            c = counts(d, f"{m}_own").sum(0)
            sc_common = dice(*(counts(d[tb], f"{m}_common")[:, [0, 1, 3]].T))
            sc_own = dice(*(counts(d[tb], f"{m}_own")[:, [0, 1, 3]].T))
            rows.append({"provider": prov, "map": m, "n_wsi": len(d), **metrics(c),
                         "reference_tumor_pixels": int(c[0] + c[3]),
                         "n_tumor_bearing": int(tb.sum()), "mean_slide_dice_common_tumor_bearing": float(sc_common.mean()),
                         "median_slide_dice_common_tumor_bearing": float(np.median(sc_common)),
                         "mean_slide_dice_own_tumor_bearing_zero_coded": float(sc_own.mean())})
    pd.DataFrame(rows).to_csv(T / "A1_provider_level_metrics.csv", index=False)
    rows = []
    for i, (prov, d) in enumerate(test.groupby("provider")):
        d = d.reset_index(drop=True)
        cnt = {m: counts(d, f"{m}_common") for m in MAPS}
        rows += paired(cnt, "pooled", 42 + 10 * (i + 1), label=f"{prov}, common support")
        tb = (d.reference_tumor_pixels_common > 0).to_numpy()
        rows += paired({m: c[tb] for m, c in cnt.items()}, "slide", 43 + 10 * (i + 1), label=f"{prov}, common support, tumor-bearing")
    pd.DataFrame(rows).to_csv(T / "A1_provider_paired_contrasts.csv", index=False)

    # ---------- B4 ----------
    cnt_c = {m: counts(test, f"{m}_common") for m in MAPS}
    tb = (test.reference_tumor_pixels_common > 0).to_numpy()
    rows = paired(cnt_c, "pooled", 42, label="common support") + \
        paired({m: c[tb] for m, c in cnt_c.items()}, "slide", 43, label="common support, tumor-bearing")
    pd.DataFrame(rows).to_csv(T / "B4_paired_contrasts_common_support.csv", index=False)
    pd.DataFrame([{"map": m, **metrics(cnt_c[m].sum(0))} for m in MAPS]).to_csv(T / "B4_common_support_dice.csv", index=False)
    cnt_o = {m: counts(test, f"{m}_own") for m in MAPS}
    tb_f = (cnt_o["fusion"][:, 0] + cnt_o["fusion"][:, 3] > 0)
    rows = paired(cnt_o, "pooled", 42, label="original: own support") + \
        paired({m: c[tb_f] for m, c in cnt_o.items()}, "slide", 43, label="original: own support, zero-coded")
    pd.DataFrame(rows).to_csv(T / "B4_paired_contrasts_original_rule.csv", index=False)
    rows = []
    for lv in LEVELS:
        keep = tb_f & (cnt_o[lv][:, 0] + cnt_o[lv][:, 3] > 0)
        sub = {m: cnt_o[m][keep] for m in ("fusion", lv)}
        r = paired(sub, "slide", 43, levels=(lv,), label="own support, slides without reference tumor in the level's support dropped")
        rows += r
    pd.DataFrame(rows).to_csv(T / "B4_per_slide_rule_sensitivity.csv", index=False)
    rows = []
    for m in MAPS:
        rows.append({"map": m, "annotated_valid_pixels": int(test.annotated_valid_pixels.sum()),
                     "support_av_pixels": int(test[f"{m}_support_av_pixels"].sum()),
                     "share_av": test[f"{m}_support_av_pixels"].sum() / test.annotated_valid_pixels.sum(),
                     "reference_tumor_pixels": int(test.reference_tumor_pixels_av.sum()),
                     "support_reference_tumor_pixels": int(test[f"{m}_support_reference_tumor_pixels"].sum()),
                     "share_reference_tumor": test[f"{m}_support_reference_tumor_pixels"].sum() / test.reference_tumor_pixels_av.sum(),
                     "wsi_with_reference_tumor_in_support": int((test[f"{m}_support_reference_tumor_pixels"] > 0).sum())})
    pd.DataFrame(rows).to_csv(T / "B4_level_support_coverage.csv", index=False)

    # ---------- A4: validation ----------
    rows = []
    for split, d in (("validation", val), ("test", test)):
        for tag, thr_note in (("own", "common threshold 0.43"), ("lvlthr", "level-specific thresholds (L1 0.40, fusion 0.43)")):
            cnt = {m: counts(d, f"{m}_{tag}") for m in ("fusion", "L1")}
            rows += paired(cnt, "pooled", 42, levels=("L1",), label=f"{split}, {thr_note}")
            tbs = (cnt["fusion"][:, 0] + cnt["fusion"][:, 3] > 0)
            rows += paired({m: c[tbs] for m, c in cnt.items()}, "slide", 43, levels=("L1",), label=f"{split}, {thr_note}, own support zero-coded")
        cnt = {m: counts(d, f"{m}_common") for m in ("fusion", "L1")}
        rows += paired(cnt, "pooled", 42, levels=("L1",), label=f"{split}, common support, 0.43")
        tbs = (d.reference_tumor_pixels_common > 0).to_numpy()
        rows += paired({m: c[tbs] for m, c in cnt.items()}, "slide", 43, levels=("L1",), label=f"{split}, common support, 0.43")
    out = pd.DataFrame(rows)
    out.to_csv(T / "A4_validation_paired_contrasts.csv", index=False)

    # ---------- B2: design effect ----------
    # Images per patient: PANDA development set, 10,616 biopsies from 2,113 cases (Bulten et al. 2022, Table 1).
    # With a random image-level split, the number of test images of a patient with k images is Binomial(k, 1577/10512).
    q = 1577 / 10512
    m_eff = {}
    for prov, k in (("radboud", 5160 / 1028), ("karolinska", 5456 / 1085)):
        kk = int(round(k)); from math import comb
        pmf = [comb(kk, j) * q ** j * (1 - q) ** (kk - j) for j in range(kk + 1)]
        e1 = sum(j * p for j, p in enumerate(pmf)); e2 = sum(j * j * p for j, p in enumerate(pmf))
        m_eff[prov] = e2 / e1
    meff = float(np.mean(list(m_eff.values())))
    rows = []
    c4 = pd.read_csv(T / "B4_paired_contrasts_common_support.csv")
    for _, r in c4[c4.contrast == "fusion-L1"].iterrows():
        z = abs(r.difference) / r.boot_se
        de_zero = (z / 1.959964) ** 2
        for rho in (0.25, 0.5, 1.0):
            de = 1 + (meff - 1) * rho
            se = r.boot_se * np.sqrt(de)
            rows.append({"analysis": r.analysis, "estimand": r.estimand, "difference": r.difference, "boot_se": r.boot_se,
                         "design_effect_at_which_ci_reaches_zero": de_zero, "assumed_images_per_patient": "Radboud 5.02, Karolinska 5.03",
                         "effective_cluster_size_in_test": meff, "rho": rho, "design_effect": de,
                         "inflated_ci_low": r.difference - 1.959964 * se, "inflated_ci_high": r.difference + 1.959964 * se})
    pd.DataFrame(rows).to_csv(T / "B2_design_effect.csv", index=False)

    # ---------- B6: tumor size ----------
    d = test[tb].copy()
    d["ref_area_mm2"] = d.reference_tumor_pixels_common * d.px_area_mm2
    for m in ("fusion", "L1"):
        d[f"{m}_dice_common"] = dice(d[f"{m}_common_tp"].to_numpy(), d[f"{m}_common_fp"].to_numpy(), d[f"{m}_common_fn"].to_numpy())
    d["quartile"] = pd.qcut(d.ref_area_mm2, 4, labels=["Q1 (smallest)", "Q2", "Q3", "Q4 (largest)"])
    rows = []
    for qn, g in d.groupby("quartile", observed=True):
        g = g.reset_index(drop=True)
        cnt = {m: counts(g, f"{m}_common") for m in MAPS}
        pr = paired(cnt, "slide", 43, levels=("L1",), label=str(qn))[0]
        rows.append({"stratum": str(qn), "n_wsi": len(g), "area_mm2_min": g.ref_area_mm2.min(), "area_mm2_max": g.ref_area_mm2.max(),
                     "karolinska": int((g.provider == "karolinska").sum()), "radboud": int((g.provider == "radboud").sum()),
                     "mean_dice_fusion": g.fusion_dice_common.mean(), "mean_dice_L1": g.L1_dice_common.mean(),
                     "fusion_minus_L1": pr["difference"], "ci_low": pr["ci_low"], "ci_high": pr["ci_high"],
                     "complete_misses_fusion": int((g.fusion_common_tp == 0).sum()), "complete_misses_L1": int((g.L1_common_tp == 0).sum())})
    pd.DataFrame(rows).to_csv(T / "B6_size_strata.csv", index=False)
    foci = []
    for _, r in test[test.n_reference_foci > 0].iterrows():
        for f in str(r.reference_foci).split(";"):
            a, hf, hl = f.split(":")
            foci.append({"image_id": r.image_id, "provider": r.provider, "area_px": int(a), "area_mm2": int(a) * r.px_area_mm2,
                         "detected_fusion": int(hf), "detected_L1": int(hl)})
    foci = pd.DataFrame(foci)
    foci.to_csv(T / "B6_reference_foci.csv", index=False)
    bins = [0, 0.01, 0.05, 0.25, 1, np.inf]
    labels = ["<0.01", "0.01-0.05", "0.05-0.25", "0.25-1", ">=1"]
    foci["size_mm2"] = pd.cut(foci.area_mm2, bins, labels=labels, right=False)
    agg = pd.concat([foci.assign(provider="pooled"), foci]).groupby(["provider", "size_mm2"], observed=False).agg(
        n_foci=("area_px", "size"), detected_fusion=("detected_fusion", "mean"),
        detected_L1=("detected_L1", "mean"), wsi=("image_id", "nunique")).reset_index()
    agg.to_csv(T / "B6_foci_detection.csv", index=False)
    d["fusion_pred_pixels"] = d.fusion_own_tp + d.fusion_own_fp
    d["empty_prediction"] = d.fusion_pred_pixels == 0
    miss = d[d.fusion_common_tp == 0][["image_id", "provider", "reference_tumor_pixels_common", "ref_area_mm2", "n_reference_foci",
                                        "fusion_pred_pixels", "empty_prediction", "L0_common_tp", "L1_common_tp", "L2_common_tp"]]
    miss.to_csv(T / "B6_complete_misses.csv", index=False)
    # The training-bucket rule: a patch is "tumor" only if tumor >= 5% of its valid tissue. A 512 x 512 patch spans
    # 32 x 32 (L0), 128 x 128 (L1) and 512 x 512 (L2) evaluation pixels, so 5% of a fully valid patch is 51, 819 and
    # 13,107 evaluation pixels. Foci smaller than these never yield an eligible tumor patch at that level unless
    # they share the patch with other tumor.
    rows = []
    for lv, side in (("L0", 32), ("L1", 128), ("L2", 512)):
        thr_px = 0.05 * side * side
        rows.append({"level": lv, "patch_side_eval_px": side, "five_percent_eval_px": thr_px,
                     "five_percent_mm2": thr_px * float(np.median(d.px_area_mm2)),
                     "share_of_foci_below": float((foci.area_px < thr_px).mean()),
                     "share_of_tumor_bearing_wsi_with_all_tumor_below": float((d.reference_tumor_pixels_common < thr_px).mean()),
                     "zero_overlap_wsi_with_all_tumor_below": int((miss.reference_tumor_pixels_common < thr_px).sum()),
                     "empty_prediction_wsi_with_all_tumor_below": int((miss[miss.empty_prediction].reference_tumor_pixels_common < thr_px).sum())})
    pd.DataFrame(rows).to_csv(T / "B6_borderline_rule.csv", index=False)

    # ---------- D3: reference-negative slides ----------
    grades = pd.read_csv(train_csv)[["image_id", "isup_grade", "gleason_score"]]
    neg = test[~tb].merge(grades, on="image_id", how="left")
    neg["fp_px"] = neg.fusion_own_fp
    neg["fp_mm2"] = neg.fp_px * neg.px_area_mm2
    def comp_list(s):
        return [int(x) for x in str(s).split(";")] if isinstance(s, str) and s else []
    neg["largest_fp_mm2"] = [max(comp_list(s), default=0) * a for s, a in zip(neg.fusion_fp_component_areas, neg.px_area_mm2)]
    neg["fp_foci_ge_0.01mm2"] = [sum(1 for x in comp_list(s) if x * a >= 0.01) for s, a in zip(neg.fusion_fp_component_areas, neg.px_area_mm2)]
    rows = []
    for prov, g in [("pooled", neg)] + list(neg.groupby("provider")):
        rows.append({"provider": prov, "n_reference_negative": len(g), "any_fp": int((g.fp_px > 0).sum()),
                     "fp_foci_total": int(g.n_fusion_fp_components.sum()), "fp_foci_median": g.n_fusion_fp_components.median(),
                     "fp_foci_median_among_fp": g.n_fusion_fp_components[g.fp_px > 0].median(),
                     "fp_foci_max": int(g.n_fusion_fp_components.max()),
                     "wsi_with_fp_focus_ge_0.01mm2": int((g["fp_foci_ge_0.01mm2"] > 0).sum()),
                     "fp_area_mm2_total": g.fp_mm2.sum(), "fp_area_mm2_median_among_fp": g.fp_mm2[g.fp_px > 0].median(),
                     "fp_area_mm2_p90_among_fp": g.fp_mm2[g.fp_px > 0].quantile(0.9), "fp_area_mm2_max": g.fp_mm2.max(),
                     "largest_focus_mm2_median_among_fp": g.largest_fp_mm2[g.fp_px > 0].median(),
                     "fp_pixel_share": g.fp_px.sum() / (g.fusion_own_tp + g.fusion_own_fp + g.fusion_own_tn + g.fusion_own_fn).sum()})
    pd.DataFrame(rows).to_csv(T / "D3_negative_slide_fp_by_provider.csv", index=False)
    neg["grade_group"] = np.where(neg.isup_grade == 0, "ISUP 0 (benign)", "ISUP >= 1")
    tab = neg.groupby(["provider", "grade_group"]).agg(n=("image_id", "size"), any_fp=("fp_px", lambda s: int((s > 0).sum())),
                                                      fp_area_mm2_median=("fp_mm2", "median")).reset_index()
    tab.to_csv(T / "D3_negative_slides_by_grade.csv", index=False)
    neg[["image_id", "provider", "isup_grade", "gleason_score", "fp_px", "fp_mm2", "n_fusion_fp_components", "largest_fp_mm2"]].to_csv(
        T / "D3_negative_slides.csv", index=False)

    # ---------- D5: boundary metrics on both-non-empty slides ----------
    ps = pd.read_csv(repo_tables / "per_slide_wsi_metrics.csv").merge(test[["image_id", "um_eval_px"]], on="image_id")
    ne = ps[(ps.fusion_gt_tumor_pixels > 0) & (ps.fusion_pred_tumor_pixels > 0)]
    rows = []
    for col, name in (("fusion_hd95_eval_px", "HD95"), ("fusion_assd_eval_px", "ASSD"), ("fusion_surface_dice", "surface Dice")):
        v = ne[col]
        r = {"metric": name, "n": int(v.notna().sum()), "mean": v.mean(), "median": v.median(), "sd": v.std(ddof=1)}
        if name != "surface Dice":
            vu = v * ne.um_eval_px
            r.update({"mean_um": vu.mean(), "median_um": vu.median()})
        rows.append(r)
    one_sided = ps[(ps.fusion_gt_tumor_pixels > 0) != (ps.fusion_pred_tumor_pixels > 0)]
    rows.append({"metric": "surface Dice assigned when exactly one boundary is empty", "n": len(one_sided),
                 "mean": one_sided.fusion_surface_dice.mean(), "median": one_sided.fusion_surface_dice.median(),
                 "sd": one_sided.fusion_surface_dice.std(ddof=1)})
    rows.append({"metric": "surface Dice tolerance (um), 2 grid px", "n": len(ps), "mean": 2 * ps.um_eval_px.mean(),
                 "median": 2 * ps.um_eval_px.median()})
    pd.DataFrame(rows).to_csv(T / "D5_boundary_nonempty.csv", index=False)

    # ---------- D6: primary CI with 10,000 resamples ----------
    rows = []
    for strat in (None, test.provider.to_numpy()):
        for m in MAPS:
            c = counts(test, f"{m}_own")
            sims = np.array([dice(*(c[ix].sum(0)[[0, 1, 3]])) for ix in boot_idx(len(c), 42, strat)])
            rows.append({"map": m, "dice": dice(*(c.sum(0)[[0, 1, 3]])), "ci_low": np.percentile(sims, 2.5),
                         "ci_high": np.percentile(sims, 97.5), "resamples": B, "seed": 42, "interval": "percentile",
                         "unit": "WSI", "strata": "provider" if strat is not None else "none"})
    for i, (prov, g) in enumerate(test.groupby("provider")):
        c = counts(g, "fusion_own")
        sims = np.array([dice(*(c[ix].sum(0)[[0, 1, 3]])) for ix in boot_idx(len(c), 42 + i + 1)])
        rows.append({"map": f"fusion, {prov}", "dice": dice(*(c.sum(0)[[0, 1, 3]])), "ci_low": np.percentile(sims, 2.5),
                     "ci_high": np.percentile(sims, 97.5), "resamples": B, "seed": 42 + i + 1, "interval": "percentile",
                     "unit": "WSI", "strata": "none"})
    pd.DataFrame(rows).to_csv(T / "D6_primary_ci_10000.csv", index=False)

    # ---------- raw-mask analyses ----------
    if raw is not None:
        rows = []
        for prov, g in raw.groupby("provider"):
            labels = {"radboud": {1: "stroma", 2: "benign epithelium", 3: "Gleason 3", 4: "Gleason 4", 5: "Gleason 5"},
                      "karolinska": {1: "benign tissue (stroma and epithelium)", 2: "cancerous tissue (stroma and epithelium)"}}[prov]
            for L, name in labels.items():
                n = g[f"label{L}_pixels"].sum()
                rows.append({"provider": prov, "label": L, "name": name, "pixels": int(n),
                             "predicted_tumor_fusion": g[f"label{L}_pred_fusion"].sum() / n, "predicted_tumor_L1": g[f"label{L}_pred_L1"].sum() / n})
            if prov == "radboud":
                for tag in ("inside", "outside"):
                    n = g[f"stroma_{tag}_envelope_pixels"].sum()
                    rows.append({"provider": prov, "label": 1, "name": f"stroma {tag} tumor envelope (closing, r=5 grid px)",
                                 "pixels": int(n), "predicted_tumor_fusion": g[f"stroma_{tag}_envelope_pred_fusion"].sum() / n,
                                 "predicted_tumor_L1": g[f"stroma_{tag}_envelope_pred_L1"].sum() / n})
        pd.DataFrame(rows).to_csv(T / "A1_raw_label_prediction.csv", index=False)
        px = raw.groupby("provider").agg(n=("image_id", "size"), um_image_min=("um_per_px_image", "min"),
                                         um_image_median=("um_per_px_image", "median"), um_image_max=("um_per_px_image", "max"),
                                         um_mask_median=("um_per_px_mask", "median"),
                                         mapping_mismatch_px=("mapping_mismatch_av_pixels", "sum")).reset_index()
        vc = raw.groupby("provider").um_per_px_image.apply(lambda s: s.round(4).value_counts().to_dict()).reset_index()
        px.to_csv(T / "D2_pixel_size.csv", index=False)
        vc.to_csv(T / "D2_pixel_size_values.csv", index=False)
        k = raw[raw.provider == "karolinska"].merge(test[["image_id", "reference_tumor_pixels_common"]], on="image_id")
        rows = []
        for tag, g in (("all Karolinska", k), ("with ink", k[k.ink_pixels > 0]), ("without ink", k[k.ink_pixels == 0])):
            full = test.set_index("image_id").loc[g.image_id]
            c_full = full[[f"fusion_own_{x}" for x in ("tp", "fp", "tn", "fn")]].sum().to_numpy()
            c_noink = g[[f"fusion_noink_{x}" for x in ("tp", "fp", "tn", "fn")]].sum().to_numpy()
            rows.append({"subset": tag, "n_wsi": len(g), "tumor_bearing": int((g.reference_tumor_pixels_common > 0).sum()),
                         "dice_all_pixels": metrics(c_full)["dice"], "dice_without_pixels_within_32px_of_ink": metrics(c_noink)["dice"],
                         "precision_all": metrics(c_full)["precision"], "precision_noink": metrics(c_noink)["precision"],
                         "valid_pixels_removed": int(g.valid_pixels_near_ink.sum())})
        pd.DataFrame(rows).to_csv(T / "D13_pen_ink.csv", index=False)
    print("done")


if __name__ == "__main__":
    main()
