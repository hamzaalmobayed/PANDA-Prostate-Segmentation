"""Fill the numbers of the revised manuscript from the result files.

The manuscript template contains tokens <<NAME>>. Every token is resolved here
from one result file of the repository; nothing is typed by hand. The script
writes the WITH_FIGURES .tex, the WITHOUT_FIGURES .tex (the same text without
the \\includegraphics lines) and NUMBER_SOURCES.csv, which lists each number
with its place in the manuscript (section and line of the .tex file) and the
file and field it comes from.

Usage:
    python fill_manuscript_numbers.py <template.tex> <PANDA_Reproducibility> <precise_summary_dir> <sicap_summary_dir> <out_dir>
"""
import json
import re
import sys
from pathlib import Path

import pandas as pd

TOK = re.compile(r"<<([A-Za-z0-9_]+)>>")


def f4(x):
    return f"{x:.4f}"


def sgn(x, nd=4):
    s = f"{x:.{nd}f}"
    return f"$-${s[1:]}" if s.startswith("-") else s


def pct(x, nd=1):
    return f"{100 * x:.{nd}f}\\%"


def mm2(x):
    return f"{x:.2f}" if x >= 0.1 else (f"{x:.3f}" if x >= 0.01 else f"{x:.4f}")


def thousands(n):
    return f"{int(n):,}"


class Registry:
    def __init__(self, rev, precise, sicap):
        self.rev, self.precise, self.sicap = rev, precise, sicap
        self.vals = {}

    def put(self, name, value, source, field):
        self.vals[name] = (value, source, field)

    def csv(self, base, name):
        return pd.read_csv(base / name)


def build(reg: Registry):
    T = reg.rev / "revision_2026-10-30" / "tables"
    rel = lambda p: "PANDA_Reproducibility/" + str(Path(p).resolve().as_posix()).split("PANDA_Reproducibility/")[-1]

    # ---- A1 provider x level
    a1 = pd.read_csv(T / "A1_provider_level_metrics.csv")
    src = rel(T / "A1_provider_level_metrics.csv")
    pk = {"pooled": "p", "karolinska": "k", "radboud": "r"}
    for _, r in a1.iterrows():
        p = pk[r.provider]
        for col in ("dice", "precision", "recall"):
            reg.put(f"A1_{p}_{r['map']}_{col}", f4(r[col]), src, f"provider={r.provider}, map={r['map']}, {col}")
    tot = a1[(a1.provider == "pooled") & (a1["map"] == "fusion")].reference_tumor_pixels.iloc[0]
    k = a1[(a1.provider == "karolinska") & (a1["map"] == "fusion")].reference_tumor_pixels.iloc[0]
    reg.put("A1_k_tumor_share", pct(k / tot), src, "Karolinska reference tumor pixels / all")
    rows = []
    names = {"karolinska": "Karolinska", "radboud": "Radboud", "pooled": "Pooled"}
    for prov in ("karolinska", "radboud", "pooled"):
        for mp in ("L0", "L1", "L2", "fusion"):
            r = a1[(a1.provider == prov) & (a1["map"] == mp)].iloc[0]
            first = names[prov] + f" ($n={r.n_wsi}$)" if mp == "L0" else ""
            label = "Fusion" if mp == "fusion" else mp
            rows.append(f"{first} & {label} & {f4(r.dice)} & {f4(r.iou)} & {f4(r.precision)} & {f4(r.recall)} & "
                        f"{f4(r.specificity)} & {f4(r.mean_slide_dice_common_tumor_bearing)}\\\\")
        if prov != "pooled":
            rows.append("\\midrule")
    reg.put("TABLE_A1", "\n".join(rows) + "\n", src, "Table: provider x level metrics")

    d6 = pd.read_csv(T / "D6_primary_ci_10000.csv"); s6 = rel(T / "D6_primary_ci_10000.csv")
    f = d6[(d6["map"] == "fusion") & (d6.strata == "none")].iloc[0]
    fs = d6[(d6["map"] == "fusion") & (d6.strata == "provider")].iloc[0]
    reg.put("D6_p_lo", f4(f.ci_low), s6, "fusion, unstratified, ci_low"); reg.put("D6_p_hi", f4(f.ci_high), s6, "fusion, unstratified, ci_high")
    reg.put("D6_ps_lo", f4(fs.ci_low), s6, "fusion, provider-stratified, ci_low"); reg.put("D6_ps_hi", f4(fs.ci_high), s6, "fusion, provider-stratified, ci_high")
    for p, nm in (("k", "fusion, karolinska"), ("r", "fusion, radboud")):
        r = d6[d6["map"] == nm].iloc[0]
        reg.put(f"D6_{p}_lo", f4(r.ci_low), s6, f"{nm}, ci_low"); reg.put(f"D6_{p}_hi", f4(r.ci_high), s6, f"{nm}, ci_high")

    # ---- B4 / A4 paired
    def put_contrast(prefix, df, src, analysis, estimand, contrast):
        r = df[(df.analysis == analysis) & (df.estimand == estimand) & (df.contrast == contrast)].iloc[0]
        reg.put(prefix, sgn(r.difference), src, f"{analysis}; {estimand}; {contrast}; difference")
        reg.put(prefix + "_lo", sgn(r.ci_low), src, f"{analysis}; {estimand}; {contrast}; ci_low")
        reg.put(prefix + "_hi", sgn(r.ci_high), src, f"{analysis}; {estimand}; {contrast}; ci_high")
        return r
    b4c = pd.read_csv(T / "B4_paired_contrasts_common_support.csv"); sc = rel(T / "B4_paired_contrasts_common_support.csv")
    b4o = pd.read_csv(T / "B4_paired_contrasts_original_rule.csv"); so = rel(T / "B4_paired_contrasts_original_rule.csv")
    b4s = pd.read_csv(T / "B4_per_slide_rule_sensitivity.csv"); ss = rel(T / "B4_per_slide_rule_sensitivity.csv")
    a4 = pd.read_csv(T / "A4_validation_paired_contrasts.csv"); sa = rel(T / "A4_validation_paired_contrasts.csv")
    put_contrast("B4_c_pool_L1", b4c, sc, "common support", "pooled", "fusion-L1")
    put_contrast("B4_c_slide_L1", b4c, sc, "common support, tumor-bearing", "slide", "fusion-L1")
    put_contrast("B4_o_pool_L1", b4o, so, "original: own support", "pooled", "fusion-L1")
    put_contrast("B4_o_slide_L1", b4o, so, "original: own support, zero-coded", "slide", "fusion-L1")
    put_contrast("B4_s_slide_L1", b4s, ss, b4s.analysis.iloc[0], "slide", "fusion-L1")
    put_contrast("A4_v_c_pool", a4, sa, "validation, common support, 0.43", "pooled", "fusion-L1")
    put_contrast("A4_v_c_slide", a4, sa, "validation, common support, 0.43", "slide", "fusion-L1")
    put_contrast("A4_v_o_pool", a4, sa, "validation, common threshold 0.43", "pooled", "fusion-L1")
    put_contrast("A4_v_l_pool", a4, sa, "validation, level-specific thresholds (L1 0.40, fusion 0.43)", "pooled", "fusion-L1")
    pc = pd.read_csv(T / "A1_provider_paired_contrasts.csv"); spc = rel(T / "A1_provider_paired_contrasts.csv")
    for p, prov in (("k", "karolinska"), ("r", "radboud")):
        put_contrast(f"A1_pc_{p}_pool_L1", pc, spc, f"{prov}, common support", "pooled", "fusion-L1")
        put_contrast(f"A1_pc_{p}_slide_L1", pc, spc, f"{prov}, common support, tumor-bearing", "slide", "fusion-L1")

    def row(df, an, est, con, label, n=None):
        r = df[(df.analysis == an) & (df.estimand == est) & (df.contrast == con)].iloc[0]
        return (f"{label} & Fusion$-${con.split('-')[1]} & {thousands(r.n_wsi)} & {sgn(r.difference)} & "
                f"[{sgn(r.ci_low)}, {sgn(r.ci_high)}]; {r.p_holm.replace('<', '$<$')}\\\\")
    rows = []
    for con in ("fusion-L0", "fusion-L1", "fusion-L2"):
        rows.append(row(b4c, "common support", "pooled", con, "Test, pooled Dice" if con == "fusion-L0" else ""))
    for con in ("fusion-L0", "fusion-L1", "fusion-L2"):
        rows.append(row(b4c, "common support, tumor-bearing", "slide", con, "Test, per-slide Dice" if con == "fusion-L0" else ""))
    rows.append("\\midrule")
    rows.append(row(a4, "validation, common support, 0.43", "pooled", "fusion-L1", "Validation, pooled Dice"))
    rows.append(row(a4, "validation, common support, 0.43", "slide", "fusion-L1", "Validation, per-slide Dice"))
    rows.append("\\midrule")
    rows.append("\\multicolumn{5}{l}{\\textit{Each level on its own support (earlier analysis)}}\\\\")
    rows.append(row(b4o, "original: own support", "pooled", "fusion-L1", "Test, pooled Dice"))
    rows.append(row(b4o, "original: own support, zero-coded", "slide", "fusion-L1", "Test, per-slide Dice$^a$"))
    rows.append(row(a4, "validation, common threshold 0.43", "pooled", "fusion-L1", "Validation, pooled Dice"))
    rows.append(row(a4, "validation, level-specific thresholds (L1 0.40, fusion 0.43)", "pooled", "fusion-L1", "Validation, pooled Dice$^b$"))
    rows.append("\\midrule\\multicolumn{5}{l}{\\footnotesize $^a$ Slides without reference tumor in L1's own support scored Dice 0 for L1. $^b$ L1 at 0.40.}\\\\")
    reg.put("TABLE_B4", "\n".join(rows) + "\n", f"{sc}; {so}; {sa}", "Table: paired contrasts")

    cov = pd.read_csv(T / "B4_level_support_coverage.csv"); scov = rel(T / "B4_level_support_coverage.csv")
    for _, r in cov.iterrows():
        if r["map"] != "fusion":
            reg.put(f"B4_cov_{r['map']}_av", pct(r.share_av), scov, f"{r['map']} share of annotated pixels in support")
            reg.put(f"B4_cov_{r['map']}_tum", pct(r.share_reference_tumor), scov, f"{r['map']} share of reference tumor in support")
    own = a1[a1.provider == "pooled"].set_index("map")
    cmn = pd.read_csv(T / "B4_common_support_dice.csv").set_index("map") if (T / "B4_common_support_dice.csv").exists() else None
    rows = []
    for mp in ("L0", "L1", "L2", "fusion"):
        r = cov.set_index("map").loc[mp]
        rows.append(f"{'Fusion' if mp == 'fusion' else mp} & {f4(own.loc[mp, 'dice'])} & {f4(cmn.loc[mp, 'dice'])} & {100 * r.share_reference_tumor:.1f}\\\\")
    reg.put("TABLE_B4COV", "\n".join(rows) + "\n", f"{src}; {rel(T / 'B4_common_support_dice.csv')}; {scov}", "Table: own vs common support")

    de = pd.read_csv(T / "B2_design_effect.csv"); sde = rel(T / "B2_design_effect.csv")
    r = de[(de.estimand == "pooled")].iloc[0]
    reg.put("B2_de_pool", f"{r.design_effect_at_which_ci_reaches_zero:.1f}", sde, "design effect at which the pooled CI reaches 0")
    reg.put("B2_meff", f"{r.effective_cluster_size_in_test:.2f}", sde, "expected test WSIs per patient (max design effect at rho=1)")

    # ---- B6
    b6 = pd.read_csv(T / "B6_size_strata.csv"); s6b = rel(T / "B6_size_strata.csv")
    q1, q4 = b6.iloc[0], b6.iloc[-1]
    reg.put("B6_q1_max", mm2(q1.area_mm2_max), s6b, "Q1 upper area")
    reg.put("B6_q1_fus", f4(q1.mean_dice_fusion), s6b, "Q1 mean fused Dice")
    reg.put("B6_q1_diff", sgn(q1.fusion_minus_L1), s6b, "Q1 fusion-L1"); reg.put("B6_q1_lo", sgn(q1.ci_low), s6b, "Q1 ci_low"); reg.put("B6_q1_hi", sgn(q1.ci_high), s6b, "Q1 ci_high")
    reg.put("B6_q4_diff", sgn(q4.fusion_minus_L1), s6b, "Q4 fusion-L1"); reg.put("B6_q4_lo", sgn(q4.ci_low), s6b, "Q4 ci_low"); reg.put("B6_q4_hi", sgn(q4.ci_high), s6b, "Q4 ci_high")
    reg.put("B6_q1_r", str(int(q1.radboud)), s6b, "Q1 Radboud WSIs"); reg.put("B6_q1_n", str(int(q1.n_wsi)), s6b, "Q1 WSIs")
    rows = []
    for _, r in b6.iterrows():
        rows.append(f"{r.stratum.split(' ')[0]} ({mm2(r.area_mm2_min)}--{mm2(r.area_mm2_max)}) & {r.n_wsi} ({r.karolinska}/{r.radboud}) & "
                    f"{f4(r.mean_dice_fusion)} & {f4(r.mean_dice_L1)} & {sgn(r.fusion_minus_L1)} ({sgn(r.ci_low)} to {sgn(r.ci_high)}) & "
                    f"{r.complete_misses_fusion}/{r.complete_misses_L1}\\\\")
    reg.put("TABLE_B6", "\n".join(rows) + "\n", s6b, "Table: size strata")
    miss = pd.read_csv(T / "B6_complete_misses.csv"); sm = rel(T / "B6_complete_misses.csv")
    reg.put("B6_zero_extra", str(int((~miss.empty_prediction).sum())), sm, "zero-overlap WSIs with non-empty prediction")
    reg.put("B6_miss_min", mm2(miss.ref_area_mm2.min()), sm, "min reference area of zero-overlap WSIs")
    reg.put("B6_miss_max", mm2(miss.ref_area_mm2.max()), sm, "max reference area of zero-overlap WSIs")
    br = pd.read_csv(T / "B6_borderline_rule.csv").set_index("level"); sbr = rel(T / "B6_borderline_rule.csv")
    for lv in ("L0", "L1", "L2"):
        reg.put(f"B6_lim_{lv}", mm2(br.loc[lv, "five_percent_mm2"]), sbr, f"{lv} 5% of a fully annotated patch (mm2)")
    reg.put("B6_below_L1", pct(br.loc["L1", "share_of_tumor_bearing_wsi_with_all_tumor_below"]), sbr, "share of tumor-bearing WSIs below the L1 limit")
    reg.put("B6_below_L2", pct(br.loc["L2", "share_of_tumor_bearing_wsi_with_all_tumor_below"]), sbr, "share below the L2 limit")

    # ---- D3
    d3 = pd.read_csv(T / "D3_negative_slide_fp_by_provider.csv").set_index("provider"); s3 = rel(T / "D3_negative_slide_fp_by_provider.csv")
    g = pd.read_csv(T / "D3_negative_slides_by_grade.csv"); sg = rel(T / "D3_negative_slides_by_grade.csv")
    reg.put("D3_isup0", str(int(g[g.grade_group.str.startswith("ISUP 0")].n.sum())), sg, "reference-negative with ISUP 0")
    reg.put("D3_isup_pos", str(int(g[~g.grade_group.str.startswith("ISUP 0")].n.sum())), sg, "reference-negative with ISUP >= 1")
    for p, prov in (("k", "karolinska"), ("r", "radboud")):
        reg.put(f"D3_isup_pos_{p}", str(int(g[(g.provider == prov) & ~g.grade_group.str.startswith("ISUP 0")].n.sum())), sg, f"{prov} ISUP>=1")
        reg.put(f"D3_{p}_any", str(int(d3.loc[prov, "any_fp"])), s3, f"{prov} any FP"); reg.put(f"D3_{p}_n", str(int(d3.loc[prov, "n_reference_negative"])), s3, f"{prov} n")
        reg.put(f"D3_{p}_ge001", str(int(d3.loc[prov, "wsi_with_fp_focus_ge_0.01mm2"])), s3, f"{prov} WSIs with FP component >= 0.01 mm2")
        reg.put(f"D3_{p}_med_foci", f"{d3.loc[prov, 'fp_foci_median_among_fp']:.0f}", s3, f"{prov} median FP components among WSIs with FP")
    reg.put("D3_p_ge001", str(int(d3.loc["pooled", "wsi_with_fp_focus_ge_0.01mm2"])), s3, "pooled WSIs with FP component >= 0.01 mm2")
    reg.put("D3_p_med_area", mm2(d3.loc["pooled", "fp_area_mm2_median_among_fp"]), s3, "median FP area among WSIs with FP")
    reg.put("D3_p_p90_area", mm2(d3.loc["pooled", "fp_area_mm2_p90_among_fp"]), s3, "90th percentile FP area")
    reg.put("D3_p_max_area", mm2(d3.loc["pooled", "fp_area_mm2_max"]), s3, "max FP area")
    reg.put("D3_p_med_largest", mm2(d3.loc["pooled", "largest_focus_mm2_median_among_fp"]), s3, "median largest FP component")

    # ---- D5 / D2
    d5 = pd.read_csv(T / "D5_boundary_nonempty.csv").set_index("metric"); s5 = rel(T / "D5_boundary_nonempty.csv")
    reg.put("D5_hd_med_um", f"{d5.loc['HD95', 'median_um']:.0f}", s5, "HD95 median um"); reg.put("D5_hd_mean_um", f"{d5.loc['HD95', 'mean_um']:.0f}", s5, "HD95 mean um")
    reg.put("D5_as_med_um", f"{d5.loc['ASSD', 'median_um']:.0f}", s5, "ASSD median um"); reg.put("D5_sd_mean", f4(d5.loc["surface Dice", "mean"]), s5, "surface Dice mean")
    reg.put("D5_tol_um", f"{d5.loc['surface Dice tolerance (um), 2 grid px', 'median']:.0f}", s5, "surface-Dice tolerance (median um)")
    rows = []
    for metric, unit in (("HD95", "grid px"), ("ASSD", "grid px")):
        r = d5.loc[metric]
        rows.append(f"{metric} ({unit}) & {r['mean']:.2f} & {r['median']:.2f} & {r.mean_um:.0f} & {r.median_um:.0f}\\\\")
    r = d5.loc["surface Dice"]
    rows.append(f"Surface Dice & {f4(r['mean'])} & {f4(r['median'])} & -- & --\\\\")
    reg.put("TABLE_D5", "\n".join(rows) + "\n", s5, "Table: boundary metrics")
    rl = pd.read_csv(T / "A1_raw_label_prediction.csv"); srl = rel(T / "A1_raw_label_prediction.csv")
    def rl_share(prov, name_start):
        return rl[(rl.provider == prov) & rl.name.str.startswith(name_start)].predicted_tumor_fusion.iloc[0]
    reg.put("A1_r_stroma_in", pct(rl_share("radboud", "stroma inside")), srl, "Radboud stroma inside tumor envelope predicted tumor (fusion)")
    reg.put("A1_r_stroma_out", pct(rl_share("radboud", "stroma outside")), srl, "Radboud stroma outside envelope predicted tumor")
    reg.put("A1_r_benign", pct(rl_share("radboud", "benign epithelium")), srl, "Radboud benign epithelium predicted tumor")
    reg.put("A1_k_canc", pct(rl_share("karolinska", "cancerous")), srl, "Karolinska cancerous region predicted tumor")
    reg.put("A1_k_canc_missed", pct(1 - rl_share("karolinska", "cancerous")), srl, "Karolinska cancerous region not predicted")
    reg.put("A1_k_benign", pct(rl_share("karolinska", "benign")), srl, "Karolinska benign tissue predicted tumor")
    ink = pd.read_csv(T / "D13_pen_ink.csv").set_index("subset"); sink = rel(T / "D13_pen_ink.csv")
    reg.put("D13_n_ink", str(int(ink.loc["with ink", "n_wsi"])), sink, "Karolinska WSIs with ink-like pixels")
    reg.put("D13_dice_noink", f4(ink.loc["all Karolinska", "dice_without_pixels_within_32px_of_ink"]), sink, "Karolinska Dice without pixels near ink")
    pv = pd.read_csv(T / "D2_pixel_size_values.csv"); spv = rel(T / "D2_pixel_size_values.csv")
    kv = pv[pv.provider == "karolinska"].set_index("level_1").um_per_px_image
    reg.put("D2_k_n452", thousands(kv.loc[0.452]), spv, "Karolinska images at 0.452 um")
    reg.put("D2_k_n503", thousands(kv.loc[0.5032]), spv, "Karolinska images at 0.5032 um")
    px = pd.read_csv(T / "D2_pixel_size.csv").set_index("provider"); spx = rel(T / "D2_pixel_size.csv")
    for prov in ("radboud", "karolinska"):
        lo, hi = px.loc[prov, "um_image_min"], px.loc[prov, "um_image_max"]
        reg.put(f"D2_{prov}_um", f"{lo:.3f}" if abs(hi - lo) < 5e-4 else f"{lo:.3f}--{hi:.3f}", spx, f"{prov} level-0 um/px from TIFF tags (min-max)")

    # ---- SICAPv2
    S = reg.sicap
    sf = json.load(open(S / "SICAP_official_facts.json")); ssf = rel(S / "SICAP_official_facts.json")
    for k_, t in (("tiles_all", "S_tiles_all"), ("slides_all", "S_slides_all"), ("patients_all", "S_patients_all"),
                  ("tiles_test", "S_tiles_test"), ("slides_test", "S_slides_test"), ("patients_test", "S_patients_test"),
                  ("tiles_listed", "S_tiles_listed"), ("tiles_unlisted", "S_tiles_unlisted")):
        reg.put(t, thousands(sf[k_]), ssf, k_)
    reg.put("S_u_fp_share", pct(sf["unlisted_share_of_fp_pixels"]), ssf, "unlisted share of FP pixels")
    ss_ = pd.read_csv(S / "SICAP_official_summary.csv"); sss = rel(S / "SICAP_official_summary.csv")
    def srow(set_, unit, coords="const"):
        return ss_[(ss_.set == set_) & (ss_.unit_scored == unit) & (ss_.coords == coords)].iloc[0]
    t = srow("test_partition", "patch")
    reg.put("S_t_tile_dice", f4(t.dice), sss, "test partition, per patch, Dice"); reg.put("S_t_tile_prec", f4(t.precision), sss, "precision"); reg.put("S_t_tile_rec", f4(t.recall), sss, "recall")
    reg.put("S_t_tile_pat_lo", f4(t.ci_patient_low), sss, "patient CI low"); reg.put("S_t_tile_pat_hi", f4(t.ci_patient_high), sss, "patient CI high")
    t = srow("test_partition", "deduplicated pixel")
    reg.put("S_t_dd_dice", f4(t.dice), sss, "test partition, deduplicated, Dice")
    for u in ("pat", "wsi"):
        nm = "patient" if u == "pat" else "wsi"
        reg.put(f"S_t_dd_{u}_lo", f4(t[f"ci_{nm}_low"]), sss, f"dedup {nm} CI low"); reg.put(f"S_t_dd_{u}_hi", f4(t[f"ci_{nm}_high"]), sss, f"dedup {nm} CI high")
    reg.put("S_a_tile_dice", f4(srow("all", "patch").dice), sss, "all patches, per patch, Dice")
    reg.put("S_l_tile_dice", f4(srow("partition_listed", "patch").dice), sss, "listed, Dice"); reg.put("S_l_tile_prec", f4(srow("partition_listed", "patch").precision), sss, "listed precision")
    reg.put("S_u_tile_prec", f4(srow("unlisted", "patch").precision), sss, "unlisted precision")
    reg.put("S_coord_diff", sgn(srow("test_partition", "patch", "region").dice - srow("test_partition", "patch").dice), sss, "region minus constant coordinates, test partition, per patch")
    rows = []
    for set_, unit, label in (("test_partition", "patch", "Test partition"), ("test_partition", "deduplicated pixel", "Test partition, deduplicated"),
                              ("partition_listed", "patch", "Listed in partition files"), ("unlisted", "patch", "Not listed"),
                              ("all", "patch", "All patches"), ("all", "deduplicated pixel", "All patches, deduplicated")):
        r = srow(set_, unit)
        rows.append(f"{label} & {thousands(r.patches)} & {r.wsis} & {f4(r.dice)} ({f4(r.ci_patient_low)}--{f4(r.ci_patient_high)}) & "
                    f"{f4(r.precision)} & {f4(r.recall)} & {f4(r.specificity)}\\\\")
    reg.put("TABLE_SICAP", "\n".join(rows) + "\n", sss, "Table: SICAPv2")

    # ---- PRECISE (run reported in the manuscript; analyses of precise_first_run_analyses.py)
    P = reg.precise
    pci = pd.read_csv(P / "PRECISE_ci.csv"); spci = rel(P / "PRECISE_ci.csv")
    pj = json.load(open(P / "PRECISE_summary.json")); spj = rel(P / "PRECISE_summary.json")
    pp = pd.read_csv(P / "PRECISE_paired.csv").iloc[0]; spp = rel(P / "PRECISE_paired.csv")
    lab = pd.read_csv(P / "PRECISE_labels.csv").set_index("label"); slab = rel(P / "PRECISE_labels.csv")
    def c(mp, unit):
        return pci[(pci["map"] == mp) & (pci.unit == unit)].iloc[0]
    reg.put("P_f_subj_lo", f4(c("fusion", "subject").ci_low), spci, "fusion subject CI low"); reg.put("P_f_subj_hi", f4(c("fusion", "subject").ci_high), spci, "subject CI high")
    reg.put("P_f_wsi_lo", f4(c("fusion", "WSI").ci_low), spci, "fusion WSI CI low (10,000 resamples)"); reg.put("P_f_wsi_hi", f4(c("fusion", "WSI").ci_high), spci, "WSI CI high")
    reg.put("P_fl1_diff", sgn(pp.difference), spp, "fusion-L1"); reg.put("P_fl1_lo", sgn(pp.ci_low), spp, "ci_low"); reg.put("P_fl1_hi", sgn(pp.ci_high), spp, "ci_high")
    reg.put("P_neg_area_total", mm2(pj["neg_fp_area_mm2_total"]), spj, "tumor-free FP area mm2")
    reg.put("P_neg_comp", thousands(pj["neg_components_total"]), spj, "tumor-free FP components")
    reg.put("P_neg_comp_med", f"{pj['neg_components_median']:.0f}", spj, "median FP components per tumor-free WSI")
    reg.put("P_neg_maxcomp", mm2(pj["neg_largest_component_mm2"]), spj, "largest FP component mm2")
    for L in (2, 4, 5, 6, 7):
        reg.put(f"P_lab{L}_pred", pct(lab.loc[L, "predicted_tumor_share"]), slab, f"label {L} predicted-tumor share")
    rows = []
    for L, r in lab.iterrows():
        share = "--" if L == 0 else pct(r.share_of_annotated, 2)
        pred = "--" if pd.isna(r.predicted_tumor_share) else pct(r.predicted_tumor_share)
        short = {4: "HGPIN", 6: "Atypical intraductal prolif."}.get(L, r["class"])
        rows.append(f"{L} & {short} & {r['use']} & {thousands(r.pixels)} & {share} & {pred}\\\\")
    reg.put("TABLE_PRECISE_LABELS", "\n".join(rows) + "\n", slab, "Table: PRECISE labels")
    first = pd.read_csv(reg.rev / "external_precise_wsi/results/tables/external_wsi_metrics_per_level_and_fusion.csv").set_index("subset")
    sfirst = "PANDA_Reproducibility/external_precise_wsi/results/tables/external_wsi_metrics_per_level_and_fusion.csv"
    rows = [f"Pooled Dice (95\% CI by subject) & {f4(first.loc['fusion', 'dice'])} ({f4(c('fusion', 'subject').ci_low)}--{f4(c('fusion', 'subject').ci_high)})\\\\",
            f"Pooled Dice (95\% CI by WSI) & {f4(first.loc['fusion', 'dice'])} ({f4(c('fusion', 'WSI').ci_low)}--{f4(c('fusion', 'WSI').ci_high)})\\\\"]
    for k_, nm in (("iou", "IoU"), ("precision", "Precision"), ("recall", "Recall"), ("specificity", "Specificity"), ("accuracy", "Accuracy")):
        rows.append(f"{nm} & {f4(first.loc['fusion', k_])}\\\\")
    for lv in ("L0", "L1", "L2"):
        rows.append(f"{lv} pooled Dice & {f4(first.loc[lv, 'dice'])}\\\\")
    rows.append(f"Fusion minus L1 (95\% CI by subject) & {sgn(pp.difference)} ({sgn(pp.ci_low)} to {sgn(pp.ci_high)})\\\\")
    pw = pd.read_csv(reg.rev / "external_precise_wsi/results/tables/per_slide_wsi_metrics.csv")
    pw = pw[(pw.fusion_tp + pw.fusion_fn) > 0]
    rows.append(f"Mean / median Dice, tumor-bearing WSIs & {f4(pw.fusion_dice.mean())} / {f4(pw.fusion_dice.median())}\\\\")
    spw = "PANDA_Reproducibility/external_precise_wsi/results/tables/per_slide_wsi_metrics.csv"
    reg.put("TABLE_PRECISE", "\n".join(rows) + "\n", f"{sfirst}; {spci}; {spp}; {spw}", "Table: PRECISE")

    # ---- patch preparation (former Fig 2F-G): candidate grid of the Fig 2 WSI and selected training buckets
    pools = reg.rev / "manifests" / "coordinate_pools"
    fig2_wsi = "362d30ace92ce17128d176137362b8c2"  # the WSI of Fig 2 (figure_scripts/provenance/Fig02_preprocessing_sources.json)
    stats = {lv: json.loads((pools / f"{fig2_wsi}_L{lv}.stats.json").read_text()) for lv in (0, 1, 2)}
    sstats = "; ".join(rel(pools / f"{fig2_wsi}_L{lv}.stats.json") for lv in (0, 1, 2))
    z2_p = T / "Z2_extraction_counts.json"
    z2 = json.loads(z2_p.read_text())
    sel_q = reg.rev / "qc" / "05_select_final.json"
    stored = {(s["level"], s["category"]): s["available"] for s in json.loads(sel_q.read_text())["details"]["selection"]}
    cap = {stats[lv]["candidate_cap_per_class_slide_level"] for lv in (0, 1, 2)}
    assert len(cap) == 1
    cap = cap.pop()
    rows = ["\\multicolumn{4}{l}{\\textit{Candidate selection on the training WSI in Fig~\\ref{fig:prep}}}\\\\",
            "Level & L0 & L1 & L2\\\\\\midrule"]
    for key, lab in (("grid_positions_total", "Grid positions"), ("rejected_low_tissue", "Rejected: tissue fraction below threshold"),
                     ("rejected_low_valid_annotation", "Rejected: little annotated tissue"),
                     ("rejected_mixed_borderline", "Rejected: tumor fraction $>0$ and $<0.05$"),
                     ("eligible_normal_before_cap", "Eligible normal (tumor fraction 0)"),
                     ("eligible_tumor_before_cap", "Eligible tumor (tumor fraction $\\geq0.05$)"),
                     ("normal_candidates_after_cap", f"Pooled normal (at most {cap})"),
                     ("tumor_candidates_after_cap", f"Pooled tumor (at most {cap})")):
        rows.append(f"{lab} & " + " & ".join(thousands(stats[lv][key]) for lv in (0, 1, 2)) + "\\\\")
    for cat in ("normal", "tumor"):
        rows.append(f"Stored {cat} & " + " & ".join(thousands(z2["fig2_wsi_stored"][f"L{lv}_{cat}"]) for lv in (0, 1, 2)) + "\\\\")
    sel_p = reg.rev / "selected" / "train_selected.csv"
    sel = pd.read_csv(sel_p, usecols=["image_id", "level", "category", "provider"])
    b = sel.groupby(["level", "category", "provider"]).size()
    rows += ["\\midrule", "\\multicolumn{4}{l}{\\textit{Stored and selected training patches, all training WSIs}}\\\\",
             "& & \\multicolumn{2}{c}{Selected}\\\\",
             "Bucket & Stored & Karolinska / Radboud & Total\\\\\\midrule"]
    for lv in (0, 1, 2):
        for cat in ("normal", "tumor"):
            k_, r_ = int(b.get((lv, cat, "karolinska"), 0)), int(b.get((lv, cat, "radboud"), 0))
            rows.append(f"L{lv} {cat} & {thousands(stored[(lv, cat)])} & {thousands(k_)} / {thousands(r_)} & {thousands(k_ + r_)}\\\\")
    kt, rt = int(b.xs("karolinska", level="provider").sum()), int(b.xs("radboud", level="provider").sum())
    rows.append(f"Total & {thousands(sum(stored.values()))} & {thousands(kt)} / {thousands(rt)} & {thousands(kt + rt)}\\\\")
    reg.put("TABLE_PREP", "\n".join(rows) + "\n", f"{sstats}; {rel(z2_p)}; {rel(sel_q)}; {rel(sel_p)}",
            "Table: candidate selection of the Fig 2 WSI; stored and selected buckets")
    reg.put("Z2_cap", str(cap), sstats, "candidate_cap_per_class_slide_level")
    reg.put("Z2_max_stored", str(z2["stored_per_wsi_level_class_max"]), rel(z2_p), "largest number of stored patches per WSI, level and class")
    reg.put("Z2_L0n_stored", thousands(stored[(0, "normal")]), rel(sel_q), "available (stored) L0 normal training patches")

    # ---- pathologist review: S.E.D. statement and S2 Table
    rf_p = T / "review_facts.json"
    reg.put("SED_years", str(json.loads(rf_p.read_text())["sed_years_in_pathology_including_residency"]), rel(rf_p),
            "S.E.D. years in pathology including residency (statement of 31 Oct 2026)")
    s2_p = reg.rev / "supporting_information" / "S2_Table_summary.json"
    s2 = json.loads(s2_p.read_text())
    reg.put("S2_slides", str(s2["slides"]), rel(s2_p), "slides of the 100 reviewed SICAPv2 images")
    reg.put("S2_patients", str(s2["patients"]), rel(s2_p), "patients of the 100 reviewed SICAPv2 images")
    reg.put("S2_test_phrase", "none of them belongs" if s2["in_test_partition"] == 0 else f"{s2['in_test_partition']} of them belong",
            rel(s2_p), "reviewed SICAPv2 images in the official test partition")
    reg.put("S2_saved", thousands(s2["saved_result_folders"]), rel(s2_p), "first-run SICAPv2 images with saved per-image results")
    reg.put("S2_dice_low", str(s2["dice_below_0.80"]), rel(s2_p), "reviewed SICAPv2 images with first-run Dice < 0.80")

    # ---- SICAPv2 scored region (tissue rule)
    sr_p = reg.sicap / "SICAP_scored_region.csv"
    sr = pd.read_csv(sr_p).set_index("subset")
    for key, sub in (("t", "test partition"), ("a", "all tiles")):
        reg.put(f"Z6_{key}_unscored", pct(sr.loc[sub, "reference_unscored_share"]), rel(sr_p), f"{sub}: reference tumor pixels outside the scored region")
        reg.put(f"Z6_{key}_unscored_bright", pct(sr.loc[sub, "unscored_reference_bright_share"]), rel(sr_p),
                f"{sub}: bright (mean RGB > 215) share of the unscored reference pixels")
        reg.put(f"Z6_{key}_bright_scored", pct(sr.loc[sub, "reference_bright_scored_share"]), rel(sr_p),
                f"{sub}: share of bright reference pixels inside the scored region")
    lvl = sel.groupby(["level", "category"]).size()
    reg.put("PREP_n_sel", thousands(len(sel)), rel(sel_p), "rows of the selected training manifest")
    reg.put("PREP_bucket_min", thousands(lvl.min()), rel(sel_p), "smallest level-class bucket")
    reg.put("PREP_bucket_max", thousands(lvl.max()), rel(sel_p), "largest level-class bucket")
    reg.put("PREP_n_tumor", thousands((sel.category == "tumor").sum()), rel(sel_p), "selected tumor patches")
    reg.put("PREP_n_normal", thousands((sel.category == "normal").sum()), rel(sel_p), "selected normal patches")
    reg.put("PREP_n_wsi", thousands(sel.image_id.nunique()), rel(sel_p), "training WSIs with at least one selected patch")


def main():
    tpl, rev, precise, sicap, out = map(Path, sys.argv[1:6])
    out.mkdir(parents=True, exist_ok=True)
    reg = Registry(rev, precise, sicap)
    build(reg)
    text = tpl.read_text(encoding="utf-8")
    missing = sorted(set(TOK.findall(text)) - set(reg.vals))
    if missing:
        print("UNRESOLVED TOKENS:", missing)
    rows = []
    lines = text.split("\n")
    section = ""
    for i, ln in enumerate(lines, 1):
        msec = re.search(r"\\(sub)?section\*\{([^}]*)\}", ln)
        if msec:
            section = msec.group(2)
        for name in TOK.findall(ln):
            if name in reg.vals:
                v, src, field = reg.vals[name]
                rows.append({"token": name, "value": v.strip().replace("\n", " ")[:120] if not name.startswith("TABLE") else "(table rows)",
                             "section": section, "tex_line": i, "source_file": src, "field": field})
    filled = TOK.sub(lambda mt: reg.vals[mt.group(1)][0] if mt.group(1) in reg.vals else mt.group(0), text)
    (out / "PANDA_FINAL_WITH_FIGURES.tex").write_text(filled, encoding="utf-8", newline="\n")
    without = "\n".join(l for l in filled.split("\n") if not l.lstrip().startswith("\\includegraphics"))
    (out / "PANDA_FINAL_WITHOUT_FIGURES.tex").write_text(without, encoding="utf-8", newline="\n")
    pd.DataFrame(rows).to_csv(out / "NUMBER_SOURCES.csv", index=False)
    print(f"{len(rows)} token uses filled; {len(missing)} unresolved")


if __name__ == "__main__":
    main()
