"""Write S1 Appendix (LaTeX) of the revised manuscript from the result files.

S1 Appendix holds the material moved out of the main text (implementation and
audit details, the stain target, the augmentation figure, the patch-level
audit) and the supporting sensitivity tables. Every number is read from a
repository file named next to its table.

Usage: python make_s1_appendix.py <PANDA_Reproducibility> <sicap_summary_dir> <figures_dir> <out.tex>
"""
import json
import sys
from pathlib import Path

import pandas as pd

f4 = lambda x: f"{x:.4f}"


def sg(x):
    s = f"{x:.4f}"
    return f"$-${s[1:]}" if s.startswith("-") else s


def table(caption, cols, header, rows, source):
    return ("\\begin{table}[H]\\centering\\footnotesize\n\\caption{\\textbf{" + caption + "} Source: \\texttt{"
            + source.replace("_", "\\_") + "}.}\n\\begin{tabular}{" + cols + "}\\toprule\n" + header + "\\\\\\midrule\n"
            + "\n".join(r + "\\\\" for r in rows) + "\n\\bottomrule\\end{tabular}\\end{table}\n\n")


def main():
    R, S, F, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
    T = R / "revision_2026-10-30" / "tables"
    h = pd.read_csv(R / "runs/final_main_training/history.csv") if (R / "runs/final_main_training/history.csv").exists() else None
    body = []
    body.append("\\section*{S1 Appendix. Implementation details, patch-level audit, and supporting analyses}\n")
    body.append("\\subsection*{Implementation and audit details}\n")
    txt = ("Candidate patch positions were checkpointed so that extraction could resume after storage or power interruptions. "
           "Training used channels-last memory layout and persistent data-loader workers. Epoch numbering in the training history "
           "begins at 1.")
    if h is not None:
        b = h.loc[h.val_dice.idxmax()]; l = h.loc[h.val_loss.idxmin()]
        txt += (f" The selected checkpoint had patch-level validation Dice {b.val_dice:.8f} at epoch {int(b.epoch)}; the minimum "
                f"validation loss, {l.val_loss:.6f}, occurred at epoch {int(l.epoch)} (\\texttt{{runs/final\\_main\\_training/history.csv}}).")
    body.append(txt + "\n\n")
    body.append("\\begin{figure}[H]\\centering\\includegraphics[width=0.35\\linewidth]{stain_target_train.png}\n"
                "\\caption{\\textbf{Training-derived Vahadane target.} The fixed target image used for training, validation, test, and WSI inference.}\\end{figure}\n\n")
    body.append("\\begin{figure}[H]\\centering\\includegraphics[width=\\linewidth]{Fig05_augmentation.png}\n"
                "\\caption{\\textbf{Training augmentations.} Seven configured transforms, each at its strongest setting, applied to one normalized L0 training patch.}\\end{figure}\n\n")

    pm = pd.read_csv(R / "paper_results/tables/patch_metrics_per_level_and_pooled.csv")
    rows = [f"{r.subset.replace('_', ' ')} & {f4(r.dice)} & {f4(r.iou)} & {f4(r.precision)} & {f4(r.recall)} & {f4(r.specificity)} & {f4(r.accuracy)}"
            for r in pm.itertuples()]
    body.append("\\subsection*{Patch-level audit}\nThe audit scores the stored test patches (at most two per class, level, and WSI). "
                "Overlapping patches count tissue locations repeatedly, and the L0 patches contribute 93.8\\% of the audited pixels, "
                "so the pooled value is dominated by L0. These values are not comparable with the reconstructed WSI results.\n\n")
    body.append(table("Patch-level audit on the stored test patches.", "lrrrrrr", "Subset & Dice & IoU & Precision & Recall & Specificity & Accuracy",
                      rows, "paper_results/tables/patch_metrics_per_level_and_pooled.csv"))
    body.append("\\begin{figure}[H]\\centering\\includegraphics[width=\\linewidth]{Fig10_confusion.png}\n"
                "\\caption{\\textbf{Patch-level confusion matrices.} (A) Pooled across levels, (B) L0, (C) L1, and (D) L2.}\\end{figure}\n\n")

    pc = pd.read_csv(T / "A1_provider_paired_contrasts.csv")
    rows = [f"{r.analysis.split(',')[0].capitalize()} & {'pooled' if r.estimand == 'pooled' else 'per slide'} & {r.contrast.replace('fusion-', 'Fusion$-$')} & {r.n_wsi} & {sg(r.difference)} & [{sg(r.ci_low)}, {sg(r.ci_high)}] & {r.p_holm.replace('<', '$<$')}"
            for r in pc.itertuples()]
    body.append("\\subsection*{Supporting analyses}\n")
    body.append(table("Paired comparisons of fusion and single levels within each provider (common scoring region, 10,000 WSI resamples).",
                      "lllrrll", "Provider & Estimand & Contrast & WSI $n$ & Difference & 95\\% CI & Holm $p$", rows,
                      "revision_2026-10-30/tables/A1_provider_paired_contrasts.csv"))
    de = pd.read_csv(T / "B2_design_effect.csv")
    rows = [f"{'pooled' if r.estimand == 'pooled' else 'per slide'} & {sg(r.difference)} & {r.rho:.2f} & {r.design_effect:.2f} & [{sg(r.inflated_ci_low)}, {sg(r.inflated_ci_high)}] & {r.design_effect_at_which_ci_reaches_zero:.2f}"
            for r in de.itertuples()]
    body.append(table("Design-effect sensitivity of the fusion-minus-L1 contrasts (common scoring region). The variance is inflated by $1+(m-1)\\rho$ with $m$ the expected number of test WSIs per patient.",
                      "lrrrlr", "Estimand & Difference & $\\rho$ & Design effect & Inflated 95\\% CI & Design effect for CI to reach 0", rows,
                      "revision_2026-10-30/tables/B2_design_effect.csv"))
    fd = pd.read_csv(T / "B6_foci_detection.csv")
    rows = [f"{r.provider.capitalize()} & {r.size_mm2} & {r.n_foci:,} & {100 * r.detected_fusion:.1f} & {100 * r.detected_L1:.1f}" for r in fd.itertuples()]
    body.append(table("Detection of reference tumor components (8-connected on the evaluation grid) by size. A component counts as detected when the map predicts tumor on any of its pixels (common scoring region). On Radboud WSIs, components are mostly single glands, not lesions.",
                      "llrrr", "Provider & Area (mm$^2$) & Components & Detected by fusion (\\%) & Detected by L1 (\\%)", rows,
                      "revision_2026-10-30/tables/B6_foci_detection.csv"))
    ms = pd.read_csv(T / "B6_complete_misses.csv")
    rows = [f"{r.provider.capitalize()} & {r.ref_area_mm2:.4f} & {int(r.n_reference_foci)} & {'yes' if r.empty_prediction else 'no'} & {int(r.L0_common_tp)} & {int(r.L1_common_tp)} & {int(r.L2_common_tp)}"
            for r in ms.sort_values("ref_area_mm2").itertuples()]
    body.append(table("Tumor-bearing test WSIs without any true-positive fused pixel.", "lrrlrrr",
                      "Provider & Reference tumor (mm$^2$) & Components & Empty prediction & L0 TP & L1 TP & L2 TP", rows,
                      "revision_2026-10-30/tables/B6_complete_misses.csv"))
    gr = pd.read_csv(T / "D3_negative_slides_by_grade.csv")
    rows = [f"{r.provider.capitalize()} & {r.grade_group} & {r.n} & {r.any_fp} & {r.fp_area_mm2_median:.4f}" for r in gr.itertuples()]
    body.append(table("Reference-negative test WSIs by PANDA slide-level ISUP grade.", "llrrr",
                      "Provider & Slide-level grade & WSIs & With false positives & Median false-positive area (mm$^2$)", rows,
                      "revision_2026-10-30/tables/D3_negative_slides_by_grade.csv"))
    bs = pd.read_csv(R / "paper_results/tables/boundary_metrics_summary.csv")
    body.append(table("Boundary metrics of the fused map with the implementation's empty-boundary convention (all defined WSIs).",
                      "l" + "r" * (len(bs.columns) - 1), " & ".join(c.replace("_", " ") for c in bs.columns),
                      [" & ".join(f"{v:.4g}" if isinstance(v, float) else str(v).replace("_", " ") for v in r) for r in bs.itertuples(index=False)],
                      "paper_results/tables/boundary_metrics_summary.csv"))
    old = json.load(open(R / "external_sicapv2/results/tables/global_external_metrics.json"))
    rows = [f"{k} & {f4(old[k])}" for k in ("dice", "iou", "precision", "recall", "specificity", "accuracy")]
    body.append(table("First SICAPv2 run on 18,765 patches of a renamed Kaggle copy (all patches of its Train folder; constant coordinate). Replaced in the main text by the evaluation of the official release.",
                      "lr", "Metric & Value", rows, "external_sicapv2/results/tables/global_external_metrics.json"))
    ss = pd.read_csv(S / "SICAP_official_summary.csv")
    rows = [f"{r.set.replace('_', ' ')} & {r.unit_scored} & {'constant' if r.coords == 'const' else 'region-derived'} & {f4(r.dice)} & {f4(r.precision)} & {f4(r.recall)}"
            for r in ss.itertuples()]
    body.append(table("SICAPv2 official release: all patch sets, per patch and deduplicated, with constant and region-derived coordinates.",
                      "lllrrr", "Patch set & Scored unit & Coordinates & Dice & Precision & Recall", rows,
                      "external_sicapv2_official/summary/SICAP_official_summary.csv"))
    doc = ("\\documentclass[10pt]{article}\n\\usepackage[utf8]{inputenc}\\usepackage[T1]{fontenc}\\usepackage{lmodern}"
           "\\usepackage[margin=1in]{geometry}\\usepackage{graphicx}\\usepackage{booktabs}\\usepackage{float}\\usepackage{amsmath}\n"
           "\\usepackage[labelfont=bf,labelsep=period]{caption}\\renewcommand{\\figurename}{S1 Appendix, Fig}\\renewcommand{\\tablename}{S1 Appendix, Table}\n"
           "\\graphicspath{{" + F.as_posix() + "/}}\n\\begin{document}\n" + "".join(body) + "\\end{document}\n")
    out.write_text(doc, encoding="utf-8", newline="\n")
    print("written", out)


if __name__ == "__main__":
    main()
