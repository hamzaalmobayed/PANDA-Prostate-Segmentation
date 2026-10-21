# PANDA Reproducibility

<!-- PANDA_REPRODUCIBILITY_INDEX_V1 -->

## Reproducibility artifact index

The following exports accompany the PANDA multiresolution
prostate tumor segmentation manuscript.

### Dataset and evaluation

- Dataset: PANDA prostate biopsy whole-slide images
- Verified image-mask pairs: 10,512
- Training WSIs: 7,358
- Validation WSIs: 1,577
- Held-out test WSIs: 1,577
- Selected training patches: 80,000
- Network: EfficientNet-B4 encoder with U-Net-style decoder
- Input: 512 x 512 patches
- Patch stride: 128
- Logical scales: L0, L1 and L2
- Primary frozen fusion threshold: 0.43
- Reported held-out pooled fusion Dice: 0.8522

The study uses WSI-disjoint partitions; patient-level
independence is not established by these partitions.

### Analysis outputs

| Analysis | File |
|---|---|
| Paired pooled Dice bootstrap | `paper_results/B_followup_analyses/tables/B1_paired_bootstrap_pooled_dice.csv` |
| Tumor-bearing paired Dice bootstrap | `paper_results/B_followup_analyses/tables/B2_paired_bootstrap_tumor_only_mean_dice.csv` |
| Per-slide Dice summaries | `paper_results/B_followup_analyses/tables/B2_per_slide_dice_summaries.csv` |
| Benign false-positive thresholds | `paper_results/B_followup_analyses/tables/B2_B3_negative_slide_fp_thresholds.csv` |
| Common-support per-slide evaluation | `paper_results/B_followup_analyses/tables/B4_common_support_per_slide.csv` |
| Common-support pooled evaluation | `paper_results/B_followup_analyses/tables/B4_common_support_pooled.csv` |
| Selected training patches | `selected/train_selected.csv` |

### Figure provenance

See `FIGURE_PROVENANCE_AUDIT.csv`.

An entry is verified only after the source WSI/patch ID,
annotation, model prediction, generation script and
parameters are independently checked. Unverified entries
must not be treated as evidence of figure provenance.

### Software versions

Use the exact dependency versions from the executed
experimental environment, rather than guessing package
versions from the manuscript.

### Source datasets

PANDA: https://www.kaggle.com/competitions/prostate-cancer-grade-assessment/data

Original PANDA WSIs and masks are not redistributed
as part of these analysis exports.

### Important external-evaluation caveat

The manuscript's SICAPv2 evaluation used an all-images
fallback index, not the official test partition.
PRECISE was evaluated using a separate full-WSI protocol.
These results are not directly interchangeable with
the PANDA held-out evaluation.

<!-- PANDA_A2_FIGURE_DOCUMENTATION -->

## Figure and script traceability

- `FIGURE_PROVENANCE_DETAILED.csv`: candidate generation
  scripts, image filenames, SHA-256 values, and unresolved
  source identifiers for the manuscript figures.
- `HARDCODED_PATHS_SUMMARY.csv`: scripts containing
  machine-specific absolute paths.
- `CHANGELOG.md`: reproducibility work and limitations.

The image-producing scripts are not assumed to regenerate
the final assembled figure without further verification.
Manual assembly and labeling may have been performed with
Microsoft Paint.

Detailed source-case identities, reference annotations,
predictions and actual executed commands must be checked
before any figure is marked VERIFIED.
