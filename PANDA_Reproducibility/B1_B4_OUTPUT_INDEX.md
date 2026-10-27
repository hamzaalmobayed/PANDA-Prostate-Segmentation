# PANDA B1–B4 Reproducibility Exports

All outputs below are written by `PANDA_Pipeline/PANDA_B_ANALYSIS_ONE_CLICK.py`
(functions `b1`–`b4`). They use frozen predictions; no model is retrained and no
test threshold is tuned.

| Analysis | Output |
|---|---|
| Paired bootstrap, pooled Dice (10,000 resamples, seed 42) | `paper_results/B_followup_analyses/tables/B1_paired_bootstrap_pooled_dice.csv` |
| Paired bootstrap, tumor-bearing mean Dice (10,000 resamples, seed 43) | `paper_results/B_followup_analyses/tables/B2_paired_bootstrap_tumor_only_mean_dice.csv` |
| Per-slide Dice summaries | `paper_results/B_followup_analyses/tables/B2_per_slide_dice_summaries.csv` |
| Reference-negative false-positive thresholds | `paper_results/B_followup_analyses/tables/B2_B3_negative_slide_fp_thresholds.csv` |
| Common-support per-slide scores | `paper_results/B_followup_analyses/tables/B4_common_support_per_slide.csv` |
| Common-support pooled scores | `paper_results/B_followup_analyses/tables/B4_common_support_pooled.csv` |
| Selected training patch manifest | `selected/train_selected.csv` |

B1–B3 read `paper_results/tables/per_slide_wsi_metrics.csv`. B4 reads the 1,577
test caches in `paper_results/wsi_cache/`; the copy in this repository is partial,
so B4 needs the full cache set.

**B4 precision.** The caches store probabilities as float16. B4 applies
`prob >= 0.43` to the float16 array, so NumPy compares against float16(0.43) =
0.4299316. The original evaluation cast the maps to float32 first, so pixels
stored as 0.4299316 scored negative there. B4 therefore gains 5,135 TP and 6,821
FP fused pixels in total (at most 70 per WSI) and never loses a positive. Pooled
fusion Dice is 0.852160 against 0.852159. Both files can be reproduced exactly
from the caches: B4 as written, `per_slide_wsi_metrics.csv` with the float32 cast
(`analysis_scripts/rebuild_level_thresholds.py` checks this).

Original source WSIs and annotations are not redistributed here.
