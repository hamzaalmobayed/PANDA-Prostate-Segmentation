# PANDA reproducibility record

This directory is generated automatically by the pipeline.

- Random seed: 42
- Best model checkpoint: D:\PANDA_PROSTATE\runs\final_main_training\best.pt
- Validation-selected threshold: 0.4300
- Test set is held out from model/threshold selection.
- WSI results are reconstructed from retained original Test WSIs by dense streaming.
- Fusion is support-aware across L0/L1/L2.
- Vahadane reference is fitted/selected from Train only; the same fixed reference is applied at Validation/Test inference.
- `tables/split_ids_reproducibility.csv` contains exact slide IDs and split membership.
- `tables/dataset_counts_verified.json` contains counts recomputed from the generated split, not hard-coded manuscript numbers.
- `reproducibility_versions.json` records software/CUDA versions.
- `reproducibility_config.json` records the run configuration.
- `tables/per_slide_wsi_metrics.csv` exposes raw slide-level points behind means/SD/CIs.
- Boundary distances are reported in evaluation-grid pixels and level-0-equivalent pixels; no mm conversion is imposed.
- CAMELYON-only ITC/micro/macro/FROC analyses are intentionally not transplanted to prostate biopsy segmentation.

## Post-pilot finalization
- Pilot experiments finished before final reviewer-proof ablation statistics were consolidated.
- Paired two-sided bootstrap, effect sizes, and Holm-corrected p-values are in `stats/pilot_major_comparisons_two_sided_holm.csv`.
- This finalization step does not rerun training or the dense WSI held-out test.
