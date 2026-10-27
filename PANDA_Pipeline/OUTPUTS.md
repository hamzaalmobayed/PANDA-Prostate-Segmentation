
# Main patch hierarchy

Each L0/L1/L2 directory contains:
- Normal/
- Normal_Masks/
- Normal_Valid_Masks/
- Tumor/
- Tumor_Masks/
- Tumor_Valid_Masks/

This structure exists under:
- patches/train/<image_id>/
- patches/val/<image_id>/
- patches/test/<image_id>/

## Reconstruction metadata
The filename itself contains:
- image_id
- split
- logical level
- x0/y0 baseline WSI coordinates
- downsample
- patch size
- stride
- class

The manifests additionally contain the full metadata and file paths.

## Resume files
- pipeline_state.json
- manifests/coordinate_pools/*.progress.json
- manifests/materialize_shards/*.csv
- features/*_feature_shards/
- runs/final_main_training/resume_training.pt
- runs/final_main_training/train_phase_complete.pt
- runs/final_main_training/val_progress.json


## V16 reviewer-proof evaluation outputs
- `paper_results/tables/per_slide_wsi_metrics.csv` — now also includes valid/GT/predicted pixel denominators and boundary metrics.
- `paper_results/tables/boundary_metrics_summary.csv`
- `paper_results/stats/major_wsi_comparisons_two_sided_holm.csv`
- `paper_results/stats/provider_difference_bootstrap.json`
- `paper_results/stats/pilot_major_comparisons_two_sided_holm.csv` (after pilots)
- `paper_results/reproducibility_versions.json`
- `paper_results/reproducibility_config.json`
- `paper_results/README_REPRODUCIBILITY.md`
- `paper_results/tables/dataset_counts_verified.json`
- `paper_results/tables/split_ids_reproducibility.csv`
