SICAPv2 EXTERNAL VALIDATION — COMPLETE

Dataset: SICAPv2 v2
Public source: https://data.mendeley.com/datasets/9xxm58dvs3/2
License: CC BY 4.0
Official external test images evaluated: 18765
Source-WSI groups: 18765

IMPORTANT SCIENTIFIC NOTE
SICAPv2 is distributed as 512x512 10x histology patches with per-pixel Gleason-pattern masks.
The PANDA model is applied without retraining. The frozen PANDA best checkpoint, frozen PANDA
validation threshold, fixed TRAIN-derived Vahadane target, and coordinate-aware network are reused.
Any nonzero SICAP Gleason-pattern mask pixel is mapped to binary tumor. Metrics are restricted
to H&E-derived valid tissue. No SICAPv2 labels are used to tune threshold or model weights.

OUTPUTS
- tables/per_external_image_metrics.csv
- tables/per_source_wsi_aggregated_metrics.csv
- tables/global_external_metrics.json
- tables/global_external_metrics_95CI_source_wsi_bootstrap.csv
- figures/external_confusion_matrix.png
- figures/per_image_dice_distribution.png
- figures/source_wsi_dice.png
- visuals_all_images/<source_wsi>/<image_id>/combined_original_gt_model_error.png

The combined PNG preserves the original 512x512 H&E panel pixel-for-pixel and uses lossless PNG.
