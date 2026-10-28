# PANDA Reproducibility

Exports, manifests and scripts that accompany the manuscript on multiresolution
prostate tumor segmentation in PANDA biopsy whole-slide images.

## Study summary

- Dataset: PANDA prostate biopsy whole-slide images (WSIs)
- Verified image-mask pairs: 10,512
- WSI-disjoint split: 7,358 training, 1,577 validation, 1,577 held-out test
- Selected training patches: 80,000 (six level x class buckets)
- Network: EfficientNet-B4 encoder with U-Net-style decoder and coordinate embedding
- Input: 512 x 512 patches, stride 128, logical levels L0, L1 and L2 (downsample 1, 4, 16)
- Fused operating threshold: 0.43, selected on validation data only
- Held-out pooled fusion Dice: 0.8522

The partitions are WSI-disjoint; patient-level independence is not established
because PANDA does not provide suitable patient identifiers.

## Environment and paths

```bash
pip install -r PANDA_Reproducibility/requirements.txt   # pinned, from the executed environment
```

The pipeline takes its folders as arguments:

```bash
python PANDA_Pipeline/run_full_panda_paper.py --data-root <PANDA folder with train_images/ and train_label_masks/> --work-root <work folder>
```

The figure and analysis scripts read two environment variables:

| Variable | Meaning |
|---|---|
| `PANDA_DATA_ROOT` | work folder written by the pipeline (`patches/`, `selected/`, `manifests/`, `runs/`, `paper_results/`, `stain_target_train.png`) |
| `PANDA_ARTICLE_ROOT` | checkout of this repository |
| `PANDA_WORK_ROOT` | work folder for `PANDA_B_ANALYSIS_ONE_CLICK.py` and the SICAPv2 script (same folder as `PANDA_DATA_ROOT`) |

The `D:\...` paths left in `PANDA_Pipeline/` are the executed defaults. Each is
remapped through these variables (`_panda_portable_path`) or overridden by a
command-line option; see `HARDCODED_PATHS_SUMMARY.csv`. In the SICAPv2 script,
`PANDA_DATA_ROOT` also sets `--data-root`, which that script does not read; pass
`--data-root` explicitly if you set the variable to the work folder.

## Manuscript figures

All 17 figures and S1 Table are drawn by `figure_scripts/` from the study data;
nothing is typed in or drawn by hand. The figures and captions carry no case
identifiers, patch coordinates or file names; these are kept in the logs
described below.

```bash
python figure_scripts/run_all_figures.py <PANDA_DATA_ROOT>   # all figures, S1 Table, logs, hash check
python figure_scripts/provenance_index.py figure_scripts/provenance <figures_dir> <PANDA_DATA_ROOT> --verify
```

| Figure | Produced by | Source data |
|---|---|---|
| Fig 1, 2 | `figure_scripts/make_figures_1_2_4_7.py` | stored training patches, masks and coordinate pools of one training WSI; `selected/train_selected.csv` |
| Fig 3 | `figure_scripts/make_figure_3.py` | nested stored L0/L1/L2 tumor patches and masks of one training WSI |
| Fig 4 | `figure_scripts/make_figures_1_2_4_7.py` | one L0 training patch and its training-time Vahadane copy (`cache/stain_v10/vahadane/`); `stain_target_train.png` |
| Fig 5 | `figure_scripts/make_figure_5.py` | one selected L0 training patch (training-time Vahadane copy) and the transforms of `PairAugStainHeavy` |
| Fig 6 | `figure_scripts/make_figures_6_8_9_10_12.py` | `selected/train_selected.csv` |
| Fig 7 | `figure_scripts/make_figures_1_2_4_7.py` | one L1 training patch; `runs/final_main_training/best.pt` |
| Fig 8 | `figure_scripts/make_figures_6_8_9_10_12.py` | `runs/final_main_training/history.csv` |
| Fig 9 | `figure_scripts/make_figures_6_8_9_10_12.py` | `paper_results/tables/per_slide_wsi_metrics.csv` |
| Fig 10 | `figure_scripts/make_figures_6_8_9_10_12.py` | `paper_results/tables/patch_metrics_per_level_and_pooled.csv` |
| Fig 11 | `figure_scripts/make_figures_11_16_17.py` | `paper_results/test_visuals_bad_review/` (one reviewed test WSI, S1 Table) and its WSI cache |
| Fig 12 | `figure_scripts/make_figures_6_8_9_10_12.py` | `paper_results/tables/threshold_sensitivity_validation.csv`, `best_threshold.json` |
| Fig 13 | `figure_scripts/GENERATE_FIG13_LITERATURE.py` | values of the cited studies; `paper_results/tables/global_wsi_metrics_95CI.csv` |
| Fig 14, 15 | `figure_scripts/make_figures_14_15.py` | `paper_results/test_visuals_bad_review/` (WSIs in S1 Table) and their WSI caches; one SICAPv2 image |
| Fig 16 | `figure_scripts/make_figures_11_16_17.py` | SICAPv2 images, masks, predictions and error maps of the external evaluation |
| Fig 17 | `figure_scripts/make_figures_11_16_17.py` | PRECISE WSI caches and saved visuals |

`figure_scripts/provenance/` holds one JSON log per figure. Each log records:

- the output files and the script, each with its SHA-256;
- the command and the software versions;
- per panel: the case (slide ID, provider, split), the level and level-0
  coordinates of each patch, and the image, tumor-mask and valid-mask files;
- for review crops, their level-0 field, located exactly in the WSI cache;
- every source file, relative to `PANDA_DATA_ROOT`, with its SHA-256;
- the parameters and the values printed in the figure.

`FIGURE_PROVENANCE_AUDIT.csv` (one row per figure) and
`FIGURE_PROVENANCE_DETAILED.csv` (one row per source file) are built from these
logs and are also copied to this folder. `FIGURE_PROVENANCE.md` is a readable
index.

Figs 6, 8, 9, 10, 12 and 13 run from the files in this repository. The image
figures also need the pipeline work folder, which holds the stored patches,
the training-time stain cache and the full WSI caches.

The `PANDA_Pipeline/GENERATE_*.py` figure scripts are superseded. The Fig 8, 10
and 12 versions there use typed-in values.

## Manuscript tables

| Table | Output file | Written by |
|---|---|---|
| 1 Cohort | `paper_results/tables/dataset_counts_verified.json`, `split_ids_reproducibility.csv` | `PANDA_Pipeline/run_full_panda_paper.py` (`panda_stream_paper.py`) |
| 2 Pilot configurations | `paper_results/tables/pilot_ablation_results.csv` | `run_full_panda_paper.py` (pilot stage) |
| 3 Encoder comparison | `paper_results/tables/pilot_encoder_comparison.csv` | `run_full_panda_paper.py` (pilot stage) |
| 4 WSI metrics | `paper_results/tables/wsi_metrics_per_level_and_fusion.csv`, `global_wsi_metrics_95CI.csv` | `panda_stream_paper.py` |
| 5 Paired bootstraps | `paper_results/B_followup_analyses/tables/B1_*.csv`, `B2_*.csv` | `PANDA_Pipeline/PANDA_B_ANALYSIS_ONE_CLICK.py` |
| 6 Common support | `paper_results/B_followup_analyses/tables/B4_*.csv` | `PANDA_B_ANALYSIS_ONE_CLICK.py` |
| 7 Patch audit | `paper_results/tables/patch_metrics_per_level_and_pooled.csv` | `panda_stream_paper.py` |
| 8 Boundary metrics | `paper_results/tables/boundary_metrics_summary.csv` | `panda_stream_paper.py` |
| 9 Level-specific thresholds | `paper_results/tables/level_specific_thresholds_table9.csv`, `level_specific_threshold_sweep_validation.csv` | `analysis_scripts/rebuild_level_thresholds.py` |
| 10 Provider metrics | `paper_results/tables/provider_metrics_95CI.csv` | `panda_stream_paper.py` |
| 11 Provider transfer | `paper_results/tables/pilot_leave_one_provider_out.csv` | `run_full_panda_paper.py` (pilot stage) |
| 12 Literature context | values from the cited studies | |
| 13 SICAPv2 | `external_sicapv2/results/tables/global_external_metrics*.{json,csv}` | `PANDA_Pipeline/run_external_sicapv2_test_LIVE_VISUALS.py` |
| 14 PRECISE | `external_precise_wsi/results/tables/*.csv` | `PANDA_Pipeline/run_external_precise_wsi_test.py` |

S1 Table (the 30 WSIs of the pathologist-guided review) is in
`supporting_information/` and is produced by `figure_scripts/make_S1_table.py`.

## Analysis scripts

`analysis_scripts/` rebuilds results from the saved WSI caches and manifests
(`python <script> <output_dir>` with `PANDA_DATA_ROOT` set):

| Script | Rebuilds | Check against archived outputs |
|---|---|---|
| `rebuild_level_thresholds.py` | Table 9: validation sweep 0.30–0.70 for L0, L1, L2 and fusion; validation-selected thresholds applied to test | fused sweep identical to `threshold_sensitivity_validation.csv`; test counts at 0.43 identical to `per_slide_wsi_metrics.csv` for all 1,577 WSIs and all four maps |
| `coverage_from_caches.py` | `coverage_analysis/` | identical files |
| `provider_tumor_definition.py` | `paper_results/tables/provider_tumor_definition_test_patches.csv` (share of test tumor patches labeled tumor over all annotated-valid pixels; third limitation) | |
| `make_cache_archives.py` | the two cache archives of the release above, with their SHA-256 manifest | |

## Known gaps

- **WSI caches.** `paper_results/wsi_cache/` here is partial (951 validation and
  1,564 test caches of 1,577 each). The missing PANDA caches and all 27 PRECISE
  caches are in the release `panda-wsi-cache-complement-20261027`
  (`panda_wsi_cache_complement.zip`, `precise_wsi_cache.zip`, SHA-256 manifest);
  several exceed GitHub's 100 MB file limit. Extract both archives here to get
  the full cache sets that B4 and the first two analysis scripts need.
- **B4 precision.** The caches store probabilities as float16. The original
  evaluation (`panda_stream_paper.py`) cast them to float32 before applying
  0.43, so float16(0.43) = 0.4299316 scored negative. B4 compared the float16
  array with 0.43 directly; NumPy then casts the threshold to float16 and the
  same pixels score positive. This adds 5,135 TP and 6,821 FP fused pixels in
  total (at most 70 per WSI, 0.034% of predicted pixels) and never removes a
  positive. Pooled fusion Dice is 0.852160 against 0.852159. With the float32
  cast, the caches reproduce `per_slide_wsi_metrics.csv` exactly.
- **Stale config.** `paper_results/reproducibility_config.json` keeps the planned
  defaults (for example, early-stopping patience 7). The executed training
  settings are in `runs/final_main_training/config.json` (patience 5, 30 epochs,
  batch 8, lr 2e-4, weight decay 1e-4).
- **Original data.** Original PANDA WSIs and masks are not redistributed:
  https://www.kaggle.com/competitions/prostate-cancer-grade-assessment/data
- **External results.** The SICAPv2 evaluation used an all-images fallback index,
  not the official test partition. PRECISE was evaluated with a separate full-WSI
  protocol. Neither is interchangeable with the PANDA held-out evaluation.
