# PANDA Reproducibility Changelog

## 28 October 2026

### Figures
- All 17 manuscript figures are now drawn by `figure_scripts/` from the study
  data. `run_all_figures.py` rebuilds them with S1 Table, writes one JSON log
  per figure to `figure_scripts/provenance/` and checks every hash
  (`provenance_index.py --verify`).
- Case identifiers, patch coordinates and file names were removed from the
  figures and captions. They are kept in the logs, together with the level-0
  field of each review crop, located exactly in its WSI cache.
- New scripts:
  - `make_figure_5.py`: Fig 5, the source patch identified, using the
    training-time Vahadane copy and the pipeline's transforms.
  - `make_figures_6_8_9_10_12.py`: Figs 6, 8, 9, 10 and 12, read from the
    exports. The earlier Fig 8 curves were typed in and did not match
    `runs/final_main_training/history.csv`. The undefined 0.40–0.45 band of
    Fig 12 is no longer drawn.
  - `make_figures_11_16_17.py`: Fig 11 (source case identified in the
    pathologist review, now marked in S1 Table) and Figs 16 and 17.
  - `GENERATE_FIG13_LITERATURE.py`: Fig 13.
  - `provenance_log.py`, `provenance_index.py`, `run_all_figures.py`.
- Figs 14 and 15: the legend now uses the error-map colours.
- Fig 13: the Damkliang et al. test Dice is 63.7% (0.637 in the paper's
  Results; 0.64 in its abstract).
- `PANDA_Pipeline/figures/`: replaced by the current figures; Figs 5 and 12 added.
- `FIGURE_PROVENANCE_AUDIT.csv` and `FIGURE_PROVENANCE_DETAILED.csv` are
  rebuilt from the logs. All 17 figures are verified.

### Added
- `analysis_scripts/make_cache_archives.py`: builds the archives of release
  `panda-wsi-cache-complement-20261027`.

## 27 October 2026 (second update)

### Added
- `analysis_scripts/rebuild_level_thresholds.py`: rebuilds Table 9 from the WSI
  caches. Output: `paper_results/tables/level_specific_threshold_sweep_validation.csv`
  and `level_specific_thresholds_table9.csv`. Every Table 9 value is reproduced;
  the fused sweep matches `threshold_sensitivity_validation.csv` and the test
  counts at 0.43 match `per_slide_wsi_metrics.csv` for all 1,577 WSIs.
- `analysis_scripts/coverage_from_caches.py`: generating script for
  `coverage_analysis/`; reproduces the archived files.
- `analysis_scripts/provider_tumor_definition.py` and
  `paper_results/tables/provider_tumor_definition_test_patches.csv`: source of
  the provider tumor-definition sentence in the third limitation.
- `figure_scripts/make_figure_3.py`: Fig 3 drawn from nested stored L0/L1/L2
  patches of the same WSI, with the field of each finer level marked.

### Removed
- `PANDA_Pipeline/GENERATE_MULTILEVEL_PATCH_MASK_OVERLAY.py`, replaced by
  `figure_scripts/make_figure_3.py`.

### Corrected
- B4 precision note: the original evaluation read the same float16 caches but
  cast them to float32 before thresholding; it did not use separate
  full-precision maps. Both per-slide files are reproducible from the caches.
- `HARDCODED_PATHS_SUMMARY.csv` now records how each executed default path is
  overridden.

## 27 October 2026

### Repository history
- The previous upload history (109,153 single-file commits) was replaced by a
  single commit, to remove personal files committed on 9 October 2026. The
  owner keeps a private backup of the old history. Release tags now point to
  the new commit; release assets are unchanged.

### Removed
- `PANDA_Pipeline/requirements.txt` (unpinned). Use the pinned
  `PANDA_Reproducibility/requirements.txt`.
- The 23 `PANDA_Pipeline/IMPORTANT_V*.txt` build notes; the version history
  below keeps what they recorded.
- `PANDA_FAST_MISSING_AUDIT.txt`, `paper_results/reviewer_proof_finalization.json`
  and `FIGURE_PROVENANCE_CANDIDATES.csv` (superseded upload and audit notes).
- `PANDA_Pipeline/GENERATE_PANDA_WHOLE_SLIDE_SEGMENTATION_WORKFLOW.py` and
  `GENERATE_PANDA_PATCH_PREPARATION_PIPELINE.py`. They drew the earlier Figs 1
  and 2 (twelve provider x level x class strata) and are replaced by
  `figure_scripts/make_figures_1_2_4_7.py`.

### Added
- `figure_scripts/`:
  - `make_figures_1_2_4_7.py` draws Figs 1, 2, 4 and 7 from stored training
    patches, reference masks, the training-derived stain target and the final
    checkpoint.
  - `make_figures_14_15.py` draws Figs 14 and 15 from the saved review outputs.
  - `make_S1_table.py` builds S1 Table.
  - `provenance/` lists every source file of these figures with its SHA-256.
- `supporting_information/S1_Table.{csv,xlsx}`: the 30 WSIs of the
  pathologist-guided review.
- `PANDA_Pipeline/figures/`: Figs 1, 2, 4, 7, 14 and 15 as used in the manuscript.

### Updated
- `README.md`: environment, paths, a manuscript figure/table → script map, and known gaps.
- `PANDA_Pipeline/README.md`: folder paths given as arguments, not machine paths.
- `FIGURE_PROVENANCE_AUDIT.csv` and `FIGURE_PROVENANCE_DETAILED.csv`: Figs 1, 2,
  4, 7, 14 and 15 are verified with source IDs; Fig 3 has its source case.
- `B1_B4_OUTPUT_INDEX.md`: producing script and the float16 note for B4.

### B4 against `per_slide_wsi_metrics.csv`
B4 rescored the float16 probabilities stored in `paper_results/wsi_cache`.
`prob >= 0.43` on a float16 array uses float16(0.43) = 0.4299316, so a few
pixels just below 0.43 count as positive. Across the 1,577 test WSIs:
- TP +5,135 and FP +6,821, with FN and TN falling by the same amounts;
- at most 70 pixels per WSI, and no positive is ever removed;
- pooled fusion Dice 0.852160 against 0.852159 in the original float32 evaluation.

## Earlier in October 2026

- Added six B1–B4 statistical and common-support CSV exports.
- Added `selected/train_selected.csv` (80,000 selected patches).
- Added the pinned `requirements.txt` from the experiment environment.
- Added the B1–B4 output index, the figure inventory and the audit of
  machine-specific paths.

## Pipeline version history (from the former IMPORTANT_V*.txt notes)

Versions V6–V15 changed robustness and speed, not the scientific configuration.

| Version | Change |
|---|---|
| V6–V7 | atomic, SSD-safe state writes and resumable pilot epochs; fixed `pandas.Series.mask` attribute access in the patch dataset |
| V9–V10 | faster DataLoader (4 workers, persistent, pinned memory); lossless on-disk Vahadane cache keyed by the stain-target identity |
| V11–V12 | final model trained before the pilots; all patch files kept (no pruning) |
| V13–V15 | device-tuned loader settings; checkpoint saves retried after storage interruptions; Vahadane cache prebuilt before training |

From V16 the evaluation and outputs changed as follows:

| Version | Change |
|---|---|
| V16–V18 | added per-slide denominators, HD95/ASSD/surface Dice, paired slide bootstraps with Holm correction; fixed order: final training, validation threshold, held-out test, dense WSI reconstruction and fusion, then pilots |
| V19–V21 | batched dense WSI inference, streamed validation sweep, shared low-resolution tissue mask per WSI; watchdog that restarts after storage loss (no change to models or metrics) |
| V22–V25 | test visuals saved live: Dice ≥ 0.80 cases to `test_visuals_good_live/`, and the first 30 test WSIs with Dice < 0.80 in test order to `test_visuals_bad_review/` (used for the pathologist review), one folder per case |
| V28–V31 | packaged pipeline; separate SICAPv2 external evaluation without retraining (official `Test.xlsx` not resolved, so an all-images fallback index was used); download fallbacks for SICAPv2 |

Settings fixed across these versions:
- batch size 8, AdamW (lr 2e-4, weight decay 1e-4);
- BCE + Dice loss, at most 30 epochs;
- Vahadane target from training data only, applied to all splits and to inference.
