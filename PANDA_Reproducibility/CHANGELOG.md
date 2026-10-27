# PANDA Reproducibility Changelog

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
