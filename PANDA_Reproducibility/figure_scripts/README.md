# Figure scripts

The scripts in this folder draw, from the stored study data:

- the 12 manuscript figures (numbering of the revision of 30 October 2026);
- the two drawn figures of S1 Appendix;
- S1 Table.

Every figure is drawn with SciencePlots and the PLOS font settings: each script calls
`provenance_log.plos_style()` before drawing (Arial, 8 pt for all text) and sets no style of its
own. The figures carry no titles; panel letters and labels only.

Set `PANDA_DATA_ROOT` to the pipeline work folder, or pass it as the first argument.

```bash
python run_all_figures.py <PANDA_DATA_ROOT> <SICAPv2 folder>   # all figures, S1 Table, provenance index and hash check
SICAPV2_ROOT=<SICAPv2 folder> python provenance_index.py provenance ../figures <PANDA_DATA_ROOT> --verify   # check only
```

## Manuscript figures

| Figure | Script | Output | Reads |
|---|---|---|---|
| Fig 1 | `make_figures_1_2_4.py` | `Fig01_pipeline.png`, `Fig1.tif` | stored training patches and masks of one training WSI |
| Fig 2 | `make_figures_1_2_4.py` | `Fig02_preprocessing.png`, `Fig2.tif` | the same WSI: stored L2 patches, coordinate pool, the training-time Vahadane cache, `selected/train_selected.csv` |
| Fig 3 | `make_figure_3.py` | `Fig03_multiresolution.png`, `Fig3.tif` | nested stored L0/L1/L2 tumor patches and masks of one training WSI |
| Fig 4 | `make_figures_1_2_4.py` | `Fig04_architecture.png`, `Fig4.tif` | one stored L1 patch, `stain_target_train.png`, `runs/final_main_training/best.pt` |
| Fig 5 | `make_figures_5_8.py` | `Fig05_training_dynamics.png`, `Fig5.tif` | `runs/final_main_training/history.csv` |
| Fig 6 | `make_figure_6.py` | `Fig06_per_slide_dice.png`, `Fig6.tif` | `revision_2026-10-30/tables/cache_per_slide_stats_test.csv` |
| Fig 7 | `make_figures_7_12.py` | `Fig07_qualitative_good.png`, `Fig7.tif` | review detail images of one test WSI, its WSI cache, `per_slide_wsi_metrics.csv` |
| Fig 8 | `make_figures_5_8.py` | `Fig08_threshold_sweep.png`, `Fig8.tif` | `paper_results/tables/threshold_sensitivity_validation.csv`, `best_threshold.json` |
| Fig 9 | `make_figures_9_10.py` | `Fig09_model_related_errors.png`, `Fig9.tif` | review detail images and WSI caches of three test WSIs |
| Fig 10 | `make_figures_9_10.py` | `Fig10_annotation_discrepancies.png`, `Fig10.tif` | review detail images and WSI caches of three test WSIs; one SICAPv2 tile (`external_sicapv2_official/results/figure_tile/`) |
| Fig 11 | `make_figure_11.py` | `Fig11_sicap_external.png`, `Fig11.tif` | `external_sicapv2_official/results/tables/per_tile_metrics.csv`, the SICAPv2 release, the checkpoint |
| Fig 12 | `make_figures_7_12.py` | `Fig12_precise_external.png`, `Fig12.tif` | PRECISE WSI caches and saved visuals of two WSIs |

Fig 2 no longer has the count panels F–G of the earlier version: the candidate-grid counts of the
WSI and the training-bucket counts are a table of the manuscript (`TABLE_PREP` in
`analysis_scripts/revision_2026-10-30/fill_manuscript_numbers.py`).

## S1 Appendix and S1 Table

| Item | Script | Output | Reads |
|---|---|---|---|
| S1 Appendix, augmentation figure | `make_S1_figures.py` | `S1_augmentation.png` | one selected L0 training patch (training-time Vahadane copy) and the transforms of `PairAugStainHeavy` |
| S1 Appendix, confusion matrices | `make_S1_figures.py` | `S1_confusion.png` | `paper_results/tables/patch_metrics_per_level_and_pooled.csv` |
| S1 Appendix, stain target | (image file) | `stain_target_train.png` | the training-derived Vahadane target itself |
| S1 Table | `make_S1_table.py` | `S1_Table.xlsx`, `S1_Table.csv` | `bad_case_manifest_live.csv`, `per_slide_wsi_metrics.csv` |

`analysis_scripts/revision_2026-10-30/make_s1_appendix.py` writes `S1_Appendix.tex` and copies the
three images next to it.

Each figure is written as PNG and PLOS TIFF (RGB, LZW, 300 dpi, at most 2250 px wide; PNG →
RGB → LZW TIFF in `provenance_log.plos_export`).

## Earlier numbering

| Revision of 30 Oct 2026 | Before |
|---|---|
| Figs 1, 2, 3 | Figs 1, 2, 3 |
| Fig 4 | Fig 7 |
| Fig 5 | Fig 8 |
| Fig 6 | new (replaces Fig 9) |
| Fig 7 | Fig 11 |
| Fig 8 | Fig 12 |
| Figs 9, 10 | Figs 14, 15 |
| Fig 11 | new (replaces Fig 16) |
| Fig 12 | Fig 17 |
| S1 Appendix | Figs 5 and 10, and the target panel of Fig 4 |
| removed | Figs 4 (merged into Fig 2E and S1 Appendix), 6, 13 |

## Inputs not on every machine

The stored training patches (`patches/train/...`) and the training-time Vahadane cache files
(`cache/stain_v10/vahadane/...`) used by Figs 1–4 and the S1 augmentation figure are on the
pipeline machine. `rebuild_inputs_from_kaggle.py` writes them again from the Kaggle WSIs and label
masks with the pipeline's own readers and writers (JPEG quality 92 without chroma subsampling,
PNG masks, PNG stain cache) and stops unless every file named in a figure log has the logged
SHA-256. The figures of 30 October 2026 were drawn from inputs re-created this way; all logged
files matched byte for byte.

## Logs

The figures and captions carry no case identifiers, patch coordinates or file names.
These are kept in `provenance/` only:

- `FigNN_<name>_sources.json` and `S1_<name>_sources.json`, one per figure (written by
  `provenance_log.py`): the output files and their SHA-256; the script and helper with their
  SHA-256; the command; the generation time; the software versions; and, per panel:
  - the case (slide ID, provider, split);
  - the level and level-0 coordinates of each patch;
  - the image, tumor-mask and valid-mask files;
  - for review crops, the level-0 field located exactly in the WSI cache.

  Each log also holds the parameters used and the values printed in the figure.
- `FIGURE_PROVENANCE_AUDIT.csv`: one row per figure.
- `FIGURE_PROVENANCE_DETAILED.csv`: one row per source file, with its SHA-256.
- `FIGURE_PROVENANCE.md`: a readable index.
- Intermediate arrays needed for exact reuse: the normalized Fig 4 input and the Fig 4
  probability map.

`provenance_index.py --verify` recomputes every hash: scripts, figure files and source files. It
exits with status 1 on any difference. The SICAPv2 release images of Fig 11 are checked under
`SICAPV2_ROOT` when that variable is set.

## Notes

- The normalized patch in Fig 2E and in the S1 augmentation figure is the copy the training
  pipeline wrote to `cache/stain_v10/vahadane/`, so it is the exact training input.
- The Fig 4 input is not in that cache. It is normalized with the training target and a fixed
  NumPy seed, because TIAToolbox dictionary learning uses NumPy's global random state.
- Fig 10D: the image of the first SICAPv2 run (`0.jpg` of the renamed Kaggle copy) is
  byte-identical to the official release tile `16B0001851_Block_Region_1_0_0_xini_6803_yini_59786.jpg`.
  The Kaggle copy and the first run's prediction files are not kept, so the panel shows the release
  tile with its lossless mask and a prediction recomputed under the first-run settings
  (`analysis_scripts/revision_2026-10-30/sicap_figure_tile.py`). Its TP/FP/TN/FN counts agree
  with the official run's stored counts to within a few pixels (GPU rounding); both are in the
  tile's JSON.
- The earlier Figs 5 and 11 (now S1 augmentation figure and Fig 7) had no recorded source. Each
  was matched to stored data by image correlation, and the matches are recorded in the logs:
  - augmentation figure: 0.977, against 0.20 for the next of 109,060 L0 patches;
  - Fig 7: 0.999, against 0.69 for the next review case.
- The earlier training-dynamics, confusion-matrix and threshold figures were drawn from values
  typed into scripts. They are now read from the exports:
  - the training curves did not match `history.csv` at every epoch;
  - the confusion counts were identical to the export;
  - the threshold values were identical to the export. The 0.40–0.45 "optimal region" band had
    no stated rule and is no longer drawn.
