# Figure scripts

All 17 manuscript figures and S1 Table are drawn by the scripts in this folder from
the stored study data. Set `PANDA_DATA_ROOT` to the pipeline work folder (or pass it
as the second argument).

```bash
python run_all_figures.py <PANDA_DATA_ROOT>      # all figures, S1 Table, provenance index and hash check
python provenance_index.py provenance ../figures <PANDA_DATA_ROOT> --verify   # check only
```

| Script | Figures | Reads |
|---|---|---|
| `make_figures_1_2_4_7.py` | 1, 2, 4, 7 | stored training patches and masks of one training WSI, coordinate pools, `selected/train_selected.csv`, the training-time Vahadane cache, `stain_target_train.png`, `runs/final_main_training/best.pt` |
| `make_figure_3.py` | 3 | nested stored L0/L1/L2 tumor patches and masks of one training WSI |
| `make_figure_5.py` | 5 | one selected L0 training patch (training-time Vahadane copy) and the transforms of `PairAugStainHeavy` |
| `make_figures_6_8_9_10_12.py` | 6, 8, 9, 10, 12 | `selected/train_selected.csv`, `runs/final_main_training/history.csv`, `paper_results/tables/{per_slide_wsi_metrics,patch_metrics_per_level_and_pooled,threshold_sensitivity_validation}.csv`, `best_threshold.json` |
| `make_figures_11_16_17.py` | 11, 16, 17 | review detail images of one test WSI, SICAPv2 evaluation outputs, PRECISE WSI caches and visuals |
| `GENERATE_FIG13_LITERATURE.py` | 13 | values of the cited studies; `paper_results/tables/global_wsi_metrics_95CI.csv` |
| `make_figures_14_15.py` | 14, 15 | review detail images of six test WSIs; one SICAPv2 image |
| `make_S1_table.py` | S1 Table | `bad_case_manifest_live.csv`, `per_slide_wsi_metrics.csv` |

Each figure is written as PNG and PLOS TIFF (RGB, LZW, 300 dpi, at most 2250 px wide).

## Logs

The figures and captions carry no case identifiers, patch coordinates or file names.
These are kept in `provenance/` only:

- `FigNN_<name>_sources.json`, one per figure (written by `provenance_log.py`):
  the output files and their SHA-256; the script and helper with their SHA-256; the
  command; the generation time; the software versions; and, per panel:
  - the case (slide ID, provider, split);
  - the level and level-0 coordinates of each patch;
  - the image, tumor-mask and valid-mask files;
  - for review crops, the level-0 field located exactly in the WSI cache.

  Each log also holds the parameters used and the values printed in the figure.
- `FIGURE_PROVENANCE_AUDIT.csv`: one row per figure.
- `FIGURE_PROVENANCE_DETAILED.csv`: one row per source file, with its SHA-256.
- `FIGURE_PROVENANCE.md`: a readable index.
- Intermediate arrays needed for exact reuse: the normalized Fig 7 input, the Fig 7
  probability map and the Fig 13 PDF.

`provenance_index.py --verify` recomputes every hash: scripts, figure files and source
files. It exits with status 1 on any difference.

## Notes

- The normalized patch in Figs 2E, 4B and 5 is the copy the training pipeline wrote to
  `cache/stain_v10/vahadane/`, so it is the exact training input.
- The Fig 7 input is not in that cache. It is normalized with the training target and a
  fixed NumPy seed, because TIAToolbox dictionary learning uses NumPy's global random
  state.
- The earlier Figs 5 and 11 had no recorded source. Each was matched to stored data by
  image correlation, and the matches are recorded in the logs:
  - Fig 5: 0.977, against 0.20 for the next of 109,060 L0 patches.
  - Fig 11: 0.999, against 0.69 for the next review case.
- The earlier Figs 8, 10 and 12 were drawn from values typed into scripts. They are now
  read from the exports:
  - The Fig 8 curves did not match `history.csv` at every epoch.
  - The Fig 10 counts were identical to the export.
  - The Fig 12 values were identical to the export. The 0.40–0.45 "optimal region" band
    had no stated rule and is no longer drawn.
