# Licenses of the files in this repository

- **Code** (all `*.py`, `*.bat`, `*.ps1` files and the configuration files): MIT license (`LICENSE`).
- **Files derived from PANDA images or masks**: Creative Commons Attribution-NonCommercial-ShareAlike 4.0
  International (CC BY-NC-SA 4.0), the license under which the PANDA data are distributed
  (the PANDA challenge pages and Bulten et al., Nat Med 2022, write it "CC BY-SA-NC 4.0"). The share-alike
  term requires these files to be shared under the same license. They are:
  - `PANDA_Reproducibility/stain_target_train.png` (the training-derived Vahadane target, a PANDA patch);
  - `PANDA_Reproducibility/figures/Fig01_pipeline.*`, `Fig02_preprocessing.*`, `Fig03_multiresolution.*`,
    `Fig04_architecture.*`, `Fig07_qualitative_good.*`, `Fig09_model_related_errors.*`,
    `Fig10_annotation_discrepancies.*`, `S1_augmentation.*` (figures with PANDA image panels), and their copies in
    `PANDA_Reproducibility/manuscript_revision_2026-10-30/`;
  - `PANDA_Reproducibility/figure_scripts/provenance/Fig04_input_normalized.png` and `Fig04_probability.npy`;
  - `PANDA_Reproducibility/paper_results/test_visuals_bad_review/` and `test_visuals_good_live/` (review images of
    PANDA test WSIs);
  - `PANDA_Reproducibility/paper_results/wsi_cache/` and the release archive `panda_wsi_cache_complement.zip`
    (probability maps and reference masks of PANDA WSIs on the evaluation grid);
  - the stored training patches that `figure_scripts/rebuild_inputs_from_kaggle.py` writes (not in the repository).
- **Files derived from SICAPv2 and PRECISE**: CC BY 4.0, the license of those datasets
  (`external_sicapv2*/`, `external_precise_wsi/`).
- **Tables of numbers** (CSV and JSON result files): MIT, with the license above for any file that contains
  PANDA image data.

The PANDA, SICAPv2 and PRECISE images themselves are not redistributed here; they are available from their
public releases.
