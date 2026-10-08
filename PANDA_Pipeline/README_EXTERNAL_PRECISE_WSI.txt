Add run_external_precise_wsi_test.py to the same final_article_pkg folder as run_full_panda_paper.py.

Run:
    python run_external_precise_wsi_test.py

Default:
    downloads only PRECISE H&E WSIs + H&E masks, not IHC.
    all 27 H&E WSIs ~= 28.2 GB extracted payload based on the Zenodo archive listing.

Optional compact mode:
    set PRECISE_MAX_SLIDES=10
    python run_external_precise_wsi_test.py

The compact mode deterministically picks the smallest H&E+mask pairs by file size before any model results are known.
