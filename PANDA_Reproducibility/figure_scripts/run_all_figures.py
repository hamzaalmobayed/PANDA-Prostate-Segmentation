"""Rebuild the 12 manuscript figures, the S1 Appendix figures and S1 Table, then the provenance index.

    python run_all_figures.py <PANDA_DATA_ROOT> <SICAPv2 folder> [figures_dir] [provenance_dir] [s1_dir] [script ...]

Figure numbers are those of the revision of 30 Oct 2026. Defaults (relative to
the folder above figure_scripts/): figures/, figure_scripts/provenance/ and
supporting_information/ (or that folder itself when it has no
supporting_information/) for S1_Table.xlsx/.csv. Each script writes its PNG and
PLOS TIFF to figures_dir and its JSON log to provenance_dir; afterwards
provenance_index.py writes the audit tables and checks every hash.

PANDA_DATA_ROOT is the pipeline work folder (patches/, selected/, manifests/,
cache/, runs/, latest_run.json, stain_target_train.png, paper_results/,
external_precise_wsi/, external_sicapv2/, external_sicapv2_official/,
revision_2026-10-30/). The SICAPv2 folder is the official release (images/,
masks/), used by Fig 11. rebuild_inputs_from_kaggle.py re-creates the stored
training patches and stain-cache files from the Kaggle WSIs where they are
missing.
"""
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
# absolute paths: the scripts run with this folder as working directory
WORK = Path(sys.argv[1]).resolve()
SICAP = Path(sys.argv[2]).resolve()
FIGS = (Path(sys.argv[3]) if len(sys.argv) > 3 else PKG / "figures").resolve()
PROV = (Path(sys.argv[4]) if len(sys.argv) > 4 else HERE / "provenance").resolve()
# S1 Table: supporting_information/ in the repository, the package folder otherwise
S1 = (Path(sys.argv[5]) if len(sys.argv) > 5 else
      PKG / "supporting_information" if (PKG / "supporting_information").is_dir() else PKG).resolve()
SCRIPTS = ["make_figures_1_2_4.py",   # Figs 1, 2, 4
           "make_figure_3.py",        # Fig 3
           "make_figures_5_8.py",     # Figs 5, 8
           "make_figure_6.py",        # Fig 6
           "make_figures_7_12.py",    # Figs 7, 12
           "make_figures_9_10.py",    # Figs 9, 10
           "make_figure_11.py",       # Fig 11
           "make_S1_figures.py"]      # S1 Appendix: augmentations, confusion matrices
env = dict(os.environ, PANDA_DATA_ROOT=str(WORK), PANDA_PROVENANCE_DIR=str(PROV), SICAPV2_ROOT=str(SICAP),
           PYTHONIOENCODING="utf-8")
only = set(sys.argv[6:])  # optional script names to run
for s in SCRIPTS:
    if only and s not in only:
        continue
    print(f"== {s}", flush=True)
    subprocess.run([sys.executable, str(HERE / s), str(FIGS), str(WORK)], check=True, cwd=HERE, env=env)
if not only or "make_S1_table.py" in only:
    subprocess.run([sys.executable, str(HERE / "make_S1_table.py"), str(S1), str(WORK)], check=True, cwd=HERE, env=env)
subprocess.run([sys.executable, str(HERE / "provenance_index.py"), str(PROV), str(FIGS), str(WORK), "--verify"],
               check=True, cwd=HERE, env=env)
