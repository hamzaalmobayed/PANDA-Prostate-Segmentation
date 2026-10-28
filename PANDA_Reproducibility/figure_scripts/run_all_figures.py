"""Rebuild all 17 manuscript figures and S1 Table, then the provenance index.

    python run_all_figures.py <PANDA_DATA_ROOT> [figures_dir] [provenance_dir] [s1_dir]

Defaults (relative to the folder above figure_scripts/): figures/,
figure_scripts/provenance/ and supporting_information/ (or that folder itself
when it has no supporting_information/) for S1_Table.xlsx/.csv. Each script writes its PNG and
PLOS TIFF to figures_dir and its JSON log to provenance_dir; afterwards
provenance_index.py writes the audit tables and checks every hash.
"""
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
# absolute paths: the scripts run with this folder as working directory
WORK = Path(sys.argv[1]).resolve()
FIGS = (Path(sys.argv[2]) if len(sys.argv) > 2 else PKG / "figures").resolve()
PROV = (Path(sys.argv[3]) if len(sys.argv) > 3 else HERE / "provenance").resolve()
# S1 Table: supporting_information/ in the repository, the package folder otherwise
S1 = (Path(sys.argv[4]) if len(sys.argv) > 4 else
      PKG / "supporting_information" if (PKG / "supporting_information").is_dir() else PKG).resolve()
SCRIPTS = ["make_figures_1_2_4_7.py",    # Figs 1, 2, 4, 7
           "make_figure_3.py",           # Fig 3
           "make_figure_5.py",           # Fig 5
           "make_figures_6_8_9_10_12.py",  # Figs 6, 8, 9, 10, 12
           "make_figures_11_16_17.py",   # Figs 11, 16, 17
           "GENERATE_FIG13_LITERATURE.py",  # Fig 13
           "make_figures_14_15.py"]      # Figs 14, 15
env = dict(os.environ, PANDA_DATA_ROOT=str(WORK), PANDA_PROVENANCE_DIR=str(PROV), PYTHONIOENCODING="utf-8")
only = set(sys.argv[5:])  # optional script names to run
for s in SCRIPTS:
    if only and s not in only:
        continue
    print(f"== {s}", flush=True)
    subprocess.run([sys.executable, str(HERE / s), str(FIGS), str(WORK)], check=True, cwd=HERE, env=env)
if not only or "make_S1_table.py" in only:
    subprocess.run([sys.executable, str(HERE / "make_S1_table.py"), str(S1), str(WORK)], check=True, cwd=HERE, env=env)
subprocess.run([sys.executable, str(HERE / "provenance_index.py"), str(PROV), str(FIGS), str(WORK), "--verify"],
               check=True, cwd=HERE, env=env)
