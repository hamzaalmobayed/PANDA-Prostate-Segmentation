"""Build S1 Table: the 30 PANDA test WSIs of the pathologist-guided review.

Sources: bad_case_manifest_live.csv (screening order, written during the test
run) and per_slide_wsi_metrics.csv (fused evaluation). Figure panels were
identified by matching each Fig 14/15 panel to the saved review images.
"""
import os
import sys
from pathlib import Path
import pandas as pd

if len(sys.argv) < 3 and "PANDA_DATA_ROOT" not in os.environ:
    sys.exit("usage: make_S1_table.py <output_dir> [work_root]  (or set PANDA_DATA_ROOT)")
WORK = Path(sys.argv[2] if len(sys.argv) > 2 else os.environ["PANDA_DATA_ROOT"])
OUT = Path(sys.argv[1])
FIGURE_PANEL = {
    "15bf6b34c8ce10b0eddf687e3a670e4d": "Fig 14A",
    "13813facc985c2ea007fdf764ef98a5a": "Fig 14B",
    "b49d54b4f7e73808440c36fa240614c0": "Fig 14C",
    "9848ee7cc45f7cd373b155191085f06a": "Fig 15A",
    "cf1c7e7e632db44c5ca6de473da247e6": "Fig 15B",
    "cfbe34e8826d4e2bb02061d9ef056e83": "Fig 15C",
}
man = pd.read_csv(WORK / "paper_results" / "test_visuals_bad_review" / "bad_case_manifest_live.csv")
per = pd.read_csv(WORK / "paper_results" / "tables" / "per_slide_wsi_metrics.csv",
                  usecols=["image_id", "fusion_dice", "fusion_gt_tumor_pixels", "fusion_fp", "fusion_valid_pixels"])
t = man.merge(per, on="image_id", how="left", validate="one_to_one")
assert len(t) == 30 and t.fusion_dice_y.notna().all()
assert (abs(t.fusion_dice_x - t.fusion_dice_y) < 1e-6).all(), "manifest Dice differs from per-slide file"
s1 = pd.DataFrame({
    "Screening order": t.order_saved,
    "PANDA image_id": t.image_id,
    "Provider": t.provider.str.capitalize(),
    "Stored fused Dice": t.fusion_dice_y.round(4),
    "Reference tumor on fused support": (t.fusion_gt_tumor_pixels > 0).map({True: "Yes", False: "No"}),
    "Reference tumor pixels (16x grid)": t.fusion_gt_tumor_pixels.astype(int),
    "False-positive pixels (16x grid)": t.fusion_fp.astype(int),
    "Valid evaluation pixels (16x grid)": t.fusion_valid_pixels.astype(int),
    "Shown in": t.image_id.map(FIGURE_PANEL).fillna(""),
})
OUT.mkdir(parents=True, exist_ok=True)
s1.to_excel(OUT / "S1_Table.xlsx", index=False, sheet_name="S1 Table")
s1.to_csv(OUT / "S1_Table.csv", index=False)
print(s1.to_string(index=False))
print("reference-negative:", (s1["Reference tumor on fused support"] == "No").sum())
