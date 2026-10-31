"""S2 Table: the 100 SICAPv2 images of the pathologist-guided review.

The review used the per-image result folders of the first SICAPv2 run (renamed
Kaggle copy, 0.jpg, 1.jpg, ...). The images are the first 100 with Dice < 0.80
in the processing order of that run (the row order of
external_sicapv2/results/tables/per_external_image_metrics.csv) among the
images that have a saved result folder. The folder names are listed in
external_sicapv2/results/review/saved_result_folders.txt.

Each result folder holds combined_original_gt_model_error.png, whose first
panel is the original 512 x 512 H&E image, pixel for pixel (rows 170-681,
columns 0-511). Every image is identified in the official release v2 by an
exact match of the decoded pixels; the official name gives slide, patient and
partition.

Usage: python make_S2_table.py <PANDA_Reproducibility> <review result folders> <SICAPv2 folder> <out_dir>
"""
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

REP, RES, SIC, OUT = (Path(a) for a in sys.argv[1:5])
N, DICE_MAX = 100, 0.80

m = pd.read_csv(REP / "external_sicapv2/results/tables/per_external_image_metrics.csv")
saved = set((REP / "external_sicapv2/results/review/saved_result_folders.txt").read_text().split())
m["stem"] = m.image_name.str[:-4]
sel = m[m.stem.isin(saved) & (m.dice < DICE_MAX)].head(N).copy()
assert len(sel) == N
skipped = m[~m.stem.isin(saved) & (m.dice < DICE_MAX) & (m.index < sel.index.max())]

digest = lambda a: hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
want = {}
for s in sel.stem:
    comp = np.asarray(Image.open(RES / s / s / "combined_original_gt_model_error.png").convert("RGB"))
    want[digest(comp[170:682, 0:512])] = s
found = {}
for p in sorted((SIC / "images").glob("*.jpg")):
    h = digest(np.asarray(Image.open(p).convert("RGB")))
    if h in want:
        found[want[h]] = p.name
assert len(found) == N, f"{N - len(found)} images not matched"

idx = pd.read_csv(REP / "external_sicapv2_official/results/tables/official_index.csv").set_index("image_name")
sel["official"] = sel.stem.map(found)
s2 = pd.DataFrame({
    "Review order": range(1, N + 1),
    "First-run image (renamed copy)": sel.image_name.values,
    "Official release image": sel.official.values,
    "Slide": [idx.loc[o, "slide"] for o in sel.official],
    "Patient": [idx.loc[o, "patient_id"] for o in sel.official],
    "Official test partition": ["Yes" if idx.loc[o, "test_partition"] else "No" for o in sel.official],
    "First-run Dice": sel.dice.round(4).values,
    "Reference tumor in valid tissue": np.where(sel.gt_tumor_pixels.values > 0, "Yes", "No"),
})
OUT.mkdir(parents=True, exist_ok=True)
s2.to_excel(OUT / "S2_Table.xlsx", index=False, sheet_name="S2 Table")
s2.to_csv(OUT / "S2_Table.csv", index=False)
summary = {"n": N, "reference_negative": int((s2["Reference tumor in valid tissue"] == "No").sum()),
           "slides": int(s2.Slide.nunique()), "patients": int(s2.Patient.nunique()),
           "in_test_partition": int((s2["Official test partition"] == "Yes").sum()),
           "skipped_without_result_folder": skipped.image_name.tolist()}
pd.Series(summary).to_json(OUT / "S2_Table_summary.json", indent=1)
print(s2.head().to_string(index=False)); print(summary)
