"""How much of the SICAPv2 reference tumor lies outside the scored tissue region.

The SICAPv2 evaluation scores only the tissue pixels that the PANDA tissue rule
detects on each H&E tile (norm_pool.prepare_sicap: generate_tissue_mask, with
its fallbacks). This script needs no model: for every tile of the official
release it counts the reference tumor pixels (mask > 0), those outside the
scored region, and the bright pixels (mean RGB > 215, unstained lumina and
background) among them. Results are given for the official test partition
and for all tiles.

Usage: python sicap_scored_region.py <SICAPv2 folder> <PANDA_Reproducibility>
Output: external_sicapv2_official/summary/SICAP_scored_region.csv
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "PANDA_Pipeline"))
from norm_pool import generate_tissue_mask  # noqa: E402

SIC, REP = Path(sys.argv[1]), Path(sys.argv[2])
BRIGHT = 215
idx = pd.read_csv(REP / "external_sicapv2_official/results/tables/official_index.csv")
rows = []
for r in idx.itertuples():
    rgb = np.asarray(Image.open(SIC / "images" / r.image_name).convert("RGB"), dtype=np.uint8)
    h, w = rgb.shape[:2]
    m = cv2.imread(str(SIC / "masks" / r.image_name.replace(".jpg", ".png")), cv2.IMREAD_GRAYSCALE)
    if m.shape[:2] != (h, w):
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
    ref = m > 0
    valid = generate_tissue_mask(rgb).astype(bool)          # same fallbacks as norm_pool.prepare_sicap
    if valid.mean() < 0.01:
        valid = rgb.mean(axis=2) < 245
    if valid.mean() < 0.01:
        valid = np.ones((h, w), bool)
    bright = rgb.mean(axis=2) > BRIGHT
    rows.append((r.test_partition, int(ref.sum()), int((ref & ~valid).sum()), int((ref & ~valid & bright).sum()),
                 int((ref & bright).sum()), int((ref & bright & valid).sum())))
d = pd.DataFrame(rows, columns=["test_partition", "ref", "ref_unscored", "ref_unscored_bright", "ref_bright", "ref_bright_scored"])
out = []
for name, sub in (("test partition", d[d.test_partition]), ("all tiles", d)):
    s = sub.drop(columns="test_partition").sum()
    out.append({"subset": name, "tiles": len(sub), "tiles_with_reference": int((sub.ref > 0).sum()),
                "reference_px": int(s.ref), "reference_unscored_px": int(s.ref_unscored),
                "reference_unscored_share": s.ref_unscored / s.ref,
                "unscored_reference_bright_share": s.ref_unscored_bright / max(s.ref_unscored, 1),
                "reference_bright_px": int(s.ref_bright), "reference_bright_scored_share": s.ref_bright_scored / max(s.ref_bright, 1),
                "bright_threshold_mean_rgb": BRIGHT})
res = pd.DataFrame(out)
dst = REP / "external_sicapv2_official/summary/SICAP_scored_region.csv"
res.to_csv(dst, index=False)
print(res.to_string(index=False))
