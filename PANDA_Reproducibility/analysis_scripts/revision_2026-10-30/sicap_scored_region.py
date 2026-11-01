"""How much of the SICAPv2 reference tumor lies outside the scored tissue region.

The SICAPv2 evaluation scores only the tissue pixels that the PANDA tissue rule
detects on each H&E tile (norm_pool.prepare_sicap: generate_tissue_mask, with
its fallbacks). This script needs no model: for every tile of the official
release it counts the reference tumor pixels (mask > 0), those outside the
scored region, and the bright pixels (mean RGB > 215, unstained lumina and
background) among them. Results are given for the official test partition
and for all tiles.

The unscored reference pixels that are not bright are split by the step of the
rule that leaves them out (1 Nov 2026). The rule thresholds the HSV saturation
at max(Otsu threshold of the tile, fixed floor 0.20 = 51), then removes pixels
by morphological opening, closing and removal of components below 500 px:
* below_floor: saturation <= 51;
* floor_to_otsu: saturation above 51 but at or below the tile's Otsu threshold;
* cleanup: saturation above the threshold, removed by the morphological cleanup;
* fallback: tiles on which a fallback of the rule (near-white pixels) applied.
Saturation medians of the scored reference and of the unscored, not bright
reference are taken from 256-bin histograms.

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
from norm_pool import TISSUE_DEFAULT_THRESHOLD, generate_tissue_mask, sicap_valid  # noqa: E402

SIC, REP = Path(sys.argv[1]), Path(sys.argv[2])
BRIGHT = 215
FLOOR = int(round(TISSUE_DEFAULT_THRESHOLD * 255))
GROUPS = ("below_floor", "floor_to_otsu", "cleanup", "fallback")
idx = pd.read_csv(REP / "external_sicapv2_official/results/tables/official_index.csv")
rows, hist = [], {k: np.zeros((2, 256), np.int64) for k in (True, False)}
for r in idx.itertuples():
    rgb = np.asarray(Image.open(SIC / "images" / r.image_name).convert("RGB"), dtype=np.uint8)
    h, w = rgb.shape[:2]
    m = cv2.imread(str(SIC / "masks" / r.image_name.replace(".jpg", ".png")), cv2.IMREAD_GRAYSCALE)
    if m.shape[:2] != (h, w):
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
    ref = m > 0
    valid = sicap_valid(rgb)                                 # same fallbacks as norm_pool.prepare_sicap
    bright = rgb.mean(axis=2) > BRIGHT
    sat = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[..., 1]
    otsu, _ = cv2.threshold(sat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr = max(int(otsu), FLOOR)
    tm = generate_tissue_mask(rgb) > 0
    fallback = not np.array_equal(valid, tm) or np.array_equal(tm, (rgb.mean(axis=2) < 240) & (rgb.min(axis=2) < 245))
    nb = ref & ~valid & ~bright
    g = dict.fromkeys(GROUPS, 0)
    if fallback:
        g["fallback"] = int(nb.sum())
    else:
        g.update(below_floor=int((nb & (sat <= FLOOR)).sum()), floor_to_otsu=int((nb & (sat > FLOOR) & (sat <= thr)).sum()),
                 cleanup=int((nb & (sat > thr)).sum()))
    hist[r.test_partition][0] += np.bincount(sat[ref & valid], minlength=256)
    hist[r.test_partition][1] += np.bincount(sat[nb], minlength=256)
    rows.append((r.test_partition, int(ref.sum()), int((ref & ~valid).sum()), int((ref & ~valid & bright).sum()),
                 int((ref & bright).sum()), int((ref & bright & valid).sum()), int(fallback), *g.values()))
d = pd.DataFrame(rows, columns=["test_partition", "ref", "ref_unscored", "ref_unscored_bright", "ref_bright", "ref_bright_scored",
                                "fallback_tile", *GROUPS])


def median(hst):
    return int(np.searchsorted(np.cumsum(hst), hst.sum() / 2))


out = []
for name, sub, hs in (("test partition", d[d.test_partition], hist[True]), ("all tiles", d, hist[True] + hist[False])):
    s = sub.drop(columns="test_partition").sum()
    nb = s.ref_unscored - s.ref_unscored_bright
    out.append({"subset": name, "tiles": len(sub), "tiles_with_reference": int((sub.ref > 0).sum()),
                "reference_px": int(s.ref), "reference_unscored_px": int(s.ref_unscored),
                "reference_unscored_share": s.ref_unscored / s.ref,
                "unscored_reference_bright_share": s.ref_unscored_bright / max(s.ref_unscored, 1),
                "reference_bright_px": int(s.ref_bright), "reference_bright_scored_share": s.ref_bright_scored / max(s.ref_bright, 1),
                "bright_threshold_mean_rgb": BRIGHT,
                "reference_unscored_not_bright_px": int(nb), "reference_unscored_not_bright_share": nb / s.ref,
                **{f"unscored_not_bright_{k}_px": int(s[k]) for k in GROUPS},
                **{f"unscored_not_bright_{k}_share": s[k] / max(nb, 1) for k in GROUPS},
                "fallback_tiles": int(s.fallback_tile), "saturation_floor": FLOOR,
                "saturation_median_scored_reference": median(hs[0]),
                "saturation_median_unscored_not_bright_reference": median(hs[1])})
res = pd.DataFrame(out)
dst = REP / "external_sicapv2_official/summary/SICAP_scored_region.csv"
res.to_csv(dst, index=False)
print(res.T.to_string())
