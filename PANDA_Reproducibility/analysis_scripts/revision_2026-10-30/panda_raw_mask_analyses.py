"""Analyses of the 1,577 PANDA test WSIs that need the original images and masks.

For each test WSI this script reads, from the Kaggle PANDA archive, the image
(only its TIFF resolution tags and its 16x pyramid level) and the label mask
(its 16x level), and the saved WSI cache. It writes one row per WSI with:

* pixel size: the level-0 micrometres per pixel given by the TIFF
  XResolution/ResolutionUnit tags of the image and of the mask (D2);
* a check that the provider mapping of the raw labels reproduces the cache's
  binary reference on annotated-valid pixels;
* raw-label composition of the fused evaluation region and how much of each
  label the fused map (and L1) predicts as tumor at 0.43 (A1: does the output
  include stroma?). For Radboud, stroma (label 1) is split into stroma inside
  the tumor envelope (morphological closing of labels 3-5 with a disk of radius
  5 evaluation pixels, about 38 um) and stroma outside it;
* pen ink (D13): ink pixels on the 16x image (dark-green, blue or black hues
  that H&E does not produce) and the fused counts with every evaluation pixel
  within 32 grid pixels (about 0.23 mm) of ink removed.

The PANDA 16x pyramid level is the evaluation grid of the caches, so no
resampling is needed beyond cropping to the cache shape.

Usage:
    python panda_raw_mask_analyses.py <repo_root> <panda_wsi_cache_complement.zip> <prostate-cancer-grade-assessment.zip> <output_dir>
"""
import io
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import tifffile

sys.path.insert(0, str(Path(__file__).parent))
import cache_stats as cs  # noqa: E402

THR = np.float32(0.43)
ENVELOPE_RADIUS = 5
INK_MARGIN = 32


def um_per_px(page):
    t = {k.name: k.value for k in page.tags.values()}
    if "XResolution" not in t or "ResolutionUnit" not in t:
        return np.nan
    num, den = t["XResolution"]
    px_per_unit = num / den if den else np.nan
    unit = int(t["ResolutionUnit"])
    per_cm = {2: px_per_unit / 2.54, 3: px_per_unit}.get(unit, np.nan)
    return 1e4 / per_cm if per_cm else np.nan


def level16(tif_bytes, channel0=False):
    with tifffile.TiffFile(io.BytesIO(tif_bytes)) as tf:
        mpp = um_per_px(tf.pages[0])
        lv = tf.series[0].levels
        a = lv[2].asarray() if len(lv) > 2 else cv2.resize(lv[0].asarray(), None, fx=1 / 16, fy=1 / 16,
                                                             interpolation=cv2.INTER_NEAREST)
    if channel0 and a.ndim == 3:
        a = a[..., 0]
    return mpp, a


def fit(a, shape):
    h, w = shape
    out = np.zeros((h, w) + a.shape[2:], a.dtype) if a.ndim == 2 else np.full((h, w) + a.shape[2:], 255, a.dtype)
    hh, ww = min(h, a.shape[0]), min(w, a.shape[1])
    out[:hh, :ww] = a[:hh, :ww]
    return out


def ink_mask(rgb):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[..., 0].astype(int), hsv[..., 1].astype(int), hsv[..., 2].astype(int)
    green = (h >= 35) & (h <= 95) & (s >= 60) & (v <= 220)
    blue = (h > 95) & (h <= 125) & (s >= 90) & (v <= 220)
    black = (v <= 45)
    m = (green | blue | black).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    keep = np.zeros_like(m)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] >= 20:
            keep[lab == i] = 1
    return keep.astype(bool)


def one(args):
    name, blob, img_bytes, mask_bytes, provider = args
    image_id = name.split("_")[1]
    z = np.load(io.BytesIO(blob))
    gt = z["gt"].astype(bool); av = z["annotated_valid"].astype(bool)
    valid = z["valid"].astype(bool)
    fus = z["fusion"].astype(np.float32) >= THR
    l1 = (z["prob_L1"].astype(np.float32) >= THR) & z["support_L1"].astype(bool)
    shape = gt.shape
    mpp_img, rgb = level16(img_bytes)
    mpp_mask, lab = level16(mask_bytes, channel0=True)
    rgb, lab = fit(rgb[..., :3], shape), fit(lab, shape)
    r = {"image_id": image_id, "provider": provider, "um_per_px_image": mpp_img, "um_per_px_mask": mpp_mask,
         "eval_grid_h": shape[0], "eval_grid_w": shape[1]}
    tumor_raw = np.isin(lab, (3, 4, 5)) if provider == "radboud" else (lab == 2)
    r["mapping_mismatch_av_pixels"] = int((tumor_raw[av] != gt[av]).sum())
    r["label0_in_av_pixels"] = int((lab[av] == 0).sum())
    labels = (1, 2, 3, 4, 5) if provider == "radboud" else (1, 2)
    for L in labels:
        sel = valid & (lab == L)
        r[f"label{L}_pixels"] = int(sel.sum())
        r[f"label{L}_pred_fusion"] = int((sel & fus).sum())
        r[f"label{L}_pred_L1"] = int((sel & l1).sum())
    if provider == "radboud":
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ENVELOPE_RADIUS + 1,) * 2)
        env = cv2.morphologyEx(np.isin(lab, (3, 4, 5)).astype(np.uint8), cv2.MORPH_CLOSE, k).astype(bool)
        for tag, sel in (("inside", valid & (lab == 1) & env), ("outside", valid & (lab == 1) & ~env)):
            r[f"stroma_{tag}_envelope_pixels"] = int(sel.sum())
            r[f"stroma_{tag}_envelope_pred_fusion"] = int((sel & fus).sum())
            r[f"stroma_{tag}_envelope_pred_L1"] = int((sel & l1).sum())
    ink = ink_mask(rgb)
    r["ink_pixels"] = int(ink.sum())
    near = cv2.dilate(ink.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * INK_MARGIN + 1,) * 2)).astype(bool)
    r["valid_pixels_near_ink"] = int((valid & near).sum())
    keep = valid & ~near
    for k_, v in zip(("tp", "fp", "tn", "fn"), cs.counts(fus, gt, keep)):
        r[f"fusion_noink_{k_}"] = v
    return r


def main():
    repo, complement, panda_zip, out = map(Path, sys.argv[1:5])
    out.mkdir(parents=True, exist_ok=True)
    split = pd.read_csv(repo / "PANDA_Reproducibility/paper_results/tables/split_ids_reproducibility.csv")
    test = split[split.split == "test"].set_index("image_id")
    names = [f"test_{i}_ds16_validsupport_v2.npz" for i in sorted(test.index)]
    in_git = set(cs.subprocess.run(["git", "-C", str(repo), "ls-tree", "--name-only", f"HEAD:{cs.CACHE_DIR}"],
                                   capture_output=True, text=True, check=True).stdout.split())
    zc = zipfile.ZipFile(complement); in_zip = {Path(n).name: n for n in zc.namelist()}
    zp = zipfile.ZipFile(panda_zip)

    def source():
        git_names = [n for n in names if n in in_git]
        gen = cs.git_blobs(repo, git_names)
        for n in names:
            blob = next(gen)[1] if n in in_git else zc.read(in_zip[n])
            iid = n.split("_")[1]
            yield (n, blob, zp.read(f"train_images/{iid}.tiff"), zp.read(f"train_label_masks/{iid}_mask.tiff"),
                   test.loc[iid, "provider"])

    rows = []
    with ProcessPoolExecutor(max_workers=3) as ex:
        for i, r in enumerate(cs.bounded_map(ex, one, source(), window=6), 1):
            rows.append(r)
            if i % 100 == 0:
                print(f"{i}/{len(names)}", flush=True)
    pd.DataFrame(rows).to_csv(out / "panda_test_raw_mask_per_slide.csv", index=False)
    print("rows", len(rows))


if __name__ == "__main__":
    main()
