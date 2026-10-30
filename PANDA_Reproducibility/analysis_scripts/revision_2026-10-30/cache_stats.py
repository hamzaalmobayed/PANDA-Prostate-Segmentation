"""Per-slide statistics from the saved PANDA WSI caches (revision of 30 Oct 2026).

Reads every validation and test cache once and writes one row per WSI with
the counts that the revision analyses need, so that no later analysis reads
the caches again:

* own-support counts (tp, fp, tn, fn) for L0, L1, L2 and fusion at the common
  threshold 0.43 and at the validation-selected level thresholds of Table 9
  (L0 0.47, L1 0.40, L2 0.54, fusion 0.43). A single level is scored on
  annotated_valid & support_L<l>, fusion on valid, as in panda_stream_paper.py;
  maps are cast to float32 before thresholding (README, "B4 precision").
* common-support counts at 0.43: every map scored on annotated_valid &
  fusion_support, unsupported single-level pixels counted as negative (Table 6).
* support coverage: annotated-valid and reference-tumor pixels inside each
  level's support.
* test only: 8-connected components of the reference tumor (with their area and
  whether fusion or L1 predicts any pixel of them, on the common region) and of
  the fused false positives.

Caches are read from the repository (git blobs at HEAD, so the 45 GB cache
folder need not be checked out) and from panda_wsi_cache_complement.zip.

Usage:
    python cache_stats.py <repo_root> <panda_wsi_cache_complement.zip> <output_dir>
"""
import io
import subprocess
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

MAPS = ("L0", "L1", "L2", "fusion")
COMMON = 0.43
LEVEL_THR = {"L0": 0.47, "L1": 0.40, "L2": 0.54, "fusion": 0.43}
CACHE_DIR = "PANDA_Reproducibility/paper_results/wsi_cache"


def counts(pred, gt, region):
    p, g = pred[region], gt[region]
    return [int((p & g).sum()), int((p & ~g).sum()), int((~p & ~g).sum()), int((~p & g).sum())]


def components(mask):
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    return n - 1, lab, st[1:, cv2.CC_STAT_AREA]


def one(args):
    name, blob = args
    split, image_id = name.split("_")[0], name.split("_")[1]
    z = np.load(io.BytesIO(blob))
    gt = z["gt"].astype(bool)
    av = z["annotated_valid"].astype(bool)
    fsup = z["fusion_support"].astype(bool)
    common = av & fsup
    r = {"split": split, "image_id": image_id, "annotated_valid_pixels": int(av.sum()),
         "reference_tumor_pixels_av": int((gt & av).sum()), "common_region_pixels": int(common.sum()),
         "reference_tumor_pixels_common": int((gt & common).sum())}
    pred_common = {}
    for m in MAPS:
        if m == "fusion":
            prob, sup, own = z["fusion"].astype(np.float32), fsup, z["valid"].astype(bool)
        else:
            prob, sup = z[f"prob_{m}"].astype(np.float32), z[f"support_{m}"].astype(bool)
            own = av & sup
        for tag, t in (("own", COMMON), ("lvlthr", LEVEL_THR[m])):
            for k, v in zip(("tp", "fp", "tn", "fn"), counts(prob >= np.float32(t), gt, own)):
                r[f"{m}_{tag}_{k}"] = v
        pc = (prob >= np.float32(COMMON)) & sup
        pred_common[m] = pc
        for k, v in zip(("tp", "fp", "tn", "fn"), counts(pc, gt, common)):
            r[f"{m}_common_{k}"] = v
        r[f"{m}_support_av_pixels"] = int((av & sup).sum())
        r[f"{m}_support_reference_tumor_pixels"] = int((gt & av & sup).sum())
    if split == "test":
        ref = gt & common
        n, lab, area = components(ref)
        r["n_reference_foci"] = n
        if n:
            hit_f = np.bincount(lab[pred_common["fusion"] & ref].ravel(), minlength=n + 1)[1:] > 0
            hit_l1 = np.bincount(lab[pred_common["L1"] & ref].ravel(), minlength=n + 1)[1:] > 0
            r["reference_foci"] = ";".join(f"{a}:{int(f)}:{int(l)}" for a, f, l in zip(area, hit_f, hit_l1))
        else:
            r["reference_foci"] = ""
        valid = z["valid"].astype(bool)
        fp = (z["fusion"].astype(np.float32) >= np.float32(COMMON)) & valid & ~gt
        n, _, area = components(fp)
        r["n_fusion_fp_components"] = n
        r["fusion_fp_component_areas"] = ";".join(str(int(a)) for a in sorted(area, reverse=True))
    return r


def bounded_map(ex, fn, items, window=8):
    """Like ex.map, but keeps at most `window` inputs in memory (Executor.map reads its whole input first)."""
    from collections import deque
    pending = deque()
    for it in items:
        pending.append(ex.submit(fn, it))
        if len(pending) >= window:
            yield pending.popleft().result()
    while pending:
        yield pending.popleft().result()


def git_blobs(repo, names):
    p = subprocess.Popen(["git", "-C", str(repo), "cat-file", "--batch"], stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE)
    try:
        for n in names:
            p.stdin.write(f"HEAD:{CACHE_DIR}/{n}\n".encode()); p.stdin.flush()
            header = p.stdout.readline().split()
            if header[-1] == b"missing":
                raise FileNotFoundError(n)
            size = int(header[2])
            data = p.stdout.read(size); p.stdout.read(1)
            yield n, data
    finally:
        p.stdin.close(); p.wait()


def main():
    repo, complement, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    split = pd.read_csv(repo / "PANDA_Reproducibility/paper_results/tables/split_ids_reproducibility.csv")
    want = {f"{s}_{i}_ds16_validsupport_v2.npz" for s, i in zip(split.split, split.image_id) if s in ("val", "test")}
    in_git = set(subprocess.run(["git", "-C", str(repo), "ls-tree", "--name-only", f"HEAD:{CACHE_DIR}"],
                                capture_output=True, text=True, check=True).stdout.split())
    zf = zipfile.ZipFile(complement)
    in_zip = {Path(n).name: n for n in zf.namelist()}
    missing = sorted(want - in_git - set(in_zip))
    if missing:
        sys.exit(f"{len(missing)} caches missing, e.g. {missing[0]}")
    from_git = sorted(want & in_git)
    from_zip = sorted(want - in_git)
    print(f"{len(want)} caches: {len(from_git)} from git, {len(from_zip)} from the release archive", flush=True)

    def source():
        yield from git_blobs(repo, from_git)
        for n in from_zip:
            yield n, zf.read(in_zip[n])

    rows = []
    with ProcessPoolExecutor(max_workers=4) as ex:
        for i, r in enumerate(bounded_map(ex, one, source()), 1):
            rows.append(r)
            if i % 200 == 0:
                print(f"{i}/{len(want)}", flush=True)
    df = pd.DataFrame(rows).merge(split[["image_id", "provider"]], on="image_id", how="left")
    for s in ("val", "test"):
        d = df[df.split == s].sort_values("image_id")
        d.to_csv(out / f"cache_per_slide_stats_{s}.csv", index=False)
        print(s, len(d), flush=True)


if __name__ == "__main__":
    main()
