#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SICAPv2 external test on the official Mendeley release (revision of 30 Oct 2026).

The first SICAPv2 run used a renamed Kaggle copy (0.jpg, 1.jpg, ...), so the
slide, patient and partition information was lost, every tile received the
constant coordinate (0.5, 0.5), and the 18 images of the copy's Val folder
were not evaluated. This script evaluates the official release
(doi:10.17632/9xxm58dvs3.2, SICAPv2.zip, SHA-256 a28aa6a0...7355916) with the
same per-tile protocol as run_external_sicapv2_test_LIVE_VISUALS.py:

* frozen PANDA checkpoint, training-derived Vahadane target, threshold 0.43;
* valid pixels = core.generate_tissue_mask on the H&E tile (same fallbacks);
* reference tumor = mask value > 0. The release masks are lossless PNG with
  values 0 (non-cancerous) and 3, 4, 5 (Gleason patterns 3, 4, 5), so no
  compression threshold is involved.

Each tile is predicted twice: with the constant coordinate (0.5, 0.5), as in
the first run (primary), and with its centre relative to the bounding box of
its slide region from the xini/yini file-name fields (sensitivity).

Tiles overlap by 50% (xini/yini step 1024 at 40x = 256 px at 10x). Besides
the per-tile counts, every slide region is rebuilt on its 256-px lattice:
each 10x pixel is scored once, with the mean probability of the tiles that
cover it (as in the PANDA dense reconstruction), the reference of any covering
tile (overlapping masks agree) and the union of the tiles' valid masks. This
is done separately for three tile subsets: the official patient-level test
partition (partition/Test/Test.xlsx), all tiles listed in the partition files
(partition/Test/Train.xlsx + Test.xlsx), and all tiles.

Usage:
    set PANDA_DATA_ROOT=<work root with runs/, stain_target_train.png, paper_results/tables/best_threshold.json>
    python run_external_sicapv2_official.py <SICAPv2 folder> <output folder>   (launcher)
"""
import os
import re
import sys
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image

from panda_final_pipeline import FinalConfig
import panda_camelyon_style_pipeline as core
import panda_stream_paper as paper

RX = re.compile(r"^(?P<slide>[^_]+)_Block_Region_(?P<region>\d+)_(?P<r>\d+)_(?P<c>\d+)_xini_(?P<x>-?\d+)_yini_(?P<y>-?\d+)\.jpg$")
TILE, HALF = 512, 256
CONST = (0.5, 0.5)
SUBSETS = ("test_partition", "partition_listed", "all")

def _quiet():
    import logging
    for name in ("tiatoolbox", ""):
        logging.getLogger(name).setLevel(logging.ERROR)


def build_index(root):
    names = sorted(p.name for p in (root / "images").glob("*.jpg"))
    d = pd.DataFrame([RX.match(n).groupdict() | {"image_name": n} for n in names])
    for c in ("region", "r", "c", "x", "y"):
        d[c] = d[c].astype(int)
    labels = pd.read_excel(root / "wsi_labels.xlsx").rename(columns={"slide_id": "slide"})
    d = d.merge(labels[["slide", "patient_id", "Gleason_primary", "Gleason_secondary"]], on="slide", how="left")
    test = set(pd.read_excel(root / "partition/Test/Test.xlsx").image_name)
    train = set(pd.read_excel(root / "partition/Test/Train.xlsx").image_name)
    d["test_partition"] = d.image_name.isin(test)
    d["partition_listed"] = d.image_name.isin(test | train)
    d["all"] = True
    d["image_path"] = [str(root / "images" / n) for n in d.image_name]
    d["mask_path"] = [str(root / "masks" / n.replace(".jpg", ".png")) for n in d.image_name]
    g = d.groupby(["slide", "region"])
    x0, y0 = g.x.transform("min"), g.y.transform("min")
    span_x = g.x.transform("max") - x0 + 2 * 1024
    span_y = g.y.transform("max") - y0 + 2 * 1024
    d["coord_x_region"] = ((d.x - x0 + 1024) / span_x).clip(0, 1)
    d["coord_y_region"] = ((d.y - y0 + 1024) / span_y).clip(0, 1)
    return d


def tile_counts(prob, gt, valid, thr):
    return list(paper.counts(prob >= thr, gt, valid))


def main():
    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    (out / "tables").mkdir(parents=True, exist_ok=True)
    _quiet()
    idx = build_index(root)
    idx.drop(columns=["image_path", "mask_path"]).to_csv(out / "tables" / "official_index.csv", index=False)
    print(f"[INDEX] {len(idx)} tiles, {idx.slide.nunique()} slides, {idx.patient_id.nunique()} patients, "
          f"test partition {int(idx.test_partition.sum())}, listed {int(idx.partition_listed.sum())}", flush=True)

    cfg = FinalConfig()
    model, device, ck = core.load_best_model(cfg, None)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        model = model.to(memory_format=torch.channels_last)
    thr = float(json.load(open(Path(cfg.work) / "paper_results/tables/best_threshold.json"))["threshold"])
    workers = int(os.environ.get("SICAP_PREP_WORKERS", "4"))
    bs = int(os.environ.get("SICAP_BATCH", "16"))

    tile_csv, region_csv = out / "tables" / "per_tile_metrics.csv", out / "tables" / "per_region_dedup_counts.csv"
    done_regions = set()
    tile_rows, region_rows = [], []
    if tile_csv.exists() and region_csv.exists():
        tile_rows = pd.read_csv(tile_csv).to_dict("records")
        region_rows = pd.read_csv(region_csv).to_dict("records")
        done_regions = {(r["slide"], int(r["region"])) for r in region_rows}

    # Official test-partition regions first, so that the primary result is available early.
    order = (idx.groupby(["slide", "region"]).test_partition.any().sort_values(ascending=False).index.tolist())
    import norm_pool
    pool = ProcessPoolExecutor(max_workers=workers, initializer=norm_pool.init, initargs=(str(cfg.stain_target_path),))

    @torch.inference_mode()
    def predict(rgbs, coords):
        x = torch.from_numpy(np.stack(rgbs)).permute(0, 3, 1, 2).float().div_(255.0)
        c = torch.tensor(np.asarray(coords, np.float32))
        x, c = x.to(device), c.to(device)
        if device.type == "cuda":
            x = x.contiguous(memory_format=torch.channels_last)
        with torch.amp.autocast("cuda", enabled=bool(cfg.amp and device.type == "cuda")):
            return torch.sigmoid(model(x, c))[:, 0].float().cpu().numpy()

    for k, (slide, region) in enumerate(order, 1):
        if (slide, region) in done_regions:
            continue
        t = idx[(idx.slide == slide) & (idx.region == region)].reset_index(drop=True)
        prepared = list(pool.map(norm_pool.prepare_sicap, zip(t.image_path, t.mask_path), chunksize=2))
        probs = {"const": [], "region": []}
        for s in range(0, len(t), bs):
            rgbs = [p[0] for p in prepared[s:s + bs]]
            probs["const"].extend(predict(rgbs, [CONST] * len(rgbs)))
            rc = list(zip(t.coord_x_region[s:s + bs], t.coord_y_region[s:s + bs]))
            probs["region"].extend(predict(rgbs, rc))
        for i, r in t.iterrows():
            _, gt, valid = prepared[i]
            row = {c: r[c] for c in ("image_name", "slide", "region", "r", "c", "patient_id",
                                     "test_partition", "partition_listed")}
            for mode in ("const", "region"):
                for name, v in zip(("tp", "fp", "tn", "fn"), tile_counts(probs[mode][i], gt, valid, thr)):
                    row[f"{mode}_{name}"] = v
            bm = paper.boundary_metrics_pixels(probs["const"][i] >= thr, gt, valid, eval_downsample=1.0)
            row.update({f"const_{k_}": v for k_, v in bm.items() if k_ in ("hd95_eval_px", "assd_eval_px", "surface_dice")})
            tile_rows.append(row)
        # 256-px lattice reconstruction of the region, per subset and coordinate mode
        nr, nc = int(t.r.max()) + 2, int(t.c.max()) + 2
        for subset in SUBSETS:
            sel = t.index[t[subset]].tolist()
            if not sel:
                continue
            for mode in ("const", "region"):
                acc = {}
                for i in sel:
                    p, (_, gt, valid) = probs[mode][i], prepared[i]
                    r0, c0 = int(t.r[i]), int(t.c[i])
                    for dr in (0, 1):
                        for dc in (0, 1):
                            key = (r0 + dr, c0 + dc)
                            sl = (slice(dr * HALF, (dr + 1) * HALF), slice(dc * HALF, (dc + 1) * HALF))
                            a = acc.get(key)
                            if a is None:
                                acc[key] = [p[sl].astype(np.float32).copy(), 1, gt[sl].copy(), valid[sl].copy()]
                            else:
                                a[0] += p[sl]; a[1] += 1; a[2] |= gt[sl]; a[3] |= valid[sl]
                tot = np.zeros(4, np.int64)
                for s_, n_, g_, v_ in acc.values():
                    tot += np.asarray(tile_counts(s_ / n_, g_, v_, thr), np.int64)
                region_rows.append({"slide": slide, "region": region, "patient_id": t.patient_id[0],
                                    "subset": subset, "mode": mode, "n_tiles": len(sel), "n_blocks": len(acc),
                                    **dict(zip(("tp", "fp", "tn", "fn"), map(int, tot)))})
        pd.DataFrame(tile_rows).to_csv(tile_csv, index=False)
        pd.DataFrame(region_rows).to_csv(region_csv, index=False)
        print(f"[REGION {k}/{len(order)}] {slide} r{region} tiles={len(t)} test={int(t.test_partition.sum())}", flush=True)
    pool.shutdown()
    json.dump({"dataset": "SICAPv2", "version": 2, "doi": "10.17632/9xxm58dvs3.2",
               "archive_sha256": "a28aa6a0e3831217bf31575cecfc1846607ace7ff25a4392f07052bbc7355916",
               "threshold": thr, "checkpoint": str(ck), "tiles": int(len(idx)),
               "reference": "mask value > 0 (PNG values 0, 3, 4, 5)",
               "valid": "core.generate_tissue_mask on the H&E tile",
               "coordinates": {"const": CONST, "region": "tile centre relative to its slide-region bounding box"}},
              open(out / "tables" / "run_settings.json", "w"), indent=2)
    print("[DONE]", flush=True)

