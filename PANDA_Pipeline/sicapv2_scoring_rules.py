#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SICAPv2 Dice of the official test partition under three scoring rules (1 Nov 2026).

The official run (sicapv2_official.py) scores only the tissue pixels that the
PANDA tissue rule detects on each H&E tile: saturation above max(Otsu, 0.20),
then morphological opening and closing and removal of components smaller than
500 px, with the fallbacks of norm_pool.prepare_sicap. Its probabilities were
not saved, so the test-partition tiles are predicted again with the same
checkpoint, Vahadane target, threshold and constant coordinate (0.5, 0.5),
and TP/FP/TN/FN are counted under three rules:

* rule:  the tissue rule of the official run (its counts are reproduced);
* all:   every pixel of the tile;
* fixed: the tissue rule with the fixed saturation threshold 0.20 instead of Otsu.

Pooled Dice is computed over patches and over the reconstructed slide regions
(each 10x pixel scored once, with the mean probability of the tiles covering
it, as in sicapv2_official.py). The per-tile counts of the tissue rule are
compared with the official run's per_tile_metrics.csv.

Usage:
    set PANDA_DATA_ROOT=<work root with runs/, stain_target_train.png, paper_results/tables/best_threshold.json>
    python run_sicapv2_scoring_rules.py <SICAPv2 folder> <PANDA_Reproducibility>   (launcher)
Output: external_sicapv2_official/results/tables/scoring_rules_per_tile.csv
        external_sicapv2_official/summary/SICAP_scoring_rules.csv
"""
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from panda_final_pipeline import FinalConfig
import panda_camelyon_style_pipeline as core
from sicapv2_official import CONST, HALF, _quiet, build_index, tile_counts

RULES = ("rule", "all", "fixed")
COUNTS = ("tp", "fp", "tn", "fn")


def main():
    root, rep = Path(sys.argv[1]), Path(sys.argv[2])
    off = rep / "external_sicapv2_official"
    _quiet()
    idx = build_index(root)
    idx = idx[idx.test_partition].reset_index(drop=True)
    cfg = FinalConfig()
    model, device, ck = core.load_best_model(cfg, None)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        model = model.to(memory_format=torch.channels_last)
    thr = float(json.load(open(Path(cfg.work) / "paper_results/tables/best_threshold.json"))["threshold"])
    bs = int(os.environ.get("SICAP_BATCH", "16"))
    import norm_pool
    pool = ProcessPoolExecutor(max_workers=int(os.environ.get("SICAP_PREP_WORKERS", "4")),
                               initializer=norm_pool.init, initargs=(str(cfg.stain_target_path),))

    @torch.inference_mode()
    def predict(rgbs):
        x = torch.from_numpy(np.stack(rgbs)).permute(0, 3, 1, 2).float().div_(255.0)
        c = torch.tensor(np.asarray([CONST] * len(rgbs), np.float32))
        x, c = x.to(device), c.to(device)
        if device.type == "cuda":
            x = x.contiguous(memory_format=torch.channels_last)
        with torch.amp.autocast("cuda", enabled=bool(cfg.amp and device.type == "cuda")):
            return torch.sigmoid(model(x, c))[:, 0].float().cpu().numpy()

    tile_rows, region_tot = [], {r: np.zeros(4, np.int64) for r in RULES}
    regions = idx.groupby(["slide", "region"]).size().index.tolist()
    for k, (slide, region) in enumerate(regions, 1):
        t = idx[(idx.slide == slide) & (idx.region == region)].reset_index(drop=True)
        prepared = list(pool.map(norm_pool.prepare_sicap_rules, zip(t.image_path, t.mask_path), chunksize=2))
        probs = []
        for s in range(0, len(t), bs):
            probs.extend(predict([p[0] for p in prepared[s:s + bs]]))
        acc = {}
        for i, r in t.iterrows():
            p, (_, gt, valid) = probs[i], prepared[i]
            row = {"image_name": r.image_name, "slide": slide, "region": region, "patient_id": r.patient_id,
                   "ref_px": int(gt.sum())}
            for rule in RULES:
                row.update({f"{rule}_{n}": v for n, v in zip(COUNTS, tile_counts(p, gt, valid[rule], thr))})
                row[f"{rule}_scored_px"] = int(valid[rule].sum())
            tile_rows.append(row)
            r0, c0 = int(r.r), int(r.c)
            for dr in (0, 1):
                for dc in (0, 1):
                    key = (r0 + dr, c0 + dc)
                    sl = (slice(dr * HALF, (dr + 1) * HALF), slice(dc * HALF, (dc + 1) * HALF))
                    a = acc.get(key)
                    if a is None:
                        acc[key] = [p[sl].astype(np.float32).copy(), 1, gt[sl].copy(),
                                    {rule: valid[rule][sl].copy() for rule in RULES}]
                    else:
                        a[0] += p[sl]; a[1] += 1; a[2] |= gt[sl]
                        for rule in RULES:
                            a[3][rule] |= valid[rule][sl]
        for s_, n_, g_, v_ in acc.values():
            for rule in RULES:
                region_tot[rule] += np.asarray(tile_counts(s_ / n_, g_, v_[rule], thr), np.int64)
        print(f"[REGION {k}/{len(regions)}] {slide} r{region} tiles={len(t)}", flush=True)
    pool.shutdown()

    tiles = pd.DataFrame(tile_rows)
    tiles.to_csv(off / "results/tables/scoring_rules_per_tile.csv", index=False)
    ref = pd.read_csv(off / "results/tables/per_tile_metrics.csv").set_index("image_name").loc[tiles.image_name]
    diff = np.abs(tiles[[f"rule_{n}" for n in COUNTS]].to_numpy() - ref[[f"const_{n}" for n in COUNTS]].to_numpy())
    out = []
    for rule in RULES:
        for unit, (tp, fp, tn, fn) in (("patches", tiles[[f"{rule}_{n}" for n in COUNTS]].sum().to_numpy()),
                                       ("slide regions", region_tot[rule])):
            out.append({"rule": rule, "unit": unit, "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
                        "dice": 2 * tp / (2 * tp + fp + fn), "precision": tp / (tp + fp), "recall": tp / (tp + fn),
                        "reference_scored_share": (tp + fn) / tiles.ref_px.sum() if unit == "patches" else np.nan})
    res = pd.DataFrame(out)
    res["tiles"] = len(tiles)
    res["threshold"] = thr
    res["official_tile_counts_max_abs_diff"] = int(diff.max())
    res["official_tile_counts_diff_px"] = int(diff.sum())
    res.to_csv(off / "summary/SICAP_scoring_rules.csv", index=False)
    print(res.to_string(index=False))
    print("[DONE]", flush=True)
