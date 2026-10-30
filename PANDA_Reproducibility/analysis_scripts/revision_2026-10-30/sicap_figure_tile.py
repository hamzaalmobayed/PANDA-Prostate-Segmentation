"""Prediction and error map of the SICAPv2 tile shown in Fig 10D (revision of 30 Oct 2026).

Fig 10D showed image 0.jpg of the renamed Kaggle copy used in the first
SICAPv2 run. That copy and its prediction files are not kept. Its H&E file is
byte-identical (SHA-256 e741a2de...) to the official release tile
16B0001851_Block_Region_1_0_0_xini_6803_yini_59786.jpg. This script predicts
that tile as sicapv2_official.py does with the constant coordinate (0.5, 0.5)
of the first run, in the tile's batch of that run, checks that the TP/FP/TN/FN
counts agree with the stored const_* counts of per_tile_metrics.csv (up to
0.05% of the tile's valid pixels: GPU rounding moves a few pixels across the
threshold; both sets of counts are written to the JSON), and writes the
tile, its mask, the prediction mask and the error map read by
figure_scripts/make_figures_9_10.py. The reference is the release's
lossless PNG mask (> 0).

Usage:
    set PANDA_DATA_ROOT=<work root used for the official run>
    python sicap_figure_tile.py <SICAPv2 folder> <PANDA_Reproducibility>
"""
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

PIPE = Path(__file__).resolve().parents[3] / "PANDA_Pipeline"
sys.path.insert(0, str(PIPE))

TILE = "16B0001851_Block_Region_1_0_0_xini_6803_yini_59786"
KAGGLE_SHA = "e741a2de01625da644e7e03c094eeaed992f0bf5b5b4e1725af1a6703b69089c"
CONST = (0.5, 0.5)
TN, TP, FN, FP = (230, 230, 230), (60, 170, 90), (240, 190, 40), (210, 60, 60)


def main():
    sic, repo = Path(sys.argv[1]), Path(sys.argv[2])
    out = repo / "external_sicapv2_official" / "results" / "figure_tile"
    out.mkdir(parents=True, exist_ok=True)
    img, msk = sic / "images" / f"{TILE}.jpg", sic / "masks" / f"{TILE}.png"
    assert hashlib.sha256(img.read_bytes()).hexdigest() == KAGGLE_SHA, "not the Kaggle 0.jpg image"

    import norm_pool
    import panda_camelyon_style_pipeline as core
    import panda_stream_paper as paper
    from panda_final_pipeline import FinalConfig

    import sicapv2_official as so

    cfg = FinalConfig()
    norm_pool.init(str(cfg.stain_target_path))
    # the tile's batch of the official run (same region order, batch size 16), so that the
    # GPU arithmetic, and hence the thresholded pixels, are the same
    idx = so.build_index(sic)
    slide, region = idx.set_index("image_name").loc[f"{TILE}.jpg", ["slide", "region"]]
    t = idx[(idx.slide == slide) & (idx.region == region)].reset_index(drop=True)
    i = int(t.index[t.image_name == f"{TILE}.jpg"][0])
    bs = 16
    s = (i // bs) * bs
    batch = [norm_pool.prepare_sicap(p) for p in zip(t.image_path[s:s + bs], t.mask_path[s:s + bs])]
    _, gt, valid = batch[i - s]
    model, device, ck = core.load_best_model(cfg, None)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        model = model.to(memory_format=torch.channels_last)
    thr = float(json.load(open(Path(cfg.work) / "paper_results/tables/best_threshold.json"))["threshold"])
    with torch.inference_mode():
        x = torch.from_numpy(np.stack([b[0] for b in batch])).permute(0, 3, 1, 2).float().div_(255.0).to(device)
        c = torch.tensor([CONST] * len(batch), dtype=torch.float32, device=device)
        if device.type == "cuda":
            x = x.contiguous(memory_format=torch.channels_last)
        with torch.amp.autocast("cuda", enabled=bool(cfg.amp and device.type == "cuda")):
            prob = torch.sigmoid(model(x, c))[i - s, 0].float().cpu().numpy()
    pred = prob >= thr
    got = dict(zip(("tp", "fp", "tn", "fn"), paper.counts(pred, gt, valid)))
    row = pd.read_csv(repo / "external_sicapv2_official/results/tables/per_tile_metrics.csv").set_index("image_name").loc[f"{TILE}.jpg"]
    stored = {k: int(row[f"const_{k}"]) for k in got}
    max_diff = max(abs(got[k] - stored[k]) for k in got)
    # GPU rounding (cuDNN algorithm choice) moves a few pixels across the threshold
    assert max_diff <= 0.0005 * sum(got.values()), (got, stored)

    err = np.full(gt.shape + (3,), 255, np.uint8)
    err[~gt & ~pred & valid] = TN
    err[gt & pred & valid] = TP
    err[gt & ~pred & valid] = FN
    err[~gt & pred & valid] = FP
    shutil.copy2(img, out / img.name)  # SICAPv2 is CC BY 4.0; the two files make the figure input self-contained
    shutil.copy2(msk, out / msk.name)
    Image.fromarray((pred * 255).astype(np.uint8)).save(out / f"{TILE}_pred_mask.png")
    Image.fromarray(err).save(out / f"{TILE}_error_map.png")
    dice = 2 * got["tp"] / (2 * got["tp"] + got["fp"] + got["fn"])
    info = {"tile": f"{TILE}.jpg", "kaggle_copy_name": "0.jpg", "image_sha256": KAGGLE_SHA,
            "mask_sha256": hashlib.sha256(msk.read_bytes()).hexdigest(), "coordinates": CONST, "threshold": thr,
            "checkpoint": str(Path(ck).name), "device": device.type,
            "batch": {"region": f"{slide} region {region}", "tiles": [s, s + len(batch)], "position": i - s},
            "counts": got, "counts_official_run_per_tile_metrics_const": stored,
            "max_abs_count_difference_px": max_diff, "valid_px": sum(got.values()), "dice": round(dice, 4),
            "error_map_colours": {"overlap": TP, "false_negative": FN, "false_positive": FP, "true_negative_tissue": TN,
                                  "outside_valid": [255, 255, 255]}}
    (out / f"{TILE}.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
