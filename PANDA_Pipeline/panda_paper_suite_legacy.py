
from __future__ import annotations

"""
PANDA prostate segmentation — PAPER RESULTS & PILOT EXPERIMENT SUITE
===================================================================

This module sits on top of panda_camelyon_style_pipeline.py.

It adds the paper-facing outputs requested for a prostate-cancer version of the
CAMELYON study, while avoiding CAMELYON-only clinical analyses such as
ITC/micrometastasis/macrometastasis FROC subtype analysis.

Main outputs
------------
- Training curves.
- Patch-level metrics and confusion matrices for L1/L2/L3/L4 and all-level pooled.
- Validation threshold sweep and optimal threshold selection.
- WSI-level reconstruction on a common physical grid for each level + four-level fusion.
- Per-slide WSI Dice/IoU/precision/recall/specificity/accuracy.
- Global WSI metrics and 95% slide-bootstrap confidence intervals.
- Per-provider (Radboud/Karolinska) metrics + 95% CIs.
- Per-slide Dice distribution figure.
- Provider comparison figure.
- Ground truth / prediction / overlay / error-map qualitative figures.
- Best/median/worst slide examples.
- Statistical paired bootstrap comparison: fusion vs strongest single level.
- Paper-ready CSV/JSON summaries.
- Pilot ablation study on a fixed small slide-disjoint subset.
- Pilot normalization comparison: none vs Macenko vs Vahadane.
- Pilot KMeans ablation and coordinate ablation.
- Optional pilot encoder comparison using segmentation_models_pytorch if installed.
- Optional pilot leave-one-provider-out experiment.

Scientific scope
----------------
PANDA is prostate biopsy tissue. CAMELYON-specific lesion subtypes
(ITC, micrometastasis, macrometastasis) are NOT copied. A note is generated explaining
why CAMELYON-style FROC-by-metastasis-subtype is not reported here.

Run after the core pipeline has completed training:
  python panda_paper_suite.py paper-all --work-root D:/PANDA_PROSTATE --data-root D:/PANDA

Pilot experiments:
  python panda_paper_suite.py pilot-experiments --work-root D:/PANDA_PROSTATE --data-root D:/PANDA

Everything:
  python run_full_panda_paper.py --data-root D:/PANDA --work-root D:/PANDA_PROSTATE
"""

import argparse
import gc
import json
import math
import os
import random
import shutil
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import panda_camelyon_style_pipeline as core

try:
    from tiatoolbox.tools.stainnorm import MacenkoNormalizer
except Exception:
    MacenkoNormalizer = None

try:
    import segmentation_models_pytorch as smp
except Exception:
    smp = None


# ---------------------------------------------------------------------------
# Config additions
# ---------------------------------------------------------------------------

PAPER_EVAL_DOWNSAMPLE = 16.0
BOOTSTRAP_ITERS = 1000
THRESHOLDS = np.round(np.arange(0.30, 0.701, 0.01), 2)
PILOT_TRAIN_PATCHES = 8000
PILOT_VAL_PATCHES = 2000
PILOT_EPOCHS = 8
PILOT_PATIENCE = 3
PILOT_BATCH_SIZE = 8


def paper_root(cfg: core.Config) -> Path:
    p = cfg.work / "paper_results"
    p.mkdir(parents=True, exist_ok=True)
    for sub in ["tables", "figures", "wsi_cache", "qualitative", "experiments", "stats"]:
        (p / sub).mkdir(parents=True, exist_ok=True)
    return p


def latest_run_dir(cfg: core.Config) -> Path:
    f = cfg.work / "latest_run.json"
    if not f.exists():
        raise core.QCError("latest_run.json not found. Run core training first.")
    return Path(json.load(open(f, "r", encoding="utf-8"))["run_dir"])


def _safe_div(a, b):
    return float(a) / float(b) if b else 0.0


def metrics_from_counts(tp: int, fp: int, tn: int, fn: int) -> Dict[str, float]:
    return {
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "dice": _safe_div(2 * tp, 2 * tp + fp + fn),
        "iou": _safe_div(tp, tp + fp + fn),
        "precision": _safe_div(tp, tp + fp),
        "recall": _safe_div(tp, tp + fn),
        "specificity": _safe_div(tn, tn + fp),
        "accuracy": _safe_div(tp + tn, tp + tn + fp + fn),
    }


def counts_from_binary(pred: np.ndarray, gt: np.ndarray, valid: Optional[np.ndarray] = None):
    p = pred.astype(bool)
    g = gt.astype(bool)
    if valid is not None:
        v = valid.astype(bool)
        p = p[v]; g = g[v]
    tp = int(np.logical_and(p, g).sum())
    fp = int(np.logical_and(p, ~g).sum())
    tn = int(np.logical_and(~p, ~g).sum())
    fn = int(np.logical_and(~p, g).sum())
    return tp, fp, tn, fn


def save_json(path: Path, obj: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)


# ---------------------------------------------------------------------------
# PAPER FIGURE 1 — pipeline overview
# ---------------------------------------------------------------------------

def make_pipeline_overview(cfg: core.Config):
    out = paper_root(cfg) / "figures" / "01_pipeline_overview.png"
    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.axis("off")
    labels = [
        "PANDA WSI\n+ label mask",
        "WSI-level\n70/15/15 split",
        "Tissue mask\n+ QC",
        "4 logical\npyramid scales",
        "512×512\npatches",
        "8 buckets\n4 levels × 2 classes",
        "EffNet-B4\nfeatures + KMeans",
        "Vahadane\n+ augmentation",
        "ONE U-Net\nEffNet-B4",
        "Per-level +\nWSI fusion results",
    ]
    xs = np.linspace(0.04, 0.96, len(labels))
    for i, (x, lab) in enumerate(zip(xs, labels)):
        ax.text(x, 0.52, lab, ha="center", va="center", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="black"))
        if i < len(labels) - 1:
            ax.annotate("", xy=(xs[i+1]-0.035, 0.52), xytext=(x+0.035, 0.52),
                        arrowprops=dict(arrowstyle="->", lw=1.3))
    ax.set_title("PANDA Prostate Cancer Segmentation Pipeline", fontsize=14, pad=18)
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Training curves
# ---------------------------------------------------------------------------

def make_training_curves(cfg: core.Config):
    run = latest_run_dir(cfg)
    hpath = run / "history.csv"
    if not hpath.exists():
        raise core.QCError("history.csv missing.")
    h = pd.read_csv(hpath)
    out = paper_root(cfg) / "figures" / "02_training_curves.png"
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(h["epoch"], h["train_loss"], label="Training Loss")
    ax1.plot(h["epoch"], h["val_loss"], label="Validation Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax2 = ax1.twinx()
    ax2.plot(h["epoch"], h["val_dice"], marker="o", label="Validation Dice")
    ax2.set_ylabel("Dice")
    best_idx = int(h["val_dice"].idxmax())
    best_epoch = int(h.loc[best_idx, "epoch"])
    best_dice = float(h.loc[best_idx, "val_dice"])
    ax1.axvline(best_epoch, ls="--", lw=1)
    ax1.text(best_epoch, ax1.get_ylim()[1] * 0.92,
             f"Best epoch={best_epoch}\nDice={best_dice:.4f}",
             ha="center", va="top", fontsize=9,
             bbox=dict(boxstyle="round", fc="white", ec="black"))
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], loc="best")
    ax1.set_title("Training Dynamics")
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Patch probability cache and patch-level metrics
# ---------------------------------------------------------------------------

@torch.inference_mode()
def predict_patch_probabilities(cfg: core.Config, frame: pd.DataFrame, model, device,
                                split_name: str, cache_name: str) -> pd.DataFrame:
    root = paper_root(cfg)
    cache = root / "wsi_cache" / f"{cache_name}_patch_predictions.csv"
    if cache.exists():
        old = pd.read_csv(cache)
        if len(old) == len(frame):
            return old
    ds = core.PandaPatchDataset(frame, cfg, split_name, augment=False, normalize=True)
    dl = core.make_loader(ds, cfg, shuffle=False)
    records = []
    offset = 0
    for x, y, levels, coords, ids in dl:
        b = len(x)
        x = x.to(device, non_blocking=True)
        coords = coords.to(device, non_blocking=True)
        if cfg.channels_last and device.type == "cuda":
            x = x.contiguous(memory_format=torch.channels_last)
        with torch.amp.autocast("cuda", enabled=cfg.amp and device.type == "cuda"):
            z = model(x, coords)
            probs = torch.sigmoid(z).detach().cpu().numpy()[:, 0]
        y_np = y.numpy()[:, 0] > 0.5
        src = frame.iloc[offset:offset+b]
        for j in range(b):
            r = src.iloc[j]
            p = probs[j]
            g = y_np[j]
            # Store enough statistics to recompute patch-level threshold metrics cheaply.
            records.append({
                **r.to_dict(),
                "prob_mean": float(p.mean()),
                "prob_max": float(p.max()),
                "gt_positive_pixels": int(g.sum()),
                "patch_pixels": int(g.size),
                # compact per-patch probability and GT arrays are stored separately below
                "_row_index": int(offset + j),
            })
            np.savez_compressed(
                root / "wsi_cache" / f"{cache_name}_patch_{offset+j:08d}.npz",
                prob=p.astype(np.float16),
                gt=g.astype(np.uint8),
            )
        offset += b
    out = pd.DataFrame(records)
    out.to_csv(cache, index=False)
    return out


def patch_metrics_from_cache(cfg: core.Config, pred_df: pd.DataFrame, cache_name: str,
                             threshold: float, subset: Optional[pd.Series] = None) -> Dict[str, Any]:
    if subset is None:
        idxs = list(pred_df.index)
    else:
        idxs = list(pred_df[subset].index)
    tp = fp = tn = fn = 0
    root = paper_root(cfg)
    for i in idxs:
        row_index = int(pred_df.loc[i, "_row_index"])
        z = np.load(root / "wsi_cache" / f"{cache_name}_patch_{row_index:08d}.npz")
        p = z["prob"].astype(np.float32) >= threshold
        g = z["gt"].astype(bool)
        c = counts_from_binary(p, g)
        tp += c[0]; fp += c[1]; tn += c[2]; fn += c[3]
    m = metrics_from_counts(tp, fp, tn, fn)
    m["n_patches"] = len(idxs)
    return m


def make_confusion_matrix_figure(name: str, m: Dict[str, Any], out: Path):
    mat = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]], dtype=np.int64)
    total = max(1, int(mat.sum()))
    fig, ax = plt.subplots(figsize=(4.4, 4))
    im = ax.imshow(mat, cmap="Blues")
    for (i, j), v in np.ndenumerate(mat):
        ax.text(j, i, f"{v:,}\n({100*v/total:.1f}%)", ha="center", va="center", fontsize=11)
    ax.set_xticks([0, 1], ["Non-tumor", "Tumor"])
    ax.set_yticks([0, 1], ["Non-tumor", "Tumor"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"{name}\nDice={m['dice']:.4f} | IoU={m['iou']:.4f}")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# WSI reconstruction on common physical evaluation grid
# ---------------------------------------------------------------------------

def _slide_dims_from_split(cfg: core.Config) -> Dict[str, Tuple[int, int]]:
    d = pd.read_csv(cfg.audit_csv)
    return {
        str(r["image_id"]): (int(r.image_w0), int(r.image_h0))
        for _, r in d.iterrows()
        if bool(r.get("valid_for_segmentation", True))
    }


def _accumulate_patch_to_grid(prob: np.ndarray, gt: np.ndarray,
                              x0: int, y0: int, source_ds: float,
                              eval_ds: float, sum_prob: np.ndarray,
                              sum_gt: np.ndarray, sum_w: np.ndarray):
    field0 = prob.shape[0] * source_ds
    ow = max(1, int(round(field0 / eval_ds)))
    oh = max(1, int(round(field0 / eval_ds)))
    pr = cv2.resize(prob.astype(np.float32), (ow, oh), interpolation=cv2.INTER_LINEAR)
    gr = cv2.resize(gt.astype(np.uint8), (ow, oh), interpolation=cv2.INTER_NEAREST).astype(np.float32)
    ox = int(round(x0 / eval_ds)); oy = int(round(y0 / eval_ds))
    H, W = sum_prob.shape
    x2 = min(W, ox + ow); y2 = min(H, oy + oh)
    if x2 <= ox or y2 <= oy:
        return
    h = y2 - oy; w = x2 - ox
    sum_prob[oy:y2, ox:x2] += pr[:h, :w]
    sum_gt[oy:y2, ox:x2] += gr[:h, :w]
    sum_w[oy:y2, ox:x2] += 1.0


def reconstruct_slide(cfg: core.Config, pred_df: pd.DataFrame, cache_name: str,
                      image_id: str, eval_ds: float = PAPER_EVAL_DOWNSAMPLE) -> Dict[str, Any]:
    root = paper_root(cfg)
    cache = root / "wsi_cache" / f"slide_{image_id}_ds{int(eval_ds)}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        return {
            "gt": z["gt"].astype(bool),
            "valid": z["valid"].astype(bool),
            **{f"prob_L{lv}": z[f"prob_L{lv}"].astype(np.float32) for lv in cfg.levels},
            "fusion": z["fusion"].astype(np.float32),
        }
    dims = _slide_dims_from_split(cfg)
    if image_id not in dims:
        raise core.QCError(f"Missing dimensions for slide {image_id}")
    w0, h0 = dims[image_id]
    W = max(1, int(math.ceil(w0 / eval_ds)))
    H = max(1, int(math.ceil(h0 / eval_ds)))
    level_probs = {}
    gt_ref = None
    valid_ref = None

    sdf = pred_df[pred_df.image_id.astype(str) == str(image_id)].copy()
    for lv in cfg.levels:
        sub = sdf[sdf.level == lv]
        sp = np.zeros((H, W), np.float32)
        sg = np.zeros((H, W), np.float32)
        sw = np.zeros((H, W), np.float32)
        for _, r in sub.iterrows():
            row_index = int(r["_row_index"])
            z = np.load(root / "wsi_cache" / f"{cache_name}_patch_{row_index:08d}.npz")
            prob = z["prob"].astype(np.float32)
            gt = z["gt"].astype(np.uint8)
            _accumulate_patch_to_grid(
                prob, gt, int(r.x0), int(r.y0),
                float(r.target_downsample), eval_ds, sp, sg, sw
            )
        valid = sw > 0
        p = np.zeros_like(sp)
        g = np.zeros_like(sg, dtype=bool)
        p[valid] = sp[valid] / sw[valid]
        g[valid] = (sg[valid] / sw[valid]) >= 0.5
        level_probs[lv] = p
        if gt_ref is None:
            gt_ref = g.copy()
            valid_ref = valid.copy()
        else:
            # use union of observed areas; GT should be physically consistent across levels
            gt_ref = np.logical_or(gt_ref, g)
            valid_ref = np.logical_or(valid_ref, valid)

    stack = np.stack([level_probs[lv] for lv in cfg.levels], axis=0)
    support = np.stack([(level_probs[lv] > 0) | valid_ref for lv in cfg.levels], axis=0)
    fusion = stack.mean(axis=0)

    np.savez_compressed(
        cache,
        gt=gt_ref.astype(np.uint8),
        valid=valid_ref.astype(np.uint8),
        fusion=fusion.astype(np.float16),
        **{f"prob_L{lv}": level_probs[lv].astype(np.float16) for lv in cfg.levels},
    )
    return {
        "gt": gt_ref, "valid": valid_ref,
        **{f"prob_L{lv}": level_probs[lv] for lv in cfg.levels},
        "fusion": fusion,
    }


def reconstruct_split(cfg: core.Config, pred_df: pd.DataFrame, cache_name: str,
                      split_name: str, eval_ds: float = PAPER_EVAL_DOWNSAMPLE):
    ids = sorted(pred_df.image_id.astype(str).unique())
    index = []
    for image_id in ids:
        rec = reconstruct_slide(cfg, pred_df, cache_name, image_id, eval_ds)
        index.append({
            "image_id": image_id,
            "shape_h": rec["gt"].shape[0], "shape_w": rec["gt"].shape[1],
            "valid_pixels": int(rec["valid"].sum()),
            "tumor_pixels_gt": int(np.logical_and(rec["gt"], rec["valid"]).sum()),
        })
    pd.DataFrame(index).to_csv(paper_root(cfg) / "tables" / f"{split_name}_reconstruction_index.csv", index=False)


# ---------------------------------------------------------------------------
# Threshold optimization on validation WSI fusion
# ---------------------------------------------------------------------------

def threshold_sweep_wsi(cfg: core.Config, val_pred_df: pd.DataFrame,
                        eval_ds: float = PAPER_EVAL_DOWNSAMPLE):
    rows = []
    ids = sorted(val_pred_df.image_id.astype(str).unique())
    for thr in THRESHOLDS:
        tp = fp = tn = fn = 0
        for image_id in ids:
            rec = reconstruct_slide(cfg, val_pred_df, "val", image_id, eval_ds)
            pred = rec["fusion"] >= float(thr)
            c = counts_from_binary(pred, rec["gt"], rec["valid"])
            tp += c[0]; fp += c[1]; tn += c[2]; fn += c[3]
        rows.append({"threshold": float(thr), **metrics_from_counts(tp, fp, tn, fn)})
    df = pd.DataFrame(rows)
    out_csv = paper_root(cfg) / "tables" / "threshold_sensitivity_validation.csv"
    df.to_csv(out_csv, index=False)
    best = df.loc[df.dice.idxmax()].to_dict()
    save_json(paper_root(cfg) / "tables" / "best_threshold.json", best)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df.threshold, df.dice, marker="o", ms=3, label="Dice")
    ax.plot(df.threshold, df.precision, marker="s", ms=3, label="Precision")
    ax.plot(df.threshold, df.recall, marker="^", ms=3, label="Recall")
    ax.axvline(best["threshold"], ls="--", lw=1)
    ax.text(best["threshold"], max(df.dice.max(), df.precision.max()) * 0.98,
            f"Optimal={best['threshold']:.2f}\nDice={best['dice']:.4f}",
            ha="center", va="top",
            bbox=dict(boxstyle="round", fc="white", ec="black"))
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Validation Threshold Sensitivity")
    ax.legend()
    fig.tight_layout()
    fig.savefig(paper_root(cfg) / "figures" / "03_threshold_sensitivity.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)
    return float(best["threshold"])


# ---------------------------------------------------------------------------
# WSI metrics per level, fusion, per-slide, provider
# ---------------------------------------------------------------------------

def _metric_row_for_prob(prob, gt, valid, threshold):
    pred = prob >= threshold
    return metrics_from_counts(*counts_from_binary(pred, gt, valid))


def evaluate_wsi_test(cfg: core.Config, test_pred_df: pd.DataFrame,
                      threshold: float, eval_ds: float = PAPER_EVAL_DOWNSAMPLE):
    split = pd.read_csv(cfg.split_csv)
    meta = split.set_index(split.image_id.astype(str))
    per_slide = []
    ids = sorted(test_pred_df.image_id.astype(str).unique())
    for image_id in ids:
        rec = reconstruct_slide(cfg, test_pred_df, "test", image_id, eval_ds)
        provider = str(meta.loc[image_id, "provider"]) if image_id in meta.index else "unknown"
        row = {"image_id": image_id, "provider": provider}
        for lv in cfg.levels:
            m = _metric_row_for_prob(rec[f"prob_L{lv}"], rec["gt"], rec["valid"], threshold)
            for k, v in m.items():
                row[f"L{lv}_{k}"] = v
        fm = _metric_row_for_prob(rec["fusion"], rec["gt"], rec["valid"], threshold)
        for k, v in fm.items():
            row[f"fusion_{k}"] = v
        per_slide.append(row)

    ps = pd.DataFrame(per_slide)
    ps.to_csv(paper_root(cfg) / "tables" / "per_slide_wsi_metrics.csv", index=False)

    # Aggregate confusion counts across slides for each level and fusion.
    global_rows = []
    for key in [f"L{lv}" for lv in cfg.levels] + ["fusion"]:
        tp = int(ps[f"{key}_tp"].sum()); fp = int(ps[f"{key}_fp"].sum())
        tn = int(ps[f"{key}_tn"].sum()); fn = int(ps[f"{key}_fn"].sum())
        global_rows.append({"subset": key, **metrics_from_counts(tp, fp, tn, fn)})
    g = pd.DataFrame(global_rows)
    g.to_csv(paper_root(cfg) / "tables" / "wsi_metrics_per_level_and_fusion.csv", index=False)

    # Per-slide summary.
    s = {
        "n_slides": int(len(ps)),
        "mean_dice": float(ps.fusion_dice.mean()),
        "sd_dice": float(ps.fusion_dice.std(ddof=1)) if len(ps) > 1 else 0.0,
        "min_dice": float(ps.fusion_dice.min()),
        "max_dice": float(ps.fusion_dice.max()),
        "median_dice": float(ps.fusion_dice.median()),
    }
    save_json(paper_root(cfg) / "tables" / "per_slide_summary.json", s)
    return ps, g


def bootstrap_ci_from_slide_counts(ps: pd.DataFrame, key: str, iters: int = BOOTSTRAP_ITERS,
                                   seed: int = 42) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    n = len(ps)
    if n == 0:
        return {}
    vals = {m: [] for m in ["dice", "iou", "precision", "recall", "specificity", "accuracy"]}
    for _ in range(iters):
        idx = rng.integers(0, n, size=n)
        b = ps.iloc[idx]
        tp = int(b[f"{key}_tp"].sum()); fp = int(b[f"{key}_fp"].sum())
        tn = int(b[f"{key}_tn"].sum()); fn = int(b[f"{key}_fn"].sum())
        mm = metrics_from_counts(tp, fp, tn, fn)
        for m in vals:
            vals[m].append(mm[m])
    out = {}
    for m, arr in vals.items():
        out[m] = {
            "low95": float(np.percentile(arr, 2.5)),
            "high95": float(np.percentile(arr, 97.5)),
        }
    return out


def make_bootstrap_tables(cfg: core.Config, ps: pd.DataFrame):
    rows = []
    for key in [f"L{lv}" for lv in cfg.levels] + ["fusion"]:
        ci = bootstrap_ci_from_slide_counts(ps, key, BOOTSTRAP_ITERS, cfg.seed)
        tp = int(ps[f"{key}_tp"].sum()); fp = int(ps[f"{key}_fp"].sum())
        tn = int(ps[f"{key}_tn"].sum()); fn = int(ps[f"{key}_fn"].sum())
        mm = metrics_from_counts(tp, fp, tn, fn)
        for metric in ["dice", "iou", "precision", "recall", "specificity", "accuracy"]:
            rows.append({
                "subset": key, "metric": metric, "value": mm[metric],
                "ci_low": ci[metric]["low95"], "ci_high": ci[metric]["high95"],
                "bootstrap_iterations": BOOTSTRAP_ITERS,
            })
    df = pd.DataFrame(rows)
    df.to_csv(paper_root(cfg) / "tables" / "global_wsi_metrics_95CI.csv", index=False)
    return df


def provider_results(cfg: core.Config, ps: pd.DataFrame):
    rows = []
    for provider, sub in ps.groupby("provider"):
        tp = int(sub.fusion_tp.sum()); fp = int(sub.fusion_fp.sum())
        tn = int(sub.fusion_tn.sum()); fn = int(sub.fusion_fn.sum())
        mm = metrics_from_counts(tp, fp, tn, fn)
        ci = bootstrap_ci_from_slide_counts(sub.reset_index(drop=True), "fusion",
                                            BOOTSTRAP_ITERS, cfg.seed)
        rows.append({
            "provider": provider, "n_slides": len(sub),
            **{k: mm[k] for k in ["dice", "iou", "precision", "recall", "specificity", "accuracy"]},
            "dice_ci_low": ci["dice"]["low95"], "dice_ci_high": ci["dice"]["high95"],
            "iou_ci_low": ci["iou"]["low95"], "iou_ci_high": ci["iou"]["high95"],
            "precision_ci_low": ci["precision"]["low95"], "precision_ci_high": ci["precision"]["high95"],
            "recall_ci_low": ci["recall"]["low95"], "recall_ci_high": ci["recall"]["high95"],
        })
    df = pd.DataFrame(rows)
    df.to_csv(paper_root(cfg) / "tables" / "provider_metrics_95CI.csv", index=False)

    if len(df):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        x = np.arange(len(df))
        y = df.dice.to_numpy()
        lo = y - df.dice_ci_low.to_numpy()
        hi = df.dice_ci_high.to_numpy() - y
        ax.errorbar(x, y, yerr=[lo, hi], fmt="o", capsize=5)
        ax.set_xticks(x, df.provider)
        ax.set_ylabel("Dice")
        ax.set_title("WSI Segmentation Performance by Data Provider")
        ax.set_ylim(max(0, min(y) - 0.1), min(1.0, max(y) + 0.1))
        fig.tight_layout()
        fig.savefig(paper_root(cfg) / "figures" / "06_provider_comparison.png",
                    dpi=300, bbox_inches="tight")
        plt.close(fig)
    return df


# ---------------------------------------------------------------------------
# Per-slide plot
# ---------------------------------------------------------------------------

def make_per_slide_plot(cfg: core.Config, ps: pd.DataFrame):
    s = ps.sort_values("fusion_dice").reset_index(drop=True)
    out = paper_root(cfg) / "figures" / "05_per_slide_dice.png"
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.scatter(np.arange(1, len(s)+1), s.fusion_dice * 100, s=24)
    mean = float(s.fusion_dice.mean() * 100)
    mn = float(s.fusion_dice.min() * 100)
    mx = float(s.fusion_dice.max() * 100)
    ax.axhline(mean, ls="--", lw=1)
    ax.axhline(mn, ls=":", lw=1)
    ax.axhline(mx, ls=":", lw=1)
    ax.text(len(s), mean, f" Mean={mean:.2f}%", va="bottom", ha="right")
    ax.text(len(s), mn, f" Min={mn:.2f}%", va="bottom", ha="right")
    ax.text(len(s), mx, f" Max={mx:.2f}%", va="bottom", ha="right")
    ax.set_xlabel("Test slide (sorted by Dice)")
    ax.set_ylabel("Dice (%)")
    ax.set_title("Per-slide WSI Dice Distribution")
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Patch confusion matrices with optimal threshold
# ---------------------------------------------------------------------------

def make_patch_confusions(cfg: core.Config, test_pred_df: pd.DataFrame, threshold: float):
    rows = []
    for lv in cfg.levels:
        m = patch_metrics_from_cache(cfg, test_pred_df, "test", threshold,
                                     test_pred_df.level == lv)
        name = f"L{lv}"
        rows.append({"subset": name, **m})
        make_confusion_matrix_figure(name, m,
            paper_root(cfg) / "figures" / f"04_confusion_{name}.png")
    pooled = patch_metrics_from_cache(cfg, test_pred_df, "test", threshold)
    rows.append({"subset": "pooled_all_levels", **pooled})
    make_confusion_matrix_figure("Pooled all levels", pooled,
        paper_root(cfg) / "figures" / "04_confusion_pooled_all_levels.png")
    df = pd.DataFrame(rows)
    df.to_csv(paper_root(cfg) / "tables" / "patch_metrics_per_level_and_pooled.csv", index=False)
    return df


# ---------------------------------------------------------------------------
# Qualitative WSI error figures
# ---------------------------------------------------------------------------

def _resize_thumbnail_to_grid(image_path: str, size_hw: Tuple[int, int]) -> np.ndarray:
    sr = core.SlideScaleReader(image_path)
    H, W = size_hw
    thumb = sr.slide.get_thumbnail((W, H)).convert("RGB").resize((W, H))
    arr = np.asarray(thumb)
    sr.close()
    return arr


def _error_rgb(gt: np.ndarray, pred: np.ndarray, valid: np.ndarray) -> np.ndarray:
    H, W = gt.shape
    out = np.zeros((H, W, 3), np.uint8) + 255
    tp = gt & pred & valid
    fn = gt & ~pred & valid
    fp = ~gt & pred & valid
    tn = ~gt & ~pred & valid
    out[tn] = [230, 230, 230]
    out[tp] = [60, 170, 90]
    out[fn] = [240, 190, 40]
    out[fp] = [210, 60, 60]
    return out


def make_qualitative_slide(cfg: core.Config, test_pred_df: pd.DataFrame, image_id: str,
                           threshold: float, tag: str):
    split = pd.read_csv(cfg.split_csv)
    row = split[split.image_id.astype(str) == str(image_id)].iloc[0]
    rec = reconstruct_slide(cfg, test_pred_df, "test", image_id, PAPER_EVAL_DOWNSAMPLE)
    gt = rec["gt"]; valid = rec["valid"]; prob = rec["fusion"]
    pred = prob >= threshold
    rgb = _resize_thumbnail_to_grid(str(row.image_path), gt.shape)

    gt_overlay = rgb.copy()
    gt_overlay[gt & valid] = (0.6 * gt_overlay[gt & valid] + 0.4 * np.array([0, 255, 0])).astype(np.uint8)
    pr_overlay = rgb.copy()
    pr_overlay[pred & valid] = (0.6 * pr_overlay[pred & valid] + 0.4 * np.array([255, 0, 0])).astype(np.uint8)
    err = _error_rgb(gt, pred, valid)

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    data = [
        ("Original H&E", rgb),
        ("Ground Truth", gt.astype(np.uint8) * 255),
        ("Predicted Mask", pred.astype(np.uint8) * 255),
        ("GT Overlay", gt_overlay),
        ("Prediction Overlay", pr_overlay),
        ("Error Map", err),
    ]
    for ax, (title, im) in zip(axes.flat, data):
        ax.imshow(im, cmap="gray" if im.ndim == 2 else None)
        ax.set_title(title); ax.axis("off")
    fig.suptitle(f"{tag.capitalize()} WSI example — {image_id}")
    fig.tight_layout()
    out = paper_root(cfg) / "qualitative" / f"{tag}_{image_id}_six_panel.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)

    cv2.imwrite(str(paper_root(cfg) / "qualitative" / f"{tag}_{image_id}_probability.png"),
                (np.clip(prob, 0, 1) * 255).astype(np.uint8))
    cv2.imwrite(str(paper_root(cfg) / "qualitative" / f"{tag}_{image_id}_mask.png"),
                pred.astype(np.uint8) * 255)
    Image.fromarray(pr_overlay).save(
        paper_root(cfg) / "qualitative" / f"{tag}_{image_id}_overlay.png")
    Image.fromarray(err).save(
        paper_root(cfg) / "qualitative" / f"{tag}_{image_id}_error_map.png")
    return out


def make_qualitative_examples(cfg: core.Config, ps: pd.DataFrame,
                              test_pred_df: pd.DataFrame, threshold: float):
    s = ps.sort_values("fusion_dice").reset_index(drop=True)
    picks = {
        "worst": str(s.iloc[0].image_id),
        "median": str(s.iloc[len(s)//2].image_id),
        "best": str(s.iloc[-1].image_id),
    }
    for tag, image_id in picks.items():
        make_qualitative_slide(cfg, test_pred_df, image_id, threshold, tag)
    save_json(paper_root(cfg) / "qualitative" / "selected_examples.json", picks)
    return picks


# ---------------------------------------------------------------------------
# Multi-level patch example + normalization + augmentation paper figures
# ---------------------------------------------------------------------------

def make_preprocessing_figures(cfg: core.Config):
    manifest = pd.read_csv(cfg.patch_manifest)
    train = manifest[manifest.split == "train"].copy()
    qfig = paper_root(cfg) / "figures"

    # Find a slide with all four levels and tumor examples if possible.
    grouped = train.groupby("image_id").level.nunique()
    candidates = grouped[grouped >= len(cfg.levels)].index.tolist()
    if candidates:
        image_id = str(candidates[0])
        fig, axes = plt.subplots(2, len(cfg.levels), figsize=(4*len(cfg.levels), 7))
        for j, lv in enumerate(cfg.levels):
            sub = train[(train.image_id.astype(str) == image_id) & (train.level == lv)]
            if len(sub) == 0: continue
            r = sub.sort_values("tumor_fraction", ascending=False).iloc[0]
            rgb = np.asarray(Image.open(r.img).convert("RGB"))
            m = np.asarray(Image.open(r["mask"]).convert("L")) > 127
            axes[0, j].imshow(rgb); axes[0, j].set_title(f"L{lv} image")
            axes[1, j].imshow(m, cmap="gray"); axes[1, j].set_title(f"L{lv} tumor mask")
            axes[0, j].axis("off"); axes[1, j].axis("off")
        fig.suptitle("Multilevel Patch Examples")
        fig.tight_layout()
        fig.savefig(qfig / "07_multilevel_patch_examples.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    # Vahadane before / after.
    core.auto_choose_stain_target(cfg, train)
    sample = train.sample(1, random_state=cfg.seed).iloc[0]
    rgb = np.asarray(Image.open(sample.img).convert("RGB"))
    target = np.asarray(Image.open(cfg.stain_target_path).convert("RGB"))
    norm = core.VahadaneProcessor(target)
    after = norm.transform(rgb)
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(rgb); axes[0].set_title("Original")
    axes[1].imshow(after); axes[1].set_title("Vahadane normalized")
    for ax in axes: ax.axis("off")
    fig.tight_layout()
    fig.savefig(qfig / "08_vahadane_before_after.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Augmentation panel.
    mask = Image.open(sample["mask"]).convert("L")
    original = Image.fromarray(rgb)
    aug = core.PairAugStainHeavy(cfg, cfg.seed)
    imgs = [("Original", original)]
    for i in range(5):
        a, _ = aug(original.copy(), mask.copy())
        imgs.append((f"Augmented {i+1}", a))
    fig, axes = plt.subplots(2, 3, figsize=(10, 7))
    for ax, (title, im) in zip(axes.flat, imgs):
        ax.imshow(im); ax.set_title(title); ax.axis("off")
    fig.tight_layout()
    fig.savefig(qfig / "09_augmentation_examples.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Paired bootstrap statistical significance
# ---------------------------------------------------------------------------

def paired_bootstrap_fusion_vs_best_level(cfg: core.Config, ps: pd.DataFrame,
                                          iters: int = BOOTSTRAP_ITERS):
    means = {f"L{lv}": float(ps[f"L{lv}_dice"].mean()) for lv in cfg.levels}
    best = max(means, key=means.get)
    rng = np.random.default_rng(cfg.seed)
    n = len(ps)
    diffs = []
    for _ in range(iters):
        idx = rng.integers(0, n, size=n)
        b = ps.iloc[idx]
        diffs.append(float(b.fusion_dice.mean() - b[f"{best}_dice"].mean()))
    diffs = np.asarray(diffs)
    # two-sided bootstrap sign p-value
    p = float(2 * min((diffs <= 0).mean(), (diffs >= 0).mean()))
    result = {
        "comparison": f"fusion_vs_{best}",
        "best_single_level": best,
        "observed_mean_difference": float(ps.fusion_dice.mean() - ps[f"{best}_dice"].mean()),
        "bootstrap_iterations": iters,
        "difference_95CI": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
        "two_sided_bootstrap_p": min(1.0, p),
    }
    save_json(paper_root(cfg) / "stats" / "paired_bootstrap_fusion_vs_best_level.json", result)
    return result


# ---------------------------------------------------------------------------
# PILOT EXPERIMENTS
# ---------------------------------------------------------------------------

class MacenkoProcessor:
    """TIAToolbox Macenko wrapper that always supplies writable RGB arrays.

    PIL-backed ``np.asarray`` arrays can be read-only. TIAToolbox's ``rgb2od``
    performs an in-place sanitation step, so passing a read-only array raises
    ``ValueError: assignment destination is read-only``.
    """
    def __init__(self, target: np.ndarray):
        if MacenkoNormalizer is None:
            raise RuntimeError("MacenkoNormalizer unavailable in this TIAToolbox installation.")
        self.norm = MacenkoNormalizer()
        target_w = np.array(target, dtype=np.uint8, copy=True, order="C")
        self.norm.fit(target_w)
    def transform(self, rgb: np.ndarray):
        rgb_w = np.array(rgb, dtype=np.uint8, copy=True, order="C")
        out = self.norm.transform(rgb_w)
        return np.array(out, dtype=np.uint8, copy=True)


class ExperimentDataset(core.PandaPatchDataset):
    def __init__(self, frame, cfg, split, augment, normalization="vahadane"):
        super().__init__(frame, cfg, split, augment, normalize=False)
        self.normalization = normalization
        self._mac = None
        self._vah = None
        # Loaded only once per Dataset/worker instead of reopening the target PNG
        # for every single patch.
        self._stain_target = None
    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        # IMPORTANT: use bracket access for manifest columns. pandas.Series.mask is a
        # built-in method, so r.mask returns a FUNCTION rather than the "mask" path.
        # This caused PIL.Image.open(<function Series.mask ...>) in V6.
        # RGB is loaded below through the persistent stain cache for normalized
        # experiments; non-normalized pilots still read the original patch directly.
        img = None if self.normalization != "none" else Image.open(r["img"]).convert("RGB")
        tumor = np.asarray(Image.open(r["mask"]).convert("L")) > 127
        if "valid_mask" not in self.df.columns:
            raise core.QCError("Pilot manifest is missing valid_mask; refusing background-as-normal evaluation.")
        valid = np.asarray(Image.open(r["valid_mask"]).convert("L")) > 127

        ternary = np.full(tumor.shape, 128, dtype=np.uint8)
        ternary[valid & ~tumor] = 0
        ternary[valid & tumor] = 255
        mask = Image.fromarray(ternary, mode="L")

        if self.normalization != "none":
            if self._stain_target is None:
                self._stain_target = np.array(
                    Image.open(self.cfg.stain_target_path).convert("RGB"),
                    dtype=np.uint8, copy=True
                )
            target = self._stain_target
            if self.normalization == "vahadane":
                if self._vah is None: self._vah = core.VahadaneProcessor(target)
                img = core.load_or_build_stain_cache(
                    self.cfg, r["img"], "vahadane", self._vah.transform
                )
            elif self.normalization == "macenko":
                if self._mac is None: self._mac = MacenkoProcessor(target)
                img = core.load_or_build_stain_cache(
                    self.cfg, r["img"], "macenko", self._mac.transform
                )
            else:
                raise ValueError(self.normalization)

        if self.pair_aug is not None:
            img, mask = self.pair_aug(img, mask)

        x = torch.from_numpy(np.asarray(img).copy()).permute(2,0,1).float()/255.0
        ma = np.asarray(mask)
        y_np = np.full(ma.shape, -1.0, dtype=np.float32)
        y_np[ma < 64] = 0.0
        y_np[ma > 192] = 1.0
        y = torch.from_numpy(y_np).unsqueeze(0)
        coords = torch.tensor([float(r["coord_x"]), float(r["coord_y"])], dtype=torch.float32)
        level = torch.tensor(int(r["level"]), dtype=torch.long)
        return x, y, level, coords, str(r["image_id"])


class SMPWrapper(nn.Module):
    def __init__(self, encoder_name: str):
        super().__init__()
        if smp is None:
            raise RuntimeError("segmentation_models_pytorch is not installed.")
        self.net = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
        )
    def forward(self, x, coords=None):
        return self.net(x)


def _balanced_sample(df: pd.DataFrame, total: int, seed: int, use_kmeans_selected: bool = False):
    if len(df) <= total:
        return df.copy()
    if {"level", "category"}.issubset(df.columns):
        groups = list(df.groupby(["level", "category"]))
        per = max(1, total // max(1, len(groups)))
        parts = []
        for _, g in groups:
            parts.append(g.sample(min(per, len(g)), random_state=seed))
        out = pd.concat(parts, ignore_index=True)
        if len(out) < total:
            remain = df.drop(index=out.index, errors="ignore")
            if len(remain):
                out = pd.concat([out, remain.sample(min(total-len(out), len(remain)), random_state=seed)])
        return out.head(total).reset_index(drop=True)
    return df.sample(total, random_state=seed).reset_index(drop=True)


def _pilot_frames(cfg: core.Config):
    manifest = pd.read_csv(cfg.patch_manifest)
    train = manifest[manifest.split == "train"].copy()
    val = manifest[manifest.split == "val"].copy()
    selected = pd.read_csv(cfg.selected_train_csv) if cfg.selected_train_csv.exists() else train
    return train, val, selected


def _fixed_val_sample(val: pd.DataFrame, cfg: core.Config):
    return _balanced_sample(val, PILOT_VAL_PATCHES, cfg.seed + 11)


def run_pilot_training(cfg: core.Config, name: str, train_frame: pd.DataFrame,
                       val_frame: pd.DataFrame, normalization: str = "vahadane",
                       use_coords: bool = True, encoder: str = "efficientnet_b4",
                       epochs: int = PILOT_EPOCHS):
    """
    Pilot training with SSD-disconnect-safe SAME-BATCH recovery.

    Guarantees for a temporary external-SSD disconnect while Python stays alive:
    - already completed TRAIN batches are NOT repeated;
    - TRAIN -> VAL phase boundary is committed before validation starts;
    - validation progress is committed after every completed VAL batch;
    - a failed TRAIN or VAL DataLoader/worker is recreated at the same pending batch;
    - after TRAIN / before VAL and after VAL / before next epoch are durable phase boundaries.

    For a full Python/PC crash, heavy model checkpoints are periodic because saving the
    full model+optimizer after every training batch would create extreme disk writes.
    Set PANDA_PILOT_CHECKPOINT_EVERY=1 for strict per-batch durable checkpoints.
    """
    import hashlib

    out = paper_root(cfg) / "experiments" / name
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        dev = torch.cuda.current_device()
        print(f"[PILOT CUDA] {name}: ENABLED -> {torch.cuda.get_device_name(dev)} | AMP={bool(cfg.amp)}")
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass
    else:
        print(f"[PILOT CUDA] {name}: NOT AVAILABLE -> CPU")

    trds = ExperimentDataset(train_frame, cfg, "train", augment=True, normalization=normalization)
    vads = ExperimentDataset(val_frame, cfg, "val", augment=False, normalization=normalization)
    if normalization != "none" and bool(getattr(cfg, "persistent_stain_cache", True)):
        try:
            paths = pd.concat([train_frame[["img"]], val_frame[["img"]]], ignore_index=True)["img"].astype(str).tolist()
            existing = sum(core.stain_cache_path(cfg, x, normalization).exists() for x in paths)
            print(f"[STAIN CACHE] {name}: {existing}/{len(paths)} already cached | first epoch builds misses once | later epochs reuse lossless PNGs")
        except Exception as e:
            print(f"[STAIN CACHE] {name}: coverage check skipped: {e!r}")

    # FAST pilot I/O defaults for the user's 8C/12T-class laptop + RTX GPU.
    # Scientific settings (batch size/model/loss/split/normalization) are untouched.
    # Both TRAIN and VAL use workers for speed; if D: disappears, the parent process
    # catches the worker failure, waits for reconnect, and recreates the loader at the
    # same pending batch.
    auto_workers = max(1, min(6, max(1, (os.cpu_count() or 4) - 2)))
    pilot_workers = max(0, int(os.environ.get("PANDA_PILOT_WORKERS", str(auto_workers))))
    pilot_val_workers = max(0, int(os.environ.get("PANDA_PILOT_VAL_WORKERS", str(pilot_workers))))
    pilot_prefetch = max(1, int(os.environ.get(
        "PANDA_PILOT_PREFETCH", str(getattr(cfg, "prefetch_factor", 4))
    )))
    pilot_persistent = bool(getattr(cfg, "persistent_workers", True)) and pilot_workers > 0
    print(
        f"[PILOT DATALOADER] {name}: TRAIN workers={pilot_workers} | "
        f"VAL workers={pilot_val_workers} | prefetch={pilot_prefetch if pilot_workers > 0 else 0} | "
        f"pin_memory={device.type == 'cuda'} | AMP={bool(cfg.amp and device.type == 'cuda')} | "
        f"channels_last={device.type == 'cuda'} | cudnn_benchmark={bool(torch.backends.cudnn.benchmark)}"
    )

    if encoder == "efficientnet_b4_custom":
        model = core.UNetEfficientNetB4(
            1, cfg.dropout_p, use_coords, cfg.pretrained, cfg.coord_embed_dim
        ).to(device)
    else:
        model = SMPWrapper(encoder).to(device)

    try:
        if device.type == "cuda":
            model = model.to(memory_format=torch.channels_last)
    except Exception:
        pass

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    lossfn = core.CombinedLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and device.type == "cuda")

    resume_path = out / "pilot_resume.pt"
    train_phase_path = out / "pilot_train_phase_complete.pt"
    val_progress_path = out / "pilot_val_progress.pt"
    history_path = out / "history.csv"
    best_path = out / "best.pt"

    # Mirror the active resume state to the INTERNAL Windows disk. This remains
    # readable even while the external D: SSD is disconnected.
    local_root = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "PANDA_PILOT_RESUME" / name
    local_root.mkdir(parents=True, exist_ok=True)
    local_resume = local_root / "pilot_resume.pt"
    local_train_phase = local_root / "pilot_train_phase_complete.pt"
    local_val_progress = local_root / "pilot_val_progress.pt"

    try:
        anchor = Path(str(train_frame.iloc[0]["img"])).anchor or Path(str(out)).anchor
    except Exception:
        anchor = Path(str(out)).anchor
    storage_root = Path(anchor) if anchor else Path(str(out).split(os.sep)[0] + os.sep)

    def storage_ready():
        try:
            if not storage_root.exists():
                return False
            next(iter(storage_root.iterdir()), None)
            return True
        except Exception:
            return False

    def looks_storage_error(e):
        s = f"{type(e).__name__}: {e}".lower()
        needles = [
            "errno 22", "invalid argument", "device is not ready", "cannot find the path",
            "system cannot find", "input/output", "i/o error", "bad file descriptor",
            "no such file", "worker exited unexpectedly", "dataloader worker",
            "broken pipe", "winerror 3", "winerror 21", "winerror 53", "winerror 64",
            "unsupported or missing image file", "unexpected pos", "inline_container.cc",
        ]
        return (not storage_ready()) or any(x in s for x in needles)

    def wait_storage(reason):
        print(f"[PILOT SSD WAIT] {name}: {reason} | waiting for {storage_root}", flush=True)
        while not storage_ready():
            time.sleep(10)
        # Give Windows/OpenSlide/USB bridge a little time after remount.
        time.sleep(2)
        print(f"[PILOT SSD RECONNECTED] {name}: continuing SAME pending batch/phase", flush=True)

    def atomic_torch(path: Path, obj):
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "wb") as f:
            torch.save(obj, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)

    def atomic_csv(path: Path, frame: pd.DataFrame):
        tmp = path.with_name(path.name + ".tmp")
        frame.to_csv(tmp, index=False)
        os.replace(tmp, path)

    def save_heavy(state, mirror_external=True):
        # Internal-disk state first, so a D: dropout cannot destroy the newest checkpoint.
        atomic_torch(local_resume, state)
        if mirror_external:
            while True:
                try:
                    atomic_torch(resume_path, state)
                    break
                except Exception as e:
                    if not looks_storage_error(e):
                        raise
                    wait_storage(f"saving resume checkpoint: {e}")

    def save_train_phase(state):
        atomic_torch(local_train_phase, state)
        while True:
            try:
                atomic_torch(train_phase_path, state)
                break
            except Exception as e:
                if not looks_storage_error(e):
                    raise
                wait_storage(f"saving TRAIN->VAL boundary: {e}")

    def save_val_progress(state):
        # Lightweight; safe to save after EVERY validation batch.
        atomic_torch(local_val_progress, state)
        while True:
            try:
                atomic_torch(val_progress_path, state)
                break
            except Exception as e:
                if not looks_storage_error(e):
                    raise
                wait_storage(f"saving validation progress: {e}")

    def newest_existing(paths):
        good=[]
        for p in paths:
            try:
                if p.exists(): good.append((p.stat().st_mtime, p))
            except Exception:
                pass
        return max(good, default=(None,None))[1]

    def make_plan(n_items, ep, shuffle):
        idx = list(range(n_items))
        if shuffle:
            digest = int(hashlib.sha1(name.encode("utf-8")).hexdigest()[:8], 16)
            g = torch.Generator()
            g.manual_seed(int(cfg.seed) + ep * 1000003 + digest)
            idx = torch.randperm(n_items, generator=g).tolist()
        bs = int(PILOT_BATCH_SIZE)
        return [idx[i:i+bs] for i in range(0, len(idx), bs)]

    def make_loader(ds, batches, workers):
        kw = dict(
            batch_sampler=batches,
            num_workers=workers,
            pin_memory=(device.type == "cuda"),
            worker_init_fn=core.fast_worker_init_fn,
        )
        if workers > 0:
            kw.update(
                persistent_workers=False,
                prefetch_factor=pilot_prefetch,
            )
        return DataLoader(ds, **kw)

    def meter_from_counts(d):
        m = core.PixelMeter()
        if d:
            m.tp=int(d.get("tp",0)); m.fp=int(d.get("fp",0)); m.tn=int(d.get("tn",0)); m.fn=int(d.get("fn",0))
        return m

    def meter_counts(m):
        return {"tp":int(m.tp),"fp":int(m.fp),"tn":int(m.tn),"fn":int(m.fn)}

    best = -1.0
    bad = 0
    hist = []
    epoch = 1
    phase = "train"
    next_batch = 0
    tl = 0.0
    n = 0

    # Prefer newest local/internal checkpoint, but remain backward compatible with V25's
    # old epoch-only pilot_resume.pt on D:.
    rp = newest_existing([local_resume, resume_path])
    if rp is not None:
        try:
            ck = torch.load(rp, map_location=device)
            model.load_state_dict(ck["model"])
            opt.load_state_dict(ck["optimizer"])
            if ck.get("scaler") is not None and scaler.is_enabled():
                scaler.load_state_dict(ck["scaler"])
            best = float(ck.get("best", -1.0))
            bad = int(ck.get("bad", 0))
            hist = list(ck.get("history", []))
            if "epoch" in ck:
                epoch = int(ck.get("epoch",1))
                phase = str(ck.get("phase","train"))
                next_batch = int(ck.get("next_batch",0))
                tl = float(ck.get("train_loss_sum",0.0))
                n = int(ck.get("train_n",0))
                print(f"[PILOT EXACT RESUME] {name}: epoch={epoch}/{epochs} phase={phase} next_batch={next_batch}")
            else:
                # V25 migration: only a completed-epoch checkpoint exists.
                epoch = int(ck.get("completed_epoch",0)) + 1
                phase = "train"; next_batch=0; tl=0.0; n=0
                print(f"[PILOT V25 MIGRATION] {name}: continuing at epoch {epoch}/{epochs}; old checkpoint had no batch-level state")
        except Exception as e:
            print(f"[PILOT RESUME WARNING] Could not load {rp}: {e!r}; trying external checkpoint if different.")

    checkpoint_every = max(1, int(os.environ.get("PANDA_PILOT_CHECKPOINT_EVERY", "25")))
    print(f"[PILOT RESUME MODE] {name}: SSD disconnect = same-batch in-memory recovery | full-crash durable TRAIN checkpoint every {checkpoint_every} batch(es)")
    if checkpoint_every == 1:
        print(f"[PILOT RESUME MODE] {name}: STRICT durable per-batch checkpointing enabled; expect slower training and heavy internal-disk writes.")

    epoch_bar = tqdm(total=max(0, epochs-epoch+1), desc=f"PILOT {name}", unit="epoch", dynamic_ncols=True)

    while epoch <= epochs:
        # ---------------- TRAIN ----------------
        if phase == "train":
            model.train()
            plan = make_plan(len(trds), epoch, True)
            processed = min(next_batch, len(plan))

            while processed < len(plan):
                dl = make_loader(trds, plan[processed:], pilot_workers)
                try:
                    train_bar = tqdm(
                        dl, total=len(plan)-processed, leave=False, dynamic_ncols=True,
                        desc=f"{name} | epoch {epoch}/{epochs} TRAIN resume@{processed}", unit="batch"
                    )
                    for x, y, lv, coords, ids in train_bar:
                        x = x.to(device, non_blocking=True)
                        y = y.to(device, non_blocking=True)
                        coords = coords.to(device, non_blocking=True)
                        if device.type == "cuda":
                            try: x = x.contiguous(memory_format=torch.channels_last)
                            except Exception: pass
                        opt.zero_grad(set_to_none=True)
                        with torch.amp.autocast("cuda", enabled=cfg.amp and device.type == "cuda"):
                            z = model(x, coords if use_coords else None)
                            loss = lossfn(z, y)
                        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
                        tl += float(loss.item()) * len(x)
                        n += len(x)
                        processed += 1
                        train_bar.set_postfix(loss=f"{float(loss.item()):.4f}", refresh=False)

                        if processed % checkpoint_every == 0:
                            state = {
                                "epoch":epoch,"phase":"train","next_batch":processed,
                                "model":model.state_dict(),"optimizer":opt.state_dict(),
                                "scaler":scaler.state_dict() if scaler.is_enabled() else None,
                                "best":best,"bad":bad,"history":hist,"experiment":name,
                                "train_loss_sum":tl,"train_n":n,
                            }
                            save_heavy(state, mirror_external=False)
                except Exception as e:
                    if not looks_storage_error(e):
                        raise
                    # IMPORTANT: model/optimizer stay in RAM. No completed batch is replayed.
                    print(f"[PILOT TRAIN I/O INTERRUPTION] {name} epoch={epoch} completed_batch={processed}: {e}", flush=True)
                    wait_storage(f"TRAIN epoch {epoch}")
                    next_batch = processed
                    continue

            # Commit exact TRAIN->VAL boundary BEFORE creating/starting VAL.
            train_loss = tl / max(1, n)
            phase_state = {
                "epoch":epoch,"phase":"val","next_batch":0,
                "model":model.state_dict(),"optimizer":opt.state_dict(),
                "scaler":scaler.state_dict() if scaler.is_enabled() else None,
                "best":best,"bad":bad,"history":hist,"experiment":name,
                "train_loss_sum":tl,"train_n":n,"train_loss":train_loss,
            }
            save_train_phase(phase_state)
            save_heavy(phase_state, mirror_external=True)
            phase="val"; next_batch=0

        # ---------------- VAL ----------------
        if phase == "val":
            # Reload exact train-boundary state only on process restart; in normal flow the
            # same in-RAM model is already there. If a boundary checkpoint exists, it is authoritative.
            tp = newest_existing([local_train_phase, train_phase_path])
            if tp is not None:
                phase_ck = torch.load(tp, map_location=device)
                if int(phase_ck.get("epoch", epoch)) == epoch:
                    model.load_state_dict(phase_ck["model"])
                    opt.load_state_dict(phase_ck["optimizer"])
                    if phase_ck.get("scaler") is not None and scaler.is_enabled():
                        try: scaler.load_state_dict(phase_ck["scaler"])
                        except Exception: pass
                    tl=float(phase_ck.get("train_loss_sum",tl)); n=int(phase_ck.get("train_n",n))

            vmeter = core.PixelMeter()
            val_slide_counts = {}
            vnext = 0
            vp = newest_existing([local_val_progress, val_progress_path])
            if vp is not None:
                try:
                    vs = torch.load(vp, map_location="cpu")
                    if int(vs.get("epoch",-1)) == epoch:
                        vnext=int(vs.get("next_batch",0))
                        vmeter=meter_from_counts(vs.get("meter"))
                        val_slide_counts=dict(vs.get("val_slide_counts",{}))
                        print(f"[PILOT VAL RESUME] {name}: epoch={epoch}/{epochs} next_batch={vnext}")
                except Exception as e:
                    print(f"[PILOT VAL RESUME WARNING] {name}: {e!r}")

            model.eval()
            val_plan = make_plan(len(vads), epoch, False)
            processed = min(vnext, len(val_plan))
            while processed < len(val_plan):
                # Fast VAL workers. If the SSD disappears while Windows workers are
                # starting/reading, the exception is caught below and the loader is
                # recreated at exactly `processed` after reconnect.
                vadl = make_loader(vads, val_plan[processed:], pilot_val_workers)
                try:
                    val_bar = tqdm(
                        vadl, total=len(val_plan)-processed, leave=False, dynamic_ncols=True,
                        desc=f"{name} | epoch {epoch}/{epochs} VAL resume@{processed}", unit="batch"
                    )
                    with torch.inference_mode():
                        for x, y, lv, coords, ids in val_bar:
                            x=x.to(device,non_blocking=True); y=y.to(device,non_blocking=True); coords=coords.to(device,non_blocking=True)
                            if device.type == "cuda":
                                try: x=x.contiguous(memory_format=torch.channels_last)
                                except Exception: pass
                            with torch.amp.autocast("cuda", enabled=cfg.amp and device.type == "cuda"):
                                z=model(x, coords if use_coords else None)
                            vmeter.update(z,y,0.5)

                            pred=(torch.sigmoid(z)>=0.5)
                            valid_px=y>=0
                            gt_px=y>0.5
                            for bi,image_id in enumerate(ids):
                                iid=str(image_id)
                                vv=valid_px[bi]; pp=pred[bi][vv]; gg=gt_px[bi][vv]
                                c=val_slide_counts.setdefault(iid,[0,0,0,0])
                                c[0]+=int((pp & gg).sum().item()); c[1]+=int((pp & ~gg).sum().item())
                                c[2]+=int((~pp & ~gg).sum().item()); c[3]+=int((~pp & gg).sum().item())

                            processed += 1
                            save_val_progress({
                                "epoch":epoch,"next_batch":processed,"meter":meter_counts(vmeter),
                                "val_slide_counts":val_slide_counts,
                            })
                except Exception as e:
                    if not looks_storage_error(e):
                        raise
                    print(f"[PILOT VAL I/O INTERRUPTION] {name} epoch={epoch} completed_batch={processed}: {e}", flush=True)
                    wait_storage(f"VAL epoch {epoch}")
                    # processed/metrics are already saved for every completed batch.
                    continue

            vm=vmeter.metrics()
            row={"epoch":epoch,"train_loss":tl/max(1,n),**{f"val_{k}":v for k,v in vm.items()}}
            hist=[h for h in hist if int(h.get("epoch",-1)) != epoch]
            hist.append(row)

            if vm["dice"] > best:
                best=vm["dice"]; bad=0
                while True:
                    try:
                        atomic_torch(best_path,{"model":model.state_dict(),"val":vm,"epoch":epoch})
                        best_slide_rows=[]
                        for iid,c in sorted(val_slide_counts.items()):
                            tp_,fp_,tn_,fn_=c
                            div=lambda a,b: float(a)/float(b) if b else 0.0
                            best_slide_rows.append({
                                "image_id":iid,"tp":tp_,"fp":fp_,"tn":tn_,"fn":fn_,
                                "dice":div(2*tp_,2*tp_+fp_+fn_),"iou":div(tp_,tp_+fp_+fn_),
                                "precision":div(tp_,tp_+fp_),"recall":div(tp_,tp_+fn_),
                                "specificity":div(tn_,tn_+fp_),"accuracy":div(tp_+tn_,tp_+fp_+tn_+fn_),
                                "best_epoch":epoch,
                            })
                        atomic_csv(out/"best_val_per_slide_metrics.csv",pd.DataFrame(best_slide_rows))
                        break
                    except Exception as e:
                        if not looks_storage_error(e): raise
                        wait_storage(f"saving best epoch {epoch}: {e}")
            else:
                bad += 1

            while True:
                try:
                    atomic_csv(history_path,pd.DataFrame(hist))
                    break
                except Exception as e:
                    if not looks_storage_error(e): raise
                    wait_storage(f"saving history epoch {epoch}: {e}")

            print(f"[PILOT PROGRESS] {name} | epoch {epoch}/{epochs} | train_loss={row['train_loss']:.4f} | val_dice={float(vm['dice']):.4f} | best={float(best):.4f}")
            epoch_bar.update(1)
            epoch_bar.set_postfix(train_loss=f"{row['train_loss']:.4f}",val_dice=f"{float(vm['dice']):.4f}",best=f"{float(best):.4f}")

            # Commit NEXT epoch only after this whole epoch is complete.
            next_state={
                "epoch":epoch+1,"phase":"train","next_batch":0,
                "model":model.state_dict(),"optimizer":opt.state_dict(),
                "scaler":scaler.state_dict() if scaler.is_enabled() else None,
                "best":best,"bad":bad,"history":hist,"experiment":name,
                "train_loss_sum":0.0,"train_n":0,
            }
            save_heavy(next_state, mirror_external=True)
            for p in [local_val_progress,val_progress_path,local_train_phase,train_phase_path]:
                try: p.unlink(missing_ok=True)
                except Exception: pass

            if bad >= PILOT_PATIENCE:
                print(f"[PILOT EARLY STOP] {name}: patience={PILOT_PATIENCE}")
                epoch += 1
                phase="train"; next_batch=0; tl=0.0; n=0
                break

            epoch += 1
            phase="train"; next_batch=0; tl=0.0; n=0

    epoch_bar.close()

    # Export raw slide-level validation points from the BEST validation epoch.
    best_slide_csv=out/"best_val_per_slide_metrics.csv"
    if best_slide_csv.exists():
        shutil.copy2(best_slide_csv,out/"val_per_slide_metrics.csv")

    result={
        "experiment":name,"n_train":len(train_frame),"n_val":len(val_frame),
        "normalization":normalization,"use_coords":use_coords,"encoder":encoder,
        "best_val_dice":best,
        "best_val_iou":max([r.get("val_iou",0) for r in hist],default=0),
        "best_val_precision":max([r.get("val_precision",0) for r in hist],default=0),
        "best_val_recall":max([r.get("val_recall",0) for r in hist],default=0),
        "device":str(device),
        "cuda_name":torch.cuda.get_device_name(torch.cuda.current_device()) if device.type=="cuda" else None,
    }

    summary_tmp=out/"summary.json.tmp"
    while True:
        try:
            with open(summary_tmp,"w",encoding="utf-8") as f:
                json.dump(result,f,indent=2,ensure_ascii=False,default=str); f.flush(); os.fsync(f.fileno())
            os.replace(summary_tmp,out/"summary.json")
            break
        except Exception as e:
            if not looks_storage_error(e): raise
            wait_storage(f"saving experiment summary: {e}")

    # Completion cleanup: only after summary.json exists.
    for p in [resume_path,train_phase_path,val_progress_path,local_resume,local_train_phase,local_val_progress]:
        try: p.unlink(missing_ok=True)
        except Exception: pass
    try:
        if local_root.exists() and not any(local_root.iterdir()): local_root.rmdir()
    except Exception:
        pass

    del model
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return result

def pilot_ablation(cfg: core.Config):
    train, val, selected = _pilot_frames(cfg)
    val_fixed = _fixed_val_sample(val, cfg)
    l2 = train[train.level == 2].copy()
    # A0 baseline: single level, random, no norm, no coords
    experiments = []
    experiments.append(("A0_baseline_L2_random_no_norm_no_coords",
                        l2.sample(min(PILOT_TRAIN_PATCHES, len(l2)), random_state=cfg.seed),
                        "none", False))
    experiments.append(("A1_plus_vahadane",
                        l2.sample(min(PILOT_TRAIN_PATCHES, len(l2)), random_state=cfg.seed),
                        "vahadane", False))
    experiments.append(("A2_plus_multilevel",
                        train.sample(min(PILOT_TRAIN_PATCHES, len(train)), random_state=cfg.seed),
                        "vahadane", False))
    experiments.append(("A3_plus_class_balance",
                        _balanced_sample(train, PILOT_TRAIN_PATCHES, cfg.seed),
                        "vahadane", False))
    experiments.append(("A4_plus_kmeans_selection",
                        _balanced_sample(selected, PILOT_TRAIN_PATCHES, cfg.seed),
                        "vahadane", False))
    experiments.append(("A5_plus_coordinate_encoding_full",
                        _balanced_sample(selected, PILOT_TRAIN_PATCHES, cfg.seed),
                        "vahadane", True))
    rows=[]
    for name, tr, norm, coords in experiments:
        rows.append(run_pilot_training(cfg, name, tr, val_fixed, norm, coords,
                                       encoder="efficientnet_b4_custom"))
    df=pd.DataFrame(rows)
    df.to_csv(paper_root(cfg)/"tables"/"pilot_ablation_results.csv", index=False)
    return df


def pilot_normalization_comparison(cfg: core.Config):
    train, val, selected = _pilot_frames(cfg)
    tr = _balanced_sample(selected, PILOT_TRAIN_PATCHES, cfg.seed)
    va = _fixed_val_sample(val, cfg)
    rows=[]
    for norm in ["none", "macenko", "vahadane"]:
        if norm=="macenko" and MacenkoNormalizer is None:
            rows.append({"experiment":"normalization_macenko","skipped":"Macenko unavailable"})
            continue
        rows.append(run_pilot_training(cfg, f"normalization_{norm}", tr, va, norm, True,
                                       encoder="efficientnet_b4_custom"))
    df=pd.DataFrame(rows)
    df.to_csv(paper_root(cfg)/"tables"/"pilot_normalization_comparison.csv", index=False)
    return df


def pilot_kmeans_and_coords(cfg: core.Config):
    train, val, selected = _pilot_frames(cfg)
    va = _fixed_val_sample(val, cfg)
    balanced_random = _balanced_sample(train, PILOT_TRAIN_PATCHES, cfg.seed)
    kmeans = _balanced_sample(selected, PILOT_TRAIN_PATCHES, cfg.seed)
    rows = [
        run_pilot_training(cfg, "kmeans_off_balanced_random", balanced_random, va,
                           "vahadane", True, "efficientnet_b4_custom"),
        run_pilot_training(cfg, "kmeans_on", kmeans, va,
                           "vahadane", True, "efficientnet_b4_custom"),
        run_pilot_training(cfg, "coords_off", kmeans, va,
                           "vahadane", False, "efficientnet_b4_custom"),
        run_pilot_training(cfg, "coords_on", kmeans, va,
                           "vahadane", True, "efficientnet_b4_custom"),
    ]
    df=pd.DataFrame(rows)
    df.to_csv(paper_root(cfg)/"tables"/"pilot_kmeans_coordinate_ablation.csv", index=False)
    return df


def pilot_encoder_comparison(cfg: core.Config):
    if smp is None:
        save_json(paper_root(cfg)/"experiments"/"encoder_comparison_skipped.json",
                  {"reason":"segmentation_models_pytorch not installed"})
        return pd.DataFrame()
    train, val, selected = _pilot_frames(cfg)
    tr=_balanced_sample(selected, PILOT_TRAIN_PATCHES, cfg.seed)
    va=_fixed_val_sample(val, cfg)
    encoders = [
        ("resnet50","resnet50"),
        ("resnet101","resnet101"),
        ("efficientnet_b0","efficientnet-b0"),
        ("efficientnet_b2","efficientnet-b2"),
        ("efficientnet_b4","efficientnet-b4"),
    ]
    rows=[]
    for name, enc in encoders:
        try:
            rows.append(run_pilot_training(cfg, f"encoder_{name}", tr, va,
                                           "vahadane", False, enc))
        except Exception as e:
            rows.append({"experiment":f"encoder_{name}","error":repr(e)})
    df=pd.DataFrame(rows)
    df.to_csv(paper_root(cfg)/"tables"/"pilot_encoder_comparison.csv", index=False)
    return df


def pilot_leave_one_provider_out(cfg: core.Config):
    train, val, selected = _pilot_frames(cfg)
    providers = sorted(train.provider.dropna().astype(str).unique())
    rows=[]
    if len(providers) < 2:
        save_json(paper_root(cfg)/"experiments"/"provider_holdout_skipped.json",
                  {"reason":"Need at least two providers"})
        return pd.DataFrame()
    for held in providers:
        tr = train[train.provider.astype(str) != held]
        te = train[train.provider.astype(str) == held]
        tr = _balanced_sample(tr, PILOT_TRAIN_PATCHES, cfg.seed)
        te = _balanced_sample(te, PILOT_VAL_PATCHES, cfg.seed+3)
        rows.append(run_pilot_training(cfg, f"provider_holdout_{held}", tr, te,
                                       "vahadane", True, "efficientnet_b4_custom"))
        rows[-1]["held_out_provider"]=held
    df=pd.DataFrame(rows)
    df.to_csv(paper_root(cfg)/"tables"/"pilot_leave_one_provider_out.csv", index=False)
    return df



def _holm_adjust_local(pvals):
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    if n == 0:
        return np.asarray([], dtype=float)
    order = np.argsort(p)
    adj = np.empty(n, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        v = min(1.0, (n-rank)*p[idx])
        running = max(running, v)
        adj[idx] = running
    return adj


def _pilot_paired_bootstrap(cfg: core.Config, exp_a: str, exp_b: str,
                            metric: str = "dice", iters: int = BOOTSTRAP_ITERS):
    pa = paper_root(cfg)/"experiments"/exp_a/"val_per_slide_metrics.csv"
    pb = paper_root(cfg)/"experiments"/exp_b/"val_per_slide_metrics.csv"
    if not pa.exists() or not pb.exists():
        return {"comparison": f"{exp_a}_vs_{exp_b}", "status": "not_available",
                "reason": "Per-slide validation metrics missing (experiment may predate V16 or be incomplete)."}
    a = pd.read_csv(pa)[["image_id", metric]].rename(columns={metric:"a"})
    b = pd.read_csv(pb)[["image_id", metric]].rename(columns={metric:"b"})
    m = a.merge(b, on="image_id", how="inner")
    if len(m) < 2:
        return {"comparison": f"{exp_a}_vs_{exp_b}", "status": "not_available",
                "reason": "Fewer than 2 common validation slides."}
    d = m.a.to_numpy(float) - m.b.to_numpy(float)
    rng = np.random.default_rng(int(cfg.seed) + abs(hash(exp_a + exp_b)) % 10000)
    boot = np.empty(iters, dtype=np.float64)
    for i in range(iters):
        idx = rng.integers(0, len(d), size=len(d))
        boot[i] = float(d[idx].mean())
    p = float(2 * min((boot <= 0).mean(), (boot >= 0).mean()))
    sd = float(d.std(ddof=1))
    return {
        "comparison": f"{exp_a}_vs_{exp_b}",
        "experiment_a": exp_a, "experiment_b": exp_b,
        "metric": metric, "n_paired_slides": int(len(m)),
        "mean_a": float(m.a.mean()), "mean_b": float(m.b.mean()),
        "mean_difference": float(d.mean()),
        "difference_95CI_low": float(np.percentile(boot,2.5)),
        "difference_95CI_high": float(np.percentile(boot,97.5)),
        "two_sided_bootstrap_p": min(1.0,p),
        "cohen_dz": float(d.mean()/sd) if sd > 0 else 0.0,
        "probability_superiority": float(((d>0).sum()+0.5*(d==0).sum())/len(d)),
        "bootstrap_iterations": int(iters),
        "status": "ok",
    }


def pilot_major_comparison_statistics(cfg: core.Config):
    """
    Pre-specified major ablation comparisons. Holm correction is applied across
    all comparisons that have paired slide-level validation points.
    """
    comparisons = [
        ("A1_plus_vahadane", "A0_baseline_L2_random_no_norm_no_coords"),
        ("A2_plus_multilevel", "A1_plus_vahadane"),
        ("A3_plus_class_balance", "A2_plus_multilevel"),
        ("A4_plus_kmeans_selection", "A3_plus_class_balance"),
        ("A5_plus_coordinate_encoding_full", "A4_plus_kmeans_selection"),
        ("normalization_macenko", "normalization_none"),
        ("normalization_vahadane", "normalization_none"),
        ("kmeans_on", "kmeans_off_balanced_random"),
        ("coords_on", "coords_off"),
        ("encoder_resnet50", "encoder_efficientnet_b4"),
        ("encoder_resnet101", "encoder_efficientnet_b4"),
        ("encoder_efficientnet_b0", "encoder_efficientnet_b4"),
        ("encoder_efficientnet_b2", "encoder_efficientnet_b4"),
    ]
    rows = [_pilot_paired_bootstrap(cfg,a,b) for a,b in comparisons]
    df = pd.DataFrame(rows)
    ok = (df.get("status","") == "ok") if len(df) else pd.Series([], dtype=bool)
    if len(df) and ok.any():
        adj = _holm_adjust_local(df.loc[ok,"two_sided_bootstrap_p"].to_numpy())
        df.loc[ok,"p_holm"] = adj
        df.loc[ok,"significant_holm_0.05"] = adj < 0.05
    out = paper_root(cfg)/"stats"/"pilot_major_comparisons_two_sided_holm.csv"
    df.to_csv(out, index=False)
    return df


def write_not_applicable_notes(cfg: core.Config):
    note = {
        "CAMELYON_specific_FROC_subtypes": "NOT REPORTED",
        "reason": (
            "CAMELYON's ITC/micrometastasis/macrometastasis size categories are "
            "lymph-node metastasis definitions and should not be transplanted into "
            "PANDA prostate-biopsy segmentation. The prostate paper reports pixel/WSI "
            "segmentation overlap, per-provider robustness, threshold sensitivity, "
            "and qualitative error maps instead."
        ),
    }
    save_json(paper_root(cfg)/"tables"/"not_applicable_camelyon_specific_analyses.json", note)


# ---------------------------------------------------------------------------
# Final summary and one-call paper run
# ---------------------------------------------------------------------------

def paper_all(cfg: core.Config):
    root = paper_root(cfg)
    make_pipeline_overview(cfg)
    make_training_curves(cfg)
    make_preprocessing_figures(cfg)

    model, device, ckpt = core.load_best_model(cfg, None)
    manifest = pd.read_csv(cfg.patch_manifest)
    val = manifest[manifest.split == "val"].reset_index(drop=True)
    test = manifest[manifest.split == "test"].reset_index(drop=True)

    val_pred = predict_patch_probabilities(cfg, val, model, device, "val", "val")
    test_pred = predict_patch_probabilities(cfg, test, model, device, "test", "test")

    reconstruct_split(cfg, val_pred, "val", "val")
    threshold = threshold_sweep_wsi(cfg, val_pred)
    reconstruct_split(cfg, test_pred, "test", "test")

    patch_df = make_patch_confusions(cfg, test_pred, threshold)
    ps, global_df = evaluate_wsi_test(cfg, test_pred, threshold)
    ci_df = make_bootstrap_tables(cfg, ps)
    provider_df = provider_results(cfg, ps)
    make_per_slide_plot(cfg, ps)
    examples = make_qualitative_examples(cfg, ps, test_pred, threshold)
    stat = paired_bootstrap_fusion_vs_best_level(cfg, ps)
    write_not_applicable_notes(cfg)

    summary = {
        "checkpoint": ckpt,
        "optimal_validation_threshold": threshold,
        "paper_eval_downsample": PAPER_EVAL_DOWNSAMPLE,
        "n_test_slides": len(ps),
        "patch_metrics_csv": str(root/"tables"/"patch_metrics_per_level_and_pooled.csv"),
        "wsi_metrics_csv": str(root/"tables"/"wsi_metrics_per_level_and_fusion.csv"),
        "per_slide_csv": str(root/"tables"/"per_slide_wsi_metrics.csv"),
        "provider_csv": str(root/"tables"/"provider_metrics_95CI.csv"),
        "ci_csv": str(root/"tables"/"global_wsi_metrics_95CI.csv"),
        "qualitative_examples": examples,
        "fusion_vs_best_level_test": stat,
    }
    save_json(root/"paper_results_summary.json", summary)
    print(json.dumps(summary, indent=2))
    return summary


def pilot_experiments(cfg: core.Config, include_encoder: bool = True,
                      include_provider_holdout: bool = True):
    root=paper_root(cfg)
    results={}
    results["ablation_rows"]=len(pilot_ablation(cfg))
    results["normalization_rows"]=len(pilot_normalization_comparison(cfg))
    results["kmeans_coord_rows"]=len(pilot_kmeans_and_coords(cfg))
    if include_encoder:
        results["encoder_rows"]=len(pilot_encoder_comparison(cfg))
    if include_provider_holdout:
        results["provider_holdout_rows"]=len(pilot_leave_one_provider_out(cfg))
    stat_df = pilot_major_comparison_statistics(cfg)
    results["paired_major_comparisons_available"] = int((stat_df.get("status","")=="ok").sum()) if len(stat_df) else 0
    results["paired_major_comparisons_csv"] = str(root/"stats"/"pilot_major_comparisons_two_sided_holm.csv")
    save_json(root/"experiments"/"pilot_experiments_summary.json", results)
    return results


def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("command", choices=["paper-all","pilot-experiments","training-curves",
                                      "threshold","wsi-results","qualitative"])
    p.add_argument("--data-root", default=None)
    p.add_argument("--work-root", default=None)
    p.add_argument("--no-encoder", action="store_true")
    p.add_argument("--no-provider-holdout", action="store_true")
    return p.parse_args()


def main():
    args=parse_args()
    cfg=core.Config()
    if args.data_root: cfg.data_root=args.data_root
    if args.work_root: cfg.work_root=args.work_root
    cfg.ensure_dirs()
    core.seed_everything(cfg.seed)
    core.setup_torch_speed()

    if args.command=="paper-all":
        paper_all(cfg)
    elif args.command=="pilot-experiments":
        pilot_experiments(cfg, not args.no_encoder, not args.no_provider_holdout)
    elif args.command=="training-curves":
        make_training_curves(cfg)
    elif args.command=="threshold":
        model,device,_=core.load_best_model(cfg,None)
        man=pd.read_csv(cfg.patch_manifest)
        val=man[man.split=="val"].reset_index(drop=True)
        pred=predict_patch_probabilities(cfg,val,model,device,"val","val")
        print("best threshold:", threshold_sweep_wsi(cfg,pred))
    elif args.command=="wsi-results":
        model,device,_=core.load_best_model(cfg,None)
        man=pd.read_csv(cfg.patch_manifest)
        test=man[man.split=="test"].reset_index(drop=True)
        pred=predict_patch_probabilities(cfg,test,model,device,"test","test")
        best=json.load(open(paper_root(cfg)/"tables"/"best_threshold.json","r",encoding="utf-8"))
        ps,g=evaluate_wsi_test(cfg,pred,float(best["threshold"]))
        make_bootstrap_tables(cfg,ps); provider_results(cfg,ps); make_per_slide_plot(cfg,ps)
    elif args.command=="qualitative":
        model,device,_=core.load_best_model(cfg,None)
        man=pd.read_csv(cfg.patch_manifest)
        test=man[man.split=="test"].reset_index(drop=True)
        pred=predict_patch_probabilities(cfg,test,model,device,"test","test")
        best=json.load(open(paper_root(cfg)/"tables"/"best_threshold.json","r",encoding="utf-8"))
        ps=pd.read_csv(paper_root(cfg)/"tables"/"per_slide_wsi_metrics.csv")
        make_qualitative_examples(cfg,ps,pred,float(best["threshold"]))


if __name__ == "__main__":
    main()
