"""Light worker functions for Vahadane normalization in separate processes.

On Windows, worker processes re-import the launching script, so the two
revision scripts keep their heavy imports out of the launcher and their
workers use only this module: NumPy, OpenCV, PIL and TIAToolbox's
VahadaneNormalizer (through light_vahadane, about 0.19 GB per process instead
of 0.9 GB). generate_tissue_mask is a copy of the function of the same name in
panda_camelyon_style_pipeline.py with the executed settings (threshold 0.20,
disk 2, minimum object 500 px); check_equivalence() compares both.
"""
import numpy as np
import cv2
from PIL import Image

TISSUE_DEFAULT_THRESHOLD, MORPH_DISK_SIZE, MIN_OBJECT_SIZE = 0.20, 2, 500
_PROC = None


def init(target_path):
    global _PROC
    import logging
    logging.getLogger("tiatoolbox").setLevel(logging.ERROR)
    import light_vahadane
    np.random.seed(42)
    _PROC = light_vahadane.VahadaneProcessor(np.array(Image.open(target_path).convert("RGB"), dtype=np.uint8, copy=True))


def normalize(rgb):
    return _PROC.transform(rgb)


def generate_tissue_mask(rgb):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    sat = hsv[..., 1]
    try:
        otsu, _ = cv2.threshold(sat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        threshold = max(int(otsu), int(round(TISSUE_DEFAULT_THRESHOLD * 255)))
    except Exception:
        threshold = int(round(TISSUE_DEFAULT_THRESHOLD * 255))
    mask = sat > threshold
    k = max(1, int(MORPH_DISK_SIZE))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
    m = mask.astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
    n, labels, stats, _ = cv2.connectedComponentsWithStats((m > 0).astype(np.uint8), 8)
    clean = np.zeros_like(m, dtype=np.uint8)
    for i in range(1, n):
        if int(stats[i, cv2.CC_STAT_AREA]) >= int(MIN_OBJECT_SIZE):
            clean[labels == i] = 255
    if float((clean > 0).mean()) < 0.0005:
        clean = (((rgb.mean(axis=2) < 240) & (rgb.min(axis=2) < 245)).astype(np.uint8) * 255)
    return clean


def prepare_sicap(paths):
    """prepare_sample() of run_external_sicapv2_test_LIVE_VISUALS.py, returning uint8 data."""
    img_path, mask_path = paths
    rgb = np.asarray(Image.open(img_path).convert("RGB"), dtype=np.uint8)
    h, w = rgb.shape[:2]
    m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if m.shape[:2] != (h, w):
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
    gt = m > 0
    valid = generate_tissue_mask(rgb).astype(bool)
    if valid.mean() < 0.01:
        valid = rgb.mean(axis=2) < 245
    if valid.mean() < 0.01:
        valid = np.ones((h, w), bool)
    return _PROC.transform(rgb), gt, valid


def check_equivalence(paths):
    import panda_camelyon_style_pipeline as core
    from panda_final_pipeline import FinalConfig
    cfg = FinalConfig()
    for p in paths:
        rgb = np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8)
        assert np.array_equal(generate_tissue_mask(rgb), core.generate_tissue_mask(rgb, cfg)[0]), p
    return len(paths)
