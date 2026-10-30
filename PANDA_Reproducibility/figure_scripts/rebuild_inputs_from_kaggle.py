"""Re-create the stored training patches and stain-cache files that the figure scripts read.

The figure scripts read stored PANDA training patches (patches/train/...) and
two training-time Vahadane cache files (cache/stain_v10/vahadane/...). These
are on the pipeline machine only. This script writes them again from the
Kaggle WSIs and label masks with the pipeline's own readers and writers:

* image: SlideScaleReader.read_at_downsample, JPEG quality 92, no chroma
  subsampling (panda_final_pipeline.py);
* tumor mask: raw_to_binary_mask, PNG compress level 3;
* valid mask: raw_valid_tissue_mask x 255, PNG compress level 3;
* stain cache: TIAToolbox VahadaneNormalizer fitted on stain_target_train.png,
  PNG compress level 1, file name = first 12 hex digits of SHA-1 of the patch
  path stored in the manifest (stain_cache_path in
  panda_camelyon_style_pipeline.py).

Every written file whose path appears in a figure log is compared with the
SHA-256 recorded there; the script stops on any difference.

Usage:
    python rebuild_inputs_from_kaggle.py <work_root> <prostate-cancer-grade-assessment.zip> <provenance_dir>

work_root must already hold manifests/patches.csv, selected/train_selected.csv
and stain_target_train.png. Manifest paths start with D:\\PANDA_PROSTATE; they
are written under work_root.
"""
import hashlib
import json
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "PANDA_Pipeline"))
import panda_camelyon_style_pipeline as core  # noqa: E402
from tiatoolbox.tools.stainnorm import VahadaneNormalizer  # noqa: E402

WORK, ZIP, PROV = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
SLIDES = ("362d30ace92ce17128d176137362b8c2", "0018ae58b01bdadc8e347995b69f99aa")
PREFIX = "D:\\PANDA_PROSTATE\\"
DS = {0: 1, 1: 4, 2: 16}
SEED = 42


def local(p):
    p = str(p)
    assert p.startswith(PREFIX), p
    return WORK / p[len(PREFIX):].replace("\\", "/")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


logged = {}
for f in PROV.glob("Fig*_sources.json"):
    def walk(o):
        if isinstance(o, dict):
            if "path" in o and "sha256" in o:
                logged[o["path"]] = o["sha256"]
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    walk(json.loads(f.read_text(encoding="utf-8")))


def check(path):
    key = path.relative_to(WORK).as_posix()
    if key not in logged:
        return "not logged"
    assert sha(path) == logged[key], f"SHA-256 differs from the figure log: {key}"
    return "match"


rows = pd.concat([c[c.image_id.isin(SLIDES)] for c in pd.read_csv(WORK / "manifests" / "patches.csv", chunksize=200_000)])
status = {"match": 0, "not logged": 0}
with zipfile.ZipFile(ZIP) as z, tempfile.TemporaryDirectory() as tmp:
    for sid, grp in rows.groupby("image_id"):
        wsi = z.extract(f"train_images/{sid}.tiff", tmp)
        msk = z.extract(f"train_label_masks/{sid}_mask.tiff", tmp)
        im = core.SlideScaleReader(wsi)
        mr = core.MaskScaleReader(msk, im.dimensions)
        for r in grp.itertuples():
            rgb, _ = im.read_at_downsample(int(r.x0), int(r.y0), DS[int(r.level)], 512)
            raw, _ = mr.read_raw(int(r.x0), int(r.y0), DS[int(r.level)], 512)
            files = {local(r.img): (Image.fromarray(rgb), {"format": "JPEG", "quality": 92, "subsampling": 0}),
                     local(r.mask): (Image.fromarray(core.raw_to_binary_mask(raw, r.provider).astype(np.uint8)),
                                     {"format": "PNG", "compress_level": 3}),
                     local(r.valid_mask): (Image.fromarray(core.raw_valid_tissue_mask(raw, r.provider).astype(np.uint8) * 255),
                                           {"format": "PNG", "compress_level": 3})}
            for path, (pic, kw) in files.items():
                path.parent.mkdir(parents=True, exist_ok=True)
                pic.save(path, **kw)
                status[check(path)] += 1
        im.close(); mr.close()
print("patch files written:", len(rows) * 3, status)

# training-time Vahadane copies named in the figure logs
cached = {Path(p).stem: p for p in logged if p.startswith("cache/stain_v10/vahadane/")}
norm = VahadaneNormalizer()
np.random.seed(SEED)
norm.fit(np.array(Image.open(WORK / "stain_target_train.png").convert("RGB"), dtype=np.uint8, copy=True, order="C"))
sel = pd.read_csv(WORK / "selected" / "train_selected.csv", usecols=["image_id", "img"])
sel = sel[sel.image_id.isin(SLIDES)]
for img in sel.img:
    key = hashlib.sha1(str(Path(str(img))).encode("utf-8")).hexdigest()[:12]
    if key not in cached:
        continue
    np.random.seed(SEED)
    src = np.array(Image.open(local(img)).convert("RGB"), dtype=np.uint8, copy=True, order="C")
    out = np.array(norm.transform(src), dtype=np.uint8, copy=True)
    path = WORK / cached.pop(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(out, mode="RGB").save(path, format="PNG", compress_level=1)
    print("stain cache", path.name, check(path))
assert not cached, f"cache files not rebuilt: {sorted(cached)}"
