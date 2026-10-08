
# PANDA Prostate — FINAL named patch/mask folders + SSD-safe resume

## Run

Recommended:

```text
RUN_ALL_RESUMABLE.bat
```

or:

```bash
python run_full_panda_paper.py
```

Default paths:

```text
DATA = D:\prostat
WORK = D:\PANDA_PROSTATE
```

## Exact folder structure

```text
D:\PANDA_PROSTATE\patches\
├── train\
│   └── <image_id>\
│       ├── L0\
│       │   ├── Normal\
│       │   ├── Normal_Masks\
│       │   ├── Normal_Valid_Masks\
│       │   ├── Tumor\
│       │   ├── Tumor_Masks\
│       │   └── Tumor_Valid_Masks\
│       ├── L1\
│       │   └── same 6 folders
│       └── L2\
│           └── same 6 folders
├── val\
│   └── <image_id>\L0|L1|L2\same 6 folders
└── test\
    └── <image_id>\L0|L1|L2\same 6 folders
```

The folders are created before patch extraction, so an empty folder is normal until that slide/level/class has accepted patches.

## Reconstruction-safe patch names

Example:

```text
00a7fb...__split-train__L0__x0-18432__y0-9216__ds-1__ps-512__stride-128__class-Normal.jpg
```

Corresponding tumor mask:

```text
00a7fb...__split-train__L0__x0-18432__y0-9216__ds-1__ps-512__stride-128__class-Normal__tumor_mask.png
```

Corresponding valid/annotated-tissue mask:

```text
00a7fb...__split-train__L0__x0-18432__y0-9216__ds-1__ps-512__stride-128__class-Normal__valid_mask.png
```

The name contains:
- original WSI `image_id`
- split
- logical pyramid level
- baseline WSI coordinate `x0`
- baseline WSI coordinate `y0`
- target downsample
- patch size
- stride
- Normal/Tumor class

Therefore a patch can always be placed back at its original WSI location.

The CSV manifest also retains:
- `image_id`
- `provider`
- `split`
- `level`
- `x0`, `y0`
- normalized coordinates
- target downsample
- native level/downsample
- tissue fraction
- tumor fraction
- valid annotated fraction
- image path
- tumor-mask path
- valid-mask path

## Why two mask types?

`*_tumor_mask.png`:
binary prostate tumor GT.

`*_valid_mask.png`:
pixels that are truly annotated tissue in PANDA. Background/unknown pixels are ignored by loss and metrics instead of being counted as Normal.

## Final Train selection

Six independent buckets:
- L0 Normal
- L0 Tumor
- L1 Normal
- L1 Tumor
- L2 Normal
- L2 Tumor

80,000 total, as equally distributed as mathematically possible.

## Resume / SSD disconnect

This package preserves:
- stage-level PASS resume
- coordinate-scan progress
- patch-materialization shards
- feature-batch shards
- Train/Validation training checkpoints
- WSI result cache
- exact-path SSD reconnect waiting
- OpenSlide transient retry handling

## Train WSI deletion

As requested for SSD space, Train WSI TIFFs may be deleted from the working SSD only after extraction + patch/mask QC proves the six Train buckets are sufficiently populated. Validation/Test WSI and all original label masks remain.


## Detailed extraction console log

After EVERY completed WSI, the terminal now prints:

```text
[WSI DONE] 12/7358 | split=train | image_id=00a7...
  L0: Normal=4 | Tumor=2 | Total=6
  L1: Normal=4 | Tumor=1 | Total=5
  L2: Normal=3 | Tumor=0 | Total=3
       Tumor EMPTY -> no Tumor candidates >= threshold; low_tissue=..., low_valid=..., mixed=...
  >>> TOTAL PATCHES SAVED FROM WSI = 14
```

If nothing is saved:

```text
>>> NO PATCHES SAVED FROM THIS WSI
```

and each empty Normal/Tumor bucket prints the reason.

Durable logs are also written to:

```text
D:\PANDA_PROSTATE\logs\extraction_train_cap4.csv
D:\PANDA_PROSTATE\logs\extraction_val_cap2.csv
D:\PANDA_PROSTATE\logs\extraction_test_cap2.csv
```

Per-image/per-level diagnostic JSON files are written under:

```text
D:\PANDA_PROSTATE\manifests\coordinate_pools\
D:\PANDA_PROSTATE\manifests\materialize_shards\
```

The logs count:
- low-tissue rejection
- low valid-annotation rejection
- mixed/borderline rejection
- pure Normal candidates
- Tumor candidates
- exact-resolution rejection
- class changes after exact GT read
- number actually saved


## GLOBAL SSD WAIT — EVERY PIPELINE STAGE

The runner now wraps EVERY stage in a global SSD/storage guard.

If the SSD disappears during audit, extraction, feature extraction, KMeans/result saving,
normalization, pilot experiments, training, validation, Test inference, WSI reconstruction,
or result/checkpoint writing, a storage-related exception does not intentionally terminate
the pipeline. It waits for the data/work storage to return and become stable, then retries
the same stage.

Fine-grained checkpoints preserve completed coordinate scans, saved patches, feature batches,
training checkpoints, validation progress, completed test WSI caches, and PASSed stages.

If Python itself dies, RUN_ALL_RESUMABLE.bat restarts it automatically.

Scientific errors such as a genuinely corrupt file, unknown mask semantics, leakage, or failed
alignment are NOT hidden as SSD disconnects.


# ULTRAFAST EXTRACTION — ONE OPEN PER WSI

This build is backward-compatible with the current `D:\PANDA_PROSTATE` progress.

Do NOT delete the work folder.

If 883/7358 or any other number of WSI are already complete, their:
- extraction log rows,
- coordinate CSVs,
- materialization shards,
- patch images,
- tumor masks,
- valid masks

are reused and not rebuilt.

For every NEW/unfinished WSI the extraction now does:

```text
Open WSI once
Open label mask once
Build low-resolution Tissue/Tumor/Valid masks once
    ↓
L0 exact stride-128 grid (vectorized)
L1 exact stride-128 grid (vectorized)
L2 exact stride-128 grid (vectorized)
    ↓
Read/save only the accepted patches
Close WSI + mask once
```

Previous slow behavior repeatedly reopened/rebuilt data for L0/L1/L2.

Scientific settings are unchanged:
- L0/L1/L2
- patch 512x512
- stride 128
- Normal = zero tumor in valid annotated tissue
- Tumor >= tumor threshold
- mixed/borderline excluded from the six balancing buckets
- same six buckets
- same final 80,000 selection
- same EfficientNet-B4 + KMeans
- same filenames/folders/coordinates
- same SSD disconnect global wait/resume behavior

This is an I/O/vectorization optimization only.
