"""Pack the WSI caches that are not in the repository into release archives.

panda_wsi_cache_complement.zip holds every PANDA cache missing from
PANDA_Reproducibility/paper_results/wsi_cache/ in the repository (listed in
<missing_list>); precise_wsi_cache.zip holds all PRECISE caches. Paths inside
the archives are relative to PANDA_Reproducibility/, so extracting there
completes the cache folders. wsi_cache_archive_manifest.csv gives the size and
SHA-256 of every packed file.

Usage:
    python make_cache_archives.py <output_dir> <missing_list> [work_root]
"""
import csv
import hashlib
import os
import sys
import zipfile
from pathlib import Path

if len(sys.argv) < 4 and "PANDA_DATA_ROOT" not in os.environ:
    sys.exit("usage: make_cache_archives.py <output_dir> <missing_list> [work_root]  (or set PANDA_DATA_ROOT)")
OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
MISSING = Path(sys.argv[2]).read_text().split()
WORK = Path(sys.argv[3] if len(sys.argv) > 3 else os.environ["PANDA_DATA_ROOT"])

archives = {
    "panda_wsi_cache_complement.zip": [WORK / "paper_results" / "wsi_cache" / n for n in MISSING],
    "precise_wsi_cache.zip": sorted((WORK / "external_precise_wsi" / "wsi_cache").glob("*.npz")),
}
rows = []
for name, files in archives.items():
    with zipfile.ZipFile(OUT / name, "w", zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as z:
        for f in files:
            arc = f.relative_to(WORK).as_posix()
            z.write(f, arc)
            rows.append({"archive": name, "path": arc, "bytes": f.stat().st_size,
                         "sha256": hashlib.sha256(f.read_bytes()).hexdigest()})
    with zipfile.ZipFile(OUT / name) as z:
        bad = z.testzip()
    assert bad is None, f"corrupt member {bad} in {name}"
    print(f"{name}: {len(files)} files, {(OUT / name).stat().st_size / 1e6:.0f} MB", flush=True)
with open(OUT / "wsi_cache_archive_manifest.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=["archive", "path", "bytes", "sha256"]); w.writeheader(); w.writerows(rows)
print("manifest rows:", len(rows))
