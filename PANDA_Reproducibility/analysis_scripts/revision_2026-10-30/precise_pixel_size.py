"""Level-0 pixel size of the 27 PRECISE H&E WSIs, and the false-positive areas
of the tumor-free WSIs with each WSI's own pixel size (2 Nov 2026).

precise_first_run_analyses.py converts evaluation-grid pixels to mm2 with
0.243 um per level-0 pixel (PRECISE README) for every WSI. The TIFF resolution
tags of the H&E files give 0.2433 um for 25 WSIs and 0.1766 um for two
(sub-24_ses-01, sub-25_ses-01). This script reads those tags (XResolution and
ResolutionUnit of the first IFD) from the PRECISE data.zip on Zenodo by HTTP
range requests, without downloading the archive, and rescales the areas of
PRECISE_negative_wsi.csv. Pixel counts, Dice and the other metrics do not
depend on the pixel size and are unchanged.

Usage: python precise_pixel_size.py <precise_he_manifest.csv> <precise_summary_dir>
Output (in <precise_summary_dir>):
  PRECISE_pixel_size.csv        um per pixel at level 0, 2 and 4 (evaluation grid) per WSI
  PRECISE_negative_wsi_area.csv false-positive area and largest component with each WSI's pixel size
  PRECISE_area_summary.json
"""
import json
import struct
import sys
import urllib.request
import zlib
from pathlib import Path

import pandas as pd

URL = "https://zenodo.org/api/records/20721779/files/data.zip/content"   # PRECISE, doi:10.5281/zenodo.20721779
UM_README = 0.243                                                       # value used by precise_first_run_analyses.py
DS = 16                                                                 # evaluation grid = level 4


def get(off, n):
    req = urllib.request.Request(URL, headers={"Range": f"bytes={off}-{off + n - 1}"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def zip_entries():
    head = urllib.request.urlopen(urllib.request.Request(URL, headers={"Range": "bytes=0-0"}), timeout=120)
    size = int(head.headers["Content-Range"].split("/")[1])
    tail = get(size - 65536, 65536)
    i = tail.rfind(b"PK\x06\x06")                                       # zip64 end of central directory
    cd_size, cd_off = struct.unpack("<QQ", tail[i + 40:i + 56])
    cd, p, out = get(cd_off, cd_size), 0, {}
    while cd[p:p + 4] == b"PK\x01\x02":
        meth, = struct.unpack("<H", cd[p + 10:p + 12])
        csz, usz = struct.unpack("<II", cd[p + 20:p + 28])
        nl, el, kl = struct.unpack("<HHH", cd[p + 28:p + 34])
        loff, = struct.unpack("<I", cd[p + 42:p + 46])
        name, extra, q = cd[p + 46:p + 46 + nl].decode(), cd[p + 46 + nl:p + 46 + nl + el], 0
        while q < len(extra):                                           # zip64 extra field
            hid, hl = struct.unpack("<HH", extra[q:q + 4])
            if hid == 1:
                vals, k = struct.unpack(f"<{hl // 8}Q", extra[q + 4:q + 4 + hl - hl % 8]), 0
                if usz == 0xFFFFFFFF: usz = vals[k]; k += 1
                if csz == 0xFFFFFFFF: csz = vals[k]; k += 1
                if loff == 0xFFFFFFFF: loff = vals[k]
            q += 4 + hl
        out[name] = (meth, csz, loff)
        p += 46 + nl + el + kl
    return out


def level0_um(entry):
    """Size and um per pixel from the first IFD of a zipped (BigTIFF or classic) TIFF."""
    meth, csz, loff = entry
    nl, el = struct.unpack("<HH", get(loff, 30)[26:30])
    start = loff + 30 + nl + el
    raw = get(start, min(csz, 1 << 20))
    buf = zlib.decompressobj(-15).decompress(raw) if meth == 8 else raw
    E = ">" if buf[:2] == b"MM" else "<"
    big = struct.unpack(E + "H", buf[2:4])[0] == 43
    ifd = struct.unpack(E + "Q", buf[8:16])[0] if big else struct.unpack(E + "I", buf[4:8])[0]
    n_fmt, ent, n_len = (E + "Q", 20, 8) if big else (E + "H", 12, 2)
    cnt, = struct.unpack(n_fmt, buf[ifd:ifd + n_len])
    tags = {}
    for t in range(cnt):
        e = buf[ifd + n_len + t * ent:ifd + n_len + (t + 1) * ent]
        tag, typ = struct.unpack(E + "HH", e[:4])
        val = e[12:20] if big else e[8:12]
        if typ == 3:
            tags[tag] = struct.unpack(E + "H", val[:2])[0]
        elif typ == 5:
            if not big:
                o = struct.unpack(E + "I", val)[0]; val = buf[o:o + 8]
            num, den = struct.unpack(E + "II", val[:8]); tags[tag] = num / den
        elif typ in (4, 16):
            tags[tag] = struct.unpack(E + "Q", val)[0] if typ == 16 else struct.unpack(E + "I", val[:4])[0]
    unit = {2: 25_400.0, 3: 10_000.0}[tags[296]]                        # um per inch / per cm
    return tags[256], tags[257], unit / tags[282]


man, out = Path(sys.argv[1]), Path(sys.argv[2])
entries = zip_entries()
rows = []
for r in pd.read_csv(man).itertuples():
    w, h, um = level0_um(entries[r.image_member])
    rows.append({"image_id": r.image_id, "width_level0": w, "height_level0": h, "um_per_px_level0": um,
                 "um_per_px_level2": 4 * um, "um_per_px_eval_level4": DS * um})
px = pd.DataFrame(rows)
px.to_csv(out / "PRECISE_pixel_size.csv", index=False)

neg = pd.read_csv(out / "PRECISE_negative_wsi.csv").merge(px[["image_id", "um_per_px_eval_level4"]], on="image_id")
old = (UM_README * DS / 1000) ** 2                                     # mm2 per grid pixel in the first analysis
new = (neg.um_per_px_eval_level4 / 1000) ** 2
neg["largest_component_px"] = (neg.largest_component_mm2 / old).round().astype(int)
neg["fp_area_mm2"] = neg.fp_pixels * new
neg["largest_component_mm2"] = neg.largest_component_px * new
neg.to_csv(out / "PRECISE_negative_wsi_area.csv", index=False)

um0 = px.um_per_px_level0.round(4)
alt = px[um0 != um0.mode()[0]]
summ = {"um_per_px_level0_values": {str(k): int(v) for k, v in um0.value_counts().items()},
        "wsi_other_pixel_size": alt.image_id.tolist(),
        "um_per_px_level0_other": float(alt.um_per_px_level0.iloc[0]) if len(alt) else None,
        "neg_fp_area_mm2_total": float(neg.fp_area_mm2.sum()),
        "neg_largest_component_mm2": float(neg.largest_component_mm2.max()),
        "neg_largest_component_wsi": neg.loc[neg.largest_component_mm2.idxmax(), "image_id"],
        "neg_fp_area_mm2_total_at_0.243um": float((neg.fp_pixels * old).sum())}
json.dump(summ, open(out / "PRECISE_area_summary.json", "w"), indent=2)
print(px.to_string(index=False)); print(json.dumps(summ, indent=2))
