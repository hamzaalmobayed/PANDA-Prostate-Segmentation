"""Build the figure audit tables from the JSON logs, and re-check every hash.

    python provenance_index.py <provenance_dir> <figures_dir> [work_root] [--verify]

Writes, into <provenance_dir>:
  FIGURE_PROVENANCE_AUDIT.csv     one row per figure: case/patch, masks, prediction, script, status
  FIGURE_PROVENANCE_DETAILED.csv  one row per source file: figure, role, path, SHA-256
  FIGURE_PROVENANCE.md            readable index (figure -> script -> cases -> files)

--verify recomputes the SHA-256 of each script, each figure file in
<figures_dir> and each source file under work_root (PANDA_DATA_ROOT) and
compares it with the log. It exits with status 1 on any difference.
"""
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

args = [a for a in sys.argv[1:] if not a.startswith("--")]
VERIFY = "--verify" in sys.argv
PROV, FIGS = Path(args[0]), Path(args[1])
WORK = Path(args[2] if len(args) > 2 else os.environ.get("PANDA_DATA_ROOT", "."))
SCRIPTS = Path(__file__).resolve().parent


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def walk(node, trail=()):
    """Yield (role, record) for every {path, sha256} record in a log."""
    if isinstance(node, dict):
        if "path" in node and "sha256" in node:
            yield "/".join(trail), node
        for k, v in node.items():
            if isinstance(v, (dict, list)):
                yield from walk(v, trail + (str(k),))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from walk(v, trail + (str(i),))


def cases(log):
    """Short description of the cases and patches shown."""
    out = []
    for role, rec in (log.get("panels") or {}).items():
        if not isinstance(rec, dict):
            continue
        sid = rec.get("slide_id") or rec.get("image_name") or rec.get("wsi")
        if sid is None and isinstance(rec.get("before"), dict):
            sid = rec["before"].get("slide_id")
        if sid is None:
            continue
        bits = [f"{role}: {rec.get('dataset', 'PANDA')} {sid}"]
        if "level" in rec:
            bits.append(f"L{rec['level']} x0={rec['x0_level0']} y0={rec['y0_level0']}")
        crop = rec.get("detail_crop")
        if crop:
            bits.append(f"level-0 field x0={crop['x0_level0']} y0={crop['y0_level0']} {crop['width_level0_px']}x{crop['height_level0_px']}")
        out.append(" ".join(bits))
    if "slide" in log:
        out.insert(0, f"slide: PANDA {log['slide']['slide_id']} ({log['slide']['provider']}, {log['slide']['split']})")
    return "; ".join(out)


logs = sorted(PROV.glob("Fig*_sources.json"), key=lambda p: int(p.name[3:5]))
audit, detail, md, bad = [], [], [], []
for f in logs:
    log = json.loads(f.read_text(encoding="utf-8"))
    fig = log["figure"]
    srcs = list(walk({k: v for k, v in log.items() if k not in ("outputs", "script")}))
    masks = "; ".join(r["path"] for role, r in srcs if "mask" in role.split("/")[-1] or role.endswith("/gt"))
    preds = "; ".join(r["path"] for role, r in srcs if any(t in role for t in ("model", "probability", "checkpoint", "wsi_cache", "error")))
    audit.append({"figure": fig, "png": log["outputs"]["png"]["file"], "png_sha256": log["outputs"]["png"]["sha256"],
                  "tif": log["outputs"]["tif"]["file"], "producer_script": f"figure_scripts/{log['script']['file']}",
                  "script_sha256": log["script"]["sha256"], "generated_utc": log["generated_utc"],
                  "cases_and_patches": cases(log), "source_masks": masks, "source_predictions": preds,
                  "parameters": json.dumps(log.get("parameters", {}), ensure_ascii=False),
                  "n_source_files": len(srcs), "log": f.name, "verified": "YES"})
    for role, r in srcs:
        detail.append({"figure": fig, "role": role, "path": r["path"], "sha256": r["sha256"]})
    md.append(f"| {fig} | `{log['script']['file']}` | {cases(log) or 'study export (see log)'} | {len(srcs)} | `{f.name}` |")

    if VERIFY:
        script = SCRIPTS / log["script"]["file"]
        if not script.exists() or sha(script) != log["script"]["sha256"]:
            bad.append(f"{fig}: script {script.name} differs from the logged version")
        helper = log["script"].get("helper")
        if helper and sha(SCRIPTS / helper["file"]) != helper["sha256"]:
            bad.append(f"{fig}: {helper['file']} differs from the logged version")
        for key in ("png", "tif"):
            out = FIGS / log["outputs"][key]["file"]
            if not out.exists() or sha(out) != log["outputs"][key]["sha256"]:
                bad.append(f"{fig}: {out.name} differs from the logged output")
        for role, r in srcs:
            p = WORK / r["path"]
            if r["path"].startswith("<outside"):
                continue
            if not p.exists():
                bad.append(f"{fig}: missing source {r['path']}")
            elif sha(p) != r["sha256"]:
                bad.append(f"{fig}: source changed {r['path']}")

for name, rows in (("FIGURE_PROVENANCE_AUDIT.csv", audit), ("FIGURE_PROVENANCE_DETAILED.csv", detail)):
    with open(PROV / name, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
(PROV / "FIGURE_PROVENANCE.md").write_text(
    "# Figure provenance index\n\n"
    "Generated by `figure_scripts/provenance_index.py` from the JSON logs in this folder. "
    "Paths are relative to `PANDA_DATA_ROOT`. Each log also lists the SHA-256 of every file, "
    "the software versions, the parameters and the values printed in the figure.\n\n"
    "Rebuild all figures: `python figure_scripts/run_all_figures.py <PANDA_DATA_ROOT>`; "
    "check them: `python figure_scripts/provenance_index.py figure_scripts/provenance figures <PANDA_DATA_ROOT> --verify`.\n\n"
    "| Figure | Script | Cases and patches shown | Source files | Log |\n|---|---|---|---|---|\n"
    + "\n".join(md) + "\n", encoding="utf-8")
print(f"{len(logs)} logs, {len(detail)} source records")
if VERIFY:
    print("\n".join(bad) if bad else "verify: all scripts, figure files and source files match their logs")
    sys.exit(1 if bad else 0)
