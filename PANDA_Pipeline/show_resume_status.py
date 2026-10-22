

def _panda_portable_path(original):
    """Resolve legacy PANDA paths through configurable project roots."""
    import os
    from pathlib import Path

    text = str(original)
    normalized = text.replace("/", "\\")

    roots = [
        (
            "PANDA_ARTICLE_ROOT",
            'D:\\PANDA_PROSTATE_FINAL_ARTICLE_COMPLETE_SAFE_RESUME_V31_KAGGLE_SICAP_DOWNLOAD_FIX\\final_article_pkg'
        ),
        (
            "PANDA_DATA_ROOT",
            r"D:\PANDA_PROSTATE"
        ),
    ]

    for variable, legacy_root in roots:
        legacy = legacy_root.replace("/", "\\").rstrip("\\")
        if (
            normalized.lower() == legacy.lower()
            or normalized.lower().startswith(legacy.lower() + "\\")
        ):
            root = os.environ.get(variable, legacy)
            suffix = normalized[len(legacy):].lstrip("\\")
            return str(Path(root).joinpath(*[
                x for x in suffix.split("\\") if x
            ]))

    return original



from pathlib import Path
import json, argparse
p=argparse.ArgumentParser()
p.add_argument("--work-root",default=_panda_portable_path(r"D:\PANDA_PROSTATE"))
a=p.parse_args()
root=Path(a.work_root)
for f in [root/"pipeline_state.json", root/"runs"/"final_main_training"/"resume_training.pt"]:
    print(f, "EXISTS" if f.exists() else "MISSING")
if (root/"pipeline_state.json").exists():
    print(json.dumps(json.load(open(root/"pipeline_state.json","r",encoding="utf-8")),indent=2))
