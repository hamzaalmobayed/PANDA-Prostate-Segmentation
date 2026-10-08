
from pathlib import Path
import json, argparse
p=argparse.ArgumentParser()
p.add_argument("--work-root",default=r"D:\PANDA_PROSTATE")
a=p.parse_args()
root=Path(a.work_root)
for f in [root/"pipeline_state.json", root/"runs"/"final_main_training"/"resume_training.pt"]:
    print(f, "EXISTS" if f.exists() else "MISSING")
if (root/"pipeline_state.json").exists():
    print(json.dumps(json.load(open(root/"pipeline_state.json","r",encoding="utf-8")),indent=2))
