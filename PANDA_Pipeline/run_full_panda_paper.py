
from __future__ import annotations

import os
# Performance-only worker startup tuning for Windows spawn.
# Prevent each DataLoader worker from doing Albumentations' online version check.
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import argparse, json, time, traceback
from pathlib import Path
from typing import Dict,Any,Callable

import panda_camelyon_style_pipeline as core
from panda_final_pipeline import (
    FinalConfig, step_audit_final, step_split_final, step_alignment_final,
    step_extract_final, step_delete_train_wsi, step_select_final,
    step_normalization_final, step_dataloader_final, step_sanity_final,
    step_pilot_experiments, step_prune_unselected, step_train_final, list_final_outputs
)
import panda_stream_paper as paper
import panda_resilience as resilience
import panda_paper_suite_legacy as legacy_paper

MAX_RETRIES=3

def state_path(cfg): return cfg.work/"pipeline_state.json"

def load_state(cfg):
    """SSD-safe state read. Never pretend the state is empty just because D: vanished."""
    p=state_path(cfg)
    while True:
        resilience.wait_for_storage_stable(cfg, "reading pipeline_state.json")
        try:
            if not p.exists():
                return {"completed":[],"failed_stage":None,"history":[]}
            with open(p,"r",encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            if resilience.storage_is_currently_missing(cfg) or resilience.looks_like_io_error(e):
                print("[STATE READ WAIT] Storage disappeared while reading pipeline state; waiting for reconnect...")
                resilience.wait_for_storage_stable(cfg, "pipeline state read")
                continue
            # Corrupt JSON is a scientific/resume integrity problem; do not silently reset progress.
            raise

def save_state(cfg,s):
    """SSD-safe atomic state write. If D: disappears here, wait forever and retry."""
    p=state_path(cfg)
    while True:
        resilience.wait_for_storage_stable(cfg, "saving pipeline_state.json")
        try:
            p.parent.mkdir(parents=True,exist_ok=True)
            resilience.atomic_json(p, s)
            return
        except Exception as e:
            if resilience.storage_is_currently_missing(cfg) or resilience.looks_like_io_error(e):
                print("[STATE WRITE WAIT] Storage disappeared while saving pipeline state; waiting for reconnect...")
                resilience.wait_for_storage_stable(cfg, "pipeline state write")
                continue
            raise

def event(cfg,stage,status,details=None):
    # IMPORTANT: this function itself is SSD-safe. The previous version could crash
    # inside the exception handler while trying to mkdir('D:/...') after D: vanished.
    s=load_state(cfg)
    s.setdefault("history",[]).append({"time":time.strftime("%F %T"),"stage":stage,"status":status,"details":details or {}})
    if status=="PASS":
        if stage not in s.setdefault("completed",[]): s["completed"].append(stage)
        s["failed_stage"]=None
    else:
        s["failed_stage"]=stage
    save_state(cfg,s)

def passed(cfg,stage): return stage in load_state(cfg).get("completed",[])

def critical(exc):
    m=repr(exc).lower()
    return any(x in m for x in ["unknown provider","unexpected raw mask values","data leakage","alignment","ground truth","mask semantics"])

def safe_repair(cfg,stage,exc):
    m=repr(exc).lower(); repaired=False
    if "out of memory" in m and cfg.batch_size>1:
        cfg.batch_size=max(1,cfg.batch_size//2); repaired=True
        try:
            import torch
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        except: pass
    if any(x in m for x in ["worker","multiprocessing","broken pipe","spawn"]):
        cfg.num_workers=0; repaired=True
    if stage=="normalization" and any(x in m for x in ["vahadane","stain target","normalizer"]):
        try:
            if cfg.stain_target_path.exists(): cfg.stain_target_path.unlink()
            repaired=True
        except: pass
    if resilience.looks_like_io_error(exc):
        resilience.wait_for_storage(cfg, f"stage {stage}")
        repaired=True
    if stage=="paper-results" and any(x in m for x in ["cache","npz","shape","missing"]):
        # Do not blindly delete valid caches on a storage disconnect.
        if not resilience.looks_like_io_error(exc):
            import shutil
            shutil.rmtree(cfg.work/"paper_results"/"wsi_cache",ignore_errors=True)
        repaired=True
    return repaired

def validate(cfg,stage):
    if stage=="audit":
        assert cfg.audit_csv.exists()
        assert cfg.excluded_missing_masks_csv.exists()
        q=cfg.qc_dir/"01c_native_pyramid.json"; assert q.exists() and json.load(open(q))["passed"]
    elif stage=="split":
        assert cfg.split_csv.exists()
    elif stage=="alignment":
        q=cfg.qc_dir/"03_alignment.json"; assert q.exists() and json.load(open(q))["passed"]
    elif stage=="extract":
        q=cfg.qc_dir/"04_extract_final.json"; assert q.exists() and json.load(open(q))["passed"]
    elif stage=="delete-train-wsi":
        q=cfg.qc_dir/"04b_delete_train_wsi.json"; assert q.exists() and json.load(open(q))["passed"]
    elif stage=="select":
        q=cfg.qc_dir/"05_select_final.json"; assert q.exists() and json.load(open(q))["passed"]
    elif stage=="normalization":
        q=cfg.qc_dir/"06_normalization.json"; assert q.exists() and json.load(open(q))["passed"]
    elif stage=="dataloader":
        q=cfg.qc_dir/"07_dataloader.json"; assert q.exists() and json.load(open(q))["passed"]
    elif stage=="sanity":
        q=cfg.qc_dir/"08_sanity.json"; assert q.exists() and json.load(open(q))["passed"]
    elif stage=="pilot-experiments":
        pr=cfg.work/"paper_results"
        required=[
            pr/"tables"/"pilot_ablation_results.csv",
            pr/"tables"/"pilot_normalization_comparison.csv",
            pr/"tables"/"pilot_kmeans_coordinate_ablation.csv",
            pr/"experiments"/"pilot_experiments_summary.json",
        ]
        for q in required:
            assert q.exists(), q
        # Encoder/provider experiments can be explicitly skipped only when their dependencies/data make that scientific comparison unavailable.
        assert ((pr/"tables"/"pilot_encoder_comparison.csv").exists() or
                (pr/"experiments"/"encoder_comparison_skipped.json").exists())
        assert ((pr/"tables"/"pilot_leave_one_provider_out.csv").exists() or
                (pr/"experiments"/"provider_holdout_skipped.json").exists())
    elif stage=="prune":
        q=cfg.qc_dir/"08b_prune.json"; assert q.exists() and json.load(open(q))["passed"]
    elif stage=="train":
        lr=cfg.work/"latest_run.json"; assert lr.exists()
        rd=Path(json.load(open(lr))["run_dir"]); assert (rd/"best.pt").exists() and (rd/"history.csv").exists()
    elif stage=="reviewer-proof-finalize":
        q=Path(cfg.work_root)/"paper_results"/"reviewer_proof_finalization.json"
        assert q.exists() and json.load(open(q,encoding="utf-8"))["passed"]
    elif stage=="paper-results":
        pr=cfg.work/"paper_results"
        for p in [pr/"tables"/"best_threshold.json",pr/"tables"/"wsi_metrics_per_level_and_fusion.csv",
                  pr/"tables"/"global_wsi_metrics_95CI.csv",pr/"tables"/"provider_metrics_95CI.csv",
                  pr/"figures"/"01_training_curves.png",pr/"figures"/"02_threshold_sensitivity.png",
                  pr/"figures"/"04_per_slide_dice.png"]:
            assert p.exists(), p

def run_stage(cfg,stage,fn):
    """
    Global SSD-safe stage runner.
    SSD/storage interruption never intentionally terminates the pipeline:
    wait for reconnect, then retry the same stage. Fine-grained checkpoints
    inside long stages preserve completed sub-work.
    """
    if passed(cfg,stage):
        print(f"[SKIP PASS] {stage}")
        return

    normal_attempt = 0

    while True:
        resilience.wait_for_storage_stable(cfg, f"before stage {stage}")

        try:
            normal_attempt += 1
            print("")
            print(f"========== {stage} attempt {normal_attempt} ==========")
            fn()
            validate(cfg,stage)
            event(cfg,stage,"PASS",{"attempt":normal_attempt})
            print(f"[PASS] {stage}")
            return

        except KeyboardInterrupt:
            raise

        except Exception as e:
            traceback.print_exc()

            # Classify storage failure BEFORE attempting to append a FAIL event.
            # Otherwise a disconnected D: can cause a second exception while the
            # exception handler itself tries to save pipeline_state.json.
            storage_related = (
                resilience.storage_is_currently_missing(cfg)
                or resilience.looks_like_io_error(e)
                or (
                    hasattr(resilience, "is_openslide_transient_open_error")
                    and resilience.is_openslide_transient_open_error(e)
                )
            )

            if storage_related:
                print("")
                print(f"[SSD/STORAGE INTERRUPTION] stage={stage}")
                print("[SSD/STORAGE INTERRUPTION] Pipeline will wait; it will not intentionally stop.")
                print("[SSD/STORAGE INTERRUPTION] Reconnect the SSD to continue from saved progress.")
                resilience.wait_for_storage_stable(cfg, f"stage {stage}")
                normal_attempt = max(0, normal_attempt - 1)
                continue

            # Only genuine non-storage failures are durably recorded as FAIL.
            event(cfg,stage,"FAIL",{"attempt":normal_attempt,"error":repr(e)})

            if critical(e):
                raise

            if normal_attempt <= MAX_RETRIES and safe_repair(cfg,stage,e):
                event(cfg,stage,"RETRY",{"attempt":normal_attempt})
                print("[AUTO-REPAIR] Retrying the same stage...")
                continue

            raise



def step_all_deferred_pilots(cfg):
    """
    Run ALL deferred/stopped experiments at the very end of the compute-heavy pipeline.
    Existing resume markers and completed experiment folders remain compatible.
    """
    print("=" * 80)
    print("[FINAL ORDER] Core final model + held-out Test + WSI results are complete.")
    print("[FINAL ORDER] NOW resuming ALL deferred experiments:")
    print("  A0-A5 ablations, normalization, KMeans, coordinates, encoders, provider holdout")
    print("[FINAL ORDER] This is the LAST compute-heavy/training stage.")
    print("=" * 80)
    return step_pilot_experiments(cfg)


def step_reviewer_proof_finalize(cfg):
    """
    FINAL lightweight reporting stage.

    Runs only AFTER pilot experiments have finished.
    It does NOT train a model, run dense WSI inference, reconstruct slides,
    or repeat the held-out Test. It only consolidates reviewer-proof
    statistics/reporting that depend on completed pilot outputs.
    """
    pr = Path(cfg.work_root) / "paper_results"
    (pr / "stats").mkdir(parents=True, exist_ok=True)
    (pr / "tables").mkdir(parents=True, exist_ok=True)

    # Recompute paired/Holm pilot statistics now that all available pilots are complete.
    stat_df = legacy_paper.pilot_major_comparison_statistics(cfg)

    # Compact machine-readable finalization record.
    record = {
        "passed": True,
        "purpose": "Post-pilot reviewer-proof statistical/reporting finalization only",
        "does_not_repeat_training": True,
        "does_not_repeat_wsi_test": True,
        "does_not_repeat_wsi_reconstruction": True,
        "pilot_comparisons_available": int((stat_df.get("status", "") == "ok").sum()) if len(stat_df) else 0,
        "pilot_comparisons_csv": str(pr / "stats" / "pilot_major_comparisons_two_sided_holm.csv"),
        "note": "CAMELYON-only ITC/micro/macro/FROC analyses remain intentionally not applicable to PANDA."
    }
    with open(pr / "reviewer_proof_finalization.json", "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)

    # Append a clear final status section to the reproducibility README, if present.
    readme = pr / "README_REPRODUCIBILITY.md"
    if readme.exists():
        body = readme.read_text(encoding="utf-8")
        marker = "\n## Post-pilot finalization\n"
        if marker not in body:
            body += marker
            body += "- Pilot experiments finished before final reviewer-proof ablation statistics were consolidated.\n"
            body += "- Paired two-sided bootstrap, effect sizes, and Holm-corrected p-values are in `stats/pilot_major_comparisons_two_sided_holm.csv`.\n"
            body += "- This finalization step does not rerun training or the dense WSI held-out test.\n"
            readme.write_text(body, encoding="utf-8")

    print("[REVIEWER-PROOF FINALIZE] PASS")
    print(f"[REVIEWER-PROOF FINALIZE] paired pilot comparisons available = {record['pilot_comparisons_available']}")
    return record


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data-root",default=r"D:\prostat")
    p.add_argument("--work-root",default=r"D:\PANDA_PROSTATE")
    p.add_argument("--restart-from",default=None,choices=[
        "audit","split","alignment","extract","delete-train-wsi","select","normalization",
        "dataloader","sanity","pilot-experiments","train","paper-results"
    ])
    args=p.parse_args()
    cfg=FinalConfig(data_root=args.data_root,work_root=args.work_root)
    resilience.wait_for_storage_stable(cfg, 'startup')
    cfg.ensure_dirs()
    resilience.install(legacy_paper)
    core.seed_everything(cfg.seed); core.setup_torch_speed()

    # Explicit CUDA diagnostic: the pipeline uses CUDA automatically when PyTorch sees it.
    try:
        import torch
        if torch.cuda.is_available():
            dev = torch.cuda.current_device()
            print(f"[CUDA] ENABLED | device={dev} | {torch.cuda.get_device_name(dev)} | AMP={bool(getattr(cfg, 'amp', True))}")
            print(f"[CUDA] torch={torch.__version__} | cuda_runtime={torch.version.cuda}")
        else:
            print("[CUDA] NOT AVAILABLE -> PyTorch will run on CPU. Check NVIDIA driver / CUDA-enabled PyTorch if GPU was expected.")
    except Exception as e:
        print(f"[CUDA CHECK WARNING] {e!r}")

    # FINAL-FIRST ORDER (V11): pilots are article ablations and do not configure the final model.
    # Keep all extracted/selected patch files. No pruning is performed in this version.
    # This lets the final model + validation/test/WSI results finish first, then pilots resume, then reviewer-proof pilot statistics are finalized.
    order=["audit","split","alignment","extract","delete-train-wsi","select","normalization",
           "dataloader","sanity","train","paper-results","pilot-experiments","reviewer-proof-finalize"]
    if args.restart_from:
        s=load_state(cfg); idx=order.index(args.restart_from)
        s["completed"]=[x for x in s.get("completed",[]) if x in order and order.index(x)<idx]
        s["failed_stage"]=args.restart_from; save_state(cfg,s)

    stages=[
        ("audit",lambda:step_audit_final(cfg)),
        ("split",lambda:step_split_final(cfg)),
        ("alignment",lambda:step_alignment_final(cfg)),
        ("extract",lambda:step_extract_final(cfg)),
        ("delete-train-wsi",lambda:step_delete_train_wsi(cfg)),
        ("select",lambda:step_select_final(cfg)),
        ("normalization",lambda:step_normalization_final(cfg)),
        ("dataloader",lambda:step_dataloader_final(cfg)),
        ("sanity",lambda:step_sanity_final(cfg)),
        # Final scientific result first. The final configuration is fixed and does NOT depend on pilot outputs.
        ("train",lambda:step_train_final(cfg)),
        # Includes validation threshold selection, held-out Test, WSI reconstruction/fusion and paper metrics.
        ("paper-results",lambda:paper.paper_all(cfg)),
        # Resume/finish the article ablations only after the final model/result is safely complete.
        ("pilot-experiments",lambda:step_all_deferred_pilots(cfg)),
        # Only AFTER all available pilots: consolidate reviewer-proof ablation statistics/reporting.
        # This is lightweight and does NOT rerun Test/WSI inference.
        ("reviewer-proof-finalize",lambda:step_reviewer_proof_finalize(cfg)),
    ]
    total_stages = len(stages)
    for i, (name, fn) in enumerate(stages, 1):
        print(f"\n[PIPELINE PROGRESS] stage {i}/{total_stages}: {name}")
        run_stage(cfg, name, fn)
    print("\nALL STAGES PASS")
    print(json.dumps(list_final_outputs(cfg),indent=2))

if __name__=="__main__": main()
