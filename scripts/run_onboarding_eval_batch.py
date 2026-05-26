from pathlib import Path
import json
import subprocess
import sys
import shutil

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
OUT = ROOT / "outputs" / "eval_runs"
OUT.mkdir(parents=True, exist_ok=True)

PIPELINE_SCRIPT = ROOT / "scripts" / "run_onboarding_pipeline.py"
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

CASES_PATH = DATA / "onboarding_eval_cases.json"
MASTER_SUMMARY_PATH = OUT / "eval_case_index.json"


def run_case(case):
    case_id = case["case_id"]
    case_dir = OUT / case_id
    case_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(PIPELINE_SCRIPT),
        "--question", case["question"],
        "--model_name", MODEL_NAME,
        "--top_docs", "20",
        "--top_passages", "8",
    ]

    print(f"\n=== Running {case_id} ===")
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        raise RuntimeError(f"Pipeline failed for {case_id}")

    # copy relevant outputs into case-specific folder
    copy_files = [
        ROOT / "outputs" / "tables" / "onboarding_pipeline_output.json",
        ROOT / "outputs" / "tables" / "grounding_evidence_top_passages.csv",
        ROOT / "outputs" / "tables" / "guarded_eligibility_v3.json",
        ROOT / "outputs" / "tables" / "consent_explanation_strict_postprocessed.json",
        ROOT / "outputs" / "tables" / "teachback_questions_targeted.json",
    ]

    saved = {}
    for p in copy_files:
        if p.exists():
            dst = case_dir / p.name
            shutil.copy2(p, dst)
            saved[p.name] = str(dst)

    return {
        "case_id": case_id,
        "category": case["category"],
        "question": case["question"],
        "notes": case.get("notes", ""),
        "case_dir": str(case_dir),
        "files": saved
    }


def main():
    with open(CASES_PATH, "r") as f:
        cases = json.load(f)

    records = []
    for case in cases:
        rec = run_case(case)
        records.append(rec)

    with open(MASTER_SUMMARY_PATH, "w") as f:
        json.dump(records, f, indent=2)

    print("\nSaved master index:", MASTER_SUMMARY_PATH)


if __name__ == "__main__":
    main()
