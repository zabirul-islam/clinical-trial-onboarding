from pathlib import Path
import json
import argparse

ROOT = Path(__file__).resolve().parents[1]
EVAL_ROOT = ROOT / "outputs" / "eval_runs_v2"

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case_id", type=str, required=True)
    args = parser.parse_args()

    case_dir = EVAL_ROOT / args.case_id
    if not case_dir.exists():
        raise FileNotFoundError(f"Missing case directory: {case_dir}")

    pipe = load_json(case_dir / "onboarding_pipeline_output.json")

    print("=" * 100)
    print("CASE:", args.case_id)
    print("QUESTION:", pipe.get("input_question", ""))
    print("\nCONTEXT STATUS:", pipe.get("context_control", {}).get("status", ""))
    print("SELECTED DOC:", pipe.get("context_control", {}).get("selected_doc_id", ""))
    print("WEIGHT SHARE:", pipe.get("context_control", {}).get("selected_doc_weight_share", ""))
    print("\nDECISION:", pipe.get("eligibility_assessment", {}).get("decision", ""))
    print("ANSWER:", pipe.get("eligibility_assessment", {}).get("patient_facing_answer", ""))

    print("\nREBUILT SUMMARY:")
    print(pipe.get("consent_explanation", {}).get("patient_facing_summary_rebuilt", {}).get("text", ""))

    print("\nTEACH-BACK QUESTIONS:")
    for q in pipe.get("teachback", {}).get("teachback_questions", []):
        print("-", q)

if __name__ == "__main__":
    main()
