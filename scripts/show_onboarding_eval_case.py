from pathlib import Path
import json
import argparse

ROOT = Path(__file__).resolve().parents[1]
EVAL_ROOT = ROOT / "outputs" / "eval_runs"

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
    elig = load_json(case_dir / "guarded_eligibility_v3.json")
    expl = load_json(case_dir / "consent_explanation_strict_postprocessed.json")
    teach = load_json(case_dir / "teachback_questions_targeted.json")

    print("=" * 100)
    print("CASE:", args.case_id)
    print("QUESTION:", pipe.get("input_question", ""))
    print("\nDECISION:", elig.get("decision", ""))
    print("ANSWER:", elig.get("patient_facing_answer", ""))

    print("\nSUPPORTED FACTS:")
    for x in elig.get("supported_patient_facts", []):
        print("-", x)

    print("\nMISSING FACTS:")
    for x in elig.get("missing_patient_facts", []):
        print("-", x)

    print("\nUNRESOLVED STUDY REQUIREMENTS:")
    for x in elig.get("unresolved_study_requirements", []):
        print("-", x)

    print("\nEXPLANATION FIELDS:")
    for field in ["study_purpose", "who_may_join", "procedures_or_participation", "risks_or_burdens", "time_commitment"]:
        val = expl.get(field, {})
        print(f"- {field}: {val.get('text', '')}")

    print("\nWHAT IS UNCLEAR:")
    for x in expl.get("what_is_unclear", []):
        print("-", x)

    print("\nREBUILT SUMMARY:")
    print(pipe.get("consent_explanation", {}).get("patient_facing_summary_rebuilt", {}).get("text", ""))

    print("\nTEACH-BACK QUESTIONS:")
    for q in teach.get("teachback_questions", []):
        print("-", q)

if __name__ == "__main__":
    main()
