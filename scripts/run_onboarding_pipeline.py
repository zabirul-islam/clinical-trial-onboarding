from pathlib import Path
import argparse
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "tables"

QUESTION_EXPLAIN = "Can you explain in simple words what this study is about, who may join, what participation seems to involve, and what is still unclear?"
FALLBACK = "not clearly stated in the retrieved evidence"


def run_cmd(cmd):
    print("\nRUNNING:", " ".join(cmd))
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def safe_text(field_obj):
    if not isinstance(field_obj, dict):
        return FALLBACK
    return str(field_obj.get("text", FALLBACK)).strip()


def rebuild_patient_summary(expl):
    pieces = []

    study_purpose = safe_text(expl.get("study_purpose"))
    who_may_join = safe_text(expl.get("who_may_join"))
    procedures = safe_text(expl.get("procedures_or_participation"))
    risks = safe_text(expl.get("risks_or_burdens"))
    time_commitment = safe_text(expl.get("time_commitment"))

    if study_purpose != FALLBACK:
        pieces.append(f"Study purpose: {study_purpose}")
    if who_may_join != FALLBACK:
        pieces.append(f"Who may join: {who_may_join}")
    if procedures != FALLBACK:
        pieces.append(f"What participation may involve: {procedures}")
    if risks != FALLBACK:
        pieces.append(f"Risks or burdens: {risks}")
    if time_commitment != FALLBACK:
        pieces.append(f"Time commitment: {time_commitment}")

    unclear = expl.get("what_is_unclear", [])
    unclear = [str(x).strip() for x in unclear if str(x).strip() and str(x).strip().lower() != FALLBACK.lower()]
    if unclear:
        pieces.append("What is still unclear: " + "; ".join(unclear))

    if not pieces:
        return FALLBACK

    return " ".join(pieces)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--top_docs", type=int, default=20)
    parser.add_argument("--top_passages", type=int, default=8)
    args = parser.parse_args()

    # 1. Retrieve evidence
    run_cmd([
        sys.executable, "scripts/retrieve_grounding_evidence.py",
        "--question", args.question,
        "--top_docs", str(args.top_docs),
        "--top_passages", str(args.top_passages),
    ])

    # 2. Guarded eligibility
    run_cmd([
        sys.executable, "scripts/generate_guarded_eligibility_v3.py",
        "--question", args.question,
        "--model_name", args.model_name,
    ])

    # 3. Strict consent explanation
    run_cmd([
        sys.executable, "scripts/generate_consent_explanation_strict.py",
        "--question", QUESTION_EXPLAIN,
        "--model_name", args.model_name,
    ])

    # 4. Postprocess strict explanation
    run_cmd([
        sys.executable, "scripts/postprocess_consent_explanation_strict.py",
    ])

    # 5. Teach-back questions
    run_cmd([
        sys.executable, "scripts/generate_teachback_questions.py",
        "--model_name", args.model_name,
    ])

    # 6. Load outputs
    eligibility = load_json(OUT / "guarded_eligibility_v3.json")
    explanation = load_json(OUT / "consent_explanation_strict_postprocessed.json")
    teachback = load_json(OUT / "teachback_questions.json")

    # 7. Rebuild safe summary
    rebuilt_summary = rebuild_patient_summary(explanation)
    explanation["patient_facing_summary_rebuilt"] = {
        "text": rebuilt_summary,
        "source": "rebuilt_from_postprocessed_fields"
    }

    final = {
        "input_question": args.question,
        "model_name": args.model_name,
        "evidence_file": str(OUT / "grounding_evidence_top_passages.csv"),
        "eligibility_assessment": eligibility,
        "consent_explanation": explanation,
        "teachback": teachback,
    }

    final_path = OUT / "onboarding_pipeline_output.json"
    with open(final_path, "w") as f:
        json.dump(final, f, indent=2)

    print("\nSaved:", final_path)
    print("\nRebuilt patient-facing summary:\n")
    print(rebuilt_summary)


if __name__ == "__main__":
    main()
