from pathlib import Path
import argparse
import json
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "tables"

QUESTION_EXPLAIN = "Can you explain in simple words what this study is about, who may join, what participation seems to involve, and what is still unclear?"
FALLBACK = "not clearly stated in the retrieved evidence"

RETRIEVE_SCRIPT = ROOT / "scripts" / "retrieve_grounding_evidence.py"
ENFORCE_SCRIPT = ROOT / "scripts" / "enforce_single_trial_context.py"

GEN_ELIG = ROOT / "scripts" / "generate_guarded_eligibility_v3.py"
GEN_CONSENT = ROOT / "scripts" / "generate_consent_explanation_strict.py"
POST_CONSENT = ROOT / "scripts" / "postprocess_consent_explanation_strict.py"
POST_UNCLEAR = ROOT / "scripts" / "postprocess_unclear_items.py"
GEN_TEACH = ROOT / "scripts" / "generate_teachback_questions_targeted.py"

RAW_EVIDENCE = OUT / "grounding_evidence_top_passages_raw.csv"
MAIN_EVIDENCE = OUT / "grounding_evidence_top_passages.csv"
FILTERED_EVIDENCE = OUT / "grounding_evidence_single_trial.csv"
STATUS_JSON = OUT / "single_trial_context_status.json"


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


def write_abstention_outputs(question, model_name, context_status):
    selected_doc = context_status.get("selected_doc_id", "")
    share = context_status.get("selected_doc_weight_share", 0.0)

    elig = {
        "decision": "cannot_determine",
        "patient_facing_answer": "I found multiple partially related studies and cannot reliably answer from a single trial context. Please provide a specific study title or NCT number.",
        "supported_patient_facts": [],
        "missing_patient_facts": [
            "the exact study you want to ask about"
        ],
        "unresolved_study_requirements": [
            "a single dominant trial context could not be established from the retrieved evidence"
        ],
        "reasoning_summary": f"Retrieved evidence was split across multiple trials. The top candidate trial was {selected_doc} with weight share {share:.3f}, which was not strong enough for a reliable single-trial answer.",
        "safety_note": "Please provide the study title, trial name, or NCT number so the system can answer from one specific protocol."
    }

    consent = {
        "study_purpose": {"text": FALLBACK, "support_passages": []},
        "who_may_join": {"text": FALLBACK, "support_passages": []},
        "procedures_or_participation": {"text": FALLBACK, "support_passages": []},
        "risks_or_burdens": {"text": FALLBACK, "support_passages": []},
        "time_commitment": {"text": FALLBACK, "support_passages": []},
        "what_is_unclear": [
            "A single study context could not be established from the retrieved evidence.",
            "The system needs one specific trial before it can explain study details safely."
        ],
    }

    rebuilt_summary = "I found multiple partially related studies and cannot safely summarize one trial without a more specific study identifier."

    teach = {
        "teachback_questions": [
            "Can you share the study title or NCT number you want to ask about?",
            "Do you want help with one specific trial rather than a general study search?"
        ],
        "checked_concepts": [
            "single-trial identification",
            "study disambiguation"
        ]
    }

    final = {
        "input_question": question,
        "model_name": model_name,
        "context_control": context_status,
        "evidence_file": str(FILTERED_EVIDENCE if FILTERED_EVIDENCE.exists() else MAIN_EVIDENCE),
        "eligibility_assessment": elig,
        "consent_explanation": {
            **consent,
            "patient_facing_summary_rebuilt": {
                "text": rebuilt_summary,
                "source": "single_trial_abstention"
            }
        },
        "teachback": teach,
    }

    with open(OUT / "guarded_eligibility_v3.json", "w") as f:
        json.dump(elig, f, indent=2)
    with open(OUT / "consent_explanation_with_unclear.json", "w") as f:
        json.dump(consent, f, indent=2)
    with open(OUT / "teachback_questions_targeted.json", "w") as f:
        json.dump(teach, f, indent=2)
    with open(OUT / "onboarding_pipeline_output.json", "w") as f:
        json.dump(final, f, indent=2)

    print("Saved abstention outputs.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--top_docs", type=int, default=20)
    parser.add_argument("--top_passages", type=int, default=8)
    parser.add_argument("--min_weight_share", type=float, default=0.55)
    parser.add_argument("--min_passage_count", type=int, default=3)
    parser.add_argument("--keep_top_passages", type=int, default=6)
    args = parser.parse_args()

    # 1. Retrieve mixed context as before
    run_cmd([
        sys.executable, str(RETRIEVE_SCRIPT),
        "--question", args.question,
        "--top_docs", str(args.top_docs),
        "--top_passages", str(args.top_passages),
    ])

    # 2. Enforce single-trial context
    run_cmd([
        sys.executable, str(ENFORCE_SCRIPT),
        "--min_weight_share", str(args.min_weight_share),
        "--min_passage_count", str(args.min_passage_count),
        "--keep_top_passages", str(args.keep_top_passages),
    ])

    context_status = load_json(STATUS_JSON)

    # 3. If too mixed, abstain early
    if context_status["status"] == "abstain_mixed_context":
        write_abstention_outputs(args.question, args.model_name, context_status)
        return

    # 4. Replace main evidence file with filtered single-trial file for downstream generators
    if RAW_EVIDENCE.exists():
        shutil.copy2(MAIN_EVIDENCE, RAW_EVIDENCE)
    shutil.copy2(FILTERED_EVIDENCE, MAIN_EVIDENCE)

    # 5. Run your existing generation stack on the single-trial evidence only
    run_cmd([
        sys.executable, str(GEN_ELIG),
        "--question", args.question,
        "--model_name", args.model_name,
    ])

    run_cmd([
        sys.executable, str(GEN_CONSENT),
        "--question", QUESTION_EXPLAIN,
        "--model_name", args.model_name,
    ])

    run_cmd([sys.executable, str(POST_CONSENT)])
    run_cmd([sys.executable, str(POST_UNCLEAR)])
    run_cmd([sys.executable, str(GEN_TEACH)])

    # 6. Assemble final output
    eligibility = load_json(OUT / "guarded_eligibility_v3.json")
    explanation = load_json(OUT / "consent_explanation_with_unclear.json")
    teachback = load_json(OUT / "teachback_questions_targeted.json")

    rebuilt_summary = rebuild_patient_summary(explanation)
    explanation["patient_facing_summary_rebuilt"] = {
        "text": rebuilt_summary,
        "source": "rebuilt_from_postprocessed_fields"
    }

    final = {
        "input_question": args.question,
        "model_name": args.model_name,
        "context_control": context_status,
        "evidence_file": str(MAIN_EVIDENCE),
        "eligibility_assessment": eligibility,
        "consent_explanation": explanation,
        "teachback": teachback,
    }

    with open(OUT / "onboarding_pipeline_output.json", "w") as f:
        json.dump(final, f, indent=2)

    print("\nSaved:", OUT / "onboarding_pipeline_output.json")
    print("\nSingle-trial selected doc:", context_status.get("selected_doc_id"))
    print("Weight share:", context_status.get("selected_doc_weight_share"))


if __name__ == "__main__":
    main()
