from pathlib import Path
import argparse
import json
import shutil
import subprocess
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "tables"

QUESTION_EXPLAIN = "Can you explain in simple words what this study is about, who may join, what participation seems to involve, and what is still unclear?"
FALLBACK = "not clearly stated in the retrieved evidence"

RETRIEVE_SCRIPT = ROOT / "scripts" / "retrieve_grounding_evidence.py"
TRIAL_SCORE_SCRIPT = ROOT / "scripts" / "score_trials_from_passages.py"

GEN_ELIG = ROOT / "scripts" / "generate_guarded_eligibility_v3.py"
GEN_CONSENT = ROOT / "scripts" / "generate_consent_explanation_strict.py"
POST_CONSENT = ROOT / "scripts" / "postprocess_consent_explanation_strict.py"
POST_UNCLEAR = ROOT / "scripts" / "postprocess_unclear_items.py"
GEN_TEACH = ROOT / "scripts" / "generate_teachback_questions_targeted.py"

MAIN_EVIDENCE = OUT / "grounding_evidence_top_passages.csv"
RAW_EVIDENCE = OUT / "grounding_evidence_top_passages_raw.csv"
TRIAL_SCORES = OUT / "trial_level_scores.csv"
TRIAL_CONTEXT = OUT / "grounding_evidence_trial_first.csv"
STATUS_JSON = OUT / "trial_first_context_status.json"


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
    for label, key in [
        ("Study purpose", "study_purpose"),
        ("Who may join", "who_may_join"),
        ("What participation may involve", "procedures_or_participation"),
        ("Risks or burdens", "risks_or_burdens"),
        ("Time commitment", "time_commitment"),
    ]:
        txt = safe_text(expl.get(key))
        if txt != FALLBACK:
            pieces.append(f"{label}: {txt}")

    unclear = expl.get("what_is_unclear", [])
    unclear = [str(x).strip() for x in unclear if str(x).strip() and str(x).strip().lower() != FALLBACK.lower()]
    if unclear:
        pieces.append("What is still unclear: " + "; ".join(unclear))

    return " ".join(pieces) if pieces else FALLBACK


def is_generic_study_reference(question: str) -> bool:
    q = question.lower()
    generic_phrases = [
        "this study",
        "joined this study",
        "join this study",
        "if i joined this study",
    ]
    anchor_terms = [
        "nct",
        "osteoporosis",
        "vertebral",
        "fracture",
        "lynch",
        "brain injury",
        "breast",
        "hepatitis",
        "melanoma",
    ]
    has_generic = any(p in q for p in generic_phrases)
    has_anchor = any(t in q for t in anchor_terms)
    return has_generic and not has_anchor


def write_abstention(question, model_name, status):
    elig = {
        "decision": "cannot_determine",
        "patient_facing_answer": "I could not identify one reliable study from the retrieved evidence. Please provide a study title or NCT number so I can answer from a single trial context.",
        "supported_patient_facts": [],
        "missing_patient_facts": ["the exact study you want to ask about"],
        "unresolved_study_requirements": ["a reliable dominant single-trial context could not be established"],
        "reasoning_summary": (
            f"Trial selection was not reliable enough for safe answering. "
            f"Top trial: {status.get('selected_doc_id', '')}, "
            f"dominance ratio: {status.get('dominance_ratio', 0.0):.3f}, "
            f"score share: {status.get('trial_score_share', 0.0):.3f}, "
            f"raw max cross-score: {status.get('raw_max_cross', 0.0):.3f}."
        ),
        "safety_note": "Please provide a study title, NCT number, or direct protocol reference."
    }

    expl = {
        "study_purpose": {"text": FALLBACK, "support_passages": []},
        "who_may_join": {"text": FALLBACK, "support_passages": []},
        "procedures_or_participation": {"text": FALLBACK, "support_passages": []},
        "risks_or_burdens": {"text": FALLBACK, "support_passages": []},
        "time_commitment": {"text": FALLBACK, "support_passages": []},
        "what_is_unclear": [
            "A reliable dominant single-trial context could not be established from the retrieved evidence.",
            "The system needs one specific trial before it can answer safely."
        ],
        "patient_facing_summary_rebuilt": {
            "text": "I could not identify one reliable study from the retrieved evidence, so I cannot safely summarize a single trial yet.",
            "source": "trial_first_abstention"
        }
    }

    teach = {
        "teachback_questions": [
            "Can you share the study title or NCT number you want to ask about?",
            "Would you like help narrowing this to one specific trial?"
        ],
        "checked_concepts": ["single-trial identification", "study disambiguation"]
    }

    final = {
        "input_question": question,
        "model_name": model_name,
        "context_control": status,
        "evidence_file": str(TRIAL_CONTEXT if TRIAL_CONTEXT.exists() else MAIN_EVIDENCE),
        "eligibility_assessment": elig,
        "consent_explanation": expl,
        "teachback": teach,
    }

    with open(OUT / "guarded_eligibility_v3.json", "w") as f:
        json.dump(elig, f, indent=2)
    with open(OUT / "consent_explanation_with_unclear.json", "w") as f:
        json.dump(expl, f, indent=2)
    with open(OUT / "teachback_questions_targeted.json", "w") as f:
        json.dump(teach, f, indent=2)
    with open(OUT / "onboarding_pipeline_output.json", "w") as f:
        json.dump(final, f, indent=2)

    print("Saved abstention outputs.")


def hard_gate_fields(expl, allowed_passages):
    allowed = set(allowed_passages)
    for key in ["study_purpose", "who_may_join", "procedures_or_participation", "risks_or_burdens", "time_commitment"]:
        obj = expl.get(key, {})
        supports = obj.get("support_passages", [])
        if not supports:
            expl[key] = {"text": FALLBACK, "support_passages": []}
            continue
        if any(p not in allowed for p in supports):
            expl[key] = {"text": FALLBACK, "support_passages": []}
    return expl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--top_docs", type=int, default=20)
    parser.add_argument("--top_passages", type=int, default=8)
    parser.add_argument("--keep_trial_passages", type=int, default=6)
    parser.add_argument("--min_dominance_ratio", type=float, default=1.35)
    parser.add_argument("--min_trial_score_share", type=float, default=0.28)
    parser.add_argument("--min_selected_raw_max_cross", type=float, default=1.0)
    parser.add_argument("--max_best_rank", type=int, default=2)
    args = parser.parse_args()

    run_cmd([
        sys.executable, str(RETRIEVE_SCRIPT),
        "--question", args.question,
        "--top_docs", str(args.top_docs),
        "--top_passages", str(args.top_passages),
    ])

    if MAIN_EVIDENCE.exists():
        shutil.copy2(MAIN_EVIDENCE, RAW_EVIDENCE)

    run_cmd([sys.executable, str(TRIAL_SCORE_SCRIPT)])

    trial_df = pd.read_csv(TRIAL_SCORES)
    raw_df = pd.read_csv(RAW_EVIDENCE)

    top = trial_df.iloc[0]
    second_score = float(trial_df.iloc[1]["trial_score"]) if len(trial_df) > 1 else 1e-6
    total_score = float(trial_df["trial_score"].sum())
    top_score = float(top["trial_score"])
    dominance_ratio = top_score / max(second_score, 1e-6)
    score_share = top_score / max(total_score, 1e-6)

    selected_doc = str(top["doc_id"])
    selected_df = raw_df[raw_df["doc_id"] == selected_doc].sort_values("passage_rank").head(args.keep_trial_passages)
    selected_df.to_csv(TRIAL_CONTEXT, index=False)

    raw_max_cross = float(top.get("raw_max_cross", 0.0))
    best_rank = int(top.get("best_rank", 999))
    generic_question = is_generic_study_reference(args.question)

    status = {
        "status": "trial_first_ok",
        "selected_doc_id": selected_doc,
        "selected_trial_score": top_score,
        "selected_doc_passage_count": int(top["n_passages"]),
        "dominance_ratio": round(dominance_ratio, 6),
        "trial_score_share": round(score_share, 6),
        "raw_max_cross": round(raw_max_cross, 6),
        "best_rank": best_rank,
        "generic_question": generic_question,
        "n_unique_docs_in_raw_context": int(raw_df["doc_id"].nunique()),
        "raw_context_rows": int(len(raw_df)),
        "filtered_context_rows": int(len(selected_df)),
        "thresholds": {
            "min_dominance_ratio": args.min_dominance_ratio,
            "min_trial_score_share": args.min_trial_score_share,
            "min_selected_raw_max_cross": args.min_selected_raw_max_cross,
            "max_best_rank": args.max_best_rank,
            "keep_trial_passages": args.keep_trial_passages,
        }
    }

    accept = (
        dominance_ratio >= args.min_dominance_ratio and
        score_share >= args.min_trial_score_share and
        raw_max_cross >= args.min_selected_raw_max_cross and
        best_rank <= args.max_best_rank and
        not generic_question
    )

    if not accept:
        status["status"] = "abstain_no_reliable_trial"
        with open(STATUS_JSON, "w") as f:
            json.dump(status, f, indent=2)
        write_abstention(args.question, args.model_name, status)
        return

    with open(STATUS_JSON, "w") as f:
        json.dump(status, f, indent=2)

    shutil.copy2(TRIAL_CONTEXT, MAIN_EVIDENCE)

    run_cmd([sys.executable, str(GEN_ELIG), "--question", args.question, "--model_name", args.model_name])
    run_cmd([sys.executable, str(GEN_CONSENT), "--question", QUESTION_EXPLAIN, "--model_name", args.model_name])
    run_cmd([sys.executable, str(POST_CONSENT)])
    run_cmd([sys.executable, str(POST_UNCLEAR)])
    run_cmd([sys.executable, str(GEN_TEACH)])

    elig = load_json(OUT / "guarded_eligibility_v3.json")
    expl = load_json(OUT / "consent_explanation_with_unclear.json")
    teach = load_json(OUT / "teachback_questions_targeted.json")

    allowed_passages = list(range(1, len(selected_df) + 1))
    expl = hard_gate_fields(expl, allowed_passages)
    expl["patient_facing_summary_rebuilt"] = {
        "text": rebuild_patient_summary(expl),
        "source": "rebuilt_from_trial_first_gated_fields"
    }

    final = {
        "input_question": args.question,
        "model_name": args.model_name,
        "context_control": status,
        "evidence_file": str(MAIN_EVIDENCE),
        "eligibility_assessment": elig,
        "consent_explanation": expl,
        "teachback": teach,
    }

    with open(OUT / "consent_explanation_with_unclear.json", "w") as f:
        json.dump(expl, f, indent=2)
    with open(OUT / "onboarding_pipeline_output.json", "w") as f:
        json.dump(final, f, indent=2)

    print("Saved:", OUT / "onboarding_pipeline_output.json")


if __name__ == "__main__":
    main()
