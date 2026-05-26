from pathlib import Path
import json
import csv

ROOT = Path(__file__).resolve().parents[1]
EVAL_ROOT = ROOT / "outputs" / "eval_runs_v2"
INDEX_PATH = EVAL_ROOT / "eval_case_index.json"
OUT_CSV = EVAL_ROOT / "onboarding_eval_summary_v2.csv"
OUT_JSON = EVAL_ROOT / "onboarding_eval_summary_v2.json"

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def safe_join(items):
    if not items:
        return ""
    return " ; ".join(str(x) for x in items)

def main():
    with open(INDEX_PATH, "r") as f:
        index = json.load(f)

    rows = []
    detailed = []

    for rec in index:
        case_dir = Path(rec["case_dir"])
        pipe = load_json(case_dir / "onboarding_pipeline_output.json")

        context = pipe.get("context_control", {})
        elig = pipe.get("eligibility_assessment", {})
        expl = pipe.get("consent_explanation", {})
        teach = pipe.get("teachback", {})

        row = {
            "case_id": rec["case_id"],
            "category": rec["category"],
            "question": rec["question"],
            "context_status": context.get("status", ""),
            "selected_doc_id": context.get("selected_doc_id", ""),
            "selected_doc_weight_share": context.get("selected_doc_weight_share", ""),
            "decision": elig.get("decision", ""),
            "patient_facing_answer": elig.get("patient_facing_answer", ""),
            "supported_patient_facts": safe_join(elig.get("supported_patient_facts", [])),
            "missing_patient_facts": safe_join(elig.get("missing_patient_facts", [])),
            "unresolved_study_requirements": safe_join(elig.get("unresolved_study_requirements", [])),
            "study_purpose": expl.get("study_purpose", {}).get("text", ""),
            "who_may_join": expl.get("who_may_join", {}).get("text", ""),
            "procedures_or_participation": expl.get("procedures_or_participation", {}).get("text", ""),
            "risks_or_burdens": expl.get("risks_or_burdens", {}).get("text", ""),
            "time_commitment": expl.get("time_commitment", {}).get("text", ""),
            "what_is_unclear": safe_join(expl.get("what_is_unclear", [])),
            "rebuilt_summary": expl.get("patient_facing_summary_rebuilt", {}).get("text", ""),
            "teachback_questions": safe_join(teach.get("teachback_questions", [])),
        }
        rows.append(row)

        detailed.append({
            "case_id": rec["case_id"],
            "category": rec["category"],
            "question": rec["question"],
            "notes": rec.get("notes", ""),
            "context_control": context,
            "eligibility_assessment": elig,
            "consent_explanation": expl,
            "teachback": teach,
        })

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with open(OUT_JSON, "w") as f:
        json.dump(detailed, f, indent=2)

    print("Saved:", OUT_CSV)
    print("Saved:", OUT_JSON)

if __name__ == "__main__":
    main()
