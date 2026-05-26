from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "tables"

ELIG_PATH = OUT / "guarded_eligibility_v3.json"
CONSENT_PATH = OUT / "consent_explanation_with_unclear.json"
OUTPUT_PATH = OUT / "teachback_questions_targeted.json"


def to_question_from_missing(item: str) -> str:
    s = item.strip().rstrip(".")
    s_low = s.lower()

    if "x-ray" in s_low or "documented" in s_low or "fracture" in s_low:
        return "Can you explain what kind of fracture documentation this study still seems to require?"

    if "blood coagulation" in s_low or "hematologic" in s_low:
        return "Can you tell me what blood or clotting-related checks may still need to be confirmed?"

    if "severe or chronic disabling conditions" in s_low:
        return "Can you explain what health conditions might still need to be checked before joining this study?"

    if s_low.startswith("your "):
        return "Can you tell me what part of your medical history or current condition still needs to be confirmed?"

    return "Can you explain what information still needs to be confirmed before joining this study?"


def main():
    with open(ELIG_PATH, "r") as f:
        elig = json.load(f)

    with open(CONSENT_PATH, "r") as f:
        consent = json.load(f)

    questions = []
    checked_concepts = []

    # 1. one question about the basic join profile
    if elig.get("supported_patient_facts"):
        questions.append("Based on what we discussed, who seems to be the kind of person this study is looking for?")
        checked_concepts.append("basic join profile")

    # 2. target concrete missing facts
    for item in elig.get("missing_patient_facts", [])[:2]:
        questions.append(to_question_from_missing(item))
        checked_concepts.append(item)

    # 3. target unresolved study-side requirement
    unresolved = elig.get("unresolved_study_requirements", [])
    if unresolved:
        questions.append("Can you explain what the study team would still need to verify before confirming eligibility?")
        checked_concepts.append("unresolved study requirements")

    # deduplicate while preserving order
    seen = set()
    q_clean = []
    c_clean = []
    for q, c in zip(questions, checked_concepts):
        key = q.lower()
        if key not in seen:
            q_clean.append(q)
            c_clean.append(c)
            seen.add(key)

    out = {
        "teachback_questions": q_clean,
        "checked_concepts": c_clean,
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(out, f, indent=2)

    print("Saved:", OUTPUT_PATH)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
