from pathlib import Path
import json
import re
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "tables"

INPUT_JSON = OUT / "consent_explanation_strict.json"
EVIDENCE_CSV = OUT / "grounding_evidence_top_passages.csv"
OUTPUT_JSON = OUT / "consent_explanation_strict_postprocessed.json"

FALLBACK = "not clearly stated in the retrieved evidence"

# Field-specific lexical anchors.
# If a field is populated with non-fallback text, at least one support passage
# should contain some of these anchors.
FIELD_KEYWORDS = {
    "study_purpose": [
        "purpose", "aim", "goal", "record the effect", "study is", "hypothesis"
    ],
    "who_may_join": [
        "inclusion", "eligible", "women", "men", "age", "postmenopausal",
        "osteoporosis", "fracture", "ambulatory", "consent"
    ],
    "procedures_or_participation": [
        "session", "exercise", "course", "programme", "program", "undergo",
        "participation", "questionnaire", "measurement", "visit", "treatment"
    ],
    "risks_or_burdens": [
        "risk", "risks", "burden", "burdens", "adverse", "side effect",
        "complication", "pain", "discomfort", "harm", "unsafe"
    ],
    "time_commitment": [
        "week", "weeks", "month", "months", "year", "years", "hour", "hours",
        "session", "follow-up", "follow up", "visit", "visits", "3-month",
        "three-month", "one year", "after one year"
    ],
    "patient_facing_summary": []
}

# Some phrases are especially suspicious if they appear without explicit evidence.
SUSPICIOUS_PATTERNS = {
    "risks_or_burdens": [
        r"side effects?",
        r"discomfort",
        r"complication",
        r"pain",
    ],
    "time_commitment": [
        r"follow-?up visits?",
        r"approximately \d+",
        r"may be necessary",
    ],
    "procedures_or_participation": [
        r"vertebroplasty",
    ],
}


def normalize(text: str) -> str:
    return " ".join(str(text).lower().split())


def load_support_texts(evidence_df: pd.DataFrame, support_passages):
    subset = evidence_df[evidence_df["passage_rank"].isin(support_passages)].copy()
    return [normalize(x) for x in subset["passage_text"].tolist()]


def any_keyword_present(texts, keywords):
    if not keywords:
        return True
    merged = " ".join(texts)
    return any(k in merged for k in keywords)


def suspicious_claim_present(field_name: str, field_text: str) -> bool:
    pats = SUSPICIOUS_PATTERNS.get(field_name, [])
    t = normalize(field_text)
    for pat in pats:
        if re.search(pat, t):
            return True
    return False


def support_mentions_suspicious_claim(field_name: str, support_texts) -> bool:
    pats = SUSPICIOUS_PATTERNS.get(field_name, [])
    if not pats:
        return True
    merged = " ".join(support_texts)
    for pat in pats:
        if re.search(pat, merged):
            return True
    return False


def fallback_object():
    return {"text": FALLBACK, "support_passages": []}


def main():
    if not INPUT_JSON.exists():
        print(f"Missing input JSON: {INPUT_JSON}")
        sys.exit(1)
    if not EVIDENCE_CSV.exists():
        print(f"Missing evidence CSV: {EVIDENCE_CSV}")
        sys.exit(1)

    with open(INPUT_JSON, "r") as f:
        data = json.load(f)
    evidence = pd.read_csv(EVIDENCE_CSV)

    object_fields = [
        "study_purpose",
        "who_may_join",
        "procedures_or_participation",
        "risks_or_burdens",
        "time_commitment",
        "patient_facing_summary",
    ]

    changes = []

    for field in object_fields:
        if field not in data or not isinstance(data[field], dict):
            data[field] = fallback_object()
            changes.append(f"{field}: missing or invalid object -> fallback")
            continue

        text = str(data[field].get("text", "")).strip()
        supports = data[field].get("support_passages", [])

        if text == FALLBACK:
            data[field]["support_passages"] = []
            continue

        if not isinstance(supports, list) or len(supports) == 0:
            data[field] = fallback_object()
            changes.append(f"{field}: no usable support_passages -> fallback")
            continue

        # Keep only integer passage ranks that exist.
        valid_supports = []
        valid_ranks = set(evidence["passage_rank"].astype(int).tolist())
        for s in supports:
            if isinstance(s, int) and s in valid_ranks:
                valid_supports.append(s)

        if len(valid_supports) == 0:
            data[field] = fallback_object()
            changes.append(f"{field}: support_passages not found in evidence -> fallback")
            continue

        support_texts = load_support_texts(evidence, valid_supports)

        # Rule 1: field-specific lexical anchor check
        keywords = FIELD_KEYWORDS.get(field, [])
        if not any_keyword_present(support_texts, keywords):
            data[field] = fallback_object()
            changes.append(f"{field}: support passages lack field-specific evidence -> fallback")
            continue

        # Rule 2: suspicious claims must be explicitly supported
        if suspicious_claim_present(field, text) and not support_mentions_suspicious_claim(field, support_texts):
            data[field] = fallback_object()
            changes.append(f"{field}: suspicious claim not explicitly supported -> fallback")
            continue

        # If it passed, keep cleaned supports only.
        data[field]["support_passages"] = valid_supports

    # Clean what_is_unclear
    if "what_is_unclear" not in data or not isinstance(data["what_is_unclear"], list) or len(data["what_is_unclear"]) == 0:
        data["what_is_unclear"] = [FALLBACK]
        changes.append("what_is_unclear: missing/empty -> fallback list")
    else:
        cleaned = []
        for x in data["what_is_unclear"]:
            s = str(x).strip()
            if s:
                cleaned.append(s)
        if len(cleaned) == 0:
            cleaned = [FALLBACK]
        data["what_is_unclear"] = cleaned

    with open(OUTPUT_JSON, "w") as f:
        json.dump(data, f, indent=2)

    print("Saved:", OUTPUT_JSON)
    print("\nPost-processing changes:")
    if changes:
        for c in changes:
            print("-", c)
    else:
        print("No changes made.")


if __name__ == "__main__":
    main()
