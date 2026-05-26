from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "tables"

ELIG_PATH = OUT / "guarded_eligibility_v3.json"
CONSENT_PATH = OUT / "consent_explanation_strict_postprocessed.json"
OUTPUT_PATH = OUT / "consent_explanation_with_unclear.json"


def normalize_sentence(s: str) -> str:
    s = " ".join(str(s).strip().split())
    if not s:
        return s
    return s[0].upper() + s[1:]


def main():
    with open(ELIG_PATH, "r") as f:
        elig = json.load(f)

    with open(CONSENT_PATH, "r") as f:
        consent = json.load(f)

    unclear = []

    for x in elig.get("missing_patient_facts", []):
        x = normalize_sentence(x)
        if x:
            unclear.append(x)

    for x in elig.get("unresolved_study_requirements", []):
        x = normalize_sentence(x)
        if x:
            unclear.append(x)

    # remove duplicates while keeping order
    seen = set()
    cleaned = []
    for x in unclear:
        key = x.lower()
        if key not in seen:
            cleaned.append(x)
            seen.add(key)

    if not cleaned:
        cleaned = ["Not clearly stated in the retrieved evidence"]

    consent["what_is_unclear"] = cleaned

    with open(OUTPUT_PATH, "w") as f:
        json.dump(consent, f, indent=2)

    print("Saved:", OUTPUT_PATH)
    print("\nUpdated what_is_unclear:")
    for x in cleaned:
        print("-", x)


if __name__ == "__main__":
    main()
