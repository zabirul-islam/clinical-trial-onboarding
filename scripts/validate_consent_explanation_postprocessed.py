from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "tables"

JSON_PATH = OUT / "consent_explanation_strict_postprocessed.json"
FALLBACK = "not clearly stated in the retrieved evidence"

REQUIRED_OBJECT_FIELDS = [
    "study_purpose",
    "who_may_join",
    "procedures_or_participation",
    "risks_or_burdens",
    "time_commitment",
    "patient_facing_summary",
]


def main():
    if not JSON_PATH.exists():
        print(f"Missing file: {JSON_PATH}")
        sys.exit(1)

    with open(JSON_PATH, "r") as f:
        data = json.load(f)

    errors = []

    for field in REQUIRED_OBJECT_FIELDS:
        if field not in data or not isinstance(data[field], dict):
            errors.append(f"{field}: missing or invalid object")
            continue

        text = data[field].get("text", None)
        supports = data[field].get("support_passages", None)

        if not isinstance(text, str):
            errors.append(f"{field}: text must be string")
        if not isinstance(supports, list):
            errors.append(f"{field}: support_passages must be list")
            continue

        if text == FALLBACK and len(supports) != 0:
            errors.append(f"{field}: fallback text must have empty support_passages")
        if text != FALLBACK and len(supports) == 0:
            errors.append(f"{field}: non-fallback text must have support_passages")

        for s in supports:
            if not isinstance(s, int):
                errors.append(f"{field}: support_passages must contain integers only")
                break

    if "what_is_unclear" not in data or not isinstance(data["what_is_unclear"], list):
        errors.append("what_is_unclear: must be a list")

    if errors:
        print("VALIDATION FAILED")
        for e in errors:
            print("-", e)
        sys.exit(1)

    print("VALIDATION PASSED")


if __name__ == "__main__":
    main()
