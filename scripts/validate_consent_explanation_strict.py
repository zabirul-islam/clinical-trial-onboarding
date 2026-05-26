from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "tables"

JSON_PATH = OUT / "consent_explanation_strict.json"

REQUIRED_OBJECT_FIELDS = [
    "study_purpose",
    "who_may_join",
    "procedures_or_participation",
    "risks_or_burdens",
    "time_commitment",
    "patient_facing_summary",
]
REQUIRED_LIST_FIELD = "what_is_unclear"
FALLBACK_TEXT = "not clearly stated in the retrieved evidence"


def main():
    if not JSON_PATH.exists():
        print(f"Missing file: {JSON_PATH}")
        sys.exit(1)

    with open(JSON_PATH, "r") as f:
        data = json.load(f)

    errors = []

    for field in REQUIRED_OBJECT_FIELDS:
        if field not in data:
            errors.append(f"Missing field: {field}")
            continue

        val = data[field]
        if not isinstance(val, dict):
            errors.append(f"{field} must be an object")
            continue

        if "text" not in val or "support_passages" not in val:
            errors.append(f"{field} must contain text and support_passages")
            continue

        text = str(val["text"]).strip()
        supports = val["support_passages"]

        if not isinstance(supports, list):
            errors.append(f"{field}.support_passages must be a list")
            continue

        if text == FALLBACK_TEXT and len(supports) != 0:
            errors.append(f"{field} uses fallback text but has non-empty support_passages")

        if text != FALLBACK_TEXT and len(supports) == 0:
            errors.append(f"{field} has non-fallback text but empty support_passages")

        for p in supports:
            if not isinstance(p, int):
                errors.append(f"{field}.support_passages must contain integers only")
                break

    if REQUIRED_LIST_FIELD not in data:
        errors.append(f"Missing field: {REQUIRED_LIST_FIELD}")
    elif not isinstance(data[REQUIRED_LIST_FIELD], list):
        errors.append(f"{REQUIRED_LIST_FIELD} must be a list")

    if errors:
        print("VALIDATION FAILED")
        for e in errors:
            print("-", e)
        sys.exit(1)

    print("VALIDATION PASSED")


if __name__ == "__main__":
    main()
