from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "tables"

orig_path = OUT / "consent_explanation_strict.json"
post_path = OUT / "consent_explanation_strict_postprocessed.json"

with open(orig_path, "r") as f:
    orig = json.load(f)

with open(post_path, "r") as f:
    post = json.load(f)

fields = [
    "study_purpose",
    "who_may_join",
    "procedures_or_participation",
    "risks_or_burdens",
    "time_commitment",
    "patient_facing_summary",
]

for field in fields:
    print("=" * 80)
    print(field)
    print("- ORIGINAL")
    print(orig.get(field))
    print("- POSTPROCESSED")
    print(post.get(field))

print("=" * 80)
print("what_is_unclear")
print("- ORIGINAL")
print(orig.get("what_is_unclear"))
print("- POSTPROCESSED")
print(post.get("what_is_unclear"))
