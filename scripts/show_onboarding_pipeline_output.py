from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "tables"
PATH = OUT / "onboarding_pipeline_output.json"

with open(PATH, "r") as f:
    data = json.load(f)

print("=" * 80)
print("INPUT QUESTION")
print(data["input_question"])

print("\n" + "=" * 80)
print("ELIGIBILITY DECISION")
print(data["eligibility_assessment"]["decision"])
print(data["eligibility_assessment"]["patient_facing_answer"])

print("\nSUPPORTED FACTS")
for x in data["eligibility_assessment"].get("supported_patient_facts", []):
    print("-", x)

print("\nMISSING FACTS")
for x in data["eligibility_assessment"].get("missing_patient_facts", []):
    print("-", x)

print("\nUNRESOLVED STUDY REQUIREMENTS")
for x in data["eligibility_assessment"].get("unresolved_study_requirements", []):
    print("-", x)

print("\n" + "=" * 80)
print("CONSENT EXPLANATION (POSTPROCESSED)")
for field in ["study_purpose", "who_may_join", "procedures_or_participation", "risks_or_burdens", "time_commitment"]:
    print(f"\n{field}:")
    print(data["consent_explanation"][field]["text"])
    print("support:", data["consent_explanation"][field]["support_passages"])

print("\nWHAT IS UNCLEAR")
for x in data["consent_explanation"].get("what_is_unclear", []):
    print("-", x)

print("\nREBUILT PATIENT-FACING SUMMARY")
print(data["consent_explanation"]["patient_facing_summary_rebuilt"]["text"])

print("\n" + "=" * 80)
print("TEACH-BACK QUESTIONS")
for q in data["teachback"].get("teachback_questions", []):
    print("-", q)
