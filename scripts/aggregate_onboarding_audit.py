from pathlib import Path
import csv
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
AUDIT_PATH = ROOT / "data" / "processed" / "onboarding_eval_audit_template_v2.csv"
OUT_PATH = ROOT / "outputs" / "eval_runs" / "onboarding_eval_audit_summary.txt"

def pct(n, d):
    return 0.0 if d == 0 else 100.0 * n / d

def main():
    with open(AUDIT_PATH, newline="") as f:
        rows = list(csv.DictReader(f))

    total = len(rows)

    def count_eq(field, value):
        return sum(1 for r in rows if r[field].strip().lower() == value)

    def count_in(field, values):
        vals = {v.lower() for v in values}
        return sum(1 for r in rows if r[field].strip().lower() in vals)

    summary = []
    summary.append(f"Total cases: {total}")

    metrics = [
        ("Topic relevance = yes", "topic_relevance", ["yes"]),
        ("Topic relevance = no", "topic_relevance", ["no"]),
        ("Eligibility overstatement = none", "eligibility_overstatement", ["none"]),
        ("Eligibility overstatement = severe", "eligibility_overstatement", ["severe"]),
        ("Unsupported explanation = none", "unsupported_explanation", ["none"]),
        ("Unsupported explanation = severe", "unsupported_explanation", ["severe"]),
        ("Fallback used correctly = yes", "fallback_used_correctly", ["yes"]),
        ("Missing facts reasonable = yes", "missing_facts_reasonable", ["yes"]),
        ("Unresolved requirements reasonable = yes", "unresolved_requirements_reasonable", ["yes"]),
        ("Teach-back targeted = yes", "teachback_targeted", ["yes"]),
        ("Overall usable = yes", "overall_usable", ["yes"]),
        ("Needs domain expert review = yes", "needs_domain_expert_review", ["yes"]),
        ("Needs domain expert review = maybe", "needs_domain_expert_review", ["maybe"]),
    ]

    for label, field, vals in metrics:
        n = count_in(field, vals)
        summary.append(f"{label}: {n}/{total} ({pct(n,total):.1f}%)")

    # also dump raw counters
    summary.append("\nRaw category counts:")
    for field in [
        "topic_relevance",
        "eligibility_overstatement",
        "unsupported_explanation",
        "fallback_used_correctly",
        "missing_facts_reasonable",
        "unresolved_requirements_reasonable",
        "teachback_targeted",
        "patient_facing_clarity",
        "overall_usable",
        "needs_domain_expert_review",
    ]:
        c = Counter(r[field].strip().lower() for r in rows if r[field].strip())
        summary.append(f"{field}: {dict(c)}")

    out_text = "\n".join(summary)
    OUT_PATH.write_text(out_text)
    print(out_text)
    print("\nSaved:", OUT_PATH)

if __name__ == "__main__":
    main()
