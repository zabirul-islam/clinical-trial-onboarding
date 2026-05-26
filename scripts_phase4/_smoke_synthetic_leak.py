"""
Smoke test: 3 hand-crafted synthetic-leak cases.

Goal: verify the semantic-leak judge actually flags real cross-trial
content leakage, not just outputs 0 every time. Each synthetic case is
designed so that:
  - the selected trial t* is real,
  - the candidate pool contains a different real trial t' (also a real NCT),
  - the answer is hand-written to PARAPHRASE distinctive content of t'
    while pretending to be about t* (no NCT mentioned, no distinctive
    title tokens — would escape both lexical detectors).

Pass criterion: each case is flagged semantic_leak=1 with suspect_doc==t'
by at least one judge. We run only Sonnet here for speed.

Cost: 3 calls ≈ $0.05.
"""
from __future__ import annotations
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from semantic_leak_judge import (  # noqa: E402
    GenRecord, SonnetJudge, build_prompt, parse_judge_json,
)

# ──────────────────────────────────────────────────────────────
# Three synthetic cases — selected trial t*, candidate pool with t',
# and an answer that *paraphrases* t' content while ostensibly
# answering about t*.
# ──────────────────────────────────────────────────────────────
CASES = [
    {
        "case_id": "smoke_synth_01",
        "selected_doc": "NCT00543023",
        "selected_passages":
            "[eligibility] Inclusion: postmenopausal women age 55-85 with "
            "documented vertebral osteoporotic fracture. BMD T-score < -2.5. "
            "Exclusion: prior bisphosphonate therapy within 12 months.",
        "candidates": [
            {
                "doc_id": "NCT01655537",
                "passages": [
                    "Phase 3 randomized double-blind study of weekly "
                    "subcutaneous abaloparatide 80 mcg vs. placebo for "
                    "24 months in women with postmenopausal osteoporosis. "
                    "Primary endpoint: new vertebral fracture incidence at "
                    "18 months.",
                    "Pharmacokinetics: peak serum 0.5-1h post-injection; "
                    "half-life 1.7 hours.",
                ],
            },
            {
                "doc_id": "NCT04123512",
                "passages": [
                    "Sponsor: Boehringer Ingelheim. Open-label single-arm "
                    "registry of romosozumab in Japanese postmenopausal "
                    "women over 60 with prior fragility fracture.",
                ],
            },
        ],
        # Leak: paraphrases NCT01655537 (abaloparatide details) inside an
        # answer claiming to be about NCT00543023.
        "answer_blob":
            "Yes, you may be a fit for this study. The trial randomizes "
            "postmenopausal women to receive a weekly subcutaneous "
            "anabolic agent at an 80-microgram dose versus placebo over a "
            "24-month treatment period, and the primary outcome is new "
            "vertebral fractures measured at the 18-month visit. The drug "
            "reaches peak serum levels within an hour and has a short "
            "half-life of under two hours, so injections are scheduled "
            "weekly. Visits will include BMD scans of the spine.",
    },
    {
        "case_id": "smoke_synth_02",
        "selected_doc": "NCT01482130",
        "selected_passages":
            "[eligibility] Adults with type 2 diabetes, HbA1c 7.0-10.0%, "
            "on stable metformin for >= 3 months. Intervention: oral "
            "semaglutide 14 mg daily vs. placebo for 26 weeks. Primary: "
            "change in HbA1c from baseline.",
        "candidates": [
            {
                "doc_id": "NCT03548935",
                "passages": [
                    "Phase 3 multicenter trial of intravenous tirzepatide "
                    "infusions every 4 weeks for 52 weeks in adults with "
                    "type 2 diabetes and BMI > 30. Primary endpoint: "
                    "weight loss of >= 5% at week 52. Secondary: change in "
                    "fasting plasma glucose and lipid panel.",
                ],
            },
            {
                "doc_id": "NCT04567321",
                "passages": [
                    "Cohort study of empagliflozin 10 mg in adults with "
                    "heart failure and preserved ejection fraction.",
                ],
            },
        ],
        # Leak: paraphrases NCT03548935 (tirzepatide IV q4w, weight-loss
        # endpoint) but pretends to describe NCT01482130 (semaglutide oral).
        "answer_blob":
            "If you're eligible, you would receive an intravenous infusion "
            "every four weeks for one year. The primary outcome is at least "
            "five percent body-weight loss measured at week 52, and the "
            "study also tracks fasting glucose and a full lipid panel as "
            "secondary outcomes. You'll need to come in monthly for the "
            "infusion and quarterly for labs.",
    },
    {
        "case_id": "smoke_synth_03",
        "selected_doc": "NCT00865358",
        "selected_passages":
            "[eligibility] Patients with previously untreated stage III/IV "
            "Hodgkin lymphoma. Intervention: ABVD x 6 cycles. Primary: "
            "5-year overall survival.",
        "candidates": [
            {
                "doc_id": "NCT02181738",
                "passages": [
                    "Brentuximab vedotin + nivolumab combination in "
                    "relapsed Hodgkin lymphoma. Dosing: brentuximab 1.8 "
                    "mg/kg + nivolumab 3 mg/kg every 21 days for up to "
                    "16 cycles. Primary endpoint: objective response rate "
                    "by PET-CT.",
                ],
            },
            {
                "doc_id": "NCT01777152",
                "passages": [
                    "Observational registry of patients with Hodgkin "
                    "lymphoma in pediatric age (under 18).",
                ],
            },
        ],
        # Leak: paraphrases NCT02181738 (brentuximab+nivolumab) — those
        # drugs are NOT in NCT00865358 (ABVD) — but answer presents them
        # as the regimen.
        "answer_blob":
            "Treatment in this trial pairs an anti-CD30 antibody-drug "
            "conjugate at 1.8 mg per kilogram with a PD-1 inhibitor at "
            "3 mg per kilogram, given together every three weeks for as "
            "many as sixteen cycles. The main goal is the proportion of "
            "patients whose disease shrinks on PET-CT imaging.",
    },
]


async def main():
    judge = SonnetJudge(min_interval=2.0)
    print(f"Running {len(CASES)} synthetic-leak cases through Sonnet …")
    flagged = 0
    for c in CASES:
        rec = GenRecord(
            case_id=c["case_id"],
            backbone="synthetic",
            selected_doc=c["selected_doc"],
            answer_blob=c["answer_blob"],
            selected_passages=c["selected_passages"],
            raw_path="(synthetic)",
        )
        prompt = build_prompt(rec, c["candidates"])
        out = await judge.score(prompt)
        flag = int(out.get("semantic_leak") or 0)
        flagged += flag
        suspect = out.get("suspect_doc")
        excerpt = (out.get("claim_excerpt") or "")[:120]
        print(f"\n[{c['case_id']}]  t*={c['selected_doc']}  "
              f"flag={flag}  suspect={suspect}")
        print(f"  excerpt: {excerpt}")
        print(f"  rationale: {out.get('rationale')}")

    print("\n──────────────────────────────────────────────")
    expected = len(CASES)
    print(f"flagged {flagged}/{expected} synthetic-leak cases")
    print("PASS" if flagged == expected else "FAIL — detector missed leaks")
    return 0 if flagged == expected else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
