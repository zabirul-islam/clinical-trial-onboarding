"""
T1.3 — Label each backbone-ablation case with a therapeutic area.

Reads:
  outputs/backbone_gens/<backbone_slug>/<case_id>.json   (all 30 distinct case ids)

For each unique case_id:
  - Extract patient question + selected_doc + first passage text (trial context).
  - Ask Sonnet to classify into one of:
      MSK_Bone, Cardiovascular, Metabolic_Endocrine, Oncology, Neurology_CNS, Other
  - Persist {case_id, source, category, area, area_confidence, rationale, prompt_input}.

Writes:
  outputs/phase4/area_breakdown/area_labels_30case.csv
  outputs/phase4/area_breakdown/area_labels_30case.jsonl   (full responses)

Usage:
  export ANTHROPIC_API_KEY=...
  python scripts_phase4/label_therapeutic_areas.py \\
      --gens-dir outputs/backbone_gens/Qwen__Qwen2.5-3B-Instruct \\
      --out-csv outputs/phase4/area_breakdown/area_labels_30case.csv \\
      --out-jsonl outputs/phase4/area_breakdown/area_labels_30case.jsonl

Cost: 30 calls × ~$0.01-0.02 each = ~$0.30-$0.60.
Resumable: skips case_ids already in the JSONL.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

try:
    import anthropic
except ImportError:
    anthropic = None


AREAS = [
    "MSK_Bone",
    "Cardiovascular",
    "Metabolic_Endocrine",
    "Oncology",
    "Neurology_CNS",
    "Other",
]

SYSTEM_PROMPT = (
    "You are classifying clinical-trial onboarding cases by therapeutic area. "
    "Read the patient's question and the selected trial context. Pick exactly "
    "one therapeutic area from the allowed list. Be strict; if the case crosses "
    "multiple areas or is too generic to classify, pick 'Other'. Return only a "
    "JSON object — no prose, no markdown."
)

USER_TEMPLATE = """\
## Allowed therapeutic areas
{areas}

## Definitions
- MSK_Bone: musculoskeletal, bone, fracture, osteoporosis, joint, orthopedic
- Cardiovascular: heart, vascular, hypertension, atrial, coronary, stroke prevention (cardiogenic)
- Metabolic_Endocrine: diabetes, thyroid, lipid disorder, metabolic syndrome, hormone-related
- Oncology: any cancer / tumor / chemotherapy / radiation oncology / oncology follow-up
- Neurology_CNS: stroke (CNS), seizure, Parkinson, Alzheimer, MS, ALS, sleep, psychiatric
- Other: anything not fitting the above five (e.g., rare disease, dental, dermatology, generic study questions)

## Patient question
{question}

## Selected trial id
{selected_doc}

## Trial context (first passage)
{passage_text}

## Output (return only this JSON, nothing else)
{{
  "area": "<one of the allowed labels>",
  "confidence": <integer 1-5; 5 = certain>,
  "rationale": "<one sentence>"
}}
"""


def _load_case_questions(gens_dir: Path) -> list[dict]:
    """Read all backbone-gen JSONs in a single backbone dir; one row per case."""
    rows = []
    for p in sorted(gens_dir.glob("*.json")):
        with p.open() as f:
            d = json.load(f)
        first_passage_text = ""
        passages = d.get("passages") or []
        if passages:
            first_passage_text = str(passages[0].get("text", ""))[:1500]
        rows.append(
            {
                "case_id": d.get("case_id"),
                "source": d.get("source"),
                "category": d.get("category"),
                "question": d.get("question", ""),
                "selected_doc": d.get("selected_doc", ""),
                "passage_text": first_passage_text,
            }
        )
    return rows


def _load_already_done(jsonl_path: Path) -> set[str]:
    if not jsonl_path.exists():
        return set()
    done = set()
    with jsonl_path.open() as f:
        for line in f:
            try:
                d = json.loads(line)
                if "case_id" in d and "area" in d:
                    done.add(d["case_id"])
            except json.JSONDecodeError:
                continue
    return done


def _call_sonnet(client, model: str, prompt: str, max_retries: int = 4) -> dict:
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=400,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            text = resp.content[0].text.strip()
            # Try to parse as JSON. Strip code fences if any.
            if text.startswith("```"):
                text = text.strip("`")
                if text.lower().startswith("json"):
                    text = text[4:].strip()
            return json.loads(text)
        except Exception as e:  # noqa: BLE001
            last_err = e
            sleep_s = 2 ** attempt
            print(
                f"[retry] attempt {attempt + 1}/{max_retries} after {sleep_s}s: {e}",
                file=sys.stderr,
            )
            time.sleep(sleep_s)
    raise RuntimeError(f"sonnet call failed after {max_retries} retries: {last_err}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--gens-dir",
        type=str,
        default="outputs/backbone_gens/Qwen__Qwen2.5-3B-Instruct",
        help="Single-backbone gen dir (one JSON per unique case).",
    )
    ap.add_argument(
        "--out-csv",
        type=str,
        default="outputs/phase4/area_breakdown/area_labels_30case.csv",
    )
    ap.add_argument(
        "--out-jsonl",
        type=str,
        default="outputs/phase4/area_breakdown/area_labels_30case.jsonl",
    )
    ap.add_argument(
        "--model",
        type=str,
        default="claude-sonnet-4-5-20250929",
    )
    ap.add_argument(
        "--min-interval",
        type=float,
        default=2.0,
        help="Seconds between API calls (rate gate).",
    )
    args = ap.parse_args()

    if anthropic is None:
        print("ERROR: anthropic SDK not installed", file=sys.stderr)
        return 2
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set", file=sys.stderr)
        return 2

    gens_dir = Path(args.gens_dir).resolve()
    if not gens_dir.exists():
        print(f"ERROR: gens-dir not found: {gens_dir}", file=sys.stderr)
        return 2

    cases = _load_case_questions(gens_dir)
    print(f"[load] {len(cases)} cases from {gens_dir}")

    out_csv = Path(args.out_csv)
    out_jsonl = Path(args.out_jsonl)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    already_done = _load_already_done(out_jsonl)
    print(f"[resume] {len(already_done)} cases already labeled")

    client = anthropic.Anthropic()
    last_call_ts = 0.0

    rows = []
    # If the JSONL has prior runs, load them as the starting point so the final
    # CSV is complete.
    if out_jsonl.exists():
        with out_jsonl.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    new_count = 0
    for case in cases:
        if case["case_id"] in already_done:
            continue
        prompt = USER_TEMPLATE.format(
            areas=", ".join(AREAS),
            question=case["question"] or "(no question text)",
            selected_doc=case["selected_doc"] or "(no selected trial)",
            passage_text=case["passage_text"] or "(no passage text)",
        )
        # rate gate
        elapsed = time.monotonic() - last_call_ts
        if elapsed < args.min_interval:
            time.sleep(args.min_interval - elapsed)
        try:
            parsed = _call_sonnet(client, args.model, prompt)
        except Exception as e:  # noqa: BLE001
            print(f"[fail] {case['case_id']}: {e}", file=sys.stderr)
            continue
        last_call_ts = time.monotonic()

        area = str(parsed.get("area", "")).strip()
        if area not in AREAS:
            print(
                f"[warn] {case['case_id']}: model returned non-canonical area {area!r}; coercing to Other"
            )
            area = "Other"

        record = {
            "case_id": case["case_id"],
            "source": case["source"],
            "category": case["category"],
            "selected_doc": case["selected_doc"],
            "question": case["question"],
            "area": area,
            "confidence": int(parsed.get("confidence", 0)),
            "rationale": str(parsed.get("rationale", "")),
        }
        with out_jsonl.open("a") as f:
            f.write(json.dumps(record) + "\n")
        rows.append(record)
        new_count += 1
        print(
            f"[ok] {case['case_id']:>20s}  →  {area:<22s}  conf={record['confidence']}"
        )

    print(f"\n[done] new={new_count}, total={len(rows)}")

    # Write CSV (drop full question text; keep concise version)
    import csv
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "case_id",
                "source",
                "category",
                "selected_doc",
                "area",
                "confidence",
                "rationale",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r["case_id"],
                    r["source"],
                    r["category"],
                    r["selected_doc"],
                    r["area"],
                    r["confidence"],
                    r["rationale"],
                ]
            )
    print(f"[wrote] {out_csv}")

    # Print area distribution
    from collections import Counter
    dist = Counter(r["area"] for r in rows)
    print("\n=== area distribution ===")
    for area in AREAS:
        n = dist.get(area, 0)
        bar = "#" * n
        print(f"{area:<22s} {n:>3d} {bar}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
