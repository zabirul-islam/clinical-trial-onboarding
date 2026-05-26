"""
Adapter — convert 15-case audit per-variant bundles into backbone-gen-style
single JSONs that scripts_phase3/run_llm_judge.py can consume.

INPUT (server-relative paths, run from repo root):
  outputs/eval_runs/case_NN/      (V1 unconstrained)
  outputs/eval_runs_v2/case_NN/   (V2 strict abstention)
  outputs/eval_runs_final/case_NN/(V-final trial-first guarded)

OUTPUT:
  outputs/phase4/15case_audit/15case_audit_gens/V1/case_NN.json
  outputs/phase4/15case_audit/15case_audit_gens/V2/case_NN.json
  outputs/phase4/15case_audit/15case_audit_gens/V-final/case_NN.json

Each output JSON matches the backbone_gens schema:
    case_id, backbone (= variant tag), source, category, gold_nct,
    selected_doc, question, evidence, passages, raw_generation,
    parsed, parse_ok, decision, patient_facing_answer,
    missing_patient_facts, unresolved_study_requirements,
    cross_trial_leak_n, latency_sec

Usage (from server repo root):
    python scripts_phase4/build_15case_audit_gens.py \\
        --repo-root . \\
        --out outputs/phase4/15case_audit/15case_audit_gens

Notes:
- The "backbone" field is repurposed as a variant tag {V1, V2, V-final}
  so run_llm_judge.py treats variants as separate "models" and produces
  per-variant scores. Judge prompt is blinded so no identity leaks.
- The judge sees: question + evidence passages + raw_generation. We
  bundle eligibility + consent + teach-back into the raw_generation
  string in a clear patient-facing format.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Optional

import pandas as pd

VARIANTS = {
    "V1": "eval_runs",
    "V2": "eval_runs_v2",
    "V-final": "eval_runs_final",
}

# All 15 audit cases. Categories from paper §4.2.
CASE_CATEGORIES: dict[str, str] = {
    "case_01": "possible_match_insufficient_evidence",  # anchor case
    "case_02": "possible_match_insufficient_evidence",
    "case_03": "likely_mismatch_or_unlikely_match",
    "case_04": "cannot_determine",
    "case_05": "possible_match_insufficient_evidence",
    "case_06": "possible_match_insufficient_evidence",
    "case_07": "explanation_request",
    "case_08": "explanation_request",
    "case_09": "participation_request",
    "case_10": "participation_request",
    "case_11": "risk_request",
    "case_12": "consent_understanding",
    "case_13": "consent_understanding",
    "case_14": "consent_understanding",
    "case_15": "cannot_determine",
}


def _safe_load_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        with path.open() as f:
            return json.load(f)
    except json.JSONDecodeError:
        print(f"[warn] failed to decode {path}", file=sys.stderr)
        return None


def _load_grounding_passages(case_dir: Path) -> tuple[list[dict], str]:
    """Return (passages list, evidence text block) from the grounding CSV.

    Prefer trial_first / single_trial filtered file when present (V-final / V2);
    fall back to top_passages (V1).
    """
    candidates = [
        case_dir / "grounding_evidence_trial_first.csv",
        case_dir / "grounding_evidence_single_trial.csv",
        case_dir / "grounding_evidence_top_passages.csv",
        case_dir / "grounding_evidence_top_passages_raw.csv",
    ]
    for p in candidates:
        if p.exists():
            try:
                df = pd.read_csv(p)
                break
            except Exception as e:
                print(f"[warn] csv read failed for {p}: {e}", file=sys.stderr)
                continue
    else:
        return [], ""

    passages: list[dict] = []
    evidence_lines: list[str] = []
    for i, row in df.iterrows():
        passage_id = row.get("passage_id") or f"row_{i}"
        doc_id = row.get("doc_id") or row.get("nct_id") or ""
        section = row.get("section") or row.get("field") or ""
        text = (
            row.get("text")
            or row.get("passage_text")
            or row.get("passage")
            or ""
        )
        score = row.get("rerank_score") or row.get("cross_score") or row.get("score") or None
        passages.append(
            {
                "passage_id": str(passage_id),
                "doc_id": str(doc_id),
                "section": str(section),
                "text": str(text),
                "score": float(score) if score is not None and not pd.isna(score) else None,
            }
        )
        evidence_lines.append(
            f"[Passage {i + 1} | doc={doc_id} | section={section}]\n{str(text).strip()}"
        )
    return passages, "\n\n".join(evidence_lines)


def _bundle_raw_generation(pipeline_output: dict) -> str:
    """Compose the patient-facing response text the judge will score."""
    parts: list[str] = []

    elig = pipeline_output.get("eligibility_assessment") or {}
    if elig:
        parts.append("=== Eligibility ===")
        parts.append(f"Decision: {elig.get('decision', '')}")
        parts.append(f"Patient-facing answer: {elig.get('patient_facing_answer', '')}")
        if elig.get("supported_patient_facts"):
            parts.append(
                "Supported facts: "
                + "; ".join(elig["supported_patient_facts"])
            )
        if elig.get("missing_patient_facts"):
            parts.append(
                "Missing patient facts: "
                + "; ".join(elig["missing_patient_facts"])
            )
        if elig.get("unresolved_study_requirements"):
            parts.append(
                "Unresolved study requirements: "
                + "; ".join(elig["unresolved_study_requirements"])
            )
        if elig.get("reasoning_summary"):
            parts.append(f"Reasoning: {elig['reasoning_summary']}")

    consent = pipeline_output.get("consent_explanation") or {}
    if consent:
        parts.append("\n=== Consent-style explanation ===")
        for field in (
            "study_purpose",
            "who_may_join",
            "procedures_or_participation",
            "risks_or_burdens",
            "time_commitment",
        ):
            f = consent.get(field) or {}
            text = f.get("text", "") if isinstance(f, dict) else str(f)
            cites = f.get("support_passages", []) if isinstance(f, dict) else []
            cite_str = (
                f" (supports: {','.join(str(c) for c in cites)})" if cites else ""
            )
            parts.append(f"{field}: {text}{cite_str}")
        if consent.get("what_is_unclear"):
            parts.append(
                "What is unclear: " + "; ".join(consent["what_is_unclear"])
            )
        rebuilt = consent.get("patient_facing_summary_rebuilt") or {}
        if rebuilt.get("text"):
            parts.append(f"\nPatient-facing summary (rebuilt): {rebuilt['text']}")

    teach = pipeline_output.get("teachback") or {}
    if teach.get("teachback_questions"):
        parts.append("\n=== Teach-back questions ===")
        for q in teach["teachback_questions"]:
            parts.append(f"- {q}")

    return "\n".join(parts).strip()


_NCT_RE = re.compile(r"NCT0\d{7}")


def _count_cross_trial_leaks(text: str, selected_doc: str) -> int:
    if not text:
        return 0
    mentioned = set(_NCT_RE.findall(text))
    mentioned.discard(selected_doc)
    mentioned.discard("")
    return len(mentioned)


def _selected_doc_for(pipeline_output: dict, ctx_status_path: Optional[Path]) -> str:
    """Best-effort selected-trial extraction across V1/V2/V-final."""
    ctx = pipeline_output.get("context_control") or {}
    if ctx.get("selected_doc_id"):
        return str(ctx["selected_doc_id"])
    if ctx_status_path and ctx_status_path.exists():
        st = _safe_load_json(ctx_status_path) or {}
        if st.get("selected_doc_id"):
            return str(st["selected_doc_id"])
    # Fall back: first doc id in passages list
    for p in pipeline_output.get("_passages_fallback", []):
        if p.get("doc_id"):
            return str(p["doc_id"])
    return ""


def _build_record(
    case_id: str,
    variant: str,
    case_dir: Path,
    pipeline_output: dict,
) -> dict[str, Any]:
    ctx_status_path = case_dir / (
        "trial_first_context_status.json"
        if variant == "V-final"
        else "single_trial_context_status.json"
    )
    passages, evidence = _load_grounding_passages(case_dir)
    pipeline_output_copy = dict(pipeline_output)
    pipeline_output_copy["_passages_fallback"] = passages
    selected_doc = _selected_doc_for(pipeline_output_copy, ctx_status_path)

    elig = pipeline_output.get("eligibility_assessment") or {}
    raw_generation = _bundle_raw_generation(pipeline_output)

    return {
        "case_id": case_id,
        "backbone": variant,  # repurposed: variant tag for the dual-judge
        "source": "handcrafted",
        "category": CASE_CATEGORIES.get(case_id, "unknown"),
        "gold_nct": "",
        "selected_doc": selected_doc,
        "question": pipeline_output.get("input_question", ""),
        "evidence": evidence,
        "passages": passages,
        "raw_generation": raw_generation,
        "parsed": elig,
        "parse_ok": bool(elig),
        "decision": elig.get("decision", ""),
        "patient_facing_answer": elig.get("patient_facing_answer", ""),
        "missing_patient_facts": elig.get("missing_patient_facts", []) or [],
        "unresolved_study_requirements": elig.get("unresolved_study_requirements", [])
        or [],
        "cross_trial_leak_n": _count_cross_trial_leaks(raw_generation, selected_doc),
        "latency_sec": 0.0,  # unknown for these legacy runs
        "context_control": pipeline_output.get("context_control"),
        "model_name": pipeline_output.get("model_name", ""),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--repo-root",
        type=str,
        default=".",
        help="Repo root containing outputs/eval_runs*/.",
    )
    ap.add_argument(
        "--out",
        type=str,
        default="outputs/phase4/15case_audit/15case_audit_gens",
        help="Output dir; subdirs V1/V2/V-final will be created.",
    )
    ap.add_argument(
        "--cases",
        nargs="+",
        default=[f"case_{i:02d}" for i in range(1, 16)],
        help="Case ids to convert.",
    )
    args = ap.parse_args()

    repo_root = Path(args.repo_root).resolve()
    out_root = (repo_root / args.out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    summary: dict[str, dict[str, int]] = {}
    for variant, src_dir_name in VARIANTS.items():
        out_dir = out_root / variant
        out_dir.mkdir(parents=True, exist_ok=True)
        wrote = 0
        skipped = 0
        for case_id in args.cases:
            case_dir = repo_root / "outputs" / src_dir_name / case_id
            if not case_dir.exists():
                print(f"[skip] {variant} {case_id}: missing {case_dir}")
                skipped += 1
                continue
            po = _safe_load_json(case_dir / "onboarding_pipeline_output.json")
            if po is None:
                print(
                    f"[skip] {variant} {case_id}: no onboarding_pipeline_output.json"
                )
                skipped += 1
                continue
            record = _build_record(case_id, variant, case_dir, po)
            out_path = out_dir / f"{case_id}.json"
            with out_path.open("w") as f:
                json.dump(record, f, indent=2)
            wrote += 1
        summary[variant] = {"wrote": wrote, "skipped": skipped}

    print("\n=== build_15case_audit_gens summary ===")
    for v, s in summary.items():
        print(f"{v}: wrote={s['wrote']}, skipped={s['skipped']}")
    total = sum(s["wrote"] for s in summary.values())
    print(f"total written: {total}")
    if total != 45:
        print(
            f"[warn] expected 45 (3 variants × 15 cases), got {total}. Check skips."
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
