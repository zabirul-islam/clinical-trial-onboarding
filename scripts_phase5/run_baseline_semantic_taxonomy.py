#!/usr/bin/env python3
"""
Phase 5 Task 5 — semantic cross-trial-leak judge + failure taxonomy, on baselines.

Reuses the frozen phase-4 dual-judge machinery (Judge classes, rate gate, retry,
JSON parsing) by import — the canonical scripts_phase4/semantic_leak_judge.py is
left UNTOUCHED so the phase-4 audit stays reproducible. This script:
  * extends the prompt to also return a taxonomy label (T1/T2/T3) when a leak is found
  * builds candidate (non-selected) trial passages from each baseline record's OWN
    embedded `passages` split by `reference_trial` (exactly what the model saw),
    rather than reconstructing the retrieval pool
  * scores B1-B4 baseline gens (and, with --gens phase4, can re-annotate the
    existing 104 either-flagged gens — see --gens-root)

Taxonomy (frozen, configs/phase5_baselines.yaml):
  T1 cross_trial_contamination : claim supported only by a non-selected pool trial
  T2 unsupported_completion    : plausible protocol content supported by NO pool trial
  T3 ordinary_hallucination    : claim false/incoherent, not attributable to any trial

Run (where API keys live; reads synced baseline gens):
  source .api_keys
  python scripts_phase5/run_baseline_semantic_taxonomy.py --max-cases 3 --baselines B1_multi_rag   # smoke
  python scripts_phase5/run_baseline_semantic_taxonomy.py                                          # full
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts_phase4"))
import semantic_leak_judge as slj  # noqa: E402  (frozen phase-4 machinery)

BASELINE_ROOT = REPO / "outputs" / "phase5" / "baseline_gens"
OUT_DIR = REPO / "outputs" / "phase5" / "semantic_judge"

SYSTEM_PROMPT = slj.SYSTEM_PROMPT + (
    " After deciding leakage, also classify the SINGLE most salient factual error "
    "(if any) into a failure taxonomy."
)

USER_TEMPLATE = """\
## Task
Decide whether the Assistant's answer contains any factual claim about a clinical
trial that is supported ONLY by a non-selected candidate trial (NCT id != {selected_doc}),
not by the selected trial t* = {selected_doc}. Paraphrase counts.

Then classify the single most salient factual error, if any, into:
  T1 = cross_trial_contamination : the erroneous claim is actually true of a DIFFERENT
       trial shown below (information fused across trials).
  T2 = unsupported_completion    : plausible protocol-style content supported by NO shown
       trial (the model filled in likely-sounding detail).
  T3 = ordinary_hallucination    : a claim false on its face / incoherent with the case,
       not attributable to any shown trial.
  none = no clinically meaningful factual error.

Generic clinical English that fits any trial is NOT a leak and is "none".

## Selected trial t* = {selected_doc}
{selected_passages}

## Non-selected candidate trials shown to the model ({n_candidates})
{candidate_passages}

## Assistant's answer (fields the patient would see)
{answer_blob}

## Output — JSON only, no markdown fences, no prose
{{
  "semantic_leak": 0 or 1,
  "taxonomy": "T1" | "T2" | "T3" | "none",
  "suspect_doc": "<NCT id of the non-selected trial whose content was leaked, or null>",
  "claim_excerpt": "<<=250 chars verbatim from the answer, or null>",
  "rationale": "<2 sentences: the claim and which (if any) non-selected trial supports it>"
}}
"""

TAX_VALUES = {"T1", "T2", "T3", "none"}


def parse_tax(txt: str, judge: str) -> Dict[str, Any]:
    obj = slj.parse_judge_json(txt, judge)  # validates semantic_leak + base keys
    tax = obj.get("taxonomy")
    if tax not in TAX_VALUES:
        tax = "none" if obj.get("semantic_leak", 0) == 0 else "T1"
    obj["taxonomy"] = tax
    return obj


def candidates_from_record(rec: dict) -> List[Dict[str, Any]]:
    """Non-selected candidate passages = record's own passages, minus reference trial."""
    ref = str(rec.get("reference_trial") or rec.get("selected_doc") or "")
    by_doc: Dict[str, List[str]] = {}
    for p in (rec.get("passages") or []):
        doc = str(p.get("doc_id", ""))
        if doc and doc != ref:
            by_doc.setdefault(doc, []).append(
                slj._truncate(str(p.get("passage_text", "")), 700))
    return [{"doc_id": d, "passages": ps[:2]} for d, ps in by_doc.items()]


def selected_passages_text(rec: dict) -> str:
    ref = str(rec.get("reference_trial") or rec.get("selected_doc") or "")
    rows = []
    for p in (rec.get("passages") or []):
        if str(p.get("doc_id", "")) == ref:
            rows.append(f"[{p.get('section','')}] "
                        f"{slj._truncate(str(p.get('passage_text','')), 700)}")
    return "\n".join(rows[:6]) or "(empty)"


def build_prompt(rec: dict) -> str:
    cands = candidates_from_record(rec)
    blocks = []
    for c in cands:
        b = f"### Candidate trial {c['doc_id']}\n" + "".join(f"- {t}\n" for t in c["passages"])
        blocks.append(b)
    if not blocks:
        blocks = ["(no non-selected candidates — single-trial evidence)"]
    return USER_TEMPLATE.format(
        selected_doc=rec.get("reference_trial") or rec.get("selected_doc") or "",
        selected_passages=selected_passages_text(rec),
        n_candidates=len(cands),
        candidate_passages="\n".join(blocks),
        answer_blob=slj._answer_blob(rec) or "(empty)",
    )


def discover(baselines: List[str], max_cases) -> List[dict]:
    recs = []
    for b in baselines:
        for bb_dir in sorted((BASELINE_ROOT / b).glob("*")):
            files = sorted(bb_dir.glob("*.json"))
            if max_cases:
                files = files[:max_cases]
            for fp in files:
                d = json.loads(fp.read_text())
                d["_key"] = f"{b}|{bb_dir.name}|{fp.stem}"
                recs.append(d)
    return recs


def load_done(jsonl: Path) -> set:
    if not jsonl.exists():
        return set()
    done = set()
    for line in jsonl.read_text().splitlines():
        if not line.strip():
            continue
        try:
            d = json.loads(line)
        except Exception:
            continue
        if "taxonomy" in d and "_key" in d:
            done.add(d["_key"])
    return done


async def run_judge(judge, recs, out_path):
    done = load_done(out_path)
    todo = [r for r in recs if r["_key"] not in done]
    print(f"[{judge.name}] {len(todo)}/{len(recs)} to score")
    fout = out_path.open("a", encoding="utf-8")
    ok = err = 0
    for r in todo:
        prompt = build_prompt(r)
        try:
            raw = await judge.score(prompt)  # returns dict already parsed by slj
            # slj judges parse with parse_judge_json; re-parse raw text for taxonomy
            obj = raw if "taxonomy" in raw else dict(raw)
            if "taxonomy" not in obj:
                obj["taxonomy"] = "none" if obj.get("semantic_leak", 0) == 0 else "T1"
            line = {"judge": judge.name, "_key": r["_key"],
                    "baseline_id": r.get("baseline_id"), "backbone": r.get("backbone"),
                    "case_id": r.get("case_id"), "reference_trial": r.get("reference_trial"),
                    **{k: obj.get(k) for k in ("semantic_leak", "taxonomy", "suspect_doc",
                                               "claim_excerpt", "rationale")}}
            fout.write(json.dumps(line) + "\n"); fout.flush(); ok += 1
        except Exception as e:
            fout.write(json.dumps({"judge": judge.name, "_key": r["_key"], "error": str(e)}) + "\n")
            fout.flush(); err += 1
            print(f"  [!] {judge.name} {r['_key']}: {e}")
    fout.close()
    print(f"[{judge.name}] done {ok} ok / {err} err")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baselines", nargs="+",
                    default=["B1_multi_rag", "B2_prompt_guard",
                             "B3_citation_enforced", "B4_top1"])
    ap.add_argument("--judges", nargs="+", default=["sonnet", "gpt4o"])
    ap.add_argument("--max-cases", type=int, default=None)
    ap.add_argument("--min-interval", type=float, default=6.0)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # monkeypatch the taxonomy-aware prompt + parser into the imported judges
    slj.SYSTEM_PROMPT = SYSTEM_PROMPT
    slj.parse_judge_json = parse_tax

    recs = discover(args.baselines, args.max_cases)
    print(f"[discover] {len(recs)} baseline (case x backbone x config) records")
    if not recs:
        print("[!] no baseline gens found — run Task 3 first")
        return 2

    for jname in args.judges:
        judge = slj.JUDGE_REGISTRY[jname](min_interval=args.min_interval)
        out_path = OUT_DIR / f"semantic_taxonomy_{jname}.jsonl"
        asyncio.run(run_judge(judge, recs, out_path))
    print("[done]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
