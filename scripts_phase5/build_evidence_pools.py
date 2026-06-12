#!/usr/bin/env python3
"""
Phase 5 Task 3 (Step 3.1) — snapshot per-case reranked multi-trial evidence pools.

For each of the 114 benchmark cases, runs the SAME retrieval the deployed
pipeline uses (scripts/retrieve_grounding_evidence.py: BM25 -> cross-encoder)
and snapshots the resulting multi-trial passage pool to
outputs/phase5/evidence_pools/<case_id>.csv.

Retrieval is backbone-independent, so this runs ONCE and feeds all
4 baseline configs x 2 backbones.

Run (server):
  conda activate avatar_trial
  python scripts_phase5/build_evidence_pools.py --device cuda
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
POOL_DIR = ROOT / "outputs" / "phase5" / "evidence_pools"
SCRATCH = ROOT / "outputs" / "tables" / "grounding_evidence_top_passages.csv"

GEN_ROOTS = [
    ROOT / "outputs" / "backbone_gens" / "Qwen__Qwen2.5-7B-Instruct",
    ROOT / "outputs" / "phase4" / "n100_expansion" / "gens" / "Qwen__Qwen2.5-7B-Instruct",
]


def benchmark_case_ids() -> list[str]:
    """The exact 114 case_ids of the audited benchmark = files in the gens dirs."""
    ids: set[str] = set()
    for d in GEN_ROOTS:
        ids.update(p.stem for p in d.glob("*.json"))
    return sorted(ids)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=Path,
                    default=ROOT / "data/processed/onboarding_eval_cases_expanded.json")
    ap.add_argument("--top-docs", type=int, default=20)
    ap.add_argument("--top-passages", type=int, default=8)
    ap.add_argument("--device", default="cuda")  # passed through env to retrieval script
    ap.add_argument("--resume", action="store_true", default=True)
    args = ap.parse_args()

    with open(args.cases) as f:
        cases = {str(c["case_id"]): c for c in json.load(f)}
    ids = benchmark_case_ids()
    print(f"[info] {len(ids)} benchmark case_ids; {len(cases)} in manifest")
    missing = [i for i in ids if i not in cases]
    if missing:
        print(f"[FATAL] {len(missing)} case_ids not in manifest: {missing[:5]}")
        return 1

    POOL_DIR.mkdir(parents=True, exist_ok=True)
    retrieve = ROOT / "scripts" / "retrieve_grounding_evidence.py"

    done = 0
    for cid in ids:
        out_csv = POOL_DIR / f"{cid}.csv"
        if args.resume and out_csv.exists() and out_csv.stat().st_size > 0:
            done += 1
            continue
        q = str(cases[cid]["question"])
        subprocess.check_call(
            [sys.executable, str(retrieve),
             "--question", q,
             "--top_docs", str(args.top_docs),
             "--top_passages", str(args.top_passages)],
            cwd=str(ROOT),
            stdout=subprocess.DEVNULL,
        )
        shutil.copyfile(SCRATCH, out_csv)
        done += 1
        print(f"[{done}/{len(ids)}] {cid}")
    print(f"[done] pools in {POOL_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
