#!/usr/bin/env python3
"""
Phase 5 Task 3 (Step 3.1) — snapshot per-case reranked multi-trial evidence pools.

For each of the 114 benchmark cases, runs the SAME retrieval the deployed
pipeline uses (BM25 full-text -> cross-encoder rerank, identical to
scripts/retrieve_grounding_evidence.py) and snapshots the multi-trial passage
pool to outputs/phase5/evidence_pools/<case_id>.csv.

In-process: corpus, BM25, and the cross-encoder are loaded ONCE and reused
across all 114 cases. This is algorithm-identical to the canonical per-case
script (same BM25Okapi, same CROSS_ENCODER_MODEL, same sort keys, same
top_docs/top_passages) — only the one-time setup is amortized, so outputs match
while runtime drops from hours to minutes.

Retrieval is backbone-independent: runs ONCE, feeds all 4 baselines x 2 backbones.

Run (server):
  conda activate avatar_trial
  python scripts_phase5/build_evidence_pools.py
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
POOL_DIR = ROOT / "outputs" / "phase5" / "evidence_pools"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # same as canonical script

GEN_ROOTS = [
    ROOT / "outputs" / "backbone_gens" / "Qwen__Qwen2.5-7B-Instruct",
    ROOT / "outputs" / "phase4" / "n100_expansion" / "gens" / "Qwen__Qwen2.5-7B-Instruct",
]


def tokenize(text: str):
    return re.findall(r"[a-z0-9]+", str(text).lower())


def benchmark_case_ids() -> list[str]:
    ids: set[str] = set()
    for d in GEN_ROOTS:
        ids.update(p.stem for p in d.glob("*.json"))
    return sorted(ids)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=Path,
                    default=DATA / "onboarding_eval_cases_expanded.json")
    ap.add_argument("--top-docs", type=int, default=20)
    ap.add_argument("--top-passages", type=int, default=8)
    ap.add_argument("--resume", action="store_true", default=False,
                    help="skip cases whose pool csv already exists")
    args = ap.parse_args()

    with open(args.cases) as f:
        cases = {str(c["case_id"]): c for c in json.load(f)}
    ids = benchmark_case_ids()
    missing = [i for i in ids if i not in cases]
    if missing:
        print(f"[FATAL] {len(missing)} case_ids not in manifest: {missing[:5]}")
        return 1
    print(f"[info] {len(ids)} benchmark cases; loading corpus + cross-encoder once")

    import torch
    from rank_bm25 import BM25Okapi
    from sentence_transformers import CrossEncoder

    corpus = pd.read_parquet(DATA / "retrieval_corpus.parquet")
    passages = pd.read_parquet(DATA / "trial_evidence_passages.parquet")
    doc_ids = corpus["doc_id"].astype(str).tolist()
    print(f"[info] BM25 over {len(doc_ids)} docs (one-time tokenization)")
    bm25 = BM25Okapi([tokenize(t) for t in corpus["text_all"].astype(str).tolist()])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CrossEncoder(CROSS_ENCODER_MODEL, device=device)
    print(f"[info] cross-encoder on {device}")

    passages = passages.copy()
    passages["doc_id_str"] = passages["doc_id"].astype(str)

    POOL_DIR.mkdir(parents=True, exist_ok=True)
    for n, cid in enumerate(ids, 1):
        out_csv = POOL_DIR / f"{cid}.csv"
        if args.resume and out_csv.exists() and out_csv.stat().st_size > 0:
            continue
        q = str(cases[cid]["question"])
        scores = bm25.get_scores(tokenize(q))
        ranked = sorted(zip(doc_ids, scores), key=lambda x: x[1], reverse=True)[: args.top_docs]
        rank_map = {d: r for r, (d, _) in enumerate(ranked, 1)}
        score_map = {d: float(s) for d, s in ranked}
        top_ids = set(rank_map)

        cand = passages[passages["doc_id_str"].isin(top_ids)].copy()
        cand["doc_rank_bm25"] = cand["doc_id_str"].map(rank_map)
        cand["doc_score_bm25"] = cand["doc_id_str"].map(score_map)
        pairs = list(zip([q] * len(cand), cand["passage_text"].astype(str).tolist()))
        cand["cross_score"] = model.predict(pairs, batch_size=32, show_progress_bar=False)
        cand = cand.sort_values(["cross_score", "doc_score_bm25"],
                                ascending=[False, False]).reset_index(drop=True)
        cand["passage_rank"] = range(1, len(cand) + 1)
        keep = [c for c in ["passage_id", "doc_id", "section", "passage_text", "title",
                            "condition", "doc_rank_bm25", "doc_score_bm25", "cross_score",
                            "passage_rank"] if c in cand.columns]
        cand.head(args.top_passages)[keep].to_csv(out_csv, index=False)
        if n % 10 == 0 or n == len(ids):
            print(f"[{n}/{len(ids)}] {cid}", flush=True)
    print(f"[done] pools in {POOL_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
