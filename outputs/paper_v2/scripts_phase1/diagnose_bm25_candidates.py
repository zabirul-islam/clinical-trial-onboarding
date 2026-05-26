"""
Phase 1.3 diagnostic — figure out why BGE R@100 = 0.15 when BM25 R@100 = 0.42.

Checks:
  1. Row count / queries / per-query rank distribution
  2. passage_text coverage after merge
  3. doc_id format match with qrels
  4. Theoretical BM25-top100 recall ceiling from qrels (independent of reranker)

Run:
  python scripts/diagnose_bm25_candidates.py --repo-root .
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", type=Path, default=Path("."))
    args = p.parse_args()
    root = args.repo_root.resolve()

    cands = pd.read_csv(root / "outputs/tables/bm25_full_text_top100_candidates.csv")
    qrels = pd.read_csv(root / "data/processed/trec_ct2021_qrels_full.csv")
    passages = pd.read_parquet(root / "data/processed/trial_evidence_passages.parquet")

    print(f"[cands]    rows={len(cands):,}  cols={list(cands.columns)}")
    print(f"[cands]    unique queries = {cands['query_id'].nunique()}")
    print(f"[cands]    per-query rows: min={cands.groupby('query_id').size().min()}, "
          f"max={cands.groupby('query_id').size().max()}, "
          f"mean={cands.groupby('query_id').size().mean():.1f}")
    print(f"[cands]    unique doc_ids = {cands['doc_id'].nunique():,}")
    print(f"[cands]    sample doc_ids: {cands['doc_id'].head(5).tolist()}")

    print(f"\n[qrels]    rows={len(qrels):,}  cols={list(qrels.columns)}")
    print(f"[qrels]    sample doc_ids: {qrels['doc_id'].head(5).tolist()}")
    print(f"[qrels]    doc_ids starting with 'NCT' = "
          f"{qrels['doc_id'].astype(str).str.startswith('NCT').sum():,}")

    cand_set = set(cands['doc_id'].astype(str).unique())
    qrel_set = set(qrels['doc_id'].astype(str).unique())
    print(f"\n[overlap]  cands ∩ qrels doc_ids = {len(cand_set & qrel_set):,}")

    print(f"\n[passages] rows={len(passages):,}  cols={list(passages.columns)}")
    pcols = {c.lower(): c for c in passages.columns}
    text_col = next((pcols[k] for k in ["passage_text","text","trial_text","evidence_text"] if k in pcols), None)
    id_col   = next((pcols[k] for k in ["doc_id","nct_id","trial_id"] if k in pcols), None)
    print(f"[passages] text_col = {text_col!r}, id_col = {id_col!r}")
    print(f"[passages] unique {id_col}s = {passages[id_col].nunique():,}")
    print(f"[passages] sample {id_col}s: {passages[id_col].astype(str).head(5).tolist()}")

    # Passage coverage after merge
    grouped = passages.groupby(id_col)[text_col].apply(lambda s: " ".join(s.astype(str))).reset_index()
    grouped = grouped.rename(columns={id_col: "doc_id", text_col: "passage_text"})
    merged = cands.merge(grouped, on="doc_id", how="left")
    missing = merged["passage_text"].isna().sum()
    empty = (merged["passage_text"].fillna("").str.len() == 0).sum()
    print(f"\n[merge]    cands rows with NaN passage_text = {missing:,} / {len(merged):,}")
    print(f"[merge]    cands rows with empty passage_text = {empty:,} / {len(merged):,}")
    avg_len = merged["passage_text"].fillna("").str.len().mean()
    print(f"[merge]    avg passage_text length (chars) = {avg_len:.0f}")

    # Theoretical R@100 ceiling from top-100 BM25 candidates
    # R@k = #rel in top-k / total_rel_in_qrels (qrels-wide)
    rel = qrels[qrels["relevance"] > 0].copy()
    rel_per_q = rel.groupby("query_id").size().to_dict()

    ceilings = []
    for qid, g in cands.groupby("query_id"):
        top100 = set(g.sort_values("rank_bm25").head(100)["doc_id"].astype(str))
        total_rel = rel_per_q.get(qid, 0)
        if total_rel == 0:
            continue
        hit = rel[rel["query_id"] == qid]["doc_id"].astype(str).isin(top100).sum()
        ceilings.append(hit / total_rel)
    ceil_arr = np.array(ceilings)
    print(f"\n[ceiling]  BM25 top-100 R@100 ceiling: mean={ceil_arr.mean():.3f}  "
          f"median={np.median(ceil_arr):.3f}  n={len(ceil_arr)}")
    print("           (this is the MAX R@100 achievable by any reranker over these candidates)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
