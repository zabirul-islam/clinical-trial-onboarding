"""
Phase 1.3b — Re-export BM25 candidates at arbitrary depth (default top-1000).

Goal: raise R@100 ceiling from 0.15 (top-100 pool) to ~0.40+ by reranking
from a deeper BM25 pool.

Based on scripts/export_bm25_candidates.py, parametrised on:
  --topk            (default 1000)
  --topics-csv      (default data/processed/trec_ct2021_topics_full.csv)
  --out-csv         (default outputs/tables/bm25_full_text_top{K}_candidates.csv)
  --corpus-parquet  (default data/processed/retrieval_corpus.parquet)

Run:
  cd ~/Desktop/islamm11/avatar_trial_onboarding
  python scripts/export_bm25_topk.py --topk 1000
"""
from __future__ import annotations
import argparse, re
from pathlib import Path

import pandas as pd
from rank_bm25 import BM25Okapi
from tqdm import tqdm


def tokenize(text: str):
    return re.findall(r"[a-z0-9]+", str(text).lower())


def infer_query_text_col(topics: pd.DataFrame):
    cands = [c for c in topics.columns if c != "query_id"]
    for c in cands:
        if topics[c].astype(str).str.len().mean() > 20:
            return c
    raise ValueError(f"Could not infer query text column from {list(topics.columns)}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", type=Path, default=Path("."))
    p.add_argument("--topk", type=int, default=1000)
    p.add_argument("--topics-csv", type=Path, default=None)
    p.add_argument("--corpus-parquet", type=Path, default=None)
    p.add_argument("--out-csv", type=Path, default=None)
    p.add_argument("--text-col", default="text_all")
    args = p.parse_args()

    root = args.repo_root.resolve()
    data = root / "data" / "processed"
    out_dir = root / "outputs" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    corpus_path = args.corpus_parquet or (data / "retrieval_corpus.parquet")
    topics_path = args.topics_csv     or (data / "trec_ct2021_topics_full.csv")
    out_csv = args.out_csv or (out_dir / f"bm25_full_text_top{args.topk}_candidates.csv")

    print(f"[load] corpus: {corpus_path}")
    corpus = pd.read_parquet(corpus_path)
    print(f"[load] topics: {topics_path}")
    topics = pd.read_csv(topics_path)

    query_col = infer_query_text_col(topics)
    print(f"[info] query text col = {query_col!r}, corpus text col = {args.text_col!r}")
    print(f"[info] |corpus| = {len(corpus):,}, |topics| = {len(topics):,}, topk = {args.topk}")

    tokenized_corpus = [tokenize(t) for t in corpus[args.text_col].tolist()]
    bm25 = BM25Okapi(tokenized_corpus)
    doc_ids = corpus["doc_id"].astype(str).tolist()

    rows = []
    for _, topic in tqdm(topics.iterrows(), total=len(topics), desc=f"BM25 top-{args.topk}"):
        qid = topic["query_id"]
        q = str(topic[query_col])
        scores = bm25.get_scores(tokenize(q))
        ranked = sorted(zip(doc_ids, scores), key=lambda x: x[1], reverse=True)[: args.topk]
        for rank, (doc_id, score) in enumerate(ranked, start=1):
            rows.append({
                "query_id":   qid,
                "doc_id":     doc_id,
                "rank_bm25":  rank,
                "score_bm25": float(score),
            })

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"[done] wrote {len(df):,} rows → {out_csv}")

    # Quick ceiling sanity: load qrels and report R@100 ceiling from this pool
    qrels_csv = data / "trec_ct2021_qrels_full.csv"
    if qrels_csv.exists():
        qrels = pd.read_csv(qrels_csv)
        rel = qrels[qrels["relevance"] > 0]
        rel_per_q = rel.groupby("query_id").size().to_dict()
        ceilings_100, ceilings_1000 = [], []
        for qid, g in df.groupby("query_id"):
            total_rel = rel_per_q.get(qid, 0)
            if total_rel == 0:
                continue
            rel_set = set(rel[rel["query_id"] == qid]["doc_id"].astype(str))
            top100  = set(g.sort_values("rank_bm25").head(100)["doc_id"].astype(str))
            topall  = set(g["doc_id"].astype(str))
            ceilings_100.append(len(rel_set & top100) / total_rel)
            ceilings_1000.append(len(rel_set & topall) / total_rel)
        import numpy as np
        print(f"[ceiling] R@100 from this pool  : mean={np.mean(ceilings_100):.3f}")
        print(f"[ceiling] R@{args.topk} from this pool : mean={np.mean(ceilings_1000):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

