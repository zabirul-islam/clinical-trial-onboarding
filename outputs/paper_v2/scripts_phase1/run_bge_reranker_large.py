"""
Phase 1.3 — Run BAAI/bge-reranker-large as a drop-in replacement for
ms-marco-MiniLM-L-6-v2 to rerank BM25 top-100 candidates.

Inputs:
  outputs/tables/bm25_full_text_top100_candidates.csv
    columns expected: query_id, doc_id, rank, score, passage_text (if available)
    If passage_text not present, we fetch it from data/processed/trial_evidence_passages.parquet.
  data/processed/trec_ct2021_topics_full.csv  (for query text)
  data/processed/trec_ct2021_qrels_full.csv   (for ground-truth relevance)

Outputs:
  outputs/tables/bge_reranker_large_run.csv
  outputs/tables/bge_reranker_large_per_query_metrics.csv
  outputs/tables/bge_reranker_large_summary_metrics.csv

Run (with GPU):
  python scripts/run_bge_reranker_large.py \
      --repo-root . \
      --device cuda \
      --batch-size 8 \
      --max-length 512 \
      --topk 20

GPU memory: ~6 GB fp16. Fits easily on GB10.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
except Exception as e:
    print("ERROR: need torch + transformers. pip install transformers torch", file=sys.stderr)
    raise

MODEL_ID = "BAAI/bge-reranker-large"


def dcg(rels: np.ndarray) -> float:
    return float(np.sum(rels / np.log2(np.arange(2, len(rels) + 2))))


def ndcg_at_k(rels: np.ndarray, k: int) -> float:
    rels_k = rels[:k]
    ideal = np.sort(rels_k)[::-1]
    denom = dcg(ideal)
    return 0.0 if denom == 0 else dcg(rels_k) / denom


def recall_at_k(rels: np.ndarray, total_rel: int, k: int) -> float:
    if total_rel == 0:
        return 0.0
    return float((rels[:k] > 0).sum()) / total_rel


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", type=Path, default=Path("."))
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--topk", type=int, default=100,
                   help="Rerank top-K BM25 candidates per query (use 100 to match 2-stage pipeline)")
    args = p.parse_args()

    root = args.repo_root.resolve()
    table_dir = root / "outputs/tables"
    cands_path = table_dir / "bm25_full_text_top100_candidates.csv"
    topics_path = root / "data/processed/trec_ct2021_topics_full.csv"
    qrels_path = root / "data/processed/trec_ct2021_qrels_full.csv"
    passages_path = root / "data/processed/trial_evidence_passages.parquet"

    print(f"loading candidates: {cands_path}")
    cands = pd.read_csv(cands_path)
    cands.columns = [c.strip().lower() for c in cands.columns]
    assert {"query_id", "doc_id"}.issubset(cands.columns), cands.columns
    if "passage_text" not in cands.columns:
        print("joining passage_text from data/processed/trial_evidence_passages.parquet")
        passages = pd.read_parquet(passages_path)
        pcols = {c.lower(): c for c in passages.columns}
        # Heuristic: expect columns doc_id/nct_id + text
        text_col = None
        for cand in ["passage_text", "text", "trial_text", "evidence_text"]:
            if cand in pcols:
                text_col = pcols[cand]; break
        id_col = None
        for cand in ["doc_id", "nct_id", "trial_id"]:
            if cand in pcols:
                id_col = pcols[cand]; break
        assert text_col and id_col, f"Need text + id cols; got {list(pcols)}"
        passages = passages[[id_col, text_col]].rename(columns={id_col: "doc_id", text_col: "passage_text"})
        # Collapse multi-passage per doc by concatenation
        passages = passages.groupby("doc_id")["passage_text"].apply(lambda s: " ".join(s.astype(str))).reset_index()
        cands = cands.merge(passages, on="doc_id", how="left")

    cands["passage_text"] = cands["passage_text"].fillna("").astype(str)
    # Defensive sort: detect any rank/score-like column
    rank_like = next((c for c in ["rank", "rank_bm25", "position"] if c in cands.columns), None)
    score_like = next((c for c in ["score", "bm25_score", "similarity", "score_bm25"] if c in cands.columns), None)
    if rank_like:
        cands = cands.sort_values(["query_id", rank_like], ascending=[True, True])
    elif score_like:
        cands = cands.sort_values(["query_id", score_like], ascending=[True, False])
    else:
        print(f"WARN: no rank/score col found. Cols = {list(cands.columns)}. Using CSV order.", file=sys.stderr)
        cands = cands.sort_values(["query_id"], kind="stable")

    topics = pd.read_csv(topics_path)
    topics.columns = [c.strip().lower() for c in topics.columns]
    qtext = dict(zip(topics["query_id"].astype(int), topics["text"].astype(str)))

    qrels = pd.read_csv(qrels_path)
    qrels.columns = [c.strip().lower() for c in qrels.columns]
    rel_lookup = {(int(r.query_id), str(r.doc_id)): int(r.relevance) for r in qrels.itertuples()}

    print(f"loading model: {MODEL_ID}")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, torch_dtype=torch.float16)
    model.to(args.device).eval()

    out_rows = []
    per_q = []
    for qid, g in tqdm(cands.groupby("query_id"), desc="rerank"):
        qid = int(qid)
        q = qtext.get(qid, "")
        sub = g.head(args.topk).copy()
        pairs = [(q, t) for t in sub["passage_text"].tolist()]

        scores = []
        with torch.no_grad():
            for i in range(0, len(pairs), args.batch_size):
                batch = pairs[i:i+args.batch_size]
                enc = tok(batch, padding=True, truncation=True, max_length=args.max_length, return_tensors="pt").to(args.device)
                logits = model(**enc).logits.squeeze(-1)
                scores.extend(logits.float().cpu().numpy().tolist())
        sub["bge_score"] = scores
        sub = sub.sort_values("bge_score", ascending=False).reset_index(drop=True)
        sub["rank_bge"] = np.arange(1, len(sub) + 1)
        for r in sub.itertuples():
            out_rows.append({"query_id": qid, "doc_id": r.doc_id, "rank_bge": int(r.rank_bge), "bge_score": float(r.bge_score)})

        # metrics: rels over reranked list; recall denom = all judged relevant for this query
        rels = np.array([rel_lookup.get((qid, str(d)), 0) for d in sub["doc_id"]])
        total_rel = sum(1 for (q_, _), v in rel_lookup.items() if q_ == qid and v > 0)
        # recall@100 uses whichever depth is reranked
        per_q.append({
            "query_id": qid,
            "ndcg@10":   ndcg_at_k(rels, 10),
            "ndcg@20":   ndcg_at_k(rels, 20),
            "recall@10": recall_at_k(rels, total_rel, 10),
            "recall@20": recall_at_k(rels, total_rel, 20),
            "recall@100": recall_at_k(rels, total_rel, min(100, args.topk)),
        })

    pd.DataFrame(out_rows).to_csv(table_dir / "bge_reranker_large_run.csv", index=False)
    per_q_df = pd.DataFrame(per_q)
    per_q_df.to_csv(table_dir / "bge_reranker_large_per_query_metrics.csv", index=False)
    metric_cols = ["ndcg@10", "ndcg@20", "recall@10", "recall@20", "recall@100"]
    summ = per_q_df[metric_cols].mean(numeric_only=True).to_frame("mean").T
    summ.insert(0, "method", "BGE-reranker-large")
    summ.insert(1, "n_queries", len(per_q_df))
    summ.insert(2, "topk", args.topk)
    summ.to_csv(table_dir / "bge_reranker_large_summary_metrics.csv", index=False)
    print(summ.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
