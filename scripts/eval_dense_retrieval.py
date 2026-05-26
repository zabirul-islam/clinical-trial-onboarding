import math
import re
from collections import defaultdict
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import torch

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
IND = ROOT / "indices"
OUT = ROOT / "outputs" / "tables"
OUT.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOPK = 1000


def dcg_at_k(rels, k):
    rels = rels[:k]
    out = 0.0
    for i, rel in enumerate(rels, start=1):
        out += (2**rel - 1) / math.log2(i + 1)
    return out


def ndcg_at_k(rels, k):
    ideal = dcg_at_k(sorted(rels, reverse=True), k)
    actual = dcg_at_k(rels, k)
    return actual / ideal if ideal > 0 else 0.0


def recall_at_k(rels, k):
    total_rel = sum(1 for r in rels if r > 0)
    if total_rel == 0:
        return 0.0
    found_rel = sum(1 for r in rels[:k] if r > 0)
    return found_rel / total_rel


def infer_query_text_col(topics: pd.DataFrame):
    candidate_cols = [c for c in topics.columns if c != "query_id"]
    for c in candidate_cols:
        if topics[c].astype(str).str.len().mean() > 20:
            return c
    raise ValueError(f"Could not infer query text column from {list(topics.columns)}")


def main():
    topics = pd.read_csv(DATA / "trec_ct2021_topics_full.csv")
    qrels = pd.read_csv(DATA / "trec_ct2021_qrels_full.csv")

    qrels_by_query = defaultdict(dict)
    for _, row in qrels.iterrows():
        qrels_by_query[row["query_id"]][row["doc_id"]] = int(row["relevance"])

    query_col = infer_query_text_col(topics)

    index = faiss.read_index(str(IND / "dense_index_minilm_ip.faiss"))
    doc_ids = np.load(IND / "dense_doc_ids.npy", allow_pickle=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model = SentenceTransformer(MODEL_NAME, device=device)

    queries = topics[query_col].astype(str).tolist()
    qids = topics["query_id"].tolist()

    q_emb = model.encode(
        queries,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    scores, indices = index.search(q_emb, TOPK)

    run_rows = []
    metric_rows = []

    for qid, score_row, idx_row in tqdm(zip(qids, scores, indices), total=len(qids), desc="eval dense"):
        ranked_docs = []
        for rank, (score, idx) in enumerate(zip(score_row, idx_row), start=1):
            if idx < 0:
                continue
            doc_id = str(doc_ids[idx])
            ranked_docs.append((doc_id, float(score)))
            run_rows.append({
                "query_id": qid,
                "doc_id": doc_id,
                "rank": rank,
                "score": float(score),
                "run_name": "dense_minilm_eligibility_focus",
            })

        gold = qrels_by_query.get(qid, {})
        ranked_rels = [gold.get(doc_id, 0) for doc_id, _ in ranked_docs]

        metric_rows.append({
            "query_id": qid,
            "nDCG@10": ndcg_at_k(ranked_rels, 10),
            "nDCG@20": ndcg_at_k(ranked_rels, 20),
            "Recall@10": recall_at_k(ranked_rels, 10),
            "Recall@20": recall_at_k(ranked_rels, 20),
            "Recall@100": recall_at_k(ranked_rels, 100),
        })

    run_df = pd.DataFrame(run_rows)
    metrics_df = pd.DataFrame(metric_rows)
    summary = {
        "run_name": "dense_minilm_eligibility_focus",
        "n_queries": int(len(metrics_df)),
        "nDCG@10": float(metrics_df["nDCG@10"].mean()),
        "nDCG@20": float(metrics_df["nDCG@20"].mean()),
        "Recall@10": float(metrics_df["Recall@10"].mean()),
        "Recall@20": float(metrics_df["Recall@20"].mean()),
        "Recall@100": float(metrics_df["Recall@100"].mean()),
    }

    run_df.to_csv(OUT / "dense_run.csv", index=False)
    metrics_df.to_csv(OUT / "dense_per_query_metrics.csv", index=False)
    pd.DataFrame([summary]).to_csv(OUT / "dense_summary_metrics.csv", index=False)

    print("Saved:")
    print(OUT / "dense_run.csv")
    print(OUT / "dense_per_query_metrics.csv")
    print(OUT / "dense_summary_metrics.csv")
    print("\nSummary:")
    print(pd.DataFrame([summary]).to_string(index=False))


if __name__ == "__main__":
    main()
