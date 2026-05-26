import math
from collections import defaultdict
from pathlib import Path

import pandas as pd
from sentence_transformers import CrossEncoder
from tqdm import tqdm
import torch

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
OUT = ROOT / "outputs" / "tables"
OUT.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
BATCH_SIZE = 32


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
    corpus = pd.read_parquet(DATA / "retrieval_corpus.parquet")
    topics = pd.read_csv(DATA / "trec_ct2021_topics_full.csv")
    qrels = pd.read_csv(DATA / "trec_ct2021_qrels_full.csv")
    candidates = pd.read_csv(OUT / "bm25_full_text_top100_candidates.csv")

    query_col = infer_query_text_col(topics)

    topics_map = dict(zip(topics["query_id"], topics[query_col].astype(str)))
    corpus_map = dict(zip(corpus["doc_id"].astype(str), corpus["text_all"].astype(str)))

    qrels_by_query = defaultdict(dict)
    for _, row in qrels.iterrows():
        qrels_by_query[row["query_id"]][row["doc_id"]] = int(row["relevance"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Model: {MODEL_NAME}")

    model = CrossEncoder(MODEL_NAME, device=device)

    run_rows = []
    metric_rows = []

    for qid, group in tqdm(candidates.groupby("query_id"), total=candidates["query_id"].nunique(), desc="Cross-encoder rerank"):
        query_text = topics_map[qid]

        pairs = []
        doc_list = []
        for _, row in group.iterrows():
            doc_id = str(row["doc_id"])
            doc_text = corpus_map[doc_id]
            pairs.append((query_text, doc_text))
            doc_list.append(doc_id)

        scores = model.predict(pairs, batch_size=BATCH_SIZE, show_progress_bar=False)
        ranked = sorted(zip(doc_list, scores), key=lambda x: x[1], reverse=True)

        gold = qrels_by_query.get(qid, {})
        ranked_rels = []

        for rank, (doc_id, score) in enumerate(ranked, start=1):
            ranked_rels.append(gold.get(doc_id, 0))
            run_rows.append({
                "query_id": qid,
                "doc_id": doc_id,
                "rank": rank,
                "score": float(score),
                "run_name": "bm25_top100_crossenc_minilm",
            })

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
        "run_name": "bm25_top100_crossenc_minilm",
        "model_name": MODEL_NAME,
        "candidate_pool": "bm25_full_text_top100",
        "n_queries": int(len(metrics_df)),
        "nDCG@10": float(metrics_df["nDCG@10"].mean()),
        "nDCG@20": float(metrics_df["nDCG@20"].mean()),
        "Recall@10": float(metrics_df["Recall@10"].mean()),
        "Recall@20": float(metrics_df["Recall@20"].mean()),
        "Recall@100": float(metrics_df["Recall@100"].mean()),
    }

    run_df.to_csv(OUT / "crossenc_run.csv", index=False)
    metrics_df.to_csv(OUT / "crossenc_per_query_metrics.csv", index=False)
    pd.DataFrame([summary]).to_csv(OUT / "crossenc_summary_metrics.csv", index=False)

    print("Saved:")
    print(OUT / "crossenc_run.csv")
    print(OUT / "crossenc_per_query_metrics.csv")
    print(OUT / "crossenc_summary_metrics.csv")
    print("\nSummary:")
    print(pd.DataFrame([summary]).to_string(index=False))


if __name__ == "__main__":
    main()
