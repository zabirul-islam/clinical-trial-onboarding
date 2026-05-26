import math
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd
from rank_bm25 import BM25Okapi
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
OUT = ROOT / "outputs"
(OUT / "tables").mkdir(parents=True, exist_ok=True)
(OUT / "logs").mkdir(parents=True, exist_ok=True)


def tokenize(text: str):
    text = str(text).lower()
    return re.findall(r"[a-z0-9]+", text)


def dcg_at_k(rels, k):
    rels = rels[:k]
    out = 0.0
    for i, rel in enumerate(rels, start=1):
        out += (2**rel - 1) / math.log2(i + 1)
    return out


def ndcg_at_k(rels, k):
    actual = dcg_at_k(rels, k)
    ideal = dcg_at_k(sorted(rels, reverse=True), k)
    return actual / ideal if ideal > 0 else 0.0


def recall_at_k(rels, k):
    total_rel = sum(1 for r in rels if r > 0)
    if total_rel == 0:
        return 0.0
    found_rel = sum(1 for r in rels[:k] if r > 0)
    return found_rel / total_rel


def main():
    corpus = pd.read_parquet(DATA / "retrieval_corpus.parquet")
    topics = pd.read_csv(DATA / "trec_ct2021_topics_full.csv")
    qrels = pd.read_csv(DATA / "trec_ct2021_qrels_full.csv")

    text_col = "text_eligibility_focus"
    corpus = corpus[["doc_id", text_col]].copy()

    tokenized_corpus = [tokenize(t) for t in corpus[text_col].tolist()]
    bm25 = BM25Okapi(tokenized_corpus)

    doc_ids = corpus["doc_id"].tolist()
    qrels_by_query = defaultdict(dict)
    for _, row in qrels.iterrows():
        qrels_by_query[row["query_id"]][row["doc_id"]] = int(row["relevance"])

    run_rows = []
    metric_rows = []

    # infer topic text column
    candidate_cols = [c for c in topics.columns if c != "query_id"]
    text_field = None
    for c in candidate_cols:
        if topics[c].astype(str).str.len().mean() > 20:
            text_field = c
            break
    if text_field is None:
        raise ValueError(f"Could not infer query text field from columns: {list(topics.columns)}")

    for _, topic in tqdm(topics.iterrows(), total=len(topics), desc="BM25 queries"):
        qid = topic["query_id"]
        query_text = str(topic[text_field])
        scores = bm25.get_scores(tokenize(query_text))

        ranked = sorted(
            zip(doc_ids, scores),
            key=lambda x: x[1],
            reverse=True
        )[:1000]

        for rank, (doc_id, score) in enumerate(ranked, start=1):
            run_rows.append({
                "query_id": qid,
                "doc_id": doc_id,
                "rank": rank,
                "score": float(score),
                "run_name": "bm25_eligibility_focus",
            })

        gold = qrels_by_query.get(qid, {})
        ranked_rels = [gold.get(doc_id, 0) for doc_id, _ in ranked]

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
    summary = metrics_df.mean(numeric_only=True).to_dict()
    summary["n_queries"] = int(len(metrics_df))
    summary["run_name"] = "bm25_eligibility_focus"

    run_df.to_csv(OUT / "tables" / "bm25_run.csv", index=False)
    metrics_df.to_csv(OUT / "tables" / "bm25_per_query_metrics.csv", index=False)
    pd.DataFrame([summary]).to_csv(OUT / "tables" / "bm25_summary_metrics.csv", index=False)

    print("Saved:")
    print(OUT / "tables" / "bm25_run.csv")
    print(OUT / "tables" / "bm25_per_query_metrics.csv")
    print(OUT / "tables" / "bm25_summary_metrics.csv")
    print("\nSummary:")
    print(pd.DataFrame([summary]).to_string(index=False))


if __name__ == "__main__":
    main()
