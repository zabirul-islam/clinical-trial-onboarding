import math
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd
from rank_bm25 import BM25Okapi
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
OUT = ROOT / "outputs" / "tables"
OUT.mkdir(parents=True, exist_ok=True)


def tokenize(text: str):
    return re.findall(r"[a-z0-9]+", str(text).lower())


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


def evaluate_run(run_name, ranked_rows, qrels_by_query):
    metric_rows = []
    for qid, docs in ranked_rows.items():
        gold = qrels_by_query.get(qid, {})
        ranked_rels = [gold.get(doc_id, 0) for doc_id, _ in docs]

        metric_rows.append({
            "query_id": qid,
            "run_name": run_name,
            "nDCG@10": ndcg_at_k(ranked_rels, 10),
            "nDCG@20": ndcg_at_k(ranked_rels, 20),
            "Recall@10": recall_at_k(ranked_rels, 10),
            "Recall@20": recall_at_k(ranked_rels, 20),
            "Recall@100": recall_at_k(ranked_rels, 100),
        })

    metrics_df = pd.DataFrame(metric_rows)
    summary = {
        "run_name": run_name,
        "n_queries": int(len(metrics_df)),
        "nDCG@10": float(metrics_df["nDCG@10"].mean()),
        "nDCG@20": float(metrics_df["nDCG@20"].mean()),
        "Recall@10": float(metrics_df["Recall@10"].mean()),
        "Recall@20": float(metrics_df["Recall@20"].mean()),
        "Recall@100": float(metrics_df["Recall@100"].mean()),
    }
    return metrics_df, pd.DataFrame([summary])


def main():
    corpus = pd.read_parquet(DATA / "retrieval_corpus.parquet")
    topics = pd.read_csv(DATA / "trec_ct2021_topics_full.csv")
    qrels = pd.read_csv(DATA / "trec_ct2021_qrels_full.csv")

    qrels_by_query = defaultdict(dict)
    for _, row in qrels.iterrows():
        qrels_by_query[row["query_id"]][row["doc_id"]] = int(row["relevance"])

    query_col = infer_query_text_col(topics)
    doc_ids = corpus["doc_id"].tolist()

    field_specs = {
        "bm25_all_text": "text_all",
        "bm25_eligibility_focus": "text_eligibility_focus",
    }

    all_run_rows = []
    all_metrics = []
    all_summaries = []

    cached_rankings = {}

    for run_name, text_col in field_specs.items():
        tokenized_corpus = [tokenize(t) for t in corpus[text_col].tolist()]
        bm25 = BM25Okapi(tokenized_corpus)
        ranked_rows = {}

        for _, topic in tqdm(topics.iterrows(), total=len(topics), desc=run_name):
            qid = topic["query_id"]
            query_text = str(topic[query_col])
            scores = bm25.get_scores(tokenize(query_text))
            ranked = sorted(zip(doc_ids, scores), key=lambda x: x[1], reverse=True)[:1000]
            ranked_rows[qid] = ranked

            for rank, (doc_id, score) in enumerate(ranked, start=1):
                all_run_rows.append({
                    "query_id": qid,
                    "doc_id": doc_id,
                    "rank": rank,
                    "score": float(score),
                    "run_name": run_name,
                })

        cached_rankings[run_name] = ranked_rows
        metrics_df, summary_df = evaluate_run(run_name, ranked_rows, qrels_by_query)
        all_metrics.append(metrics_df)
        all_summaries.append(summary_df)

    # Reciprocal-rank fusion of the two BM25 runs
    fused = {}
    k = 60.0
    for qid in topics["query_id"].tolist():
        scores = defaultdict(float)
        for run_name in field_specs.keys():
            ranked = cached_rankings[run_name][qid]
            for rank, (doc_id, _) in enumerate(ranked, start=1):
                scores[doc_id] += 1.0 / (k + rank)
        fused_ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:1000]
        fused[qid] = fused_ranked

        for rank, (doc_id, score) in enumerate(fused_ranked, start=1):
            all_run_rows.append({
                "query_id": qid,
                "doc_id": doc_id,
                "rank": rank,
                "score": float(score),
                "run_name": "rrf_bm25_all_plus_eligibility",
            })

    metrics_df, summary_df = evaluate_run("rrf_bm25_all_plus_eligibility", fused, qrels_by_query)
    all_metrics.append(metrics_df)
    all_summaries.append(summary_df)

    run_df = pd.DataFrame(all_run_rows)
    metrics_all = pd.concat(all_metrics, ignore_index=True)
    summary_all = pd.concat(all_summaries, ignore_index=True)

    run_df.to_csv(OUT / "bm25_variants_runs.csv", index=False)
    metrics_all.to_csv(OUT / "bm25_variants_per_query_metrics.csv", index=False)
    summary_all.to_csv(OUT / "bm25_variants_summary_metrics.csv", index=False)

    print(summary_all.to_string(index=False))


if __name__ == "__main__":
    main()
