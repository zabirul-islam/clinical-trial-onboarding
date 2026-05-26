import re
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


def infer_query_text_col(topics: pd.DataFrame):
    candidate_cols = [c for c in topics.columns if c != "query_id"]
    for c in candidate_cols:
        if topics[c].astype(str).str.len().mean() > 20:
            return c
    raise ValueError(f"Could not infer query text column from {list(topics.columns)}")


def main():
    corpus = pd.read_parquet(DATA / "retrieval_corpus.parquet")
    topics = pd.read_csv(DATA / "trec_ct2021_topics_full.csv")

    text_col = "text_all"
    query_col = infer_query_text_col(topics)

    tokenized_corpus = [tokenize(t) for t in corpus[text_col].tolist()]
    bm25 = BM25Okapi(tokenized_corpus)
    doc_ids = corpus["doc_id"].astype(str).tolist()

    rows = []
    for _, topic in tqdm(topics.iterrows(), total=len(topics), desc="BM25 candidate export"):
        qid = topic["query_id"]
        query_text = str(topic[query_col])
        scores = bm25.get_scores(tokenize(query_text))
        ranked = sorted(zip(doc_ids, scores), key=lambda x: x[1], reverse=True)[:100]

        for rank, (doc_id, score) in enumerate(ranked, start=1):
            rows.append({
                "query_id": qid,
                "doc_id": doc_id,
                "rank_bm25": rank,
                "score_bm25": float(score),
            })

    out = pd.DataFrame(rows)
    out.to_csv(OUT / "bm25_full_text_top100_candidates.csv", index=False)
    print("Saved:", OUT / "bm25_full_text_top100_candidates.csv")


if __name__ == "__main__":
    main()
