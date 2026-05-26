from pathlib import Path
import argparse
import re
import pandas as pd
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
import torch

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
OUT = ROOT / "outputs" / "tables"
OUT.mkdir(parents=True, exist_ok=True)

CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def tokenize(text: str):
    return re.findall(r"[a-z0-9]+", str(text).lower())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", type=str, required=True)
    parser.add_argument("--top_docs", type=int, default=20)
    parser.add_argument("--top_passages", type=int, default=8)
    args = parser.parse_args()

    corpus = pd.read_parquet(DATA / "retrieval_corpus.parquet")
    passages = pd.read_parquet(DATA / "trial_evidence_passages.parquet")

    # BM25 over full trial text
    tokenized_corpus = [tokenize(t) for t in corpus["text_all"].astype(str).tolist()]
    bm25 = BM25Okapi(tokenized_corpus)

    doc_ids = corpus["doc_id"].astype(str).tolist()
    scores = bm25.get_scores(tokenize(args.question))
    ranked_docs = sorted(zip(doc_ids, scores), key=lambda x: x[1], reverse=True)[:args.top_docs]
    top_doc_ids = {doc_id for doc_id, _ in ranked_docs}

    doc_rank_map = {doc_id: rank for rank, (doc_id, _) in enumerate(ranked_docs, start=1)}
    doc_score_map = {doc_id: float(score) for doc_id, score in ranked_docs}

    cand = passages[passages["doc_id"].astype(str).isin(top_doc_ids)].copy()
    cand["doc_rank_bm25"] = cand["doc_id"].astype(str).map(doc_rank_map)
    cand["doc_score_bm25"] = cand["doc_id"].astype(str).map(doc_score_map)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model = CrossEncoder(CROSS_ENCODER_MODEL, device=device)

    pairs = list(zip([args.question] * len(cand), cand["passage_text"].astype(str).tolist()))
    ce_scores = model.predict(pairs, batch_size=32, show_progress_bar=False)
    cand["cross_score"] = ce_scores

    cand = cand.sort_values(["cross_score", "doc_score_bm25"], ascending=[False, False]).reset_index(drop=True)
    cand["passage_rank"] = range(1, len(cand) + 1)

    top = cand.head(args.top_passages).copy()

    top.to_csv(OUT / "grounding_evidence_top_passages.csv", index=False)

    print("Top evidence passages:")
    cols = ["passage_rank", "doc_id", "section", "doc_rank_bm25", "cross_score", "passage_text"]
    print(top[cols].to_string(index=False))
    print("\nSaved:", OUT / "grounding_evidence_top_passages.csv")


if __name__ == "__main__":
    main()
