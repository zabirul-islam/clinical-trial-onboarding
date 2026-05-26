from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
import torch

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
OUT = ROOT / "indices"
OUT.mkdir(parents=True, exist_ok=True)


def safe_name(s: str) -> str:
    return s.replace("/", "__").replace("-", "_")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--text_col", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()

    corpus = pd.read_parquet(DATA / "retrieval_corpus.parquet")
    if args.text_col not in corpus.columns:
        raise ValueError(f"text_col '{args.text_col}' not found. Available: {list(corpus.columns)}")

    texts = corpus[args.text_col].astype(str).tolist()
    doc_ids = corpus["doc_id"].astype(str).tolist()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Model: {args.model_name}")
    print(f"Text column: {args.text_col}")

    model = SentenceTransformer(args.model_name, device=device)

    embeddings = model.encode(
        texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    tag = f"{safe_name(args.model_name)}__{args.text_col}"
    faiss.write_index(index, str(OUT / f"dense_index__{tag}.faiss"))
    np.save(OUT / f"dense_doc_ids__{tag}.npy", np.array(doc_ids, dtype=object), allow_pickle=True)
    np.save(OUT / f"dense_embeddings_shape__{tag}.npy", np.array(embeddings.shape))

    stats = pd.DataFrame([{
        "model_name": args.model_name,
        "text_col": args.text_col,
        "device": device,
        "n_docs": len(doc_ids),
        "embedding_dim": int(embeddings.shape[1]),
        "batch_size": args.batch_size,
        "tag": tag,
    }])
    stats.to_csv(OUT / f"dense_index_stats__{tag}.csv", index=False)

    print("Saved:")
    print(OUT / f"dense_index__{tag}.faiss")
    print(OUT / f"dense_doc_ids__{tag}.npy")
    print(OUT / f"dense_embeddings_shape__{tag}.npy")
    print(OUT / f"dense_index_stats__{tag}.csv")


if __name__ == "__main__":
    main()
