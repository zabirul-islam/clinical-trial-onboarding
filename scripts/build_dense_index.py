from pathlib import Path
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import torch

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
OUT = ROOT / "indices"
OUT.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 128


def main():
    corpus = pd.read_parquet(DATA / "retrieval_corpus.parquet")
    texts = corpus["text_eligibility_focus"].astype(str).tolist()
    doc_ids = corpus["doc_id"].astype(str).tolist()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = SentenceTransformer(MODEL_NAME, device=device)

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, str(OUT / "dense_index_minilm_ip.faiss"))

    np.save(OUT / "dense_doc_ids.npy", np.array(doc_ids, dtype=object), allow_pickle=True)
    np.save(OUT / "dense_embeddings_shape.npy", np.array(embeddings.shape))

    stats = pd.DataFrame([{
        "model_name": MODEL_NAME,
        "device": device,
        "n_docs": len(doc_ids),
        "embedding_dim": int(embeddings.shape[1]),
        "batch_size": BATCH_SIZE,
    }])
    stats.to_csv(OUT / "dense_index_stats.csv", index=False)

    print("Saved:")
    print(OUT / "dense_index_minilm_ip.faiss")
    print(OUT / "dense_doc_ids.npy")
    print(OUT / "dense_embeddings_shape.npy")
    print(OUT / "dense_index_stats.csv")


if __name__ == "__main__":
    main()
