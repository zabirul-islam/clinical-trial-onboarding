from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INFILE = ROOT / "data" / "processed" / "trec_ct_docs.parquet"
OUTDIR = ROOT / "data" / "processed"
OUTDIR.mkdir(parents=True, exist_ok=True)


def clean_text(x):
    if pd.isna(x):
        return ""
    return " ".join(str(x).split())


def main():
    df = pd.read_parquet(INFILE)

    for col in ["title", "condition", "summary", "detailed_description", "eligibility"]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].map(clean_text)

    df["text_all"] = (
        "TITLE: " + df["title"] +
        " CONDITION: " + df["condition"] +
        " SUMMARY: " + df["summary"] +
        " DESCRIPTION: " + df["detailed_description"] +
        " ELIGIBILITY: " + df["eligibility"]
    ).str.strip()

    df["text_eligibility_focus"] = (
        "TITLE: " + df["title"] +
        " CONDITION: " + df["condition"] +
        " ELIGIBILITY: " + df["eligibility"]
    ).str.strip()

    keep_cols = [
        "doc_id",
        "title",
        "condition",
        "summary",
        "detailed_description",
        "eligibility",
        "text_all",
        "text_eligibility_focus",
    ]
    df = df[keep_cols]

    df.to_parquet(OUTDIR / "retrieval_corpus.parquet", index=False)
    df[["doc_id", "text_all", "text_eligibility_focus"]].to_csv(
        OUTDIR / "retrieval_corpus_minimal.csv", index=False
    )

    stats = {
        "n_docs": int(len(df)),
        "avg_text_all_chars": float(df["text_all"].str.len().mean()),
        "avg_text_eligibility_focus_chars": float(df["text_eligibility_focus"].str.len().mean()),
        "pct_missing_eligibility": float((df["eligibility"].str.len() == 0).mean()),
    }

    pd.DataFrame([stats]).to_csv(OUTDIR / "retrieval_corpus_stats.csv", index=False)
    print("Saved:")
    print(OUTDIR / "retrieval_corpus.parquet")
    print(OUTDIR / "retrieval_corpus_minimal.csv")
    print(OUTDIR / "retrieval_corpus_stats.csv")


if __name__ == "__main__":
    main()
