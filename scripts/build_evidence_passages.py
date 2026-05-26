from pathlib import Path
import re
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
OUT = ROOT / "data" / "processed"

MAX_CHARS = 1200
MIN_CHARS = 80


def clean_text(x):
    if pd.isna(x):
        return ""
    return " ".join(str(x).split())


def split_long_text(text, max_chars=MAX_CHARS):
    text = clean_text(text)
    if not text:
        return []

    parts = re.split(r'(?<=[\.\?\!])\s+', text)
    chunks = []
    current = ""

    for part in parts:
        if not part:
            continue
        candidate = (current + " " + part).strip() if current else part
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                chunks.append(current)
            if len(part) <= max_chars:
                current = part
            else:
                # hard split very long segment
                for i in range(0, len(part), max_chars):
                    sub = part[i:i + max_chars].strip()
                    if sub:
                        chunks.append(sub)
                current = ""
    if current:
        chunks.append(current)

    return [c for c in chunks if len(c) >= MIN_CHARS]


def main():
    df = pd.read_parquet(DATA / "trec_ct_docs.parquet")

    for col in ["title", "condition", "summary", "detailed_description", "eligibility"]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].map(clean_text)

    rows = []

    for _, row in df.iterrows():
        doc_id = str(row["doc_id"])
        title = row["title"]
        condition = row["condition"]

        # short structured fields
        short_sections = [
            ("title", title),
            ("condition", condition),
            ("summary", row["summary"]),
        ]

        for section_name, text in short_sections:
            text = clean_text(text)
            if len(text) >= MIN_CHARS:
                rows.append({
                    "passage_id": f"{doc_id}::{section_name}::0",
                    "doc_id": doc_id,
                    "section": section_name,
                    "passage_text": text,
                    "title": title,
                    "condition": condition,
                })

        # longer sections split into chunks
        for section_name in ["detailed_description", "eligibility"]:
            text = row[section_name]
            chunks = split_long_text(text)
            for i, chunk in enumerate(chunks):
                rows.append({
                    "passage_id": f"{doc_id}::{section_name}::{i}",
                    "doc_id": doc_id,
                    "section": section_name,
                    "passage_text": chunk,
                    "title": title,
                    "condition": condition,
                })

    out = pd.DataFrame(rows)
    out.to_parquet(OUT / "trial_evidence_passages.parquet", index=False)
    out.to_csv(OUT / "trial_evidence_passages_sample.csv", index=False)

    stats = pd.DataFrame([{
        "n_passages": int(len(out)),
        "n_unique_docs": int(out["doc_id"].nunique()),
        "avg_passage_chars": float(out["passage_text"].str.len().mean()),
        "median_passage_chars": float(out["passage_text"].str.len().median()),
    }])
    stats.to_csv(OUT / "trial_evidence_passages_stats.csv", index=False)

    print("Saved:")
    print(OUT / "trial_evidence_passages.parquet")
    print(OUT / "trial_evidence_passages_sample.csv")
    print(OUT / "trial_evidence_passages_stats.csv")


if __name__ == "__main__":
    main()
