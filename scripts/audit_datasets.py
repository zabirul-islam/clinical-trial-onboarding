import json
from collections import Counter
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from datasets import load_dataset
import ir_datasets


ROOT = Path(__file__).resolve().parents[1]
OUT_TABLES = ROOT / "outputs" / "tables"
OUT_FIGS = ROOT / "outputs" / "figures"

OUT_TABLES.mkdir(parents=True, exist_ok=True)
OUT_FIGS.mkdir(parents=True, exist_ok=True)


def safe_len(x):
    if x is None:
        return 0
    return len(str(x))


def audit_nli4pr():
    ds = load_dataset("Mathilde/NLI4PR")
    split_names = list(ds.keys())
    rows = []

    for split in split_names:
        df = ds[split].to_pandas()
        df["text_len_1"] = df.iloc[:, 1].astype(str).str.len()
        df["text_len_2"] = df.iloc[:, 2].astype(str).str.len()

        label_col = None
        for c in df.columns:
            vals = df[c].astype(str).str.lower().unique().tolist()
            if any(v in ["entailment", "contradiction", "neutral"] for v in vals):
                label_col = c
                break
        if label_col is None:
            for c in df.columns:
                if "label" in c.lower():
                    label_col = c
                    break

        label_counts = Counter(df[label_col].astype(str)) if label_col else Counter()

        rows.append({
            "dataset": "NLI4PR",
            "split": split,
            "n_rows": len(df),
            "columns": list(df.columns),
            "label_col": label_col,
            "label_counts": dict(label_counts),
            "avg_text_len_1": round(df["text_len_1"].mean(), 2),
            "avg_text_len_2": round(df["text_len_2"].mean(), 2),
        })

    with open(OUT_TABLES / "nli4pr_audit.json", "w") as f:
        json.dump(rows, f, indent=2)

    flat_rows = []
    for r in rows:
        base = {k: v for k, v in r.items() if k not in ["label_counts", "columns"]}
        flat_rows.append(base)
    pd.DataFrame(flat_rows).to_csv(OUT_TABLES / "nli4pr_audit_summary.csv", index=False)

    first = ds[split_names[0]].to_pandas()
    label_col = None
    for c in first.columns:
        vals = first[c].astype(str).str.lower().unique().tolist()
        if any(v in ["entailment", "contradiction", "neutral"] for v in vals):
            label_col = c
            break
    if label_col:
        counts = first[label_col].astype(str).value_counts()
        plt.figure(figsize=(6, 4))
        counts.plot(kind="bar")
        plt.title("NLI4PR label distribution")
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(OUT_FIGS / "nli4pr_label_distribution.png", dpi=220)
        plt.close()


def audit_trec_ct_docs_and_topics():
    # Use 2022 for current topics + shared corpus
    dataset = ir_datasets.load("clinicaltrials/2021/trec-ct-2022")

    sample_docs = []
    for i, doc in enumerate(dataset.docs_iter()):
        sample_docs.append({
            "doc_id": doc.doc_id,
            "title_len": safe_len(getattr(doc, "title", "")),
            "condition_len": safe_len(getattr(doc, "condition", "")),
            "summary_len": safe_len(getattr(doc, "summary", "")),
            "detailed_description_len": safe_len(getattr(doc, "detailed_description", "")),
            "eligibility_len": safe_len(getattr(doc, "eligibility", "")),
        })
        if i >= 999:
            break

    topics = [q._asdict() for q in dataset.queries_iter()]

    pd.DataFrame(sample_docs).to_csv(OUT_TABLES / "trec_ct2022_doc_sample_lengths.csv", index=False)
    pd.DataFrame(topics).to_csv(OUT_TABLES / "trec_ct2022_topics.csv", index=False)

    summary = {
        "dataset": "clinicaltrials/2021/trec-ct-2022",
        "n_topics": len(topics),
        "topic_fields": list(topics[0].keys()) if topics else [],
        "doc_length_means_from_first_1000": pd.DataFrame(sample_docs).mean(numeric_only=True).to_dict() if sample_docs else {},
        "has_qrels_in_this_audit": False,
    }

    with open(OUT_TABLES / "trec_ct2022_audit.json", "w") as f:
        json.dump(summary, f, indent=2)

    if sample_docs:
        df = pd.DataFrame(sample_docs)
        plt.figure(figsize=(7, 4))
        df[["summary_len", "detailed_description_len", "eligibility_len"]].mean().plot(kind="bar")
        plt.title("TREC Clinical Trials 2022 average field lengths (first 1000 docs)")
        plt.ylabel("characters")
        plt.tight_layout()
        plt.savefig(OUT_FIGS / "trec_ct2022_field_lengths.png", dpi=220)
        plt.close()


def audit_trec_ct2021_qrels():
    # Use 2021 for judged retrieval evaluation
    dataset = ir_datasets.load("clinicaltrials/2021/trec-ct-2021")

    topics = [q._asdict() for q in dataset.queries_iter()]
    qrels = [qrel._asdict() for qrel in dataset.qrels_iter()]

    pd.DataFrame(topics).to_csv(OUT_TABLES / "trec_ct2021_topics.csv", index=False)
    pd.DataFrame(qrels).to_csv(OUT_TABLES / "trec_ct2021_qrels.csv", index=False)

    qrels_df = pd.DataFrame(qrels)
    summary = {
        "dataset": "clinicaltrials/2021/trec-ct-2021",
        "n_topics": len(topics),
        "n_qrels": len(qrels),
        "qrel_fields": list(qrels[0].keys()) if qrels else [],
        "relevance_counts": qrels_df["relevance"].value_counts().sort_index().to_dict() if not qrels_df.empty else {},
    }

    with open(OUT_TABLES / "trec_ct2021_qrels_audit.json", "w") as f:
        json.dump(summary, f, indent=2)

    if not qrels_df.empty:
        plt.figure(figsize=(6, 4))
        qrels_df["relevance"].value_counts().sort_index().plot(kind="bar")
        plt.title("TREC Clinical Trials 2021 qrel distribution")
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(OUT_FIGS / "trec_ct2021_qrel_distribution.png", dpi=220)
        plt.close()


if __name__ == "__main__":
    audit_nli4pr()
    audit_trec_ct_docs_and_topics()
    audit_trec_ct2021_qrels()
    print("Audit complete. See outputs/tables and outputs/figures.")
