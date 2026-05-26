from pathlib import Path
import pandas as pd
from datasets import load_dataset
import ir_datasets
import json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed"
TABLES = ROOT / "outputs" / "tables"
OUT.mkdir(parents=True, exist_ok=True)
TABLES.mkdir(parents=True, exist_ok=True)


def normalize_label(x: str) -> str:
    s = str(x).strip().lower()
    if s in {"entailment", "contradiction", "neutral"}:
        return s
    return str(x)


def build_nli4pr():
    ds = load_dataset("Mathilde/NLI4PR")
    frames = []

    schema_report = {
        "source_dataset": "NLI4PR",
        "columns": list(ds["train"].to_pandas().columns),
        "field_mapping": {
            "patient_text_plain": "statement_pol",
            "patient_text_medical": "statement_medical",
            "trial_text": "premise",
            "trial_title": "NCT_title",
            "trial_id": "NCT_id",
            "label": "label",
        },
    }

    for split in ds.keys():
        df = ds[split].to_pandas()

        tmp = pd.DataFrame({
            "source_dataset": "NLI4PR",
            "split": split,
            "instance_id": [f"nli4pr_{split}_{i}" for i in range(len(df))],
            "patient_text_plain": df["statement_pol"].astype(str),
            "patient_text_medical": df["statement_medical"].astype(str),
            "trial_text": df["premise"].astype(str),
            "trial_title": df["NCT_title"].astype(str),
            "trial_id": df["NCT_id"].astype(str),
            "label": df["label"].map(normalize_label),
            "task_type": "eligibility_nli",
            "topic_id": df["topic_id"],
            "orig_id": df["id"],
        })
        frames.append(tmp)

    full = pd.concat(frames, ignore_index=True)
    full.to_csv(OUT / "benchmark_nli4pr.csv", index=False)
    full.to_parquet(OUT / "benchmark_nli4pr.parquet", index=False)

    label_summary = (
        full.groupby(["split", "label"]).size().reset_index(name="count")
    )
    label_summary.to_csv(OUT / "benchmark_nli4pr_label_summary.csv", index=False)

    with open(TABLES / "nli4pr_schema_report.json", "w") as f:
        json.dump(schema_report, f, indent=2)


def build_trec_ct():
    # 2022: topics + shared corpus
    ds22 = ir_datasets.load("clinicaltrials/2021/trec-ct-2022")
    topics22 = pd.DataFrame([q._asdict() for q in ds22.queries_iter()])
    topics22["source_dataset"] = "TREC_CT_2022"
    topics22["task_type"] = "trial_retrieval_topics"
    topics22.to_csv(OUT / "trec_ct2022_topics_full.csv", index=False)

    docs_rows = []
    for doc in ds22.docs_iter():
        docs_rows.append({
            "doc_id": doc.doc_id,
            "title": getattr(doc, "title", ""),
            "condition": getattr(doc, "condition", ""),
            "summary": getattr(doc, "summary", ""),
            "detailed_description": getattr(doc, "detailed_description", ""),
            "eligibility": getattr(doc, "eligibility", ""),
        })
    docs = pd.DataFrame(docs_rows)
    docs.to_parquet(OUT / "trec_ct_docs.parquet", index=False)

    # 2021: judged retrieval benchmark
    ds21 = ir_datasets.load("clinicaltrials/2021/trec-ct-2021")
    topics21 = pd.DataFrame([q._asdict() for q in ds21.queries_iter()])
    qrels21 = pd.DataFrame([q._asdict() for q in ds21.qrels_iter()])

    topics21.to_csv(OUT / "trec_ct2021_topics_full.csv", index=False)
    qrels21.to_csv(OUT / "trec_ct2021_qrels_full.csv", index=False)

    retrieval = qrels21.merge(topics21, on="query_id", how="left")
    retrieval["source_dataset"] = "TREC_CT_2021"
    retrieval["task_type"] = "trial_retrieval"
    retrieval.to_csv(OUT / "benchmark_trec_ct2021_retrieval.csv", index=False)


if __name__ == "__main__":
    build_nli4pr()
    build_trec_ct()
    print("Unified benchmark files written to data/processed/")
