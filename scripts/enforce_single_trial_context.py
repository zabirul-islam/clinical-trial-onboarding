from pathlib import Path
import argparse
import json
import shutil
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "tables"

INFILE = OUT / "grounding_evidence_top_passages.csv"
RAW_BACKUP = OUT / "grounding_evidence_top_passages_raw.csv"
FILTERED = OUT / "grounding_evidence_single_trial.csv"
STATUS_JSON = OUT / "single_trial_context_status.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min_weight_share", type=float, default=0.55)
    parser.add_argument("--min_passage_count", type=int, default=3)
    parser.add_argument("--keep_top_passages", type=int, default=6)
    args = parser.parse_args()

    if not INFILE.exists():
        raise FileNotFoundError(f"Missing evidence file: {INFILE}")

    df = pd.read_csv(INFILE)

    required = {"doc_id", "passage_rank"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if not RAW_BACKUP.exists():
        shutil.copy2(INFILE, RAW_BACKUP)

    work = df.copy()
    work["passage_rank"] = work["passage_rank"].astype(int)
    work["weight"] = 1.0 / work["passage_rank"].clip(lower=1)

    score_df = (
        work.groupby("doc_id", as_index=False)
        .agg(
            weighted_score=("weight", "sum"),
            passage_count=("doc_id", "count"),
            best_rank=("passage_rank", "min"),
        )
        .sort_values(["weighted_score", "passage_count", "best_rank"], ascending=[False, False, True])
        .reset_index(drop=True)
    )

    top_doc = str(score_df.loc[0, "doc_id"])
    top_score = float(score_df.loc[0, "weighted_score"])
    total_score = float(score_df["weighted_score"].sum())
    top_share = top_score / total_score if total_score > 0 else 0.0
    top_count = int(score_df.loc[0, "passage_count"])
    n_unique_docs = int(score_df["doc_id"].nunique())

    filtered = (
        work[work["doc_id"] == top_doc]
        .sort_values(["passage_rank"])
        .head(args.keep_top_passages)
        .drop(columns=["weight"])
        .copy()
    )

    abstain = False
    reason = "single_trial_ok"

    if n_unique_docs > 1:
        if top_share < args.min_weight_share or top_count < args.min_passage_count:
            abstain = True
            reason = "mixed_trial_context"

    filtered.to_csv(FILTERED, index=False)

    status = {
        "status": "abstain_mixed_context" if abstain else "single_trial_ok",
        "reason": reason,
        "selected_doc_id": top_doc,
        "selected_doc_weight_share": round(top_share, 6),
        "selected_doc_passage_count": top_count,
        "n_unique_docs_in_raw_context": n_unique_docs,
        "raw_context_rows": int(len(df)),
        "filtered_context_rows": int(len(filtered)),
        "doc_scores": score_df.to_dict(orient="records"),
        "thresholds": {
            "min_weight_share": args.min_weight_share,
            "min_passage_count": args.min_passage_count,
            "keep_top_passages": args.keep_top_passages,
        },
    }

    with open(STATUS_JSON, "w") as f:
        json.dump(status, f, indent=2)

    print("Saved:", FILTERED)
    print("Saved:", STATUS_JSON)
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
