from pathlib import Path
import argparse
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "tables"

INFILE = OUT / "grounding_evidence_top_passages.csv"
OUT_CSV = OUT / "trial_level_scores.csv"
OUT_JSON = OUT / "trial_level_scores.json"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top_k_trials", type=int, default=5)
    args = parser.parse_args()

    if not INFILE.exists():
        raise FileNotFoundError(f"Missing file: {INFILE}")

    df = pd.read_csv(INFILE)

    required = {"doc_id", "passage_rank", "cross_score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    work = df.copy()
    work["passage_rank"] = work["passage_rank"].astype(int)
    work["cross_score"] = pd.to_numeric(work["cross_score"], errors="coerce").fillna(0.0)

    work["rrf_weight"] = 1.0 / (work["passage_rank"] + 1.0)

    min_cs = work["cross_score"].min()
    work["cross_score_shifted"] = work["cross_score"] - min_cs + 1e-6

    grouped = (
        work.groupby("doc_id", as_index=False)
        .agg(
            n_passages=("doc_id", "count"),
            best_rank=("passage_rank", "min"),
            sum_rrf=("rrf_weight", "sum"),
            sum_cross=("cross_score_shifted", "sum"),
            max_cross=("cross_score_shifted", "max"),
            raw_sum_cross=("cross_score", "sum"),
            raw_max_cross=("cross_score", "max"),
        )
    )

    grouped["trial_score"] = (
        0.45 * grouped["sum_rrf"] +
        0.45 * grouped["sum_cross"] +
        0.10 * grouped["n_passages"]
    )

    grouped = grouped.sort_values(
        ["trial_score", "n_passages", "best_rank"],
        ascending=[False, False, True]
    ).reset_index(drop=True)

    grouped.to_csv(OUT_CSV, index=False)

    with open(OUT_JSON, "w") as f:
        json.dump(grouped.head(args.top_k_trials).to_dict(orient="records"), f, indent=2)

    print("Saved:", OUT_CSV)
    print("Saved:", OUT_JSON)
    print(grouped.head(args.top_k_trials).to_string(index=False))

if __name__ == "__main__":
    main()
