"""
Phase 1.2 — Bootstrap 95% CIs for retrieval metrics over TREC CT 2021 topics.

Inputs (per-query metric CSVs already produced by your pipeline):
  outputs/tables/bm25_per_query_metrics.csv
  outputs/tables/crossenc_per_query_metrics.csv
  outputs/tables/dense_per_query_metrics__sentence_transformers__all_MiniLM_L6_v2__text_all.csv
  outputs/tables/dense_per_query_metrics__BAAI__bge_base_en_v1.5__text_all.csv

Each must contain columns: query_id, ndcg@10, ndcg@20, recall@10, recall@20, recall@100
(case-insensitive; we normalize).

Output:
  outputs/tables/retrieval_bootstrap_cis.csv
  outputs/tables/retrieval_bootstrap_cis.tex

Run:
  python scripts/bootstrap_retrieval_cis.py \
      --repo-root . \
      --n-bootstrap 1000 \
      --seed 42 \
      --out outputs/tables/retrieval_bootstrap_cis.csv
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path

import numpy as np
import pandas as pd

METHODS = {
    "BM25 (full text)":           "bm25_per_query_metrics.csv",
    "Dense MiniLM (full text)":   "dense_per_query_metrics__sentence_transformers__all_MiniLM_L6_v2__text_all.csv",
    "Dense BGE-base (full text)": "dense_per_query_metrics__BAAI__bge_base_en_v1.5__text_all.csv",
    "BM25 -> Cross-encoder":      "crossenc_per_query_metrics.csv",
}
METRICS = ["ndcg@10", "ndcg@20", "recall@10", "recall@20", "recall@100"]


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    # Some pipelines use "ndcg_cut_10", etc. Map them.
    rename = {}
    for c in df.columns:
        for m in METRICS:
            key = m.replace("@", "_")  # ndcg_10
            if c == key or c.endswith(key):
                rename[c] = m
    df = df.rename(columns=rename)
    return df


def bootstrap_ci(values: np.ndarray, n_boot: int, rng: np.random.Generator) -> tuple[float, float, float]:
    n = len(values)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        boots[b] = values[idx].mean()
    mean = float(values.mean())
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return mean, float(lo), float(hi)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", type=Path, default=Path("."))
    p.add_argument("--n-bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    root = args.repo_root.resolve()
    table_dir = root / "outputs/tables"

    rows = []
    for method_name, fname in METHODS.items():
        fpath = table_dir / fname
        if not fpath.exists():
            print(f"WARN: missing {fpath}, skipping {method_name}", file=sys.stderr)
            continue
        df = normalize_columns(pd.read_csv(fpath))
        for m in METRICS:
            if m not in df.columns:
                print(f"WARN: {method_name} missing {m}", file=sys.stderr)
                continue
            vals = df[m].dropna().to_numpy(dtype=float)
            mean, lo, hi = bootstrap_ci(vals, args.n_bootstrap, rng)
            rows.append({"method": method_name, "metric": m, "mean": mean, "ci_lo_95": lo, "ci_hi_95": hi, "n_topics": len(vals)})

    out_df = pd.DataFrame(rows)
    out_path = args.out if args.out.is_absolute() else root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"wrote {out_path}")

    # Pretty LaTeX table
    tex_path = out_path.with_suffix(".tex")
    pivot = out_df.pivot(index="method", columns="metric", values="mean").reindex(METHODS.keys())
    cilo  = out_df.pivot(index="method", columns="metric", values="ci_lo_95").reindex(METHODS.keys())
    cihi  = out_df.pivot(index="method", columns="metric", values="ci_hi_95").reindex(METHODS.keys())
    cells = pd.DataFrame(index=pivot.index, columns=pivot.columns, dtype=object)
    for r in pivot.index:
        for c in pivot.columns:
            cells.loc[r, c] = f"{pivot.loc[r,c]:.3f} [{cilo.loc[r,c]:.3f},{cihi.loc[r,c]:.3f}]"
    with open(tex_path, "w") as f:
        f.write(cells.to_latex(escape=True, column_format="l" + "c" * len(cells.columns)))
    print(f"wrote {tex_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
