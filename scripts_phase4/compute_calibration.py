"""
T1.4 — Calibration metrics for the trial-first selector.

Inputs (server-relative):
  outputs/tables/selector_signals_cache.csv
    Required cols: case_id, trial_score_share (= τ),
                   selected_matches_gold, selected_in_qrels_any,
                   selected_in_qrels_strict, gold_nct, generic_question.

Method:
  - p_pred = τ (trial_score_share). This is the system's continuous
    confidence signal in "the dominant trial is the right one".
  - y_true defined by three escalating relevance thresholds:
      strict   : selected_in_qrels_strict (qrels rel == 2)
      any      : selected_in_qrels_any    (qrels rel ≥ 1)
      gold_nct : selected_matches_gold    (selected == gold NCT)
  - Filter: only rows with gold_nct populated (we need ground truth).
  - For each y_true variant, compute:
      ECE (10-bin), Brier, base rate, n
      reliability diagram (predicted-prob bin → empirical accuracy)
  - Optional split: generic_question == False (selector active) vs all.

Outputs:
  outputs/phase4/calibration/calibration_summary.json
  outputs/phase4/calibration/calibration_per_bin.csv      (per-bin counts)
  outputs/phase4/calibration/reliability_diagram.png
  outputs/phase4/calibration/reliability_diagram.pdf

Usage:
  python scripts_phase4/compute_calibration.py \\
      --selector-cache outputs/tables/selector_signals_cache.csv \\
      --out-dir outputs/phase4/calibration

CPU only, no API. ~5 sec wall.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# Use Agg backend so it works headless on the server.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


Y_TRUE_VARIANTS: list[tuple[str, str]] = [
    ("strict", "selected_in_qrels_strict"),
    ("any", "selected_in_qrels_any"),
    ("gold_nct", "selected_matches_gold"),
]


def _to_bool(s: pd.Series) -> pd.Series:
    """Coerce TRUE/FALSE/'True'/'False'/1/0 to bool."""
    if s.dtype == bool:
        return s
    return s.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def expected_calibration_error(
    p_pred: np.ndarray, y_true: np.ndarray, n_bins: int = 10
) -> tuple[float, pd.DataFrame]:
    """ECE with equal-width bins on [0, 1]; per-bin diagnostic frame."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    inds = np.digitize(p_pred, bins) - 1
    inds = np.clip(inds, 0, n_bins - 1)
    rows = []
    n_total = len(p_pred)
    ece = 0.0
    for b in range(n_bins):
        mask = inds == b
        n_b = int(mask.sum())
        if n_b == 0:
            rows.append(
                {
                    "bin_lo": float(bins[b]),
                    "bin_hi": float(bins[b + 1]),
                    "n": 0,
                    "frac": 0.0,
                    "mean_p_pred": np.nan,
                    "mean_y_true": np.nan,
                    "abs_gap": np.nan,
                }
            )
            continue
        mean_p = float(p_pred[mask].mean())
        mean_y = float(y_true[mask].mean())
        gap = abs(mean_p - mean_y)
        ece += (n_b / n_total) * gap
        rows.append(
            {
                "bin_lo": float(bins[b]),
                "bin_hi": float(bins[b + 1]),
                "n": n_b,
                "frac": n_b / n_total,
                "mean_p_pred": mean_p,
                "mean_y_true": mean_y,
                "abs_gap": gap,
            }
        )
    return float(ece), pd.DataFrame(rows)


def brier_score(p_pred: np.ndarray, y_true: np.ndarray) -> float:
    return float(np.mean((p_pred - y_true) ** 2))


def base_rate(y_true: np.ndarray) -> float:
    return float(np.mean(y_true)) if len(y_true) else float("nan")


def plot_reliability(
    bin_dfs: dict[str, pd.DataFrame],
    out_path: Path,
    n_bins: int = 10,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    width = 1.0 / n_bins * 0.8
    colors = {"strict": "#d62728", "any": "#1f77b4", "gold_nct": "#2ca02c"}

    # Diagonal
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1, label="Perfect calibration")

    for label, df in bin_dfs.items():
        valid = df.dropna(subset=["mean_p_pred", "mean_y_true"])
        ax.plot(
            valid["mean_p_pred"],
            valid["mean_y_true"],
            "o-",
            color=colors.get(label, "black"),
            markersize=8,
            linewidth=1.5,
            label=f"y_true = {label}",
        )

    ax.set_xlabel("Predicted prob (trial-score share τ)")
    ax.set_ylabel("Empirical fraction relevant")
    ax.set_title("Reliability diagram — selector calibration")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--selector-cache",
        type=str,
        default="outputs/tables/selector_signals_cache.csv",
    )
    ap.add_argument("--out-dir", type=str, default="outputs/phase4/calibration")
    ap.add_argument("--n-bins", type=int, default=10)
    ap.add_argument(
        "--filter-non-generic",
        action="store_true",
        help="if set, only score cases where generic_question == False",
    )
    args = ap.parse_args()

    cache_path = Path(args.selector_cache)
    if not cache_path.exists():
        raise FileNotFoundError(cache_path)
    df = pd.read_csv(cache_path)
    print(f"[load] {len(df)} rows from {cache_path}")

    # Keep only rows with gold_nct (we need ground truth).
    has_gold = df["gold_nct"].notna() & (df["gold_nct"].astype(str).str.startswith("NCT"))
    df = df[has_gold].reset_index(drop=True)
    print(f"[filter] kept {len(df)} rows with gold_nct populated")

    if args.filter_non_generic and "generic_question" in df.columns:
        before = len(df)
        df = df[~_to_bool(df["generic_question"])].reset_index(drop=True)
        print(f"[filter] kept {len(df)}/{before} rows where generic_question == False")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Compute per y_true variant
    summary: dict = {
        "n_total": int(len(df)),
        "filter_non_generic": bool(args.filter_non_generic),
        "y_true_variants": {},
    }
    bin_dfs: dict[str, pd.DataFrame] = {}
    bin_csv_rows: list[dict] = []

    p_pred = df["trial_score_share"].astype(float).clip(0, 1).to_numpy()

    for label, col in Y_TRUE_VARIANTS:
        if col not in df.columns:
            print(f"[skip] {label}: column {col} missing")
            continue
        y = _to_bool(df[col]).astype(int).to_numpy()
        ece, bin_df = expected_calibration_error(p_pred, y, n_bins=args.n_bins)
        brier = brier_score(p_pred, y.astype(float))
        br = base_rate(y.astype(float))

        # Naive baseline: predict the base rate for every sample.
        # Brier of constant prediction = base_rate * (1 - base_rate).
        brier_naive = br * (1.0 - br)
        # Skill score relative to base-rate predictor.
        brier_skill = 1.0 - (brier / brier_naive) if brier_naive > 0 else float("nan")

        summary["y_true_variants"][label] = {
            "y_true_col": col,
            "n": int(len(y)),
            "base_rate": br,
            "ece": ece,
            "brier": brier,
            "brier_naive": brier_naive,
            "brier_skill_score": brier_skill,
        }
        bin_dfs[label] = bin_df
        for _, row in bin_df.iterrows():
            bin_csv_rows.append({"y_true_variant": label, **row.to_dict()})

        print(
            f"[{label:>9s}] n={len(y):>3d} base_rate={br:.3f} "
            f"ECE={ece:.4f} Brier={brier:.4f} BrierNaive={brier_naive:.4f} "
            f"BSS={brier_skill:+.3f}"
        )

    # Persist
    pd.DataFrame(bin_csv_rows).to_csv(out_dir / "calibration_per_bin.csv", index=False)
    print(f"[wrote] {out_dir / 'calibration_per_bin.csv'}")

    with (out_dir / "calibration_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"[wrote] {out_dir / 'calibration_summary.json'}")

    plot_reliability(bin_dfs, out_dir / "reliability_diagram", n_bins=args.n_bins)
    print(f"[wrote] {out_dir / 'reliability_diagram.pdf'} and .png")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
