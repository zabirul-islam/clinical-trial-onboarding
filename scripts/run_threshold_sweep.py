"""
Phase 2.2 — Threshold grid sweep over cached selector signals.
Scoring now uses full TREC qrels (any-relevant ≥1, strict ≥2) instead of
single-gold collapse.

Input : outputs/tables/selector_signals_cache.csv  (must have
        selected_in_qrels_any + selected_in_qrels_strict cols)

Outputs:
  outputs/tables/threshold_sweep_grid.csv
  outputs/tables/threshold_sweep_pareto.csv
  outputs/tables/threshold_sweep_summary.json

Run:
  python scripts/run_threshold_sweep.py --repo-root .
"""
from __future__ import annotations
import argparse, json, sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


DEFAULTS = {
    "min_dominance_ratio":          1.35,
    "min_trial_score_share":        0.28,
    "min_selected_raw_max_cross":  -4.50,   # adjusted: MiniLM logits ~[-8,+4]
    "max_best_rank":                2,
}

GRID_RHO = [1.00, 1.15, 1.25, 1.35, 1.50, 1.75, 2.00, 2.50]
GRID_TAU = [0.15, 0.20, 0.25, 0.28, 0.32, 0.40, 0.50]
GRID_MU  = [-7.0, -6.0, -5.0, -4.5, -4.0, -3.0, -2.0, 0.0, 1.0]
GRID_KAP = [1, 2, 3, 5, 999]


def evaluate(df: pd.DataFrame, rho: float, tau: float, mu: float, kap: int) -> dict:
    accept_mask = (
        (df["dominance_ratio"]   >= rho) &
        (df["trial_score_share"] >= tau) &
        (df["raw_max_cross"]     >= mu)  &
        (df["best_rank"]         <= kap) &
        (~df["generic_question"].astype(bool))
    )
    n = len(df)
    n_accept = int(accept_mask.sum())

    # qrel-based scoring on TREC cases only
    is_trec = df["source"].astype(str).str.startswith("trec")
    df_t = df[is_trec].copy()
    accept_t = accept_mask[is_trec]
    rel_any = df_t["selected_in_qrels_any"].fillna(False).astype(bool)
    rel_str = df_t["selected_in_qrels_strict"].fillna(False).astype(bool)

    # any-relevance track
    tp_a = int(( rel_any &  accept_t).sum())
    fp_a = int((~rel_any &  accept_t).sum())
    fn_a = int(( rel_any & ~accept_t).sum())
    tn_a = int((~rel_any & ~accept_t).sum())
    prec_a = tp_a / (tp_a + fp_a) if (tp_a + fp_a) else 0.0
    rec_a  = tp_a / (tp_a + fn_a) if (tp_a + fn_a) else 0.0
    f1_a   = 2*prec_a*rec_a/(prec_a+rec_a) if (prec_a+rec_a) else 0.0

    # strict (grade≥2) track
    tp_s = int(( rel_str &  accept_t).sum())
    fp_s = int((~rel_str &  accept_t).sum())
    fn_s = int(( rel_str & ~accept_t).sum())
    prec_s = tp_s / (tp_s + fp_s) if (tp_s + fp_s) else 0.0
    rec_s  = tp_s / (tp_s + fn_s) if (tp_s + fn_s) else 0.0
    f1_s   = 2*prec_s*rec_s/(prec_s+rec_s) if (prec_s+rec_s) else 0.0

    # safety on generic questions
    generic_mask = df["generic_question"].astype(bool)
    n_generic = int(generic_mask.sum())
    n_generic_accept = int((accept_mask & generic_mask).sum())
    generic_abstain_rate = 1.0 - (n_generic_accept/n_generic) if n_generic else 1.0

    return {
        "rho": rho, "tau": tau, "mu": mu, "kap": kap,
        "n": n, "n_accept": n_accept,
        "accept_rate":  n_accept/n,
        "abstain_rate": 1.0 - n_accept/n,
        # any-relevance
        "tp_any": tp_a, "fp_any": fp_a, "fn_any": fn_a, "tn_any": tn_a,
        "precision_any": prec_a, "recall_any": rec_a, "f1_any": f1_a,
        # strict
        "tp_strict": tp_s, "fp_strict": fp_s, "fn_strict": fn_s,
        "precision_strict": prec_s, "recall_strict": rec_s, "f1_strict": f1_s,
        # safety
        "n_generic":             n_generic,
        "generic_accepted":      n_generic_accept,
        "generic_abstain_rate":  generic_abstain_rate,
    }


def pareto_front(rows, maxi: str, mini: str):
    front = []
    for r in rows:
        dom = False
        for q in rows:
            if (q[maxi] >= r[maxi] and q[mini] <= r[mini]
                and (q[maxi] > r[maxi] or q[mini] < r[mini])):
                dom = True; break
        if not dom: front.append(r)
    return sorted(front, key=lambda r: r[maxi])


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", type=Path, default=Path("."))
    p.add_argument("--cache", type=str, default="outputs/tables/selector_signals_cache.csv")
    args = p.parse_args()

    root = args.repo_root.resolve()
    df = pd.read_csv(root / args.cache)
    print(f"[load] {len(df):,} cached signals")

    df = df[df["dominance_ratio"].notna()].reset_index(drop=True)
    df["generic_question"] = df["generic_question"].fillna(False).astype(bool)
    for c in ["selected_in_qrels_any","selected_in_qrels_strict"]:
        if c not in df.columns:
            print(f"[error] missing {c}. Run patch_cache_with_qrels first.", file=sys.stderr)
            return 1
    print(f"[clean] {len(df):,} cases with valid signals")

    rows = []
    combos = list(product(GRID_RHO, GRID_TAU, GRID_MU, GRID_KAP))
    for rho, tau, mu, kap in tqdm(combos, desc="sweep"):
        rows.append(evaluate(df, rho, tau, mu, kap))

    grid = pd.DataFrame(rows)
    out_dir = root / "outputs/tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    grid.to_csv(out_dir / "threshold_sweep_grid.csv", index=False)
    print(f"[wrote] grid: {len(grid):,} rows → threshold_sweep_grid.csv")

    grid["unsafety"] = 1.0 - grid["generic_abstain_rate"]
    pareto = pareto_front(grid.to_dict(orient="records"), maxi="f1_any", mini="unsafety")
    pareto_df = pd.DataFrame(pareto)
    pareto_df.to_csv(out_dir / "threshold_sweep_pareto.csv", index=False)
    print(f"[wrote] pareto: {len(pareto_df):,} rows → threshold_sweep_pareto.csv")

    best_any    = grid.sort_values("f1_any",    ascending=False).iloc[0].to_dict()
    best_strict = grid.sort_values("f1_strict", ascending=False).iloc[0].to_dict()
    default_row = evaluate(df,
        DEFAULTS["min_dominance_ratio"], DEFAULTS["min_trial_score_share"],
        DEFAULTS["min_selected_raw_max_cross"], DEFAULTS["max_best_rank"])

    summary = {
        "n_cases":      int(len(df)),
        "grid_size":    int(len(grid)),
        "pareto_size":  int(len(pareto_df)),
        "defaults":     DEFAULTS,
        "default_metrics":    default_row,
        "best_f1_any":    best_any,
        "best_f1_strict": best_strict,
    }
    with open(out_dir / "threshold_sweep_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(json.dumps(summary, indent=2, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
