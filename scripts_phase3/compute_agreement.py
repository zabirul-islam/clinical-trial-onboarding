"""
Phase 3 — Inter-rater agreement between Sonnet and GPT-4o judges.

Inputs (produced by run_llm_judge.py):
  tables/phase3/judge_sonnet.jsonl
  tables/phase3/judge_gpt4o.jsonl
  tables/phase3/judge_manifest.csv  (optional — rebuilt if missing)

Outputs:
  tables/phase3/agreement_per_dim.csv        # κ_linear, κ_quadratic, ICC(2,1), Spearman ρ, |Δ|, MAE, exact_match
  tables/phase3/agreement_disagreements.csv  # cases where |Δ|>=2 on any dim (for qual review)
  tables/phase3/agreement_summary.json       # macro-averages + N

Metrics:
  Cohen's κ (linear + quadratic weights) treating 1-5 as ordinal.
  ICC(2,1)       — two-way random absolute agreement.
  Spearman ρ     — monotonic agreement.
  exact_match    — fraction identical.
  MAE, mean |Δ|.

Run:
  python scripts_phase3/compute_agreement.py
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import cohen_kappa_score

DIMS = ["factuality", "groundedness", "abstain_appropriateness",
        "safety", "patient_utility"]


def load_judge(jsonl: Path, judge_name: str) -> pd.DataFrame:
    rows = []
    for line in jsonl.read_text().splitlines():
        if not line.strip():
            continue
        try:
            d = json.loads(line)
        except Exception:
            continue
        if "error" in d:
            continue
        base = {
            "case_id": d["case_id"],
            "backbone": d["backbone"],
            "judge": d["judge"],
        }
        base.update({k: int(d["scores"][k]) for k in DIMS})
        rows.append(base)
    df = pd.DataFrame(rows)
    df = df[df["judge"] == judge_name]
    # dedupe → keep last
    df = df.drop_duplicates(["case_id", "backbone"], keep="last")
    return df.reset_index(drop=True)


def icc_2_1(y1: np.ndarray, y2: np.ndarray) -> float:
    """ICC(2,1) — two-way random, absolute agreement, single rater.
    Shrout & Fleiss (1979). Returns np.nan on degenerate input."""
    Y = np.stack([y1, y2], axis=1).astype(float)   # n × k=2
    n, k = Y.shape
    if n < 2:
        return float("nan")
    mean_row = Y.mean(axis=1, keepdims=True)
    mean_col = Y.mean(axis=0, keepdims=True)
    mean_all = Y.mean()
    SSR = k * ((mean_row - mean_all) ** 2).sum()
    SSC = n * ((mean_col - mean_all) ** 2).sum()
    SST = ((Y - mean_all) ** 2).sum()
    SSE = SST - SSR - SSC
    MSR = SSR / (n - 1)
    MSC = SSC / (k - 1) if k > 1 else 0.0
    MSE = SSE / ((n - 1) * (k - 1)) if (n - 1) * (k - 1) > 0 else 0.0
    denom = MSR + (k - 1) * MSE + k * (MSC - MSE) / n
    if denom <= 0:
        return float("nan")
    return float((MSR - MSE) / denom)


def agreement_for_dim(s: np.ndarray, g: np.ndarray) -> dict:
    mask = (~np.isnan(s)) & (~np.isnan(g))
    s, g = s[mask], g[mask]
    if len(s) < 2:
        return {k: float("nan") for k in
                ["kappa_linear", "kappa_quadratic", "icc21", "spearman",
                 "exact_match", "mae", "mean_abs_delta", "n"]}
    kl = cohen_kappa_score(s, g, weights="linear")
    kq = cohen_kappa_score(s, g, weights="quadratic")
    icc = icc_2_1(s, g)
    try:
        rho = stats.spearmanr(s, g).statistic
    except Exception:
        rho = float("nan")
    return {
        "kappa_linear":    float(kl),
        "kappa_quadratic": float(kq),
        "icc21":           icc,
        "spearman":        float(rho) if rho is not None else float("nan"),
        "exact_match":     float((s == g).mean()),
        "mae":             float(np.mean(np.abs(s - g))),
        "mean_abs_delta":  float(np.mean(np.abs(s - g))),
        "n":               int(len(s)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase3-dir", type=Path,
                    default=Path(__file__).resolve().parents[1]
                            / "tables" / "phase3")
    args = ap.parse_args()
    P = args.phase3_dir

    s = load_judge(P / "judge_sonnet.jsonl", "sonnet")
    g = load_judge(P / "judge_gpt4o.jsonl",  "gpt4o")
    print(f"[load] sonnet {len(s)}  gpt4o {len(g)}")

    merged = s.merge(g, on=["case_id", "backbone"],
                     suffixes=("_sonnet", "_gpt4o"))
    print(f"[merge] {len(merged)} paired (case×backbone)")

    if len(merged) < 2:
        print("[!] too few paired rows")
        return 2

    # per-dim agreement
    rows = []
    for d in DIMS:
        stats_d = agreement_for_dim(
            merged[f"{d}_sonnet"].to_numpy(dtype=float),
            merged[f"{d}_gpt4o"].to_numpy(dtype=float),
        )
        rows.append({"dim": d, **stats_d})
    per_dim = pd.DataFrame(rows)
    per_dim.to_csv(P / "agreement_per_dim.csv", index=False)
    print("[wrote] agreement_per_dim.csv")
    print(per_dim.round(3).to_string(index=False))

    # disagreements — |Δ| ≥ 2 on any dim
    dis_mask = np.zeros(len(merged), dtype=bool)
    for d in DIMS:
        dis_mask |= (merged[f"{d}_sonnet"] - merged[f"{d}_gpt4o"]).abs() >= 2
    dis = merged[dis_mask].copy()
    for d in DIMS:
        dis[f"{d}_delta"] = dis[f"{d}_sonnet"] - dis[f"{d}_gpt4o"]
    keep_cols = (["case_id", "backbone"]
                 + [f"{d}_sonnet" for d in DIMS]
                 + [f"{d}_gpt4o"  for d in DIMS]
                 + [f"{d}_delta"  for d in DIMS])
    dis[keep_cols].to_csv(P / "agreement_disagreements.csv", index=False)
    print(f"[wrote] agreement_disagreements.csv ({len(dis)} rows)")

    # summary json
    summary = {
        "n_paired": int(len(merged)),
        "macro_kappa_linear":    float(per_dim["kappa_linear"].mean()),
        "macro_kappa_quadratic": float(per_dim["kappa_quadratic"].mean()),
        "macro_icc21":           float(per_dim["icc21"].mean()),
        "macro_spearman":        float(per_dim["spearman"].mean()),
        "macro_exact_match":     float(per_dim["exact_match"].mean()),
        "macro_mae":             float(per_dim["mae"].mean()),
        "per_dim": per_dim.set_index("dim").to_dict("index"),
    }
    (P / "agreement_summary.json").write_text(json.dumps(summary, indent=2))
    print("[wrote] agreement_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
