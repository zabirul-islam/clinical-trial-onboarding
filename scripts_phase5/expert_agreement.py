#!/usr/bin/env python3
"""
Phase 5 Task 2 — analyze filled expert scoring sheets.

Reads outputs/expert_review/scoring_sheet_filled.csv (+ blind_key.csv to unblind)
and computes:
  - expert cross-trial leak rate + Clopper-Pearson 95% CI (overall, per system)
  - failure-type (T1/T2/T3) distribution per system
  - expert vs LLM-judge agreement per rubric dimension (Spearman, quadratic kappa)

Unit-testable: run with --selftest to validate on a synthetic filled sheet.

Run (after experts return sheets):
  python scripts_phase5/expert_agreement.py
  python scripts_phase5/expert_agreement.py --selftest
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "expert_review"
DIMS = ["factuality", "groundedness", "abstain_appropriateness", "safety", "patient_utility"]


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    from scipy.stats import beta
    lo = 0.0 if k == 0 else beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - alpha / 2, k + 1, n - k)
    return float(lo), float(hi)


def quad_kappa(a: pd.Series, b: pd.Series) -> float:
    from sklearn.metrics import cohen_kappa_score
    m = a.notna() & b.notna()
    if m.sum() < 2:
        return float("nan")
    return float(cohen_kappa_score(a[m].round().astype(int), b[m].round().astype(int),
                                   weights="quadratic"))


def analyze(sheet: pd.DataFrame, key: pd.DataFrame) -> dict:
    df = sheet.merge(key, on="blind_id", how="left")
    leak = df["cross_trial_leak_yes_no"].astype(str).str.lower().str.startswith("y")
    n, k = len(df), int(leak.sum())
    lo, hi = clopper_pearson(k, n)
    res = {"n": n, "expert_leak_rate": k / n if n else float("nan"),
           "leak_ci95": (round(lo, 4), round(hi, 4)),
           "per_system_leak": df.assign(leak=leak).groupby("system")["leak"].mean().to_dict(),
           "failure_type_dist": df.groupby("system")["failure_type_T1_T2_T3_or_none"]
               .value_counts().to_dict()}
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet", type=Path, default=OUT / "scoring_sheet_filled.csv")
    ap.add_argument("--key", type=Path, default=OUT / "blind_key.csv")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        rng = np.random.default_rng(0)
        bids = [f"b{i:03d}" for i in range(12)]
        key = pd.DataFrame({"blind_id": bids,
                            "system": ["V-final"] * 6 + ["B1_multi_rag"] * 6,
                            "case_id": bids, "backbone": "Qwen/Qwen2.5-7B-Instruct"})
        sheet = pd.DataFrame({"blind_id": bids})
        for d in DIMS:
            sheet[f"{d}_1to5"] = rng.integers(1, 6, len(bids))
        sheet["cross_trial_leak_yes_no"] = (["no"] * 9) + ["yes"] * 3
        sheet["failure_type_T1_T2_T3_or_none"] = (["none"] * 9) + ["T1", "T2", "T1"]
        res = analyze(sheet, key)
        assert res["n"] == 12
        assert abs(res["expert_leak_rate"] - 0.25) < 1e-9
        assert 0.0 <= res["leak_ci95"][0] <= res["leak_ci95"][1] <= 1.0
        print("[selftest] PASS", res)
        return 0

    if not args.sheet.exists():
        print(f"[wait] no filled sheet yet at {args.sheet} — run after experts return data.")
        return 0
    sheet = pd.read_csv(args.sheet)
    key = pd.read_csv(args.key)
    res = analyze(sheet, key)
    print(res)

    # expert vs LLM-judge per-dim agreement (V-final cases only, where judges scored)
    try:
        jr = pd.read_csv(ROOT / "outputs/phase4/n114_aggregate/judge_pooled_n114.csv")
        merged = sheet.merge(key, on="blind_id").merge(
            jr, on=["case_id", "backbone"], how="left", suffixes=("_exp", "_llm"))
        print("\n[expert vs LLM-judge] quadratic kappa / Spearman per dim:")
        for d in DIMS:
            ec, lc = f"{d}_1to5", f"{d}_mean"
            if ec in merged and lc in merged:
                m = merged[[ec, lc]].dropna()
                rho = m[ec].corr(m[lc], method="spearman") if len(m) > 2 else float("nan")
                print(f"  {d}: kappa={quad_kappa(merged[ec], merged[lc]):.3f}  rho={rho:.3f}")
    except FileNotFoundError:
        print("[skip] judge_pooled_n114.csv not found for agreement.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
