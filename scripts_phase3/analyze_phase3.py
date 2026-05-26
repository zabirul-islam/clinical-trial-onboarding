"""
Phase 3 — aggregate & analysis.

Inputs:
  tables/phase3/judge_manifest.csv
  tables/phase3/judge_{sonnet,gpt4o}.jsonl
  tables/phase2/selector_signals_cache.csv
  tables/phase2/backbone_ablation_raw.csv

Outputs:
  tables/phase3/phase3_per_backbone.csv      # mean±95% CI per dim per judge
  tables/phase3/phase3_per_backbone_pooled.csv
  tables/phase3/phase3_signal_corr.csv       # judge score × selector signal (Spearman)
  tables/phase3/phase3_failure_modes.csv     # tag counts per backbone per judge
  tables/phase3/phase3_summary.json          # headline numbers for paper

Run:
  python scripts_phase3/analyze_phase3.py
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from scipy import stats

DIMS = ["factuality", "groundedness", "abstain_appropriateness",
        "safety", "patient_utility"]

SELECTOR_COLS = ["dominance_ratio", "trial_score_share",
                 "raw_max_cross", "best_rank"]


def boot_ci(x: np.ndarray, B: int = 5000, alpha: float = 0.05,
            seed: int = 13) -> tuple[float, float, float]:
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boots = rng.choice(x, size=(B, len(x)), replace=True).mean(axis=1)
    lo = float(np.quantile(boots, alpha / 2))
    hi = float(np.quantile(boots, 1 - alpha / 2))
    return float(x.mean()), lo, hi


def load_judges(P: Path) -> pd.DataFrame:
    rows = []
    for j in ["sonnet", "gpt4o"]:
        jf = P / f"judge_{j}.jsonl"
        if not jf.exists():
            continue
        for line in jf.read_text().splitlines():
            if not line.strip():
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            if "error" in d:
                continue
            base = {
                "judge": d["judge"],
                "case_id": d["case_id"],
                "backbone": d["backbone"],
                "source": d.get("source", ""),
                "decision": d.get("decision", ""),
                "failure_modes": d.get("failure_modes", []),
            }
            base.update({k: int(d["scores"][k]) for k in DIMS})
            rows.append(base)
    return pd.DataFrame(rows)


def per_backbone(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (judge, bb), sub in df.groupby(["judge", "backbone"]):
        for d in DIMS:
            m, lo, hi = boot_ci(sub[d].to_numpy(dtype=float))
            rows.append({"judge": judge, "backbone": bb, "dim": d,
                         "mean": m, "ci_lo": lo, "ci_hi": hi, "n": len(sub)})
    return pd.DataFrame(rows)


def pooled_per_backbone(df: pd.DataFrame) -> pd.DataFrame:
    """Average scores across the 2 judges for each (case, backbone) first,
    then bootstrap over cases."""
    wide = (df.pivot_table(index=["case_id", "backbone"],
                           columns="judge", values=DIMS, aggfunc="mean")
              .reset_index())
    # average over judges
    pooled = pd.DataFrame({"case_id": wide["case_id"],
                           "backbone": wide["backbone"]})
    for d in DIMS:
        cols = [(d, j) for j in ("sonnet", "gpt4o")
                if (d, j) in wide.columns]
        pooled[d] = wide[cols].mean(axis=1) if cols else np.nan
    rows = []
    for bb, sub in pooled.groupby("backbone"):
        for d in DIMS:
            m, lo, hi = boot_ci(sub[d].to_numpy(dtype=float))
            rows.append({"backbone": bb, "dim": d,
                         "mean": m, "ci_lo": lo, "ci_hi": hi, "n": len(sub)})
    return pd.DataFrame(rows)


def signal_correlations(df: pd.DataFrame,
                        selector: pd.DataFrame) -> pd.DataFrame:
    sel = selector[["case_id"] + SELECTOR_COLS].drop_duplicates("case_id")
    merged = df.merge(sel, on="case_id", how="left")
    rows = []
    for (judge, bb), sub in merged.groupby(["judge", "backbone"]):
        for d in DIMS:
            for sig in SELECTOR_COLS:
                sub2 = sub[[d, sig]].dropna()
                if len(sub2) < 5:
                    rho, p = float("nan"), float("nan")
                else:
                    try:
                        r = stats.spearmanr(sub2[d], sub2[sig])
                        rho, p = float(r.statistic), float(r.pvalue)
                    except Exception:
                        rho, p = float("nan"), float("nan")
                rows.append({"judge": judge, "backbone": bb,
                             "dim": d, "signal": sig,
                             "spearman_rho": rho, "p_value": p,
                             "n": len(sub2)})
    return pd.DataFrame(rows)


def failure_mode_counts(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (judge, bb), sub in df.groupby(["judge", "backbone"]):
        tag_counts: dict[str, int] = {}
        for tags in sub["failure_modes"]:
            if not isinstance(tags, list):
                continue
            for t in tags:
                t = str(t).strip().lower() or "none"
                tag_counts[t] = tag_counts.get(t, 0) + 1
        for tag, n in sorted(tag_counts.items(),
                             key=lambda kv: -kv[1]):
            rows.append({"judge": judge, "backbone": bb,
                         "failure_mode": tag, "count": n,
                         "share": n / max(len(sub), 1)})
    return pd.DataFrame(rows)


def headline_summary(per_bb_pool: pd.DataFrame,
                     backbone_raw: pd.DataFrame | None) -> dict:
    out = {"per_backbone_mean_overall": {}}
    for bb, sub in per_bb_pool.groupby("backbone"):
        out["per_backbone_mean_overall"][bb] = {
            "overall_mean": float(sub["mean"].mean()),
            "per_dim":      {r["dim"]: round(float(r["mean"]), 3)
                             for _, r in sub.iterrows()},
        }
    if backbone_raw is not None:
        out["phase2_crosscheck"] = (
            backbone_raw.groupby("backbone")
                        .agg(commit_rate=("decision",
                                          lambda s: float(s.isin(
                                              ["likely_match", "unlikely_match"]
                                          ).mean())),
                             abstain_rate=("decision",
                                           lambda s: float(s.isin(
                                               ["possible_match_insufficient_evidence",
                                                "cannot_determine"]).mean())),
                             parse_fail_rate=("decision",
                                              lambda s: float(
                                                  (s == "parse_fail").mean())),
                             leak_rate=("cross_trial_leak_n",
                                        lambda s: float((s > 0).mean())))
                        .reset_index().to_dict("records")
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[1]
    ap.add_argument("--phase3-dir", type=Path,
                    default=root / "tables" / "phase3")
    ap.add_argument("--phase2-dir", type=Path,
                    default=root / "tables" / "phase2")
    args = ap.parse_args()
    P3, P2 = args.phase3_dir, args.phase2_dir

    df = load_judges(P3)
    print(f"[load] {len(df)} judge rows  "
          f"({df['judge'].nunique()} judges × "
          f"{df['backbone'].nunique()} backbones × "
          f"~{df.groupby(['judge','backbone']).size().median():.0f} cases)")

    if df.empty:
        print("[!] no judge data")
        return 2

    per_bb = per_backbone(df)
    per_bb.to_csv(P3 / "phase3_per_backbone.csv", index=False)
    print("[wrote] phase3_per_backbone.csv")

    per_bb_pool = pooled_per_backbone(df)
    per_bb_pool.to_csv(P3 / "phase3_per_backbone_pooled.csv", index=False)
    print("[wrote] phase3_per_backbone_pooled.csv")

    sel_f = P2 / "selector_signals_cache.csv"
    if sel_f.exists():
        sel = pd.read_csv(sel_f)
        sig_corr = signal_correlations(df, sel)
        sig_corr.to_csv(P3 / "phase3_signal_corr.csv", index=False)
        print("[wrote] phase3_signal_corr.csv")
    else:
        print("  [skip] selector_signals_cache.csv not found; skipping corr")

    fm = failure_mode_counts(df)
    fm.to_csv(P3 / "phase3_failure_modes.csv", index=False)
    print("[wrote] phase3_failure_modes.csv")

    raw_f = P2 / "backbone_ablation_raw.csv"
    raw = pd.read_csv(raw_f) if raw_f.exists() else None
    summary = headline_summary(per_bb_pool, raw)
    (P3 / "phase3_summary.json").write_text(json.dumps(summary, indent=2))
    print("[wrote] phase3_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
