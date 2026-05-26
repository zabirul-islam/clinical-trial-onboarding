"""
T1.6c — Aggregate the unified n=114 backbone ablation (30 curated + 84 broad)
+ dual-judge rubric scores + therapeutic-area labels.

Produces paper-ready tables for §5:
  - per-backbone safety summary on n=114 (replaces Table 6)
  - per-backbone judge-pooled rubric scores with bootstrap 95% CIs on n=114
    (replaces Table 7)
  - per-regime stratification (selector-accept vs selector-abstain) — new
  - per-area × per-backbone safety heatmap data — new
  - failure-mode taxonomy on n=114 — refresh

Inputs:
  outputs/tables/backbone_ablation_raw.csv                            (120 rows: 30 curated × 4 backbones)
  outputs/phase4/n100_expansion/backbone_ablation_n100_raw.csv        (336 rows: 84 broad × 4 backbones)
  outputs/phase3/judge_sonnet.jsonl                                   (120 lines)
  outputs/phase3/judge_gpt4o.jsonl                                    (120 lines)
  outputs/phase4/n100_expansion/judge_sonnet.jsonl                    (336 lines)
  outputs/phase4/n100_expansion/judge_gpt4o.jsonl                     (336 lines)
  outputs/phase4/area_breakdown/area_labels_n114.csv                  (114 case→area)
  outputs/tables/selector_signals_cache.csv                           (regime tag source)

Outputs (all under outputs/phase4/n114_aggregate/):
  raw_n114.csv                          # merged 456-row generation table + regime + area
  summary_n114_per_backbone.csv         # leak/commit/abstain per backbone overall
  summary_n114_per_regime.csv           # × accept/abstain
  summary_n114_per_area.csv             # × area
  judge_pooled_n114.csv                 # mean ± 95% CI per (backbone × dim)
  judge_pooled_n114_per_regime.csv      # × regime
  failure_modes_n114.csv                # tag share per backbone

CPU only, ~10 sec.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

JUDGE_DIMS = [
    "factuality",
    "groundedness",
    "abstain_appropriateness",
    "safety",
    "patient_utility",
]

DEFAULT_GATE = {
    "rho_min": 1.35,
    "tau_min": 0.28,
    "mu_min": -4.5,
    "kappa_max": 2,
}


def load_jsonl(path: Path) -> list[dict]:
    out = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "error" in d:
                continue
            out.append(d)
    return out


def merge_raw(repo_root: Path) -> pd.DataFrame:
    """Combine 30-curated + 84-broad raw generation tables."""
    df_30 = pd.read_csv(repo_root / "outputs/tables/backbone_ablation_raw.csv")
    df_30["regime_pool"] = "curated_30"
    df_84 = pd.read_csv(repo_root / "outputs/phase4/n100_expansion/backbone_ablation_n100_raw.csv")
    df_84["regime_pool"] = "broad_84"
    raw = pd.concat([df_30, df_84], ignore_index=True)
    return raw


def merge_judges(repo_root: Path) -> pd.DataFrame:
    """Combine 30-curated + 84-broad dual-judge JSONLs into a long-form DataFrame."""
    rows = []
    sources = [
        ("curated_30", "sonnet", repo_root / "outputs/phase3/judge_sonnet.jsonl"),
        ("curated_30", "gpt4o", repo_root / "outputs/phase3/judge_gpt4o.jsonl"),
        ("broad_84", "sonnet", repo_root / "outputs/phase4/n100_expansion/judge_sonnet.jsonl"),
        ("broad_84", "gpt4o", repo_root / "outputs/phase4/n100_expansion/judge_gpt4o.jsonl"),
    ]
    for regime_pool, judge, path in sources:
        if not path.exists():
            print(f"[warn] missing {path}")
            continue
        for d in load_jsonl(path):
            scores = d.get("scores") or {}
            row = {
                "regime_pool": regime_pool,
                "case_id": d.get("case_id"),
                "backbone": d.get("backbone"),
                "judge": judge,
            }
            for dim in JUDGE_DIMS:
                v = scores.get(dim)
                row[dim] = float(v) if v is not None else np.nan
            row["failure_modes"] = ",".join(d.get("failure_modes") or [])
            rows.append(row)
    return pd.DataFrame(rows)


def attach_regime_gate(raw: pd.DataFrame, repo_root: Path) -> pd.DataFrame:
    """Tag each row by selector-gate regime (accept vs abstain) using cached signals."""
    cache = pd.read_csv(repo_root / "outputs/tables/selector_signals_cache.csv")
    pass_mask = (
        (cache["dominance_ratio"] >= DEFAULT_GATE["rho_min"])
        & (cache["trial_score_share"] >= DEFAULT_GATE["tau_min"])
        & (cache["raw_max_cross"] >= DEFAULT_GATE["mu_min"])
        & (cache["best_rank"] <= DEFAULT_GATE["kappa_max"])
        & (~cache["generic_question"].fillna(False).astype(bool))
    )
    cache["regime_gate"] = np.where(pass_mask, "accept", "abstain")
    return raw.merge(cache[["case_id", "regime_gate"]], on="case_id", how="left")


def attach_area(raw: pd.DataFrame, repo_root: Path) -> pd.DataFrame:
    areas = pd.read_csv(repo_root / "outputs/phase4/area_breakdown/area_labels_n114.csv")[
        ["case_id", "area"]
    ]
    return raw.merge(areas, on="case_id", how="left")


def per_backbone_summary(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Compute leak/commit/abstain rates aggregated over group_cols."""
    def commit(s):
        return s.isin(["likely_match", "possible_match_insufficient_evidence"]).mean()

    def abstain(s):
        return (s == "cannot_determine").mean()

    def leak(s):
        return (s > 0).mean()

    grouped = df.groupby(group_cols).agg(
        n=("case_id", "nunique"),
        rows=("case_id", "size"),
        parse_ok=("parse_ok", "mean"),
        commit_rate=("decision", commit),
        abstain_rate=("decision", abstain),
        leak_rate=("cross_trial_leak_n", leak),
        mean_latency=("latency_sec", "mean"),
        mean_answer_chars=("answer_chars", "mean"),
    ).reset_index()
    return grouped


def bootstrap_judge_pooled(
    judges_df: pd.DataFrame,
    group_cols: list[str],
    n_resamples: int = 5000,
    seed: int = 42,
    ci: float = 95.0,
) -> pd.DataFrame:
    """Pool sonnet + gpt4o per (case, backbone), then bootstrap over cases."""
    rng = np.random.default_rng(seed)
    pooled = (
        judges_df.groupby(group_cols + ["case_id"])[JUDGE_DIMS].mean().reset_index()
    )
    rows = []
    lo_p = (100 - ci) / 2.0
    hi_p = 100 - lo_p
    for keys, sub in pooled.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        n = len(sub)
        if n == 0:
            continue
        scores_per_case = sub[JUDGE_DIMS].to_numpy()  # (n_cases, 5)
        means = []
        for _ in range(n_resamples):
            idx = rng.integers(0, n, size=n)
            means.append(scores_per_case[idx].mean(axis=0))
        means = np.array(means)  # (n_resamples, 5)
        lo = np.nanpercentile(means, lo_p, axis=0)
        hi = np.nanpercentile(means, hi_p, axis=0)
        m = scores_per_case.mean(axis=0)
        row = dict(zip(group_cols, keys))
        row["n_cases"] = int(n)
        for i, dim in enumerate(JUDGE_DIMS):
            row[f"{dim}_mean"] = float(m[i])
            row[f"{dim}_ci_lo"] = float(lo[i])
            row[f"{dim}_ci_hi"] = float(hi[i])
        rows.append(row)
    return pd.DataFrame(rows)


def failure_mode_share(judges_df: pd.DataFrame) -> pd.DataFrame:
    """Per-backbone share of cases flagged with each failure tag, judge-pooled."""
    if "failure_modes" not in judges_df.columns:
        return pd.DataFrame()
    tags = set()
    for s in judges_df["failure_modes"].dropna():
        for t in s.split(","):
            t = t.strip()
            if t:
                tags.add(t)

    rows = []
    for backbone, sub in judges_df.groupby("backbone"):
        n = sub["case_id"].nunique() * sub["judge"].nunique()  # rows = cases × judges
        for tag in sorted(tags):
            count = sub["failure_modes"].fillna("").apply(
                lambda s: tag in [t.strip() for t in s.split(",")]
            ).sum()
            rows.append(
                {
                    "backbone": backbone,
                    "tag": tag,
                    "count": int(count),
                    "share": count / n if n else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=str, default=".")
    ap.add_argument("--out-dir", type=str, default="outputs/phase4/n114_aggregate")
    ap.add_argument("--n-resamples", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    root = Path(args.repo_root).resolve()
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Merge raw generations
    raw = merge_raw(root)
    print(f"[raw] {len(raw)} rows = "
          f"{(raw['regime_pool']=='curated_30').sum()} curated_30 + "
          f"{(raw['regime_pool']=='broad_84').sum()} broad_84")

    # 2. Attach regime + area
    raw = attach_regime_gate(raw, root)
    raw = attach_area(raw, root)
    raw.to_csv(out_dir / "raw_n114.csv", index=False)
    print(f"[wrote] {out_dir / 'raw_n114.csv'}")

    # 3. Per-backbone overall summary
    s_overall = per_backbone_summary(raw, ["backbone"])
    s_overall.to_csv(out_dir / "summary_n114_per_backbone.csv", index=False)
    print(f"\n=== summary_n114_per_backbone ===")
    print(s_overall.to_string(index=False))

    # 4. Per regime_pool × backbone
    s_pool = per_backbone_summary(raw, ["regime_pool", "backbone"])
    s_pool.to_csv(out_dir / "summary_n114_per_regime_pool.csv", index=False)
    print(f"\n=== summary_n114_per_regime_pool ===")
    print(s_pool.to_string(index=False))

    # 5. Per gate-regime × backbone (selector accept vs abstain — cleaner story)
    s_gate = per_backbone_summary(raw, ["regime_gate", "backbone"])
    s_gate.to_csv(out_dir / "summary_n114_per_regime_gate.csv", index=False)
    print(f"\n=== summary_n114_per_regime_gate ===")
    print(s_gate.to_string(index=False))

    # 6. Per area × backbone
    s_area = per_backbone_summary(raw.dropna(subset=["area"]), ["area", "backbone"])
    s_area.to_csv(out_dir / "summary_n114_per_area.csv", index=False)
    print(f"\n=== summary_n114_per_area (head) ===")
    print(s_area.head(12).to_string(index=False))

    # 7. Judge-pooled scores with bootstrap CIs
    # drop regime_pool from judges (set by source path) so raw's column is authoritative
    judges = merge_judges(root).drop(columns=["regime_pool"], errors="ignore")
    judges = judges.merge(
        raw[["case_id", "backbone", "regime_gate", "regime_pool", "area"]].drop_duplicates(),
        on=["case_id", "backbone"],
        how="left",
    )
    print(f"\n[judges] {len(judges)} rows after merge with regime/area")

    pooled = bootstrap_judge_pooled(
        judges, ["backbone"], n_resamples=args.n_resamples, seed=args.seed
    )
    pooled.to_csv(out_dir / "judge_pooled_n114.csv", index=False)
    print(f"\n=== judge_pooled_n114 ===")
    print(pooled.to_string(index=False))

    pooled_regime = bootstrap_judge_pooled(
        judges, ["regime_gate", "backbone"], n_resamples=args.n_resamples, seed=args.seed
    )
    pooled_regime.to_csv(out_dir / "judge_pooled_n114_per_regime_gate.csv", index=False)

    pooled_pool = bootstrap_judge_pooled(
        judges, ["regime_pool", "backbone"], n_resamples=args.n_resamples, seed=args.seed
    )
    pooled_pool.to_csv(out_dir / "judge_pooled_n114_per_regime_pool.csv", index=False)

    # 8. Failure-mode taxonomy
    fm = failure_mode_share(judges)
    fm.to_csv(out_dir / "failure_modes_n114.csv", index=False)
    print(f"\n=== failure_modes_n114 (top tags per backbone) ===")
    if not fm.empty:
        for backbone, sub in fm.groupby("backbone"):
            top = sub.sort_values("share", ascending=False).head(3)
            print(f"  {backbone}:")
            for _, r in top.iterrows():
                print(f"    {r['tag']:<22s}: {r['share']:.3f} ({r['count']})")

    # 9. Headline summary JSON
    headline = {
        "n_cases": int(raw["case_id"].nunique()),
        "n_rows_total": int(len(raw)),
        "leak_rate_overall_per_backbone": s_overall.set_index("backbone")["leak_rate"].to_dict(),
        "n_with_zero_leak_per_backbone": s_overall.set_index("backbone")["leak_rate"]
            .apply(lambda x: x == 0)
            .to_dict(),
        "regime_split": {
            "n_accept": int((raw["regime_gate"] == "accept").sum() // raw["backbone"].nunique()),
            "n_abstain": int((raw["regime_gate"] == "abstain").sum() // raw["backbone"].nunique()),
        },
        "regime_pool_split": {
            "curated_30": int((raw["regime_pool"] == "curated_30").sum() // raw["backbone"].nunique()),
            "broad_84": int((raw["regime_pool"] == "broad_84").sum() // raw["backbone"].nunique()),
        },
        "area_distribution": (
            raw.drop_duplicates("case_id")["area"].value_counts().to_dict()
        ),
    }
    with (out_dir / "headline_n114.json").open("w") as f:
        json.dump(headline, f, indent=2)
    print(f"\n=== HEADLINE n=114 ===")
    print(json.dumps(headline, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
