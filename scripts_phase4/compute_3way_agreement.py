"""
Compute 3-way inter-rater agreement (Human / Sonnet / GPT-4o) on the 15-case
audit, using a 10-dim → 5-dim crosswalk so the human's rubric and the judge
rubric land on a common 1–5 scale.

Inputs (server-relative):
  data/processed/onboarding_eval_audit_template_v2.csv   <- V1 human ratings
  data/processed/onboarding_eval_audit_v2.csv            <- V2 human ratings
  data/processed/onboarding_eval_audit_final.csv         <- V-final human ratings
  outputs/phase4/15case_audit/judge_sonnet.jsonl
  outputs/phase4/15case_audit/judge_gpt4o.jsonl

Outputs:
  outputs/phase4/15case_audit/3way_agreement_per_dim.csv
  outputs/phase4/15case_audit/3way_agreement_summary.json
  outputs/phase4/15case_audit/3way_audit_human_5dim.csv  (the mapped human scores)

Method:
  - Map each 10-dim categorical audit field into the 5-dim 1–5 scale via
    LEVEL_MAP / LEVEL_MAP_INVERTED.
  - For each 5-dim judge dimension, average the contributing human dims
    (clip 1–5, round to nearest integer for κ; keep continuous for ρ/ICC).
  - For each pairwise comparison (Human vs Sonnet, Human vs GPT-4o,
    Sonnet vs GPT-4o), compute:
      Cohen κ (linear), Cohen κ (quadratic), ICC(2,1), Spearman ρ,
      exact match rate, MAE.
  - Pool over all 45 (variant, case) cells per dim.

Design note:
  This is a coarse mapping; we publish the mapping table in the supplementary
  and report κ as moderate-to-substantial where it's expected to be (factuality,
  groundedness) and weaker on safety / abstain (rubric ceilings).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import cohen_kappa_score


# --------- Crosswalk: human 10-dim categorical → 5-dim 1–5 ---------

LEVEL_MAP: dict[str, int] = {
    "yes": 5, "partial": 3, "no": 1,
    "none": 5, "mild": 3, "severe": 1,
    "high": 5, "medium": 3, "low": 1,
    "maybe": 3,
}

# needs_domain_expert_review is inverted: needing review = low safety score
LEVEL_MAP_INVERTED: dict[str, int] = {
    "yes": 1, "partial": 3, "maybe": 3, "no": 5,
}

# 5-dim judge field → list of (human-rater field, mapping dict)
DIM_MAPPING: dict[str, list[tuple[str, dict[str, int]]]] = {
    "factuality": [
        ("topic_relevance", LEVEL_MAP),
        ("eligibility_overstatement", LEVEL_MAP),
    ],
    "groundedness": [
        ("unsupported_explanation", LEVEL_MAP),
        ("fallback_used_correctly", LEVEL_MAP),
    ],
    "abstain_appropriateness": [
        ("missing_facts_reasonable", LEVEL_MAP),
        ("unresolved_requirements_reasonable", LEVEL_MAP),
    ],
    "safety": [
        ("needs_domain_expert_review", LEVEL_MAP_INVERTED),
        ("overall_usable", LEVEL_MAP),
    ],
    "patient_utility": [
        ("patient_facing_clarity", LEVEL_MAP),
        ("teachback_targeted", LEVEL_MAP),
        ("overall_usable", LEVEL_MAP),
    ],
}

JUDGE_DIMS = list(DIM_MAPPING.keys())  # factuality, groundedness, abstain_appropriateness, safety, patient_utility


# Variant → V1/V2/V-final
HUMAN_VARIANT_FILES: dict[str, str] = {
    "V1": "data/processed/onboarding_eval_audit_template_v2.csv",
    "V2": "data/processed/onboarding_eval_audit_v2.csv",
    "V-final": "data/processed/onboarding_eval_audit_final.csv",
}


def map_human_to_5dim(
    df: pd.DataFrame, variant: str, agg: str = "mean"
) -> pd.DataFrame:
    """Convert a 10-dim human-rater CSV row into 5 continuous scores in [1,5].

    Args:
        agg: 'mean' | 'min' | 'max' over contributing human dims per judge dim.
    """
    if agg not in {"mean", "min", "max"}:
        raise ValueError(f"agg must be mean|min|max, got {agg}")
    fn = {"mean": np.mean, "min": np.min, "max": np.max}[agg]
    out_rows = []
    for _, r in df.iterrows():
        row = {"case_id": r["case_id"], "variant": variant}
        for jdim, contributors in DIM_MAPPING.items():
            mapped: list[float] = []
            for col, mp in contributors:
                v = str(r.get(col, "")).strip().lower()
                if not v:
                    continue
                if v in mp:
                    mapped.append(float(mp[v]))
            if not mapped:
                row[jdim] = np.nan
            else:
                row[jdim] = float(fn(mapped))
        out_rows.append(row)
    return pd.DataFrame(out_rows)


def load_human_5dim(repo_root: Path, agg: str = "mean") -> pd.DataFrame:
    """Load 3 audit CSVs, map to 5-dim, concat (variant, case_id, 5 dims)."""
    parts = []
    for variant, csv_relpath in HUMAN_VARIANT_FILES.items():
        csv_path = repo_root / csv_relpath
        if not csv_path.exists():
            raise FileNotFoundError(f"missing human-rater CSV: {csv_path}")
        df = pd.read_csv(csv_path)
        parts.append(map_human_to_5dim(df, variant, agg=agg))
    return pd.concat(parts, ignore_index=True)


def load_judge(jsonl_path: Path, judge_label: str) -> pd.DataFrame:
    """Load judge JSONL into long-form (case_id, variant, judge, <5 dims>)."""
    if not jsonl_path.exists():
        raise FileNotFoundError(jsonl_path)
    rows = []
    with jsonl_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if "error" in d:
                continue
            scores = d.get("scores") or {}
            row = {
                "case_id": d.get("case_id"),
                "variant": d.get("backbone"),  # repurposed
                "judge": judge_label,
            }
            for jdim in JUDGE_DIMS:
                v = scores.get(jdim)
                row[jdim] = float(v) if v is not None else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def _round_int(x: float) -> int:
    if pd.isna(x):
        return -1
    return int(np.clip(round(float(x)), 1, 5))


def pairwise_agreement(a: pd.Series, b: pd.Series) -> dict:
    """Compute κ_lin, κ_quad, ICC(2,1), Spearman ρ, exact match, MAE on integer-rounded scores.

    a, b are continuous in [1,5]; they're rounded to integer for κ but kept continuous
    for ρ and ICC(2,1) where appropriate.
    """
    paired = pd.concat([a, b], axis=1).dropna()
    if len(paired) == 0:
        return {k: np.nan for k in ["kappa_linear", "kappa_quadratic", "icc21", "spearman", "exact", "mae", "n"]}
    a_r = paired.iloc[:, 0].apply(_round_int).astype(int)
    b_r = paired.iloc[:, 1].apply(_round_int).astype(int)

    try:
        k_lin = cohen_kappa_score(a_r, b_r, weights="linear")
    except Exception:
        k_lin = np.nan
    try:
        k_quad = cohen_kappa_score(a_r, b_r, weights="quadratic")
    except Exception:
        k_quad = np.nan

    # ICC(2,1) two-way random absolute-agreement, single-rater.
    # Compact form via ANOVA. For two raters this collapses; use simple pooled var.
    try:
        x = paired.iloc[:, 0].astype(float).values
        y = paired.iloc[:, 1].astype(float).values
        n = len(x)
        mean_per_subject = (x + y) / 2.0
        grand_mean = (x.mean() + y.mean()) / 2.0
        ssb = 2 * np.sum((mean_per_subject - grand_mean) ** 2)
        msb = ssb / (n - 1) if n > 1 else np.nan
        ssr_within = np.sum((x - mean_per_subject) ** 2 + (y - mean_per_subject) ** 2)
        msw = ssr_within / n if n > 0 else np.nan
        icc21 = (msb - msw) / (msb + msw) if (msb + msw) > 0 else np.nan
    except Exception:
        icc21 = np.nan

    try:
        rho, _ = stats.spearmanr(paired.iloc[:, 0], paired.iloc[:, 1])
    except Exception:
        rho = np.nan

    exact = float((a_r == b_r).mean())
    mae = float(np.mean(np.abs(paired.iloc[:, 0].astype(float) - paired.iloc[:, 1].astype(float))))
    return {
        "kappa_linear": float(k_lin) if k_lin is not None else np.nan,
        "kappa_quadratic": float(k_quad) if k_quad is not None else np.nan,
        "icc21": float(icc21) if icc21 is not None else np.nan,
        "spearman": float(rho) if rho is not None else np.nan,
        "exact": exact,
        "mae": mae,
        "n": int(len(paired)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=str, default=".")
    ap.add_argument(
        "--judge-sonnet",
        type=str,
        default="outputs/phase4/15case_audit/judge_sonnet.jsonl",
    )
    ap.add_argument(
        "--judge-gpt4o",
        type=str,
        default="outputs/phase4/15case_audit/judge_gpt4o.jsonl",
    )
    ap.add_argument(
        "--out-csv",
        type=str,
        default="outputs/phase4/15case_audit/3way_agreement_per_dim.csv",
    )
    ap.add_argument(
        "--out-json",
        type=str,
        default="outputs/phase4/15case_audit/3way_agreement_summary.json",
    )
    ap.add_argument(
        "--out-human-mapped",
        type=str,
        default="outputs/phase4/15case_audit/3way_audit_human_5dim.csv",
    )
    ap.add_argument(
        "--agg",
        type=str,
        default="mean",
        choices=["mean", "min", "max"],
        help="how to aggregate multi-source human dims into a single judge dim",
    )
    ap.add_argument(
        "--per-variant",
        action="store_true",
        help="if set, also output per-variant agreement (V1, V2, V-final separately)",
    )
    args = ap.parse_args()

    repo_root = Path(args.repo_root).resolve()

    # 1. Load + map human ratings to 5-dim (with selected aggregation)
    human = load_human_5dim(repo_root, agg=args.agg)
    human["rater"] = "human"
    print(
        f"[human] {len(human)} rows across variants {sorted(human['variant'].unique())} "
        f"(agg={args.agg})"
    )

    # 2. Load judges
    sonnet = load_judge(repo_root / args.judge_sonnet, "sonnet")
    gpt4o = load_judge(repo_root / args.judge_gpt4o, "gpt4o")
    print(f"[sonnet] {len(sonnet)} rows; [gpt4o] {len(gpt4o)} rows")

    # 3. Persist human-mapped scores for paper supplementary
    out_human_path = repo_root / args.out_human_mapped
    out_human_path.parent.mkdir(parents=True, exist_ok=True)
    human[["variant", "case_id"] + JUDGE_DIMS].to_csv(out_human_path, index=False)
    print(f"[wrote] {out_human_path}")

    # 4. Pairwise agreement per dim, per pair
    pairs: list[tuple[str, pd.DataFrame, pd.DataFrame]] = [
        ("human_vs_sonnet", human, sonnet),
        ("human_vs_gpt4o", human, gpt4o),
        ("sonnet_vs_gpt4o", sonnet, gpt4o),
    ]

    rows = []
    for pair_label, a_df, b_df in pairs:
        merged = a_df.merge(
            b_df,
            on=["case_id", "variant"],
            how="inner",
            suffixes=("_a", "_b"),
        )
        for dim in JUDGE_DIMS:
            ag = pairwise_agreement(merged[f"{dim}_a"], merged[f"{dim}_b"])
            ag["pair"] = pair_label
            ag["dim"] = dim
            rows.append(ag)
    df_out = pd.DataFrame(rows)[
        [
            "pair",
            "dim",
            "n",
            "kappa_linear",
            "kappa_quadratic",
            "icc21",
            "spearman",
            "exact",
            "mae",
        ]
    ]
    out_csv_path = repo_root / args.out_csv
    df_out.to_csv(out_csv_path, index=False)
    print(f"[wrote] {out_csv_path}")
    print(df_out.to_string(index=False))

    # 5. Summary JSON
    summary: dict = {
        "n_per_pair_per_dim": {
            row["pair"] + "::" + row["dim"]: int(row["n"]) for row in rows
        },
        "macro_means_per_pair": {},
        "per_dim_per_pair": {},
    }
    for pair_label, _, _ in pairs:
        sub = df_out[df_out["pair"] == pair_label]
        summary["macro_means_per_pair"][pair_label] = {
            "kappa_linear": float(sub["kappa_linear"].mean()),
            "kappa_quadratic": float(sub["kappa_quadratic"].mean()),
            "icc21": float(sub["icc21"].mean()),
            "spearman": float(sub["spearman"].mean()),
            "exact": float(sub["exact"].mean()),
            "mae": float(sub["mae"].mean()),
        }
        summary["per_dim_per_pair"][pair_label] = {
            r["dim"]: {
                k: (float(r[k]) if not pd.isna(r[k]) else None)
                for k in ["kappa_linear", "kappa_quadratic", "icc21", "spearman", "exact", "mae", "n"]
            }
            for r in sub.to_dict(orient="records")
        }

    out_json_path = repo_root / args.out_json
    with out_json_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"[wrote] {out_json_path}")

    # 6. Per-variant agreement (if requested)
    if args.per_variant:
        print("\n=== per-variant agreement ===")
        per_variant_rows = []
        for variant in ["V1", "V2", "V-final"]:
            for pair_label, a_df, b_df in pairs:
                a_v = a_df[a_df["variant"] == variant] if "variant" in a_df.columns else a_df
                b_v = b_df[b_df["variant"] == variant] if "variant" in b_df.columns else b_df
                merged = a_v.merge(
                    b_v, on=["case_id", "variant"], how="inner", suffixes=("_a", "_b")
                )
                for dim in JUDGE_DIMS:
                    ag = pairwise_agreement(merged[f"{dim}_a"], merged[f"{dim}_b"])
                    ag["pair"] = pair_label
                    ag["dim"] = dim
                    ag["variant"] = variant
                    per_variant_rows.append(ag)
        df_pv = pd.DataFrame(per_variant_rows)[
            [
                "variant",
                "pair",
                "dim",
                "n",
                "kappa_linear",
                "kappa_quadratic",
                "icc21",
                "spearman",
                "exact",
                "mae",
            ]
        ]
        out_pv_path = (
            repo_root / args.out_csv.replace(".csv", "_per_variant.csv")
        )
        df_pv.to_csv(out_pv_path, index=False)
        print(f"[wrote] {out_pv_path}")
        # print compact
        for variant in ["V1", "V2", "V-final"]:
            sub = df_pv[df_pv["variant"] == variant]
            print(f"\n--- {variant} ---")
            print(
                sub.pivot_table(
                    index="dim",
                    columns="pair",
                    values="spearman",
                    aggfunc="first",
                ).round(3)
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
