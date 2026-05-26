"""
T1.5 — Per-gate ablation via counterfactual reuse of the n=114 --no-gate run.

Insight: the n=114 expansion was run with --no-gate, so every (case × backbone)
generation exists regardless of which gates the case fails. We can therefore
counterfactually compute leave-one-out gate ablation by filtering the existing
n=114 table to the subset accepted under each variant.

Variants:
    V-final          (all gates active; deployment baseline)
    V-final-no-rho   (drop dominance gate)
    V-final-no-tau   (drop trial-score-share gate)
    V-final-no-mu    (drop raw-cross-encoder gate)
    V-final-no-kappa (drop best-rank gate)
    V-final-no-generic (drop generic-question heuristic)

For each variant:
    - n_accepted             (cases that would clear that variant's gates)
    - n_only_this_gate       (cases that ONLY this gate had been rejecting)
    - per-backbone leak / commit / abstain rates on accepted cases
    - delta vs V-final on each metric

Outputs (under outputs/phase4/per_gate_ablation/):
    per_gate_summary.csv
    per_gate_delta_vs_vfinal.csv
    per_gate_only_this_gate.csv         (cases newly accepted by each variant)
    per_gate_summary.json

CPU only, ~5 sec. Inputs are all already on disk:
    outputs/phase4/n114_aggregate/raw_n114.csv
    outputs/tables/selector_signals_cache.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# Production thresholds (paper §3.3 / Algorithm 1)
GATE = {
    "rho_min": 1.35,
    "tau_min": 0.28,
    "mu_min": -4.5,
    "kappa_max": 2,
}
SIGNAL_COLS = ["dominance_ratio", "trial_score_share", "raw_max_cross", "best_rank", "generic_question"]


def gate_pass_per_signal(cache: pd.DataFrame) -> dict[str, pd.Series]:
    """Per-signal pass mask. True = case clears that single gate."""
    return {
        "rho": cache["dominance_ratio"] >= GATE["rho_min"],
        "tau": cache["trial_score_share"] >= GATE["tau_min"],
        "mu": cache["raw_max_cross"] >= GATE["mu_min"],
        "kappa": cache["best_rank"] <= GATE["kappa_max"],
        "generic": ~cache["generic_question"].fillna(False).astype(bool),
    }


def variant_acceptance(cache: pd.DataFrame, drop_gate: str | None) -> pd.Series:
    """Return per-case boolean: would this case be accepted under the variant?

    drop_gate=None → V-final (all gates).
    drop_gate='rho' → all gates except ρ; any case clearing other 4 gates accepted.
    """
    masks = gate_pass_per_signal(cache)
    if drop_gate is not None and drop_gate not in masks:
        raise ValueError(f"unknown gate: {drop_gate}")
    keep = [k for k in masks if k != drop_gate]
    out = pd.Series(True, index=cache.index)
    for k in keep:
        out &= masks[k]
    return out


def per_backbone_metrics(df: pd.DataFrame, group_col: str = "backbone") -> pd.DataFrame:
    """Compute leak / commit / abstain / parse_ok per group."""
    def leak(s):
        return float((s > 0).mean())
    def commit(s):
        return float(s.isin(["likely_match", "possible_match_insufficient_evidence"]).mean())
    def abstain(s):
        return float((s == "cannot_determine").mean())
    return df.groupby(group_col).agg(
        n=("case_id", "nunique"),
        rows=("case_id", "size"),
        parse_ok=("parse_ok", "mean"),
        commit_rate=("decision", commit),
        abstain_rate=("decision", abstain),
        leak_rate=("cross_trial_leak_n", leak),
    ).reset_index()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=str, default=".")
    ap.add_argument(
        "--raw",
        type=str,
        default="outputs/phase4/n114_aggregate/raw_n114.csv",
    )
    ap.add_argument(
        "--cache",
        type=str,
        default="outputs/tables/selector_signals_cache.csv",
    )
    ap.add_argument(
        "--out-dir",
        type=str,
        default="outputs/phase4/per_gate_ablation",
    )
    args = ap.parse_args()

    root = Path(args.repo_root).resolve()
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(root / args.raw)
    cache = pd.read_csv(root / args.cache)
    print(f"[load] raw={len(raw)} rows, cache={len(cache)} rows")

    # Restrict cache to cases present in raw (n=114 pool)
    n114_ids = set(raw["case_id"].astype(str))
    cache = cache[cache["case_id"].astype(str).isin(n114_ids)].reset_index(drop=True)
    print(f"[filter] cache restricted to n=114 pool: {len(cache)} rows")

    masks_per_signal = gate_pass_per_signal(cache)

    # Compute V-final acceptance + each variant
    variants: list[tuple[str, str | None]] = [
        ("V-final", None),
        ("V-final-no-rho", "rho"),
        ("V-final-no-tau", "tau"),
        ("V-final-no-mu", "mu"),
        ("V-final-no-kappa", "kappa"),
        ("V-final-no-generic", "generic"),
    ]
    case_to_signals = cache.set_index("case_id")

    per_variant_metrics: list[dict] = []
    per_variant_only: list[dict] = []

    # V-final's accepted set, for delta computation
    vfinal_accept_ids = set(
        cache.loc[variant_acceptance(cache, None), "case_id"].astype(str)
    )

    for vname, drop in variants:
        accept_mask = variant_acceptance(cache, drop)
        accept_ids = set(cache.loc[accept_mask, "case_id"].astype(str))

        # Cases newly accepted by this variant (vs V-final)
        only_this = accept_ids - vfinal_accept_ids
        per_variant_only.append({
            "variant": vname,
            "drop_gate": drop,
            "n_accepted": len(accept_ids),
            "n_only_this_gate": len(only_this),
            "newly_accepted_case_ids": sorted(only_this),
        })

        # Filter raw to accepted cases
        sub = raw[raw["case_id"].astype(str).isin(accept_ids)].copy()
        m = per_backbone_metrics(sub)
        m["variant"] = vname
        m["n_accepted_cases"] = len(accept_ids)
        per_variant_metrics.append(m)

    df_metrics = pd.concat(per_variant_metrics, ignore_index=True)
    df_metrics = df_metrics[
        [
            "variant",
            "n_accepted_cases",
            "backbone",
            "n",
            "rows",
            "parse_ok",
            "commit_rate",
            "abstain_rate",
            "leak_rate",
        ]
    ]
    df_metrics.to_csv(out_dir / "per_gate_summary.csv", index=False)
    print("\n=== per_gate_summary ===")
    print(df_metrics.to_string(index=False))

    # Delta vs V-final: how much does dropping each gate change leak / commit / abstain?
    base = df_metrics[df_metrics["variant"] == "V-final"][
        ["backbone", "leak_rate", "commit_rate", "abstain_rate", "parse_ok", "n_accepted_cases"]
    ].rename(
        columns={
            "leak_rate": "leak_base",
            "commit_rate": "commit_base",
            "abstain_rate": "abstain_base",
            "parse_ok": "parse_ok_base",
            "n_accepted_cases": "n_base",
        }
    )
    deltas = df_metrics[df_metrics["variant"] != "V-final"].merge(base, on="backbone")
    deltas["delta_leak"] = deltas["leak_rate"] - deltas["leak_base"]
    deltas["delta_commit"] = deltas["commit_rate"] - deltas["commit_base"]
    deltas["delta_abstain"] = deltas["abstain_rate"] - deltas["abstain_base"]
    deltas["delta_n"] = deltas["n_accepted_cases"] - deltas["n_base"]
    deltas = deltas[
        [
            "variant",
            "backbone",
            "n_accepted_cases",
            "n_base",
            "delta_n",
            "leak_rate",
            "leak_base",
            "delta_leak",
            "commit_rate",
            "commit_base",
            "delta_commit",
            "abstain_rate",
            "abstain_base",
            "delta_abstain",
        ]
    ]
    deltas.to_csv(out_dir / "per_gate_delta_vs_vfinal.csv", index=False)
    print("\n=== per_gate_delta_vs_vfinal ===")
    print(
        deltas[
            ["variant", "backbone", "delta_n", "delta_leak", "delta_commit", "delta_abstain"]
        ].to_string(index=False)
    )

    # Per-variant only-this-gate cases
    df_only = pd.DataFrame(per_variant_only)
    df_only.to_csv(out_dir / "per_gate_only_this_gate.csv", index=False)
    print("\n=== per_gate_only_this_gate ===")
    print(df_only[["variant", "drop_gate", "n_accepted", "n_only_this_gate"]].to_string(index=False))

    # Headline JSON
    headline = {
        "n_pool": int(len(cache)),
        "vfinal_accept": int(len(vfinal_accept_ids)),
        "per_variant_n_accepted": {
            r["variant"]: int(r["n_accepted"]) for r in per_variant_only
        },
        "per_variant_n_only_this_gate": {
            r["variant"]: int(r["n_only_this_gate"]) for r in per_variant_only
        },
        "per_variant_max_leak_across_backbones": {},
    }
    for vname in [v[0] for v in variants]:
        sub = df_metrics[df_metrics["variant"] == vname]
        headline["per_variant_max_leak_across_backbones"][vname] = float(
            sub["leak_rate"].max()
        )
    with (out_dir / "per_gate_summary.json").open("w") as f:
        json.dump(headline, f, indent=2)
    print("\n=== HEADLINE per-gate ablation ===")
    print(json.dumps(headline, indent=2))

    # Verdict text for paper
    print("\n=== VERDICT ===")
    any_leak = any(v > 0 for v in headline["per_variant_max_leak_across_backbones"].values())
    if not any_leak:
        print("CLEAN: no variant produces any leak on any backbone (max leak = 0 everywhere).")
        print("→ Paper claim: structural safety holds even when individual gates are removed,")
        print("  i.e., no single gate is solely responsible for the zero-leak result.")
    else:
        print("MIXED: at least one variant + backbone shows leak > 0. Investigate which gate.")
        for v, m in headline["per_variant_max_leak_across_backbones"].items():
            if m > 0:
                print(f"  {v}: max leak = {m:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
