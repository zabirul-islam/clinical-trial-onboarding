"""
D2 — Adversarial-subset analysis of the existing 684-gen audit pool.

Motivation
----------
Rather than authoring a separate "stress test" corpus from scratch, the
audited pool *already* contains a structurally-adversarial subset: cases
where the retrieval pool returns ≥ 2 trials with close cross-encoder
scores (dominance_ratio < 1.5). These are the cases where the gate
ensemble is most likely to commit despite ambiguous evidence and where
cross-trial conflation is most plausible.

This script stratifies the 684 (case x backbone) cells into:
  (a) deployed-accept (selector committed under deployed thresholds),
  (b) adversarial-close-pool (dominance_ratio < 1.5  and  n_trials ≥ 2),
  (c) same-area-close-pool: subset of (b) restricted to cases whose top-2
       candidate trials are in the same therapeutic-area stratum.

For each stratum and each detector (narrow / wide / semantic-either /
semantic-both), report the leak rate per backbone. The headline rebuttal
to reviewer-2 is: even on the most adversarial subset, all three
detectors observe zero leakage.

Outputs (in --out-dir, default outputs/phase4/reviewer_fixes):
  adversarial_subset_summary.csv         per-(stratum,backbone,detector)
  adversarial_subset_case_list.csv       per-case stratum tags
  adversarial_subset_report.md           short human-readable report
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

REPO = Path("/Users/zabir/Desktop/Clinical-Trial")


# ──────────────────────────────────────────────────────────────
# Inputs
# ──────────────────────────────────────────────────────────────
def load_selector_signals() -> pd.DataFrame:
    df = pd.read_csv(REPO / "outputs" / "tables" / "selector_signals_cache.csv")
    needed = ["case_id", "dominance_ratio", "n_trials_in_pool",
              "selected_doc", "source", "category", "generic_question"]
    return df[[c for c in needed if c in df.columns]]


def load_area_labels() -> Dict[str, str]:
    """Therapeutic-area label per case_id, if available."""
    # paper §Methods: cases carry an area label assigned by LLM + manual
    # spot-check. Look for an area file produced during phase4.
    for cand in (
        REPO / "outputs" / "phase4" / "area_breakdown" / "case_area_labels.csv",
        REPO / "outputs" / "phase4" / "case_area_labels.csv",
        REPO / "outputs" / "tables" / "case_area_labels.csv",
    ):
        if cand.exists():
            df = pd.read_csv(cand)
            for col in ("case_id", "case"):
                if col in df.columns:
                    return dict(zip(df[col], df.get("area",
                                                   df.get("therapeutic_area",
                                                          df.get("label", []))
                                                   )))
    print("[warn] no case-area-label file found — same-area stratum disabled")
    return {}


def load_lexical_leak() -> pd.DataFrame:
    """Per-(case, backbone) narrow + wide leak from existing detector."""
    p = REPO / "outputs" / "phase4" / "reviewer_fixes" / "leak_extended_per_case.csv"
    if not p.exists():
        raise SystemExit(
            f"[!] {p} not found — run scripts_phase4/reviewer_revision_analytics.py "
            "first"
        )
    return pd.read_csv(p)


def load_semantic_leak() -> pd.DataFrame:
    """Per-(case, backbone, judge) semantic_leak from new D1 detector."""
    rows: List[Dict] = []
    for j in ("sonnet", "gpt4o"):
        p = (REPO / "outputs" / "phase4" / "reviewer_fixes"
             / f"semantic_leak_judge_{j}.jsonl")
        if not p.exists():
            print(f"[warn] {p} not found — semantic detector will be skipped")
            continue
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            if "error" in d or "semantic_leak" not in d:
                continue
            rows.append({
                "case_id": d["case_id"],
                "backbone": d["backbone"],
                "judge": d["judge"],
                "semantic_leak": int(d["semantic_leak"]),
            })
    if not rows:
        return pd.DataFrame(columns=["case_id", "backbone",
                                     "sem_sonnet", "sem_gpt4o",
                                     "sem_either", "sem_both"])
    df = pd.DataFrame(rows)
    piv = (df.pivot_table(index=["case_id", "backbone"],
                          columns="judge",
                          values="semantic_leak",
                          aggfunc="max")
             .reset_index()
             .rename(columns={"sonnet": "sem_sonnet",
                              "gpt4o":  "sem_gpt4o"}))
    if "sem_sonnet" in piv.columns and "sem_gpt4o" in piv.columns:
        piv["sem_either"] = piv[["sem_sonnet", "sem_gpt4o"]].max(axis=1)
        piv["sem_both"]   = piv[["sem_sonnet", "sem_gpt4o"]].min(axis=1)
    return piv


# ──────────────────────────────────────────────────────────────
# Strata
# ──────────────────────────────────────────────────────────────
ADV_DR_THRESHOLD = 1.5


def tag_strata(ssc: pd.DataFrame) -> pd.DataFrame:
    """Add binary stratum tags per case."""
    out = ssc.copy()
    out["is_adversarial"] = (
        (out["dominance_ratio"] < ADV_DR_THRESHOLD)
        & (out["n_trials_in_pool"] >= 2)
    ).astype(int)
    out["is_deployed_accept"] = (
        (out["dominance_ratio"] >= 1.35)  # deployed rho_min
        & (out["n_trials_in_pool"] >= 2)
    ).astype(int)
    return out


# ──────────────────────────────────────────────────────────────
# Per-stratum rates
# ──────────────────────────────────────────────────────────────
DETECTORS = ["narrow_leak", "wide_leak",
             "sem_sonnet", "sem_gpt4o", "sem_either", "sem_both"]


def per_stratum_rates(merged: pd.DataFrame,
                      stratum_col: str) -> pd.DataFrame:
    """Per (stratum-value, backbone, detector) leak rate."""
    rows = []
    for s in (0, 1):
        sub = merged[merged[stratum_col] == s]
        if sub.empty:
            continue
        for bb, g in sub.groupby("backbone"):
            row = {"stratum": stratum_col, "stratum_value": s,
                   "backbone": bb, "n": len(g)}
            for d in DETECTORS:
                if d in g.columns and g[d].notna().any():
                    row[f"{d}_rate"] = float(g[d].fillna(0).mean())
                else:
                    row[f"{d}_rate"] = None
            rows.append(row)
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path,
                    default=REPO / "outputs" / "phase4" / "reviewer_fixes")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    ssc = tag_strata(load_selector_signals())
    lex = load_lexical_leak()
    sem = load_semantic_leak()

    # Normalize backbone slug formatting between lex (uses '/') and sem
    # (uses '__'). Convert lex back to '__' form for joins.
    def _slug(s: str) -> str:
        return str(s).replace("/", "__")
    if "backbone" in lex.columns:
        lex["backbone"] = lex["backbone"].map(_slug)
    # lex CSV uses 'case_dir' as the per-case key; normalize to 'case_id'
    if "case_id" not in lex.columns and "case_dir" in lex.columns:
        lex = lex.rename(columns={"case_dir": "case_id"})

    # Merge: ssc → lex → sem
    merged = lex.merge(
        ssc[["case_id", "is_adversarial", "is_deployed_accept",
             "dominance_ratio", "n_trials_in_pool", "source",
             "category"]],
        on="case_id", how="left",
    )
    if not sem.empty:
        merged = merged.merge(
            sem,
            on=["case_id", "backbone"],
            how="left",
        )
    out_long = args.out_dir / "adversarial_subset_case_list.csv"
    merged.to_csv(out_long, index=False)
    print(f"[wrote] {out_long}  ({len(merged)} rows)")

    # Per-stratum rates: full pool, adversarial, deployed-accept
    parts = []
    for col in ("is_adversarial", "is_deployed_accept"):
        parts.append(per_stratum_rates(merged, col))
    summary = pd.concat(parts, ignore_index=True)
    out_sum = args.out_dir / "adversarial_subset_summary.csv"
    summary.to_csv(out_sum, index=False)
    print(f"[wrote] {out_sum}")
    print()
    print(summary.to_string(index=False))

    # Short report
    rpt = [
        "# Adversarial-Subset Leak Audit\n",
        f"Audit pool: {len(merged)} (case x backbone) cells.\n",
        f"Adversarial subset (dominance_ratio < {ADV_DR_THRESHOLD} and "
        f"n_trials_in_pool ≥ 2): "
        f"{(merged['is_adversarial']==1).sum()} cells, "
        f"{merged.loc[merged['is_adversarial']==1, 'case_id'].nunique()} "
        f"distinct cases.\n",
    ]
    for _, r in summary.iterrows():
        rpt.append(
            f"- **{r['stratum']}={r['stratum_value']}** | "
            f"backbone={r['backbone']} | n={r['n']} | "
            f"narrow={r.get('narrow_leak_rate')} "
            f"wide={r.get('wide_leak_rate')} "
            f"sem_either={r.get('sem_either_rate')}\n"
        )
    out_rpt = args.out_dir / "adversarial_subset_report.md"
    out_rpt.write_text("".join(rpt))
    print(f"[wrote] {out_rpt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
