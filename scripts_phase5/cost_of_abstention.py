#!/usr/bin/env python3
"""
Phase 5 Task 7 — cost-of-abstention analysis.

Quantifies the utility actually forgone by the selector's abstentions: among the
abstain-regime cases, how often was a *relevant* trial in fact present in the
case's candidate pool (so a non-abstaining system could in principle have
helped). Uses only persisted artifacts — no new inference.

Relevance signal, in priority order per case:
  1. gold_nct present in the case's evidence pool (TREC/curated cases with gold)
  2. selected_in_qrels_any (selector's pick was a qrels-relevant trial)

Outputs outputs/phase5/cost_of_abstention.csv + a printed summary with
Clopper-Pearson 95% CIs.

Run (after evidence pools are synced local):
  python scripts_phase5/cost_of_abstention.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
POOL_DIR = ROOT / "outputs" / "phase5" / "evidence_pools"
OUT = ROOT / "outputs" / "phase5"


def clopper_pearson(k: int, n: int, alpha: float = 0.05):
    from scipy.stats import beta
    if n == 0:
        return (float("nan"), float("nan"))
    lo = 0.0 if k == 0 else beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - alpha / 2, k + 1, n - k)
    return round(float(lo), 4), round(float(hi), 4)


def pool_doc_ids(case_id: str) -> set[str]:
    fp = POOL_DIR / f"{case_id}.csv"
    if not fp.exists():
        return set()
    return set(pd.read_csv(fp)["doc_id"].astype(str))


def main() -> int:
    raw = pd.read_csv(ROOT / "outputs/phase4/n114_aggregate/raw_n114.csv")
    cache = pd.read_csv(ROOT / "outputs/tables/selector_signals_cache.csv")
    qrels = cache.set_index("case_id")["selected_in_qrels_any"].to_dict() \
        if "selected_in_qrels_any" in cache else {}

    # one row per case (regime is per-case; backbone-invariant gate)
    cases = raw.drop_duplicates("case_id")[["case_id", "regime_gate", "gold_nct"]].copy()

    rows = []
    for _, r in cases.iterrows():
        cid = str(r["case_id"])
        gold = str(r["gold_nct"]) if pd.notna(r["gold_nct"]) and str(r["gold_nct"]).startswith("NCT") else ""
        pool = pool_doc_ids(cid)
        gold_in_pool = bool(gold) and gold in pool
        relevant_available = gold_in_pool or bool(qrels.get(cid, False))
        rows.append({
            "case_id": cid,
            "regime_gate": r["regime_gate"],
            "gold_nct": gold,
            "pool_size": len(pool),
            "gold_in_pool": gold_in_pool,
            "selected_in_qrels_any": bool(qrels.get(cid, False)),
            "relevant_available": relevant_available,
        })
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "cost_of_abstention.csv", index=False)

    abst = df[df["regime_gate"] == "abstain"]
    acc = df[df["regime_gate"] == "accept"]
    n_ab = len(abst)
    k_missed = int(abst["relevant_available"].sum())
    lo, hi = clopper_pearson(k_missed, n_ab)
    print(f"abstain cases: {n_ab}; accept cases: {len(acc)}")
    print(f"abstain cases with a relevant trial available in pool "
          f"(missed-utility): {k_missed}/{n_ab} = {k_missed/n_ab:.3f}  CI95 {lo,hi}")
    print(f"  of which gold_nct-in-pool: {int(abst['gold_in_pool'].sum())}")
    print(f"  abstain cases with NO relevant trial available "
          f"(correct abstention): {n_ab - k_missed}/{n_ab} = {(n_ab-k_missed)/n_ab:.3f}")
    print(f"pools found for {df['pool_size'].gt(0).sum()}/{len(df)} cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
