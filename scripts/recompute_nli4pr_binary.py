"""
Phase 1.4b — Recompute NLI4PR alignment metrics under correct BINARY framing.

NLI4PR has only two labels in this corpus: entailment (3939) / contradiction (3068).
No neutral. Original 3-class eval inflated false "undetermined" row.

We report four scorings:

  A) Strict-match   : likely_match                    -> entail
                       everything else                -> contradict
  B) Lenient-match  : likely_match OR possible_match  -> entail
                       unlikely OR cannot_determine   -> contradict
  C) Abstention-aware (preferred for the paper):
       likely_match                    -> entail
       unlikely_match                  -> contradict
       possible_match_insufficient OR
       cannot_determine OR parse_fail  -> ABSTAIN (excluded from acc/F1)
       Report coverage + conditional metrics.
  D) Full-coverage: same as (C) but forced pick for abstentions (random label),
       included for completeness.

Outputs: outputs/tables/nli4pr_eligibility_alignment_binary.json
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd


BINARY = ["entail", "contradict"]


def conf(yt, yp, labels=BINARY):
    idx = {l: i for i, l in enumerate(labels)}
    M = np.zeros((2, 2), dtype=int)
    for t, p in zip(yt, yp):
        if t in idx and p in idx:
            M[idx[t], idx[p]] += 1
    return M


def kappa(yt, yp, labels=BINARY):
    M = conf(yt, yp, labels)
    N = M.sum()
    if N == 0: return float("nan")
    po = np.trace(M) / N
    pe = ((M.sum(0) * M.sum(1)) / (N * N)).sum()
    return float("nan") if pe == 1 else float((po - pe) / (1 - pe))


def macro_f1(yt, yp, labels=BINARY):
    M = conf(yt, yp, labels)
    f1s = []
    for i in range(len(labels)):
        tp = M[i, i]; fp = M[:, i].sum() - tp; fn = M[i, :].sum() - tp
        pr = tp / (tp + fp) if (tp + fp) else 0.0
        rc = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * pr * rc / (pr + rc) if (pr + rc) else 0.0)
    return float(np.mean(f1s)), {l: f for l, f in zip(labels, f1s)}


def score(yt, yp, name):
    M = conf(yt, yp)
    N = M.sum()
    acc = float(np.trace(M) / N) if N else float("nan")
    f1m, f1p = macro_f1(yt, yp)
    k = kappa(yt, yp)
    majority = max((yt == "entail").mean(), (yt == "contradict").mean()) if len(yt) else float("nan")
    return {
        "name": name, "n": int(N),
        "accuracy": acc, "majority_baseline_acc": float(majority),
        "macro_f1": f1m, "f1_per_class": f1p,
        "cohens_kappa": k, "confusion": M.tolist(), "labels": BINARY,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", type=Path, default=Path("."))
    p.add_argument("--csv", type=str,
                   default="outputs/tables/nli4pr_eligibility_alignment.csv")
    args = p.parse_args()

    root = args.repo_root.resolve()
    df = pd.read_csv(root / args.csv)
    print(f"loaded {len(df):,} rows from {args.csv}")

    # Gold: collapse 3-class -> binary (drop rows where gold_class == undetermined, i.e. 'neutral')
    gold_map = {"match": "entail", "mismatch": "contradict", "undetermined": None}
    df["gold_bin"] = df["gold_class"].map(gold_map)
    df = df[df["gold_bin"].notna()].reset_index(drop=True)
    print(f"after drop gold=None: {len(df):,}")

    # Prediction bucketing uses raw pred_decision
    dec = df["pred_decision"].astype(str).str.strip().str.lower().str.replace(" ", "_")
    parse_ok = df["parse_ok"].astype(bool) if "parse_ok" in df.columns else pd.Series([True]*len(df))

    # === A) Strict
    pred_a = np.where(dec == "likely_match", "entail", "contradict")
    sA = score(df["gold_bin"].values, pred_a, "A_strict")

    # === B) Lenient
    pred_b = np.where(dec.isin(["likely_match", "possible_match_insufficient_evidence"]),
                      "entail", "contradict")
    sB = score(df["gold_bin"].values, pred_b, "B_lenient")

    # === C) Abstention-aware
    abstain = dec.isin(["possible_match_insufficient_evidence", "cannot_determine"]) | (~parse_ok)
    kept = ~abstain
    pred_c = np.where(dec[kept] == "likely_match", "entail", "contradict")
    sC = score(df.loc[kept, "gold_bin"].values, pred_c, "C_abstention_aware")
    sC["coverage"] = float(kept.mean())
    sC["abstained_n"] = int(abstain.sum())

    # === D) Forced-pick for abstentions: random label by prior
    rng = np.random.default_rng(42)
    base_rate_entail = (df["gold_bin"] == "entail").mean()
    forced = pred_b.copy()  # lenient as base
    forced[abstain.values] = rng.choice(["entail", "contradict"],
                                        size=int(abstain.sum()),
                                        p=[base_rate_entail, 1 - base_rate_entail])
    sD = score(df["gold_bin"].values, forced, "D_forced_pick")

    # Decision histogram
    dec_hist = dec.value_counts().to_dict()
    parse_rate = float(parse_ok.mean())

    summary = {
        "n_total": int(len(df)),
        "parse_ok_rate": parse_rate,
        "decision_histogram": dec_hist,
        "scorings": [sA, sB, sC, sD],
    }

    out = root / "outputs/tables/nli4pr_eligibility_alignment_binary.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"\n[wrote] {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
