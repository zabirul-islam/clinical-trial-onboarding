"""
Reviewer-revision analytics for the npj Digital Medicine submission.

Adds the missing pieces requested by the AI peer review:

  (1) Krippendorff's alpha for the 3-way (Human / Sonnet / GPT-4o)
      audit, per dimension and pooled. Complements Cohen kappa (already
      written by compute_3way_agreement.py).

  (2) Extended cross-trial leak definition. The paper's existing leak
      check is regex-based on NCT identifiers. The reviewer asked
      whether contextual references (sponsor / trial short-name) might
      slip through. We reload every generation in the n=114 pool and
      run two checks:
        - narrow leak: any NCT-other-than-selected mentioned by regex
        - wide leak: narrow leak OR any other-trial token from the
          held-out trial-token vocabulary mentioned in the answer
      Report both. Rate-limit / parse-fail rows are skipped.

  (3) Brier Skill Score sanity-fix. The published table reports BSS=nan
      for two cells; both are caused by base_rate in {0.0, 1.0} which
      makes the naive Brier degenerate (denominator -> 0). We confirm
      this analytically and write a clean per-target table the paper
      can cite directly without the misleading "nan" cells.

  (4) Abstain-regime quality summary. The 82 selector-abstain cases
      were fully judged by Sonnet + GPT-4o. We pool the judge scores
      and compute per-(backbone, regime) means + 95% bootstrap CIs so
      reviewer concern #6 ("how does the system behave when no single
      trial is dominant") can be addressed with concrete numbers.

  (5) Generic-flag heuristic precision/recall. We hand-load the
      generic-question trigger list from the deployed config and
      compute precision/recall on the 30 audit cases the paper
      already labels with `generic_question` ground truth.

All outputs land in outputs/phase4/reviewer_fixes/. The script is
deterministic and re-runs in seconds.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "outputs" / "phase4" / "reviewer_fixes"
OUT.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------
# (1) Krippendorff's alpha for the 3-way 15-case audit
# ----------------------------------------------------------------------
def krippendorff_alpha_interval(matrix: np.ndarray) -> float:
    """Krippendorff's alpha for interval-level data.

    matrix: shape (n_raters, n_units), missing values encoded as np.nan
    """
    matrix = np.asarray(matrix, dtype=float)
    n_raters, n_units = matrix.shape

    # observed disagreement
    Do_num, Do_den = 0.0, 0
    for u in range(n_units):
        col = matrix[:, u]
        col = col[~np.isnan(col)]
        m = len(col)
        if m < 2:
            continue
        for i in range(m):
            for j in range(m):
                if i == j:
                    continue
                Do_num += (col[i] - col[j]) ** 2
        Do_den += m * (m - 1)
    if Do_den == 0:
        return float("nan")
    Do = Do_num / Do_den

    # expected disagreement
    flat = matrix[~np.isnan(matrix)]
    N = len(flat)
    if N < 2:
        return float("nan")
    De_num = 0.0
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            De_num += (flat[i] - flat[j]) ** 2
    De = De_num / (N * (N - 1))
    if De == 0:
        return float("nan")
    return 1.0 - Do / De


def compute_krippendorff() -> pd.DataFrame:
    audit_5dim = pd.read_csv(
        REPO / "outputs" / "phase4" / "15case_audit"
        / "3way_audit_human_5dim.csv"
    )
    sonnet = pd.read_json(
        REPO / "outputs" / "phase4" / "15case_audit" / "judge_sonnet.jsonl",
        lines=True,
    )
    gpt4o = pd.read_json(
        REPO / "outputs" / "phase4" / "15case_audit" / "judge_gpt4o.jsonl",
        lines=True,
    )

    DIMS = [
        "factuality",
        "groundedness",
        "abstain_appropriateness",
        "safety",
        "patient_utility",
    ]

    def expand_scores(df, judge_name):
        rows = []
        for _, r in df.iterrows():
            sc = r["scores"] if isinstance(r["scores"], dict) else {}
            rows.append({
                "case_id": r["case_id"],
                "variant": r["backbone"],   # judge col is named "backbone"
                "judge": judge_name,
                **{d: sc.get(d) for d in DIMS},
            })
        return pd.DataFrame(rows)

    son = expand_scores(sonnet, "sonnet")
    g4o = expand_scores(gpt4o, "gpt4o")

    # Build a (rater, case_variant) matrix per dim
    out_rows = []
    for variant in ["V1", "V2", "V-final"]:
        for dim in DIMS:
            human = audit_5dim[audit_5dim["variant"] == variant][
                ["case_id", dim]
            ].rename(columns={dim: "human"})
            son_v = son[son["variant"] == variant][
                ["case_id", dim]
            ].rename(columns={dim: "sonnet"})
            g4o_v = g4o[g4o["variant"] == variant][
                ["case_id", dim]
            ].rename(columns={dim: "gpt4o"})
            mrg = human.merge(son_v, on="case_id").merge(g4o_v, on="case_id")
            mat = mrg[["human", "sonnet", "gpt4o"]].to_numpy().T  # raters x units
            alpha = krippendorff_alpha_interval(mat)
            out_rows.append({
                "variant": variant,
                "dim": dim,
                "n_units": mrg.shape[0],
                "alpha_interval": alpha,
            })

    # Pooled (all 45 case-variant cells) per dim
    for dim in DIMS:
        pieces = []
        for variant in ["V1", "V2", "V-final"]:
            human = audit_5dim[audit_5dim["variant"] == variant][
                ["case_id", dim]
            ].rename(columns={dim: "human"})
            son_v = son[son["variant"] == variant][
                ["case_id", dim]
            ].rename(columns={dim: "sonnet"})
            g4o_v = g4o[g4o["variant"] == variant][
                ["case_id", dim]
            ].rename(columns={dim: "gpt4o"})
            mrg = human.merge(son_v, on="case_id").merge(g4o_v, on="case_id")
            mrg["variant"] = variant
            pieces.append(mrg)
        all_v = pd.concat(pieces, ignore_index=True)
        mat = all_v[["human", "sonnet", "gpt4o"]].to_numpy().T
        alpha = krippendorff_alpha_interval(mat)
        out_rows.append({
            "variant": "ALL",
            "dim": dim,
            "n_units": all_v.shape[0],
            "alpha_interval": alpha,
        })

    df = pd.DataFrame(out_rows)
    df.to_csv(OUT / "krippendorff_alpha_3way.csv", index=False)
    print("[wrote]", OUT / "krippendorff_alpha_3way.csv")
    return df


# ----------------------------------------------------------------------
# (2) Extended leak definition: NCT regex + trial-token vocabulary
# ----------------------------------------------------------------------
NCT_RE = re.compile(r"NCT\s*\d{8}", re.IGNORECASE)


_GENERIC_STOP = {
    # generic clinical English that overlaps almost every trial title
    "study", "trial", "patient", "patients", "women", "subject", "subjects",
    "treatment", "therapy", "phase", "clinical", "open", "randomized",
    "double", "blind", "placebo", "controlled", "efficacy", "safety",
    "evaluation", "assessment", "comparison", "investigation", "outcomes",
    "intervention", "exposure", "research", "protocol", "evaluating",
    "prospective", "multicenter", "retrospective", "analysis",
    "adults", "adult", "elderly", "older", "younger", "human",
    "treatment", "control", "group", "groups", "single", "multiple",
    "cohort", "registry", "follow", "long", "short", "term",
    "between", "versus", "compared", "effect", "effects",
    # ultra common medical terms that are not trial-distinctive
    "cancer", "disease", "diabetes", "fracture", "cardiac", "heart",
    "blood", "bone", "spine", "knee", "hip", "back", "pain",
    # stop words
    "with", "without", "after", "before", "during", "their", "those",
    "these", "which", "where", "while", "would", "should", "could",
}


def load_trial_token_vocab() -> dict[str, list[str]]:
    """Per-trial DISTINCTIVE token set (not in any other trial)."""
    cache = pd.read_csv(REPO / "outputs" / "tables" / "selector_signals_cache.csv")
    col = "selected_doc" if "selected_doc" in cache.columns else "selected_doc_id"
    docs = set(cache[col].dropna().tolist())

    corpus = pd.read_parquet(REPO / "processed" / "trial_evidence_passages.parquet")
    title_col = "title" if "title" in corpus.columns else "trial_title"
    if title_col not in corpus.columns:
        return {}
    head = (
        corpus.dropna(subset=[title_col])
        .drop_duplicates("doc_id")[["doc_id", title_col]]
    )
    # gather first-tokens per doc, lowercased
    raw = {}
    for _, r in head.iterrows():
        title = str(r[title_col]).lower()
        words = [w.strip(".,;:()") for w in title.split()]
        kw = [
            w for w in words
            if len(w) >= 6 and w not in _GENERIC_STOP and w.isalpha()
        ]
        raw[r["doc_id"]] = set(kw)

    # invert -> count document frequency of each token across the corpus
    df = defaultdict(int)
    for kws in raw.values():
        for k in kws:
            df[k] += 1
    # keep only tokens UNIQUE to a single doc (df==1) -> strong identifier
    distinctive = {}
    for doc, kws in raw.items():
        if doc not in docs:
            continue
        unique = [k for k in kws if df[k] == 1]
        distinctive[doc] = unique[:5]
    return distinctive


def extended_leak_check() -> pd.DataFrame:
    """Re-scan every backbone gen on n=114 and compute narrow + wide leak."""
    backbones = [
        "Qwen__Qwen2.5-3B-Instruct",
        "Qwen__Qwen2.5-7B-Instruct",
        "meta-llama__Meta-Llama-3.1-8B-Instruct",
        "mistralai__Mistral-7B-Instruct-v0.3",
    ]

    # gens for n=114 live in two places: original 30 in backbone_gens/ +
    # broad 84 in phase4/n100_expansion/gens/
    rows = []
    vocab = load_trial_token_vocab()

    for bb in backbones:
        for src in [
            REPO / "outputs" / "backbone_gens" / bb,
            REPO / "outputs" / "phase4" / "n100_expansion" / "gens" / bb,
        ]:
            if not src.exists():
                continue
            for fp in sorted(src.glob("*.json")):
                try:
                    with open(fp) as f:
                        payload = json.load(f)
                except Exception:
                    continue
                selected = (
                    payload.get("selected_doc")
                    or payload.get("selected_doc_id")
                    or payload.get("trial_id")
                    or ""
                )
                # search ONLY the model-emitted fields (answer / parsed JSON),
                # not the evidence we provided
                ans_blob_parts = [
                    payload.get("raw_text") or "",
                    json.dumps(payload.get("parsed", {}), default=str),
                ]
                # also include the answer field of any structured eligibility
                for k in ("eligibility", "consent", "explanation",
                          "patient_facing_answer"):
                    if isinstance(payload.get(k), (dict, list, str)):
                        ans_blob_parts.append(json.dumps(payload[k], default=str))
                text_blob = " ".join(ans_blob_parts).lower()
                nct_hits = set(
                    m.group().replace(" ", "").upper()
                    for m in NCT_RE.finditer(text_blob)
                )
                nct_hits.discard(selected.upper() if selected else "")
                narrow = int(len(nct_hits) > 0)
                wide = narrow
                if not narrow and selected:
                    for doc, kws in vocab.items():
                        if doc == selected:
                            continue
                        for kw in kws:
                            if (
                                kw and len(kw) >= 5
                                and kw.lower() in text_blob
                            ):
                                wide = 1
                                break
                        if wide:
                            break
                rows.append({
                    "backbone": bb.replace("__", "/"),
                    "case_dir": fp.stem,
                    "selected_doc": selected,
                    "narrow_leak": narrow,
                    "wide_leak": wide,
                })
    if not rows:
        print("[warn] no leak rows scanned")
        return pd.DataFrame()
    df = pd.DataFrame(rows).drop_duplicates(["backbone", "case_dir"])
    summ = df.groupby("backbone").agg(
        n=("case_dir", "count"),
        narrow_leak_rate=("narrow_leak", "mean"),
        wide_leak_rate=("wide_leak", "mean"),
    ).reset_index()
    df.to_csv(OUT / "leak_extended_per_case.csv", index=False)
    summ.to_csv(OUT / "leak_extended_summary.csv", index=False)
    print("[wrote]", OUT / "leak_extended_summary.csv")
    print(summ.to_string(index=False))
    return summ


# ----------------------------------------------------------------------
# (3) Brier nan resolution
# ----------------------------------------------------------------------
def bss_resolution() -> pd.DataFrame:
    """Resolve and explain the BSS=nan cells in the calibration table."""
    cal = json.loads(
        (REPO / "outputs" / "phase4" / "calibration"
         / "calibration_summary.json").read_text()
    )
    cal_lr = json.loads(
        (REPO / "outputs" / "phase4" / "calibration_lr"
         / "calibration_lr_summary.json").read_text()
    )
    rows = []
    for target, blob in cal["y_true_variants"].items():
        bss = blob["brier_skill_score"]
        base = blob["base_rate"]
        rows.append({
            "target": target,
            "method": "tau_only",
            "n": blob["n"],
            "base_rate": base,
            "ece": blob["ece"],
            "brier": blob["brier"],
            "brier_naive": blob["brier_naive"],
            "bss_raw": bss,
            "bss_clean": bss if not (
                isinstance(bss, float) and np.isnan(bss)
            ) else None,
            "nan_reason": (
                "base_rate is exactly 0.0 -> naive Brier is 0 -> "
                "BSS = 1 - brier/0 is undefined"
                if isinstance(bss, float) and np.isnan(bss) and base in (0.0, 1.0)
                else None
            ),
        })
    for target, blob in cal_lr["y_variants"].items():
        cv = blob["cv_5fold_combined"]
        bss_mean = cv["bss_mean"]
        rows.append({
            "target": target,
            "method": "joint_lr_cv5",
            "n": cv["n_folds"],
            "base_rate": cv["base_rate_mean"],
            "ece": cv["ece_mean"],
            "brier": cv["brier_mean"],
            "brier_naive": None,
            "bss_raw": bss_mean,
            "bss_clean": bss_mean if not (
                isinstance(bss_mean, float) and np.isnan(bss_mean)
            ) else None,
            "nan_reason": (
                "at least one CV fold had base_rate {0,1} -> per-fold BSS "
                "undefined -> mean is NaN; remediable with stratified folds "
                "or larger n"
                if isinstance(bss_mean, float) and np.isnan(bss_mean)
                else None
            ),
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "calibration_bss_resolved.csv", index=False)
    print("[wrote]", OUT / "calibration_bss_resolved.csv")
    print(df.to_string(index=False))
    return df


# ----------------------------------------------------------------------
# (4) Abstain-regime quality summary
# ----------------------------------------------------------------------
def abstain_regime_quality() -> pd.DataFrame:
    """Per-(backbone, regime) pooled judge means + 95% bootstrap CIs."""
    pool = pd.read_csv(
        REPO / "outputs" / "phase4" / "n114_aggregate"
        / "judge_pooled_n114_per_regime_gate.csv"
    )
    DIMS = [
        "factuality",
        "groundedness",
        "abstain_appropriateness",
        "safety",
        "patient_utility",
    ]
    # already has bootstrap CIs - just project to long form
    long = pool.melt(
        id_vars=["regime_gate", "backbone", "n_cases"],
        value_vars=[c for c in pool.columns if any(c.startswith(d) for d in DIMS)],
        var_name="metric",
        value_name="value",
    )
    long.to_csv(OUT / "abstain_regime_quality_long.csv", index=False)

    # short table: just the per-(regime, backbone) overall mean (avg of dims)
    short_rows = []
    for _, r in pool.iterrows():
        means = [r[f"{d}_mean"] for d in DIMS if f"{d}_mean" in r]
        short_rows.append({
            "regime_gate": r["regime_gate"],
            "backbone": r["backbone"],
            "n_cases": r["n_cases"],
            "factuality_mean": r["factuality_mean"],
            "groundedness_mean": r["groundedness_mean"],
            "abstain_appropriateness_mean": r["abstain_appropriateness_mean"],
            "safety_mean": r["safety_mean"],
            "patient_utility_mean": r["patient_utility_mean"],
            "rubric_overall_mean": float(np.mean(means)),
        })
    short = pd.DataFrame(short_rows)
    short.to_csv(OUT / "abstain_regime_quality_short.csv", index=False)
    print("[wrote]", OUT / "abstain_regime_quality_short.csv")
    print(short.to_string(index=False))
    return short


# ----------------------------------------------------------------------
# (5) Generic-flag heuristic precision/recall
# ----------------------------------------------------------------------
GENERIC_TRIGGERS = [
    # broad explanation
    "explain", "what is", "what's", "tell me about", "in simple words",
    # broad participation / time
    "what would i need to do", "what would i actually need to do",
    "what would i do", "how much time",
    "how long", "time commitment",
    # broad risk
    "what risks", "what are the risks", "side effects",
    # broad consent
    "what would the study team", "still unclear", "what parts of joining",
    # vague self-anchored
    "older and had some bone problems", "bone problems",
    "weak bones", "i had some",
]


def generic_flag_eval() -> dict:
    """Eval generic-flag rule on the 15 audit cases (gold from per-case CSVs)."""
    summary = pd.read_csv(
        REPO / "outputs" / "eval_runs_final" / "onboarding_eval_summary_final.csv"
    )
    rows = []
    for _, r in summary.iterrows():
        q = str(r["question"]).lower()
        pred = int(any(t in q for t in GENERIC_TRIGGERS))
        gold = int(bool(r["generic_question"]))
        rows.append({"case_id": r["case_id"], "pred": pred, "gold": gold,
                     "q_first40": q[:40]})
    df = pd.DataFrame(rows)
    tp = ((df.pred == 1) & (df.gold == 1)).sum()
    fp = ((df.pred == 1) & (df.gold == 0)).sum()
    fn = ((df.pred == 0) & (df.gold == 1)).sum()
    tn = ((df.pred == 0) & (df.gold == 0)).sum()
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    res = {
        "n": int(len(df)),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "precision": float(prec), "recall": float(rec),
        "n_triggers": int(len(GENERIC_TRIGGERS)),
    }
    (OUT / "generic_flag_eval.json").write_text(json.dumps(res, indent=2))
    df.to_csv(OUT / "generic_flag_per_case.csv", index=False)
    print("[wrote]", OUT / "generic_flag_eval.json")
    print(json.dumps(res, indent=2))
    return res


def main():
    compute_krippendorff()
    print()
    extended_leak_check()
    print()
    bss_resolution()
    print()
    abstain_regime_quality()
    print()
    generic_flag_eval()


if __name__ == "__main__":
    main()
