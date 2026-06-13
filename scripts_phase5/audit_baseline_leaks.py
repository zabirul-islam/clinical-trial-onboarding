#!/usr/bin/env python3
"""
Phase 5 Task 4 — deterministic lexical leak audit on baseline generations.

Ports the phase-4 narrow (NCT regex) + wide (distinctive trial-token) detectors
verbatim from scripts_phase4/reviewer_revision_analytics.py and applies them to
the B1-B4 baseline gens. Reference trial per baseline = rec["reference_trial"]
(rank-1 passage's trial), so "other-trial" content counts as cross-trial.

Distinctive-token vocabulary is loaded from a portable cache
(outputs/phase5/trial_token_vocab.csv: doc_id,tokens) built once by
--build-vocab on the server (needs the trial-passage parquet); falls back to
the parquet directly if present.

Run:
  # once on server (has the parquet):
  python scripts_phase5/audit_baseline_leaks.py --build-vocab
  # anywhere (uses cached vocab):
  python scripts_phase5/audit_baseline_leaks.py
  python scripts_phase5/audit_baseline_leaks.py --selftest
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
GENS_ROOT = ROOT / "outputs" / "phase5" / "baseline_gens"
VOCAB_CACHE = ROOT / "outputs" / "phase5" / "trial_token_vocab.csv"
OUT = ROOT / "outputs" / "phase5"

NCT_RE = re.compile(r"NCT\s*\d{8}", re.IGNORECASE)

_GENERIC_STOP = {
    "study", "trial", "patient", "patients", "women", "subject", "subjects",
    "treatment", "therapy", "phase", "clinical", "open", "randomized",
    "double", "blind", "placebo", "controlled", "efficacy", "safety",
    "evaluation", "assessment", "comparison", "investigation", "outcomes",
    "intervention", "exposure", "research", "protocol", "evaluating",
    "prospective", "multicenter", "retrospective", "analysis",
    "adults", "adult", "elderly", "older", "younger", "human",
    "control", "group", "groups", "single", "multiple",
    "cohort", "registry", "follow", "long", "short", "term",
    "between", "versus", "compared", "effect", "effects",
    "cancer", "disease", "diabetes", "fracture", "cardiac", "heart",
    "blood", "bone", "spine", "knee", "hip", "back", "pain",
    "with", "without", "after", "before", "during", "their", "those",
    "these", "which", "where", "while", "would", "should", "could",
}


def build_vocab() -> dict[str, list[str]]:
    """Distinctive (corpus-unique) >=6-char title tokens per trial."""
    pq = ROOT / "data" / "processed" / "trial_evidence_passages.parquet"
    if not pq.exists():
        pq = ROOT / "processed" / "trial_evidence_passages.parquet"
    corpus = pd.read_parquet(pq)
    title_col = "title" if "title" in corpus.columns else "trial_title"
    head = corpus.dropna(subset=[title_col]).drop_duplicates("doc_id")[["doc_id", title_col]]
    raw = {}
    for _, r in head.iterrows():
        words = [w.strip(".,;:()") for w in str(r[title_col]).lower().split()]
        raw[r["doc_id"]] = {w for w in words
                            if len(w) >= 6 and w not in _GENERIC_STOP and w.isalpha()}
    df = defaultdict(int)
    for kws in raw.values():
        for k in kws:
            df[k] += 1
    return {doc: [k for k in kws if df[k] == 1] for doc, kws in raw.items()}


def load_vocab() -> dict[str, list[str]]:
    if VOCAB_CACHE.exists():
        v = pd.read_csv(VOCAB_CACHE)
        return {r["doc_id"]: str(r["tokens"]).split("|") if pd.notna(r["tokens"]) else []
                for _, r in v.iterrows()}
    return build_vocab()


def detect(text_blob: str, reference_trial: str, vocab: dict[str, list[str]]) -> tuple[int, int]:
    """Returns (narrow, wide) for a model-emitted text blob (lower-cased)."""
    tb = text_blob.lower()
    ref = (reference_trial or "").upper()
    nct_hits = {m.group().replace(" ", "").upper() for m in NCT_RE.finditer(tb)}
    nct_hits.discard(ref)
    narrow = int(len(nct_hits) > 0)
    wide = narrow
    if not narrow and reference_trial:
        for doc, kws in vocab.items():
            if doc == reference_trial:
                continue
            if any(kw and len(kw) >= 5 and kw.lower() in tb for kw in kws):
                wide = 1
                break
    return narrow, wide


def emitted_blob(rec: dict) -> str:
    parts = [rec.get("raw_generation") or rec.get("raw_text") or "",
             json.dumps(rec.get("parsed", {}), default=str),
             str(rec.get("patient_facing_answer", ""))]
    return " ".join(parts)


def run_audit() -> pd.DataFrame:
    vocab = load_vocab()
    rows = []
    for baseline_dir in sorted(GENS_ROOT.glob("*")):
        if not baseline_dir.is_dir():
            continue
        for bb_dir in sorted(baseline_dir.glob("*")):
            for fp in sorted(bb_dir.glob("*.json")):
                rec = json.loads(fp.read_text())
                ref = rec.get("reference_trial") or rec.get("selected_doc") or ""
                narrow, wide = detect(emitted_blob(rec), ref, vocab)
                rows.append({
                    "baseline_id": baseline_dir.name,
                    "backbone": bb_dir.name.replace("__", "/"),
                    "case_id": fp.stem,
                    "reference_trial": ref,
                    "parse_ok": bool(rec.get("parse_ok")),
                    "decision": rec.get("decision", ""),
                    "narrow_leak": narrow,
                    "wide_leak": wide,
                })
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-vocab", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        vocab = {"NCT00000001": ["osteoporosis", "raloxifene"],
                 "NCT00000002": ["pembrolizumab"]}
        # fixture 1: planted NCT leak (other trial id) -> narrow+wide
        n, w = detect("you may qualify for NCT00000002 instead", "NCT00000001", vocab)
        assert (n, w) == (1, 1), (n, w)
        # fixture 2: planted distinctive-token leak (no NCT) -> wide only
        n, w = detect("this study uses pembrolizumab therapy", "NCT00000001", vocab)
        assert (n, w) == (0, 1), (n, w)
        # fixture 3: clean (only reference-trial vocab / generic) -> no leak
        n, w = detect("raloxifene is given for this osteoporosis study", "NCT00000001", vocab)
        assert (n, w) == (0, 0), (n, w)
        print("[selftest] PASS — narrow+wide detectors behave on 3 fixtures")
        return 0

    if args.build_vocab:
        v = build_vocab()
        VOCAB_CACHE.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{"doc_id": d, "tokens": "|".join(t)} for d, t in v.items()]) \
            .to_csv(VOCAB_CACHE, index=False)
        print(f"[wrote] {VOCAB_CACHE} ({len(v)} trials)")
        return 0

    df = run_audit()
    if df.empty:
        print("[wait] no baseline gens yet under", GENS_ROOT)
        return 0
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "baseline_leak_per_case.csv", index=False)

    def cp(k, n):
        from scipy.stats import beta
        lo = 0.0 if k == 0 else beta.ppf(0.025, k, n - k + 1)
        hi = 1.0 if k == n else beta.ppf(0.975, k + 1, n - k)
        return round(lo, 4), round(hi, 4)

    summ = df.groupby(["baseline_id", "backbone"]).agg(
        n=("case_id", "count"),
        parse_ok_rate=("parse_ok", "mean"),
        narrow_leak_rate=("narrow_leak", "mean"),
        wide_leak_rate=("wide_leak", "mean"),
    ).reset_index()
    summ["narrow_ci95"] = summ.apply(
        lambda r: cp(int(r.n * r.narrow_leak_rate), int(r.n)), axis=1)
    summ["wide_ci95"] = summ.apply(
        lambda r: cp(int(r.n * r.wide_leak_rate), int(r.n)), axis=1)
    summ.to_csv(OUT / "baseline_leak_summary.csv", index=False)
    print(summ.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
