#!/usr/bin/env python3
"""
Phase 5 Task 6 — claim-level NLI redaction module.

Splits patient-facing fields into sentences and drops any sentence not entailed
by the accepted-trial evidence E* (premise). Closes Theorem boundary (iv) — the
paraphrased-semantic channel — constructively.

Design (frozen in configs/phase5_baselines.yaml):
  premise    = accepted-trial passages, chunked to model max length
               (max-entailment over chunks)
  hypothesis = each sentence of the patient-facing fields
  checkpoint = MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli
  threshold  = calibrated on the 15-case dev pool ONLY
               (objective: highest T with benign-drop <= 0.10)

The scorer is injected, so the decision logic is unit-testable on CPU with a
stub (no GPU / model download required for tests).

Run (server, real model):
  python scripts_phase5/nli_redact.py --calibrate
  python scripts_phase5/nli_redact.py --apply 684
  python scripts_phase5/nli_redact.py --apply 912
Test (anywhere):
  python scripts_phase5/nli_redact.py --selftest
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
FALLBACK = "not clearly stated in the retrieved evidence"
# Leak-TARGETED claim verification (final design).
#
# Two earlier framings failed and the failures are themselves findings:
#   (1) sentence-NLI on patient_facing_answer: the guarded summary is an
#       intentionally generic hedge that entails nothing specific (NLI~0), so a
#       "drop low-entailment" filter nukes benign content (dev benign-drop=1.0).
#   (2) NLI on supported_patient_facts vs the selected trial: those are *patient*
#       attributes ("55-year-old woman"), which a trial's *criteria* text does not
#       naturally entail (wrong relation; dev benign-drop=0.9).
#
# Correct, collateral-light design: a claim is cross-trial contamination iff it is
# better supported by a NON-selected pool trial than by the selected trial. We
# redact a claim sentence S iff  e_other(S) >= T  AND  e_other(S) > e_sel(S),
# where e_sel = entailment vs the selected trial's passages and e_other = max
# entailment vs any non-selected pool trial's passages. Generic/benign content is
# entailed by neither -> kept. This directly instruments the T1 channel and leaves
# single-trial-evidence outputs (no "other" trial) untouched by construction.
CLAIM_SENTENCE_FIELDS = ("patient_facing_answer", "reasoning_summary")
CLAIM_LIST_FIELDS = ("supported_patient_facts",)
SKIP_SENTENCES = {FALLBACK.lower(), ""}

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def split_sentences(text: str) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    return [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]


def chunk_premise(passages: list[str], max_chars: int = 1800) -> list[str]:
    chunks, cur = [], ""
    for p in passages:
        if len(cur) + len(p) + 1 > max_chars and cur:
            chunks.append(cur)
            cur = ""
        cur = (cur + "\n" + p).strip()
    if cur:
        chunks.append(cur)
    return chunks or [""]


# scorer signature: (premise_chunk: str, hypothesis: str) -> entailment_prob in [0,1]
Scorer = Callable[[str, str], float]


def sentence_entailment(scorer: Scorer, premise_chunks: list[str], sent: str) -> float:
    """Max entailment over premise chunks."""
    return max((scorer(c, sent) for c in premise_chunks), default=0.0)


def claim_units(rec: dict) -> list[tuple[str, str]]:
    """Yield (field, claim_text): prose sentences + concrete list items."""
    parsed = rec.get("parsed") or {}
    out = []
    for field in CLAIM_SENTENCE_FIELDS:
        src = rec.get(field) or parsed.get(field) or ""
        if str(src).strip().lower() in SKIP_SENTENCES:
            continue
        for s in split_sentences(str(src)):
            out.append((field, s))
    for field in CLAIM_LIST_FIELDS:
        lst = rec.get(field) or parsed.get(field) or []
        if isinstance(lst, list):
            for c in lst:
                s = str(c).strip()
                if s and s.lower() not in SKIP_SENTENCES:
                    out.append((field, s))
    return out


def split_pool_premises(rec: dict, pool_passages: dict | None):
    """Return (sel_chunks, other_chunks) for the selected vs non-selected trials.

    pool_passages: optional {trial_id: [passage_text,...]} for the case's full
    retrieval pool. Falls back to the record's own embedded passages.
    """
    ref = str(rec.get("reference_trial") or rec.get("selected_doc") or "")
    sel_txt, other_txt = [], []
    src = pool_passages
    if not src:
        src = {}
        for p in (rec.get("passages") or []):
            src.setdefault(str(p.get("doc_id", "")), []).append(str(p.get("passage_text", "")))
    for trial, txts in src.items():
        (sel_txt if trial == ref else other_txt).extend(txts)
    if not sel_txt:
        sel_txt = [str(rec.get("evidence", ""))]
    return chunk_premise(sel_txt), (chunk_premise(other_txt) if other_txt else [])


def redact_record(rec: dict, scorer: Scorer, threshold: float,
                  pool_passages: dict | None = None, margin: float = 0.0) -> dict:
    """Drop a claim iff better-entailed by a non-selected pool trial (T1 channel)."""
    sel_chunks, other_chunks = split_pool_premises(rec, pool_passages)
    out = json.loads(json.dumps(rec, default=str))  # deep copy
    parsed = out.get("parsed") or {}
    dropped = []
    drop_sent = set()
    drop_listitem: dict[str, set] = {}
    for field, claim in claim_units(out):
        e_other = sentence_entailment(scorer, other_chunks, claim) if other_chunks else 0.0
        if e_other < threshold:
            continue
        e_sel = sentence_entailment(scorer, sel_chunks, claim)
        if e_other > e_sel + margin:
            dropped.append({"field": field, "claim": claim,
                            "e_other": round(e_other, 4), "e_sel": round(e_sel, 4)})
            if field in CLAIM_LIST_FIELDS:
                drop_listitem.setdefault(field, set()).add(claim)
            else:
                drop_sent.add(claim)
    # rebuild sentence fields (drop flagged sentences)
    for field in CLAIM_SENTENCE_FIELDS:
        src = out.get(field) or parsed.get(field) or ""
        if not str(src).strip() or not drop_sent:
            continue
        kept = [s for s in split_sentences(str(src)) if s not in drop_sent]
        new = " ".join(kept) if kept else FALLBACK
        if field in out:
            out[field] = new
        if field in parsed:
            parsed[field] = new
    # rebuild list fields
    for field in CLAIM_LIST_FIELDS:
        drops = drop_listitem.get(field, set())
        if not drops:
            continue
        for container in (out, parsed):
            if isinstance(container.get(field), list):
                container[field] = [c for c in container[field] if str(c).strip() not in drops]
    out["parsed"] = parsed
    out["nli_dropped"] = dropped
    out["nli_threshold"] = threshold
    out["nli_n_dropped"] = len(dropped)
    return out


# ---------------------------------------------------------------------------
def _load_real_scorer(checkpoint: str) -> Scorer:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tok = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    # label order for this family: 0=entailment, 1=neutral, 2=contradiction
    ent_idx = 0
    labels = getattr(model.config, "id2label", {})
    for i, name in labels.items():
        if str(name).lower().startswith("entail"):
            ent_idx = int(i)

    def scorer(premise: str, hypothesis: str) -> float:
        with torch.no_grad():
            x = tok(premise, hypothesis, return_tensors="pt", truncation=True, max_length=512)
            if torch.cuda.is_available():
                x = {k: v.cuda() for k, v in x.items()}
            probs = model(**x).logits.softmax(-1)[0]
        return float(probs[ent_idx].item())
    return scorer


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--apply", type=int, choices=[684, 912])
    args = ap.parse_args()

    if args.selftest:
        # stub scorer: entailment ~ keyword overlap with the premise text
        def stub(premise: str, hyp: str) -> float:
            sel_kw = {"osteoporosis", "raloxifene", "vertebral"}
            oth_kw = {"leukemia", "pediatric", "immunotherapy"}
            p = premise.lower(); h = hyp.lower()
            if any(k in p for k in oth_kw) and any(k in h for k in oth_kw):
                return 0.95
            if any(k in p for k in sel_kw) and any(k in h for k in sel_kw):
                return 0.95
            return 0.05

        # two-trial pool: selected (osteoporosis) + other (leukemia).
        rec = {
            "reference_trial": "NCT_SEL",
            "passages": [
                {"doc_id": "NCT_SEL", "passage_text": "postmenopausal women with osteoporosis and a vertebral fracture; raloxifene arm"},
                {"doc_id": "NCT_OTH", "passage_text": "pediatric leukemia immunotherapy study"},
            ],
            "patient_facing_answer": "You may qualify based on your osteoporosis. The trial provides leukemia immunotherapy.",
            "parsed": {"patient_facing_answer": "You may qualify based on your osteoporosis. The trial provides leukemia immunotherapy."},
        }
        out = redact_record(rec, stub, threshold=0.5)
        # only the leukemia sentence is better-entailed by the OTHER trial -> dropped
        assert out["nli_n_dropped"] == 1, out["nli_dropped"]
        assert "leukemia" not in out["patient_facing_answer"], out["patient_facing_answer"]
        assert "osteoporosis" in out["patient_facing_answer"]
        # single-trial evidence (no other premise) -> nothing dropped by construction
        rec2 = {"reference_trial": "NCT_SEL",
                "passages": [{"doc_id": "NCT_SEL", "passage_text": "osteoporosis raloxifene"}],
                "patient_facing_answer": "The trial provides leukemia immunotherapy."}
        out2 = redact_record(rec2, stub, threshold=0.5)
        assert out2["nli_n_dropped"] == 0, "single-trial pool must never redact"
        assert len(split_sentences("A b c. D e f! G h?")) == 3
        print("[selftest] PASS — leak-targeted drop (e_other>e_sel), single-trial untouched")
        return 0

    # ---- real model paths (server GPU) ----
    import csv
    import yaml
    cfg = yaml.safe_load((ROOT / "configs" / "phase5_baselines.yaml").read_text())
    checkpoint = cfg["nli"]["checkpoint"]

    DEV_DIR = ROOT / "outputs/phase4/15case_audit/15case_audit_gens/V-final"
    FROZEN_ROOTS = [
        ROOT / "outputs/backbone_gens",
        ROOT / "outputs/phase4/n100_expansion/gens",
        ROOT / "outputs/phase4/zeroshot_baseline/gens",
    ]
    BASELINE_ROOT = ROOT / "outputs/phase5/baseline_gens"
    NLI_OUT = ROOT / "outputs/phase5/nli"
    NLI_OUT.mkdir(parents=True, exist_ok=True)

    POOL_DIR = ROOT / "outputs/phase5/evidence_pools"
    import pandas as pd

    def pool_for(case_id: str) -> dict | None:
        fp = POOL_DIR / f"{case_id}.csv"
        if not fp.exists():
            return None
        df = pd.read_csv(fp)
        out: dict[str, list] = {}
        for _, r in df.iterrows():
            out.setdefault(str(r["doc_id"]), []).append(str(r["passage_text"]))
        return out

    def load_dir_recs(root: Path):
        for fp in sorted(root.rglob("*.json")):
            if fp.name.endswith(".nli_redacted.json"):
                continue
            try:
                yield fp, json.loads(fp.read_text())
            except Exception:
                continue

    if args.calibrate:
        scorer = _load_real_scorer(checkpoint)
        # benign-drop = fraction of dev V-final claims the leak-targeted rule would
        # redact at threshold T. Objective (frozen): highest T with benign-drop<=0.10.
        # Evaluated over the dev pool with each case's multi-trial evidence pool, so
        # "other-trial" entailment is actually computable.
        per_claim_min_other = []  # (e_other, e_sel) for every dev claim
        for fp, rec in load_dir_recs(DEV_DIR):
            pool = pool_for(str(rec.get("case_id", "")))
            sel_chunks, other_chunks = split_pool_premises(rec, pool)
            if not other_chunks:
                continue
            for _field, claim in claim_units(rec):
                e_other = sentence_entailment(scorer, other_chunks, claim)
                e_sel = sentence_entailment(scorer, sel_chunks, claim)
                per_claim_min_other.append((e_other, e_sel))
        n = len(per_claim_min_other)
        rows = []
        for T in [x / 100 for x in range(50, 100, 5)]:
            dropped = sum(1 for eo, es in per_claim_min_other if eo >= T and eo > es)
            rows.append({"threshold": T, "benign_drop_rate": round(dropped / n, 4) if n else 0.0,
                         "n_dev_claims": n})
        chosen = max((r["threshold"] for r in rows if r["benign_drop_rate"] <= 0.10), default=0.5)
        with (NLI_OUT / "calibration.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["threshold", "benign_drop_rate", "n_dev_claims"])
            w.writeheader(); w.writerows(rows)
        cfg["nli"]["calibration"]["chosen_threshold"] = float(chosen)
        (ROOT / "configs" / "phase5_baselines.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
        print(f"[calibrate] n_dev_claims={n}; chosen_threshold={chosen}")
        print("\n".join(f"  T={r['threshold']:.2f} drop={r['benign_drop_rate']:.3f}" for r in rows))
        return 0

    if args.apply:
        T = cfg["nli"]["calibration"].get("chosen_threshold")
        if T is None:
            print("[FATAL] run --calibrate first (chosen_threshold is null)")
            return 1
        scorer = _load_real_scorer(checkpoint)

        # consensus semantic-flagged cases (both phase-4 judges agree leak=1)
        def consensus_keys() -> set:
            import collections
            cnt = collections.Counter()
            for j in ("sonnet", "gpt4o"):
                p = ROOT / "outputs/phase4/reviewer_fixes" / f"semantic_leak_judge_{j}.jsonl"
                if not p.exists():
                    continue
                for line in p.read_text().splitlines():
                    if not line.strip():
                        continue
                    d = json.loads(line)
                    if d.get("semantic_leak") == 1:
                        cnt[f"{d.get('case_id')}|{d.get('backbone')}"] += 1
            return {k for k, v in cnt.items() if v >= 2}

        consensus = consensus_keys()
        roots = FROZEN_ROOTS if args.apply == 684 else [BASELINE_ROOT]
        rows = []
        for root in roots:
            for fp, rec in load_dir_recs(root):
                pool = pool_for(str(rec.get("case_id", "")))
                out = redact_record(rec, scorer, T, pool_passages=pool)
                key = f"{rec.get('case_id')}|{rec.get('backbone')}"
                rows.append({
                    "path": str(fp.relative_to(ROOT)),
                    "case_id": rec.get("case_id"), "backbone": rec.get("backbone"),
                    "baseline_id": rec.get("baseline_id", "V-final"),
                    "n_dropped": out["nli_n_dropped"],
                    "is_consensus_flagged": key in consensus,
                })
                # persist redacted copy alongside
                rfp = fp.with_suffix(".nli_redacted.json")
                rfp.write_text(json.dumps(out, indent=2, default=str))
        df = pd.DataFrame(rows)
        tag = str(args.apply)
        df.to_csv(NLI_OUT / f"redaction_{tag}.csv", index=False)
        total = len(df); any_drop = int((df["n_dropped"] > 0).sum())
        cons = df[df["is_consensus_flagged"]]
        print(f"[apply {tag}] gens={total}; with>=1 drop={any_drop} ({any_drop/total:.3f}); "
              f"total sentences dropped={int(df['n_dropped'].sum())}")
        if len(cons):
            print(f"  consensus-flagged gens={len(cons)}; "
                  f"redacted (>=1 drop)={int((cons['n_dropped']>0).sum())}/{len(cons)}")
        print(f"  benign collateral (non-consensus gens with a drop)="
              f"{int((df[~df['is_consensus_flagged']]['n_dropped']>0).sum())}/{total-len(cons)}")
        return 0

    print("[info] pass --selftest, --calibrate, or --apply {684|912}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
