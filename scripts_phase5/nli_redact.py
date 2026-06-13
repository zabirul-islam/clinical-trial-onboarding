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
PATIENT_FIELDS = ("patient_facing_answer", "reasoning_summary")
# fallback strings and structured non-prose fields are never NLI-filtered
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


def redact_record(rec: dict, scorer: Scorer, threshold: float) -> dict:
    """Returns a copy with low-entailment sentences dropped + an audit log."""
    passages = [str(p.get("passage_text", "")) for p in (rec.get("passages") or [])]
    if not passages:
        passages = [str(rec.get("evidence", ""))]
    chunks = chunk_premise(passages)

    out = json.loads(json.dumps(rec, default=str))  # deep copy
    parsed = out.get("parsed") or {}
    dropped = []
    for field in PATIENT_FIELDS:
        src = out.get(field) or parsed.get(field) or ""
        if str(src).strip().lower() in SKIP_SENTENCES:
            continue
        kept = []
        for sent in split_sentences(str(src)):
            ent = sentence_entailment(scorer, chunks, sent)
            if ent >= threshold:
                kept.append(sent)
            else:
                dropped.append({"field": field, "sentence": sent, "entailment": round(ent, 4)})
        new_text = " ".join(kept) if kept else FALLBACK
        if field in out:
            out[field] = new_text
        if field in parsed:
            parsed[field] = new_text
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
        # stub scorer: entailed iff hypothesis shares a keyword with premise
        def stub(premise: str, hyp: str) -> float:
            kws = {"osteoporosis", "raloxifene", "vertebral", "fracture"}
            p_has = any(k in premise.lower() for k in kws)
            h_has = any(k in hyp.lower() for k in kws)
            return 0.95 if (p_has and h_has) else 0.10

        rec = {
            "passages": [{"passage_text": "Inclusion: postmenopausal women with osteoporosis and a vertebral fracture; raloxifene arm."}],
            "patient_facing_answer": "You may qualify based on osteoporosis. The study also enrolls pediatric leukemia patients.",
            "parsed": {"patient_facing_answer": "You may qualify based on osteoporosis. The study also enrolls pediatric leukemia patients."},
        }
        out = redact_record(rec, stub, threshold=0.5)
        assert out["nli_n_dropped"] == 1, out["nli_dropped"]
        assert "pediatric leukemia" not in out["patient_facing_answer"]
        assert "osteoporosis" in out["patient_facing_answer"]
        # all-dropped -> fallback
        rec2 = {"passages": [{"passage_text": "osteoporosis raloxifene"}],
                "patient_facing_answer": "This trial studies pediatric leukemia immunotherapy."}
        out2 = redact_record(rec2, stub, threshold=0.5)
        assert out2["patient_facing_answer"] == FALLBACK, out2["patient_facing_answer"]
        # fallback string is never filtered
        rec3 = {"passages": [{"passage_text": "x"}], "patient_facing_answer": FALLBACK}
        out3 = redact_record(rec3, stub, threshold=0.5)
        assert out3["nli_n_dropped"] == 0
        # sentence splitter
        assert len(split_sentences("A b c. D e f! G h?")) == 3
        print("[selftest] PASS — split, drop, fallback, skip-fallback all correct")
        return 0

    # ---- real model paths (server GPU) ----
    import csv
    import yaml
    cfg = yaml.safe_load((REPO / "configs" / "phase5_baselines.yaml").read_text())
    checkpoint = cfg["nli"]["checkpoint"]

    DEV_DIR = REPO / "outputs/phase4/15case_audit/15case_audit_gens/V-final"
    FROZEN_ROOTS = [
        REPO / "outputs/backbone_gens",
        REPO / "outputs/phase4/n100_expansion/gens",
        REPO / "outputs/phase4/zeroshot_baseline/gens",
    ]
    BASELINE_ROOT = REPO / "outputs/phase5/baseline_gens"
    NLI_OUT = REPO / "outputs/phase5/nli"
    NLI_OUT.mkdir(parents=True, exist_ok=True)

    def load_dir_recs(root: Path):
        for fp in sorted(root.rglob("*.json")):
            try:
                yield fp, json.loads(fp.read_text())
            except Exception:
                continue

    if args.calibrate:
        scorer = _load_real_scorer(checkpoint)
        # benign-drop rate per threshold on the dev pool (design-before-results:
        # objective fixed in config — choose highest T with benign-drop <= 0.10)
        ents = []
        for fp, rec in load_dir_recs(DEV_DIR):
            passages = [str(p.get("passage_text", "")) for p in (rec.get("passages") or [])] \
                or [str(rec.get("evidence", ""))]
            chunks = chunk_premise(passages)
            for field in PATIENT_FIELDS:
                src = rec.get(field) or (rec.get("parsed") or {}).get(field) or ""
                if str(src).strip().lower() in SKIP_SENTENCES:
                    continue
                for sent in split_sentences(str(src)):
                    ents.append(sentence_entailment(scorer, chunks, sent))
        ents = sorted(ents)
        n = len(ents)
        rows, chosen = [], None
        for T in [x / 100 for x in range(50, 100, 5)]:
            drop = sum(1 for e in ents if e < T) / n if n else 0.0
            rows.append({"threshold": T, "benign_drop_rate": round(drop, 4), "n_dev_sentences": n})
            if drop <= 0.10:
                chosen = T  # highest T satisfying the ceiling (loop ascends)
        # highest T with drop<=0.10: re-scan descending
        chosen = max((r["threshold"] for r in rows if r["benign_drop_rate"] <= 0.10),
                     default=0.5)
        with (NLI_OUT / "calibration.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["threshold", "benign_drop_rate", "n_dev_sentences"])
            w.writeheader(); w.writerows(rows)
        cfg["nli"]["calibration"]["chosen_threshold"] = float(chosen)
        (REPO / "configs" / "phase5_baselines.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
        print(f"[calibrate] n_dev_sentences={n}; chosen_threshold={chosen}")
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
                p = REPO / "outputs/phase4/reviewer_fixes" / f"semantic_leak_judge_{j}.jsonl"
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
                out = redact_record(rec, scorer, T)
                key = f"{rec.get('case_id')}|{rec.get('backbone')}"
                rows.append({
                    "path": str(fp.relative_to(REPO)),
                    "case_id": rec.get("case_id"), "backbone": rec.get("backbone"),
                    "baseline_id": rec.get("baseline_id", "V-final"),
                    "n_dropped": out["nli_n_dropped"],
                    "is_consensus_flagged": key in consensus,
                })
                # persist redacted copy alongside
                rfp = fp.with_suffix(".nli_redacted.json")
                rfp.write_text(json.dumps(out, indent=2, default=str))
        import pandas as pd
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
