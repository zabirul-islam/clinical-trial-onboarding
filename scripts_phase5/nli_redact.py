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

    print("[info] --calibrate / --apply require the real NLI checkpoint on GPU.")
    print("[info] implemented in Task 6 Steps 6.3/6.4 on the DGX server.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
