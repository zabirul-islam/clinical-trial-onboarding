"""
Phase 1.4 — NLI4PR eligibility-alignment eval. V3 logic INLINED (no package import).

Why this matters: TREC CT 2022 has no qrels in this repo. NLI4PR (n=7007)
provides labeled patient-voiced eligibility entailment ({entailment,
contradiction, neutral}) as a downstream ground-truth signal for the
eligibility module.

What this does:
  For each NLI4PR instance, run the SAME V3 guarded-eligibility prompt
  used in scripts/generate_guarded_eligibility_v3.py, but with the trial_text
  from NLI4PR used directly as the evidence passage. Map the V3 decision to
  {match, mismatch, undetermined} and compare to the gold label.

Decision mapping (V3 actual categories):
  likely_match                              -> match
  possible_match_insufficient_evidence      -> match
  unlikely_match                            -> mismatch
  cannot_determine                          -> undetermined

Gold-label collapse:
  entailment    -> match
  contradiction -> mismatch
  neutral       -> undetermined

Metrics: accuracy, macro-F1 (with per-class), Cohen's kappa, confusion matrix.

Run:
  python scripts/eval_nli4pr_eligibility_alignment.py \\
      --repo-root . \\
      --split test \\
      --n-sample 500 \\
      --backbone Qwen/Qwen2.5-3B-Instruct \\
      --max-new-tokens 340

Outputs:
  outputs/tables/nli4pr_eligibility_alignment.csv
  outputs/tables/nli4pr_eligibility_alignment_summary.json

GPU memory: ~8 GB fp16 for 3B. 500 instances × ~2 s/inst ≈ 15-20 min on GB10.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
except Exception:
    print("ERROR: need torch + transformers. pip install transformers torch",
          file=sys.stderr)
    raise


# ---------------------------------------------------------------------------
# V3 prompt — copied VERBATIM from scripts/generate_guarded_eligibility_v3.py
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a clinical trial onboarding assistant.
You must answer ONLY from the provided evidence passages.

Safety rules:
1. Never say definitively eligible or definitively enrolled.
2. Use uncertainty-aware categories when exclusions, site checks, imaging requirements, or other facts remain unresolved.
3. Distinguish between:
   - supported patient facts
   - missing patient facts
   - unresolved study-side requirements
4. Do not give diagnosis or treatment advice.
5. Use plain language.
6. Output valid JSON only.
"""

USER_TEMPLATE = """Patient question:
{question}

Evidence passages:
{evidence}

Return JSON with exactly these keys:
{{
  "decision": "likely_match | possible_match_insufficient_evidence | unlikely_match | cannot_determine",
  "patient_facing_answer": "...",
  "supported_patient_facts": ["...", "..."],
  "missing_patient_facts": ["...", "..."],
  "unresolved_study_requirements": ["...", "..."],
  "reasoning_summary": "...",
  "safety_note": "..."
}}

Decision policy:
- likely_match: strong match to main inclusion profile, but final review is still needed.
- possible_match_insufficient_evidence: some important criteria match, but missing facts or unresolved requirements remain.
- unlikely_match: important criteria appear not to match.
- cannot_determine: too little relevant evidence to judge.
"""


# Decision → class maps
V3_CAT_TO_CLASS = {
    "likely_match":                         "match",
    "possible_match_insufficient_evidence": "match",
    "unlikely_match":                       "mismatch",
    "cannot_determine":                     "undetermined",
}
NLI_TO_CLASS = {
    "entailment":    "match",
    "contradiction": "mismatch",
    "neutral":       "undetermined",
}
LABELS = ["match", "mismatch", "undetermined"]


def parse_decision(gen_text: str) -> tuple[str, dict]:
    """Extract V3 decision string + full parsed JSON dict (best effort)."""
    start = gen_text.find("{")
    end = gen_text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return "cannot_determine", {"_parse_error": "no_json_braces"}
    candidate = gen_text[start:end + 1]
    try:
        parsed = json.loads(candidate)
    except Exception as e:
        return "cannot_determine", {"_parse_error": f"json_decode: {e}"}
    dec = str(parsed.get("decision", "cannot_determine")).strip()
    # Canonicalise: V3 may emit "likely match" or casing variants
    dec = dec.lower().replace(" ", "_")
    if dec not in V3_CAT_TO_CLASS:
        # Some models emit the whole option list. Grab the first known token.
        for k in V3_CAT_TO_CLASS:
            if k in dec:
                dec = k; break
        else:
            dec = "cannot_determine"
    return dec, parsed


def confusion(y_true, y_pred, labels):
    idx = {l: i for i, l in enumerate(labels)}
    M = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(y_true, y_pred):
        if t in idx and p in idx:
            M[idx[t], idx[p]] += 1
    return M


def cohens_kappa(y_true, y_pred, labels):
    M = confusion(y_true, y_pred, labels)
    N = M.sum()
    if N == 0:
        return float("nan")
    po = np.trace(M) / N
    pe = ((M.sum(0) * M.sum(1)) / (N * N)).sum()
    return float("nan") if pe == 1 else float((po - pe) / (1 - pe))


def macro_f1(y_true, y_pred, labels):
    M = confusion(y_true, y_pred, labels)
    f1s = []
    for i, _ in enumerate(labels):
        tp = M[i, i]
        fp = M[:, i].sum() - tp
        fn = M[i, :].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec  = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        f1s.append(f1)
    return float(np.mean(f1s)), {l: f for l, f in zip(labels, f1s)}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", type=Path, default=Path("."))
    p.add_argument("--split", default="test")
    p.add_argument("--n-sample", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--backbone", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--max-new-tokens", type=int, default=340)
    p.add_argument("--max-evidence-chars", type=int, default=6000,
                   help="Truncate trial_text to N chars before prompting (prevents OOM on long trials)")
    p.add_argument("--save-every", type=int, default=50,
                   help="Flush partial results every N instances")
    args = p.parse_args()

    root = args.repo_root.resolve()
    nli_csv = root / "data/processed/benchmark_nli4pr.csv"
    print(f"loading NLI4PR: {nli_csv}")
    df = pd.read_csv(nli_csv)
    df.columns = [c.strip().lower() for c in df.columns]

    # split filter
    if "split" in df.columns:
        df = df[df["split"] == args.split].reset_index(drop=True)
    print(f"[{args.split}] rows = {len(df):,}")

    if len(df) > args.n_sample:
        df = df.sample(n=args.n_sample, random_state=args.seed).reset_index(drop=True)
    print(f"sampled {len(df):,} instances (seed={args.seed})")

    # Pick the three required columns defensively
    cols = {c.lower(): c for c in df.columns}
    def pick(*opts):
        for o in opts:
            if o in cols: return cols[o]
        return None
    iid_col   = pick("instance_id", "id")
    trial_col = pick("trial_id", "nct_id", "doc_id")
    pt_col    = pick("patient_text_plain", "patient_text", "patient", "hypothesis")
    tt_col    = pick("trial_text", "premise", "evidence", "trial")
    lbl_col   = pick("label", "gold_label", "entailment_label")
    assert all([pt_col, tt_col, lbl_col]), f"missing required NLI4PR cols; got {list(df.columns)}"

    # Load backbone
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"loading backbone: {args.backbone}  (device={device})")
    tok = AutoTokenizer.from_pretrained(args.backbone)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.backbone,
        dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )
    model.eval()

    out_dir = root / "outputs/tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "nli4pr_eligibility_alignment.csv"

    rows = []
    for i, r in enumerate(tqdm(df.itertuples(index=False), total=len(df), desc="V3-inline")):
        rec = r._asdict() if hasattr(r, "_asdict") else dict(zip(df.columns, r))
        patient   = str(rec[pt_col])[: args.max_evidence_chars]
        trial     = str(rec[tt_col])[: args.max_evidence_chars]
        gold      = str(rec[lbl_col]).strip().lower()
        iid       = rec.get(iid_col, f"inst_{i}") if iid_col else f"inst_{i}"
        trial_id  = rec.get(trial_col, "") if trial_col else ""

        # V3 evidence is just the trial_text (single passage) — matches NLI4PR structure
        evidence_block = f"[Passage 1 | doc={trial_id} | section=trial_text]\n{trial}"
        user_msg = USER_TEMPLATE.format(question=patient, evidence=evidence_block)
        messages = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_msg}]
        if tok.chat_template is not None:
            prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt = SYSTEM_PROMPT + "\n\n" + user_msg

        inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=8192)
        if device == "cuda":
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            outs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                temperature=None,
                pad_token_id=tok.eos_token_id,
            )
        gen = tok.decode(outs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        dec, parsed = parse_decision(gen)
        pred_class = V3_CAT_TO_CLASS.get(dec, "undetermined")
        gold_class = NLI_TO_CLASS.get(gold, "undetermined")

        rows.append({
            "instance_id":   iid,
            "trial_id":      trial_id,
            "gold_label":    gold,
            "gold_class":    gold_class,
            "pred_decision": dec,
            "pred_class":    pred_class,
            "parse_ok":      "_parse_error" not in parsed,
            "gen_snippet":   gen[:300].replace("\n", " "),
        })

        if (i + 1) % args.save_every == 0:
            pd.DataFrame(rows).to_csv(out_csv, index=False)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_csv, index=False)

    yt, yp = out_df["gold_class"].tolist(), out_df["pred_class"].tolist()
    acc = float((out_df["pred_class"] == out_df["gold_class"]).mean())
    f1m, f1_per = macro_f1(yt, yp, LABELS)
    kappa = cohens_kappa(yt, yp, LABELS)
    M = confusion(yt, yp, LABELS).tolist()

    summary = {
        "n":            int(len(out_df)),
        "accuracy":     acc,
        "macro_f1":     f1m,
        "f1_per_class": f1_per,
        "cohens_kappa": kappa,
        "labels":       LABELS,
        "confusion":    M,
        "backbone":     args.backbone,
        "parse_ok_rate": float(out_df["parse_ok"].mean()),
    }
    with open(out_dir / "nli4pr_eligibility_alignment_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

