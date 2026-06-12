#!/usr/bin/env python3
"""
Phase 5 Task 3 — baseline generation harness (B1-B4) on the n=114 benchmark.

Baseline configs are FROZEN in configs/phase5_baselines.yaml (pre-registered,
committed before any run). Prompt scaffold, JSON schema, parser, and model
loading are copied verbatim from scripts_phase4/run_backbone_ablation_n100.py
(the phase-4 open-weight harness) so that V-final and baselines differ ONLY
in evidence construction and guard:

  B1_multi_rag          top-K passages across trials, no guard
  B2_prompt_guard       same evidence + frozen anti-mixing instruction
  B3_citation_enforced  same evidence; fields must cite passage numbers;
                        uncited narrative fields -> fallback string
  B4_top1               passages of the rank-1 passage's trial only; no 5-gate

Reference trial (leak detectors evaluate "other trial" against this):
  trial owning the rank-1 passage of the case's reranked pool (frozen).

Output: outputs/phase5/baseline_gens/<baseline_id>/<backbone_slug>/<case_id>.json
(backbone_gens schema + baseline_id, reference_trial, evidence_trials).

Run (server):
  conda activate avatar_trial
  python scripts_phase5/run_baselines.py --max-cases 3 --baselines B1_multi_rag   # smoke
  python scripts_phase5/run_baselines.py                                          # full sweep
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
except Exception:
    print("ERROR: need torch + transformers.", file=sys.stderr)
    raise

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "configs" / "phase5_baselines.yaml").read_text())
POOL_DIR = ROOT / "outputs" / "phase5" / "evidence_pools"
OUT_ROOT = ROOT / "outputs" / "phase5" / "baseline_gens"
FALLBACK = "not clearly stated in the retrieved evidence"

NCT_REGEX = re.compile(r"NCT\d{8}", re.IGNORECASE)

# --- copied verbatim from scripts_phase4/run_backbone_ablation_n100.py ---
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

CITATION_ADDENDUM = """
Additional REQUIRED key for every narrative field:
also return "support_passages": {{"patient_facing_answer": [passage numbers], "reasoning_summary": [passage numbers]}}.
Every claim must be supported by the cited passage numbers. If you cannot cite
a passage for a field, write exactly "{fallback}" in that field.
"""


def parse_json_best_effort(text: str) -> tuple[dict, bool]:
    s, e = text.find("{"), text.rfind("}")
    if s == -1 or e == -1 or e <= s:
        return {}, False
    try:
        return json.loads(text[s:e + 1]), True
    except Exception:
        return {}, False


def load_backbone(name: str, device: str):
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(
        name,
        dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )
    m.eval()
    return tok, m


def build_evidence(passages_df: pd.DataFrame) -> str:
    blocks = []
    for _, r in passages_df.iterrows():
        blocks.append(
            f"[Passage {int(r.get('passage_rank', 1))} "
            f"| doc={r.get('doc_id', '')} "
            f"| section={r.get('section', 'unknown')}]\n"
            f"{str(r.get('passage_text', ''))}"
        )
    return "\n\n".join(blocks)
# --- end copied block ---


def evidence_for(baseline_id: str, pool: pd.DataFrame, k: int) -> tuple[pd.DataFrame, str]:
    """Returns (passages_df, reference_trial)."""
    pool = pool.sort_values("passage_rank").reset_index(drop=True)
    ref_trial = str(pool.iloc[0]["doc_id"])
    if baseline_id == "B4_top1":
        ps = pool[pool["doc_id"].astype(str) == ref_trial].head(k)
    else:  # B1-B3: multi-trial, rank order preserved
        ps = pool.head(k)
    ps = ps.reset_index(drop=True)
    ps["passage_rank"] = range(1, len(ps) + 1)
    return ps, ref_trial


def build_user_msg(baseline_id: str, question: str, evidence: str) -> str:
    msg = USER_TEMPLATE.format(question=question, evidence=evidence)
    if baseline_id == "B2_prompt_guard":
        instr = CFG["baselines"]["B2_prompt_guard"]["extra_instruction"].strip()
        msg = instr + "\n\n" + msg
    elif baseline_id == "B3_citation_enforced":
        msg = msg + CITATION_ADDENDUM.format(fallback=FALLBACK)
    return msg


def postprocess_b3(parsed: dict, n_passages: int) -> dict:
    """B3 guard: drop narrative fields without valid passage citations."""
    cited = parsed.get("support_passages") or {}
    valid = lambda v: (isinstance(v, list) and len(v) > 0
                       and all(isinstance(x, (int, float)) and 1 <= int(x) <= n_passages
                               for x in v))
    out = dict(parsed)
    for field in ("patient_facing_answer", "reasoning_summary"):
        if not valid(cited.get(field)):
            out[field] = FALLBACK
    return out


def run_one(tok, model, device, system: str, user: str, max_new: int):
    msgs = [{"role": "system", "content": system},
            {"role": "user", "content": user}]
    try:
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    except Exception:
        prompt = system + "\n\n" + user
    inp = tok(prompt, return_tensors="pt", truncation=True, max_length=8192)
    if device == "cuda":
        inp = {k: v.to(model.device) for k, v in inp.items()}
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.pad_token_id)
    gen = tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)
    dt = time.time() - t0
    parsed, ok = parse_json_best_effort(gen)
    return gen, parsed, ok, dt


def count_narrow_leak(text: str, reference_trial: str) -> int:
    hits = {m.group(0).upper() for m in NCT_REGEX.finditer(text)}
    return len(hits - {reference_trial.upper()})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baselines", nargs="+", default=list(CFG["baselines"].keys()))
    ap.add_argument("--backbones", nargs="+", default=CFG["backbones"])
    ap.add_argument("--max-cases", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--resume", action="store_true", default=True)
    args = ap.parse_args()

    with open(ROOT / "data/processed/onboarding_eval_cases_expanded.json") as f:
        cases = {str(c["case_id"]): c for c in json.load(f)}
    pool_files = sorted(POOL_DIR.glob("*.json")) or sorted(POOL_DIR.glob("*.csv"))
    cids = [p.stem for p in pool_files]
    if args.max_cases:
        cids = cids[: args.max_cases]
    print(f"[info] {len(cids)} cases x {len(args.baselines)} baselines x {len(args.backbones)} backbones")

    k = int(CFG["evidence"]["multi_trial_k"])
    max_new = int(CFG["decoding"]["max_new_tokens"])

    for bb in args.backbones:
        slug = bb.replace("/", "__")
        print(f"\n=== backbone: {bb} ===")
        tok, model = load_backbone(bb, args.device)
        for baseline_id in args.baselines:
            out_dir = OUT_ROOT / baseline_id / slug
            out_dir.mkdir(parents=True, exist_ok=True)
            n_done = 0
            for cid in cids:
                out_fp = out_dir / f"{cid}.json"
                if args.resume and out_fp.exists():
                    n_done += 1
                    continue
                pool = pd.read_csv(POOL_DIR / f"{cid}.csv")
                if len(pool) == 0:
                    continue
                ps, ref_trial = evidence_for(baseline_id, pool, k)
                evidence = build_evidence(ps)
                user = build_user_msg(baseline_id, str(cases[cid]["question"]), evidence)
                gen, parsed, ok, dt = run_one(tok, model, args.device,
                                              SYSTEM_PROMPT, user, max_new)
                if baseline_id == "B3_citation_enforced" and ok:
                    parsed = postprocess_b3(parsed, len(ps))
                decision = (str(parsed.get("decision", "cannot_determine"))
                            .lower().replace(" ", "_")) if ok else "parse_fail"
                rec = {
                    "case_id": cid,
                    "backbone": bb,
                    "baseline_id": baseline_id,
                    "source": cases[cid].get("source", ""),
                    "category": cases[cid].get("category", ""),
                    "gold_nct": cases[cid].get("gold_nct") or "",
                    "selected_doc": ref_trial,          # reference trial for detectors
                    "reference_trial": ref_trial,
                    "evidence_trials": sorted(ps["doc_id"].astype(str).unique().tolist()),
                    "question": str(cases[cid]["question"]),
                    "evidence": evidence,
                    "passages": ps.to_dict("records"),
                    "raw_generation": gen,
                    "parsed": parsed,
                    "parse_ok": bool(ok),
                    "decision": decision,
                    "patient_facing_answer": str(parsed.get("patient_facing_answer", "")) if ok else "",
                    "missing_patient_facts": parsed.get("missing_patient_facts", []) or [],
                    "unresolved_study_requirements": parsed.get("unresolved_study_requirements", []) or [],
                    "narrow_leak_n": count_narrow_leak(gen, ref_trial),
                    "latency_sec": round(dt, 3),
                }
                out_fp.write_text(json.dumps(rec, indent=2, default=str))
                n_done += 1
                print(f"[{baseline_id}|{slug}] {n_done}/{len(cids)} {cid} "
                      f"parse_ok={ok} leak_n={rec['narrow_leak_n']} {dt:.1f}s", flush=True)
        del tok, model
        if args.device == "cuda":
            torch.cuda.empty_cache()
    print("[done]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
