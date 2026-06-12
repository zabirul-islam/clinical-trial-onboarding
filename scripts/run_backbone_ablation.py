"""
Phase 2.3 — Backbone ablation on a 30-case subset.

Runs the full trial-first eligibility pipeline (retrieval already cached via
Phase 2.1; here we only re-run the LLM stage) for 4 backbones:
  - Qwen/Qwen2.5-3B-Instruct   (current)
  - Qwen/Qwen2.5-7B-Instruct
  - meta-llama/Meta-Llama-3.1-8B-Instruct
  - mistralai/Mistral-7B-Instruct-v0.3

For each (case × backbone) we record:
  decision                              (likely_match | possible_match_insuff | unlikely_match | cannot_determine)
  parse_ok
  latency_sec
  contains_nct_id_other_than_selected   (cross-trial leakage flag)
  patient_facing_answer_length_chars
  missing_patient_facts_count
  unresolved_study_requirements_count

The script reuses cached trial-first context from selector_signals_cache.csv
(skips cases the selector abstains on at default thresholds).

Outputs:
  outputs/tables/backbone_ablation_raw.csv       (per-case-per-backbone)
  outputs/tables/backbone_ablation_summary.csv   (per-backbone aggregates)

Run:
  python scripts/run_backbone_ablation.py --repo-root . \\
      --n-cases 30 --device cuda \\
      --backbones Qwen/Qwen2.5-3B-Instruct Qwen/Qwen2.5-7B-Instruct

Memory note: 7B fp16 ~14 GB, 8B ~16 GB. GB10 96 GB → fits one at a time.
"""
from __future__ import annotations
import json
import argparse, json, re, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
except Exception:
    print("ERROR: need torch + transformers.", file=sys.stderr); raise


NCT_REGEX = re.compile(r"NCT\d{8}", re.IGNORECASE)

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
        dtype=torch.float16 if device == "cuda" else torch.float32,
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


def count_cross_trial_leakage(text: str, selected_doc: str) -> int:
    hits = {m.group(0).upper() for m in NCT_REGEX.finditer(text)}
    return len(hits - {selected_doc.upper()})


def run_one(tok, model, device, question: str, evidence: str, max_new: int) -> tuple[str, dict, bool, float]:
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": USER_TEMPLATE.format(question=question, evidence=evidence)},
    ]
    if tok.chat_template is not None:
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    else:
        prompt = SYSTEM_PROMPT + "\n\n" + msgs[1]["content"]
    inp = tok(prompt, return_tensors="pt", truncation=True, max_length=8192)
    if device == "cuda":
        inp = {k: v.to(model.device) for k, v in inp.items()}
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    dt = time.time() - t0
    gen = tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    parsed, ok = parse_json_best_effort(gen)
    return gen, parsed, ok, dt


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", type=Path, default=Path("."))
    p.add_argument("--cache", type=str, default="outputs/tables/selector_signals_cache.csv")
    p.add_argument("--cases", type=str, default="data/processed/onboarding_eval_cases_expanded.json")
    p.add_argument("--passages", type=str, default="data/processed/trial_evidence_passages.parquet")
    p.add_argument("--n-cases", type=int, default=30,
                   help="Pick first N accepted-by-default cases (reproducible)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-new-tokens", type=int, default=340)
    p.add_argument("--passages-per-trial", type=int, default=6)
    p.add_argument("--backbones", nargs="+", default=[
        "Qwen/Qwen2.5-3B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct",
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "mistralai/Mistral-7B-Instruct-v0.3",
    ])
    args = p.parse_args()

    root = args.repo_root.resolve()
    cache = pd.read_csv(root / args.cache)
    with open(root / args.cases) as f:
        cases = {str(c["case_id"]): c for c in json.load(f)}
    passages_all = pd.read_parquet(root / args.passages)

    # Default thresholds (from run_onboarding_pipeline_trial_first.py)
    DEF = {"rho": 1.35, "tau": 0.28, "mu": -4.5, "kap": 2}
    accept = (
        (cache["dominance_ratio"]   >= DEF["rho"]) &
        (cache["trial_score_share"] >= DEF["tau"]) &
        (cache["raw_max_cross"]     >= DEF["mu"])  &
        (cache["best_rank"]         <= DEF["kap"]) &
        (~cache["generic_question"].fillna(False).astype(bool))
    )
    accepted = cache[accept].reset_index(drop=True)
    print(f"[info] {len(accepted)} of {len(cache)} cases accepted by defaults; sampling {args.n_cases}")
    pick = accepted.sample(n=min(args.n_cases, len(accepted)),
                           random_state=args.seed).reset_index(drop=True)

    rows = []
    for bb in args.backbones:
        print(f"\n=== backbone: {bb} ===")
        try:
            tok, model = load_backbone(bb, args.device)
        except Exception as e:
            print(f"[skip] {bb}: {e}")
            continue
        for _, r in tqdm(pick.iterrows(), total=len(pick), desc=bb.split('/')[-1]):
            cid = str(r["case_id"])
            case = cases.get(cid, {})
            q = str(case.get("question", ""))
            sel = str(r["selected_doc"])
            ps = passages_all[passages_all["doc_id"].astype(str) == sel] \
                    .sort_values("passage_id").head(args.passages_per_trial)
            if len(ps) == 0:
                continue
            # assign passage_rank 1..K
            ps = ps.reset_index(drop=True)
            ps["passage_rank"] = np.arange(1, len(ps) + 1)
            evidence = build_evidence(ps)
            gen, parsed, ok, dt = run_one(tok, model, args.device, q, evidence, args.max_new_tokens)
            decision = str(parsed.get("decision", "cannot_determine")).lower().replace(" ", "_") if ok else "parse_fail"
            pfa = str(parsed.get("patient_facing_answer", "")) if ok else ""
            leak = count_cross_trial_leakage(gen, sel)
            rows.append({
                "case_id":     cid,
                "source":      case.get("source", ""),
                "category":    case.get("category", ""),
                "gold_nct":    case.get("gold_nct") or "",
                "selected_doc": sel,
                "backbone":    bb,
                "decision":    decision,
                "parse_ok":    ok,
                "latency_sec": round(dt, 3),
                "answer_chars": len(pfa),
                "missing_facts_n":      len(parsed.get("missing_patient_facts", []) or []),
                "unresolved_reqs_n":    len(parsed.get("unresolved_study_requirements", []) or []),
                "cross_trial_leak_n":   int(leak),
                "gen_snippet": gen[:240].replace("\n", " "),
            })

            # --- Phase 3 full dump for LLM judge ---
            slug = bb.replace("/", "__")
            gen_dir = root / "outputs" / "backbone_gens" / slug
            gen_dir.mkdir(parents=True, exist_ok=True)
            (gen_dir / f"{cid}.json").write_text(json.dumps({
                "case_id": cid,
                "backbone": bb,
                "source":   case.get("source", ""),
                "category": case.get("category", ""),
                "gold_nct": case.get("gold_nct") or "",
                "selected_doc": sel,
                "question": q,
                "evidence": evidence,
                "passages": ps.to_dict("records"),
                "raw_generation": gen,
                "parsed": parsed,
                "parse_ok": bool(ok),
                "decision": decision,
                "patient_facing_answer": pfa,
                "missing_patient_facts":           parsed.get("missing_patient_facts", []) or [],
                "unresolved_study_requirements":   parsed.get("unresolved_study_requirements", []) or [],
                "cross_trial_leak_n": int(leak),
                "latency_sec":        round(dt, 3),
            }, indent=2, default=str))


        # Free VRAM between backbones
        del tok, model
        if args.device == "cuda":
            torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    out_dir = root / "outputs/tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "backbone_ablation_raw.csv", index=False)

    if len(df) == 0:
        print("[warn] no rows generated"); return 0

    agg = df.groupby("backbone").agg(
        n            = ("case_id", "count"),
        parse_ok_rate= ("parse_ok", "mean"),
        mean_latency = ("latency_sec", "mean"),
        median_latency= ("latency_sec", "median"),
        mean_answer_chars = ("answer_chars", "mean"),
        leak_rate    = ("cross_trial_leak_n", lambda s: float((s > 0).mean())),
        commit_rate  = ("decision", lambda s: float(s.isin(["likely_match", "unlikely_match"]).mean())),
        abstain_rate = ("decision", lambda s: float(s.isin(["possible_match_insufficient_evidence", "cannot_determine"]).mean())),
        parse_fail_rate = ("decision", lambda s: float((s == "parse_fail").mean())),
    ).reset_index()
    agg.to_csv(out_dir / "backbone_ablation_summary.csv", index=False)
    print(agg.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
