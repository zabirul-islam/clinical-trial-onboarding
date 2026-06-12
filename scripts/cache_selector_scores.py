"""
Phase 2.1 — Cache trial-first selector signals for all 150 expanded cases.

For each case, runs BM25 → cross-encoder → trial-score aggregation and saves
the 4 selector thresholds plus selected_doc + gold_nct (if any). Expensive
(~10-15s per case = ~25-35 min for 150), so we cache once and sweep offline.

Signals cached:
  dominance_ratio  = top_trial_score / second_trial_score       (ρ)
  trial_score_share= top_trial_score / sum(all_trial_scores)    (τ)
  raw_max_cross    = top trial's max raw cross-encoder score     (μ)
  best_rank        = top trial's best passage rank in CE top-K   (κ)
  selected_doc     = NCT id of top-scored trial
  gold_nct         = ground-truth NCT (TREC cases only; None otherwise)
  generic_question = heuristic flag for under-specified questions

Output: outputs/tables/selector_signals_cache.csv

Run:
  cd ~/Desktop/islamm11/avatar_trial_onboarding
  python scripts/cache_selector_scores.py --repo-root . \\
      --cases data/processed/onboarding_eval_cases_expanded.json \\
      --top-docs 20 --top-passages 8 --device cuda
"""
from __future__ import annotations
import argparse, json, re, subprocess, sys, shutil, time
from pathlib import Path

import pandas as pd
from tqdm import tqdm


def tokenize(text: str):
    return re.findall(r"[a-z0-9]+", str(text).lower())


GENERIC_PHRASES = [
    "this study", "joined this study", "join this study", "if i joined this study",
]
ANCHOR_TERMS = [
    "nct", "osteoporosis", "vertebral", "fracture", "lynch",
    "brain injury", "breast", "hepatitis", "melanoma",
]


def is_generic(q: str) -> bool:
    q = q.lower()
    return any(p in q for p in GENERIC_PHRASES) and not any(a in q for a in ANCHOR_TERMS)


def run_retrieval(question: str, root: Path, top_docs: int, top_passages: int):
    """Invoke existing retrieve_grounding_evidence.py + score_trials_from_passages.py."""
    retrieve = root / "scripts" / "retrieve_grounding_evidence.py"
    score = root / "scripts" / "score_trials_from_passages.py"
    subprocess.run(
        [sys.executable, str(retrieve),
         "--question", question,
         "--top_docs", str(top_docs),
         "--top_passages", str(top_passages)],
        check=True, cwd=root,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    subprocess.run(
        [sys.executable, str(score)],
        check=True, cwd=root,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def extract_signals(root: Path) -> dict:
    out = root / "outputs" / "tables"
    trial_df = pd.read_csv(out / "trial_level_scores.csv")
    if len(trial_df) == 0:
        return {
            "dominance_ratio": 0.0, "trial_score_share": 0.0,
            "raw_max_cross": 0.0, "best_rank": 999,
            "selected_doc": "", "n_trials_in_pool": 0,
        }
    top = trial_df.iloc[0]
    second = float(trial_df.iloc[1]["trial_score"]) if len(trial_df) > 1 else 1e-6
    total = float(trial_df["trial_score"].sum())
    top_s = float(top["trial_score"])
    return {
        "dominance_ratio":   top_s / max(second, 1e-6),
        "trial_score_share": top_s / max(total, 1e-6),
        "raw_max_cross":     float(top.get("raw_max_cross", 0.0)),
        "best_rank":         int(top.get("best_rank", 999)),
        "selected_doc":      str(top["doc_id"]),
        "n_trials_in_pool":  int(len(trial_df)),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", type=Path, default=Path("."))
    p.add_argument("--cases", type=Path,
                   default=Path("data/processed/onboarding_eval_cases_expanded.json"))
    p.add_argument("--top-docs", type=int, default=20)
    p.add_argument("--top-passages", type=int, default=8)
    p.add_argument("--out", type=Path,
                   default=Path("outputs/tables/selector_signals_cache.csv"))
    p.add_argument("--resume", action="store_true",
                   help="Skip cases already in out CSV (by case_id)")
    args = p.parse_args()

    root = args.repo_root.resolve()
    cases_path = args.cases if args.cases.is_absolute() else root / args.cases
    out_path = args.out if args.out.is_absolute() else root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(cases_path) as f:
        cases = json.load(f)
    print(f"[load] {len(cases)} cases from {cases_path}")

    done = set()
    if args.resume and out_path.exists():
        done_df = pd.read_csv(out_path)
        done = set(done_df["case_id"].astype(str))
        print(f"[resume] {len(done)} already cached; will skip")

    rows = []
    t0 = time.time()
    for c in tqdm(cases, desc="cache signals"):
        cid = str(c["case_id"])
        if cid in done:
            continue
        q = str(c["question"])
        try:
            run_retrieval(q, root, args.top_docs, args.top_passages)
            sig = extract_signals(root)
            sig.update({
                "case_id":  cid,
                "source":   c.get("source", ""),
                "category": c.get("category", ""),
                "gold_nct": c.get("gold_nct") or "",
                "trec_query_id": c.get("trec_query_id"),
                "generic_question": is_generic(q),
                "selected_matches_gold": (sig["selected_doc"] == (c.get("gold_nct") or "NONE")),
                "question_preview": q[:160].replace("\n", " "),
            })
            rows.append(sig)
        except Exception as e:
            print(f"[warn] {cid} failed: {e}", file=sys.stderr)
            rows.append({
                "case_id": cid, "source": c.get("source", ""),
                "category": c.get("category", ""),
                "gold_nct": c.get("gold_nct") or "",
                "trec_query_id": c.get("trec_query_id"),
                "error": str(e)[:200],
            })
        # Flush every 10
        if len(rows) % 10 == 0:
            df_so_far = pd.DataFrame(rows)
            if args.resume and out_path.exists():
                prev = pd.read_csv(out_path)
                df_so_far = pd.concat([prev, df_so_far], ignore_index=True)
            df_so_far.to_csv(out_path, index=False)

    df = pd.DataFrame(rows)
    if args.resume and out_path.exists():
        prev = pd.read_csv(out_path)
        df = pd.concat([prev, df], ignore_index=True)
    df.to_csv(out_path, index=False)
    dt = time.time() - t0
    print(f"[done] wrote {len(df):,} rows → {out_path}  ({dt/60:.1f} min)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
