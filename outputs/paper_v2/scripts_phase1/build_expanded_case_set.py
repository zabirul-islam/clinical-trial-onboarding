"""
Phase 1.1 — Expand onboarding eval case set from n=15 → n=150.

Inputs (relative to repo root):
  data/processed/onboarding_eval_cases.json            (15 hand-crafted)
  data/processed/trec_ct2021_topics_full.csv           (75 topics)
  data/processed/trec_ct2022_topics_full.csv           (50 topics)
  data/processed/trec_ct2021_qrels_full.csv            (for gold NCT lookup)

Output:
  data/processed/onboarding_eval_cases_expanded.json   (n=150)

Composition:
  -  15 existing hand-crafted (ids: case_01..case_15)
  -  50 TREC CT 2021 topics, intent=possible_match_insufficient_evidence
  -  50 TREC CT 2022 topics, intent=possible_match_insufficient_evidence
  -  35 paraphrased-intent cases (TREC topic base × 5 intents × 7 bases)

Run:
  python scripts/build_expanded_case_set.py \
      --repo-root . \
      --seed 42 \
      --out data/processed/onboarding_eval_cases_expanded.json
"""
from __future__ import annotations
import argparse, json, random, sys
from pathlib import Path

import pandas as pd


INTENT_PARAPHRASES = {
    "explanation_request": "Can you explain what this study is really about in plain terms?",
    "participation_request": "What would I actually need to do if I joined this study?",
    "risk_request": "What are the risks or side-effects I should be aware of in this study?",
    "cannot_determine": "I have similar symptoms but I'm not sure if I would qualify. What should I check?",
    "consent_understanding": "Before I agree to anything, what is the study team going to confirm with me?",
}


def topic_text_to_query(topic_text: str) -> str:
    """Convert TREC topic narrative to first-person onboarding query."""
    t = topic_text.strip()
    # Simple heuristic: strip numeric IDs and collapse whitespace
    t = " ".join(t.split())
    # Prepend a patient-style framing
    return f"I am considering a clinical trial. Here is my situation: {t} Am I eligible?"


def load_existing_cases(p: Path) -> list[dict]:
    with open(p) as f:
        return json.load(f)


def build_trec_cases(
    topics_csv: Path,
    qrels_csv: Path | None,
    n: int,
    source: str,
    rng: random.Random,
) -> list[dict]:
    topics = pd.read_csv(topics_csv)
    # Column names differ: 2021 has (query_id,text); 2022 has (query_id,text,source_dataset,task_type)
    topics = topics[["query_id", "text"]].copy()
    topics["text"] = topics["text"].astype(str).str.strip()
    topics = topics[topics["text"].str.len() > 50].reset_index(drop=True)
    if len(topics) > n:
        topics = topics.sample(n=n, random_state=rng.randint(0, 10**6)).reset_index(drop=True)

    gold_lookup: dict[int, str] = {}
    if qrels_csv is not None and qrels_csv.exists():
        q = pd.read_csv(qrels_csv)
        # Pick highest-relevance NCT per query
        q = q.sort_values(["query_id", "relevance"], ascending=[True, False])
        first = q.drop_duplicates("query_id", keep="first")
        gold_lookup = dict(zip(first["query_id"].astype(int), first["doc_id"].astype(str)))

    cases = []
    for i, row in topics.iterrows():
        qid = int(row["query_id"])
        cases.append(
            {
                "case_id": f"case_{source}_{qid:03d}",
                "source": source,
                "trec_query_id": qid,
                "gold_nct": gold_lookup.get(qid),
                "category": "possible_match_insufficient_evidence",
                "question": topic_text_to_query(row["text"]),
                "notes": f"From {source} topic {qid}.",
            }
        )
    return cases


def build_paraphrased_intent_cases(
    topics_2021: pd.DataFrame,
    topics_2022: pd.DataFrame,
    rng: random.Random,
) -> list[dict]:
    """7 bases × 5 intents = 35 paraphrased-intent cases."""
    combo = pd.concat(
        [topics_2021[["query_id", "text"]].assign(src="trec2021"),
         topics_2022[["query_id", "text"]].assign(src="trec2022")],
        ignore_index=True,
    )
    combo = combo[combo["text"].astype(str).str.len() > 50].reset_index(drop=True)
    bases = combo.sample(n=7, random_state=rng.randint(0, 10**6)).reset_index(drop=True)

    cases = []
    k = 0
    for _, base in bases.iterrows():
        base_query = topic_text_to_query(str(base["text"]))
        for intent, paraphrase in INTENT_PARAPHRASES.items():
            k += 1
            cases.append(
                {
                    "case_id": f"case_paraphrase_{k:03d}",
                    "source": f"paraphrase_{base['src']}",
                    "trec_query_id": int(base["query_id"]),
                    "gold_nct": None,
                    "category": intent,
                    "question": f"{base_query.split('Am I eligible?')[0].strip()} {paraphrase}",
                    "notes": f"Paraphrased intent: {intent}. Base: {base['src']} topic {base['query_id']}.",
                }
            )
    return cases


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", type=Path, default=Path("."))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-2021", type=int, default=50)
    p.add_argument("--n-2022", type=int, default=50)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    rng = random.Random(args.seed)

    root = args.repo_root.resolve()
    existing = load_existing_cases(root / "data/processed/onboarding_eval_cases.json")
    for c in existing:
        c.setdefault("source", "handcrafted")
        c.setdefault("gold_nct", None)
        c.setdefault("trec_query_id", None)

    trec2021 = build_trec_cases(
        topics_csv=root / "data/processed/trec_ct2021_topics_full.csv",
        qrels_csv=root / "data/processed/trec_ct2021_qrels_full.csv",
        n=args.n_2021,
        source="trec2021",
        rng=rng,
    )
    trec2022 = build_trec_cases(
        topics_csv=root / "data/processed/trec_ct2022_topics_full.csv",
        qrels_csv=None,  # 2022 qrels may not be in repo
        n=args.n_2022,
        source="trec2022",
        rng=rng,
    )

    # For paraphrased intents, reuse the already-loaded topic dataframes
    t2021_full = pd.read_csv(root / "data/processed/trec_ct2021_topics_full.csv")
    t2022_full = pd.read_csv(root / "data/processed/trec_ct2022_topics_full.csv")
    paraphrases = build_paraphrased_intent_cases(t2021_full, t2022_full, rng)

    all_cases = existing + trec2021 + trec2022 + paraphrases
    out = args.out if args.out.is_absolute() else root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(all_cases, f, indent=2)

    print(f"wrote {len(all_cases)} cases to {out}")
    by_src: dict[str, int] = {}
    for c in all_cases:
        by_src[c["source"]] = by_src.get(c["source"], 0) + 1
    print("breakdown:", by_src)
    return 0


if __name__ == "__main__":
    sys.exit(main())
