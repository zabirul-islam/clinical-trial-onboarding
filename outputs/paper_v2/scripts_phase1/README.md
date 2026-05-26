# Phase 1 scripts — drop-in for your Clinical-Trial repo

Your server layout:
```
~/Desktop/islamm11/avatar_trial_onboarding/
├── configs/
├── data/
│   ├── interim/
│   ├── processed/      <-- benchmark CSVs, TREC topics, NLI4PR
│   └── raw/
├── final_system_backup/
├── indices/
├── outputs/
│   └── tables/         <-- per-query retrieval metrics
├── scripts/            <-- put the 4 new scripts here
└── src/
    └── avatar_onboarding/
```

All scripts assume repo root = `~/Desktop/islamm11/avatar_trial_onboarding`.
**Always run from repo root, never from `data/` or `scripts/`.**

## Environment verify

```bash
cd ~/Desktop/islamm11/avatar_trial_onboarding
python -c "import torch, transformers, pandas, numpy; print(torch.__version__, transformers.__version__)"
pip install tqdm  # if missing
```

## 1.1  Expand case set (n=15 → n=150) — ~5 s

```bash
cd ~/Desktop/islamm11/avatar_trial_onboarding
python scripts/build_expanded_case_set.py \
    --repo-root . \
    --seed 42 \
    --out data/processed/onboarding_eval_cases_expanded.json
```

Expected stdout:
```
wrote 150 cases to .../data/processed/onboarding_eval_cases_expanded.json
breakdown: {'handcrafted': 15, 'trec2021': 50, 'trec2022': 50,
            'paraphrase_trec2021': X, 'paraphrase_trec2022': Y}
```
Paste the stdout back so I can verify breakdown.

## 1.2  Bootstrap retrieval CIs — ~20 s

```bash
cd ~/Desktop/islamm11/avatar_trial_onboarding
python scripts/bootstrap_retrieval_cis.py \
    --repo-root . \
    --n-bootstrap 1000 \
    --seed 42 \
    --out outputs/tables/retrieval_bootstrap_cis.csv
```

Produces `outputs/tables/retrieval_bootstrap_cis.csv` and `.tex`. Paste both.

Any `WARN: missing ...` → tell me which file.

## 1.3  BGE-reranker-large rerun — ~15–25 min on GB10

```bash
cd ~/Desktop/islamm11/avatar_trial_onboarding
python scripts/run_bge_reranker_large.py \
    --repo-root . \
    --device cuda \
    --batch-size 8 \
    --max-length 512 \
    --topk 20
```

Produces in `outputs/tables/`:
- `bge_reranker_large_run.csv`
- `bge_reranker_large_per_query_metrics.csv`
- `bge_reranker_large_summary_metrics.csv`

Paste the summary one-row CSV.

## 1.4  NLI4PR eligibility alignment — ~15–20 min on GB10

V3 prompt is inlined. No import from `avatar_onboarding` needed.

```bash
cd ~/Desktop/islamm11/avatar_trial_onboarding
python scripts/eval_nli4pr_eligibility_alignment.py \
    --repo-root . \
    --split test \
    --n-sample 500 \
    --backbone Qwen/Qwen2.5-3B-Instruct \
    --max-new-tokens 340
```

Produces in `outputs/tables/`:
- `nli4pr_eligibility_alignment.csv`  (per-instance predictions)
- `nli4pr_eligibility_alignment_summary.json`  (accuracy, macro-F1, kappa, confusion)

Paste the summary JSON back.

## 1.3b  BGE-reranker diagnostic — run BEFORE trusting 1.3 numbers

If 1.3 reports R@100 far below BM25 R@100, something is broken in the
candidate CSV (passage_text missing, doc_id format mismatch with qrels,
or rank column not sorted). Run this 3-second diagnostic:

```bash
cd ~/Desktop/islamm11/avatar_trial_onboarding
python scripts/diagnose_bm25_candidates.py --repo-root .
```

Paste the full stdout. It prints:
- cands row count, unique queries, per-query rank distribution
- qrels doc_id format (NCT prefix count)
- overlap of cand doc_ids with qrel doc_ids
- passage_text coverage after merge (NaN / empty / avg length)
- **theoretical BM25 top-100 R@100 ceiling** — the max R@100 any reranker
  can achieve on these candidates. If this is 0.42 but 1.3 reports 0.15,
  the reranker is scoring on empty text or the wrong ranks.

---

## Quick sanity check (do this first, 3 s)

```bash
cd ~/Desktop/islamm11/avatar_trial_onboarding
ls data/processed/onboarding_eval_cases.json \
   data/processed/trec_ct2021_topics_full.csv \
   data/processed/trec_ct2022_topics_full.csv \
   data/processed/trec_ct2021_qrels_full.csv \
   data/processed/trial_evidence_passages.parquet \
   data/processed/benchmark_nli4pr.csv \
   outputs/tables/bm25_per_query_metrics.csv \
   outputs/tables/crossenc_per_query_metrics.csv
```

All 8 should exist. Report any missing.
