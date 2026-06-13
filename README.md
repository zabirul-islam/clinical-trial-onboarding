# Audited Grounded Pipeline for Clinical Trial Onboarding

Companion code and frozen audit bundle for the paper
**"Structural Safety in Patient-Facing AI for Clinical Trial Onboarding:
An Audited Grounded Pipeline"**
(under review, npj Digital Medicine).

The pipeline turns a patient utterance into a single-trial-grounded onboarding
response (eligibility triage, consent-style explanation, teach-back questions).
A five-condition selector ensemble enforces single-trial evidence at the
retrieval layer; a strict postprocessor rewrites uncited fields to a fallback
string; a JSON schema collapses non-parsing outputs to `cannot-determine`.
Three pre-specified detectors (narrow NCT regex, wide entity-resolved
token-match, blinded dual LLM-judge semantic auditor) instrument the
parametric-memory leak channel.

## Headline numbers (frozen audit, 684 generations × 6 backbones)

| Detector | Rate |
|---|---|
| Lexical (narrow NCT regex) | 0 / 684 |
| Lexical (wide entity-resolved) | 0 / 684 |
| Semantic LLM-judge — Sonnet 4.5 | 23 / 681 (3.4%) |
| Semantic LLM-judge — GPT-4o | 93 / 684 (13.6%) |
| Semantic — either-judge OR | 104 / 684 (15.2%) |
| **Semantic — both-judges AND (consensus)** | **12 / 684 (1.75%)** |

Cohen $\kappa = 0.16$, raw agreement 86.5%. Manual review of the 12
consensus-flagged cases: 8 unambiguous paraphrased cross-trial content,
2 mixed-attribution model errors, 2 likely judge over-flag. Residual
concentrates on the adversarial subset ($\rho < 1.5$, $\geq 2$ pool trials)
and on the smallest open-weight backbone (Qwen-2.5-3B, 5.7% adversarial).

## Strengthening (phase 5): baselines, failure taxonomy, claim-level verification

Four pre-registered baselines (912 generations, 2 deployment backbones) isolate
the mechanism: only structural single-trial control closes the cross-trial channel.

| System | Lexical (wide) | Semantic (dual-judge consensus) | T1 | T2 | T3 |
|---|---|---|---|---|---|
| V-final (single-trial guard) | 0.000 | 0.018 | — | — | — |
| B1 multi-trial RAG | 0.022 | 0.583 | 124 | 9 | 0 |
| B2 prompt-only guard | 0.053 | 0.509 | 107 | 8 | 0 |
| B3 citation-enforced | 0.211 | 0.590 | 125 | 9 | 0 |
| **B4 top-1 selection** | **0.000** | **0.013** | **0** | 2 | 1 |

Taxonomy: **T1** cross-trial contamination, **T2** unsupported clinical completion,
**T3** ordinary hallucination. Structural control (B4, V-final) eliminates T1 entirely.
A claim-level NLI verifier (DeBERTa-v3-large MNLI; redact a claim iff a non-selected
pool trial entails it better than the selected trial) removes **8 / 12** consensus
semantic leaks — exactly the genuine cross-trial ones — at a 15.1% per-claim utility
cost. Cost-of-abstention: of 82 abstain cases, 78% had no relevant trial available
(correct abstention); B4 top-1 attains the highest dual-judge-pooled utility (3.47/5)
at near-zero leakage (Pareto-favourable). A blinded clinical-expert review packet
(30 cases) is released for independent validation.

## What's in this repository

```
scripts_phase3/              Phase-3 pipeline + dual-judge evaluation harness
scripts_phase4/              Phase-4 reviewer-revision analytics
  semantic_leak_judge.py     The third (semantic) leak detector
  adversarial_subset_analysis.py
  reviewer_revision_analytics.py
scripts_phase5/              Phase-5 strengthening (baselines, taxonomy, NLI)
  build_evidence_pools.py    Per-case multi-trial retrieval pools (BM25→cross-enc)
  run_baselines.py           B1–B4 baseline generation harness
  audit_baseline_leaks.py    Lexical (narrow+wide) audit of baselines
  run_baseline_semantic_taxonomy.py  Dual-judge semantic + T1/T2/T3 taxonomy
  nli_redact.py              Claim-level NLI verification + counterfactual audit
  cost_of_abstention.py      Missed-utility analysis over abstain cases
  build_expert_packet.py     Blinded clinical-expert review packet builder
  build_phase5_figures.py    Baseline-leak, taxonomy, safety–utility figures
  sync_manifest.py, sync_diff.py   Server/local sync-gate tooling
src/                         Core library (selector, postprocessor, etc.)
configs/phase5_baselines.yaml   Frozen pre-registration (baselines, NLI, taxonomy)
outputs/phase5/baseline_gens/   912 baseline generations (B1–B4 × 2 backbones)
outputs/phase5/evidence_pools/  114 per-case multi-trial pools
outputs/phase5/semantic_judge/  Dual-judge semantic+taxonomy JSONLs + summaries
outputs/phase5/rubric_judge/    Per-baseline dual-judge rubric scores
outputs/phase5/nli/             NLI calibration + redaction audit
outputs/phase5/leak_taxonomy_master.csv   Master leak + taxonomy table
outputs/expert_review/          30 blinded dossiers + scoring instrument
outputs/paper_v2/            LaTeX source + compiled main + supplement PDFs
                             (both generic and npj formats)
outputs/phase4/reviewer_fixes/
  semantic_leak_judge_{sonnet,gpt4o}.jsonl    Per-call judge outputs
  semantic_leak_summary.csv                   Per-backbone rates
  semantic_leak_irr.csv                       Inter-rater reliability
  adversarial_subset_summary.csv              Stratified rates
  adversarial_subset_case_list.csv            Per-case audit
  leak_extended_per_case.csv                  Lexical detector outputs
outputs/backbone_gens/                        30 anchor cases × 4 open-weight
outputs/phase4/n100_expansion/gens/           84 expansion cases × 4 open-weight
outputs/phase4/zeroshot_baseline/gens/        114 cases × 2 closed-API
outputs/tables/selector_signals_cache.csv     Selector telemetry
```

## What's NOT here (large corpus → HuggingFace dataset)

- `processed/trial_evidence_passages.parquet` (~660 MB, derived from
  public ClinicalTrials.gov + TREC CT 2021/2022 corpus)
  → **HuggingFace dataset:**
  https://huggingface.co/datasets/zabir1996/clinical-trial-onboarding-corpus

Download with:
```bash
hf download zabir1996/clinical-trial-onboarding-corpus \
    trial_evidence_passages.parquet --repo-type dataset \
    --local-dir processed/
```

The BM25 top-100 candidate cache is small enough to ship in this repo
(`outputs/tables/bm25_full_text_top100_candidates.csv`).
Both are derivable from public sources via `scripts/build_retrieval_corpus.py`
+ Pyserini.

## Reproduce the headline numbers

Lexical detectors (no API; reads existing generation JSONs):

```bash
python scripts_phase4/reviewer_revision_analytics.py
# writes outputs/phase4/reviewer_fixes/leak_extended_{per_case,summary}.csv
```

Semantic detector (requires Anthropic + OpenAI API keys, ~$13, ~2.3h):

```bash
echo 'export ANTHROPIC_API_KEY=sk-ant-...'   >> .api_keys
echo 'export OPENAI_API_KEY=sk-...'          >> .api_keys
set -a; source .api_keys; set +a
python -u scripts_phase4/semantic_leak_judge.py \
    --judges sonnet gpt4o --min-interval 6.0 \
    --out-dir outputs/phase4/reviewer_fixes
```

Adversarial-subset stratification (no API):

```bash
python scripts_phase4/adversarial_subset_analysis.py
```

## Reproduce the phase-5 strengthening

```bash
# 1. evidence pools + baseline generations (GPU)
python scripts_phase5/build_evidence_pools.py
python scripts_phase5/run_baselines.py            # 912 gens, B1–B4 × 2 backbones

# 2. lexical audit (no API)
python scripts_phase5/audit_baseline_leaks.py --build-vocab   # once (needs corpus parquet)
python scripts_phase5/audit_baseline_leaks.py

# 3. dual-judge semantic + taxonomy, and rubric (API keys)
set -a; source .api_keys; set +a
python scripts_phase5/run_baseline_semantic_taxonomy.py --judges sonnet gpt4o
for b in B1_multi_rag B2_prompt_guard B3_citation_enforced B4_top1; do
  python scripts_phase3/run_llm_judge.py --gens-dir outputs/phase5/baseline_gens/$b \
      --out-dir outputs/phase5/rubric_judge/$b --judges sonnet gpt4o
done

# 4. NLI claim verification (GPU) + cost-of-abstention + figures (no API)
python scripts_phase5/nli_redact.py --calibrate
python scripts_phase5/nli_redact.py --apply 684
python scripts_phase5/nli_redact.py --apply 912
python scripts_phase5/cost_of_abstention.py
python scripts_phase5/build_phase5_figures.py
```

Self-tests (no API/GPU): `python scripts_phase5/audit_baseline_leaks.py --selftest`,
`python scripts_phase5/nli_redact.py --selftest`, `python scripts_phase5/expert_agreement.py --selftest`.

## Build the paper

```bash
cd outputs/paper_v2
pdflatex main.tex && bibtex main && pdflatex main && pdflatex main
pdflatex main_npj.tex && bibtex main_npj && pdflatex main_npj && pdflatex main_npj
pdflatex supplement.tex && pdflatex supplement.tex
pdflatex supplement_npj.tex && pdflatex supplement_npj.tex
```

## Citation

After acceptance the paper-side BibTeX entry will be added here. For now,
cite as work-in-progress:

```bibtex
@unpublished{islam2026audited,
  author = {Md Zabirul Islam and Ge Wang},
  title  = {Structural Safety in Patient-Facing AI for Clinical Trial Onboarding:
            An Audited Grounded Pipeline},
  year   = {2026},
  note   = {Manuscript under review.}
}
```

## License

MIT. See [LICENSE](LICENSE).

## Contact

- Ge Wang (corresponding) — `wangg6@rpi.edu`
- Md Zabirul Islam — `islamm11@rpi.edu`
