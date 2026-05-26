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

## What's in this repository

```
scripts_phase3/              Phase-3 pipeline + dual-judge evaluation harness
scripts_phase4/              Phase-4 reviewer-revision analytics
  semantic_leak_judge.py     The third (semantic) leak detector
  adversarial_subset_analysis.py
  reviewer_revision_analytics.py
src/                         Core library (selector, postprocessor, etc.)
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
