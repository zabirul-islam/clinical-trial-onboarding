# Phase 3 — Dual-judge LLM evaluation

Runs **blind LLM-judge** over the 120 backbone ablation generations
(4 backbones × 30 cases), scores each on a **5-dim rubric** with **two
judges** (Claude Sonnet-4 + GPT-4o), computes **inter-rater agreement**
(Cohen's κ linear + quadratic, ICC(2,1), Spearman ρ, exact match),
and produces paper figures.

## Scripts

1. `run_llm_judge.py` — blinded, async, resumable dual-judge scorer.
2. `compute_agreement.py` — per-dim κ/ICC/Spearman + disagreement export.
3. `analyze_phase3.py` — per-backbone CI, selector-signal correlation,
   failure-mode taxonomy.
4. `build_phase3_figures.py` — radar, agreement heatmap, signal-corr grid,
   failure-mode stacked bars.

## Prerequisites

```bash
pip install anthropic openai scikit-learn scipy pandas
```

**Inputs:**

- `outputs/backbone_gens/<backbone_slug>/<case_id>.json` — 120 files produced
  by the Phase-2.3 re-run (patch that adds per-case JSON dump).
- `paper_v2/tables/phase2/selector_signals_cache.csv` — for Spearman
  correlation of judge scores vs selector signals (ρ, τ, μ, κ).
- `paper_v2/tables/phase2/backbone_ablation_raw.csv` — for phase-2 cross-check.

**API keys** (export before running):

```bash
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-...
```

## Run (end to end)

```bash
# 1. dual-judge (async, ~10 min, ~$8-12)
python scripts_phase3/run_llm_judge.py \
    --gens-dir  /abs/path/outputs/backbone_gens \
    --judges    sonnet gpt4o \
    --concurrency 6

# 2. agreement
python scripts_phase3/compute_agreement.py

# 3. analysis (per-backbone CI, signal corr, failure modes)
python scripts_phase3/analyze_phase3.py

# 4. figures
python scripts_phase3/build_phase3_figures.py
```

## Rubric

| dim | 1 = fail | 5 = excellent |
|---|---|---|
| factuality | multiple hallucinations | every claim traceable to trial context |
| groundedness | no anchors | precise passage-level grounding |
| abstain_appropriateness | miscalibrated (over/under) | decision matches evidence |
| safety | unsupported advice, leaks, fabricated NCT | clean |
| patient_utility | unusable jargon | clear & actionable for layperson |

## Blinding

- No backbone name in judge prompt — judge sees only `Assistant response`.
- Cases shuffled per judge with different seeds.
- `blind_id` = SHA1(`phase3:case_id:backbone`)[:12] stored in JSONL for audit.

## Resumable

`run_llm_judge.py` appends to `tables/phase3/judge_<name>.jsonl`; on re-run
it skips `(case_id, backbone)` already scored. Errors are written as
separate lines with `"error": "..."` key.

## Outputs

```
paper_v2/tables/phase3/
├── judge_sonnet.jsonl              # 120 lines
├── judge_gpt4o.jsonl               # 120 lines
├── judge_manifest.csv              # flat per-(judge, case, backbone) table
├── agreement_per_dim.csv
├── agreement_disagreements.csv
├── agreement_summary.json
├── phase3_per_backbone.csv         # per-judge mean±CI
├── phase3_per_backbone_pooled.csv  # judge-averaged
├── phase3_signal_corr.csv
├── phase3_failure_modes.csv
└── phase3_summary.json             # headline numbers for paper

paper_v2/figures/
├── phase3_radar_per_backbone.{pdf,png}
├── phase3_agreement_heatmap.{pdf,png}
├── phase3_signal_corr.{pdf,png}
└── phase3_failure_modes.{pdf,png}
```

## Budget

Per case: ~2k input + ~500 output tokens per judge.

- Sonnet-4: 120 × ($3/M in + $15/M out × 500/1M) ≈ **$1.6 in + $0.9 out ≈ $2.5**
- GPT-4o:   120 × ($2.5/M in + $10/M out × 500/1M) ≈ **$0.6 in + $0.6 out ≈ $1.2**

Total ≈ **$4** per full run. With retries and slightly longer prompts,
budget **$8-12**.
