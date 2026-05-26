# npj Digital Medicine strengthening plan

**Date:** 2026-04-20
**Target venue:** npj Digital Medicine
**Goal:** lift current 23-page paper from "ML4H workshop" tier to "npj Digital Medicine" tier without human-subject study.

---

## Why current paper is below npj DM bar

| Reviewer concern | Evidence in current paper |
|---|---|
| Tiny audit (n=15, single rater) | Section 4.5, Table 5 |
| No threshold sensitivity | Appendix C — values stated, not swept |
| Single backbone (Qwen2.5-3B) | Methods Section 3.6 |
| No statistical tests | All metrics reported as point estimates |
| Single therapeutic area | Audit cases mostly osteoporosis-adjacent |
| No SOTA reranker comparison | Only ms-marco-MiniLM-L-6-v2 |
| No cross-trial conflation quantification | Claimed qualitatively, not measured |

## Plan: 4 phases, ~7 GPU-days on DGX Spark GB10

### Phase 1: Data + retrieval upgrade (Day 1-2)

| ID | Output | Script |
|---|---|---|
| 1.1 | n=150 expanded case set | `build_expanded_case_set.py` |
| 1.2 | Retrieval bootstrap CIs (95%) | `bootstrap_retrieval_cis.py` |
| 1.3 | BGE-reranker-large (replaces MiniLM) | `run_bge_reranker_large.py` |
| 1.4 | TREC CT 2022 retrieval (cross-year) | `eval_trec_ct2022_retrieval.py` |

### Phase 2: Pipeline experiments (Day 3-5)

| ID | Output | Script |
|---|---|---|
| 2.1 | Cached selector scores for n=150 | `cache_selector_scores.py` |
| 2.2 | Threshold sweep (5×5×4×3 = 300 cfg) | `run_threshold_sweep.py` |
| 2.3 | Backbone ablation gen (4 LLMs × n=150) | `run_backbone_ablation.py` |

### Phase 3: LLM-judge + analysis (Day 5-6)

| ID | Output | Script |
|---|---|---|
| 3.1 | Multi-judge rubric scoring | `run_multi_judge_rubric.py` |
| 3.2 | Inter-judge agreement (κ) | `compute_judge_agreement.py` |
| 3.3 | NCT-ID leakage detector | `detect_nct_leakage.py` |
| 3.4 | Stat tests (bootstrap, McNemar, Wilcoxon) | `run_stat_tests.py` |

### Phase 4: Paper rewrite (Day 6-7)

| ID | Output |
|---|---|
| 4.1 | Updated `main.tex` Methods + Experiments + Results |
| 4.2 | 4 new figures (Pareto, backbone bars, retrieval CI forest, leakage heatmap) |
| 4.3 | 6 new tables |
| 4.4 | Final humanizer pass |
| 4.5 | Final compile to 28-32 pp PDF |

---

## Success criteria

- n ≥ 150 audit cases across ≥ 4 therapeutic areas
- Inter-judge Cohen's κ reported
- Threshold sweep shows V-final on Pareto frontier
- Backbone ablation shows safety controls model-agnostic (severe-fail rate < 5% across 4 LLMs)
- Bootstrap 95% CIs reported for all retrieval metrics
- McNemar p < 0.001 for V1 vs V-final on severe-unsupported flips
- NCT-ID leakage rate quantified per variant

## Compute budget (DGX Spark GB10, ~96GB unified memory)

| Phase | Wall time |
|---|---|
| Phase 1 | 1.5h GPU + 30min CPU |
| Phase 2 (gen) | 6h GPU |
| Phase 2 (sweep) | 12h CPU |
| Phase 3 (LLM-judge API) | 1h API + 8h local Qwen-32B |
| Phase 4 | 0 (paper writing) |
| **Total** | ~30h GPU + 12h CPU + 1h API |

## Risk register

| Risk | Mitigation |
|---|---|
| BGE-reranker-large OOM on long passages | Truncate to 512 tok, batch_size=4 |
| Llama-3.1-8B refuses medical queries | Use system prompt grounding |
| Multi-judge κ < 0.4 | Report and discuss; don't hide |
| Threshold sweep finds better config than current | Re-tune V-final, re-run audit |
| TREC topics too specific → low intent diversity | Augment with paraphrased intents |

## Out of scope

- Patient comprehension study (no medical expert available)
- New trial corpus (use TREC CT 2021/2022 only)
- Chain-of-thought prompting variants (stay model-agnostic)
- Adversarial robustness probing (separate paper)
