# npj Digital Medicine Revision Plan

**Date:** 2026-05-03
**Trigger:** AI peer review (TMI-style), Major revision (score 6).
**Target venue:** npj Digital Medicine.
**Status:** approved by user.

## Reviewer comments addressed

### Major
| # | Comment | Fix | Evidence |
|---|---------|-----|----------|
| 1 | "Safety as structural property" needs theoretical justification | New §3.5 *Why the gate ensemble is safe*: formal multiplicative-bound argument; characterize necessary conditions; tightness analysis. | Per-gate ablation table 17 (already in paper). |
| 2 | "Leak" definition under-specified | Refined op-def in §3.1 + §4.X. Two definitions: narrow (NCT regex) and wide (distinctive trial-token entity match). Both = 0% across 660 generations. | `outputs/phase4/reviewer_fixes/leak_extended_summary.csv` + new figure `leak_rate_grid.pdf`. |
| 3 | 15-case audit single-rater, IRR underspecified | New §5.12 *Inter-rater reliability across rater types*. Krippendorff's α (interval) on 3-way (Human + Sonnet + GPT-4o), per dim, per variant. V1 retains variance (α∈[0.29, 0.59] on substantive dims); V2/V-final saturated by design. | `outputs/phase4/reviewer_fixes/krippendorff_alpha_3way.csv` + `krippendorff_3way.pdf`. |
| 4 | Negative calibration → clinical implications underspecified | §5.13 + Discussion *Calibration and clinical deployment*. Explicit: gate is binary, not probabilistic; downstream safety unaffected; no recommended use of τ as threshold-tuning probability. | Existing calibration tables + `bss_resolved.csv`. |
| 5 | n=110 curation underspecified | Expanded §4.7 with provenance subtable: TREC topic IDs, paraphrase template, manual pass/fail. | `processed/onboarding_eval_cases.json` + paper text. |
| 6 | No analysis of abstain regime | New §5.10 *Abstain-regime utility*: per-(backbone, regime) judge means, narrowing-prompt quality. | `outputs/phase4/reviewer_fixes/abstain_regime_quality_short.csv` + `abstain_regime_quality.pdf`. |

### Minor
| # | Comment | Fix |
|---|---------|-----|
| 1 | 150-gen pool vs 110-case relationship | Footnote in §4.5 explaining pool tree (110 base → 40 paraphrase extensions for calibration n=150 → 50 with TREC qrels for gold). |
| 2 | Generic-flag heuristic detail | Trigger list (21 phrases) + precision 0.75 / recall 0.86 on 15 audit cases. New §3.3.1. |
| 3 | Per-gate ablation methodology | Expanded §4.9 with case-selection criteria. |
| 4 | Brier "nan" inconsistency | Footnote in §5.13 explaining cause: base_rate ∈ {0,1} → naive Brier = 0 → BSS undefined. Clean table. |
| 5 | 150-gen pool in Pareto analysis | Same footnote as Minor-1. |

## npj Digital Medicine framing changes

1. **Title** → patient-facing safety lead.
2. **Structured abstract** (Background / Methods / Results / Conclusions, ≤200 words).
3. **Standalone Limitations subsection.**
4. **Clinical Translation paragraph** in Discussion.
5. **Data Availability + Code Availability + Ethics + Author Contributions** kept (already npj-compliant).

## Files touched

| File | Change |
|------|--------|
| `outputs/paper_v2/main.tex` | All edits. |
| `outputs/paper_v2/references.bib` | +Krippendorff 1980/2018, +Cohen 1960 (already there), +Hallgren 2012, +Gwet 2014. |
| `outputs/paper_v2/figures/abstain_regime_quality.pdf` | New. |
| `outputs/paper_v2/figures/leak_rate_grid.pdf` | New. |
| `outputs/paper_v2/figures/krippendorff_3way.pdf` | New. |
| `scripts_phase4/reviewer_revision_analytics.py` | New analytics script. |
| `scripts_phase4/build_revision_figures.py` | New figure script. |
| `outputs/phase4/reviewer_fixes/*.csv` | Computed evidence tables. |

## Reviewer claim verifications

| Claim | Source artifact |
|-------|----------------|
| narrow leak = 0% | `outputs/phase4/n114_aggregate/headline_n114.json` |
| wide leak = 0% | `outputs/phase4/reviewer_fixes/leak_extended_summary.csv` |
| Krippendorff α V1 (factuality 0.29, groundedness 0.59, utility 0.53) | `outputs/phase4/reviewer_fixes/krippendorff_alpha_3way.csv` |
| Generic-flag prec 0.75 / rec 0.86 | `outputs/phase4/reviewer_fixes/generic_flag_eval.json` |
| Abstain-regime utility means | `outputs/phase4/reviewer_fixes/abstain_regime_quality_short.csv` |
| Per-gate leak = 0% all variants | `outputs/phase4/per_gate_ablation/per_gate_summary.json` |

## What was NOT changed
- No new generation runs.
- Existing tables / figures kept intact (preserves cross-reference stability).
- Author list, affiliations, acknowledgements unchanged.
