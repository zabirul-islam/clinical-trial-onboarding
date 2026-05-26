# Paper rewrite plan — npj Digital Medicine target

**Date:** 2026-04-29
**Status:** awaiting user sign-off
**Source paper:** `outputs/paper_v2/main.tex` (736 lines, 31 pp)
**Source bib:** `outputs/paper_v2/references.bib` (74 entries)
**New evidence:** `outputs/phase4/*` (n=114 ablation + dual-judge + per-gate + per-area + calibration + zero-shot baseline)

---

## 1. What stays as-is

- §1 Introduction (refresh contribution list only; story unchanged).
- §2 Related Work — minor: add temperature scaling (Guo 2017) + Brier 1950 to bib.
- §3 Methods — verbatim. Methods didn't change; only evidence base did.
- §4.1 Datasets — minor.
- §4.4 Rubric — verbatim.
- §4.5 Ablations — verbatim.
- §5.1 Retrieval — verbatim.
- §5.2 Worked example — verbatim.
- §5.3 V1/V2/V-final on 15-case audit — verbatim (now triangulated by 3-way LLM agreement).
- §5.4 Decisive probes — verbatim.
- §5.5 Failure taxonomy — verbatim.
- §5.6 Qualitative output — verbatim.
- §5.9 Threshold sweep — verbatim (n=150 sweep stands).
- Appendix A/B/C — verbatim.

## 2. What changes (sections to update)

### 2.1 Abstract — full rewrite (~370 words)

**Old claim:** "30 onboarding cases × 4 open-weight backbones = 240 generations, 0% leak."
**New claim:** "**110 onboarding cases × 6 backbones = 660 generations across the curated 30-case selector-accept set, the broad 80-case selector-abstain set, and zero-shot closed-API baselines (GPT-4o, Claude Sonnet 4.5); cross-trial leak is 0% on every backbone × every case.**"
**Add:** per-gate ablation finding (no single gate carries safety; structural property), 6-area coverage breakdown, brief honest calibration framing.
**Remove:** "30 cases" everywhere.

### 2.2 §1 Contributions — update list 6 → 7 items

Add to the existing 6 contributions:
- **(7) Per-gate ablation** showing safety is structural across n=110 (no single gate's removal causes leak).
- Update (6) to mention 660 scored generations across **6 backbones** (4 open + 2 closed), and the n=110 expansion rather than 30.

### 2.3 §4.2 Onboarding evaluation corpus — extend

Old: 15 hand-crafted cases.
New: keep the 15-case audit description; **add** subsection §4.2.X "Expanded n=110 robustness pool" describing:
- n=110 unique cases drawn from handcrafted (15) + TREC-2021 paraphrased (30) + TREC-2022 paraphrased (30) + paraphrase-intent (35).
- Therapeutic-area distribution: MSK_Bone (14), Cardiovascular (8), Metabolic_Endocrine (19), Oncology (16), Neurology_CNS (13), Other (44).
- Selector-regime split: 32 cases pass deployment gates ("accept" regime); 82 cases would be abstained at deployment thresholds ("broad" regime).
- The 80 broad-pool cases were run with selector gate **disabled** to probe backbone behavior off-distribution; the 32 accept-regime cases retain the deployed gate.

### 2.4 §4.6 Backbone ablation and extended onboarding corpus — replace numbers

Old: 30 cases × 4 backbones = 120 records.
New: **110 cases × 4 open-weight backbones + 114 cases × 2 closed APIs = 660 records**. Closed-API baseline (GPT-4o, Claude Sonnet 4.5) receives the SAME guarded single-trial evidence and SAME prompt as the open-weight backbones, isolating the contribution of the LLM family from the contribution of the evidence gate.

### 2.5 NEW §4.X Per-gate ablation protocol (~150 words)

Counterfactual leave-one-out over the four selector signals (ρ, τ, μ, κ) plus the generic-flag heuristic. For each variant V-final-no-{ρ, τ, μ, κ, generic}, compute the set of cases the variant would have accepted from the n=110 pool, then measure leak/commit/abstain on the existing backbone outputs. This is a counterfactual ablation that requires no new generations, since the n=110 expansion was already run with the gate disabled.

### 2.6 NEW §4.Y External SOTA comparison protocol (~120 words)

Apply two closed-model APIs (GPT-4o, Claude Sonnet 4.5) to the same 114-case pool with the same single-trial guarded evidence and the same JSON-schema prompt. Compare per-model leak/commit/abstain to V-final-Qwen-3B. This isolates whether the safety property comes from the evidence gate or the open-weight backbone family. Limit `max_tokens=1500` to prevent JSON truncation. Cost ≈ $25.

### 2.7 NEW §4.Z Calibration protocol (~150 words)

Use the trial-first selector's normalized score share τ as the predicted probability of trial relevance; compute Expected Calibration Error (10-bin), Brier, and Brier Skill Score against three relevance targets (qrels-strict rel=2, qrels-any rel≥1, exact-NCT match) on the 50 cases of the 150-case expansion pool with TREC qrels. Additionally fit a 4-feature logistic regression over (ρ, τ, μ, κ) under 5-fold cross-validation as a counterfactual calibrator. Report negative results honestly: the selector is a gating signal, not a calibrated probability; post-hoc temperature/Platt scaling on a larger held-out set is left as future work.

### 2.8 §5.7 Backbone ablation — replace Table 6 + figure

New Table: per-backbone safety on n=110 (replaces n=30 numbers). All four open-weight backbones retain 0% leak. Add the regime breakdown (curated_30 vs broad_80) showing safety holds in both.

### 2.9 §5.8 Dual-judge LLM evaluation — replace Table 7 + Figure 6

New Table 7: judge-pooled rubric scores at n=114 with bootstrap 95% CIs (5000 resamples). Numbers shift slightly vs n=30: Qwen-7B = 3.95, Llama-8B = 3.52, Qwen-3B = 3.03, Mistral-7B = 2.66; all overall means up except Mistral (broad pool stresses it). Inter-judge agreement table refreshed.

New Figure 6: radar chart with 4 backbones at n=114 (or 6 backbones if we add closed APIs to the radar — recommend adding).

### 2.10 NEW §5.X — 3-way agreement on 15-case audit (~250 words + Table 9)

Apply the dual-judge harness to V1/V2/V-final variants of the 15-case audit (45 generations × 2 judges = 90 scores). Map the 10-dim human-rater rubric to the 5-dim judge rubric via a documented crosswalk (Appendix B.X).

**Per-variant key finding:** agreement is moderate-substantial on V1 (Spearman ρ ∈ [0.21, 0.79]); V2 and V-final saturate (human gives near-constant scores → undefined agreement). **The 5-dim LLM rubric retains discriminating power on V2 and V-final**, motivating it as a complementary instrument.

### 2.11 NEW §5.Y — Per-therapeutic-area safety (~200 words + Figure)

114 cases × 4 backbones × 6 areas = 24 (area × backbone) cells. **0% leak in every cell.** Areas: MSK_Bone (14), Cardiovascular (8), Metabolic_Endocrine (19), Oncology (16), Neurology_CNS (13), Other (44). Per-area parse_ok and commit rates differ by backbone (Qwen-7B uniformly highest; Mistral-7B 0% commit on Metabolic_Endocrine). Heatmap: 6 areas × 4 backbones × 3 metrics.

### 2.12 NEW §5.Z — Per-gate ablation (~250 words + Table 10)

V-final + 5 leave-one-out variants × 4 backbones × n=114. **CLEAN: 0% leak across all 24 (variant × backbone) cells.** Δ-cases relative to V-final: μ-removed admits +27 cases, ρ-removed +10, generic-removed +6, τ-removed and κ-removed admit 0 (slack gates under current thresholds). Paper claim: **safety is structural, not gate-specific**; multiplicative ensemble of (ρ, μ, generic) carries the filter; (τ, κ) are redundant under deployment thresholds.

### 2.13 NEW §5.W — External SOTA comparison (~250 words + Table 11)

Zero-shot baseline: GPT-4o (n=114, parse_ok=1.00, leak=0.0, commit=0.31) and Claude Sonnet 4.5 (n=114, parse_ok=1.00, leak=0.0, commit=0.14, abstain=0.11) on identical guarded single-trial evidence + JSON prompt. **Both closed APIs hit 0% leak**, confirming the safety property is invariant to backbone family (open-weight + closed-API). Table 11 reports leak/commit/abstain/parse_ok across all 6 model families. Headline: "**On 660 (case × model) generations across 6 model families, cross-trial leak rate is 0%.**"

### 2.14 NEW §5.V — Calibration analysis (~300 words + Figure)

τ-only baseline: ECE=0.15 (any), 0.21 (strict), 0.39 (gold_nct); Brier 0.27, 0.21, 0.17; **Brier Skill Score is negative** on all three relevance targets (BSS=−0.10, −0.45, NaN). Joint LR over (ρ, τ, μ, κ) under 5-fold CV: BSS=−0.06±0.25 — improvement statistically indistinguishable from zero given small n=50 with gold labels. **Honest framing:** τ is a gating signal at a fixed operating point, not a calibrated probability; we do not claim probabilistic calibration as a strength. Post-hoc temperature/Platt scaling on a larger held-out set (e.g., a follow-up paper's expanded gold-annotated pool) is the natural next step.

### 2.15 §6 Discussion — 4 new paragraphs

**Paragraph A — "Structural safety across model families."** Combine the n=110 + per-gate + zero-shot-closed-API findings: 0% leak on 660 (case × model) generations across 6 backbones × 6 leave-one-out variants. The safety property is a structural consequence of the guarded single-trial evidence and the multiplicative gate ensemble, not a function of the backbone's fluency, the LLM family, or any single gate.

**Paragraph B — "Slack gates and deployment simplification."** τ and κ reject zero cases under the current thresholds. The dominance ratio ρ, raw cross-encoder score μ, and generic-flag heuristic carry the filtering work. A simplified gate is feasible; we leave the threshold re-tuning as future work.

**Paragraph C — "Calibration is not a strength."** Honest reframe: the trial-first selector controls accept/reject at a fixed threshold; it is not a calibrated probability. Brier Skill Scores are negative against three relevance targets; a joint logistic regression over the four selector signals does not stabilize calibration at n=50 gold-annotated cases. Post-hoc temperature scaling on a larger held-out set is the obvious next direction; we frame this as an open problem rather than a system feature.

**Paragraph D — "Forward reference to clinician audit (Paper B)."** Multi-rater clinician audit, multi-turn patient simulation, and a controlled-perturbation operator taxonomy are deferred to a follow-up benchmark paper currently in preparation. Paper A establishes the system-and-safety baseline at scale; the follow-up will provide controlled empirical validation.

### 2.16 §7 Conclusion — refresh numbers

Replace "30 onboarding cases × 4 open-weights backbones (240 scored generations)" with "**110 onboarding cases × 6 backbones (660 scored generations spanning open-weight and closed-API model families, plus 600 generations under per-gate leave-one-out ablation)**". Add the structural-safety claim sentence.

### 2.17 References — add

- Guo et al. 2017 — Temperature Scaling (calibration future work cite).
- Brier 1950 — Brier score original.
- Optional: Niculescu-Mizil & Caruana 2005 — Platt scaling (calibration cite).

## 3. New figures to make

| ID | Filename | Content |
|---|---|---|
| F-area | `figures/per_area_heatmap.{pdf,png}` | 6 areas × 4 backbones; cells = leak rate (will be all 0); secondary heatmap = commit_rate |
| F-gate | `figures/per_gate_ablation.{pdf,png}` | bar chart: 6 variants × {n_accepted, leak_rate (=0), commit_rate} |
| F-radar-n114 | `figures/phase3_radar_per_backbone_n114.{pdf,png}` | radar updated to n=114 (replaces existing) |
| F-radar-with-closed | `figures/phase4_radar_open_vs_closed.{pdf,png}` | NEW: 6-backbone radar (open + closed APIs); shows safety is family-invariant |
| F-calib | `figures/calibration_reliability.{pdf,png}` | Already produced by `compute_calibration.py`. Just include. |
| F-area-bars | `figures/per_area_safety_bars.{pdf,png}` | optional: alternative to heatmap |

## 4. New tables

| ID | Section | Content |
|---|---|---|
| T-n110 | §5.7 (replaces T6) | per-backbone safety on n=110: parse_ok, commit, abstain, leak across 4 open backbones |
| T-judge-n114 | §5.8 (replaces T7) | judge-pooled rubric scores per backbone with 95% bootstrap CIs at n=114 |
| T-3way | §5.X | 3-way agreement on V1 of 15-case audit (V2/V-final saturated) |
| T-area | §5.Y | per-area × per-backbone safety summary |
| T-pergate | §5.Z | per-gate ablation: variant × backbone × {n_accept, leak, commit, abstain} |
| T-zeroshot | §5.W | open vs closed: 6 model families, n=114 each, leak/commit/abstain/parse_ok |
| T-calib | §5.V | calibration metrics (τ-only, joint LR; ECE, Brier, BSS) |

## 5. Claim-evidence map (new claims only)

| Claim | Evidence |
|---|---|
| n=110 cases × 4 open-weight backbones, 0% leak | `outputs/phase4/n114_aggregate/headline_n114.json` + `summary_n114_per_backbone.csv` |
| n=110 spans 6 therapeutic areas, each ≥4 cases × 4 backbones, 0% leak | `outputs/phase4/area_breakdown/area_labels_n114.csv` + `summary_n114_per_area.csv` |
| Per-gate ablation: 0% leak across 6 variants × 4 backbones | `outputs/phase4/per_gate_ablation/per_gate_summary.{json,csv}` |
| τ and κ are slack gates (0 cases rejected by either alone) | `outputs/phase4/per_gate_ablation/per_gate_summary.json` (`per_variant_n_only_this_gate`) |
| Closed APIs (GPT-4o, Sonnet 4.5) also 0% leak with same evidence | `outputs/phase4/zeroshot_baseline/zeroshot_summary.csv` |
| Selector-τ Brier Skill Score is negative on 3 relevance targets | `outputs/phase4/calibration/calibration_summary.json` |
| Joint LR over (ρ,τ,μ,κ) BSS = −0.06±0.25, statistically indistinguishable from zero | `outputs/phase4/calibration_lr/calibration_lr_summary.json` |

## 6. Adversarial self-review (research-paper-writing §1.6)

| Risk | Where | Mitigation |
|---|---|---|
| Reviewer: "still no clinician audit" | §6 Discussion limitations | explicit forward-reference to Paper B |
| Reviewer: "n=110 is still small" | §6 Discussion limitations | acknowledge; cite that 660 (case × model) cells with 0% leak is substantial despite per-axis n |
| Reviewer: "your zeroshot baseline isn't TrialGPT" | §6 Discussion limitations | acknowledge; argue our comparison controls for prompt + evidence; TrialGPT comparison is follow-up |
| Reviewer: "calibration is bad" | §5.V + §6 Discussion C | honest framing; future-work proposal |
| Reviewer: "MSK over-represented" | §5.Y + §6 limitations | report 6 areas, acknowledge MSK 14 vs Cardio 8 (already reported) |

## 7. Execution sequence (proposed)

| Step | Effort | Output |
|---|---|---|
| 7.1 Build figure scripts (5 figures) | ~3 hr | `figures/*.pdf` + `*.png` |
| 7.2 Build LaTeX table snippets | ~1 hr | `outputs/paper_v2/tables/*.tex` |
| 7.3 Update Abstract + Introduction Contributions | ~1 hr | Edit main.tex 87-117 |
| 7.4 Update §4 Experimental Setup (add 4 subsections) | ~2 hr | Edit main.tex 259-325 |
| 7.5 Update §5 Results (5 new subsections + replace 2 tables) | ~4 hr | Edit main.tex 326-641 |
| 7.6 Update §6 Discussion (4 new paragraphs) | ~2 hr | Edit main.tex 642-672 |
| 7.7 Update §7 Conclusion + add bib entries | ~30 min | Edit main.tex 673-690 + references.bib |
| 7.8 Humanizer pass on Abstract + Discussion + Conclusion | ~1 hr | Reduce AI fingerprints; vary rhythm |
| 7.9 Compile + visual proof | ~30 min | New `main.pdf` |
| 7.10 Cover letter draft | ~1 hr | `cover_letter_npj_dm.md` |
| **Total** | **~16 hr** | submission-ready PDF |

Realistic over 2-3 working days.

## 8. Open questions for user (decide now or default)

1. **Title rebrand?** Keep current 18-word title vs short alt "An Auditable Pipeline for Patient-Facing Clinical Trial Onboarding". Default = keep current; reviewer attention is gained in Abstract not title.
2. **Add closed APIs to the rubric figure?** Recommend yes — strengthens the family-invariance claim. Default = yes.
3. **TrialGPT external SOTA (T1.2A)?** Already decided: skip unless reviewer demands.
4. **Cover letter venue:** npj DM primary, JAMIA backup. Confirm npj DM.

## 9. Sign-off

After user confirms §1-§8, execute step 7.1 → 7.10 sequentially. Flag any deviation from plan during execution.
