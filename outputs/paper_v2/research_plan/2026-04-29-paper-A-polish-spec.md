# Paper A polish spec — npj Digital Medicine target

**Date:** 2026-04-29
**Author:** Zabir (with Claude as planning aid)
**Target venue (primary):** npj Digital Medicine
**Backup venues:** JAMIA, npj Health Systems, ML4H Findings (last resort)
**Time budget:** 4 weeks
**Constraint:** No clinician available for human eval. Plan assumes no human-subjects work this round. Clinician validation is **explicitly deferred to Paper B**.
**Status:** Draft for user review

---

## 0. Server / local sync state

Server has more than local. As of 2026-04-29 server has:

| Server path | Local mirror? | Comment |
|---|---|---|
| `scripts_phase3/{run_llm_judge,compute_agreement,analyze_phase3,build_phase3_figures}.py + README.md` | NEW: synced 2026-04-29 | dual-judge harness, reusable for T1.1 |
| `outputs/phase3/` (judge JSONL + agreement + figures) | NOT SYNCED | needed locally to inspect 30-case judge results |
| `outputs/backbone_gens/{4 backbones}/{30 cases}.json` | partial (5 visible) | rest needed for Tier-1 work |
| `logs/{backbone_ablation_v2.log, phase3_judge.log}` | NOT SYNCED | trace logs, useful for debug |
| `indices/dense_*.faiss + .npy` | NOT SYNCED | retrieval indices, only need if re-running retrieval |
| `docs_phase2_README.md` | NOT SYNCED | Phase 2 doc, possibly already-superseded |
| `scripts/{cache_selector_scores,run_threshold_sweep,run_backbone_ablation,run_bge_reranker_large,bootstrap_retrieval_cis,build_expanded_case_set,eval_nli4pr_eligibility_alignment,recompute_nli4pr_binary,export_bm25_topk}.py` | partially synced | server scripts overlap with `outputs/paper_v2/scripts_phase1/` drafts |

**Action item before impl plan:** `rsync` server `outputs/phase3/` and full `outputs/backbone_gens/` to local. Without these I can't inspect actual judge results to validate the impl plan.

```bash
rsync -avz islamm11@zabi-nvidia-gpu.bme.rpi.edu:~/Desktop/islamm11/avatar_trial_onboarding/outputs/phase3/ \
  ~/Desktop/Clinical-Trial/outputs/phase3/
rsync -avz islamm11@zabi-nvidia-gpu.bme.rpi.edu:~/Desktop/islamm11/avatar_trial_onboarding/outputs/backbone_gens/ \
  ~/Desktop/Clinical-Trial/outputs/backbone_gens/
rsync -avz islamm11@zabi-nvidia-gpu.bme.rpi.edu:~/Desktop/islamm11/avatar_trial_onboarding/logs/ \
  ~/Desktop/Clinical-Trial/logs/
rsync -avz islamm11@zabi-nvidia-gpu.bme.rpi.edu:~/Desktop/islamm11/avatar_trial_onboarding/docs_phase2_README.md \
  ~/Desktop/Clinical-Trial/
```

---

## 1. Where Paper A stands today

The current draft (`outputs/paper_v2/main.pdf`, 31 pp) already contains:

- Two-stage BM25→cross-encoder retrieval on TREC CT 2021 (Recall@100 = 0.96).
- Trial-first single-context selector (ρ, τ, μ, κ gates + generic-question flag).
- Three-variant ablation V1 / V2 / V-final on the 15-case rubric audit.
- Decisive probe checks on anchor / generic-participation / vague queries.
- 30-case backbone ablation across Qwen-2.5-3B/7B, Llama-3.1-8B, Mistral-7B (120 generations).
- Blind dual-judge LLM evaluation (Sonnet 4.5 + GPT-4o, 240 scored generations) on a 5-dimension rubric.
- 2,520-point threshold-grid sweep with Pareto frontier and unsafety = 0 across all Pareto points.
- McNemar tests on V1 → V-final paired outcomes with p-values.
- Bootstrap 95% CIs on the per-backbone rubric means.
- Inter-judge agreement (κ, ICC, Spearman, exact-match, MAE).
- Failure-mode taxonomy and per-backbone tag distribution.

This is already a strong systems paper. It is *not* yet a strong npj DM paper. The gap between the two is in **clinical rigor, generalization claims, and external comparison**, not in engineering.

---

## 2. What npj DM reviewers will hit

### 2.1 Self-acknowledged limitations (paper §6 Limitations)

1. 15-case audit is single-rater.
2. Threshold sweep is retrospective on the same case pool, not held-out across institutions.
3. The production backbone is 3B; 7B variant is more fluent.
4. No human-subject teach-back evaluation.

### 2.2 Additional reviewer concerns we anticipate

5. **No external SOTA comparison.** Paper compares only V1 / V2 / V-final variants of the *same* pipeline. No comparison to TrialGPT (Jin et al. 2024, *Nature Communications*), PRISM (Gupta et al. 2024, *npj Digital Medicine*), or DeepEnroll on the same case set. Reviewers will say "compared to what?".
6. **"Therapeutic-area diversity" is asserted, not shown.** The 30-case corpus claims cardio/metabolic/oncology coverage. Inspection of `outputs/tables/backbone_ablation_raw.csv` shows the corpus is dominated by `possible_match_insufficient_evidence` (76/120 = 63% of rows) and `consent_understanding` (24/120 = 20%). Per-area breakdown is not in the paper. This is fixable.
7. **No calibration metrics.** Paper reports decisions and rubric scores, but no ECE / Brier / reliability diagrams on the eligibility decision when gold labels exist (TREC qrels). Standard metric for clinical AI papers; missing.
8. **No per-gate ablation.** Paper has V1 (all gates off) vs V2 (binary refuse) vs V-final (all gates on). It does not show *which* gate (ρ vs τ vs μ vs κ vs generic-flag) carries the safety load. Reviewers will ask.
9. **30-case audit pool is small.** The original strengthening plan (`2026-04-20-npj-dm-strengthening.md`) targeted n=150. Only 30 was actually run. The 2,520-point threshold sweep is on *those same 30 cases* (plus 120 unlabeled signals from a 150 expansion that isn't part of the rubric eval). npj DM reviewers will accept ~100 cases; 30 is shaky.
10. **Single reranker.** Paper uses ms-marco-MiniLM-L-6-v2. Modern medical/biomedical RAG papers typically also report BGE-reranker-large or a similarly larger reranker. Not a dealbreaker, but expected.
11. **No ablation on the generic-flag rule.** It's keyword-based, which is fragile. Paper should show (a) sensitivity to the keyword set, and (b) what happens if generic-flag is replaced by an LLM-based generic-classifier.

### 2.3 Limitations we leave on the table

| Limitation | Why we're keeping it |
|---|---|
| No clinician audit | Constraint of this round; explicitly deferred to Paper B |
| No human-subject teach-back study | Same — Paper B |
| Single-institution corpus (TREC) | Out of scope; framed honestly in Limitations |
| No multilingual eval | Out of scope; framed honestly |
| No real EHR / patient-note inputs | Out of scope; framed honestly |
| No prospective deployment metrics | Out of scope; framed honestly |

---

## 3. What we'll add (Tier 1 — must do)

These close the gaps (5)–(11) above. None require a clinician.

### T1.1 — Second LLM judge on the 15-case audit

Closes (1). Apply the existing dual-judge harness (`scripts_phase3/run_llm_judge.py`) to the **original 15-case rubric** (Section 4.4 in paper, 10-dim audit). Currently scored by single human rater (the author). After this, every case has the human-rater score *and* two LLM-judge scores.

**Reuse strategy.** Three variants V1, V2, V-final × 15 cases = 45 generations to score. The existing `run_llm_judge.py` reads `<gens_dir>/<backbone_slug>/<case_id>.json`. Adapter step:

1. Add `scripts_phase3/build_15case_audit_gens.py` that reads `outputs/eval_runs/case_*/`, `outputs/eval_runs_v2/case_*/`, `outputs/eval_runs_final/case_*/` and writes `outputs/15case_audit_gens/{V1,V2,V-final}/case_XX.json` in the same schema the judge expects.
2. Run `run_llm_judge.py --gens-dir outputs/15case_audit_gens --judges sonnet gpt4o`.
3. Run `compute_agreement.py` over the resulting JSONL — but extend it to also load the human-rater 10-dim scores from `outputs/eval_runs_final/onboarding_eval_audit_summary_final.txt` and report 3-way agreement (human / Sonnet / GPT-4o).

Note: the current 5-dim phase3 rubric (factuality / groundedness / abstain_appropriateness / safety / patient_utility) is *not* identical to the 10-dim audit rubric (topic-relevance / overstatement / unsupported / fallback / missing-fact / unresolved-req / teach-back / clarity / overall-usable / needs-expert-review). Two options:

- **Map 10-dim → 5-dim** for the agreement comparison. E.g., human "severe_unsupported" maps to LLM-judge "groundedness ≤ 2"; "overall_usable=yes" maps to "patient_utility ≥ 4". Document the mapping table.
- **Re-run human + LLM on a 5-dim audit** of the same 15 cases. Heavier but cleaner. Recommend skipping; mapping is enough.

Pick mapping. Heavier path goes into Tier 2.

Report inter-rater agreement: (a) human rater vs Sonnet, (b) human rater vs GPT-4o, (c) Sonnet vs GPT-4o on the mapped 15-case audit. Cohen's κ (linear + quadratic) and Spearman ρ.

**Result expected:** moderate-to-substantial agreement on factuality / groundedness / fallback correctness; weaker on the subjective dimensions. This pattern matches the 30-case dual-judge result and demonstrates the original audit, while single-rater, is consistent with LLM-judge consensus.

### T1.2 — External SOTA comparison

Closes (5). Run **TrialGPT** (Jin et al. 2024, public code at https://github.com/ncbi-nlp/TrialGPT) on the same 30-case backbone-ablation corpus. Score with the same dual-judge harness on the same 5-dimension rubric. Report side-by-side with V-final (Qwen-2.5-3B prod backbone).

Optionally also run PRISM (Gupta et al. 2024, public). PRISM uses patient EHR records as input — adapter needed to ingest our patient-facing utterances. Skip if adapter is non-trivial.

**Result expected:** TrialGPT does not enforce single-trial consistency, so it should produce non-zero cross-trial leak rate on broad participation queries. This is the headline external comparison.

### T1.3 — Per-therapeutic-area breakdown

Closes (6). Annotate the 30 cases with therapeutic area: `MSK / Bone`, `Cardiovascular`, `Metabolic / Endocrine`, `Oncology`, `Neurology / CNS`, `Other`. Use a simple LLM-assisted labeler with manual spot-check on all 30. Then re-aggregate every safety metric (cross-trial leak, severe-overstatement, severe-unsupported, abstain rate, judge-pooled rubric scores) per area. Report a per-area heatmap.

**Result expected:** safety result holds across all areas (no area shows a leak); rubric fluency may vary by area, which is honest and reportable.

### T1.4 — Calibration metrics

Closes (7). For cases with a gold NCT (TREC qrels-derived), score the V-final eligibility decision against gold using:

- Expected Calibration Error (ECE) with 10 bins.
- Brier score on the binarized "any match" vs "no match" decision.
- Reliability diagram.

**Caveat — read before implementing.** The current pipeline does not emit a continuous probability; it emits a discrete decision in {LM, PMIE, LMM, CD} plus an abstain status. ECE/Brier require a probability. Two options:

1. **Selector-signal proxy (recommended).** Use the trial-first selector's normalized score `S(t⋆) / Σ S(t)` (i.e., τ from the paper) as the predicted probability of "this is the right trial". This is already cached in `outputs/tables/selector_signals_cache.csv`. ECE/Brier are then computed on this continuous signal vs the gold-NCT match label. This is honest because τ is what the system actually uses to gate decisions.

2. **LLM token-probability proxy.** Re-run V-final inference with `output_scores=True` on the eligibility-decision token (e.g., probability mass on "likely_match" vs other tokens). Heavier; only do if option 1 isn't compelling.

Pick option 1 first. If the calibration is poor, add option 2 as a more granular probe.

**Result expected:** V-final is well-calibrated on commit cases (where the guard accepts) and over-conservative on abstain cases. Both are reportable.

### T1.5 — Per-gate ablation

Closes (8). Define five "leave-one-out" variants of V-final:

- V-final-no-ρ: dominance gate off.
- V-final-no-τ: trial-score-share gate off.
- V-final-no-μ: raw cross-encoder score gate off.
- V-final-no-κ: best-rank gate off.
- V-final-no-generic: generic-flag heuristic off.

Run all five on the 30-case corpus + all four backbones (where the existing pipeline allows). Report per-variant cross-trial leak, severe overstatement, severe unsupported, and abstain rate.

**Result expected:** the generic-flag and the ρ gate carry most of the safety load on broad / vague queries; μ and κ matter most on retrieval-drift cases.

### T1.6 — Expanded audit pool to n ≥ 100

Closes (9). Run `build_expanded_case_set.py` (already drafted in `outputs/paper_v2/scripts_phase1/`, never executed). Target 100 cases minimum (vs original plan's 150). Use the same backbone-ablation harness on the expanded pool. Re-run dual-judge eval on the expansion (this is API-budget bound — see §5).

**Result expected:** safety result holds at scale; rubric scores stay within the 95% CI of the 30-case result. If they do, we report 100-case numbers as the headline. If they shift materially, we report both and analyze.

### T1.7 — Paper rewrite

Closes nothing directly but is required to integrate T1.1–T1.6 and to reframe for npj DM:

- Move "Trustworthy clinical-AI deployment" framing earlier (§1 Introduction).
- Add §3.X "External baselines" subsection introducing TrialGPT and the comparison protocol.
- Add §5.X "Calibration" and §5.X "Per-gate ablation" results subsections.
- Rewrite §6 Limitations to position multi-rater clinician audit as Paper B (forward reference; do not over-promise).
- Add Figure: deployment workflow (study coordinator review path), npj DM-style.
- Tighten ethics & equity to npj DM expectations (FDA diversity guidance, 2022; Clark et al. 2019).
- Final humanizer pass on Abstract, Introduction, Conclusion. Cut AI-fingerprint phrasing.

---

## 4. What we'll add (Tier 2 — nice, do if time)

| ID | What | Why | Effort |
|---|---|---|---|
| T2.1 | BGE-reranker-large rerun (Phase 1.3 from old plan) | Modern reranker baseline | 0.5 day |
| T2.2 | TREC CT 2022 cross-year retrieval (Phase 1.4) | Cross-year generalization | 0.5 day |
| T2.3 | Bootstrap 95% CIs on retrieval metrics | Closes minor gap | 0.5 day |
| T2.4 | NLI4PR eligibility-alignment study (Phase 1.5) | Shows pipeline can also do NLI4PR-style criterion-level NLI; useful bridge to Paper B | 1 day |
| T2.5 | Generic-flag sensitivity analysis | Closes (11) | 0.5 day |
| T2.6 | Selector-signal correlation with rubric stratified by therapeutic area | Strengthens the "guard absorbs correlation" claim | 0.5 day |

If we land Tier 1 in 3 weeks, we have one week of slack for Tier 2. Otherwise drop Tier 2 entirely.

---

## 5. Compute and API budget (REVISED 2026-04-29)

Original $300 estimate was wrong by 6×. Real cost per `scripts_phase3/README.md`: 240 dual-judge calls = ~$4. Existing 30-case Phase 3 run already paid; only delta matters.

| Step | Calls | Real $ | Time |
|---|---|---|---|
| T1.1 (15-case × 3 variants × 2 judges) | 90 | ~$1.50 | 20 min wall + rate-gate |
| T1.2 TrialGPT (30 outputs × 2 judges) | 60 | ~$1.00 | 4 h (TrialGPT compute + judging) |
| T1.3 area labeling (LLM-assisted, ~30 calls) | 30 | <$1 | 1 h |
| T1.4 calibration (CPU only, no API) | 0 | $0 | 30 min |
| T1.5 per-gate (5 variants × 30 cases × 4 backbones × 2 judges) | 1200 | ~$20 | ~6 h GPU + ~3 h API |
| T1.6 n=70 expansion (70 × 4 backbones × 2 judges) | 560 | ~$10 | ~3 h GPU + ~2 h API |
| T1.7 paper rewrite (no compute) | 0 | $0 | 5-7 days |
| T2.1–T2.6 (Tier 2, GPU only) | 0 | $0 | 2-3 h GPU total |
| Buffer (retries, errors, re-runs) | — | ~$15 | — |
| **Total delta** | **~1940** | **≈ $50** | ~12 GPU-h + ~8 API-h |

**Budget cap: $75** (covers $50 plan + 50% buffer). Vastly under typical research API budgets. Existing rate gate (8 s inter-call) handles tier-1 limits.

Plan B (single-judge Sonnet only): cuts cost to ~$25 but loses inter-rater agreement story. Don't recommend unless API key constrained.

---

## 6. Timeline (4 weeks)

| Week | Tasks | Deliverable |
|---|---|---|
| 1 | T1.1 + T1.3 + T1.4 (low-effort, fast results) | New tables: 15-case dual-judge κ, per-area heatmap, calibration plots |
| 2 | T1.2 + T1.5 (external SOTA + per-gate ablation) | TrialGPT comparison table; per-gate ablation bar chart |
| 3 | T1.6 (n=100 expansion) + Tier 2 if time | Updated backbone ablation at n=100 |
| 4 | T1.7 paper rewrite + final humanizer pass + cover letter | Submitted PDF |

This is tight. If anything slips, drop T1.6 (keep n=30 as the headline) and ship in 3 weeks. n=30 with rigorous SOTA comparison is preferable to n=100 without.

---

## 7. Paper B forward-reference (so reviewers know what's coming)

The Limitations and Discussion sections explicitly state:

> "Multi-rater clinician audit, multi-turn patient simulation, and a controlled-perturbation operator taxonomy are deferred to a follow-up benchmark paper (Paper B, in preparation). Paper A establishes the system-and-safety baseline; Paper B will provide the controlled empirical benchmark."

This forward-reference does three things: (a) signals that the limitation is acknowledged, (b) commits us to a follow-up that strengthens the line of work rather than hiding from it, (c) lets the npj DM editor see this as a coherent program of research, not a one-off.

---

## 8. Risks and mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| TrialGPT public code drift / dependency hell | Medium | Medium | Pin commit hash; if blocking >1 day, fall back to a simpler rule-based "no-guard RAG" baseline using same retrieval as Paper A, no trial-first selector |
| API rate limits during n=100 dual-judge eval | Medium | Low | Existing rate gate handles this; budget extra wall time |
| BGE-reranker-large doesn't beat MiniLM cross-encoder on this corpus | Low | Low | Report honestly; doesn't affect main result |
| Per-area annotation reveals coverage hole (e.g., 0 cardio cases) | Medium | Medium | Add 5-10 synthetic cardio cases to the n=100 expansion |
| n=100 result diverges from n=30 result | Low | High | Report both; analyze; this is real signal, not noise |
| Reviewer #2 still asks for clinician eval | High | Medium | Cover letter + Limitations explicitly position Paper B as the planned follow-up |

---

## 9. Open questions for user (status 2026-04-29)

1. ~~**API budget cap:** is $300 OK?~~ **RESOLVED.** Real cost ~$50; budget $75. User flagged the over-estimate.
2. **TrialGPT vs PRISM vs both:** Default = TrialGPT only (primary external SOTA). PRISM deferred unless EHR adapter is cheap. **Confirm or override.**
3. ~~**n=100 vs stay at n=30:**~~ **RESOLVED.** User picked n=100. T1.6 in scope.
4. **Email Jin et al. courtesy note?** Optional, no blocker. Default: yes, draft after T1.2 results in hand. **Confirm or skip.**
5. **Title rebrand?** Current 18-word title vs short alt "An Auditable Pipeline for Patient-Facing Clinical Trial Onboarding". Default: switch to short for npj DM. **Confirm or override.**
6. **Audit-rubric mapping (10-dim → 5-dim) vs full re-rate of 15 cases on 5-dim?** Default: mapping (cheaper, 1 hr work, documented in supplementary). **Confirm or override.**

I will proceed with the defaults above unless you override. Defaults flagged in §3 / §4 / §7 of the impl plan.

---

## 10. Sign-off

User confirms:
- §3 Tier 1 list is right.
- §4 Tier 2 priority order is OK.
- §5 budget is acceptable.
- §6 timeline is realistic given other commitments.
- §9 open questions are answered.

After sign-off → write granular impl plan to `outputs/paper_v2/research_plan/2026-04-29-paper-A-polish-plan.md` with exact scripts, paths, commands, and per-task acceptance criteria. Then execute on server task by task.

This spec lives at:
`outputs/paper_v2/research_plan/2026-04-29-paper-A-polish-spec.md`
