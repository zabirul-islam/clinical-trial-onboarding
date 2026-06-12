# Design: npj Revision — Phased Full Strengthening

**Date:** 2026-06-12
**Status:** Approved by author (all 8 sections)
**Driver:** Advisor feedback — paper risks desk rejection as "technical audit with limited clinical validation." Six requested enhancements.

## Goal

Strengthen the clinical-trial onboarding paper (master `outputs/paper_v2/main.tex`/`supplement.tex`, npj derivatives `*_npj.tex`) before npj Digital Medicine submission by adding baseline contrast, claim-level verification, a failure taxonomy, a safety–utility frontier, moderated claims, and a clinical-expert review track.

## Why this design

The paper currently has no contrast: every leak cell is 0.00. Baselines give the missing evidence that cross-trial conflation actually occurs without the guard, and that prompt-only mitigation is insufficient. Human expert review answers the "LLM judges only" objection. The NLI module converts Theorem boundary (iv) from speculation to measurement.

## Constraints and decisions (from author)

- Compute: NVIDIA DGX Spark server, project at `~/Desktop/islamm11/avatar_trial_onboarding`, conda env `avatar_trial`. SSH from this Mac to be tested; fallback = runbook executed by author.
- Baseline matrix: 4 configs × 2 backbones (Qwen-2.5-7B-Instruct, Qwen-2.5-3B-Instruct) × 114 cases = 912 generations.
- Judge budget: semantic dual-judge + 5-dim rubric on all 912 (~3.6k API calls) approved.
- NLI: build the actual redaction module, not measurement-only.
- Expert review: packet ships first; submission waits for and integrates expert results.
- Phasing: Phase 1 packet + text items → Phase 2 compute → Phase 3 integration.

## Components

### 1. Baseline harness — `scripts_phase5/run_baselines.py` (DGX)

Reuses existing retrieval (BM25→cross-encoder), prompt templates, postprocessor, and audit-JSON output format so the existing narrow/wide lexical detectors and `semantic_leak_judge.py` run unchanged.

| ID | Config | Evidence handed to generator | Guard |
|----|--------|------------------------------|-------|
| B1 | Standard multi-trial RAG | top-6 passages across trials | none (selector bypassed) |
| B2 | Prompt-only safeguard | same as B1 | instruction: "use only the single most relevant trial; never mix trials" |
| B3 | Citation-enforced | same as B1 | every field must cite passage IDs; uncited fields dropped (existing postprocessor logic); no single-trial gate |
| B4 | Top-1 selection | passages of the trial owning the rank-1 passage | trivial structural guard; no 5-gate ensemble |

Scoring: lexical detectors on all 912 (deterministic, free); semantic dual-judge (Claude Sonnet 4.5 + GPT-4o, existing blinding protocol) on all; 5-dim rubric dual-judge on all. Outputs persisted in the same per-case JSON layout as phase 4.

Expected result (falsifiable): B1/B2 show non-zero lexical and/or semantic leak; B3 partial; B4 low leak but worse selection quality than the 5-gate ensemble. If B1 shows zero lexical leak, that is itself a reportable finding and the semantic channel comparison becomes primary.

### 2. NLI claim-verification module — `scripts_phase5/nli_redact.py` (DGX)

- Sentence-split patient-facing fields (consent explanation, eligibility narrative, rebuilt summary).
- Premise: accepted-trial passages E*; hypothesis: each sentence; model: DeBERTa-v3-large-MNLI (or equivalent local NLI checkpoint already downloadable on DGX).
- Sentences below entailment threshold are dropped; summary rebuilt from surviving fields; dropped content logged to audit JSON.
- **Threshold calibrated on the 15-case dev pool only** — never on the 12 consensus-flagged cases (design-before-results).
- Counterfactual evaluation: apply to frozen 684 generations + new 912. Report: (a) how many of the 12 consensus semantic leaks are redacted; (b) collateral cost = fraction of benign sentences dropped + pre/post rubric delta on a judge-scored sample; (c) updated Theorem boundary (iv) text.

### 3. Failure taxonomy (text + re-annotation)

Three exclusive categories, formally defined in Methods:

- **T1 cross-trial contamination**: claim supported only by a non-selected trial in the retrieval pool.
- **T2 unsupported clinical completion**: claim supported by no pool trial; plausible protocol-style content generated from parametric memory.
- **T3 ordinary hallucination**: claim false on its face or incoherent with the case.

Re-annotate the 12 consensus-flagged and 104 either-judge-flagged generations; annotate all baseline flags the same way. New table: taxonomy × system × backbone.

### 4. Safety–utility analysis

- New headline frontier figure: leak rate (lexical + semantic) vs utility (judge-pooled overall), points = V-final + B1–B4 per backbone, abstention rate annotated per point.
- Cost-of-abstention analysis: among the 82 abstain-regime cases, fraction where a correct gold-NCT trial existed in the pool (missed-utility rate), from existing case manifest + selector cache. No new inference.
- Reuses the 2,520-tuple threshold sweep for the frontier context.

### 5. Claim moderation (text)

Global pass over master + npj tex: "zero leakage" → "zero detected lexical leakage under the audited conditions" (abstract, contributions, Table 1 caption, theorem discussion, per-area figure caption). Same softening in supplement.

### 6. Expert review packet — `outputs/expert_review/`

- ~30 generations, stratified: accept/abstain regime × semantically-flagged/clean × system (V-final + B1 + B4 for contrast), blinded IDs, randomized order.
- Per-case dossier: patient question, selected-trial passages, full patient-facing response.
- Scoring instrument: 5-dim rubric (mirrors LLM-judge rubric) + binary leak judgment + taxonomy label (T1/T2/T3) + free-text comment. CSV scoring sheets + instructions document.
- Pre-written analysis script: expert-vs-LLM-judge agreement (κ, Spearman), expert leak rate with CIs.
- No patient data; synthetic queries — noted for collaborators' IRB determination.
- Methods subsection written now; Results subsection slot filled when MGH/Yale data returns.

### 7. Paper integration

- Master tex first (maximal), npj derivatives trimmed second (per project rule: master/derivative model).
- New Results subsections: baseline comparison; NLI mitigation; taxonomy; expert validation (pending data).
- Methods additions: baseline configs, NLI module, taxonomy definitions, expert protocol.
- Abstract + contributions rewritten around: defined patient-facing safety problem + structural control + audited baselines + claim verification + human validation.

### 8. Execution mechanics and rails

- Test SSH to DGX from this Mac; verify server `outputs/` matches local before any run (case manifest hash, generation counts). Fallback: runbook with exact commands for author.
- Smoke-test each baseline config on 3 cases before the full 114-case sweep.
- All reported numbers derive from persisted JSON via scripts (reproducibility gate).
- Phase order: (1) expert packet + text items → (2) DGX compute → (3) annotation + analysis + integration → submit after expert data lands.

## Error handling

- Baseline parse failures: same degradation path as existing pipeline (fallback/abstain), reported as rates — not silently dropped.
- Judge API failures: existing retry/rate-gate logic in `semantic_leak_judge.py`; refusals logged and reported as in phase 4.
- NLI module: unit tests on hand-built entail/neutral/contradict sentence pairs before use.
- Server/local mismatch: hard stop + reconcile before generating anything.

## Testing

- Smoke runs (3 cases) per baseline config, manually inspected.
- NLI unit tests + threshold calibration report on dev pool.
- Figure scripts re-run from persisted CSV/JSON only.
- Final compile + forbidden-string sweep (existing checklist: PLACEHOLDER / prod / reviewer / etc.).

## Out of scope

- Recruiting expert reviewers (author + advisor handle MGH/Yale contacts).
- Closed-API baselines for B1–B4 (open-weight only, matching the wide-detector constraint).
- New retrieval methods, new case pools, clinical deployment claims.
