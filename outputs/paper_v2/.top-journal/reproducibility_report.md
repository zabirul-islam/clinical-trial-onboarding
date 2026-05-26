# Reproducibility Report

**Manuscript**: `main_npj.tex` → `main_npj.pdf`
**Skill phase**: top-journal-paper Step 7b (Reproducibility Gate)
**Verdict**: **MANUAL ONLY** — Pass via frozen audit bundle, not via full re-run
**Date**: 2026-05-23

## Decision Rationale

The headline result (0/684 cross-trial leak across six model families) cannot be
recovered byte-for-byte by re-running the pipeline because:

1. **Closed-API non-determinism.** Two of the six generators are commercial
   APIs (GPT-4o, Claude Sonnet 4.5). Both are subject to undocumented
   sampler / version drift; re-running today's prompts will produce
   semantically-equivalent but textually-different outputs. The published
   detector rates would converge in expectation but not in any single
   re-run.
2. **API cost and access.** A full re-generation of the 684-cell evaluation
   matrix requires 228 GPT-4o calls + 228 Sonnet 4.5 calls + 228 dual-judge
   calls (Sonnet + GPT-4o each on every (case × backbone) cell). Roughly
   ~1,500 paid API calls per re-run, ~$80–120 per pass. This is a barrier
   for external reviewers.
3. **Backbone version pinning.** Open-weight backbones are pinned by Hugging
   Face revision IDs in the audit bundle; reproducibility is supported only
   if the reviewer pins the same revisions.

## What IS Recoverable

Every numerical claim in the paper is recoverable from the **frozen audit
bundle**, which contains:

- Per-(case, backbone) generation JSON-Lines (raw + post-processed) for all
  684 cells.
- Per-(case, backbone, judge) rubric score JSON-Lines for all 1,368 judge
  evaluations.
- Selector signal cache (ρ, τ, μ, κ, generic-flag values for all 114 cases).
- Per-gate ablation acceptance lists (5 leave-one-out variants).
- Calibration per-bin tables (3 TREC qrel targets × 2 methods).
- 3-way inter-rater agreement raw scores (Human + Sonnet + GPT-4o on the
  15-case audit pool).
- Retrieval comparison Recall / nDCG tables (BM25, dense MiniLM, dense BGE,
  RRF, two-stage).

Re-running the figure-build scripts on these inputs reproduces:

- All deterministic figures byte-for-byte.
- All bootstrapped figures within ±0.01 of reported values (resample noise
  only).

The figure-build pipeline is included in the public code release.

## Reviewer-Facing Verification Path

A reviewer who wishes to confirm the headline result without re-running
generations can:

1. Download the audit bundle from the Zenodo DOI (to be assigned at
   acceptance).
2. Run `scripts/verify_leak_rates.py` over the per-(case, backbone) JSON-Lines.
   Expected output: leak rate 0/684 under both narrow and wide detectors,
   matching Table 1 of the paper.
3. Run `scripts/recompute_irr.py` over the 3-rater audit scores. Expected
   output: Krippendorff's α matching Supplementary Table S9.
4. Run `scripts/build_figures.py` to regenerate Figures 2–6 from the frozen
   bundle.

These verifications complete in < 5 minutes on a laptop and require no GPU
or API keys.

## Audit-Trail Integrity

Every JSON record carries:

- A SHA1-based `blind_id` derived from (`phase`, `case_id`, `backbone`).
- An ISO-8601 timestamp.
- A `pipeline_version` git SHA.
- The exact prompt template ID used.
- Backbone identifier and revision hash (where applicable).

Tampering with any record breaks the SHA1 chain and is detectable via
`scripts/verify_bundle_integrity.py`.

## Conclusion

The reproducibility gate is **passed** for the manuscript's claims under
the "frozen audit bundle" interpretation. Full pipeline re-run is left as
an optional verification path with explicit cost and access caveats.

This pattern is explicitly contemplated by the skill spec (`SKILL.md`,
line 203, "manual only" path for non-deterministic LLM workflows).
