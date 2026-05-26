# Referee Report — Round 1

**Manuscript:** *Structural Safety in Patient-Facing AI for Clinical Trial Onboarding: An Audited Grounded Pipeline*
**Target venue:** npj Digital Medicine
**Recommendation:** Minor Revision

---

## Summary

The authors address a specific, plausible, and under-studied safety hazard in patient-facing clinical-trial onboarding AI: *cross-trial conflation*, where one generated response mixes eligibility, procedures, and time commitments across distinct trials. They frame it as a property of the evidence layer rather than the generator, and instantiate the framing as a five-condition selector ensemble (dominance ratio rho, score share tau, raw cross-encoder max mu, best rank kappa, plus a generic-question flag) that either accepts a single dominant trial or abstains with a narrowing prompt. Downstream eligibility triage, consent-style explanation, and teach-back all draw from that one accepted trial. The empirical claim is striking and tightly scoped: on 684 generations (114 cases x 6 backbones), the cross-trial leak rate is 0 under two pre-specified detectors, across both selector regimes and all five leave-one-out per-gate ablations. The paper also reports a deterministic sufficiency theorem with three boundary conditions, a transparent negative calibration result for tau, a dual-judge LLM evaluation with appropriate blinding, a 3-rater Krippendorff alpha analysis, and a serious effort at reporting standards (CLAIM, TRIPOD+AI, CONSORT-AI/SPIRIT-AI, DECIDE-AI, FDA SaMD).

This is a careful paper. The framing is novel enough, the methodology is largely defensible, the limitations are unusually honest, and the policy/clinical scaffolding is in place. My principal concerns are about positioning, the meaning of "backbone-invariance" in the presence of Mistral's collapse, and the modest sample size relative to the strength of the headline claim.

---

## Major Comments

**1. Novelty: the framing is real, but the contribution must be sharpened.**
The reframing of conflation as an evidence-layer rather than generator-layer problem is genuinely useful, and to my knowledge no published trial-matching system (DeepEnroll, COMPOSE, TrialGPT, PRISM, Trial2Vec) reports a cross-trial leak rate. Table 3 acknowledges this qualitatively but the prose underclaims. The Introduction should state plainly: prior trial-matching systems optimize per-(patient, trial) match accuracy on single trial-patient pairs; the multi-trial *consistency* failure mode is invisible to that objective. This is the gap. Currently the contribution reads as "we tackle a different output type" — true but undersold. The cross-trial conflation construct could become a standard safety dimension for this literature; please write it as such.

**2. Theorem 1 mixes deterministic and empirical channels.**
The theorem statement claims deterministic sufficiency under (C1)-(C3). The proof sketch then concedes that "the only remaining channel is direct generation of a non-t-star identifier from parametric memory, which the wide-leak detector tests for; its empirical rate is 0/684". This is a category error: a deterministic sufficiency claim cannot lean on an empirical residual rate. Either close that channel constructively (e.g., post-hoc redaction of any NCT pattern not in the accepted evidence) and restate as fully deterministic, or rename the result as "Conditional Sufficiency" and acknowledge that one channel is closed empirically. The honest version is still a strong result; the current wording overclaims.

**3. "Backbone-invariance" interacts poorly with Mistral's commit-rate collapse.**
Mistral-7B has parse-ok 0.12 and commit 0.11 (Table 1). With 89% of Mistral outputs failing to parse or commit, "zero leak across all backbones" is partly a statement about the postprocessor's ability to convert non-parsing outputs into safe abstentions, not about Mistral's generative behavior per se. The paper later acknowledges this in the Discussion ("non-parsing answers degrade to abstention or fallback, not to leak"), which is fair. But Contribution 2 in the Introduction still asserts backbone-invariance of *safety*, full stop. Please qualify this: the leak channel is invariant; the commit channel is not, and the conditional leak rate among *committed* Mistral outputs is the relevant comparison. With 11% commit rate, the effective n for Mistral's safety claim is ~12-13 generations, which is much weaker than 114.

**4. n=114 is at the floor of defensibility for the headline claim.**
The Clopper-Pearson upper bound is reported honestly: 0.0054 pooled, 0.032 stem-clustered. The stem-clustered bound — meaning the true leak rate could be as high as ~1 in 30 cases — is the right denominator for a clinical safety claim, since cases are correlated within therapeutic area and within selector regime. This belongs in the Abstract, not buried in Methods. A reader who absorbs only "0/684, zero leak" will misweight the evidence. I would strongly recommend rewriting the Results-line of the abstract to read something like "zero observed cross-trial leakage; Clopper-Pearson 95% upper bound 3.2% under stem-clustering."

**5. The IRR ceiling explanation is partially correct but masks a subtler issue.**
The paper attributes low Cohen's kappa on the safety dimension to ceiling saturation: when most scores are 4-5, kappa's variance-dependent denominator collapses. Mechanically this is right. However, Supplementary Table S8 shows safety has kappa_lin = 0.032 across all n=120 paired scores, even though the safety means (4.14 for Qwen-3B, 4.63 for Qwen-7B, 4.05 for Mistral) are *not* at the absolute ceiling of 5. There is residual variance — and the agreement on that residual variance is essentially chance. This may indicate that judges' disagreement is concentrated on borderline cases (3-4 vs 5) and is not purely a saturation artifact. Please add a 2x2 confusion table of (judge1 >= 4, judge2 >= 4) to test whether agreement is high above the threshold even when kappa is low overall. This is the standard fix for the "kappa paradox" in clinical-AI evaluation and would strengthen the paper considerably.

---

## Minor Comments

- **Calibration framing.** The negative Brier Skill Score is a strength when read carefully — the authors document *why not to use tau as a probability* — but the abstract sentence "Calibration of the selector's score share was negative on held-out targets and is reported as a transparent limitation" lands defensively. Reframe: the selector is binary-by-design; documenting tau's non-calibration prevents downstream misuse. This is contribution, not limitation.

- **ECE present but understated.** Supplementary Table S13 reports ECE. Main text Section 2.7 mentions only Brier and BSS. Add ECE to the main-paper sentence for completeness.

- **Equity discussion is good but disease-stratified only.** The therapeutic-area heatmap is fine. The Limitations section names under-represented populations (pediatrics, rare disease, non-English) but does not stratify the existing case pool by, e.g., query length, syntactic complexity, or health-literacy proxy. A one-paragraph acknowledgment that the n=114 pool is English-only and constructed from TREC topics (themselves not stratified for health literacy) would close the loop.

- **Threat model is implicit.** The per-gate ablation tests robustness to gate removal. It does not test adversarial inputs (a user crafting a query to force conflation). One sentence stating that adversarial prompting is out-of-scope for this audit — or that the gate ensemble's input-independence makes it adversarially robust by construction — would clarify the scope.

- **Figure 2 (leak grid) carries no information.** Every cell is zero. Consider promoting the per-gate ablation table (Table 2) to a figure, since it actually shows variation across variants. The all-zero grid could go to supplementary.

- **Deployment workflow figure (Fig. 5).** A textual companion (role x action x audit artifact table) would help readers parse the four-role workflow.

- **"Backbone-invariance within the audited cohort"** appears multiple times. Compress the hedge into one consistent phrase used throughout — currently the strength of the hedge varies between Abstract, Introduction, and Discussion.

- **DOI/repo placeholders.** Both [REPO-PLACEHOLDER] and [GRANT-PLACEHOLDER] are present in the production tex. Standard for pre-acceptance, but flag to editorial.

- **Author count and contributions are appropriate** for the work described; the contributions statement is properly scoped.

---

## Checklist Verification

| Item | Status | Note |
|------|--------|------|
| TRIPOD+AI | Addressed | Cited in Introduction; reporting style consistent |
| CLAIM | Addressed | Cited in Introduction |
| FDA SaMD | Addressed | Class II SaMD positioning in Introduction; deployment caveats in Discussion |
| DECIDE-AI | Addressed | Early-stage clinical-AI framing cited |
| Calibration | Addressed | Brier + BSS in main text; ECE in supplement; negative result transparently reported |
| Fairness | Addressed (partial) | Therapeutic-area stratification present; demographic stratification absent and acknowledged as limitation |
| Reproducibility | Addressed | Zenodo + GitHub commitment; placeholders standard pre-acceptance |
| Equity | Addressed | Under-represented populations discussed; English-only scope could be more explicit |

---

## Recommendation

**Minor Revision.** This paper makes a real, scoped, and credible contribution. The headline result is striking; the empirical bound is honest; the methodology choices around blinding, dual-judge evaluation, per-gate ablation, and 3-way IRR are appropriate; reporting-standard hygiene is unusually good. The required revisions are positioning and scoping — sharpening the novelty paragraph, qualifying backbone-invariance, foregrounding the stem-clustered upper bound, and tightening Theorem 1's deterministic-vs-empirical mixing. None of these require new experiments. I recommend the editors accept conditional on these revisions and would be willing to re-review.

The paper would be stronger with a small clinician-rater audit on a subsample, but the authors flag this as deferred future work and I do not consider it blocking for this venue at the early-stage clinical-AI tier the paper occupies.

*Approx. word count: 1,420.*
