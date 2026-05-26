# Referee Report — Round 2 (Re-review)

**Manuscript:** *Structural Safety in Patient-Facing AI for Clinical Trial Onboarding: An Audited Grounded Pipeline*
**Target venue:** npj Digital Medicine
**Round:** 2 of up to 3
**Recommendation:** **Accept**

---

## Summary of Round-2 Re-review

Round 1 returned Minor Revision with five Required Revisions, all targeting positioning, scoping, and honesty rather than new experiments. The authors have implemented every one of them. Layered on top, the present Round-2 pass applied a coherent set of style and layout fixes (float placement discipline, sans-serif-to-emph conversion for conceptual terms, removal of the all-zero Figure 2, citation tidying around the kappa paradox literature, and replacement of an unanchored 0.87 number with a properly supplementary-cited qualitative claim). The result is a manuscript that has tightened on four of five scoring dimensions while leaving the only floor-level dimension (data quality) unchanged for reasons that are irreducible.

## (a) What Improved

**Methodological credibility (8 → 9).** Theorem 1 has been renamed *Conditional sufficiency for zero leak*, and the proof sketch now explicitly partitions the four leak channels into three that close deterministically under (C1)–(C3) and one (the parametric-memory channel) that closes only empirically at 0/684 on the audited pool. An alternate constructive-redaction variant is sketched as a path to fully deterministic sufficiency. This was the most pointed Round-1 criticism, and the fix is exactly the right one: it preserves the strength of the result while removing the category error of mixing deterministic and empirical claims in a single theorem.

**Novelty (7 → 8).** The Introduction now contains a standalone paragraph stating that no published trial-matching system (DeepEnroll, COMPOSE, TrialGPT, PRISM, Trial2Vec) reports a cross-trial leak rate, and explicitly frames cross-trial conflation as a *standard safety dimension* for the patient-facing trial-onboarding literature. The abstract foregrounds the 3.2% stem-clustered Clopper–Pearson upper bound. Both fixes move the contribution from "different output type" to "previously unmeasured failure mode," which is the framing the work deserves.

**Clinical and policy relevance (8 → 9).** The negative Brier Skill Score is no longer framed defensively. The abstract and §2.7 now read it as a deployment-safety contribution — *do not use τ as a per-case triage probability* — which is the actionable form. Contribution 2 has been qualified to "leak channel only," with explicit acknowledgement that Mistral-7B's 11% commit rate means utility is not backbone-invariant. This is the kind of scoping precision that survives editor and reader scrutiny.

**Clarity (8 → 9).** Three style/layout interventions compound:
1. `[t]` → `[!htbp]` + `float` + `placeins` with auto-`\FloatBarrier` at section breaks fixes the figure-drift that a 20+-page guarded-pipeline manuscript inevitably accumulates;
2. ten conceptual terms (`LM`, `PMIE`, `LMM`, `CD`, *cannot-determine*, *likely-match*, etc.) moved from sans-serif code-font to `\emph{}`, with legitimate code literals (`NCT`, `blind_id`) correctly retained in `\texttt{}` — this is the correct typographic discipline and reduces visual noise across every page;
3. the all-zero Figure 2 leak grid is demoted to Supplementary §5; the main paper now leads its Results visualization with the per-area heatmap, which carries actual within-cell variance.

## (b) What Is Still Weak

**Data quality (6, unchanged).** n=114 cases remain the floor. The stem-clustered Clopper–Pearson upper bound of ~3.2% is now properly placed in the abstract, so a reader cannot miss the limitation, but the limitation itself cannot be revised away without new data. A larger second-cohort replication would lift this dimension to 7 or 8; that work is deferred and is acknowledged as deferred. This is appropriate for the early-stage DECIDE-AI tier the paper explicitly claims.

**Three minor residuals**, none of which threaten the verdict:
- The DeepEnroll/COMPOSE/TrialGPT/PRISM/Trial2Vec novelty contrast appears in both the Introduction and the Discussion; one of the two occurrences could be compressed to a single sentence.
- The replacement of the unanchored 0.87 threshold-agreement number with qualitative "high across the open-weight cohort (Supplementary Table~S8)" loses a small amount of main-text precision in exchange for honesty. Acceptable trade.
- The four-role deployment workflow (Fig. 5) could be supplemented by a textual role × action × audit-artifact table; non-blocking.

## (c) Regressions Checked

- **Figure 2 demotion** — no regression. The demoted figure had zero within-cell variance and conveyed no information beyond what Table 1 already states. The per-area heatmap that now anchors §2.1 carries real variance in commit and parse-ok rates.
- **Supplement self-bibliography strip** — no regression. The Cohen 1960, Landis & Koch 1977, and Gwet 2014 kappa-paradox citations remain in the main bibliography and are correctly placed at the IRR discussion.
- **0.87 number removal** — minor main-text precision loss but the supplementary anchor (Table S8) is properly cited. No methodological regression.
- **`\textsf{}` → `\emph{}` conversion** — no regression. Code literals correctly retained in `\texttt{}`; only conceptual terms were converted.

## Overall Verdict

**Accept.** Round 1's five Required Revisions are all implemented, and the Round-2 style/layout pass compounds to a real clarity improvement. The manuscript now scores at or above 8 on every dimension except data quality, which sits at 6 because of an irreducible n constraint that the abstract now foregrounds honestly. The remaining residual issues are author-elective polish, not editor-required revisions. The work makes a scoped, credible, and now properly positioned contribution to the patient-facing clinical-AI literature; the cross-trial conflation construct is well-suited to become a standard safety dimension as the authors propose. I recommend acceptance without further revision.

*Round-2 scores: novelty=8, methodological_credibility=9, data_quality=6, clarity=9, clinical_policy_relevance=9. Deltas vs. Round 1: +1, +1, 0, +1, +1. Approx. word count: 720.*
