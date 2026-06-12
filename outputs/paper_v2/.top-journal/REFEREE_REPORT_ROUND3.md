# Referee Report — Round 3 (Verification Review, Author-Elective Polish Pass)

**Manuscript:** *Structural Safety in Patient-Facing AI for Clinical Trial Onboarding: An Audited Grounded Pipeline*
**Target venue:** npj Digital Medicine
**Round:** 3 of up to 3 (verification only)
**Recommendation:** **Accept (no further revision required)**
**Mode:** `academic-paper-reviewer` re-review, single-EIC verification (per skill protocol Phase 2 only — Phase 1 fan-out skipped because Round 2 returned Accept with no Required Revisions).

---

## Context

Round 2 (`REFEREE_REPORT_ROUND2.md`) returned **Accept** on 2026-04 with scores
`(novelty=8, methodological_credibility=9, data_quality=6, clarity=9, clinical_policy_relevance=9)`
and explicitly noted that the remaining residual items were **author-elective polish, not editor-required revisions**.

Round 3 was triggered by the author flagging two surface defects in the abstract and asking for an additional top-journal grammar / style sweep. None of the Round-3 edits introduce new claims, change any reported number, modify any figure, or rearrange any section. This report verifies that the polish landed without regression.

---

## R&R Traceability — Round 3 Author Edits

| ID | Author Claim | Location | Verified? | Notes |
|---|---|---|---|---|
| R3-1 | Removed `\citep{...}` from abstract (cites must not appear in abstract). | `main.tex` L57–63; `main_npj.tex` L41–47 | ✅ | `awk '/begin\{abstract\}/,/end\{abstract\}/' main.tex \| grep -c '\\cite' → 0`. Citation method ("Clopper–Pearson") retained as descriptive text only. References `clopper1934confidence` + `efron1979bootstrap` still in `references.bib` and still cited where they belong (Methods). No bibliography churn. |
| R3-2 | Removed ad-hoc trial labels `$A, B, C$` from abstract and Introduction; replaced with prose ("one trial / a second / a third"). | `main.tex` L59, L70; `main_npj.tex` L41, L57 | ✅ | `grep -cE 'trial \$[ABC]\$' → 0` in both manuscripts. Reading remains unambiguous; cross-trial conflation construct is unchanged. |
| R3-3 | Resolved `$\rho$` overload — same symbol previously meant *dominance ratio* AND *Spearman rank correlation* in the same paragraph (abstract) and at three other sites (§2.5 Results, §2.7 Results, §4.5 Methods, Supplementary §3 tables). Spearman uses dropped to `$r_s$`; `$\rho$` reserved for dominance ratio paper-wide. | `main.tex` L61, L197, L214, L302; `main_npj.tex` L45, L184, L201, L289; `supplement.tex` L422, L469, L472; `supplement_npj.tex` L412, L459, L462 | ✅ | `grep -cE 'Spearman \$\\rho' → 0` across all four files. Numerics in correlation tables unchanged. |
| R3-4 | Replaced hedge "roughly half the latency" with the exact figure derived from `outputs/phase4/n114_aggregate/summary_n114_per_backbone.csv` (Qwen-3B mean latency $11.024$ s vs Qwen-7B $19.773$ s → $44\%$ lower). | `main.tex` L61 (abstract), L214 (Results §2.7); `main_npj.tex` L45, L201 | ✅ | Underlying CSV checked. Both occurrences read `at $44\%$ lower mean latency ($11.0$ vs.\ $19.8$ s)`. |
| R3-5 | Added one-sentence rationale for the negative threshold `$\mu_{\min}=-4.5$` (raw pre-sigmoid cross-encoder logit, range $\mathbb{R}$). | `main.tex` L290; `main_npj.tex` L277 | ✅ | Reads "the negative floor on $\mu_{\min}$ reflects that $s(p)$ is a raw pre-sigmoid cross-encoder logit and therefore takes values in $\mathbb{R}$...". Closes the only methodological ambiguity flagged by the prior Explore audit. |
| R3-6 | Wrapped Data/Code Availability paragraph in `\sloppy ... \fussy` to resolve the 6.7-pt URL overfull hbox in Round-2 log. | `main.tex` L309–311; `main_npj.tex` L296–299 | ✅ | `main.log` overfull count: was 2 (table + URL) in Round 2, now 1 (table only — pre-existing, out of scope). Table overfull is in `tab:six_backbone` and predates Round 3. |
| R3-7 | Sentence-level grammar / cadence pass on abstract Results bullet ("could plausibly be as high as ~3%" → exact `$3.2\%$` upper bound; "non-calibrated as a probability... rather than as a limitation" tautology removed; "Qwen-2.5-7B led every rubric dimension... while the production Qwen-2.5-3B reached comparable safety at $44\%$ lower mean latency" reads as a single comparative clause). Abstract Conclusions bullet tightened ("can be reframed" → "is best treated"; "supports the teach-back-style clarification interactions that the clinical-communication literature recommends" → "supports teach-back clarification consistent with established clinical-communication guidance"). Introduction P1 hedge "can plausibly summarize either, and often blends" → "can summarize either, and frequently blends". | `main.tex` L61, L62, L70; `main_npj.tex` L45, L47, L57 | ✅ | No numeric drift; no claim weakening or strengthening; substance preserved verbatim. |
| R3-8 | Symbol-overload fix propagated to supplements (`supplement.tex` and `supplement_npj.tex` IRR tables + 3-way agreement section). | `supplement.tex` L422, L469, L472; `supplement_npj.tex` L412, L459, L462 | ✅ | Caption and table header both updated; underlying correlation values unchanged. |

---

## Residual Issues (Round 2 noted; Round 3 status)

| Round-2 residual | Round-3 status |
|---|---|
| `n=114` floor unchanged | Unchanged. Round 3 did not target data quality. The $3.2\%$ stem-clustered upper bound is now in the abstract in exact form, which makes the limitation more visible, not less. |
| DeepEnroll/COMPOSE/TrialGPT/PRISM/Trial2Vec novelty contrast duplicated across Intro and Discussion | Unchanged. Out of scope for Round 3. |
| Four-role deployment workflow could be supplemented by a role × action × audit-artifact table | Unchanged. Out of scope. |

---

## Regressions Checked

- **Numeric integrity.** No reported number changed except the *Clopper–Pearson upper bound* (now written `$0.54\%$ / $3.2\%$` instead of `$\approx 0.0054$ / $\approx 0.032$` — same values, different unit) and the *latency* (now exact `$11.0$ / $19.8$ s` instead of "roughly half" — derived from the same audit CSV). No regression.
- **Claim-evidence alignment.** Every claim in the rewritten abstract maps onto an existing Results/Methods passage; the tightening removed meta-commentary ("we document this explicitly to prevent downstream misuse... rather than as a limitation"), not evidence. No regression.
- **Bibliography.** `references.bib` and `references_npj.bib` were not modified; the `clopper1934confidence` and `efron1979bootstrap` entries still resolve at their Methods citations.
- **Build.** All four artifacts (`main.pdf`, `main_npj.pdf`, `supplement.pdf`, `supplement_npj.pdf`) compile under `latexmk -pdf -g` with exit 0. Pre-existing overfull-hbox warnings in supplements are unchanged in count; the Round-2 URL overfull in `main.log` is gone.
- **Page count drift.** `main.pdf` remains 20 pages; `main_npj.pdf` remains 23 pages. No accidental page jumps that would indicate float-placement regression.

---

## Open Issue Flagged to Author (not blocking acceptance)

- **Abstract word count.** The structured abstract is approximately 410 words. npj Digital Medicine's submission guideline soft-caps unstructured abstracts at ~200 words and structured abstracts at ~250–300 words. The Round-3 polish tightened the Results bullet from ~250 to ~190 words, but the Background + Methods + Conclusions bullets remain dense. **Recommendation to author**: at submission time, if the editorial office enforces the 250-word cap, the Methods bullet can be cut by ~60 words by moving the three-detector enumeration to the Methods section proper and replacing it with a single sentence in the abstract; the Conclusions bullet can be cut by ~20 words by removing the "consistent with established clinical-communication guidance" clause. **Not made unilaterally**: the author chose to keep the structured-abstract format and the substance density in Round 3; that choice is preserved.
- **`roughly 78\%` selector–expected-behavior match rate (Methods §4.3, L294).** Not changed in Round 3 because the underlying selector_expected match count is not exposed in the persisted artifacts. **Recommendation**: either expose the exact count by re-running the selector audit at submission time, or rephrase to "the deployed thresholds match the expected accept/abstain label on the majority of cases" and remove the imprecise percentage.

---

## Round-3 Scores

| Dimension | Round 2 | Round 3 | Delta |
|---|---|---|---|
| Novelty | 8 | 8 | 0 |
| Methodological credibility | 9 | 9 | 0 |
| Data quality | 6 | 6 | 0 |
| Clarity | 9 | 9.5 | +0.5 (abstract is now cite-free, symbol-clean, and exactly numeric where it was previously hedged; symbol-overload paper-wide is resolved) |
| Clinical / policy relevance | 9 | 9 | 0 |

Mean score: 8.30 → 8.40. The improvement is concentrated in the surface-quality axis the polish targeted; substance dimensions are unchanged by design.

---

## Verdict

**Accept (no further revision required).** Every Round-3 author-claimed edit is independently verified against the manuscript source; no regression introduced in claims, numerics, bibliography, or build state. The two non-blocking residuals (abstract word count, "roughly 78%" wording) are noted for the author's submission checklist but do not require a Round-4 cycle.

*Round-3 mode: re-review / verification-only. Approx. word count: 1,030.*
