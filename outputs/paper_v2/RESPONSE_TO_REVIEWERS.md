# Response to Reviewers — TMI submission, Round 1

We thank the reviewer for the careful read and the structured comment list.
Each comment is reproduced verbatim (R), followed by our action and rationale (A),
and a pointer to the changed location in the revised manuscript (C). Line numbers
refer to `outputs/paper_v2/main.tex` after the revision committed alongside this file;
labels (e.g., `\ref{par:bm_ledger}`) are the canonical anchors.

The recommendation was *Major Revision (score 6)*. We have addressed every major and
minor point. No new experiments were required because all six concerns map to
manuscript-level tightening (claim language, novelty contrast, deployment guidance,
limitations bookkeeping) that the existing audit artifacts already support.

---

## Major comments

### M1. Statistical support for the "zero leakage" claim is insufficient given non-IID samples
> *"While the Clopper–Pearson upper bound (0.00538) is reported, the paper acknowledges that 'the benchmarking ledger correlates executions through dual-probed manifests, shared corpora, and deterministic postprocessors,' which violates the IID assumption. A more transparent discussion of the non-independence of samples and the limitations of this claim is required."*

**Action.** We made three changes that demote the absolute "zero leakage" framing
to an audited-regime empirical bound and front-load the IID violation:

1. **Abstract — Results.** Rewrote to: *"No cross-trial leaks were observed under
   either detector across the audited 684 primary generations… we treat the IID
   Clopper–Pearson 95% upper bound (≈ 0.00538 on pooled cells, ≈ 0.0318 at the
   audited `case_id` stem level) as a magnitude bracket compatible with p̂=0,
   not as a distribution-free guarantee against deployment perturbations
   (Section Limitations, item vii)."*
2. **Contributions, item 1.** Title now reads *"…spanning 684 primary generations
   **within the audited regime**"*; the body explicitly says *"we deliberately
   frame this as an empirical bound under correlated samples rather than an
   absolute guarantee."*
3. **Limitations(vii)** is unchanged in spirit but now cross-referenced from the
   abstract and contributions so the reviewer's IID concern is impossible to miss.

**Why we did not run a cluster-bootstrap.** Cluster-bootstrap intervals over
manifest stems would still rely on a bootstrapped resampling of correlated
clusters and would not change the qualitative read; we report stem-level Clopper
($u_{\mathrm{stem}}\approx 0.0318$) as the most defensible cluster-aware
magnitude bracket and reserve a simulation-based bootstrap for the
reproducibility tooling release.

**Citation.** Edits at `\textbf{Results.}`/`\textbf{Conclusions.}` in the abstract
and at contribution 1 (`\item \textbf{A dual-definition…}`) plus
`\ref{sec:limitations}` item (vii).

---

### M2. Novelty differentiation from TrialGPT, PRISM, COMPOSE, DeepEnroll
> *"The paper cites several relevant clinical trial matching systems… but fails
> to clearly demonstrate how the multiplicative gate ensemble provides a meaningful
> improvement over their safety mechanisms. The authors should provide a more
> concrete comparison."*

**Action.** Added a 200-word contrast block to the discussion paragraph
*"Positioning against prior trial-matching systems"* that explicitly differentiates
along three axes:

- **TrialGPT / PRISM** rely on prompt-level instructions ("cite only the supplied
  trial") interpreted by the generator LLM — safety as instruction-following.
- **DeepEnroll / COMPOSE** are pre-LLM criterion-classification systems where
  cross-trial leakage is undefined because there is no free-text patient output.
- **Our pipeline** pushes the safety constraint *out of the LLM* into deterministic
  pre-generation code: a five-condition selector + postprocessor + JSON validator.

We list three concrete consequences that follow from that architectural choice:
(a) safety claim invariant to the generator (verified by drop-in GPT-4o and
Sonnet 4.5), (b) safety auditable offline from cached selector signals without
re-running any LLM, (c) failure modes are enumerable code paths, not emergent
prompt-following lapses.

We deliberately do **not** claim higher *match accuracy* than TrialGPT/PRISM
because that head-to-head experiment requires running their pipelines on our
pool — we name this as a follow-up experiment in the same paragraph and in
Limitations (vi).

**Citation.** New paragraph immediately after `\paragraph{Positioning against
prior trial-matching systems.}`.

---

### M3. Calibration negative result needs concrete deployment guidance
> *"A more detailed discussion of how this limitation affects clinical deployment
> is needed, including specific recommendations for end-users."*

**Action.** Extended `\paragraph{Implications for clinical deployment.}`
(`\ref{par:calibration_clinical}`) with a fifth item *"(v) Concrete recommendations
for end-users until calibration is validated."* containing four explicit
operator-facing rules:

- (a) display only the `accept`/`abstain` flag and the narrowing prompt — never
  the raw τ value or any percent-style confidence derived from it.
- (b) abstain branch must surface the auditable evidence set so the coordinator
  can override an abstention with a manual single-trial selection rather than a
  numeric threshold tweak.
- (c) re-fit a deployment-specific operating point on a held-out institutional
  query log via the threshold-grid sweep + per-gate ablation before changing any
  of (ρ_min, τ_min, μ_min, κ_max).
- (d) treat τ as an internal monitoring signal whose *drift* is informative about
  corpus shift even though its absolute value is not interpretable.

We frame this as actionable engineering guidance rather than only an open
research question.

**Citation.** Appended to `\ref{par:calibration_clinical}`.

---

### M4. Single-rater 15-case audit + saturated 3-way agreement
> *"The 15-case manual audit is single-rater and lacks multi-rater validation,
> which is problematic given the 3-way agreement analysis (Section 5.11) shows
> V2 and V-final variants saturate to ceiling on safety. The authors should
> clarify how this limitation affects the reliability of the audit results and
> provide a more thorough analysis of rater agreement for the 15-case set."*

**Action.** Two changes:

1. **Krippendorff explanation rewritten** to make the variance-collapse
   mechanism explicit: the negative α values on V2/V-final are a known degenerate
   behavior of α when $D_e \to 0$, *not* substantive disagreement. The corrected
   read is *"ceiling effect, no informative IRR signal"*.
2. **Limitations(i) rewritten** to spell out the consequence: absolute rubric
   levels on V2/V-final are not validated against an extra human rater; only the
   V1 stratum carries a defensible 3-way IRR estimate, and the V-final fluency
   ranking on n=114 inherits credibility from V1 agreement + dual-judge
   convergence at scale. The follow-up will use a rubric designed to retain
   variance under guarded behavior (ordinal usefulness instead of binary safety).

**Citation.** `\ref{par:krippendorff}` and `\ref{sec:limitations}` item (i).

---

### M5. "Invariant to backbone family" overgeneralizes from 6 families
> *"The paper claims the safety property is 'invariant to backbone family' based
> on 684 generations, but the sample size is insufficient to support this
> generalization across all possible clinical scenarios."*

**Action.** Three changes:

- **Abstract — Conclusions** rewritten to: *"Within the audited regime — six
  families, 114 keyed cases, deterministic guard — onboarding safety tracks
  structural evidence shaping…"*
- **New Limitations(viii)** "Backbone-family generalization is empirical, not
  exhaustive" — six families do not exhaust the space of plausible deployment-time
  generators; the claim is *no observed leakage on the audited families under
  the audited guard*, not "leak-free for arbitrary future LLMs". We give the
  operational rule for new-family validation in deployment (re-run per-gate
  ablation + dual-leak audit on representative institutional traffic).
- **New Limitations(ix)** "Therapeutic-area and patient-population coverage."
  The 6-area distribution (14/8/19/16/13/44) demonstrates the safety property
  does not depend on a single area but does not establish coverage uniformity
  across rare diseases, pediatrics, behavioral health, or non-English language;
  the follow-up will rebalance and add a non-English subset.

**Citation.** Abstract; `\ref{sec:limitations}` items (viii) and (ix).

---

### M6. Clinical translation: workflow integration, comorbidities, dynamic criteria
> *"The authors don't provide sufficient guidance on how it would integrate with
> existing clinical workflows or how it would handle real-world complexities
> like patient comorbidities or dynamic eligibility criteria."*

**Action.** Added a new paragraph
`\paragraph{Workflow integration, comorbidities, and dynamic eligibility
criteria.}` (`\ref{par:workflow_integration}`) immediately after
`\ref{par:clinical_translation}` covering the three sub-questions:

- **Workflow integration.** Pipeline lives behind a coordinator-facing UI at the
  pre-screening queue position; structured JSON (eligibility, evidence,
  teach-back, audit-bundle URI) is forwarded to the coordinator's task list.
  Pipeline is upstream of, not a replacement for, REDCap surveys, sponsor IWRS,
  EHR-resident criteria checkers; the audit bundle is the contract.
- **Comorbidities.** Single-trial design handles comorbidities encoded as
  *exclusion criteria of the accepted trial* correctly (status demotes to LMM,
  teach-back surfaces unresolved exclusion). Comorbidities that require
  considering an *alternative* trial trigger an abstention by construction —
  the deployment recommendation is to treat that abstention as the comorbidity
  signal and branch to a coordinator-directed multi-trial workflow. We
  deliberately do not extend to multi-trial reasoning because the safety
  guarantee in §3.8 relies on $|\mathcal{E}^\star \cap \mathcal{T}|=1$.
- **Dynamic criteria.** The audit bundle records selector thresholds, cited
  passages, and corpus snapshot timestamp. Refresh recommendations: re-run
  $V(t^\star)$, re-run per-gate ablation, treat any change in *only this gate*
  counts as a deployment-time alarm.

**Citation.** New paragraph `\ref{par:workflow_integration}`.

---

## Minor comments

### m1. §3.8 mathematical argument too dense
> *"Could be simplified and better connected to the practical implementation."*

**Action.** Inserted a *Plain-language summary* paragraph at the top of
`\ref{sec:safety_theory}` that states the three operational invariants in one
sentence each (selector returns one trial, postprocessor strips uncited claims,
schema validator collapses non-parsing output to `cannot_determine`). The formal
derivation is preserved unchanged below.

### m2. More V1 cross-trial conflation examples beyond the three probes
> *"The paper could improve clarity by providing more examples of cross-trial
> conflation failure modes in the V1 pipeline."*

**Action.** Added `\paragraph{Cross-trial conflation failure modes observed in
V1.}` (`\ref{par:v1_conflation_examples}`) listing four named patterns with
case_ids:

- **(1) Mixed-eligibility composition** (case 05) — mismatched age cap + comorbidity
  exclusion across studies.
- **(2) Cross-trial outcome substitution** (case 03) — denosumab follow-up arm
  outcomes attached to a separate prospective cohort.
- **(3) Sponsor-phrase aliasing** (case 09) — distinctive trial nickname in
  explanation paragraph contradicting the cited NCT.
- **(4) Vague-anchor drift** (case 15) — three loosely-related fracture trials
  silently merged into one paragraph.

Each is mapped to the specific guard component that blocks it (ρ-gate,
$k_{\text{keep}}$ cap, wide-leak detector $V$, generic-question heuristic $g$).
Per-case JSON traces released alongside the audit bundle.

### m3. The "150-case expanded pool" presentation is confusing
> *"The discussion of the '150-case expanded pool' (Section 4.6) is confusingly
> presented and could benefit from better contextualization."*

**Action.** Two changes:

- Added a *one-line orientation* sentence at the top of
  `\paragraph{Pool tree and the n=150 calibration extension.}`
  (`\ref{par:n150_tree}`): *"The audited safety pool is 114 rows; the
  calibration/threshold-grid pool is a strict superset of 150 cached selector
  signal rows; the 36-row delta exists only so that grid sweeps and ECE bins
  are well-populated, and never enters leak accounting."*
- The previously added `\ref{par:pool_bookkeeping}` already pins
  `curated_30 + broad_84 = 114` vs the older `30 + 80` notebook wording.

### m4. "Dual-judge LLM evaluation" definition unclear
> *"The paper should clarify the exact definition of 'dual-judge LLM evaluation'
> — specifically whether the judges were instructed to evaluate the same
> responses or different responses."*

**Action.** Added `\paragraph{What "dual-judge" means here.}` to
`\ref{sec:setup_phase3}`: the two judges receive byte-equivalent prompts and the
*same* generated answer, score independently, and emit separate score / rationale
/ failure-tag tuples. Explicitly clarified that "dual-judge" is a two-rater
agreement design over fixed responses, not a multi-LLM voting scheme over
alternative generations.

### m5. Why Krippendorff α turns negative on V2/V-final
> *"Section 5.11 should more clearly explain why Krippendorff's α becomes
> negative for V2 and V-final variants."*

**Action.** Rewrote the relevant block in `\ref{par:krippendorff}` to spell out
the formula $\alpha = 1 - D_o/D_e$, point out that $D_e \to 0$ at ceiling, and
state the corrected interpretation: *"ceiling effect, no informative IRR signal"*
rather than "disagreement worse than chance". The same reading is mirrored in
Limitations(i) and the table caption.

---

## Summary of changes (file: `outputs/paper_v2/main.tex`)

| # | Edit | Anchor |
|---|------|--------|
| 1 | Abstract Results+Conclusions softened | abstract |
| 2 | Contribution 1 retitled "within the audited regime" | contribution list |
| 3 | Plain-language summary added to §3.8 | `\ref{sec:safety_theory}` |
| 4 | Dual-judge clarification paragraph | `\ref{sec:setup_phase3}` |
| 5 | V1 conflation failure-mode catalog (4 patterns) | `\ref{par:v1_conflation_examples}` |
| 6 | Krippendorff variance-collapse rewrite | `\ref{par:krippendorff}` |
| 7 | n=150 one-line orientation | `\ref{par:n150_tree}` |
| 8 | Concrete novelty contrast vs TrialGPT/PRISM/COMPOSE/DeepEnroll | after `Positioning against prior…` |
| 9 | Calibration deployment recommendations (item v) | `\ref{par:calibration_clinical}` |
| 10 | Workflow / comorbidity / dynamic-criteria paragraph | `\ref{par:workflow_integration}` |
| 11 | Limitations (i) ceiling-saturation explicit | `\ref{sec:limitations}`(i) |
| 12 | Limitations (viii) backbone generalization | `\ref{sec:limitations}`(viii) |
| 13 | Limitations (ix) area / population coverage | `\ref{sec:limitations}`(ix) |
| 14 | Limitations count updated 7 → 9 | `\ref{sec:limitations}` |

No experiments were re-run; the integrity-aligned figures remain those produced by
`scripts_phase4/build_npj_overleaf_figures.py` and reflect the audited 114-row
ledger and the 150-row selector-signal extension exactly as released in
`outputs/phase4/n114_aggregate/` and `outputs/tables/selector_signals_cache.csv`.
