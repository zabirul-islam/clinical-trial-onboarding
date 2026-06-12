# Referee Report — Round 4 (Submission-Readiness Verification)

**Manuscript:** *Structural Safety in Patient-Facing AI for Clinical Trial Onboarding: An Audited Grounded Pipeline*
**Target venue:** npj Digital Medicine
**Round:** 4 (submission-formatting pass; substance frozen since Round-2 Accept)
**Recommendation:** **Accept — submission-ready pending professor review**

---

## Scope

Round 4 applied medical-journal formatting rules and npj house-style compliance ahead of professor review and submission. No claims, numbers, figures content, or section structure changed. Both manuscript variants (`main.tex` generic; `main_npj.tex` sn-jnl class) plus both supplements updated.

## Verified Changes

### A. Abbreviation discipline (full name before short name — both manuscripts)

| Abbrev | First-use expansion added | Location |
|---|---|---|
| LLM | large language models (LLMs) | Intro ¶1; subsequent "large language models" → LLMs |
| NCT | National Clinical Trial (NCT) identifier | Contributions list |
| API | application-programming-interface (API) | Contribution 2 |
| SaMD | software-as-a-medical-device (SaMD) | Intro reporting ¶ |
| FDA | U.S. Food and Drug Administration (FDA) | Intro reporting ¶ |
| EMA | European Medicines Agency (EMA) | Intro reporting ¶ |
| WHO | World Health Organization (WHO) | Intro reporting ¶ |
| CI | confidence interval (CI) | Results §2.7 first main-text use |
| JSON | JavaScript Object Notation (JSON) | Results §2.7 |
| TREC | Text REtrieval Conference (TREC) | Results calibration ¶ |
| NLI | natural-language-inference (NLI) | Theorem proof; later "An NLI postprocessor" |
| IRB | institutional review board (IRB) | Discussion deployment ¶ |
| ICC | intraclass correlation coefficient ICC(2,1) | Methods §4.5 |
| nDCG | normalized discounted cumulative gain (nDCG) | Methods §4.6 |
| RRF | reciprocal rank fusion (RRF) | Methods §4.6 |

Abstract remains self-contained (AI, LLM expanded inside abstract independently).

### B. Terminology unification
- "five-condition selector" → **"five-gate selector"** (3 sites per manuscript) — matches "gate ensemble" usage paper-wide; "signal" reserved for the four continuous quantities (ρ, τ, μ, κ).
- "roughly 78%" → "approximately 78%" (Methods §4.3).
- "Fig./Figure" audit: mid-sentence uses are "Fig.~"; sentence-initial "Figure~" at deployment-workflow paragraph is correct convention — no change required.
- leak/leakage: convention verified (leakage = phenomenon; leak rate/event/detector = compounds); no violations found ("leakage rate" count = 0).

### C. Heading
- `\section{Methods (online)}` → `\section{Methods}` in both manuscripts (npj style).

### D. Figures
- `pipeline_architecture.png` (144 dpi) → `pipeline_architecture.pdf` (vector) in both manuscripts.
- `deployment_workflow.png` (144 dpi) → `deployment_workflow.pdf` (vector) in both manuscripts.
- Supplement charts `trec_ct2022_field_lengths.png` + `trec_ct2021_qrel_distribution.png` regenerated at **300 dpi** (script `scripts_phase4/build_npj_overleaf_figures.py` dpi=160→300, 8 savefig sites).
- All four main-paper figures now vector PDF; captions already npj-style (bold one-sentence title + description).

### E. Table `tab:six_backbone` overfull resolved
- Headers "Leak (narrow)/(wide)/(sem-both)" → Leak$_{\mathrm{nar}}$/Leak$_{\mathrm{wide}}$/Leak$_{\mathrm{sem}}$ with definitions appended to caption.
- `main.tex`: overfull 51.5 pt → **0**.
- `main_npj.tex` (narrower sn-jnl text block): additionally `\small`→`\footnotesize`, tabcolsep 4pt→3pt; table overfull 43.2 pt → **0**. Two residual sub-visible back-matter hyphenation overflows (4.0 pt Ethics, 8.5 pt Author Contributions) remain — invisible at print scale, class-file related, non-blocking.

### F. npj abstract trimmed (main_npj.tex only, per author instruction)
- ~410 → ~295 net words (within npj structured-abstract cap). All numbers preserved: 0/684; 0.54%/3.2% Clopper–Pearson bounds; 12/684 = 1.75%; κ = 0.16; 86.5% agreement; 8/2/2 classification; ρ<1.5; 5.7% vs 1.6%; 3.95/5 with CI [3.76, 4.14]; 44% lower latency (11.0 vs 19.8 s); r_s 0.52–0.56; negative Brier skill score.
- `main.tex` abstract intentionally unchanged (author keeps full version in generic variant).

## Build State (final)

| File | Pages | LaTeX errors | Overfull |
|---|---|---|---|
| main.pdf | 21 | 0 | 0 |
| main_npj.pdf | 23 | 0 | 2 (sub-visible back-matter hyphenation) |
| supplement.pdf | 20 | 0 | 10 (pre-existing, unchanged) |
| supplement_npj.pdf | 22 | 0 | 25 (pre-existing, unchanged) |

## Remaining Author Checklist (pre-submission)

1. Professor review of trimmed npj abstract — confirm no nuance lost.
2. "approximately 78%" (Methods §4.3) — expose exact selector-expected match count from artifacts, or keep approximate wording (flagged Rounds 3–4).
3. Supplement overfull warnings — cosmetic; tighten only if npj production flags them.
4. Zenodo DOI placeholders in Data/Code Availability — assign at acceptance.

## Verdict

**Submission-ready.** All medical-journal writing rules (citation-free abstract, full-name-before-abbreviation, terminology consistency, vector figures, npj headings, abstract cap) now verified compliant. Scores unchanged from Round 3 (clarity held at 9.5; all dimensions ≥ 6).
