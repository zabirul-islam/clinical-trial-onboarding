# Round 5 — Master-Detail Restoration Note

**Manuscript:** *Structural Safety in Patient-Facing AI for Clinical Trial Onboarding: An Audited Grounded Pipeline*
**Round:** 5 (document-model correction; no substance change)
**Status:** Complete — master/derivative model now explicit and enforced.

## Document Model (now canonical)

| File | Role |
|---|---|
| `main.tex` + `supplement.tex` | **MASTER** — maximal detail, journal-agnostic. Source of truth from which venue versions are derived. |
| `main_npj.tex` + `supplement_npj.tex` | **DERIVATIVE** — npj Digital Medicine trim (lean abstract ≤300 words, no Related Work section, sn-jnl class). |

Rule going forward: corrections (citations, symbols, abbreviations, figure quality) flow to BOTH; trims/cuts apply to DERIVATIVES ONLY.

## Restored to master this round

1. **main.tex abstract — full detail restored** (Round 3 had condensed it):
   - "Within the audited regime ($n=114$ cases, six model families, three pre-specified leak detectors)" opener
   - Explicit $8/12$ / $2/12$ / $2/12$ manual-review fractions
   - "the true lexical leak rate could plausibly be as high as $\sim 3\%$ under correlated sampling" explanation alongside the exact $0.54\%/3.2\%$ bounds
   - Full calibration meta-note ("we document this explicitly to prevent downstream misuse as a triage probability rather than as a limitation")
   - Conclusions: "teach-back-style clarification interactions that the clinical-communication literature recommends"
   - All Round-3/4 fixes retained (no citations, no $A/B/C$, $r_s$ for Spearman, exact latency).

2. **main.tex — `\section{Related Work}` restored** (from main.OLD.tex §133–150, dropped in pre-release restructuring): five subsections — patient–trial matching and recruitment; RAG and medical RAG; safety/hallucination/uncertainty in clinical NLP; informed consent and teach-back; clinical deployment and equity. ~20 additional citations now used in main text (all keys already in references.bib — zero bib changes). Terminology updated to current conventions (LLMs, RAG defined at first Intro use). Absent from main_npj.tex by design.

3. **supplement.tex — judge evaluation infrastructure paragraph restored** (from main.OLD.tex §486–493): JSON-Lines resumability keyed by (case_id, backbone), error-line filtering and retry, tier-1 tokens-per-minute throttling via 8-s per-judge rate gate, exponential backoff, Retry-After handling, post-hoc-verifiable blinding IDs, per-judge seed-shuffled case order.

## Audit findings NOT restored (verified already present)

- Retrieval comparison table/numbers — already in supplement §"Retrieval comparison" (Table S14 region).
- Calibration methodology (ECE bins, BSS base rate, 4-feature LR, 5-fold CV, undef-cell footnotes) — already rich in supplement §"Calibration analysis".
- Ten-dimension rubric definitions + 10→5-dim crosswalk — already in supplement.
- $n=114$ pool decomposition (30+50+22+12) and area distribution (14/8/19/16/13/44) — already in main.tex Methods.
- TREC corpus statistics (75 topics, 35,832 judgments, field lengths) — already in supplement figure captions.

## Build state

| File | Pages | Errors | Overfull | Note |
|---|---|---|---|---|
| main.pdf | 23 (was 21) | 0 | 0 | +2pp from Related Work |
| supplement.pdf | 20 | 0 | 10 (pre-existing) | +1 paragraph |
| main_npj.pdf | 23 | — | — | **byte-identical, untouched** |
| supplement_npj.pdf | 22 | — | — | **byte-identical, untouched** |

No undefined citations; npj derivative PDFs verified unchanged by md5.
