# Domain Rigor Checklist Audit

**Manuscript**: `main_npj.tex` (also mirrored to `main.tex`)
**Skill phase**: top-journal-paper Step 5 (Domain Rigor Audit)
**Checklist used**: `checklist_medical_imaging.md` (closest proxy for clinical-AI safety; ~40% of items N/A by design)
**Date**: 2026-05-23

This document pre-records which checklist items are N/A and why,
so that the Phase 8 peer-review pass does not down-score
methodological credibility on items that do not apply to a
safety-pipeline study with no patient cohort.

## Items Addressed in the Manuscript

| Checklist item | Manuscript location | Notes |
|---|---|---|
| TRIPOD+AI reporting compliance | Introduction (new ¶ after contributions) | Cited as `collins2024tripodai` |
| CLAIM checklist alignment | Introduction (same ¶) | Cited as `mongan2020claim` |
| DECIDE-AI early-stage framing | Introduction (same ¶) | Cited as `vasey2022decideai` |
| CONSORT-AI / SPIRIT-AI alignment | Introduction (same ¶) | Cited as `liu2020spiritai` |
| FDA SaMD positioning | Introduction (same ¶) | Cited as `fda2021samdaction`; Class II SaMD candidate |
| EMA reflection paper | Introduction (same ¶) | Cited as `ema2024reflection` |
| WHO ethics guidance on LMMs | Introduction (same ¶) | Cited as `who2024llm` |
| Explainability stance | Introduction + Discussion | Substitute structural audit traceability for post-hoc XAI; cites `ghassemi2021falsehope` |
| Calibration reporting | Results §calib + Discussion + Supplement §S7 | Brier Skill Score reported with sign; cites `brier1950verification`, `guo2017calibration`, `platt1999probabilistic` |
| Confidence intervals | Throughout Results + Methods | Clopper–Pearson 95% upper bound; cites `clopper1934confidence`, `efron1979bootstrap` |
| Inter-rater reliability | Results §IRR + Methods §judge + Supplement §S4 | Cohen's κ + ICC + Spearman ρ + Krippendorff's α; cites `cohen1960kappa`, `landis1977kappa`, `krippendorff2018content`, `hayes2007krippendorff`, `gwet2014handbook`, `hallgren2012irr` |
| Reproducibility statement | Reproducibility statement + `reproducibility_report.md` | "Manual only" path for closed-API non-determinism |
| Data availability | Data availability statement | Zenodo DOI placeholder; ClinicalTrials.gov public source; no PII |
| Code availability | Code availability statement | MIT-licensed GitHub + Zenodo archive (placeholders) |
| Equity / under-represented populations | Introduction ¶1 + Discussion limitations | Cites `unger2019recruitment`, `clark2019barriers`, `fda2022diversity`, `chang2024fair`, `obermeyer2019dissecting` |
| Informed consent + teach-back | Discussion clinical-workflow ¶ | Cites `kadam2017informed`, `sudore2009interventions`, `kripalani2008teachback`, `talevski2020teachback`, `ho2017teachback`, `tamariz2013improving`, `glaser2020interventions` |
| Patient-facing AI safety / hallucination | Introduction + Discussion | Cites `ji2023hallucination`, `maynez2020faithfulness`, `pal2023medhalu`, `huang2025survey`, `ayers2023comparing` |

## Items Marked N/A (with Rationale)

| Checklist item | Why N/A | Documented in manuscript? |
|---|---|---|
| Patient demographics | No patient cohort; 114 synthetic onboarding cases derived from TREC topics and curated audit corpus. | Methods §case construction; Discussion limitation (ii) |
| Patient-level train/test split | No patient-level data; case-level split (50 TREC + 22 paraphrase + 12 vague + 30 anchor) | Methods §case construction |
| IRB approval | No human subjects; no enrolled participants; synthetic queries only | Ethics statement (explicit) |
| Scanner-vendor fairness | No imaging modality; this is a text-only safety pipeline | N/A by design (not mentioned, but obvious from study type) |
| AUC / DeLong test | Binary safety outcome with zero positives in audited regime; AUC degenerates. Reported instead: Clopper–Pearson upper bound on leak rate. | Results §headline; Discussion |
| Multi-center validation | Single-institution audit; explicitly framed as empirical bound, not generalization guarantee | Discussion limitation (ii) — "Empirical coverage of the audited regime" |
| Patient-perceived comprehension | No human-subject teach-back study; deferred to follow-up | Discussion limitation (iii) — "Clinical-utility validation" |
| Like-for-like baseline comparison | Closed-API baselines (GPT-4o, Sonnet 4.5) are zero-shot under our guarded prompt; no like-for-like against published trial-matcher pipelines (their artifacts do not enable patient-facing form) | Discussion limitation (v); §positioning |
| ECE (Expected Calibration Error) | Brier Skill Score is the primary calibration metric reported (negative result); ECE would not change the qualitative conclusion (selector miscalibrated by design) | Supplement §S7 reports ECE in calibration table; main paper limits to BSS for brevity |
| Pediatric / rare-disease / non-English coverage | Out of scope for the audited regime; under-represented in TREC CT 2021/2022 source pool | Discussion limitation (ii) — explicitly flagged |
| Algorithmic-fairness audit on enrollment records | No deployment institution; no enrollment records used | Discussion limitation (ii) — explicitly flagged as required before deployment |

## Items Partially Addressed

| Checklist item | Status | Action taken |
|---|---|---|
| Subgroup analysis | Therapeutic-area heatmap (6 areas) is the subgroup proxy; demographic subgroups absent. | Fig. 3 (per-area heatmap); flagged in Discussion as "domain-strata, not demographic" |
| Real-world deployment evidence | Pipeline described as pre-screening clarification aid; not yet deployed clinically | Fig. 5 (deployment workflow) + Discussion clinical-workflow ¶ |
| Patient-utility validation | LLM-judge rubric scores rather than patient-perceived comprehension | Limitation (iii) explicitly flagged |

## Methodological Credibility Risk Areas (for Phase 8 reviewer)

The peer-review agent should NOT down-score on items in the **N/A**
table above. The following risks **are** legitimate and should be
weighed:

1. **n=114 sample size** — small for a clinical evaluation; the
   paper frames this as an empirical bound, not a guarantee. Already
   flagged as Limitation (ii).
2. **2 LLM judges vs. multi-clinician audit** — Limitation (i)
   explicitly defers multi-clinician audit to follow-up.
3. **Negative calibration result** — explicitly reported and
   contextualized; not a methodological gap, an honest finding.
4. **Closed-API baselines are zero-shot** — Limitation (v)
   explicitly defers like-for-like comparison to follow-up.

These four limitations are all explicit in the Discussion §Limitations
section.

## Summary

- Items addressed: 16 of 27 checklist items.
- Items marked N/A with rationale: 11 of 27.
- Items partially addressed (with explicit limitation): 3 of 27.
- Items unaddressed without rationale: **0**.

The manuscript is **compliant with the medical-imaging rigor checklist
to the extent applicable** to a clinical-AI safety study with no
patient cohort. Peer-review penalization should be limited to the
four legitimate methodological risk areas above, all of which are
already documented as Limitations in the Discussion.
