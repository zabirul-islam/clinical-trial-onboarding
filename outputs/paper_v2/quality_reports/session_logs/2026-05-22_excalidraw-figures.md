# Session Log — 2026-05-22 — Excalidraw-native Figures 1 & 5

## Skills invoked
- `top-journal-paper` (Step 5.5 figure regeneration)
- `phd-workflow` (`figure` mode)
- `pedro-workflow` (principles: plan-first, orchestrator loop, quality ≥ 90)
- `figure` sub-skill (Excalidraw MCP backend; matplotlib only as fallback renderer)

## Plan
`/Users/zabir/.claude/plans/keep-this-paper-correct-compressed-quasar.md` (rewritten this session)

## Workstreams completed

### W1 — Parity confirmation (documentation only)
- Confirmed via grep that `main.tex` + `supplement.tex` have no residual `n=110`, `80-case`, `8 hand-crafted`, `same safety ceiling`, `89/110` — all 13 claim fixes from the previous session are in place.
- `main_npj.tex` + `supplement_npj.tex` were copied from the post-fix originals; content parity preserved.
- Note appended to `CLAIM-AUDIT.md`.

### W2 — Figure 1 (Grounded onboarding pipeline)
- **Source**: `outputs/paper_v2/figures/pipeline_architecture.excalidraw` (32 elements: 14 rectangles, 14 arrows, 4 text).
- **Excalidraw URL**: `https://excalidraw.com/#json=rRD2hSErig-l-iUV40919,Q59uyHynh_hdTKBTFXtdJA`
- **Fallback PDF**: rendered via new `scripts_phase4/render_excalidraw_to_pdf.py`. 14 inches wide.
- **Content**: Patient query + Trial corpus → BM25 (with nDCG@10=0.319, R@100=0.429) → Cross-encoder (MS-MARCO MiniLM, nDCG@20=0.361, R@100=0.960) → Trial-first selector (5 gate thresholds explicit: ρ≥1.35, τ≥0.28, μ≥−4.5, κ≤2, generic-Q) → 3 module boxes → Patient-facing bundle + Audit artifacts. Shared Qwen2.5-3B backbone label with single dashed arrow. Legend at bottom.
- **Score**: 90.

### W3 — Figure 5 (Clinical deployment workflow)
- **Source**: `outputs/paper_v2/figures/deployment_workflow.excalidraw` (47 elements: 15 rectangles, 15 arrows, 8 text, 9 lane/legend).
- **Excalidraw URL**: `https://excalidraw.com/#json=PE2kFRzuiv3ubEbV0sesm,eUUG8196-irJc1rzROdm-A`
- **Fallback PDF**: rendered via the new CLI.
- **Content**: Four-lane swim diagram (Patient · Pipeline · Coordinator · Audit) with the colour-zoned backgrounds. No arrow crosses a non-target box. Abstain branch routed cleanly off the bottom of the selector. Selector-logs feed (dashed green) goes directly to the audit log at the left edge of the audit lane. Final legend strip across the bottom with one swatch per role + abstain pathway.
- **Score**: 88.

### W4 — Generalised renderer
- **New file**: `scripts_phase4/render_excalidraw_to_pdf.py` (CLI: `--input <path> --output <stem>`).
- Replaces the figure-1-specific `build_figure1_from_excalidraw.py` (kept for back-compat; not deleted).
- Fixed mid-session: `strokeColor="transparent"` and `backgroundColor="transparent"` now handled; `opacity` honored for lane backgrounds.

### W5 — Compile + cross-doc verification
- All four `.tex` files compile cleanly:
  - `main.pdf` — 14 pages, 730 KB
  - `supplement.pdf` — 17 pages, 795 KB
  - `main_npj.pdf` — 16 pages, 774 KB
  - `supplement_npj.pdf` — 21 pages, 703 KB
- Cross-doc callouts: `Supplementary Table~S3,S4,S8,S9,S11,S12,S13,S14` + `Fig.~S11,S13` all resolve in both `main.tex` and `main_npj.tex` (10 unique callouts; 11 occurrences since S9 appears twice for the Krippendorff reference).
- Parity grep: empty (no residual stale numbers).

## Whole-bundle score: 91

| Workstream | Score |
|---|---|
| W1 parity | 95 |
| W2 Figure 1 | 90 |
| W3 Figure 5 | 88 |
| W4 renderer | 92 |
| W5 compile | 95 |
| Mean | 92 |

## Files touched

**New**:
- `outputs/paper_v2/figures/pipeline_architecture.excalidraw`
- `outputs/paper_v2/figures/deployment_workflow.excalidraw`
- `scripts_phase4/render_excalidraw_to_pdf.py`

**Replaced (fallback PDFs from new .excalidraw)**:
- `outputs/paper_v2/figures/pipeline_architecture.{pdf,png}`
- `outputs/paper_v2/figures/deployment_workflow.{pdf,png}`

**Edited (1 line appended)**:
- `outputs/paper_v2/CLAIM-AUDIT.md` (W1 parity note)

**Recompiled**:
- `main.pdf`, `supplement.pdf`, `main_npj.pdf`, `supplement_npj.pdf`

**Not touched** (per scope discipline):
- `main.tex`, `supplement.tex`, `main_npj.tex`, `supplement_npj.tex` (figure paths unchanged)
- `references.bib`, `references_npj.bib`
- Any analysis script, data file, or `outputs/phase*` artifact

## Excalidraw URLs (for user — open in browser, edit if wanted, File → Export to PDF)

| Figure | URL |
|---|---|
| Figure 1 — Grounded onboarding pipeline | `https://excalidraw.com/#json=rRD2hSErig-l-iUV40919,Q59uyHynh_hdTKBTFXtdJA` |
| Figure 5 — Clinical deployment workflow | `https://excalidraw.com/#json=PE2kFRzuiv3ubEbV0sesm,eUUG8196-irJc1rzROdm-A` |

Drop manually-exported PDFs into `outputs/paper_v2/figures/` over the fallback ones, then recompile. No `.tex` edit needed.

## Open items

1. **Sn-nature.bst still rejects @inproceedings entries**: deferred from prior session; not blocking — Springer typesetting pipeline re-formats the bibliography on submission. `main_npj.tex` currently uses `unsrtnat` bibstyle as a working substitute.
2. **F10 numerator** ("roughly 78% of cases"): exact count vs n=114 still requires a small analysis script run; soft-language fix in place.
3. **Optional pre-submission review**: spawn `paper-critic` + `domain-reviewer` over the final bundle once user is satisfied with the Excalidraw figures.

## Answer to user's two questions

1. **"Did you change old main.tex and supplement.tex with claim fixes?"** — Yes. 13 fixes applied in the previous session; verified clean by grep this session. Both `main_npj.tex` and `supplement_npj.tex` inherit those fixes by copy.

2. **"Use Excalidraw, not Python, for Figures 1 and 5; both versions should reference the same figures"** — Done. Native `.excalidraw` JSON files in `figures/`, shareable excalidraw.com URLs above. All four `.tex` files reference the same `figures/pipeline_architecture.pdf` and `figures/deployment_workflow.pdf` paths, so any export from Excalidraw replaces both versions in one step.
