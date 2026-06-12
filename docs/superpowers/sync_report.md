# Phase 0 Sync Report — local Mac ↔ DGX server

**Date:** 2026-06-12
**Gate status:** PASSED
**Sync commit:** `8e58230` (both sides)

## Setup

- Local: `/Users/zabir/Desktop/Clinical-Trial` (git, origin = github.com/zabirul-islam/clinical-trial-onboarding)
- Server: `islamm11@zabi-nvidia-gpu.bme.rpi.edu:~/Desktop/islamm11/avatar_trial_onboarding` (NVIDIA GB10, conda env `avatar_trial`)
- SSH: key auth installed and verified (one-time password use; no password stored).
- Server was **not** a git repo → initialized in place, fetched origin/main, index-only reset, then **per-file** restore of an inventoried divergence list (no blanket reset). Pre-overwrite backup: `~/server_pre_sync_backup_2026-06-12.tar.gz` (24 files).

## Scope decision (author-approved)

Generation/analysis code + outputs must match; **`data/` and `indices/` are server-authoritative** — no local hash parity required (generation runs on server only). Local `data/processed` is a symlink into gitignored `processed/`.

## Invariants verified on BOTH sides (match paper)

| Invariant | Value | Status |
|---|---|---|
| Curated gens (4 backbones × 30) | 120 | ✓ both |
| Extension gens (4 × 84) | 336 | ✓ both |
| Zero-shot gens (2 × 114) | 228 | ✓ both |
| Total generations | 684 | ✓ both |
| Semantic judge calls (684 × 2 JSONL lines) | 1,368 | ✓ (were local-only; now in git → both) |
| Expanded case pool | 150 cases | ✓ (was server-only; pulled) |

**Zero hash mismatches among generation JSONs, judge JSONLs, and n114 aggregate CSVs at first diff** — paper numbers were never at risk.

## Divergences found and resolutions

| Class | Count | Resolution |
|---|---|---|
| Figures + main.tex + references.bib + 2 figure scripts (local newer — 2026-06-12 npj label session) | 19 | local → commit → server restored from commit |
| `run_zeroshot_baseline.py`, `backbone_ablation_{raw,summary}.csv` (server newer, mtime) | 3 | pulled server → local → committed |
| Scratch probe artifacts (`trial_level_scores.*`, `grounding_evidence_top_passages.csv`) | 3 | ephemeral last-run files, regenerable; committed local copies (which match the paper's probe table); documented here |
| Server-only scripts + tables (run_backbone_ablation.py, run_threshold_sweep.py, cache_selector_scores.py, bge-reranker runs, bm25 top1000, 15-case 3way variants, …) | 34 | pulled → local → committed |
| Local-only analysis/paper files (npj tex/pdf, semantic-judge outputs, referee reports, phase5 tooling) | ~100 | committed → created on server via git |
| `run_backbone_ablation_n100.py` + 3 case-manifest JSONs (server-only) | 4 | scp'd local → committed |

## Final verification

Post-reconciliation manifests (roots: configs, scripts*, src, outputs):
**1,430 common files — 1,430 identical — 0 mismatches.**
Local-only remainder (35) = LaTeX build artifacts (`.aux/.bbl/...`), `.pytest_cache`, `.OLD` backups, `.claude-flow` caches — all gitignored, out of contract.

## Ongoing sync discipline (Phases 1–3)

1. Code authored locally → `git commit` + `git push origin main` → server: `git pull` (only code path).
2. Server run outputs (`outputs/phase5/...`) → `rsync -az server:…/outputs/phase5/ outputs/phase5/` (only output-return path).
3. Every analysis/figure/paper number reads from the synced **local** copy.
4. Re-run `sync_manifest.py` + `sync_diff.py` after each major output batch.
5. iCloud caveat: this local Desktop is iCloud-synced; large local file hashing can stall on evicted placeholders — keep big data server-side (already policy).
