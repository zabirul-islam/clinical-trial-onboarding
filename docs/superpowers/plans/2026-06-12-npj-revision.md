# npj Revision Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute the approved spec (`docs/superpowers/specs/2026-06-12-npj-revision-design.md`): add baseline contrast, NLI claim verification, failure taxonomy, safety–utility frontier, moderated claims, and an expert-review packet to the clinical-trial onboarding paper before npj submission.

**Architecture:** All new code lives in `scripts_phase5/` (mirrors phase3/phase4 conventions: standalone argparse scripts, persisted JSON/CSV, no shared state). Generation runs on the DGX server (`~/Desktop/islamm11/avatar_trial_onboarding`, conda env `avatar_trial`); analysis, figures, and paper edits run locally on the synced copy. Baseline generations adopt the existing `backbone_gens` JSON schema so every existing detector and judge harness works unchanged.

**Tech stack:** Python (pandas, matplotlib, transformers/torch on DGX), existing project scripts (`scripts_phase4/semantic_leak_judge.py`, `scripts_phase3/run_llm_judge.py`, BM25+cross-encoder retrieval artifacts), LaTeX (sn-jnl), rsync+ssh, git.

**Key existing artifacts (verified locally):**
- Curated 30-case gens: `outputs/backbone_gens/<backbone>/case_*.json` (schema: case_id, backbone, source, category, gold_nct, selected_doc, question, evidence, …)
- 84-case extension gens: `outputs/phase4/n100_expansion/gens/<backbone>/`
- Judge JSONL: `outputs/phase4/n100_expansion/judge_{sonnet,gpt4o}.jsonl`, `outputs/phase3/…`
- Semantic judge: `scripts_phase4/semantic_leak_judge.py` (`GEN_ROOTS` registry at line 414; argparse: `--backbones --judges --out-dir --max-cases --k-candidates --concurrency --min-interval`)
- Zero-shot baseline template: `scripts_phase4/run_zeroshot_baseline.py`
- Aggregates: `outputs/phase4/n114_aggregate/*.csv`; threshold sweep: `outputs/tables/threshold_sweep_grid.csv`; selector cache: `outputs/tables/selector_signals_cache.csv`
- Paper: `outputs/paper_v2/{main,supplement}.tex` (master), `{main,supplement}_npj.tex` (derivatives — trim from master, never extend independently)

**Hard rules:**
1. Phase 0 must pass before any Task ≥ 2 step runs on the server.
2. Scripts flow local → server **via git only**; outputs flow server → local **via rsync only**; every paper number reads from the synced local copy.
3. Configs (prompts, thresholds, checkpoints) are frozen and committed **before** the runs that use them.
4. NLI threshold calibrated only on the 15-case dev pool; objective fixed in spec (highest threshold with ≤10% benign drop).

---

## Task 0: Phase 0 — server/local sync gate (HARD GATE)

**Files:**
- Create: `scripts_phase5/sync_manifest.py`
- Create: `docs/superpowers/sync_report.md`

- [ ] **Step 0.1: Test SSH from this Mac**

Run: `ssh islamm11@Zabi-nvidia-gpu 'hostname && ls ~/Desktop/islamm11/avatar_trial_onboarding'` (try `Zabi-nvidia-gpu.local` / ask user for host/IP if unresolved).
Expected: hostname + project dir listing (configs, data, indices, outputs, scripts, scripts_phase3, scripts_phase4, src, …).
If SSH fails after reasonable attempts: switch to runbook mode — every later "run on server" step becomes a fenced command block the user pastes; outputs come back via `scp`/USB; all other steps unchanged.

- [ ] **Step 0.2: Write the manifest script**

`scripts_phase5/sync_manifest.py` — walks given roots, emits CSV `path,size,sha256` (sha256 streamed, 1 MB chunks), excludes `__pycache__`, `.git`, `logs/`, `*.log`. Args: `--roots data indices outputs scripts scripts_phase3 scripts_phase4 src configs --out manifest_<host>.csv`. ~60 lines, plain stdlib.

- [ ] **Step 0.3: Run on both sides, diff**

Local: `python scripts_phase5/sync_manifest.py --out /tmp/manifest_local.csv`
Server: same script (push via `scp` for this bootstrap step only), `--out /tmp/manifest_server.csv`; `scp` back.
Diff: pandas outer-join on path; report classes: local-only, server-only, hash-mismatch.

- [ ] **Step 0.4: Verify paper-number invariants on BOTH sides**

- 114 unique case_ids across `outputs/backbone_gens/<bb>` ∪ `outputs/phase4/n100_expansion/gens/<bb>` per backbone (30+84)
- 4 open-weight backbone dirs × 114 + 2 zeroshot dirs × 114 = 684 generation JSONs
- semantic judge JSONLs: `outputs/phase4/reviewer_fixes/semantic_leak_judge_{sonnet,gpt4o}.jsonl` = 684 lines each (1,368 calls); rubric judge JSONLs `outputs/phase4/n100_expansion/judge_{sonnet,gpt4o}.jsonl` are a different artifact (339/336 lines) — do not confuse
Run a 10-line pandas check; paste results into sync report.

- [ ] **Step 0.5: Reconcile divergences**

For each divergent file: decide newest-correct deliberately (check mtime + git log + content), log decision in `docs/superpowers/sync_report.md`. Code divergence → resolve through git (server becomes a git clone/pull of the same repo if not already; if server is not a git repo, init from local remote state). Output divergence → authoritative side wins (server for raw gens, local for paper_v2). **Hard stop + ask user if any generation JSON or judge JSONL mismatches in content.**

- [ ] **Step 0.6: Establish ongoing sync mechanics + commit**

- Server: `git remote -v` configured to same GitHub repo; document `git pull` as the only code-update path.
- Local: document `rsync -avz islamm11@<host>:~/Desktop/islamm11/avatar_trial_onboarding/outputs/phase5/ outputs/phase5/` as the only output-return path.
- Commit: `git add scripts_phase5/sync_manifest.py docs/superpowers/sync_report.md && git commit -m "phase5: sync gate manifest + report"`

---

## Task 1: Frozen run configs

**Files:**
- Create: `configs/phase5_baselines.yaml`

- [ ] **Step 1.1: Write config with ALL frozen decisions**

```yaml
# configs/phase5_baselines.yaml — frozen before any phase-5 run (spec §1, §2)
backbones:
  - Qwen/Qwen2.5-7B-Instruct
  - Qwen/Qwen2.5-3B-Instruct
decoding: {temperature: 0.0, max_new_tokens: 1500}
evidence:
  multi_trial_k: 6          # passages, no per-trial cap (B1-B3)
  source: reranked cross-encoder pool per case (same artifacts as V-final input)
baselines:
  B1_multi_rag:
    evidence: multi_trial
    guard: none
  B2_prompt_guard:
    evidence: multi_trial
    extra_instruction: >
      CRITICAL SAFETY RULE: The evidence below may contain passages from
      MULTIPLE different clinical trials. You must first identify the single
      trial most relevant to the patient's question, then answer using ONLY
      passages from that one trial. Never combine eligibility criteria,
      procedures, schedules, or any other details from different trials in
      one answer. If you cannot identify a single clearly most-relevant
      trial, say so and ask the patient for the study title or NCT number.
  B3_citation_enforced:
    evidence: multi_trial
    guard: every field must cite passage ids; postprocessor drops uncited fields
  B4_top1:
    evidence: single_trial_rank1   # trial owning rank-1 passage; no 5-gate
nli:
  checkpoint: MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli
  calibration: {pool: 15case_dev, objective: max threshold s.t. benign_drop <= 0.10}
taxonomy: {T1: cross_trial_contamination, T2: unsupported_completion, T3: ordinary_hallucination}
```

- [ ] **Step 1.2: Commit before any run**

`git commit -m "phase5: freeze baseline + NLI configs (pre-registration)"`

---

## Task 2: Expert review packet instrument (no server dependency)

**Files:**
- Create: `scripts_phase5/build_expert_packet.py`
- Create: `scripts_phase5/expert_agreement.py`
- Create: `outputs/expert_review/INSTRUCTIONS.md`
- Create: `outputs/expert_review/scoring_sheet_template.csv`

- [ ] **Step 2.1: Case selection script** — first add `outputs/expert_review/blind_key.csv` to `.gitignore`; then stratified sample of 30 V-final generations using EXISTING phase-4 flags only (regime accept/abstain × semantic-flagged/clean × backbone), reproducible seed, emits `outputs/expert_review/selection.csv` + blinded dossiers `outputs/expert_review/dossiers/<blind_id>.md` (question, selected-trial passages, full patient-facing response; blind_id = SHA1, mapping kept in gitignored `blind_key.csv`). B1/B4 dossier slots appended later by re-running with `--add-baselines` (Task 7).
- [ ] **Step 2.2: Instrument docs** — `INSTRUCTIONS.md`: 5-dim rubric (1–5, same definitions as LLM judges), binary cross-trial leak judgment, taxonomy label T1/T2/T3, free-text; estimated 60–90 min for 30 cases; note: synthetic queries, no patient data (IRB determination aid). `scoring_sheet_template.csv`: blind_id × dimensions.
- [ ] **Step 2.3: Analysis script** — `expert_agreement.py`: reads filled sheets, computes expert leak rate + Clopper–Pearson CI, expert↔LLM-judge κ/Spearman per dimension. Unit-test on a synthetic filled sheet.
- [ ] **Step 2.4: Smoke + commit** — run selection, inspect 2 dossiers manually, commit (without `blind_key.csv`).

---

## Task 3: Baseline generation harness (DGX)

**Files:**
- Create: `scripts_phase5/run_baselines.py`
- Output: `outputs/phase5/baseline_gens/<baseline_id>/<backbone_dirname>/case_*.json`

- [ ] **Step 3.1: Locate per-case multi-trial evidence on server (Phase 0 inventory).** Expected: per-case reranked pool artifacts used to build V-final inputs (grounding-evidence CSVs referenced by `semantic_leak_judge.py`). Fallback if absent: rebuild per-case top-100 BM25 + cross-encoder rerank with `scripts/run_cross_encoder_rerank.py` against existing indices.
- [ ] **Step 3.2: Write harness.** Reuses: case loading pattern from `run_zeroshot_baseline.py` (114 records from the two gens dirs — question, gold_nct, category, source); JSON-schema prompt + postprocessor from the V-final pipeline; transformers generation (greedy, fp16/bf16) with model loaded once per backbone. Per config: B1/B2/B3 build evidence block from top-6 multi-trial passages (B2 prepends frozen instruction; B3 prompt requires `support_passages` ids and postprocessor drops uncited fields); B4 selects rank-1 passage's trial then reuses the existing single-trial path minus the 5-gate checks. Output JSON: identical keys to `backbone_gens` schema + `baseline_id`, `evidence_trials` (list of NCT ids handed to the model), `selected_doc` (B4: rank-1 trial; B1–B3: null → leak detectors evaluate against `evidence_trials[0]`'s competitor set; define explicitly: for B1–B3 the "selected" reference trial = trial of the rank-1 passage, so any other-trial reference counts as cross-trial content).
- [ ] **Step 3.3: Smoke test** — `python scripts_phase5/run_baselines.py --max-cases 3 --baselines B1_multi_rag` on server; inspect the 3 JSONs by hand (prompt sanity, evidence_trials populated, parseable output). Repeat per config.
- [ ] **Step 3.4: Full sweep** — 4 configs × 2 backbones × 114 = 912 generations. Log per-case latency. Rsync `outputs/phase5/` back to local. Re-run sync manifest on `outputs/phase5`.
- [ ] **Step 3.5: Commit harness + smoke evidence.**

---

## Task 4: Lexical leak audit on baselines (local, deterministic)

**Files:**
- Create: `scripts_phase5/audit_baseline_leaks.py`
- Output: `outputs/phase5/baseline_leak_summary.csv`

- [ ] **Step 4.1:** Port the narrow (NCT-regex) + wide (distinctive-token) detectors from the phase-4 audit code onto the baseline JSONs; reference trial per Step 3.2 definition. Unit-test the detector port on 2 hand-built fixtures (one planted NCT leak, one planted distinctive-token leak) — must flag both.
- [ ] **Step 4.2:** Run over all 912; emit per-(baseline, backbone) leak/commit/parse-ok rates with Clopper–Pearson CIs. This is the paper's new headline contrast table.
- [ ] **Step 4.3: Commit.**

---

## Task 5: Semantic + rubric judging with taxonomy (API)

**Files:**
- Modify: `scripts_phase4/semantic_leak_judge.py` (GEN_SOURCES registry ~line 415; judge prompt + output schema)
- Create: `scripts_phase5/run_baseline_rubric_judges.py` (adapter around `scripts_phase3/run_llm_judge.py` protocol)
- Output: `outputs/phase5/semantic_judge/`, `outputs/phase5/rubric_judge/`

- [ ] **Step 5.1:** Extend semantic judge: add `taxonomy` field (T1/T2/T3 definitions in prompt, returned only when `semantic_leak=1`); register 8 baseline gen-dirs in `GEN_ROOTS` (line 414). Keep blinding + 6 s rate gate.
- [ ] **Step 5.2:** Smoke on 3 cases × 1 baseline × both judges; verify JSONL schema.
- [ ] **Step 5.3:** Full semantic sweep: 912 × 2 judges = 1,824 calls (~3 h at 6 s gate; run both judges concurrently as phase 4 did).
- [ ] **Step 5.4:** Rubric sweep on 912 with existing 5-dim blind protocol = 1,824 calls.
- [ ] **Step 5.5:** Re-annotation sweep: the 104 either-judge-flagged phase-4 generations re-judged with taxonomy prompt (208 calls); author adjudication sheet `outputs/phase5/taxonomy_adjudication.csv` (consensus cases pre-filled, author fills final label).
- [ ] **Step 5.6:** Aggregate: `outputs/phase5/judged_summary.csv` (per baseline × backbone: sem-both, sem-either, rubric means + bootstrap CIs, taxonomy distribution). Commit.

---

## Task 6: NLI redaction module + counterfactual audit (DGX)

**Files:**
- Create: `scripts_phase5/nli_redact.py`
- Create: `tests/test_nli_redact.py` (sentence-split + decision-rule unit tests, run locally without GPU using a stub scorer)
- Output: `outputs/phase5/nli/{calibration.csv,redaction_684.csv,redaction_912.csv}`

- [ ] **Step 6.1: Unit tests first** (stub scorer): sentence splitting of patient-facing fields; redaction rule (drop below threshold, keep fallback strings untouched, rebuild summary); logging of dropped sentences.
- [ ] **Step 6.2: Implement** with pinned checkpoint (config Task 1); premise = concatenated E* passages (chunked to model max length, max-entailment over chunks); hypothesis = each sentence.
- [ ] **Step 6.3: Calibrate on 15-case dev pool only** — sweep thresholds {0.5 … 0.95}; pick highest with benign-drop ≤ 10%; write `calibration.csv` + chosen threshold into config (committed before Step 6.4).
- [ ] **Step 6.4: Counterfactual audit** — apply to frozen 684 + new 912. Report: (a) of the 12 consensus flags, how many flagged sentences are redacted; (b) benign collateral rate overall; (c) pre/post rubric delta on a 40-generation judge-scored sample (160 calls, small add-on). 
- [ ] **Step 6.5: Commit + rsync results local.**

---

## Task 7: Analysis, figures, cost-of-abstention (local)

**Files:**
- Create: `scripts_phase5/build_phase5_figures.py`
- Create: `scripts_phase5/cost_of_abstention.py`
- Output: `outputs/paper_v2/figures/{frontier_leak_utility.pdf,baseline_leak_bars.pdf,taxonomy_breakdown.pdf}`

- [ ] **Step 7.1: Frontier figure** — x = judge-pooled overall rubric, y = semantic-both leak rate (lexical shown as marker fill), one point per (system ∈ {V-final, B1–B4}) × backbone, abstention-rate annotation. Canonical labels: Qwen-2.5-3B / Qwen-2.5-7B (memory: figure-build-gotchas).
- [ ] **Step 7.2: Cost-of-abstention** — over the 82 abstain cases: fraction with gold-NCT trial present in candidate pool (from case manifest + selector cache); report as missed-utility rate with CI.
- [ ] **Step 7.3: Taxonomy figure/table** — T1/T2/T3 distribution per system × backbone.
- [ ] **Step 7.4:** Append B1/B4 dossiers to expert packet (`build_expert_packet.py --add-baselines`), final packet zip → **ship to user for MGH/Yale**. Commit.

---

## Task 8: Paper integration (local; master first, then npj derivative)

**Files:**
- Modify: `outputs/paper_v2/main.tex`, `outputs/paper_v2/supplement.tex` (master)
- Modify: `outputs/paper_v2/main_npj.tex`, `outputs/paper_v2/supplement_npj.tex` (trim from master)

- [ ] **Step 8.1: Claim moderation pass (both master + npj):** every "zero leakage"/"zero cross-trial leakage" → "zero detected lexical leakage under the audited conditions" (abstract, contributions, Table 1 caption, theorem text, figure captions). Grep gate: `grep -i "zero leakage\|zero cross-trial leak" *.tex` → 0 hits.
- [ ] **Step 8.2: New Results subsections:** baseline comparison (Task 4+5 numbers), NLI mitigation (Task 6), taxonomy (Task 5), safety–utility frontier + cost-of-abstention (Task 7). New Methods: baseline configs (cite frozen config), NLI module, taxonomy definitions, expert protocol.
- [ ] **Step 8.3: Expert-validation subsection:** Methods complete now; Results slot with explicit "data collection ongoing" until sheets return, then `expert_agreement.py` numbers dropped in.
- [ ] **Step 8.4: Abstract + contributions rewrite** around: defined patient-facing safety problem → structural control vs four audited baselines → claim-level verification → human validation.
- [ ] **Step 8.5: Compile + sweep:** latexmk both pairs (`/Library/TeX/texbin`); forbidden-string sweep with `/opt/homebrew/bin/pdftotext` (PLACEHOLDER, (prod), reviewer, peer-review, deployed tuple, zero leakage); fix → repeat until clean. Commit.

---

## Task 9: Final verification

- [ ] **Step 9.1:** Re-run all phase-5 figure/analysis scripts from persisted CSV/JSON only — byte-identical outputs (reproducibility gate).
- [ ] **Step 9.2:** Re-run sync manifest; local ↔ server clean.
- [ ] **Step 9.3:** Cross-check every new number in tex against generated CSVs (spot-check script or manual table).
- [ ] **Step 9.4:** Final commit + summary report to user (incl. expert-packet status + what remains blocked on MGH/Yale data).

---

## Execution order & parallelism

```
Task 0 (gate) → Task 1 → {Task 2 ∥ Task 3} → Task 4 → {Task 5 ∥ Task 6} → Task 7 → Task 8 → Task 9
```
Expert packet ships at end of Task 7 (after baseline generation, before judging completes if needed).

## Risks

- **SSH unavailable** → runbook mode (Step 0.1 fallback); plan unchanged otherwise.
- **B1 shows zero lexical leak** → reportable finding; semantic channel becomes the primary contrast (spec §1 falsifiability note).
- **Multi-trial evidence artifacts missing on server** → rebuild path in Step 3.1.
- **Judge API budget/refusals** → refusals logged + reported as in phase 4; exact tally 1,824 (semantic) + 1,824 (rubric) + 208 (re-annotation) + 160 (NLI pre/post sample) = 4,016 calls vs user-approved ~3.8k — confirm the +160 with user before Step 6.4.
