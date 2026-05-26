# Paper A polish — Implementation Plan

**Goal:** Land Paper A at npj Digital Medicine bar in 4 weeks. Close 7 reviewer-attack surfaces (single-rater, no SOTA, no calibration, no per-gate ablation, small-N, no per-area breakdown, single reranker) without a clinician.

**Architecture:** Reuse existing `scripts_phase3/` dual-judge harness for everything that involves judge scoring. Add adapter scripts and analysis scripts. Run on DGX `zabi-nvidia-gpu.bme.rpi.edu` under existing `avatar_trial` conda env.

**Tech Stack:** Python 3.11, PyTorch + vllm, anthropic + openai SDKs, pandas, scikit-learn, scipy, matplotlib. All on existing server; no new env.

**Defaults assumed (override anytime):**
- TrialGPT only (no PRISM)
- Title rebrand → "An Auditable Pipeline for Patient-Facing Clinical Trial Onboarding"
- 10-dim audit ↔ 5-dim judge mapping (no full re-rate)
- Email Jin et al. after T1.2 results land

**Convention:**
- All commands assumed run from server `~/Desktop/islamm11/avatar_trial_onboarding/` unless stated.
- Environment: `conda activate avatar_trial` first.
- API keys: `export ANTHROPIC_API_KEY=sk-ant-...` and `export OPENAI_API_KEY=sk-...` in shell before judge runs.
- All new scripts go in `scripts/` (or `scripts_phase4/` for new-phase work — pick one, stick with it). I'll use `scripts_phase4/` to keep the polish-pass cleanly separable.
- Outputs land in `outputs/phase4/`.

**Frequency of sync:** after every task completes on server, rsync the relevant `outputs/phase4/` subdir down so I can inspect.

---

## File structure for new artifacts

```
scripts_phase4/
├── README.md                              # how to run phase 4
├── build_15case_audit_gens.py             # adapter: eval_runs/* → 15case_audit_gens/<variant>/<case>.json
├── score_15case_audit.py                  # wraps run_llm_judge.py for 45 generations + human-rater merge
├── label_therapeutic_areas.py             # T1.3: LLM-assisted area labeling for 30-case + n=70
├── compute_calibration.py                 # T1.4: ECE / Brier / reliability diagram
├── run_per_gate_ablation.py               # T1.5: 5 leave-one-out variants × 30 × 4 backbones
├── run_n70_expansion.py                   # T1.6: 70 new cases → backbone gens
├── run_trialgpt_baseline.py               # T1.2: TrialGPT inference on 30-case corpus
├── aggregate_phase4_tables.py             # writes paper-ready tables for §3 T1.x
└── build_phase4_figures.py                # writes paper-ready figures

outputs/phase4/
├── 15case_audit/                          # T1.1
│   ├── 15case_audit_gens/{V1,V2,V-final}/case_*.json
│   ├── judge_sonnet_15case.jsonl
│   ├── judge_gpt4o_15case.jsonl
│   ├── audit_rubric_mapping.csv           # 10-dim ↔ 5-dim crosswalk
│   ├── 3way_agreement_per_dim.csv         # human / sonnet / gpt4o
│   └── 3way_agreement_summary.json
├── trialgpt/                              # T1.2
│   ├── trialgpt_gens/<case_id>.json
│   ├── trialgpt_dualjudge_scores.csv
│   └── trialgpt_vs_vfinal_comparison.csv
├── area_breakdown/                        # T1.3
│   ├── area_labels_30case.csv
│   ├── area_labels_n100.csv               # after T1.6 lands
│   └── per_area_safety_summary.csv
├── calibration/                           # T1.4
│   ├── calibration_per_threshold.csv
│   ├── reliability_diagram.{pdf,png}
│   └── calibration_summary.json
├── per_gate_ablation/                     # T1.5
│   ├── gens/{no-rho,no-tau,no-mu,no-kappa,no-generic}/<backbone>/<case>.json
│   ├── per_gate_safety_summary.csv
│   └── per_gate_dualjudge_scores.csv
├── n100_expansion/                        # T1.6
│   ├── onboarding_eval_cases_n100.json
│   ├── gens/<backbone>/<case>.json
│   ├── n100_dualjudge_scores.csv
│   └── n100_per_backbone_summary.csv
└── tier2/                                 # T2.x
    ├── bge_reranker_large_*.csv
    ├── trec_ct2022_retrieval_*.csv
    ├── retrieval_bootstrap_cis.csv
    ├── nli4pr_eligibility_alignment.csv
    └── generic_flag_sensitivity.csv
```

---

## Phase 1 — Quick wins (Week 1)

### Task 1.1A: Build 15-case audit gen adapter

**Files:**
- Create: `scripts_phase4/build_15case_audit_gens.py`
- Read: `outputs/eval_runs/case_*/`, `outputs/eval_runs_v2/case_*/`, `outputs/eval_runs_final/case_*/`
- Write: `outputs/phase4/15case_audit/15case_audit_gens/{V1,V2,V-final}/case_NN.json`

**Why:** `run_llm_judge.py` reads `<gens_dir>/<backbone_slug>/<case_id>.json`. The 15-case audit outputs are spread across 3 sibling dirs, one per variant. Need to reshape.

- [ ] **Step 1: Inspect existing JSON schemas (1 min)**

  ```bash
  cd ~/Desktop/islamm11/avatar_trial_onboarding
  ls outputs/eval_runs_final/case_01/
  cat outputs/eval_runs_final/case_01/$(ls outputs/eval_runs_final/case_01/ | head -1) | head -50
  ```

  Confirm what JSON keys are present per variant. Paste back any schema differences I should know about.

- [ ] **Step 2: Write adapter script**

  Adapter pseudocode:
  ```python
  # scripts_phase4/build_15case_audit_gens.py
  # For each (variant, case_id) in {V1, V2, V-final} × {01..15}:
  #   - read source JSONs from eval_runs/, eval_runs_v2/, eval_runs_final/ respectively
  #   - construct unified JSON record matching backbone_gens schema:
  #       { case_id, variant, source, category, gold_nct, selected_doc,
  #         decision, parse_ok, gen_snippet, raw_eligibility, raw_explanation,
  #         raw_teachback, trial_context_passages }
  #   - write to outputs/phase4/15case_audit/15case_audit_gens/<variant>/case_NN.json
  ```

  I'll write the actual code once you paste a sample existing JSON in step 1.

- [ ] **Step 3: Verify shape**

  ```bash
  ls outputs/phase4/15case_audit/15case_audit_gens/V-final/ | wc -l   # expect: 15
  python -c "import json; d=json.load(open('outputs/phase4/15case_audit/15case_audit_gens/V-final/case_01.json')); print(list(d.keys()))"
  ```

  Expected: 15 files per variant, keys including `case_id`, `decision`, `selected_doc`, generation text.

### Task 1.1B: Score 15-case audit with dual judge

**Files:**
- Create: `scripts_phase4/score_15case_audit.py` (thin wrapper around `run_llm_judge.py`)
- Output: `outputs/phase4/15case_audit/judge_{sonnet,gpt4o}_15case.jsonl`

- [ ] **Step 1: Dry run (1 case, 1 judge) to verify wiring**

  ```bash
  cd ~/Desktop/islamm11/avatar_trial_onboarding
  python scripts_phase3/run_llm_judge.py \
    --gens-dir outputs/phase4/15case_audit/15case_audit_gens \
    --out-dir  outputs/phase4/15case_audit/ \
    --judges   sonnet \
    --concurrency 1 \
    --limit 1
  ```

  Expected: 1 line written to `outputs/phase4/15case_audit/judge_sonnet.jsonl`. Inspect it has 5 rubric scores + rationale + failure_modes.

  If `run_llm_judge.py` doesn't have a `--limit` flag, add it (5 lines), or run on a temp dir with 1 file.

- [ ] **Step 2: Full run**

  ```bash
  python scripts_phase3/run_llm_judge.py \
    --gens-dir outputs/phase4/15case_audit/15case_audit_gens \
    --out-dir  outputs/phase4/15case_audit/ \
    --judges   sonnet gpt4o \
    --concurrency 6
  ```

  Wall time: ~20 min. Cost: ~$1.50.

- [ ] **Step 3: Sanity check**

  ```bash
  wc -l outputs/phase4/15case_audit/judge_*.jsonl
  ```

  Expected: 45 lines per judge file (3 variants × 15 cases). If less, resume re-runs the missing ones.

### Task 1.1C: Build 10-dim ↔ 5-dim mapping + 3-way agreement

**Files:**
- Create: `outputs/phase4/15case_audit/audit_rubric_mapping.csv` (manual table)
- Create: `scripts_phase4/compute_3way_agreement.py`
- Output: `outputs/phase4/15case_audit/3way_agreement_per_dim.csv` + summary JSON

**Mapping table** (write as CSV, paper supplementary):

| 10-dim audit field | 5-dim judge field | Mapping rule |
|---|---|---|
| topic_relevance ∈ {yes/partial/no} | factuality | yes→4-5, partial→2-3, no→1 |
| eligibility_overstatement ∈ {none/mild/severe} | factuality | none→4-5, mild→3, severe→1-2 |
| unsupported_explanation ∈ {none/mild/severe} | groundedness | none→4-5, mild→3, severe→1-2 |
| fallback_used_correctly ∈ {yes/partial/no} | groundedness | yes→4-5, partial→3, no→1-2 |
| missing_fact_reasonableness ∈ {yes/partial/no} | abstain_appropriateness | yes→4-5, partial→3, no→1-2 |
| unresolved_req_reasonableness ∈ {yes/partial/no} | abstain_appropriateness | yes→4-5, partial→3, no→1-2 |
| teach_back_targeting ∈ {yes/partial/no} | patient_utility | yes→4-5, partial→3, no→1-2 |
| patient_facing_clarity ∈ {low/medium/high} | patient_utility | high→4-5, medium→3, low→1-2 |
| overall_usable ∈ {yes/partial/no} | safety + patient_utility avg | yes→4-5, partial→3, no→1-2 |
| needs_domain_expert_review ∈ {yes/maybe/no} | safety (inverted) | no→5, maybe→3, yes→1-2 |

- [ ] **Step 1: Write mapping CSV manually + commit**

- [ ] **Step 2: Write compute_3way_agreement.py**

  Reads:
  - human-rater 15-case scores from `outputs/eval_runs_final/onboarding_eval_audit_summary_final.txt` (or similar; if not in that file, from `outputs/tables/onboarding_eval_audit_v2.csv` or wherever the 15-case rubric was logged)
  - LLM-judge from `outputs/phase4/15case_audit/judge_*.jsonl`

  Computes Cohen's κ (linear + quadratic) and Spearman ρ between (human ↔ Sonnet), (human ↔ GPT-4o), (Sonnet ↔ GPT-4o) for each of the 5 dims (after mapping the human's 10-dim to 5-dim per the table).

- [ ] **Step 3: Run + paste output**

  ```bash
  python scripts_phase4/compute_3way_agreement.py \
    --human-csv outputs/tables/onboarding_eval_audit_v2.csv \
    --judge-jsonl-sonnet outputs/phase4/15case_audit/judge_sonnet.jsonl \
    --judge-jsonl-gpt4o  outputs/phase4/15case_audit/judge_gpt4o.jsonl \
    --mapping-csv outputs/phase4/15case_audit/audit_rubric_mapping.csv \
    --out-csv outputs/phase4/15case_audit/3way_agreement_per_dim.csv \
    --out-json outputs/phase4/15case_audit/3way_agreement_summary.json
  ```

  Paste the summary JSON.

- [ ] **Step 4: Commit**

  `git add scripts_phase4/ outputs/phase4/15case_audit/ && git commit -m "feat(t1.1): 3-way agreement on 15-case audit"`

### Task 1.3: Therapeutic-area labeling for 30-case corpus

**Files:**
- Create: `scripts_phase4/label_therapeutic_areas.py`
- Output: `outputs/phase4/area_breakdown/area_labels_30case.csv`

**What:** call Sonnet with each of the 30 case texts + selected NCT title, return label ∈ {MSK_Bone, Cardiovascular, Metabolic_Endocrine, Oncology, Neurology_CNS, Other}. Manual spot-check on 10 random cases. Save to CSV.

- [ ] **Step 1: Read 30 cases**

  ```bash
  cd ~/Desktop/islamm11/avatar_trial_onboarding
  python -c "
  import pandas as pd
  df = pd.read_csv('outputs/tables/backbone_ablation_raw.csv')
  cases = df[['case_id','source','category']].drop_duplicates()
  print(cases.shape)
  print(cases.head(35))
  "
  ```

- [ ] **Step 2: Write labeler script**

  - Read each case's question text (from `data/processed/onboarding_eval_cases_expanded.json` if exists, else reconstruct from `outputs/backbone_gens/Qwen__Qwen2.5-3B-Instruct/<case_id>.json` `question` field).
  - For each case: prompt Sonnet "Classify this clinical-trial onboarding case into one of {MSK_Bone, Cardiovascular, Metabolic_Endocrine, Oncology, Neurology_CNS, Other} based on the patient question and the selected trial. Return exactly one label."
  - Write CSV: `case_id, area, source_text, sonnet_rationale`.

- [ ] **Step 3: Run + spot check**

  ```bash
  python scripts_phase4/label_therapeutic_areas.py \
    --cases outputs/backbone_gens/Qwen__Qwen2.5-3B-Instruct \
    --out outputs/phase4/area_breakdown/area_labels_30case.csv
  ```

  Then manually open the CSV and override any mislabels. Document overrides in a `manual_overrides_30case.txt`.

  Expected: 30 rows. Roughly 6-10 per area (or skewed toward MSK if the audit cases dominate).

- [ ] **Step 4: Aggregate per-area safety**

  ```bash
  python -c "
  import pandas as pd
  areas = pd.read_csv('outputs/phase4/area_breakdown/area_labels_30case.csv')
  raw = pd.read_csv('outputs/tables/backbone_ablation_raw.csv')
  m = raw.merge(areas[['case_id','area']], on='case_id')
  agg = m.groupby(['area','backbone']).agg(
      n=('case_id','nunique'),
      leak_rate=('cross_trial_leak_n', lambda s: (s>0).mean()),
      commit_rate=('decision', lambda s: (s.isin(['likely_match','possible_match_insufficient_evidence'])).mean()),
      abstain_rate=('decision', lambda s: (s=='cannot_determine').mean()),
  ).reset_index()
  agg.to_csv('outputs/phase4/area_breakdown/per_area_safety_summary.csv', index=False)
  print(agg.to_string())
  "
  ```

  Expected: leak_rate = 0 across all (area × backbone) cells. If any cell is non-zero — that's a real finding for the paper, not a bug.

### Task 1.4: Calibration metrics

**Files:**
- Create: `scripts_phase4/compute_calibration.py`
- Output: `outputs/phase4/calibration/{calibration_summary.json, reliability_diagram.{pdf,png}}`

**What:** ECE (10 bins), Brier score, reliability diagram. Use selector's normalized trial-score share τ (from `outputs/tables/selector_signals_cache.csv`) as the predicted probability.

- [ ] **Step 1: Verify selector signals cache exists + has expected columns**

  ```bash
  head -3 outputs/tables/selector_signals_cache.csv
  ```

  Expected columns: at minimum `case_id, rho, tau, mu, kappa, top_doc, gold_nct (if joinable), final_status`.

- [ ] **Step 2: Write calibration script**

  ```python
  # scripts_phase4/compute_calibration.py
  # Inputs: selector signals + gold NCT mapping
  # Output: ECE, Brier, reliability diagram per backbone
  #
  # For each (case, backbone):
  #   - p_pred = tau (or 1 - tau if predicting "no match")
  #   - y_true = 1 if selected_doc == gold_nct else 0
  # ECE: bin p_pred into 10 equal-width bins, compute |bin_acc - bin_conf|, weight by bin frac.
  # Brier: mean((p_pred - y_true)^2).
  # Reliability diagram: matplotlib bar chart of bin_conf vs bin_acc.
  ```

- [ ] **Step 3: Run**

  ```bash
  python scripts_phase4/compute_calibration.py \
    --selector-cache outputs/tables/selector_signals_cache.csv \
    --backbone-raw   outputs/tables/backbone_ablation_raw.csv \
    --out-summary    outputs/phase4/calibration/calibration_summary.json \
    --out-diagram    outputs/phase4/calibration/reliability_diagram.pdf
  ```

  Paste the summary JSON.

  **Expected:** ECE between 0.05 and 0.20 on the commit subset. Brier in 0.15-0.30 range. If ECE > 0.30, the system is poorly calibrated and we report it honestly + propose temperature scaling as a future fix.

- [ ] **Step 4: Commit**

---

## Phase 2 — Heavy compute (Week 2)

### Task 1.5: Per-gate ablation

**Files:**
- Create: `scripts_phase4/run_per_gate_ablation.py`
- Output: `outputs/phase4/per_gate_ablation/gens/{no-rho,no-tau,no-mu,no-kappa,no-generic}/<backbone>/<case>.json`

**What:** Modify the trial-first selector to disable one gate at a time. Run 5 leave-one-out variants × 30 cases × 4 backbones = 600 generations. Re-judge.

- [ ] **Step 1: Patch selector for runtime gate-disable flags**

  Existing selector in `scripts/run_onboarding_pipeline_trial_first.py` (or similar) has hard-coded thresholds. Add CLI flags:

  ```
  --disable-gate-rho        # ρ accept-all
  --disable-gate-tau        # τ accept-all
  --disable-gate-mu         # μ accept-all
  --disable-gate-kappa      # κ accept-all
  --disable-gate-generic    # generic-flag always false
  ```

  For each disabled gate, replace the threshold check with `True`. Confirm on 1 case it produces a different selector outcome than baseline.

- [ ] **Step 2: Write driver**

  ```python
  # scripts_phase4/run_per_gate_ablation.py
  # for each gate in ['rho','tau','mu','kappa','generic']:
  #   for each backbone in BACKBONES:
  #     for each case in CASES_30:
  #       run pipeline with --disable-gate-<gate>
  #       save JSON to outputs/phase4/per_gate_ablation/gens/no-<gate>/<backbone>/<case>.json
  ```

  Use vllm for Qwen-2.5-3B/7B; HF transformers for Llama and Mistral if simpler.

- [ ] **Step 3: Run on server (~6 h GPU)**

  ```bash
  nohup python scripts_phase4/run_per_gate_ablation.py \
    --cases outputs/backbone_gens/Qwen__Qwen2.5-3B-Instruct \
    --out outputs/phase4/per_gate_ablation/gens \
    --backbones Qwen/Qwen2.5-3B-Instruct Qwen/Qwen2.5-7B-Instruct meta-llama/Meta-Llama-3.1-8B-Instruct mistralai/Mistral-7B-Instruct-v0.3 \
    > logs/per_gate_ablation.log 2>&1 &
  ```

  Tail log periodically.

- [ ] **Step 4: Re-judge with dual-judge**

  ```bash
  python scripts_phase3/run_llm_judge.py \
    --gens-dir outputs/phase4/per_gate_ablation/gens \
    --out-dir  outputs/phase4/per_gate_ablation/ \
    --judges   sonnet gpt4o \
    --concurrency 6
  ```

  Wall time: ~3 h API. Cost: ~$20.

- [ ] **Step 5: Aggregate**

  Compute per-gate-disabled cross-trial leak rate, severe overstatement rate, severe unsupported rate, abstain rate, judge-pooled rubric mean. Compare to V-final baseline. Report which gate matters most.

  ```bash
  python scripts_phase4/aggregate_phase4_tables.py --task per_gate
  ```

  Output table goes into paper §5 as "Per-gate ablation".

- [ ] **Step 6: Commit**

### Task 1.6: n=70 expansion + re-judge

**Files:**
- Server-existing: `scripts/build_expanded_case_set.py` (drafted but never run)
- Server-existing: `scripts/run_backbone_ablation.py`
- Output: `outputs/phase4/n100_expansion/`

- [ ] **Step 1: Run case expansion**

  ```bash
  cd ~/Desktop/islamm11/avatar_trial_onboarding
  python scripts/build_expanded_case_set.py \
    --repo-root . \
    --target-n 100 \
    --seed 42 \
    --out data/processed/onboarding_eval_cases_n100.json
  ```

  Expected stdout: `wrote 100 cases ... breakdown: {'handcrafted': 15, 'trec2021': N1, 'trec2022': N2, 'paraphrase_*': N3, 'synthetic_persona_*': N4}`.

  Paste the breakdown back. If `synthetic_persona_*` count is 0 (script doesn't generate them), augment the script or accept the TREC-paraphrase-only mix and reframe the per-area discussion.

- [ ] **Step 2: Run backbone ablation on the additional 70**

  ```bash
  python scripts/run_backbone_ablation.py \
    --cases-json data/processed/onboarding_eval_cases_n100.json \
    --skip-cases-already-in outputs/backbone_gens \
    --out outputs/phase4/n100_expansion/gens \
    --backbones Qwen/Qwen2.5-3B-Instruct Qwen/Qwen2.5-7B-Instruct meta-llama/Meta-Llama-3.1-8B-Instruct mistralai/Mistral-7B-Instruct-v0.3
  ```

  Wall time: ~3 h GPU.

- [ ] **Step 3: Re-judge the new 70**

  ```bash
  python scripts_phase3/run_llm_judge.py \
    --gens-dir outputs/phase4/n100_expansion/gens \
    --out-dir  outputs/phase4/n100_expansion/ \
    --judges   sonnet gpt4o \
    --concurrency 6
  ```

  Wall: ~2 h API. Cost: ~$10.

- [ ] **Step 4: Merge with existing 30-case Phase 3 results + re-aggregate**

  ```bash
  python scripts_phase4/aggregate_phase4_tables.py --task n100_merge
  ```

  Outputs:
  - `n100_per_backbone_summary.csv` — replaces Table 7 in paper.
  - `n100_dualjudge_scores.csv`
  - Updated bootstrap CIs over n=100.

- [ ] **Step 5: Re-label therapeutic areas for new 70**

  ```bash
  python scripts_phase4/label_therapeutic_areas.py \
    --cases outputs/phase4/n100_expansion/gens/Qwen__Qwen2.5-3B-Instruct \
    --out outputs/phase4/area_breakdown/area_labels_n70.csv
  cat outputs/phase4/area_breakdown/area_labels_30case.csv outputs/phase4/area_breakdown/area_labels_n70.csv > outputs/phase4/area_breakdown/area_labels_n100.csv
  ```

- [ ] **Step 6: Commit**

---

## Phase 3 — External SOTA (Week 2-3)

### Task 1.2: TrialGPT comparison

**Files:**
- Create: `scripts_phase4/run_trialgpt_baseline.py`
- Output: `outputs/phase4/trialgpt/trialgpt_gens/<case_id>.json`

**Repo:** https://github.com/ncbi-nlp/TrialGPT (Jin et al. 2024, *Nature Communications*)

- [ ] **Step 1: Clone + verify TrialGPT runs**

  ```bash
  cd ~/Desktop/islamm11
  git clone https://github.com/ncbi-nlp/TrialGPT.git
  cd TrialGPT
  pip install -r requirements.txt   # in avatar_trial env
  ```

  Run their demo on 1 case to confirm. If it expects OpenAI API + GPT-4, set keys. If it accepts open-weights via their config, use Qwen-2.5-3B for fair comparison.

  **Decision point:** if TrialGPT requires GPT-4 or proprietary models, use the version that uses GPT-4 (since it's the canonical TrialGPT result) and document the unfair-comparison caveat ("we compare V-final-Qwen-3B vs TrialGPT-GPT-4 because no open-weight TrialGPT exists; the comparison favors TrialGPT and our safety advantage holds despite the disadvantage").

- [ ] **Step 2: Adapter — feed our 30-case corpus into TrialGPT**

  TrialGPT's input is (patient summary, list of trials). Our input is (patient utterance, top-k retrieved passages from BM25→cross-encoder).

  Two options:
  - **Strict:** feed TrialGPT only the trial that V-final selected. This isolates the eligibility-reasoning component.
  - **Fair:** feed TrialGPT the same top-k=20 candidates V-final saw before its trial-first gate. This includes the cross-trial conflation risk that V-final structurally blocks.

  Use **Fair**. The whole point is to show TrialGPT lacks the trial-first gate.

  Adapter script writes one JSON per case with TrialGPT's eligibility decision + reasoning + selected NCT.

- [ ] **Step 3: Run on 30 cases**

  ```bash
  python scripts_phase4/run_trialgpt_baseline.py \
    --cases outputs/backbone_gens/Qwen__Qwen2.5-3B-Instruct \
    --top-k 20 \
    --out outputs/phase4/trialgpt/trialgpt_gens/
  ```

  Wall: ~30 min if API; ~2 h if local. Cost: ~$5-10 if GPT-4.

- [ ] **Step 4: Dual-judge TrialGPT outputs**

  ```bash
  python scripts_phase3/run_llm_judge.py \
    --gens-dir outputs/phase4/trialgpt/trialgpt_gens/__as_backbone__ \
    --out-dir  outputs/phase4/trialgpt/ \
    --judges   sonnet gpt4o \
    --concurrency 6
  ```

  Need to wrap `trialgpt_gens/` in a `__as_backbone__` subdir to match `<gens_dir>/<backbone>/<case>.json` shape that `run_llm_judge.py` expects.

  Cost: ~$1.

- [ ] **Step 5: Side-by-side comparison table**

  ```bash
  python scripts_phase4/aggregate_phase4_tables.py --task trialgpt_vs_vfinal
  ```

  Output table:

  | System | Cross-trial leak | Severe overstatement | Severe unsupported | Judge-pooled overall | Source |
  |---|---|---|---|---|---|
  | TrialGPT (GPT-4) | X% | Y% | Z% | A.B | Jin et al. 2024 + our eval |
  | V-final (Qwen-3B) | 0% | 0% | 0% | 3.22 | This paper |
  | V-final (Qwen-7B) | 0% | 0% | 0% | 3.94 | This paper |

  **Expected:** TrialGPT shows non-zero cross-trial leak on broad/generic queries because it has no trial-first gate. Headline.

- [ ] **Step 6: Commit + email Jin et al. (optional)**

  Draft 4-line courtesy email after results in. Skip if you prefer.

---

## Phase 4 — Tier 2 (Week 3, if time)

### Task 2.1: BGE-reranker-large

```bash
cd ~/Desktop/islamm11/avatar_trial_onboarding
python scripts/run_bge_reranker_large.py \
  --repo-root . --device cuda --batch-size 8 --max-length 512 --topk 20

# Result: outputs/tables/bge_reranker_large_summary_metrics.csv
```

Paste summary one-row CSV. Compare to existing MiniLM cross-encoder result.

### Task 2.2: TREC CT 2022 retrieval

```bash
python scripts/run_bm25_baseline.py --year 2022   # if script supports it
python scripts/run_cross_encoder_rerank.py --year 2022
```

If scripts don't support 2022 directly, parameterize them. Adds ~30 min work.

### Task 2.3: Bootstrap CIs on retrieval metrics

```bash
python scripts/bootstrap_retrieval_cis.py --repo-root . --n-bootstrap 1000 --seed 42 \
  --out outputs/tables/retrieval_bootstrap_cis.csv
```

Paste the CSV.

### Task 2.4: NLI4PR eligibility-alignment

```bash
python scripts/eval_nli4pr_eligibility_alignment.py \
  --repo-root . --split test --n-sample 500 \
  --backbone Qwen/Qwen2.5-3B-Instruct --max-new-tokens 340
```

Paste the summary JSON.

### Task 2.5: Generic-flag sensitivity analysis

For each of: (a) +5 phrase additions, (b) -5 phrase removals, (c) replace rule-based with Sonnet "is this generic?" classifier — recompute V-final on the 30-case corpus and report leak / abstain / commit deltas.

---

## Phase 5 — Paper rewrite (Week 4)

### Task 1.7: Integrate + reframe + ship

**Files:**
- Modify: `outputs/paper_v2/main.tex`
- Modify: `outputs/paper_v2/references.bib` (add TrialGPT, FDA Diversity, etc., already in bib likely)

- [ ] **Step 1: Title**

  Replace title block to "An Auditable Pipeline for Patient-Facing Clinical Trial Onboarding". Move current title to subtitle if you want to keep the descriptive phrasing.

- [ ] **Step 2: Abstract rewrite**

  - Lead with the audit-and-safety story, not the retrieval numbers.
  - Quantify what's new: "n=100 case corpus across 5 therapeutic areas, 3-way agreement (human + 2 LLM judges) on 15-case audit, per-gate ablation showing the generic-flag and dominance gates carry safety, ECE = X on the eligibility decision, and a side-by-side comparison with TrialGPT showing 0% cross-trial leak vs Y% for TrialGPT."
  - Final humanizer pass.

- [ ] **Step 3: §3 Methods updates**

  - Add §3.X "External baseline: TrialGPT" subsection. Cite Jin et al. 2024.
  - Add §3.X "Calibration analysis" subsection. Document selector-τ-as-probability proxy.

- [ ] **Step 4: §4 Experimental setup updates**

  - Update §4.1 Datasets to mention n=100 expansion + per-area labeling.
  - Update §4.6/4.7 to reflect new n=100 corpus.
  - Add §4.X "Per-gate ablation protocol" subsection.

- [ ] **Step 5: §5 Results updates**

  - Add Table: Per-gate ablation (5 leave-one-out variants × 4 backbones).
  - Add Table: TrialGPT vs V-final.
  - Add Figure: Per-area safety heatmap.
  - Add Figure: Reliability diagram.
  - Update Table 7 (judge-pooled rubric scores) to n=100 numbers.
  - Add §5.X "3-way agreement on 15-case audit" subsection with κ table.

- [ ] **Step 6: §6 Discussion + Limitations**

  - Add subsection: "What the per-gate ablation reveals" — which gate carries safety.
  - Add subsection: "External SOTA comparison" — TrialGPT result discussion.
  - Limitations: explicitly position multi-rater clinician audit + multi-turn dialog + operator taxonomy as **Paper B in preparation**, not as gaps. This forward-references the line of research.

- [ ] **Step 7: Final humanizer pass + ruff/lint on the paper**

  Run `humanizer` skill on Abstract, Introduction §1, Discussion §6, Conclusion §7. Cut em-dashes, "moreover", "furthermore", inflated symbolism, rule-of-three patterns.

- [ ] **Step 8: Final compile**

  ```bash
  cd outputs/paper_v2
  pdflatex main.tex
  bibtex main
  pdflatex main.tex
  pdflatex main.tex
  ```

  Open the PDF, eyeball every figure, every table. Fix layout issues.

- [ ] **Step 9: Cover letter draft**

  3-paragraph npj DM cover letter:
  1. What problem we're solving + why it matters clinically.
  2. What's new in this submission (the audit-rigor improvements).
  3. Why npj DM is the right venue (impact + audit transparency match the journal).

- [ ] **Step 10: Submit**

  Paper A live. Pop champagne. Start Paper B.

---

## Done definition

- All 7 reviewer-attack surfaces in spec §2.1–2.2 addressed with new evidence in the paper.
- Paper PDF compiles to ~32-36 pp (current 31 + ~3-5 new pages of results).
- All scripts, intermediate CSVs, and final tables persisted under `outputs/phase4/` for reproducibility.
- npj DM submission drafted + cover letter written.

## When to stop and ask

- TrialGPT install fails or produces malformed JSON for >50% of cases. Stop, ask.
- Per-gate ablation reveals one gate matters > 80% (e.g., generic-flag carries everything). That changes the paper's "structural property of the guard" framing. Surface to user.
- n=100 expansion's judge-pooled scores diverge by > 0.5 from n=30 numbers on the safety dimension. That's signal, discuss before paper rewrite.
- Calibration ECE > 0.30. Honest result, but warrants temperature-scaling section discussion. Surface.

## Sync after each task

After every task completes on server:

```bash
# user runs from Mac
rsync -avz islamm11@zabi-nvidia-gpu.bme.rpi.edu:~/Desktop/islamm11/avatar_trial_onboarding/outputs/phase4/ \
  ~/Desktop/Clinical-Trial/outputs/phase4/
rsync -avz islamm11@zabi-nvidia-gpu.bme.rpi.edu:~/Desktop/islamm11/avatar_trial_onboarding/scripts_phase4/ \
  ~/Desktop/Clinical-Trial/scripts_phase4/
```

Paste any new files' first 30 lines if shape is unclear.

---

This plan is at `outputs/paper_v2/research_plan/2026-04-29-paper-A-polish-plan.md`. Spec at `outputs/paper_v2/research_plan/2026-04-29-paper-A-polish-spec.md`.

Status: ready to execute. First step Task 1.1A Step 1 — paste the existing 15-case eval JSON schema.
