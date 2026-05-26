# Phase 2 — Pipeline experiments

Three scripts, three artifacts:

## 2.1 Cache selector signals (~25-35 min GPU)

```bash
cd ~/Desktop/islamm11/avatar_trial_onboarding
conda activate avatar_trial
python scripts/cache_selector_scores.py --repo-root . --top-docs 20 --top-passages 8
```

Output: `outputs/tables/selector_signals_cache.csv` (150 rows × signal cols).

Use `--resume` to pick up after crashes.

## 2.2 Threshold grid sweep (~30 sec CPU)

```bash
python scripts/run_threshold_sweep.py --repo-root .
```

Sweeps 8 ρ × 7 τ × 7 μ × 5 κ = 1,960 combos. Outputs:
- `threshold_sweep_grid.csv`   (full grid, 1960 rows)
- `threshold_sweep_pareto.csv` (Pareto frontier on F1 vs generic-abstain rate)
- `threshold_sweep_summary.json` (default config metrics + best-F1 config)

Paste the `default_metrics` + `best_f1_config` sections.

## 2.3 Backbone ablation (~2-3 hrs total GPU, 4 backbones × 30 cases)

```bash
python scripts/run_backbone_ablation.py --repo-root . --n-cases 30
```

Loads 4 backbones sequentially; 30 accepted cases each. Outputs:
- `backbone_ablation_raw.csv`     (per-case-per-backbone)
- `backbone_ablation_summary.csv` (parse-ok, latency, commit rate, leak rate)

**If time-constrained**, run fewer backbones:

```bash
python scripts/run_backbone_ablation.py --repo-root . --n-cases 30 \
    --backbones Qwen/Qwen2.5-3B-Instruct Qwen/Qwen2.5-7B-Instruct
```

## Order of operations

1. Run 2.1 first. **Blocking** — 2.2 and 2.3 both read its cache.
2. Run 2.2 while 2.3 runs (2.2 is CPU-only, 2.3 needs GPU). Parallel OK.
3. Paste all three outputs.
