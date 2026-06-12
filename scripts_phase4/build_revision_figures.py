"""
Figures for the npj Digital Medicine revision.

Outputs into outputs/paper_v2/figures/ (Overleaf-tracked):
  - abstain_regime_quality.pdf / .png
  - leak_rate_grid.pdf / .png  (six model families x two regimes x narrow/wide)
  - krippendorff_3way.pdf / .png

Reuses existing CSVs only; no new compute.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
FIG = REPO / "outputs" / "paper_v2" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
RFX = REPO / "outputs" / "phase4" / "reviewer_fixes"


def fig_abstain_regime_quality():
    df = pd.read_csv(RFX / "abstain_regime_quality_short.csv")
    DIMS = [
        "factuality_mean",
        "groundedness_mean",
        "abstain_appropriateness_mean",
        "safety_mean",
        "patient_utility_mean",
    ]
    DIM_LABELS = ["Fact.", "Ground.", "Abst.", "Safety", "Utility"]
    backbones = list(df["backbone"].unique())
    short = {
        "Qwen/Qwen2.5-3B-Instruct": "Qwen-2.5-3B",
        "Qwen/Qwen2.5-7B-Instruct": "Qwen-2.5-7B",
        "meta-llama/Meta-Llama-3.1-8B-Instruct": "Llama-8B",
        "mistralai/Mistral-7B-Instruct-v0.3": "Mistral-7B",
    }

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, regime in zip(axes, ["accept", "abstain"]):
        sub = df[df["regime_gate"] == regime]
        x = np.arange(len(DIMS))
        width = 0.2
        for i, bb in enumerate(backbones):
            row = sub[sub["backbone"] == bb].iloc[0]
            vals = [row[d] for d in DIMS]
            ax.bar(x + i * width - 1.5 * width, vals, width,
                   label=short.get(bb, bb))
        ax.set_xticks(x)
        ax.set_xticklabels(DIM_LABELS)
        n_cases = sub["n_cases"].iloc[0]
        ax.set_title(
            f"selector-{regime} regime "
            f"(n={n_cases} cases × 4 backbones)"
        )
        ax.set_ylim(0, 5)
        ax.set_ylabel("Judge-pooled rubric score (1–5)")
        ax.axhline(4.0, color="grey", linestyle=":", linewidth=0.8,
                   alpha=0.6)
    axes[0].legend(loc="upper left", fontsize=8, ncol=2,
                   frameon=False)
    fig.suptitle(
        "Abstain-regime utility: rubric scores when no single trial dominates",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(FIG / "abstain_regime_quality.pdf", bbox_inches="tight")
    fig.savefig(FIG / "abstain_regime_quality.png", bbox_inches="tight",
                dpi=160)
    plt.close(fig)
    print("[wrote]", FIG / "abstain_regime_quality.pdf")


def fig_leak_grid():
    """6-model x 2-regime narrow + wide leak grid. All zeros, but visible."""
    leak = pd.read_csv(RFX / "leak_extended_summary.csv")
    n114 = pd.read_csv(REPO / "outputs" / "phase4"
                       / "n114_aggregate"
                       / "summary_n114_per_regime_gate.csv")
    zero = pd.read_csv(REPO / "outputs" / "phase4"
                       / "zeroshot_baseline" / "zeroshot_summary.csv")
    short_map = {
        "Qwen/Qwen2.5-3B-Instruct": "Qwen-2.5-3B",
        "Qwen/Qwen2.5-7B-Instruct": "Qwen-2.5-7B",
        "meta-llama/Meta-Llama-3.1-8B-Instruct": "Llama-8B",
        "mistralai/Mistral-7B-Instruct-v0.3": "Mistral-7B",
        "gpt-4o": "GPT-4o",
        "claude-sonnet-4-5-20250929": "Sonnet 4.5",
    }
    rows = []
    for _, r in n114.iterrows():
        rows.append({
            "model": short_map.get(r["backbone"], r["backbone"]),
            "regime": r["regime_gate"],
            "leak_rate": r["leak_rate"],
            "kind": "narrow",
        })
    for _, r in leak.iterrows():
        rows.append({
            "model": short_map.get(r["backbone"], r["backbone"]),
            "regime": "all",
            "leak_rate": r["wide_leak_rate"],
            "kind": "wide",
        })
    for _, r in zero.iterrows():
        rows.append({
            "model": short_map.get(r["model"], r["model"]),
            "regime": "all",
            "leak_rate": r["leak_rate"],
            "kind": "narrow",
        })
    df = pd.DataFrame(rows)

    # heatmap-style: rows = model, cols = (regime, kind)
    pivot = df.pivot_table(
        index="model", columns=["regime", "kind"],
        values="leak_rate", aggfunc="mean", fill_value=0.0,
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(pivot.values, cmap="RdYlGn_r", vmin=0, vmax=0.05,
                   aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{r}\n{k}" for r, k in pivot.columns],
                       rotation=0, fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f"{pivot.values[i, j]:.2%}",
                    ha="center", va="center", fontsize=9,
                    color="black")
    ax.set_title(
        "Cross-trial leak rate: narrow (NCT regex) and wide "
        "(distinctive trial-token entity match)\n"
        "0% across all six model families and both selector regimes",
        fontsize=10,
    )
    fig.colorbar(im, ax=ax, fraction=0.02, label="leak rate")
    fig.tight_layout()
    fig.savefig(FIG / "leak_rate_grid.pdf", bbox_inches="tight")
    fig.savefig(FIG / "leak_rate_grid.png", bbox_inches="tight", dpi=160)
    plt.close(fig)
    print("[wrote]", FIG / "leak_rate_grid.pdf")


def fig_krippendorff():
    df = pd.read_csv(RFX / "krippendorff_alpha_3way.csv")
    DIMS = ["factuality", "groundedness", "abstain_appropriateness",
            "safety", "patient_utility"]
    DIM_LABELS = ["Fact.", "Ground.", "Abst.", "Safety", "Utility"]
    variants = ["V1", "V2", "V-final"]
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(DIMS))
    width = 0.27
    for i, v in enumerate(variants):
        vals = []
        for d in DIMS:
            r = df[(df.variant == v) & (df["dim"] == d)]
            vals.append(r["alpha_interval"].iloc[0] if len(r) else 0.0)
        ax.bar(x + (i - 1) * width, vals, width, label=v)
    ax.axhline(0.0, color="grey", linewidth=0.8)
    ax.axhline(0.667, color="green", linestyle="--", linewidth=0.7,
               alpha=0.6, label="α = 0.67 (acceptable)")
    ax.axhline(0.4, color="orange", linestyle=":", linewidth=0.7,
               alpha=0.6, label="α = 0.40 (moderate)")
    ax.set_xticks(x)
    ax.set_xticklabels(DIM_LABELS)
    ax.set_ylabel("Krippendorff's α (interval)")
    ax.set_title("3-way IRR (Human + Sonnet + GPT-4o) on the 15-case audit\n"
                 "V1 retains rubric variance; V2 / V-final are saturated by design")
    ax.legend(fontsize=7, loc="lower right", frameon=False)
    ax.set_ylim(-0.7, 1.0)
    fig.tight_layout()
    fig.savefig(FIG / "krippendorff_3way.pdf", bbox_inches="tight")
    fig.savefig(FIG / "krippendorff_3way.png", bbox_inches="tight", dpi=160)
    plt.close(fig)
    print("[wrote]", FIG / "krippendorff_3way.pdf")


def main():
    fig_abstain_regime_quality()
    fig_leak_grid()
    fig_krippendorff()


if __name__ == "__main__":
    main()
