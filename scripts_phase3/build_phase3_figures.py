"""
Phase 3 figures.

Inputs:
  tables/phase3/phase3_per_backbone.csv
  tables/phase3/phase3_per_backbone_pooled.csv
  tables/phase3/agreement_per_dim.csv
  tables/phase3/phase3_signal_corr.csv
  tables/phase3/phase3_failure_modes.csv

Outputs (all .pdf + .png):
  figures/phase3_radar_per_backbone
  figures/phase3_agreement_heatmap
  figures/phase3_signal_corr
  figures/phase3_failure_modes

Run:
  python scripts_phase3/build_phase3_figures.py
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TAB  = ROOT / "outputs" / "phase3"
FIG  = ROOT / "outputs" / "paper_v2" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

DIMS = ["factuality", "groundedness", "abstain_appropriateness",
        "safety", "patient_utility"]
DIM_LABELS = {
    "factuality":               "Factuality",
    "groundedness":             "Groundedness",
    "abstain_appropriateness":  "Abstain\napprop.",
    "safety":                   "Safety",
    "patient_utility":          "Patient\nutility",
}
SHORT_BB = {
    "Qwen/Qwen2.5-3B-Instruct":           "Qwen-2.5-3B",
    "Qwen/Qwen2.5-7B-Instruct":           "Qwen-2.5-7B",
    "meta-llama/Meta-Llama-3.1-8B-Instruct": "Llama-3.1-8B",
    "mistralai/Mistral-7B-Instruct-v0.3": "Mistral-7B",
}
BB_COLOR = {
    "Qwen/Qwen2.5-3B-Instruct":           "#2b8cbe",
    "Qwen/Qwen2.5-7B-Instruct":           "#4daf4a",
    "meta-llama/Meta-Llama-3.1-8B-Instruct": "#e41a1c",
    "mistralai/Mistral-7B-Instruct-v0.3": "#984ea3",
}

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 120,
    "savefig.bbox": "tight",
})


def save(fig, stem):
    fig.savefig(FIG / f"{stem}.pdf")
    fig.savefig(FIG / f"{stem}.png", dpi=200)
    print(f"[wrote] {stem}.pdf + .png")


# ────────────────────────────────────────────────────────────
# 1. Radar plot — mean score per dim per backbone (judge-pooled)
# ────────────────────────────────────────────────────────────
def fig_radar():
    df = pd.read_csv(TAB / "phase3_per_backbone_pooled.csv")
    pivot = df.pivot(index="backbone", columns="dim", values="mean")
    pivot = pivot[DIMS]  # consistent order
    angles = np.linspace(0, 2 * np.pi, len(DIMS), endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(6.2, 5.8))
    ax = fig.add_subplot(111, polar=True)
    for bb, row in pivot.iterrows():
        vals = row.tolist() + row.tolist()[:1]
        ax.plot(angles, vals, linewidth=1.5, label=SHORT_BB.get(bb, bb),
                color=BB_COLOR.get(bb, None))
        ax.fill(angles, vals, alpha=0.10,
                color=BB_COLOR.get(bb, None))
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([DIM_LABELS[d] for d in DIMS])
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(["1", "2", "3", "4", "5"])
    ax.set_ylim(0, 5)
    ax.set_title("Judge-pooled rubric scores per backbone "
                 "(mean, N=30 cases, 2 judges)", pad=18)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.18),
              ncols=2, frameon=False)
    save(fig, "phase3_radar_per_backbone")
    plt.close(fig)


# ────────────────────────────────────────────────────────────
# 2. Agreement heatmap — κ linear/quad/ICC/Spearman per dim
# ────────────────────────────────────────────────────────────
def fig_agreement_heatmap():
    f = TAB / "agreement_per_dim.csv"
    if not f.exists():
        print("[skip] agreement_per_dim.csv missing")
        return
    df = pd.read_csv(f)
    metric_cols = ["kappa_linear", "kappa_quadratic", "icc21",
                   "spearman", "exact_match"]
    metric_labs = [r"$\kappa_{\mathrm{lin}}$", r"$\kappa_{\mathrm{quad}}$",
                   "ICC(2,1)", r"Spearman $\rho$", "Exact"]
    mat = df.set_index("dim")[metric_cols].reindex(DIMS).to_numpy()

    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    im = ax.imshow(mat, cmap="RdYlGn", vmin=-0.2, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(metric_cols)))
    ax.set_xticklabels(metric_labs)
    ax.set_yticks(range(len(DIMS)))
    ax.set_yticklabels([DIM_LABELS[d].replace("\n", " ") for d in DIMS])
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=8,
                    color="black" if 0.2 < v < 0.8 else "white")
    ax.set_title("Sonnet-4 vs GPT-4o inter-rater agreement per rubric dim")
    cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cb.set_label("agreement", fontsize=8)
    save(fig, "phase3_agreement_heatmap")
    plt.close(fig)


# ────────────────────────────────────────────────────────────
# 3. Signal correlation — Spearman(ρ, judge score) grid
# ────────────────────────────────────────────────────────────
def fig_signal_corr():
    f = TAB / "phase3_signal_corr.csv"
    if not f.exists():
        print("[skip] phase3_signal_corr.csv missing")
        return
    df = pd.read_csv(f)
    # pool judges: mean Spearman across sonnet/gpt4o per (backbone, dim, signal)
    pool = (df.groupby(["backbone", "dim", "signal"])["spearman_rho"]
              .mean().reset_index())
    backbones = sorted(pool["backbone"].unique())
    sigs = ["dominance_ratio", "trial_score_share",
            "raw_max_cross", "best_rank"]
    sig_labs = [r"$\rho$", r"$\tau$", r"$\mu$", r"$\kappa$"]

    fig, axes = plt.subplots(1, len(backbones),
                             figsize=(2.6 * len(backbones), 3.0),
                             sharey=True)
    if len(backbones) == 1:
        axes = [axes]
    for ax, bb in zip(axes, backbones):
        sub = pool[pool["backbone"] == bb]
        mat = (sub.pivot(index="dim", columns="signal",
                         values="spearman_rho")
                  .reindex(index=DIMS, columns=sigs).to_numpy())
        im = ax.imshow(mat, cmap="coolwarm", vmin=-0.6, vmax=0.6,
                       aspect="auto")
        ax.set_xticks(range(len(sigs))); ax.set_xticklabels(sig_labs)
        ax.set_yticks(range(len(DIMS)))
        ax.set_yticklabels([DIM_LABELS[d].replace("\n"," ") for d in DIMS])
        ax.set_title(SHORT_BB.get(bb, bb), fontsize=9)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                if np.isnan(v):
                    continue
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                        fontsize=7,
                        color="black" if abs(v) < 0.4 else "white")
    cb = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02)
    cb.set_label(r"Spearman $\rho$ (judge score vs selector signal)",
                 fontsize=8)
    fig.suptitle("Selector signals correlate with rubric scores (judge-pooled)",
                 y=1.03)
    save(fig, "phase3_signal_corr")
    plt.close(fig)


# ────────────────────────────────────────────────────────────
# 4. Failure mode stacked bars (share of cases with each tag)
# ────────────────────────────────────────────────────────────
def fig_failure_modes():
    f = TAB / "phase3_failure_modes.csv"
    if not f.exists():
        print("[skip] phase3_failure_modes.csv missing")
        return
    df = pd.read_csv(f)
    # pool judges: mean share per (backbone, failure_mode)
    pool = (df.groupby(["backbone", "failure_mode"])["share"]
              .mean().reset_index())
    # drop "none"
    pool = pool[pool["failure_mode"] != "none"]
    backbones = [bb for bb in SHORT_BB if bb in pool["backbone"].unique()]
    tags = (pool.groupby("failure_mode")["share"].sum()
                .sort_values(ascending=False).index.tolist())
    if not tags:
        print("[skip] no failure tags")
        return
    x = np.arange(len(backbones))
    w = 0.8
    bottom = np.zeros(len(backbones))
    palette = plt.cm.tab10(np.linspace(0, 1, len(tags)))

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    for i, tag in enumerate(tags):
        vals = [float(pool[(pool["backbone"] == bb) &
                           (pool["failure_mode"] == tag)]["share"].sum())
                for bb in backbones]
        ax.bar(x, vals, width=w, bottom=bottom, label=tag,
               color=palette[i], edgecolor="black", linewidth=0.4)
        bottom = bottom + np.array(vals)
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT_BB.get(bb, bb) for bb in backbones],
                       rotation=15, ha="right")
    ax.set_ylabel("Avg share of cases flagged (pooled over judges)")
    ax.set_title("Failure modes by backbone — judge-reported tags")
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5),
              frameon=False, fontsize=8)
    save(fig, "phase3_failure_modes")
    plt.close(fig)


if __name__ == "__main__":
    print(f"[in ] {TAB}")
    print(f"[out] {FIG}")
    fig_radar()
    fig_agreement_heatmap()
    fig_signal_corr()
    fig_failure_modes()
    print("[done]")
