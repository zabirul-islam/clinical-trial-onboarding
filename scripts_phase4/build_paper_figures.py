"""
T1.7 step 7.1 — Build all paper-ready figures from phase4 outputs.

Produces 5 figures:
  1. per_area_heatmap.{pdf,png}              — 6 areas × 4 backbones leak/commit/parse_ok
  2. per_gate_ablation.{pdf,png}             — variant × backbone bars
  3. phase3_radar_per_backbone_n114.{pdf,png}  — 4 open-weight backbones, n=114
  4. phase4_radar_open_vs_closed.{pdf,png}   — 4 open + 2 closed APIs, judge-pooled where avail
  5. (calibration_reliability already produced; skip)

All figures saved to outputs/paper_v2/figures/.

Usage (server):
  python scripts_phase4/build_paper_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path
from math import pi

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/paper_v2/figures"
OUT.mkdir(parents=True, exist_ok=True)

BACKBONE_SHORT = {
    "Qwen/Qwen2.5-3B-Instruct": "Qwen-2.5-3B",
    "Qwen/Qwen2.5-7B-Instruct": "Qwen-2.5-7B",
    "meta-llama/Meta-Llama-3.1-8B-Instruct": "Llama-3.1-8B",
    "mistralai/Mistral-7B-Instruct-v0.3": "Mistral-7B",
}
BACKBONE_ORDER = list(BACKBONE_SHORT.keys())
BACKBONE_COLORS = {
    "Qwen-2.5-3B": "#1f77b4",
    "Qwen-2.5-7B": "#2ca02c",
    "Llama-3.1-8B": "#d62728",
    "Mistral-7B": "#9467bd",
    "GPT-4o": "#ff7f0e",
    "Sonnet 4.5": "#8c564b",
}
AREA_ORDER = ["MSK_Bone", "Cardiovascular", "Metabolic_Endocrine",
              "Oncology", "Neurology_CNS", "Other"]
JUDGE_DIMS = ["factuality", "groundedness", "abstain_appropriateness",
              "safety", "patient_utility"]
JUDGE_DIM_LABELS = ["Factuality", "Groundedness", "Abstain\nappropriate",
                    "Safety", "Patient\nutility"]


# ----------------------------- Figure 1: per-area heatmap ----------------------

def fig_per_area_heatmap():
    df = pd.read_csv(ROOT / "outputs/phase4/n114_aggregate/summary_n114_per_area.csv")
    df = df[df["area"].isin(AREA_ORDER)]

    metrics = ["leak_rate", "commit_rate", "parse_ok"]
    titles = ["Cross-trial leak rate", "Commit rate", "Parse OK rate"]
    cmaps = ["Reds", "YlGnBu", "Greens"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, metric, title, cmap in zip(axes, metrics, titles, cmaps):
        pivot = df.pivot_table(index="area", columns="backbone",
                               values=metric, aggfunc="first")
        pivot = pivot.reindex(index=AREA_ORDER, columns=BACKBONE_ORDER)
        # Rename backbone column headers for display
        pivot.columns = [BACKBONE_SHORT[c] for c in pivot.columns]

        im = ax.imshow(pivot.values, cmap=cmap, aspect="auto",
                       vmin=0, vmax=1.0 if metric != "leak_rate" else 0.05)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=30, ha="right", fontsize=9)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=9)
        ax.set_title(title, fontsize=11)
        # cell annotations
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                v = pivot.values[i, j]
                txt = f"{v:.2f}" if not pd.isna(v) else "—"
                color = "white" if (cmap != "Reds" and v > 0.5) else "black"
                ax.text(j, i, txt, ha="center", va="center",
                        fontsize=8, color=color)
        plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)

    fig.suptitle("Per-therapeutic-area safety on n=114 cases × 4 open-weight backbones "
                 "(zero leak in every cell)", fontsize=12)
    plt.tight_layout()
    fig.savefig(OUT / "per_area_heatmap.pdf", bbox_inches="tight")
    fig.savefig(OUT / "per_area_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[wrote] {OUT}/per_area_heatmap.{{pdf,png}}")


# ----------------------------- Figure 2: per-gate ablation ---------------------

def fig_per_gate_ablation():
    summary = pd.read_csv(ROOT / "outputs/phase4/per_gate_ablation/per_gate_summary.csv")
    only = pd.read_csv(ROOT / "outputs/phase4/per_gate_ablation/per_gate_only_this_gate.csv")
    headline = json.loads(
        (ROOT / "outputs/phase4/per_gate_ablation/per_gate_summary.json").read_text()
    )

    variants = ["V-final", "V-final-no-rho", "V-final-no-tau", "V-final-no-mu",
                "V-final-no-kappa", "V-final-no-generic"]
    variant_labels = ["V-final", r"no $\rho$", r"no $\tau$", r"no $\mu$",
                      r"no $\kappa$", "no generic-flag"]
    n_acc = [headline["per_variant_n_accepted"][v] for v in variants]
    n_only = [headline["per_variant_n_only_this_gate"][v] for v in variants]
    leak_max = [headline["per_variant_max_leak_across_backbones"][v] for v in variants]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2),
                             gridspec_kw={"width_ratios": [1, 1.3]})

    # Left: cases admitted per variant
    ax = axes[0]
    x = np.arange(len(variants))
    w = 0.4
    bars1 = ax.bar(x - w/2, n_acc, w, label="Total accepted",
                   color="#1f77b4", edgecolor="black", linewidth=0.5)
    bars2 = ax.bar(x + w/2, n_only, w, label="Only this gate",
                   color="#aec7e8", edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(variant_labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Cases (out of n=114 pool)", fontsize=10)
    ax.set_title("Selector acceptance per leave-one-out variant", fontsize=11)
    ax.legend(fontsize=9, loc="upper left")
    for b, v in zip(bars1, n_acc):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 1,
                str(v), ha="center", fontsize=8)
    for b, v in zip(bars2, n_only):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 1,
                str(v), ha="center", fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # Right: leak rate matrix per variant × backbone (all zeros — show as table-like)
    ax = axes[1]
    pivot = summary.pivot_table(index="variant", columns="backbone",
                                values="leak_rate", aggfunc="first")
    pivot = pivot.reindex(index=variants, columns=BACKBONE_ORDER)
    pivot.columns = [BACKBONE_SHORT[c] for c in pivot.columns]
    pivot.index = variant_labels
    im = ax.imshow(pivot.values, cmap="Reds", aspect="auto", vmin=0, vmax=0.05)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=20, ha="right", fontsize=9)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)
    ax.set_title("Cross-trial leak rate × backbone (all 0.00)", fontsize=11)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                    fontsize=9, color="black")
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)

    fig.suptitle("Per-gate ablation: removing any single gate keeps leak at zero "
                 "across all 4 backbones (n=114 pool)", fontsize=12)
    plt.tight_layout()
    fig.savefig(OUT / "per_gate_ablation.pdf", bbox_inches="tight")
    fig.savefig(OUT / "per_gate_ablation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[wrote] {OUT}/per_gate_ablation.{{pdf,png}}")


# ----------------------------- Figure 3: radar n=114 ---------------------------

def _radar(ax, dims, values_dict, colors, fill_alpha=0.10, lw=2):
    angles = [n / float(len(dims)) * 2 * pi for n in range(len(dims))]
    angles += angles[:1]
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(0)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dims, fontsize=9)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(["1", "2", "3", "4", "5"], fontsize=8)
    ax.set_ylim(0, 5)
    for label, vals in values_dict.items():
        v = list(vals) + [vals[0]]
        ax.plot(angles, v, lw=lw, label=label, color=colors.get(label, None))
        ax.fill(angles, v, alpha=fill_alpha, color=colors.get(label, None))


def fig_radar_n114():
    df = pd.read_csv(ROOT / "outputs/phase4/n114_aggregate/judge_pooled_n114.csv")
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, polar=True)
    values_dict = {}
    for _, r in df.iterrows():
        bb_long = r["backbone"]
        if bb_long not in BACKBONE_SHORT:
            continue
        label = BACKBONE_SHORT[bb_long]
        vals = [r[f"{d}_mean"] for d in JUDGE_DIMS]
        values_dict[label] = vals
    _radar(ax, JUDGE_DIM_LABELS, values_dict, BACKBONE_COLORS)
    ax.set_title("Judge-pooled rubric scores (n=114, 2 judges)", fontsize=11, pad=18)
    ax.legend(loc="lower right", bbox_to_anchor=(1.25, -0.05), fontsize=9)
    plt.tight_layout()
    fig.savefig(OUT / "phase3_radar_per_backbone_n114.pdf", bbox_inches="tight")
    fig.savefig(OUT / "phase3_radar_per_backbone_n114.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[wrote] {OUT}/phase3_radar_per_backbone_n114.{{pdf,png}}")


# ----------------------------- Figure 4: open vs closed bars -------------------

def fig_open_vs_closed_bars():
    """6-backbone safety bar chart (no judge scores for closed APIs yet)."""
    open_df = pd.read_csv(ROOT / "outputs/phase4/n114_aggregate/summary_n114_per_backbone.csv")
    closed_df = pd.read_csv(ROOT / "outputs/phase4/zeroshot_baseline/zeroshot_summary.csv")

    # Build unified dataframe
    rows = []
    for _, r in open_df.iterrows():
        bb_long = r["backbone"]
        if bb_long not in BACKBONE_SHORT:
            continue
        rows.append({
            "model": BACKBONE_SHORT[bb_long],
            "family": "open-weight",
            "parse_ok": float(r["parse_ok"]),
            "leak_rate": float(r["leak_rate"]),
            "commit_rate": float(r["commit_rate"]),
            "abstain_rate": float(r["abstain_rate"]),
        })
    closed_map = {"gpt-4o": "GPT-4o", "claude-sonnet-4-5-20250929": "Sonnet 4.5"}
    for _, r in closed_df.iterrows():
        if r["model"] not in closed_map:
            continue
        rows.append({
            "model": closed_map[r["model"]],
            "family": "closed-API",
            "parse_ok": float(r["parse_ok_rate"]),
            "leak_rate": float(r["leak_rate"]),
            "commit_rate": float(r["commit_rate"]),
            "abstain_rate": float(r["abstain_rate"]),
        })
    df = pd.DataFrame(rows)
    order = ["Qwen-2.5-3B", "Qwen-2.5-7B", "Llama-3.1-8B", "Mistral-7B",
             "GPT-4o", "Sonnet 4.5"]
    df["model"] = pd.Categorical(df["model"], categories=order, ordered=True)
    df = df.sort_values("model").reset_index(drop=True)

    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    metrics = ["parse_ok", "leak_rate", "commit_rate", "abstain_rate"]
    titles = ["Parse OK rate", "Cross-trial leak rate (all 0)",
              "Commit rate", "Abstain rate"]

    family_color = {"open-weight": "#1f77b4", "closed-API": "#d62728"}
    for ax, metric, title in zip(axes, metrics, titles):
        x = np.arange(len(df))
        bars = ax.bar(x, df[metric], color=[family_color[f] for f in df["family"]],
                      edgecolor="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(df["model"], rotation=30, ha="right", fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0, max(0.05, float(df[metric].max()) * 1.2) if metric == "leak_rate" else 1.05)
        ax.grid(axis="y", alpha=0.3)
        for b, v in zip(bars, df[metric]):
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.02,
                    f"{v:.2f}", ha="center", fontsize=8)

    # Legend (one for whole figure)
    handles = [plt.Rectangle((0,0),1,1, color="#1f77b4"),
               plt.Rectangle((0,0),1,1, color="#d62728")]
    fig.legend(handles, ["Open-weight (Qwen, Llama, Mistral)",
                         "Closed-API zero-shot (GPT-4o, Sonnet 4.5)"],
               loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 1.04), fontsize=10)
    fig.suptitle("Safety + behavior across 6 model families on identical guarded evidence (n=114)",
                 fontsize=11, y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT / "phase4_open_vs_closed_bars.pdf", bbox_inches="tight")
    fig.savefig(OUT / "phase4_open_vs_closed_bars.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[wrote] {OUT}/phase4_open_vs_closed_bars.{{pdf,png}}")


# ----------------------------- main --------------------------------------------

def main():
    print(f"[out] {OUT}")
    fig_per_area_heatmap()
    fig_per_gate_ablation()
    fig_radar_n114()
    fig_open_vs_closed_bars()
    print("[done] 4 figures generated.")


if __name__ == "__main__":
    main()
