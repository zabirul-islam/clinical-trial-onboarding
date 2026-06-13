#!/usr/bin/env python3
"""
Phase 5 Task 7 — figures for the npj revision.

Produces into outputs/paper_v2/figures/:
  phase5_baseline_leak_bars.{pdf,png}   lexical + semantic leak per baseline vs V-final
  phase5_taxonomy_breakdown.{pdf,png}   T1/T2/T3 stacked per system
  phase5_safety_utility_frontier.{pdf,png}  semantic-leak (y) vs utility (x), per system

Reads only persisted CSVs. Utility = GPT-4o rubric overall mean (phase-5 baselines
+ phase-4 V-final). Canonical labels Qwen-2.5-3B / Qwen-2.5-7B.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "outputs/paper_v2/figures"
FIG.mkdir(parents=True, exist_ok=True)
P5 = ROOT / "outputs/phase5"

SYS_LABEL = {
    "V-final (guarded)": "V-final\n(single-trial guard)",
    "B1_multi_rag": "B1\nmulti-trial RAG",
    "B2_prompt_guard": "B2\nprompt-only guard",
    "B3_citation_enforced": "B3\ncitation-enforced",
    "B4_top1": "B4\ntop-1 selection",
}
ORDER = ["B1_multi_rag", "B2_prompt_guard", "B3_citation_enforced", "B4_top1", "V-final (guarded)"]
DIMS = ["factuality", "groundedness", "abstain_appropriateness", "safety", "patient_utility"]


def fig_leak_bars():
    t = pd.read_csv(P5 / "leak_taxonomy_master.csv").set_index("system").reindex(ORDER)
    x = np.arange(len(t)); w = 0.38
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar(x - w/2, t["lexical_wide_leak"], w, label="Lexical (wide) leak", color="#1f77b4")
    ax.bar(x + w/2, t["semantic_consensus"], w, label="Semantic leak (dual-judge consensus)", color="#d62728")
    ax.set_xticks(x); ax.set_xticklabels([SYS_LABEL[s] for s in t.index], fontsize=8)
    ax.set_ylabel("Cross-trial leak rate")
    ax.set_title("Cross-trial leakage by system: structural single-trial control vs alternatives")
    ax.legend(fontsize=8, frameon=False)
    for i, (lx, sm) in enumerate(zip(t["lexical_wide_leak"], t["semantic_consensus"])):
        ax.text(i - w/2, lx + 0.01, f"{lx:.2f}", ha="center", fontsize=7)
        ax.text(i + w/2, sm + 0.01, f"{sm:.2f}", ha="center", fontsize=7)
    plt.tight_layout()
    fig.savefig(FIG / "phase5_baseline_leak_bars.pdf", bbox_inches="tight")
    fig.savefig(FIG / "phase5_baseline_leak_bars.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("[wrote] phase5_baseline_leak_bars")


def fig_taxonomy():
    t = pd.read_csv(P5 / "leak_taxonomy_master.csv").set_index("system")
    bl = [s for s in ORDER if s.startswith("B")]
    sub = t.reindex(bl)
    T1 = pd.to_numeric(sub["T1"], errors="coerce").fillna(0)
    T2 = pd.to_numeric(sub["T2"], errors="coerce").fillna(0)
    T3 = pd.to_numeric(sub["T3"], errors="coerce").fillna(0)
    x = np.arange(len(sub))
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.bar(x, T1, label="T1 cross-trial contamination", color="#d62728")
    ax.bar(x, T2, bottom=T1, label="T2 unsupported completion", color="#ff7f0e")
    ax.bar(x, T3, bottom=T1 + T2, label="T3 ordinary hallucination", color="#7f7f7f")
    ax.set_xticks(x); ax.set_xticklabels([SYS_LABEL[s] for s in sub.index], fontsize=8)
    ax.set_ylabel("Consensus-flagged generations (dual-judge)")
    ax.set_ylim(0, float((T1 + T2 + T3).max()) * 1.18)
    ax.set_title("Failure taxonomy by system: structural control removes T1, not T2/T3",
                 fontsize=10, pad=24)
    # legend above the plot (ncol=3) so it never overlaps the tall T1 bars
    ax.legend(fontsize=8, frameon=False, ncol=3, loc="lower center",
              bbox_to_anchor=(0.5, 1.01), columnspacing=1.4, handletextpad=0.4)
    plt.tight_layout()
    fig.savefig(FIG / "phase5_taxonomy_breakdown.pdf", bbox_inches="tight")
    fig.savefig(FIG / "phase5_taxonomy_breakdown.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("[wrote] phase5_taxonomy_breakdown")


def _rubric_overall(jsonl: Path) -> float | None:
    if not jsonl.exists():
        return None
    vals = []
    for line in jsonl.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        s = d.get("scores") or d
        dimv = [s.get(k) for k in DIMS if isinstance(s.get(k), (int, float))]
        if dimv:
            vals.append(np.mean(dimv))
    return float(np.mean(vals)) if vals else None


def _rubric_pooled(baseline_dir: Path) -> float | None:
    """Dual-judge pooled overall: average the per-judge overall means when both exist."""
    judges = [_rubric_overall(baseline_dir / f"judge_{j}.jsonl") for j in ("sonnet", "gpt4o")]
    judges = [u for u in judges if u is not None]
    return float(np.mean(judges)) if judges else None


def fig_frontier():
    t = pd.read_csv(P5 / "leak_taxonomy_master.csv").set_index("system")
    pts = []
    # baselines: dual-judge pooled rubric overall (falls back to whichever judge exists)
    for b in ["B1_multi_rag", "B2_prompt_guard", "B3_citation_enforced", "B4_top1"]:
        u = _rubric_pooled(P5 / "rubric_judge" / b)
        if u is not None:
            pts.append((b, u, float(t.loc[b, "semantic_consensus"])))
    # V-final: existing phase-4 judge-pooled overall
    jp = ROOT / "outputs/phase4/n114_aggregate/judge_pooled_n114.csv"
    if jp.exists():
        df = pd.read_csv(jp)
        meancols = [f"{d}_mean" for d in DIMS if f"{d}_mean" in df.columns]
        if meancols:
            u = float(df[meancols].mean(axis=1).mean())
            pts.append(("V-final (guarded)", u, float(t.loc["V-final (guarded)", "semantic_consensus"])))
    if not pts:
        print("[skip] frontier — no rubric scores yet")
        return
    # per-point label offsets so the two close low-leak points (B4, V-final) don't collide
    OFFSET = {
        "B4_top1": (0, -16, "center"),
        "V-final (guarded)": (8, 10, "left"),
        "B1_multi_rag": (10, 13, "left"),
        "B2_prompt_guard": (10, 2, "left"),
        "B3_citation_enforced": (0, 12, "center"),
    }
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for name, u, leak in pts:
        ax.scatter(u, leak, s=120, zorder=3)
        dx, dy, ha = OFFSET.get(name, (8, 4, "left"))
        ax.annotate(SYS_LABEL[name].replace("\n", " "), (u, leak),
                    textcoords="offset points", xytext=(dx, dy), fontsize=8, ha=ha)
    ax.set_xlabel("Utility — dual-judge rubric overall mean (1–5)")
    ax.set_ylabel("Semantic cross-trial leak rate (dual-judge consensus)")
    ax.set_title("Safety–utility: single-trial control reaches low leakage\nwithout sacrificing utility")
    ax.set_ylim(-0.05, 0.72)
    ax.margins(x=0.18)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIG / "phase5_safety_utility_frontier.pdf", bbox_inches="tight")
    fig.savefig(FIG / "phase5_safety_utility_frontier.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[wrote] phase5_safety_utility_frontier ({len(pts)} systems)")


def main():
    fig_leak_bars()
    fig_taxonomy()
    fig_frontier()
    print("[done]")


if __name__ == "__main__":
    main()
