"""Build clinical deployment workflow figure (Fig 5 of npj-style paper)."""
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D

REPO = Path(__file__).resolve().parents[1]
FIG = REPO / "outputs" / "paper_v2" / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def _box(ax, x, y, w, h, text, fc, ec="#333", fontsize=9, fontweight="normal"):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.012,rounding_size=0.022",
        linewidth=1.2, edgecolor=ec, facecolor=fc,
    )
    ax.add_patch(box)
    ax.text(
        x + w / 2, y + h / 2, text,
        ha="center", va="center",
        fontsize=fontsize, fontweight=fontweight,
        wrap=True,
    )


def _arrow(ax, x1, y1, x2, y2, text=None, color="#444", style="-|>",
           text_offset_y=0.012, text_ha="center"):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle=style, color=color, lw=1.4,
                        shrinkA=4, shrinkB=4),
    )
    if text:
        ax.text(
            (x1 + x2) / 2, (y1 + y2) / 2 + text_offset_y,
            text, ha=text_ha, va="bottom", fontsize=7.8, color=color,
        )


def main():
    # Larger canvas; more vertical breathing room between lanes.
    fig, ax = plt.subplots(figsize=(13.5, 7.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # ---- Lane labels (left edge). Wider lanes; pushed apart vertically.
    LANES = [
        (0.860, "Patient", "#E8F4FD"),
        (0.660, "Onboarding pipeline", "#F4ECFA"),
        (0.420, "Study coordinator", "#FFF8E1"),
        (0.170, "Audit / governance", "#F1F8E9"),
    ]
    LANE_H = 0.165
    for y, lab, fc in LANES:
        ax.add_patch(FancyBboxPatch(
            (0.005, y - LANE_H / 2), 0.99, LANE_H,
            boxstyle="round,pad=0.005,rounding_size=0.013",
            linewidth=0, facecolor=fc, alpha=0.55, zorder=0,
        ))
        ax.text(0.013, y + LANE_H / 2 - 0.012, lab,
                ha="left", va="top",
                fontsize=9, fontweight="bold", color="#555")

    # ---- Patient lane
    _box(ax, 0.13, 0.815, 0.16, 0.095, "Patient query\n(plain language)",
         fc="#FFFFFF", ec="#1976D2", fontweight="bold")
    _box(ax, 0.82, 0.815, 0.15, 0.095, "Teach-back\nclarifications",
         fc="#FFFFFF", ec="#1976D2")

    # ---- Pipeline lane (5 boxes spread out; wider spacing)
    PIPE_Y, PIPE_H = 0.615, 0.095
    pipe_boxes = [
        (0.13, 0.10, "Retrieval\n(BM25 + rerank)", "#7B1FA2", "normal"),
        (0.265, 0.10, "Trial-first\nselector\n(5-gate)", "#7B1FA2", "bold"),
        (0.400, 0.10, "Eligibility\ntriage", "#7B1FA2", "normal"),
        (0.535, 0.10, "Consent-style\nexplanation", "#7B1FA2", "normal"),
        (0.670, 0.10, "Teach-back\ngenerator", "#7B1FA2", "normal"),
    ]
    for x, w, text, ec, fw in pipe_boxes:
        _box(ax, x, PIPE_Y, w, PIPE_H, text, fc="#FFFFFF", ec=ec,
             fontsize=9, fontweight=fw)

    # Abstain decision box — pushed below pipeline lane, off-center
    # so it doesn't cross the "bundle build" arrow path.
    _box(ax, 0.265, 0.515, 0.10, 0.060,
         "abstain $\\to$\nnarrowing prompt",
         fc="#FFEBEE", ec="#C62828", fontsize=8)

    # ---- Coordinator lane
    # Bundle box: subtitle moved into smaller second line
    bundle_x, bundle_y, bundle_w, bundle_h = 0.13, 0.380, 0.34, 0.095
    bundle_box = FancyBboxPatch(
        (bundle_x, bundle_y), bundle_w, bundle_h,
        boxstyle="round,pad=0.012,rounding_size=0.022",
        linewidth=1.2, edgecolor="#F57C00", facecolor="#FFFFFF",
    )
    ax.add_patch(bundle_box)
    ax.text(bundle_x + bundle_w / 2, bundle_y + bundle_h * 0.72,
            "Pre-screening summary bundle",
            ha="center", va="center", fontsize=10, fontweight="bold")
    ax.text(bundle_x + bundle_w / 2, bundle_y + bundle_h * 0.30,
            "decision $\\cdot$ supported facts $\\cdot$ "
            "fallbacks $\\cdot$ audit JSON",
            ha="center", va="center", fontsize=8.2, color="#555")

    _box(ax, 0.500, 0.380, 0.20, 0.095,
         "Coordinator review\nverify $F^+, F^-, R^-$",
         fc="#FFFFFF", ec="#F57C00", fontsize=9)
    _box(ax, 0.730, 0.380, 0.24, 0.095,
         "Final eligibility decision\n(study team)",
         fc="#FFFFFF", ec="#F57C00", fontweight="bold", fontsize=9)

    # ---- Audit lane
    AUDIT_Y, AUDIT_H = 0.125, 0.095
    audit_boxes = [
        (0.030, 0.20, "Per-query audit log\nevidence, gates, JSON"),
        (0.245, 0.20, "Continuous safety monitor\nleak / parse / abstain"),
        (0.460, 0.20, "IRB / governance review\nretain N days, drift escalation"),
        (0.675, 0.295, "Threshold re-tuning\nheld-out institutional pool;\nnot driven by $\\tau$ alone"),
    ]
    for x, w, text in audit_boxes:
        _box(ax, x, AUDIT_Y, w, AUDIT_H, text, fc="#FFFFFF", ec="#558B2F",
             fontsize=8.6)

    # ---- Arrows
    # Patient query -> Retrieval
    _arrow(ax, 0.21, 0.815, 0.21, 0.710, color="#1976D2")
    # Pipeline horizontal flow
    _arrow(ax, 0.23, 0.6625, 0.265, 0.6625)
    _arrow(ax, 0.365, 0.6625, 0.400, 0.6625)
    _arrow(ax, 0.500, 0.6625, 0.535, 0.6625)
    _arrow(ax, 0.635, 0.6625, 0.670, 0.6625)
    # Selector -> abstain (when gate fails)
    _arrow(ax, 0.315, 0.615, 0.315, 0.575, color="#C62828",
           text="gate fails", text_offset_y=0.005)
    # Abstain -> coordinator bundle (narrowing prompt routed but
    # primarily a patient-facing event; arrow goes back to patient)
    _arrow(ax, 0.315, 0.515, 0.315, 0.475, color="#C62828",
           text="narrowing prompt", text_offset_y=0.004)
    # Teach-back -> patient (curved out to the right to avoid label overlap)
    _arrow(ax, 0.770, 0.710, 0.875, 0.815, color="#7B1FA2")
    ax.text(0.860, 0.755, "targeted Q's",
            ha="left", va="center", fontsize=7.8, color="#7B1FA2")
    # Bundle build arrow: from right side of consent/teach-back drop-down,
    # routed AROUND the abstain box (to the right of it).
    _arrow(ax, 0.610, 0.615, 0.420, 0.475, color="#7B1FA2",
           text="bundle build", text_offset_y=0.015)
    # Bundle -> coordinator review -> final decision
    _arrow(ax, 0.470, 0.4275, 0.500, 0.4275)
    _arrow(ax, 0.700, 0.4275, 0.730, 0.4275)

    # Audit feeds: rerouted to avoid crossing the bundle.
    # Selector logs flow down the LEFT margin, not diagonally through bundle.
    _arrow(ax, 0.265, 0.6625, 0.060, 0.6625, color="#558B2F")
    _arrow(ax, 0.060, 0.6625, 0.060, 0.220, color="#558B2F")
    _arrow(ax, 0.230, 0.1725, 0.245, 0.1725, color="#558B2F")
    _arrow(ax, 0.445, 0.1725, 0.460, 0.1725, color="#558B2F")
    _arrow(ax, 0.660, 0.1725, 0.675, 0.220, color="#558B2F")

    # ---- Title
    ax.set_title(
        "Clinical deployment workflow",
        fontsize=13, fontweight="bold", pad=10,
    )

    # legend
    legend_lines = [
        Line2D([0], [0], color="#1976D2", lw=2,
               label="Patient interaction"),
        Line2D([0], [0], color="#7B1FA2", lw=2,
               label="Pipeline (automated)"),
        Line2D([0], [0], color="#F57C00", lw=2,
               label="Coordinator (human)"),
        Line2D([0], [0], color="#558B2F", lw=2,
               label="Audit / governance"),
        Line2D([0], [0], color="#C62828", lw=2,
               label="Abstain pathway"),
    ]
    ax.legend(handles=legend_lines, loc="lower center",
              bbox_to_anchor=(0.5, -0.03),
              ncol=5, frameon=False, fontsize=8.5)

    fig.tight_layout()
    fig.savefig(FIG / "deployment_workflow.pdf", bbox_inches="tight")
    fig.savefig(FIG / "deployment_workflow.png", bbox_inches="tight",
                dpi=200)
    plt.close(fig)
    print("[wrote]", FIG / "deployment_workflow.pdf")


if __name__ == "__main__":
    main()
