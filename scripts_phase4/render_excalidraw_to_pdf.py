"""CLI: render any .excalidraw JSON file to PDF + PNG via matplotlib.

Generalised from `build_figure1_from_excalidraw.py`. Used as a *fallback*
renderer when the user has not yet manually exported the figure from
excalidraw.com. Output is functional, not pretty — user-supplied
Excalidraw export should overwrite when ready.

Usage:
    python3 render_excalidraw_to_pdf.py \\
        --input outputs/paper_v2/figures/pipeline_architecture.excalidraw \\
        --output outputs/paper_v2/figures/pipeline_architecture
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


def render(src_path: Path, out_stem: Path, width_in: float = 14.0) -> None:
    data = json.loads(src_path.read_text())
    elements = data["elements"]

    # Bounding box
    xs = [el["x"] for el in elements if "x" in el]
    ys = [el["y"] for el in elements if "y" in el]
    x2s = [el["x"] + el.get("width", 0) for el in elements if "x" in el]
    y2s = [el["y"] + el.get("height", 0) for el in elements if "y" in el]
    minx, miny = min(xs), min(ys)
    maxx, maxy = max(x2s), max(y2s)
    src_w = maxx - minx
    src_h = maxy - miny
    fig_h = width_in * (src_h / src_w)
    fig, ax = plt.subplots(figsize=(width_in, fig_h))
    ax.set_xlim(minx - 20, maxx + 20)
    ax.set_ylim(maxy + 20, miny - 20)  # invert y to match Excalidraw
    ax.set_aspect("equal")
    ax.axis("off")

    # Rectangles (including those with embedded label)
    for el in elements:
        if el["type"] not in ("rectangle", "ellipse", "diamond"):
            continue
        x, y = el["x"], el["y"]
        w, h = el["width"], el["height"]
        stroke = el.get("strokeColor", "#1e1e1e")
        fill = el.get("backgroundColor", "transparent")
        fill_rgba = (0, 0, 0, 0) if fill == "transparent" else fill
        stroke_rgba = (0, 0, 0, 0) if stroke == "transparent" else stroke
        sw = el.get("strokeWidth", 1)
        opacity = el.get("opacity", 100) / 100.0
        rect = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=2,rounding_size=6",
            linewidth=sw * 1.0, edgecolor=stroke_rgba, facecolor=fill_rgba,
            alpha=opacity if opacity < 1.0 else None,
            zorder=1,
        )
        ax.add_patch(rect)
        # Embedded label?
        label = el.get("label")
        if isinstance(label, dict):
            txt = label.get("text", "")
            fs = label.get("fontSize", 16) * 0.75
            ax.text(x + w / 2, y + h / 2, txt,
                    ha="center", va="center",
                    fontsize=fs, color=stroke, zorder=3, wrap=True)

    # Standalone text
    for el in elements:
        if el["type"] != "text":
            continue
        x, y = el["x"], el["y"]
        w = el.get("width", 0)
        h = el.get("height", 30)
        text = el.get("text", "")
        fs = el.get("fontSize", 16) * 0.75
        color = el.get("strokeColor", "#1e1e1e")
        text_align = el.get("textAlign", "left")
        ha = {"center": "center", "right": "right"}.get(text_align, "left")
        if ha == "center":
            tx = x + w / 2 if w else x
        elif ha == "right":
            tx = x + w if w else x
        else:
            tx = x
        ax.text(tx, y + (h / 2 if h else 0), text,
                ha=ha, va="center" if h else "top",
                fontsize=fs, color=color, zorder=3)

    # Arrows
    for el in elements:
        if el["type"] != "arrow":
            continue
        x0, y0 = el["x"], el["y"]
        pts = el.get("points", [])
        if not pts:
            continue
        path_pts = [(x0 + px, y0 + py) for px, py in pts]
        stroke = el.get("strokeColor", "#1e1e1e")
        sw = el.get("strokeWidth", 1)
        style = el.get("strokeStyle", "solid")
        ls = "--" if style == "dashed" else "-"
        for (xa, ya), (xb, yb) in zip(path_pts[:-1], path_pts[1:]):
            ax.plot([xa, xb], [ya, yb], color=stroke, linewidth=sw * 1.0,
                    linestyle=ls, zorder=2)
        # Arrowhead at last segment
        (xa, ya), (xb, yb) = path_pts[-2], path_pts[-1]
        arrow = FancyArrowPatch(
            (xa, ya), (xb, yb),
            arrowstyle="-|>", mutation_scale=14,
            color=stroke, linewidth=sw * 1.0,
            zorder=2,
        )
        ax.add_patch(arrow)
        # Arrow label
        label = el.get("label")
        if isinstance(label, dict):
            txt = label.get("text", "")
            fs = label.get("fontSize", 14) * 0.75
            # Place at midpoint of arrow
            mx, my = (path_pts[0][0] + path_pts[-1][0]) / 2, \
                     (path_pts[0][1] + path_pts[-1][1]) / 2
            ax.text(mx, my - 8, txt,
                    ha="center", va="center",
                    fontsize=fs, color=stroke, zorder=3,
                    bbox=dict(facecolor="white", edgecolor="none",
                              pad=1.5, alpha=0.92))

    fig.tight_layout()
    pdf_out = out_stem.with_suffix(".pdf")
    png_out = out_stem.with_suffix(".png")
    fig.savefig(pdf_out, bbox_inches="tight")
    fig.savefig(png_out, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"[wrote] {pdf_out}")
    print(f"[wrote] {png_out}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path,
                   help="Output stem (no extension); .pdf and .png appended")
    p.add_argument("--width-in", type=float, default=14.0)
    args = p.parse_args()
    render(args.input, args.output, args.width_in)


if __name__ == "__main__":
    main()
