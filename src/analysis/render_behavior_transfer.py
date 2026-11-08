"""Plot mean system loss in simplified and complete environments."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


METHODS = [
    ("static", "Static flow\npriority"),
    ("dynamic", "Dynamic accessibility\nand duration"),
    ("ga", "GA"),
    ("rl", "RL"),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    complete = json.loads(args.input.read_text(encoding="utf-8"))
    assert complete["complete"] is True
    summaries = complete["summaries"]
    simplified = [summaries[f"{key}_matched"]["mean_system_loss"] for key, _ in METHODS]
    complete_environment = [
        summaries[f"{key}_response7d"]["mean_system_loss"] for key, _ in METHODS
    ]
    assert all(np.isfinite(simplified)) and all(np.isfinite(complete_environment))

    x = np.arange(len(METHODS), dtype=float)
    width = 0.34
    simplified_color = "#4477AA"
    complete_color = "#EE9955"

    with mpl.rc_context(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 17,
            "axes.labelsize": 19,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 17,
            "axes.linewidth": 1.2,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    ):
        fig, left = plt.subplots(figsize=(12.4, 6.8), constrained_layout=True)
        right = left.twinx()

        bars_left = left.bar(
            x - width / 2,
            simplified,
            width,
            color=simplified_color,
            edgecolor="#234A75",
            linewidth=1.1,
            label="Simplified environment",
            zorder=3,
        )
        bars_right = right.bar(
            x + width / 2,
            complete_environment,
            width,
            color=complete_color,
            edgecolor="#9B572B",
            linewidth=1.1,
            hatch="//",
            label="Complete environment",
            zorder=3,
        )

        left.set_ylabel("Mean system loss — simplified environment", color=simplified_color)
        right.set_ylabel("Mean system loss — complete environment", color=complete_color)
        left.tick_params(axis="y", colors=simplified_color, width=1.1)
        right.tick_params(axis="y", colors=complete_color, width=1.1)
        left.spines["left"].set_color(simplified_color)
        right.spines["right"].set_color(complete_color)
        left.spines["top"].set_visible(False)
        right.spines["top"].set_visible(False)
        left.set_ylim(bottom=0)
        right.set_ylim(bottom=0)
        left.set_xticks(x, [label for _, label in METHODS])
        left.grid(axis="y", color="#D9D9D9", linewidth=0.9, alpha=0.8, zorder=0)
        left.set_axisbelow(True)

        handles = [bars_left, bars_right]
        labels = [handle.get_label() for handle in handles]
        left.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.04),
            ncol=2,
            frameon=False,
            handlelength=1.8,
            columnspacing=2.0,
        )

        output = args.output.with_suffix(".png")
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=600, bbox_inches="tight", facecolor="white")
        plt.close(fig)


if __name__ == "__main__":
    main()
