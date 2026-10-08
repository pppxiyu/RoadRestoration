"""Render the five daily OD-demand panels used on meeting slides 2--4."""

from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PURPLE = "#6A3D9A"
NEUTRAL = "#5E6472"


def _plot_recovery(frame: pd.DataFrame, normal_total: float, legend: str,
                   destination: Path):
    days = frame["day"].to_numpy(dtype=int)
    external = frame["external_total_od_demand"].to_numpy() / normal_total
    adjusted = frame["adjusted_total_od_demand"].to_numpy() / normal_total
    with plt.rc_context({
        "font.family": "Arial", "font.size": 18, "axes.labelsize": 19,
        "xtick.labelsize": 16, "ytick.labelsize": 16,
        "legend.fontsize": 16, "legend.frameon": False,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 1.1,
    }):
        fig, ax = plt.subplots(figsize=(12.8, 4.8))
        ax.plot(days, adjusted, color=PURPLE, linewidth=3.2,
                label="Fixed-point OD demand")
        ax.plot(days, external, color=NEUTRAL, linewidth=2.5,
                linestyle=(0, (5, 4)), label=legend)
        ax.set_ylabel("Total OD demand\nrelative to normal")
        ax.set_xlabel("Days after disaster")
        ax.set_xlim(1, int(days[-1]))
        ax.set_xticks(np.arange(1, int(days[-1]) + 1, 20))
        ax.set_ylim(0.1, 1.05)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.grid(axis="y", color="#E5E7EB", linewidth=0.9)
        ax.legend(loc="lower left", bbox_to_anchor=(0, 1.035), ncol=2,
                  borderaxespad=0, handlelength=3)
        fig.subplots_adjust(left=0.13, right=0.985, bottom=0.20, top=0.83)
        fig.savefig(destination, dpi=300, bbox_inches="tight")
        plt.close(fig)


def _plot_abc(frame: pd.DataFrame, destination: Path):
    days = frame["day"].to_numpy(dtype=int)
    A = frame["A_direct_damage_minutes"].to_numpy()
    B = frame["B_actual_congestion_minutes"].to_numpy()
    C = frame["C_reference_congestion_minutes"].to_numpy()
    gap = frame["demand_gap_percent_normal"].to_numpy()
    assert np.allclose(A + B - C, frame["mean_travel_time_difference_minutes"])
    with plt.rc_context({
        "font.family": "Arial", "font.size": 18, "axes.labelsize": 19,
        "xtick.labelsize": 16, "ytick.labelsize": 16,
        "legend.fontsize": 15, "legend.frameon": False,
        "axes.spines.top": False, "axes.linewidth": 1.1,
    }):
        fig, ax = plt.subplots(figsize=(14, 6.3))
        right = ax.twinx()
        lines = ax.plot(days, gap, color=PURPLE, linewidth=3,
                        label="OD demand gap")
        lines += right.plot(days, A, color="#D9792B", linewidth=2.7,
                            linestyle="-.", label="A: Direct road-damage delay")
        lines += right.plot(days, B, color="#3569A8", linewidth=2.7,
                            linestyle="--", label="B: Actual-network congestion")
        lines += right.plot(days, C, color="#328578", linewidth=2.7,
                            linestyle=":", label="C: Reference congestion")
        ax.set_ylabel("Demand gap (% of normal OD)", color=PURPLE)
        right.set_ylabel("External-OD-weighted time (minutes)")
        ax.tick_params(axis="y", colors=PURPLE)
        ax.spines["left"].set_color(PURPLE)
        right.spines["left"].set_visible(False)
        ax.set_ylim(-0.13, 13)
        ax.set_yticks([0, 3, 6, 9, 12])
        right.set_ylim(-0.12, 12)
        right.set_yticks([0, 2, 4, 6, 8, 10, 12])
        ax.set_xlim(1, int(days[-1]))
        ax.set_xticks(np.arange(1, int(days[-1]) + 1, 20))
        ax.set_xlabel("Days after disaster")
        ax.grid(axis="y", color="#E5E7EB", linewidth=0.9)
        fig.legend(lines, [line.get_label() for line in lines], ncol=2,
                   loc="upper center", bbox_to_anchor=(0.51, 1.0),
                   handlelength=3, columnspacing=2)
        fig.subplots_adjust(left=0.095, right=0.90, bottom=0.16, top=0.82)
        fig.savefig(destination, dpi=300, bbox_inches="tight")
        plt.close(fig)


def render_five_panels(output_dir: Path, slow: pd.DataFrame, fast: pd.DataFrame,
                       abc: pd.DataFrame, gradual: pd.DataFrame,
                       normal_total: float) -> list[Path]:
    """Save one PNG per slide image placement; the 35-day plot appears twice."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        output_dir / "01_recovery_70_days.png",
        output_dir / "02_recovery_35_days.png",
        output_dir / "03_recovery_35_days_repeat.png",
        output_dir / "04_abc_decomposition_35_days.png",
        output_dir / "05_gradual_response_95pct_in_7_days.png",
    ]
    _plot_recovery(slow, normal_total, "External recovery forecast", paths[0])
    _plot_recovery(fast, normal_total,
                   "External recovery forecast (faster)", paths[1])
    shutil.copyfile(paths[1], paths[2])
    _plot_abc(abc, paths[3])
    _plot_recovery(gradual, normal_total,
                   "External recovery forecast (faster)", paths[4])
    return paths
