"""Comparable figures for the experimental fixed-point environment path."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from viz.style import save_pub, use_pub


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/01-sim_val_n_problem_setting/04-env_behavior"
RESULTS = OUTPUT / "results"

PURPLE = "#6A3D9A"
BLUE = "#3569A8"
ORANGE = "#D9792B"
NEUTRAL = "#5E6472"
LIGHT_NEUTRAL = "#9AA1AD"

CASE_FIGURES = {
    "instantaneous": "02_instantaneous_restoration_reference",
    "flow": "03_flow_priority_reference",
    "demand": "04_demand_priority_reference",
    "none": "05_no_restoration_reference",
}
CASE_RESULTS = {
    "instantaneous": "02_instantaneous_restoration_reference.csv",
    "flow": "03_flow_priority_reference.csv",
    "demand": "04_demand_priority_reference.csv",
    "none": "05_no_restoration_reference.csv",
}
TOP_TITLES = {
    "instantaneous": "OD demand recovery with instantaneous restoration",
    "flow": "OD demand recovery with a flow-priority reference policy",
    "demand": "OD demand recovery with a demand-priority reference policy",
    "none": "OD demand recovery with no restoration",
}


def _assert_aligned(axes, tolerance=1e-8):
    """Block export if the two plot areas do not share left and right edges."""
    first = axes[0].get_position()
    second = axes[1].get_position()
    if abs(first.x0 - second.x0) > tolerance or abs(first.x1 - second.x1) > tolerance:
        raise RuntimeError("environment-behavior subplot plot areas are not aligned")


def _prepare(frame, normal_total, case):
    if case == "instantaneous":
        return frame.assign(
            external_relative=1.0,
            adjusted_relative=(
                frame["adjusted_total_od_demand"] / float(normal_total)
            ),
        )
    return frame.assign(
        external_relative=(
            frame["external_total_od_demand"] / float(normal_total)
        ),
        adjusted_relative=(
            frame["adjusted_total_od_demand"] / float(normal_total)
        ),
    )


def _draw(frame, normal_total, case):
    frame = _prepare(frame, normal_total, case)
    use_pub(slide=True)
    plt.rcParams.update(
        {
            "font.size": 17,
            "axes.labelsize": 19,
            "axes.titlesize": 20,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 13.5,
        }
    )
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(12.5, 7.7),
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.12], "hspace": 0.80},
    )

    axes[0].step(
        frame["time_interval"],
        frame["adjusted_relative"],
        where="post",
        color=PURPLE,
        linewidth=3.4,
        label="Fixed-point OD demand",
    )
    axes[0].plot(
        frame["time_interval"],
        frame["external_relative"],
        color=NEUTRAL,
        linewidth=2.4,
        linestyle=(0, (5, 4)),
        label=(
            "External forecast (normal-period OD)"
            if case == "instantaneous"
            else "External recovery forecast"
        ),
    )
    if case != "instantaneous":
        axes[0].axhline(
            1.0,
            color=LIGHT_NEUTRAL,
            linewidth=1.7,
            linestyle=(0, (2, 3)),
            label="Normal-period OD demand",
        )
    axes[0].set_title(TOP_TITLES[case], loc="left", fontweight="bold", pad=56)
    axes[0].set_ylabel("Total OD demand\nrelative to normal")
    axes[0].legend(
        loc="lower left",
        bbox_to_anchor=(0.0, 1.03),
        borderaxespad=0.0,
        ncol=(2 if case == "instantaneous" else 3),
        columnspacing=2.6,
        handlelength=3.0,
    )

    axes[1].step(
        frame["time_interval"],
        frame["accessibility_degradation"],
        where="post",
        color=BLUE,
        linewidth=3.2,
        label="Accessibility degradation in each interval",
    )
    axes[1].plot(
        frame["time_interval"],
        frame["accessibility_degradation_cumulative_mean"],
        color=ORANGE,
        linewidth=3.2,
        label="Accessibility degradation cumulative mean",
    )
    axes[1].axhline(
        1.0,
        color=NEUTRAL,
        linewidth=1.8,
        linestyle=(0, (5, 4)),
        label="Normal-network level",
    )
    axes[1].set_title(
        "Accessibility degradation under flow-priority restoration",
        loc="left",
        fontweight="bold",
        pad=56,
    )
    axes[1].set_xlabel("Time interval after disaster (3 hours per interval)")
    axes[1].set_ylabel("Accessibility\ndegradation")
    axes[1].legend(
        loc="lower left",
        bbox_to_anchor=(0.0, 1.03),
        borderaxespad=0.0,
        ncol=3,
        columnspacing=2.0,
        handlelength=3.0,
    )

    if case == "none":
        axes[0].set_ylim(0.10, 2.05)
        axes[0].set_yticks([0.25, 0.50, 0.75, 1.00, 1.50, 2.00])
        axes[1].set_ylim(0.50, 10.00)
        axes[1].set_yticks([1, 2, 4, 6, 8, 10])
    else:
        axes[0].set_ylim(0.10, 1.10)
        axes[0].set_yticks([0.25, 0.50, 0.75, 1.00])
        axes[1].set_ylim(0.50, 1.70)
        axes[1].set_yticks([0.50, 0.75, 1.00, 1.25, 1.50])

    horizon = int(frame["time_interval"].max())
    for axis in axes:
        axis.set_xlim(1, horizon)
        axis.set_xticks(np.arange(1, horizon + 1, 20))
        axis.grid(axis="y", color="#E5E7EB", linewidth=0.9)
        axis.margins(x=0)
    fig.subplots_adjust(left=0.12, right=0.985, bottom=0.105, top=0.84)
    fig.canvas.draw()
    _assert_aligned(axes)
    save_pub(fig, OUTPUT / CASE_FIGURES[case], dpi=300)
    plt.close(fig)


def render_environment_behavior(frames=None, normal_total=None):
    """Render all four cases from in-memory or persisted simulation results."""
    if frames is None:
        frames = {
            case: pd.read_csv(RESULTS / filename)
            for case, filename in CASE_RESULTS.items()
        }
    if normal_total is None:
        recovery = frames["flow"]
        normal_total = float(
            np.median(
                recovery["external_total_od_demand"]
                / recovery["mobility_relative_to_normal"]
            )
        )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for case in ("instantaneous", "flow", "demand", "none"):
        _draw(frames[case], normal_total, case)
    return [OUTPUT / f"{CASE_FIGURES[case]}.png" for case in CASE_FIGURES]


if __name__ == "__main__":
    for path in render_environment_behavior():
        print(path)
