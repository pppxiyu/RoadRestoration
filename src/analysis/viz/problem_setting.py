"""Figures for the active daily discovery problem.

Render from the same public problem and frozen scenario generator as the
experiment. Historical three-hour figures are archived outside the active
source tree and must not be mistaken for the current problem.

    python -m src.analysis.viz.problem_setting --n 11
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from src.environment.daily import recovery_fraction
from src.experiment.layout import study_dir
from src.problem.daily import (CONFUSION, build_problem, duration_draw,
                               duration_expectation, make_splits)


BLUE = "#36649C"
PURPLE = "#7655A3"
LILAC = "#A990C9"
GRAY = "#D3D6DE"
DARK = "#263444"
CLASS_WIDTH = {"local": 1.6, "major": 3.2, "highway": 5.2}
STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 17,
    "axes.labelsize": 19,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 14,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
}


def _save(fig, path: Path):
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _selection_map(problem, path):
    nodes = pd.read_csv(Path(problem["toy_dir"]) / "network" / "nodes.csv")
    position = {int(r.node_id): (float(r.x), float(r.y))
                for r in nodes.itertuples(index=False)}
    public = set(problem["public_roads"])
    high, middle, low = map(set, problem["bins"])
    colors = {
        "public": BLUE,
        "potential_high": PURPLE,
        "potential_middle": LILAC,
        "excluded": GRAY,
    }
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(11.2, 7.0), layout="constrained")
        for row in problem["edges"].itertuples(index=False):
            edge = int(row.edge_id)
            kind = ("public" if edge in public else
                    "potential_high" if edge in high else
                    "potential_middle" if edge in middle else "excluded")
            xy = np.asarray((position[int(row.u)], position[int(row.v)]))
            ax.plot(xy[:, 0], xy[:, 1], color=colors[kind],
                    lw=CLASS_WIDTH[str(row.road_class)], solid_capstyle="round",
                    zorder=3 if kind == "public" else 2 if kind != "excluded" else 1)
        xy = np.asarray(list(position.values()))
        ax.scatter(xy[:, 0], xy[:, 1], s=21, facecolor="white",
                   edgecolor=DARK, linewidth=.7, zorder=4)
        ax.set_aspect("equal")
        ax.axis("off")
        hh, hm = problem["hidden_quotas"]
        # The public count is read from the actual sampled problem, never from
        # a hard-coded percentage that may round differently across scales.
        handles = [
            Line2D([], [], color=BLUE, lw=4, label=f"Public on day 0 ({len(public)})"),
            Line2D([], [], color=PURPLE, lw=4,
                   label=f"Possible hidden high-flow roads ({len(high - public)}; draw {hh})"),
            Line2D([], [], color=LILAC, lw=4,
                   label=f"Possible hidden middle-flow roads ({len(middle - public)}; draw {hm})"),
            Line2D([], [], color=GRAY, lw=4,
                   label=f"Low-flow roads excluded ({len(low)})"),
        ]
        fig.legend(handles=handles, loc="lower center", ncol=2,
                   bbox_to_anchor=(.5, -.04), frameon=False)
        _save(fig, path)


def _hidden_frequency(problem, test_scenarios, path):
    public = set(problem["public_roads"])
    eligible = [edge for edge in problem["ranked"][:29] if edge not in public]
    high = set(problem["bins"][0])
    counts = {edge: 0 for edge in eligible}
    for scenario in test_scenarios:
        for road in scenario.roads:
            if road.edge_id not in public:
                counts[road.edge_id] += 1
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(13.2, 5.4), layout="constrained")
        x = np.arange(len(eligible))
        ax.bar(x, [counts[e] for e in eligible],
               color=[PURPLE if e in high else LILAC for e in eligible], width=.82)
        ax.set_xticks(x, [str(e) for e in eligible])
        ax.set_xlabel("Road segment (ordered by normal-period flow)")
        ax.set_ylabel("Test scenarios with hidden damage")
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=.18)
        ax.set_axisbelow(True)
        ax.legend(handles=[
            Line2D([], [], color=PURPLE, lw=7, label="High-flow candidates"),
            Line2D([], [], color=LILAC, lw=7, label="Middle-flow candidates"),
        ], loc="upper right", frameon=False)
        _save(fig, path)
    return counts


def _discovery_days(problem, test_scenarios, path):
    public = set(problem["public_roads"])
    days = [road.discovery_day for scenario in test_scenarios
            for road in scenario.roads if road.edge_id not in public]
    counts = np.bincount(days, minlength=problem["rules"].discovery_last_day + 1)
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(11.4, 5.4), layout="constrained")
        day = np.arange(1, problem["rules"].discovery_last_day + 1)
        ax.bar(day, counts[1:], color=PURPLE, width=.72)
        ax.axhline(len(days) / len(day), color=DARK, lw=2, ls="--",
                   label="Uniform-day expectation")
        ax.set_xticks(day)
        ax.set_xlabel("Day after disaster")
        ax.set_ylabel("Discoveries across 50 test scenarios")
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=.18)
        ax.set_axisbelow(True)
        ax.legend(frameon=False)
        _save(fig, path)
    return counts


def _severity_law(path):
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(10.8, 5.4), layout="constrained")
        x = np.arange(3)
        width = .25
        for i, color in enumerate((BLUE, LILAC, PURPLE)):
            ax.bar(x + (i - 1) * width,
                   [CONFUSION[s][i] for s in (1, 2, 3)],
                   width=width, color=color, label=f"True severity {i + 1}")
        ax.set_xticks(x, [f"Estimate {s}" for s in (1, 2, 3)])
        ax.set_xlabel("Public severity estimate")
        ax.set_ylabel("Probability of true severity")
        ax.set_ylim(0, .7)
        ax.legend(ncol=3, frameon=False, loc="upper right")
        ax.grid(axis="y", alpha=.18)
        ax.set_axisbelow(True)
        _save(fig, path)


def _duration_law(path):
    classes = ("local", "major", "highway")
    colors = {"local": BLUE, "major": LILAC, "highway": PURPLE}
    cells = [(c, s) for c in classes for s in (1, 2, 3)]
    rows = []
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(11.5, 6.8), layout="constrained")
        for i, (road_class, severity) in enumerate(cells):
            lower = duration_draw(road_class, severity, .025)
            upper = duration_draw(road_class, severity, .975)
            mean = duration_expectation(road_class, severity)
            estimate = max(1, int(np.floor(mean + .5)))
            rows.append(dict(road_class=road_class, severity=severity,
                             lower_2p5_day=lower, upper_97p5_day=upper,
                             actual_cell_mean_day=mean, public_estimate_day=estimate))
            y = len(cells) - 1 - i
            ax.plot([lower, upper], [y, y], color=colors[road_class],
                    lw=5, solid_capstyle="round")
            ax.scatter(mean, y, color=DARK, s=65, zorder=3)
        ax.set_yticks(range(len(cells)),
                      [f"{c.title()} · severity {s}" for c, s in reversed(cells)])
        ax.set_xlabel("Repair duration (days)")
        ax.set_xlim(left=0)
        ax.grid(axis="x", alpha=.18)
        ax.set_axisbelow(True)
        ax.legend(handles=[
            Line2D([], [], color=PURPLE, lw=5, label="Middle 95% of true durations"),
            Line2D([], [], marker="o", ls="", color=DARK, label="Cell mean"),
        ], loc="upper right", frameon=False)
        _save(fig, path)
    return pd.DataFrame(rows)


def _natural_recovery(problem, path):
    days = np.arange(0, problem["rules"].recovery_days + 1)
    fraction = recovery_fraction(days, problem["rules"])
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(11.0, 5.4), layout="constrained")
        ax.plot(days, 100 * fraction, color=BLUE, lw=3.5)
        ax.axhline(100, color=DARK, lw=1.6, ls="--",
                   label="Normal-period demand")
        ax.set_xlim(0, int(days[-1]))
        ax.set_ylim(0, 105)
        ax.set_xlabel("Day after disaster")
        ax.set_ylabel("Natural demand (% of normal)")
        ax.grid(axis="y", alpha=.18)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, loc="lower right")
        _save(fig, path)


def render_problem_setting(n=11, seed=42):
    """Render only the requested scale, using its current frozen sampling law."""
    problem = build_problem(n, seed)
    splits = make_splits(problem)
    figure_dir = study_dir("problem_setting") / "figures" / f"n{n}"
    data_dir = study_dir("problem_setting") / "data" / f"n{n}"
    figure_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    _selection_map(problem, figure_dir / "01_road_selection.png")
    counts = _hidden_frequency(problem, splits["test"],
                               figure_dir / "02_hidden_road_frequency.png")
    day_counts = _discovery_days(problem, splits["test"],
                                 figure_dir / "03_discovery_days.png")
    _severity_law(figure_dir / "04_severity_law.png")
    cells = _duration_law(figure_dir / "05_repair_duration.png")
    _natural_recovery(problem, figure_dir / "06_natural_recovery.png")
    pd.DataFrame([dict(edge_id=e, hidden_test_scenarios=counts[e],
                       normal_two_way_flow=problem["baseline_flow"][e])
                  for e in counts]).to_csv(data_dir / "hidden_road_frequency.csv", index=False)
    pd.DataFrame(dict(day=np.arange(len(day_counts)),
                      hidden_roads_discovered=day_counts)).to_csv(
                          data_dir / "discovery_days.csv", index=False)
    cells.to_csv(data_dir / "duration_cells.csv", index=False)
    return figure_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, choices=(6, 11, 17, 23), default=11)
    parser.add_argument("--seed", type=int, default=42)
    arguments = parser.parse_args()
    print(render_problem_setting(arguments.n, arguments.seed))
