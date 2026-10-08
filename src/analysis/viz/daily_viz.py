"""Daily experiment figures from saved data, using the existing Python workflow.

Figure contract: quantify the difference between two training environments under
one complete test environment; do not presume either wins. Boxplots show the
scenario distribution, paired scatter shows within-scenario differences. Training
diagnostics contain no test data. Quantitative panels use large slide-readable
fonts, restrained consistent colors, and PNG only; all source tables are retained.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {"natural": "#527DA8", "response7d": "#8265A3", "ga": "#BA8855", "roadclass": "#418C8B",
          "natural_roadclass": "#7198BD", "no_traffic": "#BA8855"}
LABELS = {"natural": "Trained with\nnatural recovery only", "response7d": "Trained with\nhuman response",
          "ga": "Genetic algorithm\nFixed priority",
          "roadclass": "Human response\nWith road class",
          "natural_roadclass": "Natural recovery only\nWith road class",
          "no_traffic": "Human response\nWithout real-time traffic inputs"}
STYLE = {"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
         "font.size": 18, "axes.labelsize": 19, "axes.titlesize": 20,
         "xtick.labelsize": 16, "ytick.labelsize": 16, "legend.fontsize": 15,
         "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 1.0,
         "legend.frameon": False, "lines.linewidth": 2.2}


def _save(figure, destination):
    figure.savefig(destination, dpi=300, bbox_inches="tight")
    plt.close(figure)


def make_training_figures(run_dir):
    run_dir = Path(run_dir)
    path = run_dir / "log" / "training.csv"
    if not path.exists():
        return
    data = pd.read_csv(path)
    if data.empty:
        return
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(11, 6), layout="constrained")
        ax.plot(data.episode, data.training_objective, color="#A9B6C5", alpha=0.55,
                lw=1.1, label="Training batch")
        valid = data.dropna(subset=["validation_objective"])
        ax.plot(valid.episode, valid.validation_objective, color="#527DA8", marker="o",
                markersize=5, label="Validation scenarios")
        ax.plot(data.episode, data.best_validation_objective, color="#8265A3", ls="--",
                label="Best validation so far")
        ax.set(xlabel="Training episode", ylabel="Cumulative transport loss\n(equivalent days)")
        ax.legend(loc="best")
        _save(fig, run_dir / "training_objective.png")
        fig, axes = plt.subplots(1, 3, figsize=(19, 5.5), layout="constrained")
        axes[0].plot(data.episode, data.absolute_td_error, color="#527DA8")
        axes[0].set(title="Prediction error", ylabel="Mean absolute learning error")
        axes[1].plot(data.episode, data.q_mean, color="#527DA8", label="Predicted action value")
        axes[1].plot(data.episode, data.target_mean, color="#8265A3", ls="--", label="Learning target")
        axes[1].set(title="Predictions and targets", ylabel="Action value")
        axes[1].legend(loc="best", fontsize=13)
        axes[2].plot(data.episode, data.gradient_norm, color="#8265A3")
        axes[2].set(title="Gradient magnitude", ylabel="Gradient norm", yscale="symlog")
        for ax in axes:
            ax.set_xlabel("Training episode")
        _save(fig, run_dir / "learning_diagnostics.png")


def make_comparison(data, output, filename="performance_distribution.png"):
    output = Path(output)
    methods = list(data.groupby("method").F.mean().sort_values(ascending=False).index)
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(max(10, 4 * len(methods)), 6.4), layout="constrained")
        series = [data.loc[data.method == m, "F"].to_numpy() for m in methods]
        box = ax.boxplot(series, patch_artist=True, widths=0.48, showfliers=False,
                         medianprops=dict(color="#252525", linewidth=2),
                         whiskerprops=dict(linewidth=1.6), capprops=dict(linewidth=1.6))
        labels = []
        rng = np.random.default_rng(0)
        for i, (name, patch, values) in enumerate(zip(methods, box["boxes"], series), start=1):
            method_rows = data[data.method == name]
            behavior = method_rows.training_behavior.iloc[0]
            if "input_variant" in method_rows and method_rows.input_variant.iloc[0] == "roadclass":
                behavior = "natural_roadclass" if behavior == "natural" else "roadclass"
            if "input_variant" in method_rows and method_rows.input_variant.iloc[0] in ("no_traffic", "roadclass_no_traffic"):
                behavior = "no_traffic"
            patch.set_facecolor(COLORS[behavior])
            patch.set_alpha(0.55)
            ax.scatter(i + rng.uniform(-0.09, 0.09, len(values)), values, s=20,
                       color=COLORS[behavior], alpha=0.55, linewidths=0)
            labels.append(LABELS[behavior])
        ax.set_xticks(np.arange(1, len(methods) + 1), labels)
        ax.set_ylabel("Cumulative transport loss\n(equivalent days)")
        ax.set_xlim(0.5, len(methods) + 0.5)
        ax.grid(axis="y", alpha=0.16)
        _save(fig, output / filename)
        for seed, frame in data.groupby("training_seed"):
            if "input_variant" in frame:
                frame = frame[frame.input_variant == "original"]
            frame = frame[frame.training_behavior.isin(["natural", "response7d"])]
            wide = frame.pivot(index="scenario", columns="training_behavior", values="F")
            if not {"natural", "response7d"} <= set(wide.columns):
                continue
            fig, ax = plt.subplots(figsize=(8.8, 7.0), layout="constrained")
            lo, hi = float(wide.min().min()), float(wide.max().max())
            pad = max((hi - lo) * 0.05, 0.1)
            ax.plot([lo-pad, hi+pad], [lo-pad, hi+pad], color="#777777", ls="--", lw=1.5,
                    label="Equal performance")
            ax.scatter(wide.natural, wide.response7d, color=COLORS["response7d"], s=42, alpha=0.75)
            ax.set(xlim=(lo-pad, hi+pad), ylim=(lo-pad, hi+pad),
                   xlabel="Loss after natural-recovery-only training",
                   ylabel="Loss after human-response training")
            ax.legend(loc="best")
            _save(fig, output / f"paired_performance_seed{seed}.png")


def make_ga_search(data, output):
    """Show optimization progress, never test feedback, in the existing style."""
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), layout="constrained")
        axes[0].plot(data.generation, data.best_F, color=COLORS["ga"], label="Best so far")
        axes[0].plot(data.generation, data.gen_mean_F, color="#9099A5", alpha=0.7,
                     label="Population mean")
        axes[0].set(xlabel="Search generation", ylabel="Mean training transport loss\n(equivalent days)")
        axes[0].legend()
        axes[1].plot(data.generation, data.cum_evals, color=COLORS["ga"])
        axes[1].set(xlabel="Search generation", ylabel="Distinct priority orders evaluated")
        _save(fig, Path(output) / "search_process.png")


def make_controlled_pair(data, output, *, methods, labels, colors, filename, y_max):
    """Two prespecified policies on the same test worlds; shared scale across pairs.

    Boxes show quartiles and medians, with 1.5-IQR whiskers. Every scenario is
    overlaid, including outliers. Points are scenarios, not training replicates.
    Labels/colors/order are fixed by the experiment plan, not by observed rank.
    """
    series = [data.loc[data.method == method, "F"].to_numpy() for method in methods]
    if len(methods) != 2 or any(len(values) == 0 for values in series):
        raise ValueError("a controlled comparison requires both named methods")
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(10.5, 6.4), layout="constrained")
        boxes = ax.boxplot(series, patch_artist=True, widths=.46, showfliers=False,
                           medianprops=dict(color="#252525", linewidth=2),
                           whiskerprops=dict(linewidth=1.6), capprops=dict(linewidth=1.6))
        # Identical offsets for paired scenarios; the full model is visually
        # identical whenever it appears in either requested comparison.
        jitter = np.random.default_rng(0).uniform(-.08, .08, max(map(len, series)))
        for i, (box, values, color) in enumerate(zip(boxes["boxes"], series, colors), 1):
            box.set_facecolor(color)
            box.set_alpha(.55)
            ax.scatter(i + jitter[:len(values)], values, s=24, color=color, alpha=.6, linewidths=0)
        ax.set_xticks([1, 2], labels)
        ax.set_ylabel("Cumulative transport loss\n(equivalent days)")
        ax.set_xlim(.5, 2.5)
        ax.set_ylim(0, y_max)
        ax.grid(axis="y", alpha=.16)
        _save(fig, Path(output) / filename)
