"""Plot the fitted Sioux Falls gravity-model travel-time effect."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from util.gravity import add_gravity_inputs, complete_od_table, fit_gravity


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/01-sim_val_n_problem_setting/01-benchmark"


_PLOT_STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "font.size": 17,
    "axes.labelsize": 18,
    "axes.linewidth": 1.2,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 12.5,
    "legend.frameon": False,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "axes.spines.top": False,
    "axes.spines.right": False,
}


def build_curve() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    table = add_gravity_inputs(complete_od_table())
    parameters, covariance, fitted_pairs, _ = fit_gravity(table)
    coefficient = parameters.set_index("parameter")["estimate"].to_numpy(float)
    covariance_matrix = covariance.to_numpy(float)

    assert (table["origin_total"] > 0).all()
    assert (table["destination_total"] > 0).all()
    origin_log = np.log(table["origin_total"].to_numpy(float))
    destination_log = np.log(table["destination_total"].to_numpy(float))
    travel_times = np.linspace(
        float(table["travel_cost_minutes"].min()),
        float(table["travel_cost_minutes"].max()),
        200,
    )
    rows = []
    individual_predictions = []
    for travel_time in travel_times:
        design = np.column_stack(
            [
                np.ones(len(table)),
                origin_log,
                destination_log,
                np.full(len(table), travel_time),
            ]
        )
        prediction = np.exp(design @ coefficient)
        individual_predictions.append(prediction)
        fitted_mean = float(prediction.mean())
        gradient = (prediction[:, None] * design).mean(axis=0)
        standard_error = float(
            np.sqrt(np.maximum(gradient @ covariance_matrix @ gradient, 0.0))
        )
        rows.append(
            {
                "travel_time_minutes": travel_time,
                "fitted_mean_flow": fitted_mean,
                "confidence_lower": max(0.0, fitted_mean - 1.96 * standard_error),
                "confidence_upper": fitted_mean + 1.96 * standard_error,
            }
        )
    curve = pd.DataFrame(rows)
    observed = fitted_pairs[
        [
            "origin",
            "destination",
            "travel_cost_minutes",
            "observed_flow",
            "predicted_flow",
        ]
    ].sort_values(["travel_cost_minutes", "origin", "destination"])
    prediction_matrix = np.asarray(individual_predictions).T
    return curve, observed, travel_times, prediction_matrix


def make_figure() -> Path:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    curve, observed, travel_times, prediction_matrix = build_curve()
    curve.to_csv(OUTPUT / "05_gravity_travel_time_curve.csv", index=False)
    observed.to_csv(OUTPUT / "05_gravity_observed_by_time.csv", index=False)

    with mpl.rc_context(_PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(6.4, 4.6), constrained_layout=True)
        ax.plot(
            travel_times,
            prediction_matrix.T,
            color="#B8A6C7",
            alpha=0.065,
            linewidth=0.65,
            zorder=1,
        )
        ax.plot(
            curve["travel_time_minutes"],
            curve["fitted_mean_flow"],
            color="#57068C",
            linewidth=3.4,
            zorder=4,
        )
        ax.set_xlabel("Equilibrium shortest-path travel time\n(minutes)", labelpad=8)
        ax.set_ylabel("OD flow\n(trips per representative period)", labelpad=10)
        travel_time_min = float(observed["travel_cost_minutes"].min())
        travel_time_max = float(observed["travel_cost_minutes"].max())
        ax.set_xlim(travel_time_min, travel_time_max)
        ax.set_ylim(bottom=0)
        tick_start = 5.0 * np.ceil(travel_time_min / 5.0)
        tick_end = 5.0 * np.floor(travel_time_max / 5.0)
        ax.set_xticks(np.arange(tick_start, tick_end + 0.1, 5.0))
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.7)
        ax.set_axisbelow(True)
        handles = [
            Line2D(
                [0],
                [0],
                color="#B8A6C7",
                linewidth=1.5,
                alpha=0.7,
                label="Individual OD-pair predictions",
            ),
            Line2D(
                [0],
                [0],
                color="#57068C",
                linewidth=3.4,
                label="Mean prediction",
            ),
        ]
        ax.legend(
            handles=handles,
            loc="upper right",
            handlelength=2.2,
            labelspacing=0.35,
        )

        output = OUTPUT / "05_gravity_travel_time_effect.png"
        fig.savefig(output, dpi=600, bbox_inches="tight", facecolor="white")
        plt.close(fig)
    return output


def make_reconstruction_figure() -> Path:
    """Plot each observed OD flow against its gravity-model reconstruction."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    table = add_gravity_inputs(complete_od_table())
    _, _, fitted_pairs, _ = fit_gravity(table)
    observed = fitted_pairs["observed_flow"].to_numpy(float)
    reconstructed = fitted_pairs["predicted_flow"].to_numpy(float)

    residual_sum_squares = float(np.sum((observed - reconstructed) ** 2))
    total_sum_squares = float(np.sum((observed - observed.mean()) ** 2))
    r_squared = 1.0 - residual_sum_squares / total_sum_squares
    pearson_correlation = float(np.corrcoef(observed, reconstructed)[0, 1])

    upper = float(np.ceil(max(observed.max(), reconstructed.max()) / 500.0) * 500.0)
    with mpl.rc_context(_PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(6.4, 4.6), constrained_layout=True)
        ax.scatter(
            observed,
            reconstructed,
            s=34,
            color="#6A1B9A",
            alpha=0.34,
            edgecolor="none",
            rasterized=True,
        )
        ax.plot(
            [0.0, upper],
            [0.0, upper],
            color="#625B6D",
            linewidth=1.7,
            linestyle="--",
            label="Perfect reconstruction",
        )
        ax.text(
            0.05,
            0.95,
            f"R² = {r_squared:.3f}\nPearson r = {pearson_correlation:.3f}\nn = {len(observed)} OD pairs",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=15,
            color="#262330",
        )
        ax.set_xlabel("Observed OD flow\n(trips per representative period)", labelpad=8)
        ax.set_ylabel("Reconstructed OD flow\n(trips per representative period)", labelpad=8)
        ax.set_xlim(0.0, upper)
        ax.set_ylim(0.0, upper)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(color="#D9D9D9", linewidth=0.8, alpha=0.7)
        ax.set_axisbelow(True)
        ax.legend(loc="lower right", handlelength=2.2)

        output = OUTPUT / "06_gravity_reconstruction_scatter.png"
        fig.savefig(output, dpi=600, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    metrics = {
        "origin_destination_pairs": int(len(observed)),
        "r_squared_definition": "1 - sum((observed - reconstructed)^2) / sum((observed - mean(observed))^2)",
        "r_squared": float(r_squared),
        "pearson_correlation": float(pearson_correlation),
    }
    (OUTPUT / "06_gravity_reconstruction_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    fitted_pairs[
        ["origin", "destination", "observed_flow", "predicted_flow", "residual"]
    ].to_csv(OUTPUT / "06_gravity_reconstruction_data.csv", index=False)
    return output


def make_formula_block() -> Path:
    """Render fitted equations with the same mathematical typography as slide 1."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    parameters, _, _, diagnostics = fit_gravity(add_gravity_inputs(complete_od_table()))
    estimates = parameters.set_index("parameter")["estimate"]
    scale = float(np.exp(estimates["intercept"]))
    origin_elasticity = float(estimates["log_origin_total"])
    destination_elasticity = float(estimates["log_destination_total"])
    cost_sensitivity = float(-estimates["travel_cost_minutes"])
    mantissa, exponent = f"{scale:.3e}".split("e")
    scale_math = rf"{float(mantissa):.3g} \times 10^{{{int(exponent)}}}"
    formula = (
        rf"${{OD}}'_{{ij}} = \left({scale_math}\right) "
        rf"O_i^{{{origin_elasticity:.4f}}} D_j^{{{destination_elasticity:.4f}}} "
        rf"\exp\!\left(-{cost_sensitivity:.5f}\,c_{{ij}}\right)$"
    )
    parameter_line = (
        rf"$\widehat{{\theta}} = "
        rf"\left(\widehat K,\widehat\alpha,\widehat\beta,\widehat\gamma\right) = "
        rf"\left({scale_math},\,{origin_elasticity:.4f},\,{destination_elasticity:.4f},\,"
        rf"{cost_sensitivity:.5f}\right)$"
    )
    with mpl.rc_context(
        {
            "font.family": "STIXGeneral",
            "mathtext.fontset": "stix",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    ):
        fig = plt.figure(figsize=(13.2, 1.7), facecolor="none")
        fig.text(
            0.5,
            0.68,
            formula,
            ha="center",
            va="center",
            fontsize=32,
            color="#262330",
        )
        fig.text(
            0.5,
            0.22,
            parameter_line,
            ha="center",
            va="center",
            fontsize=27,
            color="#57068C",
        )
        output = OUTPUT / "07_gravity_fitted_formula.png"
        fig.savefig(output, dpi=600, bbox_inches="tight", pad_inches=0.02, transparent=True)
        plt.close(fig)
    (OUTPUT / "07_gravity_fitted_formula_values.json").write_text(
        json.dumps(
            {
                "scale": scale,
                "origin_elasticity": origin_elasticity,
                "destination_elasticity": destination_elasticity,
                "cost_sensitivity_per_minute": cost_sensitivity,
                "one_additional_minute_flow_percent_decrease": diagnostics[
                    "one_additional_minute"
                ]["flow_percent_decrease"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return output


if __name__ == "__main__":
    print(make_figure())
    print(make_reconstruction_figure())
    print(make_formula_block())
