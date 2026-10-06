"""Shared gravity model and exogenous daily OD demand for the problem setting.

The gravity model is estimated once from normal-period Sioux Falls data and persisted by
``util.gravity``. A free-plateau logistic curve then scales that fitted spatial OD pattern over
the post-disaster recovery period. The daily table is exogenous: it is identical for every
repair method and contains no realized repair-duration or severity information.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config as P
from util.gravity import (
    DEFAULT_MODEL,
    add_gravity_inputs,
    complete_od_table,
    ensure_gravity_model,
    load_gravity_model,
    predict_gravity_flow,
)
from viz.style import C, save_pub, use_pub


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = (
    ROOT
    / "outputs/01-sim_val_n_problem_setting/04-env_behavior"
)
RAW_DIR = ROOT / "outputs/01-sim_val_n_problem_setting/raw"
CURVE_CSV = RAW_DIR / "exogenous_mobility_recovery_curve.csv"
OD_CSV = RAW_DIR / "exogenous_daily_od_flow.csv"
META_JSON = RAW_DIR / "exogenous_daily_od_flow_meta.json"
FIGURE = OUTPUT_DIR / "01_exogenous_od_recovery_curve"


def recovery_rate() -> float:
    """Rate that places the logistic curve at 99% of its plateau on day 60."""
    initial = float(P.RECOVERY_INITIAL_LEVEL)
    plateau = float(P.RECOVERY_PLATEAU_LEVEL)
    target = float(P.RECOVERY_SETTLING_FRACTION)
    if not 0.0 < initial < plateau < 1.0:
        raise ValueError("recovery levels must satisfy 0 < initial < plateau < 1")
    if not 0.0 < target < 1.0:
        raise ValueError("RECOVERY_SETTLING_FRACTION must lie between zero and one")
    ratio = plateau / initial - 1.0
    return math.log(ratio / (1.0 / target - 1.0)) / float(P.RECOVERY_SETTLING_DAYS)


def recovery_multiplier(day) -> np.ndarray:
    """Mobility relative to the pre-disaster level for any non-negative day."""
    day = np.asarray(day, dtype=float)
    if np.any(day < 0):
        raise ValueError("recovery day cannot be negative")
    initial = float(P.RECOVERY_INITIAL_LEVEL)
    plateau = float(P.RECOVERY_PLATEAU_LEVEL)
    ratio = plateau / initial - 1.0
    return plateau / (1.0 + ratio * np.exp(-recovery_rate() * day))


def _baseline_gravity_pattern(model: dict) -> pd.DataFrame:
    table = add_gravity_inputs(complete_od_table())
    predicted = predict_gravity_flow(
        model,
        table["origin_total"].to_numpy(float),
        table["destination_total"].to_numpy(float),
        table["travel_cost_minutes"].to_numpy(float),
    )
    # The intercept score equation should already match total observed trips. Normalizing removes
    # only optimizer-rounding drift and makes the daily-total contract exact by construction.
    predicted *= float(table["observed_flow"].sum()) / float(predicted.sum())
    return table.assign(gravity_baseline_flow=predicted)


def build_daily_od_table(model: dict, days=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the recovery curve and every day's complete 24-zone OD table."""
    if days is None:
        days = np.arange(int(P.RECOVERY_SETTLING_DAYS) + 1)
    days = np.asarray(days, dtype=int)
    multipliers = recovery_multiplier(days)
    curve = pd.DataFrame(
        {
            "day": days,
            "mobility_relative_to_pre_disaster": multipliers,
        }
    )
    baseline = _baseline_gravity_pattern(model)
    blocks = []
    for day, multiplier in zip(days, multipliers):
        block = baseline[
            [
                "origin",
                "destination",
                "travel_cost_minutes",
                "gravity_baseline_flow",
            ]
        ].copy()
        block.insert(0, "day", int(day))
        block.insert(1, "mobility_relative_to_pre_disaster", float(multiplier))
        block["od_flow"] = block["gravity_baseline_flow"] * float(multiplier)
        blocks.append(block)
    return curve, pd.concat(blocks, ignore_index=True)


def plot_recovery_curve(curve: pd.DataFrame, output_stem: Path = FIGURE) -> Path:
    """Render the scale-independent recovery assumption as a slide-readable PNG."""
    use_pub(slide=True)
    plt.rcParams.update(
        {
            "font.size": 18,
            "axes.labelsize": 20,
            "xtick.labelsize": 17,
            "ytick.labelsize": 17,
            "legend.fontsize": 15,
        }
    )
    fig, ax = plt.subplots(figsize=(9.0, 5.1))
    ax.plot(
        curve["day"],
        curve["mobility_relative_to_pre_disaster"],
        color=C["purple"],
        linewidth=4.0,
        label="Exogenous recovery path",
    )
    ax.axhline(
        1.0,
        color=C["neutral_dark"],
        linewidth=1.8,
        linestyle=(0, (5, 4)),
        label="Pre-disaster level",
    )
    ax.axhline(
        float(P.RECOVERY_PLATEAU_LEVEL),
        color=C["accent2"],
        linewidth=2.0,
        linestyle=(0, (2, 3)),
        label=f"Long-run plateau ({P.RECOVERY_PLATEAU_LEVEL:.3f})",
    )
    ax.set_xlabel("Days since disaster onset")
    ax.set_ylabel("Mobility relative to pre-disaster level")
    ax.set_xlim(0, int(P.RECOVERY_SETTLING_DAYS))
    ax.set_ylim(0, 1.06)
    ax.set_xticks(np.arange(0, int(P.RECOVERY_SETTLING_DAYS) + 1, 10))
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.9)
    ax.legend(loc="lower right")
    fig.tight_layout()
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    save_pub(fig, output_stem)
    plt.close(fig)
    return output_stem.with_suffix(".png")


def prepare_problem_setting(force_model: bool = False, render_figure: bool = True) -> dict:
    """Create or reuse the shared model, then materialize the daily exogenous OD inputs."""
    ensure_gravity_model(DEFAULT_MODEL, force=force_model)
    model = load_gravity_model(DEFAULT_MODEL)
    curve, daily_od = build_daily_od_table(model)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    curve.to_csv(CURVE_CSV, index=False)
    daily_od.to_csv(OD_CSV, index=False)
    metadata = {
        "gravity_model_path": str(DEFAULT_MODEL.relative_to(ROOT)).replace("\\", "/"),
        "gravity_model_version": int(model["model_version"]),
        "recovery_form": "L / (1 + (L / r0 - 1) * exp(-alpha * day))",
        "initial_level": float(P.RECOVERY_INITIAL_LEVEL),
        "plateau_level": float(P.RECOVERY_PLATEAU_LEVEL),
        "settling_days": int(P.RECOVERY_SETTLING_DAYS),
        "settling_fraction": float(P.RECOVERY_SETTLING_FRACTION),
        "alpha_per_day": float(recovery_rate()),
        "empirical_event_count": int(P.RECOVERY_EMPIRICAL_EVENTS),
        "empirical_source": str(P.RECOVERY_SOURCE),
        "daily_od_definition": "gravity_baseline_flow * recovery_multiplier(day)",
        "information_rule": (
            "exogenous and shared by every method; no realized repair duration, true severity, "
            "or method decision enters this table"
        ),
        "changes_repair_scoring_horizon": False,
    }
    META_JSON.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    figure = plot_recovery_curve(curve) if render_figure else None
    return {
        "model": model,
        "curve": curve,
        "daily_od": daily_od,
        "metadata": metadata,
        "figure": figure,
    }


def load_training_inputs() -> dict:
    """Load the one persisted model before training and expose its fixed recovery parameters."""
    ensure_gravity_model(DEFAULT_MODEL)
    model = load_gravity_model(DEFAULT_MODEL)
    return {
        "gravity_model": model,
        "recovery_initial_level": float(P.RECOVERY_INITIAL_LEVEL),
        "recovery_plateau_level": float(P.RECOVERY_PLATEAU_LEVEL),
        "recovery_rate_per_day": float(recovery_rate()),
        "recovery_settling_days": int(P.RECOVERY_SETTLING_DAYS),
    }


if __name__ == "__main__":
    result = prepare_problem_setting()
    print(json.dumps(result["metadata"], indent=2))
