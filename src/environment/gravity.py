"""Fit the Sioux Falls trip-distribution gravity model.

The model uses the observed origin production and destination attraction totals.
Its impedance variable is the shortest-path travel time after assigning the
normal-period OD matrix to the intact network at user equilibrium, so congestion
is part of the fitted travel cost.
It is estimated by Poisson pseudo-maximum likelihood, so observed zero-flow OD
pairs remain valid training observations.

Run from the repository root:
    python -m src.environment.gravity
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln

from src import config as P
from src.experiment.layout import study_dir
from src.problem.io import load_toy_network, od_to_matrix
from src.environment.ue import solve_ue


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = study_dir("human_behavior") / "gravity"
MODEL_VERSION = 2
TRAVEL_COST_BASIS = "baseline_user_equilibrium_congested_shortest_path_v1"
GRAVITY_UE_TARGET_RELATIVE_GAP = float(P.UE_RGAP_DEF)
GRAVITY_UE_MAX_ITERATIONS = 2000
DEFAULT_MODEL = DEFAULT_OUTPUT / "data" / "gravity_model.json"
MODEL_SOURCES = (
    ROOT / "data/siouxfalls_toy/network/od_pairs.csv",
    ROOT / "data/siouxfalls_toy/network/edges.csv",
)


def _source_fingerprint() -> dict[str, str]:
    """Hashes of the raw inputs that define the fitted model."""
    return {
        str(path.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in MODEL_SOURCES
    }


def complete_od_table() -> pd.DataFrame:
    """Return every off-diagonal OD pair, retaining observed zero flows."""
    observed = pd.read_csv(ROOT / "data/siouxfalls_toy/network/od_pairs.csv")
    zones = sorted(set(observed["origin"]) | set(observed["destination"]))
    table = pd.MultiIndex.from_product(
        [zones, zones], names=["origin", "destination"]
    ).to_frame(index=False)
    table = table[table["origin"] != table["destination"]].copy()
    table = table.merge(
        observed[["origin", "destination", "h0"]],
        on=["origin", "destination"],
        how="left",
    )
    table["observed_flow"] = table.pop("h0").fillna(0.0)
    return table


def add_gravity_inputs(table: pd.DataFrame) -> pd.DataFrame:
    """Add observed marginals and normal-period congested shortest-path time."""
    table = table.copy()
    productions = table.groupby("origin")["observed_flow"].sum()
    attractions = table.groupby("destination")["observed_flow"].sum()
    table["origin_total"] = table["origin"].map(productions)
    table["destination_total"] = table["destination"].map(attractions)

    toy = ROOT / "data/siouxfalls_toy"
    edges, observed_od, zone_ids = load_toy_network(toy)
    links, convergence = solve_ue(
        edges,
        od_to_matrix(observed_od, zone_ids),
        zone_ids,
        rgap=GRAVITY_UE_TARGET_RELATIVE_GAP,
        max_iter=GRAVITY_UE_MAX_ITERATIONS,
        quiet=True,
    )
    graph = nx.DiGraph()
    for origin, destination, cost in links[["from", "to", "cost"]].itertuples(
        index=False, name=None
    ):
        graph.add_edge(int(origin), int(destination), weight=float(cost))
    shortest = {
        int(origin): nx.single_source_dijkstra_path_length(
            graph, int(origin), weight="weight"
        )
        for origin in sorted(table["origin"].unique())
    }
    table["travel_cost_minutes"] = [
        shortest[int(origin)][int(destination)]
        for origin, destination in table[["origin", "destination"]].itertuples(index=False)
    ]
    table.attrs.update(
        travel_cost_basis=TRAVEL_COST_BASIS,
        ue_relative_gap=float(convergence.rgap),
        ue_iterations=int(convergence.iterations),
        ue_target_relative_gap=GRAVITY_UE_TARGET_RELATIVE_GAP,
        ue_max_iterations=GRAVITY_UE_MAX_ITERATIONS,
    )
    return table


def _poisson_log_likelihood(observed: np.ndarray, predicted: np.ndarray) -> float:
    return float(
        np.sum(
            observed * np.log(np.maximum(predicted, 1e-300))
            - predicted
            - gammaln(observed + 1.0)
        )
    )


def fit_gravity(
    table: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Fit the multiplicative gravity equation by Poisson pseudo-likelihood."""
    observed = table["observed_flow"].to_numpy(float)
    design_raw = np.column_stack(
        [
            np.ones(len(table)),
            np.log(table["origin_total"].to_numpy(float)),
            np.log(table["destination_total"].to_numpy(float)),
            table["travel_cost_minutes"].to_numpy(float),
        ]
    )

    means = design_raw[:, 1:].mean(axis=0)
    scales = design_raw[:, 1:].std(axis=0)
    design = np.column_stack(
        [np.ones(len(table)), (design_raw[:, 1:] - means) / scales]
    )
    initial = np.linalg.lstsq(design, np.log(observed + 1.0), rcond=None)[0]

    def objective(coef: np.ndarray) -> tuple[float, np.ndarray]:
        linear_predictor = np.clip(design @ coef, -40.0, 40.0)
        predicted = np.exp(linear_predictor)
        return (
            float(np.sum(predicted - observed * linear_predictor)),
            design.T @ (predicted - observed),
        )

    fit = minimize(
        lambda coef: objective(coef)[0],
        initial,
        jac=lambda coef: objective(coef)[1],
        method="L-BFGS-B",
        options={"ftol": 1e-15, "gtol": 1e-8, "maxiter": 5000, "maxls": 100},
    )
    if not fit.success and np.linalg.norm(fit.jac, ord=np.inf) > 1e-3:
        raise RuntimeError(f"Gravity-model fit failed: {fit.message}")

    coef = np.empty_like(fit.x)
    coef[1:] = fit.x[1:] / scales
    coef[0] = fit.x[0] - np.sum(fit.x[1:] * means / scales)
    predicted = np.exp(design_raw @ coef)

    # Heteroskedasticity-robust sandwich covariance. This does not require the
    # conditional flow variance to equal its conditional mean.
    hessian = design_raw.T @ (predicted[:, None] * design_raw)
    bread = np.linalg.inv(hessian)
    residual = observed - predicted
    meat = design_raw.T @ ((residual**2)[:, None] * design_raw)
    observation_count, parameter_count = design_raw.shape
    covariance = (
        observation_count
        / (observation_count - parameter_count)
        * bread
        @ meat
        @ bread
    )
    robust_standard_error = np.sqrt(np.diag(covariance))

    names = [
        "intercept",
        "log_origin_total",
        "log_destination_total",
        "travel_cost_minutes",
    ]
    parameters = pd.DataFrame(
        {
            "parameter": names,
            "estimate": coef,
            "robust_standard_error": robust_standard_error,
            "robust_95_percent_lower": coef - 1.96 * robust_standard_error,
            "robust_95_percent_upper": coef + 1.96 * robust_standard_error,
        }
    )
    covariance_table = pd.DataFrame(covariance, index=names, columns=names)
    covariance_table.index.name = "parameter"

    output = table.copy()
    output["predicted_flow"] = predicted
    output["residual"] = observed - predicted

    null_prediction = np.full_like(observed, observed.mean())
    deviance_terms = predicted.copy()
    positive = observed > 0
    deviance_terms[positive] = (
        observed[positive]
        * np.log(observed[positive] / np.maximum(predicted[positive], 1e-300))
        - (observed[positive] - predicted[positive])
    )
    cost_coef = float(coef[3])
    cost_lower = float(parameters.iloc[3]["robust_95_percent_lower"])
    cost_upper = float(parameters.iloc[3]["robust_95_percent_upper"])
    diagnostics = {
        "model": "Poisson pseudo-maximum-likelihood multiplicative gravity model",
        "mean_specification": (
            "exp(intercept) * origin_total^beta_origin * "
            "destination_total^beta_destination * "
            "exp(beta_cost * travel_cost_minutes)"
        ),
        "travel_cost_definition": (
            "shortest-path sum of congested link travel times after normal-period "
            "OD demand is assigned to the intact network at user equilibrium, in minutes"
        ),
        "travel_cost_basis": table.attrs.get("travel_cost_basis", TRAVEL_COST_BASIS),
        "normal_period_user_equilibrium": {
            "achieved_relative_gap": float(table.attrs.get("ue_relative_gap", np.nan)),
            "iterations": int(table.attrs.get("ue_iterations", -1)),
            "target_relative_gap": float(
                table.attrs.get("ue_target_relative_gap", GRAVITY_UE_TARGET_RELATIVE_GAP)
            ),
            "maximum_iterations": int(
                table.attrs.get("ue_max_iterations", GRAVITY_UE_MAX_ITERATIONS)
            ),
        },
        "speed_limit_audit": (
            "speed field exists in SiouxFalls_net.tntp but all 76 directed-link values are zero"
        ),
        "observations": int(observation_count),
        "zero_flow_observations": int((observed == 0).sum()),
        "converged": bool(
            fit.success or np.linalg.norm(fit.jac, ord=np.inf) <= 1e-3
        ),
        "optimizer_message": str(fit.message),
        "mean_observed_flow": float(observed.mean()),
        "mean_predicted_flow": float(predicted.mean()),
        "mean_absolute_error": float(np.mean(np.abs(residual))),
        "root_mean_squared_error": float(np.sqrt(np.mean(residual**2))),
        "poisson_deviance": float(2.0 * deviance_terms.sum()),
        "mcfadden_pseudo_r_squared": float(
            1.0
            - _poisson_log_likelihood(observed, predicted)
            / _poisson_log_likelihood(observed, null_prediction)
        ),
        "travel_cost_minutes": {
            "minimum": float(table["travel_cost_minutes"].min()),
            "median": float(table["travel_cost_minutes"].median()),
            "mean": float(table["travel_cost_minutes"].mean()),
            "maximum": float(table["travel_cost_minutes"].max()),
        },
        "one_additional_minute": {
            "flow_multiplier": float(np.exp(cost_coef)),
            "flow_percent_decrease": float(100.0 * (1.0 - np.exp(cost_coef))),
            "robust_95_percent_lower": float(100.0 * (1.0 - np.exp(cost_upper))),
            "robust_95_percent_upper": float(100.0 * (1.0 - np.exp(cost_lower))),
        },
    }
    return parameters, covariance_table, output, diagnostics


def _artifact_from_fit(
    parameters: pd.DataFrame, covariance: pd.DataFrame, diagnostics: dict
) -> dict:
    estimates = dict(zip(parameters["parameter"], parameters["estimate"]))
    return {
        "model_version": MODEL_VERSION,
        "travel_cost_basis": TRAVEL_COST_BASIS,
        "fit_settings": {
            "ue_target_relative_gap": GRAVITY_UE_TARGET_RELATIVE_GAP,
            "ue_max_iterations": GRAVITY_UE_MAX_ITERATIONS,
        },
        "source_sha256": _source_fingerprint(),
        "formula": diagnostics["mean_specification"],
        "coefficients": {key: float(value) for key, value in estimates.items()},
        "covariance": {
            row: {column: float(covariance.loc[row, column]) for column in covariance.columns}
            for row in covariance.index
        },
        "diagnostics": diagnostics,
    }


def _artifact_is_current(artifact: dict) -> bool:
    return (
        artifact.get("model_version") == MODEL_VERSION
        and artifact.get("travel_cost_basis") == TRAVEL_COST_BASIS
        and artifact.get("fit_settings")
        == {
            "ue_target_relative_gap": GRAVITY_UE_TARGET_RELATIVE_GAP,
            "ue_max_iterations": GRAVITY_UE_MAX_ITERATIONS,
        }
        and artifact.get("source_sha256") == _source_fingerprint()
        and set(artifact.get("coefficients", {}))
        == {"intercept", "log_origin_total", "log_destination_total", "travel_cost_minutes"}
    )


def fit_and_save_gravity_model(model_path: Path = DEFAULT_MODEL) -> dict:
    """Fit once from raw Sioux Falls data and persist a transparent JSON model artifact."""
    parameters, covariance, _, diagnostics = fit_gravity(
        add_gravity_inputs(complete_od_table())
    )
    artifact = _artifact_from_fit(parameters, covariance, diagnostics)
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = model_path.with_suffix(model_path.suffix + ".tmp")
    temporary.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    temporary.replace(model_path)
    return artifact


def ensure_gravity_model(model_path: Path = DEFAULT_MODEL, force: bool = False) -> dict:
    """Reuse the persisted fit unless its raw inputs or schema have changed."""
    model_path = Path(model_path)
    if model_path.exists() and not force:
        artifact = json.loads(model_path.read_text(encoding="utf-8"))
        if _artifact_is_current(artifact):
            return artifact
    return fit_and_save_gravity_model(model_path)


def load_gravity_model(model_path: Path = DEFAULT_MODEL) -> dict:
    """Load the shared fitted model and reject stale or incompatible artifacts."""
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"gravity model is missing at {model_path}; run ensure_gravity_model() first"
        )
    artifact = json.loads(model_path.read_text(encoding="utf-8"))
    if not _artifact_is_current(artifact):
        raise RuntimeError(
            f"gravity model at {model_path} does not match the current raw inputs or schema"
        )
    artifact["path"] = str(model_path)
    return artifact


def predict_gravity_flow(
    model: dict,
    origin_total,
    destination_total,
    travel_cost_minutes,
) -> np.ndarray:
    """Evaluate the persisted mean equation for scalar or array-like inputs."""
    coef = model["coefficients"]
    origin_total = np.asarray(origin_total, dtype=float)
    destination_total = np.asarray(destination_total, dtype=float)
    travel_cost_minutes = np.asarray(travel_cost_minutes, dtype=float)
    linear = (
        float(coef["intercept"])
        + float(coef["log_origin_total"]) * np.log(origin_total)
        + float(coef["log_destination_total"]) * np.log(destination_total)
        + float(coef["travel_cost_minutes"]) * travel_cost_minutes
    )
    return np.exp(np.clip(linear, -40.0, 40.0))


def run(output_dir: Path = DEFAULT_OUTPUT) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    records = output_dir / "data"
    records.mkdir(parents=True, exist_ok=True)
    parameters, covariance, predictions, diagnostics = fit_gravity(
        add_gravity_inputs(complete_od_table())
    )
    artifact = _artifact_from_fit(parameters, covariance, diagnostics)
    DEFAULT_MODEL.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_MODEL.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    parameters.to_csv(records / "04_gravity_parameters.csv", index=False)
    covariance.to_csv(records / "04_gravity_parameter_covariance.csv")
    predictions.to_csv(records / "04_gravity_od_predictions.csv", index=False)
    (records / "04_gravity_model_results.json").write_text(
        json.dumps(diagnostics, indent=2), encoding="utf-8"
    )
    return diagnostics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir), indent=2))


if __name__ == "__main__":
    main()
