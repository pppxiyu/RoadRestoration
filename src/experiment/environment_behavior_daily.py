"""Reproduce the five daily environment-behavior panels in the meeting deck.

Run ``python -m src.experiment.environment_behavior_daily`` (or
``python main.py --solve env-behavior --n 11``) from the repository root.
This is a separate experimental environment path. It never changes the
three-hour clock, demand law, rewards, or observations used by the RL agents.

One discrete repair/evaluation slot is interpreted as one day here. The same
integer nominal repair durations and the same 119-slot sample-derived horizon
are used as in the original figures; this changes their physical duration,
not their numerical state sequence. Each daily network assignment represents
one stationary traffic-demand snapshot; travel cost remains in minutes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import time
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from src import config as P
from src.environment import environment_behavior as eb
from src.environment.behavior_models import select_behavior_model
from src.environment.evaluate import _matrix_from_H, build_damaged_edges, od_travel_times
from src.environment.gravity import DEFAULT_MODEL, ensure_gravity_model, load_gravity_model
from src.methods.oracle import compute_horizon, select_oracle_instance
from src.problem.scenarios import nominal_durations, sample_scenarios
from src.environment.ue import solve_ue
from src.experiment.layout import study_dir


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = study_dir("human_behavior") / "n11_daily"
SETTLING_DAYS = (70, 35)
SETTLING_FRACTION = 0.99
DELAYED_RESPONSE_95_DAYS = 7.0
DELAYED_RESPONSE_TAU_DAYS = -DELAYED_RESPONSE_95_DAYS / math.log(0.05)
DAILY_TARGET_WEIGHT = -math.expm1(-1.0 / DELAYED_RESPONSE_TAU_DAYS)
DELAYED_TOLERANCE = 0.002
DELAYED_UE_GAP = 1e-4
DELAYED_UE_MAX_ITERATIONS = 1000
DELAYED_MAX_ITERATIONS = 100
DELAYED_RELAXATION = 0.5


def recovery_multiplier(days: np.ndarray, settling_days: int) -> np.ndarray:
    """Non-road recovery, reaching 99% of its plateau approach by day 35/70."""
    days = np.asarray(days, dtype=float)
    if np.any(days < 0):
        raise ValueError("days must be nonnegative")
    initial = float(P.RECOVERY_INITIAL_LEVEL)
    plateau = float(P.RECOVERY_PLATEAU_LEVEL)
    ratio = plateau / initial - 1.0
    rate = math.log(ratio / (1.0 / SETTLING_FRACTION - 1.0)) / settling_days
    return plateau / (1.0 + ratio * np.exp(-rate * days))


def _trajectory(days, context, recovery, external, reference, ref_diagnostics,
                actual_states, solutions):
    rows = []
    for index, day in enumerate(days):
        demand, actual_time, _, diagnostics = solutions[index]
        denominator = float(np.dot(demand, context["baseline_u"]))
        degradation = float(np.dot(demand, actual_time) / denominator)
        rows.append({
            "day": int(day),
            "external_recovery_fraction_of_normal": float(recovery[index]),
            "damaged_roads": len(actual_states[index]),
            "external_total_od_demand": float(np.sum(external[index])),
            "adjusted_total_od_demand": float(np.sum(demand)),
            "accessibility_degradation": degradation,
            **ref_diagnostics[index],
            **diagnostics,
        })
    frame = pd.DataFrame(rows)
    frame["accessibility_degradation_cumulative_mean"] = (
        frame["accessibility_degradation"].expanding().mean()
    )
    return frame


def _save_path(directory: Path, name: str, days, context, recovery, external,
               reference, ref_diagnostics, actual_states, solutions):
    frame = _trajectory(days, context, recovery, external, reference,
                        ref_diagnostics, actual_states, solutions)
    frame.to_csv(directory / f"{name}_trajectory.csv", index=False)
    np.savez_compressed(
        directory / f"{name}_od_solutions.npz",
        days=days,
        external=np.asarray(external),
        adjusted=np.asarray([item[0] for item in solutions]),
        reference_times=np.asarray(reference),
        actual_times=np.asarray([item[1] for item in solutions]),
        normal_od=context["H0"],
    )
    return frame


def _free_flow_od_times(context, damaged):
    edges = build_damaged_edges(context, damaged)
    links = pd.DataFrame({
        "from": np.r_[edges.u, edges.v],
        "to": np.r_[edges.v, edges.u],
        "cost": np.r_[edges.free_flow_time, edges.free_flow_time],
    })
    times = od_travel_times(links, context)
    if not np.isfinite(times).all():
        raise RuntimeError("ABC decomposition requires reachable OD pairs")
    return times


def _decompose_abc(days, context, external, solutions, reference_times,
                   actual_states, gamma, results_dir):
    """A=damage free-flow delay, B=actual congestion, C=reference congestion."""
    normal_free_flow = _free_flow_od_times(context, {})
    state_cache = {}
    actual_free_flow = []
    for damaged in actual_states:
        key = tuple(sorted(damaged.items()))
        if key not in state_cache:
            state_cache[key] = _free_flow_od_times(context, damaged)
        actual_free_flow.append(state_cache[key])
    actual_free_flow = np.asarray(actual_free_flow)
    actual_times = np.asarray([item[1] for item in solutions])
    adjusted = np.asarray([item[0] for item in solutions])
    reference_times = np.asarray(reference_times)
    external = np.asarray(external)
    weights = external / external.sum(axis=1, keepdims=True)
    A = actual_free_flow - normal_free_flow
    B = actual_times - actual_free_flow
    C = reference_times - normal_free_flow
    delta = actual_times - reference_times
    if not np.allclose(A + B - C, delta, atol=1e-10):
        raise AssertionError("A+B-C does not reconstruct the OD travel-time gap")
    if min(float(A.min()), float(B.min()), float(C.min())) < -1e-6:
        raise AssertionError("a travel-time decomposition component is negative")
    mean = lambda matrix: np.sum(weights * matrix, axis=1)
    normal_total = float(context["H0"].sum())
    gap_normal = 100 * (external.sum(axis=1) - adjusted.sum(axis=1)) / normal_total
    gap_external = 100 * (1 - adjusted.sum(axis=1) / external.sum(axis=1))
    reconstructed = 100 * np.sum(weights * (-np.expm1(-gamma * delta)), axis=1)
    decomposition = pd.DataFrame({
        "day": days,
        "A_direct_damage_minutes": mean(A),
        "B_actual_congestion_minutes": mean(B),
        "C_reference_congestion_minutes": mean(C),
        "mean_travel_time_difference_minutes": mean(delta),
        "demand_gap_percent_normal": gap_normal,
        "demand_gap_percent_external": gap_external,
        "reconstructed_gap_percent_external": reconstructed,
    })
    decomposition.to_csv(results_dir / "abc_decomposition_35_days.csv", index=False)
    np.savez_compressed(
        results_dir / "abc_od_components_35_days.npz",
        days=days, A=A, B=B, C=C, external_weights=weights,
        actual_free_flow_times=actual_free_flow,
        normal_free_flow_times=normal_free_flow,
    )
    return decomposition


def _solve_delayed(days, context, external, reference_times, actual_states,
                   immediate_solutions, gamma, results_dir):
    """Damped Picard/UE solve with persistent OD adjustment and UE warm starts.

    q_t = E_t + (1-w)(q_{t-1}-E_{t-1}) + w E_t[exp(-gamma*Delta c_t(q_t))-1].
    The contemporaneous travel-time term is recomputed inside every iteration.
    Reference UE travel times are reused from the immediate 35-day experiment;
    a network state is built only once per distinct repair-completion prefix.
    """
    external = np.asarray(external)
    reference_times = np.asarray(reference_times)
    previous_deviation = np.zeros_like(external[0])
    warm = None
    road_edges = {}
    adjusted = []
    actual_times = []
    traces = []
    diagnostics = []
    for index, day in enumerate(days):
        forecast = external[index]
        history = (1.0 - DAILY_TARGET_WEIGHT) * previous_deviation
        state_key = tuple(sorted(actual_states[index].items()))
        if state_key not in road_edges:
            road_edges[state_key] = build_damaged_edges(context, actual_states[index])
        edges = road_edges[state_key]
        q = (forecast + history + DAILY_TARGET_WEIGHT
             * (immediate_solutions[index][0] - forecast))
        for iteration in range(1, DELAYED_MAX_ITERATIONS + 1):
            seed = eb._feasible_reference_seed(edges, q, context, warm)
            links, convergence = solve_ue(
                edges, _matrix_from_H(q, context), context["zone_ids"],
                rgap=DELAYED_UE_GAP, max_iter=DELAYED_UE_MAX_ITERATIONS,
                quiet=True, cores=1, x0=seed,
            )
            raw, costs = eb._penalized_times(links, context)
            warm = (links, np.where(np.isfinite(raw), q, 0.0))
            full_target_adjustment = forecast * np.expm1(
                -gamma * (costs - reference_times[index])
            )
            target = forecast + history + DAILY_TARGET_WEIGHT * full_target_adjustment
            if float(target.min()) <= 0:
                raise RuntimeError(f"nonpositive OD demand on day {day}")
            residual = float(np.max(np.abs(target - q) / np.maximum(forecast, 1.0)))
            traces.append({
                "day": int(day), "iteration": iteration,
                "fixed_point_residual": residual,
                "traffic_equilibrium_relative_gap": float(convergence.rgap),
                "adjusted_total_od_demand": float(q.sum()),
            })
            if residual <= DELAYED_TOLERANCE and convergence.rgap <= 0.00011:
                break
            q = DELAYED_RELAXATION * q + (1.0 - DELAYED_RELAXATION) * target
        else:
            raise RuntimeError(f"delayed OD fixed point did not converge on day {day}")
        previous_deviation = q - forecast
        adjusted.append(q.copy())
        actual_times.append(costs.copy())
        diagnostics.append((residual, iteration, float(convergence.rgap)))

    adjusted = np.asarray(adjusted)
    actual_times = np.asarray(actual_times)
    pd.DataFrame(traces).to_csv(results_dir / "gradual_response_convergence.csv", index=False)
    np.savez_compressed(
        results_dir / "gradual_response_7day_od_solutions.npz",
        days=days, external=external, adjusted=adjusted,
        reference_times=reference_times, actual_times=actual_times,
        normal_od=context["H0"],
    )
    frame = pd.DataFrame({
        "day": days,
        "external_total_od_demand": external.sum(axis=1),
        "adjusted_total_od_demand": adjusted.sum(axis=1),
        "fixed_point_residual": [row[0] for row in diagnostics],
        "fixed_point_iterations": [row[1] for row in diagnostics],
        "traffic_equilibrium_relative_gap": [row[2] for row in diagnostics],
    })
    frame.to_csv(results_dir / "gradual_response_7day_trajectory.csv", index=False)
    return frame


def _replace_output(staging: Path, output: Path, n: int):
    """Replace only the precisely named, validated generated-output directory."""
    expected = (study_dir("human_behavior") / f"n{int(n)}_daily").resolve()
    if output.resolve() != expected or output.name != f"n{int(n)}_daily":
        raise ValueError("refusing to replace a directory outside the named output target")
    if staging.resolve() != output.parent.resolve() / f"n{int(n)}_daily_staging":
        raise ValueError("unexpected staging directory")
    if output.is_symlink() or staging.is_symlink():
        raise ValueError("refusing to replace a symlink")
    expected_images = [
        "01_recovery_70_days.png", "02_recovery_35_days.png",
        "03_recovery_35_days_repeat.png", "04_abc_decomposition_35_days.png",
        "05_gradual_response_95pct_in_7_days.png",
    ]
    from PIL import Image
    for name in expected_images:
        with Image.open(staging / name) as picture:
            picture.verify()
    archive = None
    if output.exists():
        archive_root = ROOT / ".legacy"
        archive_root.mkdir(exist_ok=True)
        archive = archive_root / (
            "04-env_behavior_previous_"
            + datetime.now().strftime("%Y%m%d_%H%M%S")
            + "_" + uuid4().hex[:8]
        )
        if archive.resolve().parent != archive_root.resolve():
            raise ValueError("unexpected archive path")
        shutil.move(str(output), str(archive))
    try:
        staging.rename(output)
    except Exception:
        if archive is not None and not output.exists():
            shutil.move(str(archive), str(output))
        raise


def run_environment_behavior_daily(n: int = 11, output: Path | None = None):
    """Compute from raw Sioux Falls inputs, save intermediate arrays, render five PNGs."""
    select_behavior_model("elastic_daily", 24.0, for_methods=False)
    started = time.perf_counter()
    expected = study_dir("human_behavior") / f"n{int(n)}_daily"
    output = expected if output is None else Path(output)
    if output.resolve() != expected.resolve():
        raise ValueError("this reproducibility command writes only its named study folder")
    staging = output.parent / f"n{int(n)}_daily_staging"
    if staging.exists():
        raise FileExistsError(f"inspect or remove the stale staging directory: {staging}")
    results = staging / "data" / "results"
    results.mkdir(parents=True)
    disrupted = select_oracle_instance(eb.TOY, n=n)
    segments = sorted(int(edge) for edge in disrupted.edge_id)
    scenarios = sample_scenarios(disrupted, P.M_SCENARIOS, P.SEED)
    horizon = int(compute_horizon(segments, scenarios))
    days = np.arange(1, horizon + 1, dtype=int)
    context = eb.build_context(eb.TOY, disrupted, ue_cores=1)
    durations = nominal_durations(disrupted, segments)
    severity = {int(row.edge_id): int(row.severity) for row in disrupted.itertuples()}
    plans = eb._repair_plans(context, disrupted, segments, durations)
    actual_states = eb._damaged_states(days, segments, severity,
                                       plans["flow"]["completions"])
    intact_states = [{} for _ in days]

    ensure_gravity_model(DEFAULT_MODEL)
    gravity_model = load_gravity_model(DEFAULT_MODEL)
    gamma = -float(gravity_model["coefficients"]["travel_cost_minutes"])
    if gamma <= 0:
        raise ValueError("gravity cost sensitivity must be positive")

    paths = {}
    for settling_days in SETTLING_DAYS:
        recovery = recovery_multiplier(days, settling_days)
        external = np.asarray([context["H0"] * level for level in recovery])
        reference_times, reference_diagnostics = eb.solve_reference_path(
            context, external, intact_states
        )
        solutions, fixed_point_trace = eb.solve_actual_path(
            context, gamma, external, reference_times, reference_diagnostics,
            intact_states, actual_states, return_traces=True,
        )
        pd.DataFrame(fixed_point_trace).rename(
            columns={"time_interval": "day"}
        ).to_csv(results / f"recovery_{settling_days}_days_joint_iterations.csv",
                 index=False)
        frame = _save_path(
            results, f"recovery_{settling_days}_days", days, context,
            recovery, external, reference_times, reference_diagnostics,
            actual_states, solutions,
        )
        if float(frame.fixed_point_residual.max()) > eb.FIXED_POINT_TOLERANCE:
            raise AssertionError("immediate fixed-point tolerance was exceeded")
        paths[settling_days] = (frame, external, reference_times, solutions)
        print(f"computed {settling_days}-day external recovery", flush=True)

    fast_frame, fast_external, fast_reference, fast_solutions = paths[35]
    abc = _decompose_abc(
        days, context, fast_external, fast_solutions, fast_reference,
        actual_states, gamma, results,
    )
    delayed = _solve_delayed(
        days, context, fast_external, fast_reference, actual_states,
        fast_solutions, gamma, results,
    )
    if float(delayed.fixed_point_residual.max()) > DELAYED_TOLERANCE:
        raise AssertionError("gradual-response fixed-point tolerance was exceeded")

    from src.analysis.viz.environment_behavior_daily_viz import render_five_panels
    rendered = render_five_panels(
        staging, paths[70][0], fast_frame, abc, delayed,
        normal_total=float(context["H0"].sum()),
    )
    source_files = [
        ROOT / "data/siouxfalls_toy/network/edges.csv",
        ROOT / "data/siouxfalls_toy/network/od_pairs.csv",
        ROOT / f"data/siouxfalls_toy/instances/disrupted_segments_oracle{n}.csv",
    ]
    source_hashes = {
        str(path.relative_to(ROOT)).replace("\\", "/"):
            hashlib.sha256(path.read_bytes()).hexdigest()
        for path in source_files
    }
    metadata = {
        "pipeline": "independent daily environment-behavior reproduction",
        "command": f"python -m src.experiment.environment_behavior_daily --n {n}",
        "dependencies": "requirements.txt (Python 3.14.5)",
        "rl_environment_modified": False,
        "day_definition": "one existing discrete slot is one day in this separate experiment",
        "daily_assignment": "one stationary traffic-demand snapshot per day; travel cost in minutes",
        "n_disrupted": n,
        "horizon_days": horizon,
        "scenario_count_for_horizon": P.M_SCENARIOS,
        "scenario_seed": P.SEED,
        "nominal_repair_durations_in_days": {str(k): int(v) for k, v in durations.items()},
        "flow_priority_order": plans["flow"]["order"],
        "flow_priority_completion_days": plans["flow"]["completions"],
        "external_recovery_settling_days": list(SETTLING_DAYS),
        "external_recovery_settling_fraction": SETTLING_FRACTION,
        "gravity_cost_sensitivity_per_minute": gamma,
        "gravity_model_sha256": hashlib.sha256(DEFAULT_MODEL.read_bytes()).hexdigest(),
        "source_sha256": source_hashes,
        "figure_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in rendered
        },
        "immediate_fixed_point_tolerance": eb.FIXED_POINT_TOLERANCE,
        "immediate_solver": "joint convex elastic-demand assignment with path column generation",
        "reference_ue_relative_gap": eb.REFERENCE_UE_RELATIVE_GAP,
        "reference_ue_warm_start": bool(P.UE_WARM_START),
        "actual_solver_parallel_workers": eb.PARALLEL_WORKERS,
        "actual_solver_constant_state_chunk_days": eb.STATE_CHUNK_INTERVALS,
        "gradual_response_95_percent_days": DELAYED_RESPONSE_95_DAYS,
        "gradual_response_tau_days": DELAYED_RESPONSE_TAU_DAYS,
        "gradual_response_daily_target_weight": DAILY_TARGET_WEIGHT,
        "gradual_response_fixed_point_tolerance": DELAYED_TOLERANCE,
        "gradual_response_ue_relative_gap": DELAYED_UE_GAP,
        "gradual_response_damping": DELAYED_RELAXATION,
        "max_70_day_fixed_point_residual": float(paths[70][0].fixed_point_residual.max()),
        "max_35_day_fixed_point_residual": float(fast_frame.fixed_point_residual.max()),
        "max_gradual_fixed_point_residual": float(delayed.fixed_point_residual.max()),
        "max_abc_identity_error_minutes": float(np.max(np.abs(
            abc.A_direct_damage_minutes + abc.B_actual_congestion_minutes
            - abc.C_reference_congestion_minutes
            - abc.mean_travel_time_difference_minutes
        ))),
        "elapsed_seconds": time.perf_counter() - started,
    }
    config_dir = staging / "data" / "config"
    config_dir.mkdir()
    (config_dir / "run_meta.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    _replace_output(staging, output, n)
    print(json.dumps(metadata, indent=2), flush=True)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=11)
    args = parser.parse_args()
    run_environment_behavior_daily(n=args.n)


if __name__ == "__main__":
    main()
