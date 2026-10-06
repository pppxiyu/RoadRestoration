"""Shared traffic/OD helpers and the legacy four-reference comparison.

This module is deliberately separate from :mod:`util.evaluate`.  Running the
existing objective evaluator, baselines, or reinforcement-learning solvers
therefore keeps the repository's established demand model unchanged.  The
current five-panel daily experiment is in :mod:`util.environment_behavior_daily`;
the module and main.py commands enter that experiment. The older four-case
comparison below is retained as an explicitly callable legacy helper.

Four reference-policy assumptions are compared against the same actual
flow-priority restoration schedule:

1. instantaneous restoration with normal-period external OD demand;
2. flow-priority restoration with an exogenous OD recovery curve;
3. demand-priority restoration with the same recovery curve;
4. no restoration with the same recovery curve.

For each interval the fixed point is

    q = q_external * exp[-gamma * (c_actual(q) - c_reference(q_external))].

The joint convex solver lives in :mod:`util.fixed_point_demand` and is not
imported by the production simulation path.
"""

from __future__ import annotations

import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

import config as P
from util.evaluate import (
    _matrix_from_H,
    build_context,
    build_damaged_edges,
    od_travel_times,
    schedule_from_permutation,
)
from util.fixed_point_demand import solve_joint_elastic_assignment
from util.gravity import DEFAULT_MODEL, ensure_gravity_model, load_gravity_model
from util.greedy import score_demand, score_flow
from util.oracle import (
    _baseline_twoway_flow,
    compute_horizon,
    select_oracle_instance,
)
from util.scenarios import nominal_durations, sample_scenarios
from util.ue import _Network, solve_ue, warm_start_seed


ROOT = Path(__file__).resolve().parents[1]
TOY = ROOT / "data" / "siouxfalls_toy"
OUTPUT = ROOT / "outputs/01-sim_val_n_problem_setting/04-env_behavior"
RESULTS = OUTPUT / "results"
CONFIG = OUTPUT / "config"

# These assumptions belong only to this experimental path.  They do not alter
# config.py or the horizon/demand model used by util.evaluate and the RL agents.
RECOVERY_SETTLING_INTERVALS = 70
RECOVERY_SETTLING_FRACTION = 0.99
FIXED_POINT_TOLERANCE = 1e-2
REFERENCE_UE_RELATIVE_GAP = 1e-3
REFERENCE_UE_MAX_ITERATIONS = 100
STATE_CHUNK_INTERVALS = 6
PARALLEL_WORKERS = min(8, os.cpu_count() or 1)

CASE_FILES = {
    "instantaneous": "02_instantaneous_restoration_reference.csv",
    "flow": "03_flow_priority_reference.csv",
    "demand": "04_demand_priority_reference.csv",
    "none": "05_no_restoration_reference.csv",
}

_WORKER_CONTEXT = None
_WORKER_GAMMA = None


def interval_recovery_multiplier(intervals) -> np.ndarray:
    """Exogenous OD recovery used only by the environment-behavior experiment."""
    intervals = np.asarray(intervals, dtype=float)
    if np.any(intervals < 0.0):
        raise ValueError("recovery intervals must be non-negative")
    initial = float(P.RECOVERY_INITIAL_LEVEL)
    plateau = float(P.RECOVERY_PLATEAU_LEVEL)
    ratio = plateau / initial - 1.0
    rate = math.log(
        ratio / (1.0 / RECOVERY_SETTLING_FRACTION - 1.0)
    ) / RECOVERY_SETTLING_INTERVALS
    return plateau / (1.0 + ratio * np.exp(-rate * intervals))


def _damaged_states(intervals, segments, severity, completions):
    return [
        {
            edge: severity[edge]
            for edge in segments
            if int(interval) < completions[edge]
        }
        for interval in intervals
    ]


def _penalized_times(links: pd.DataFrame, context: dict):
    raw = od_travel_times(links, context)
    return raw, np.where(np.isfinite(raw), raw, context["u_pen"])


def _feasible_reference_seed(edges, demand, context, warm):
    if not P.UE_WARM_START or warm is None:
        return None
    previous_links, previous_routed_demand = warm
    increment = demand - previous_routed_demand
    if float(increment.min()) >= -1e-9:
        return warm_start_seed(
            edges,
            _matrix_from_H(np.clip(increment, 0.0, None), context),
            context["zone_ids"],
            previous_links,
        )
    network = _Network(
        edges,
        _matrix_from_H(demand, context),
        context["zone_ids"],
    )
    mapped_previous_flow = network.map_flows(previous_links)
    return network.all_or_nothing(network.cost(mapped_previous_flow))


def solve_reference_path(context, external_demands, damaged_states):
    """Compute reference-policy travel times under the external OD forecast."""
    warm = None
    all_times = []
    diagnostics = []
    for demand, damaged in zip(external_demands, damaged_states):
        edges = build_damaged_edges(context, damaged)
        seed = _feasible_reference_seed(edges, demand, context, warm)
        links, convergence = solve_ue(
            edges,
            _matrix_from_H(demand, context),
            context["zone_ids"],
            rgap=REFERENCE_UE_RELATIVE_GAP,
            max_iter=REFERENCE_UE_MAX_ITERATIONS,
            quiet=True,
            cores=1,
            x0=seed,
        )
        raw, penalized = _penalized_times(links, context)
        warm = (links, np.where(np.isfinite(raw), demand, 0.0))
        all_times.append(penalized)
        diagnostics.append(
            {
                "reference_ue_iterations": int(convergence.iterations),
                "reference_ue_relative_gap": float(convergence.rgap),
            }
        )
    return all_times, diagnostics


def _initialize_worker(context, gamma):
    global _WORKER_CONTEXT, _WORKER_GAMMA
    _WORKER_CONTEXT = context
    _WORKER_GAMMA = float(gamma)


def _solve_actual_state_group(payload):
    """Solve one constant actual-road-state chunk with numerical warm starts."""
    tasks, capture_trace = payload
    actual_edges = build_damaged_edges(_WORKER_CONTEXT, tasks[0][5])
    previous_external = None
    previous_adjusted = None
    joint_warm_start = None
    solved = []

    for task in tasks:
        (
            index,
            external_demand,
            reference_times,
            reference_diagnostics,
            reference_damaged,
            actual_damaged,
        ) = task
        if actual_damaged == reference_damaged:
            demand = external_demand.copy()
            actual_times = reference_times
            diagnostics = {
                "fixed_point_iterations": 1,
                "fixed_point_residual": 0.0,
                "joint_optimizer_iterations": 0,
                "joint_path_count": 0,
                "cross_interval_warm_start_used": False,
            }
            trace = []
            previous_adjusted = demand
        else:
            if previous_adjusted is None:
                initial_demand = external_demand
            else:
                scale = np.divide(
                    external_demand,
                    previous_external,
                    out=np.ones_like(external_demand),
                    where=previous_external > 0.0,
                )
                initial_demand = previous_adjusted * scale
            demand, actual_times, diagnostics, trace, joint_warm_start = (
                solve_joint_elastic_assignment(
                    _WORKER_CONTEXT,
                    actual_edges,
                    external_demand,
                    reference_times,
                    _WORKER_GAMMA,
                    tolerance=FIXED_POINT_TOLERANCE,
                    precision_profile="one_percent",
                    warm_start=joint_warm_start,
                    initial_demand=initial_demand,
                )
            )
            previous_adjusted = demand
        solved.append(
            (
                index,
                demand,
                actual_times,
                reference_diagnostics,
                diagnostics,
                trace if capture_trace else [],
            )
        )
        previous_external = external_demand
    return solved


def solve_actual_path(
    context,
    gamma,
    external_demands,
    reference_times,
    reference_diagnostics,
    reference_states,
    actual_states,
    return_traces=False,
):
    """Solve every interval without changing the repository's production evaluator."""
    tasks = [
        (
            index,
            external_demands[index],
            reference_times[index],
            reference_diagnostics[index],
            reference_states[index],
            actual_states[index],
        )
        for index in range(len(external_demands))
    ]
    groups = []
    current_state = None
    for task in tasks:
        state_key = tuple(sorted(task[5].items()))
        if (
            not groups
            or state_key != current_state
            or len(groups[-1]) >= STATE_CHUNK_INTERVALS
        ):
            groups.append([])
            current_state = state_key
        groups[-1].append(task)

    workers = min(PARALLEL_WORKERS, len(groups))
    results = {}
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_initialize_worker,
        initargs=(context, gamma),
    ) as executor:
        futures = [
            executor.submit(_solve_actual_state_group, (group, return_traces))
            for group in groups
        ]
        for future in as_completed(futures):
            for index, demand, times, ref_diag, fp_diag, trace in future.result():
                results[index] = (demand, times, ref_diag, fp_diag, trace)
    ordered = [results[index] for index in range(len(tasks))]
    solutions = [entry[:4] for entry in ordered]
    if return_traces:
        traces = [
            {"time_interval": index + 1, **row}
            for index, entry in enumerate(ordered)
            for row in entry[4]
        ]
        return solutions, traces
    return solutions


def _repair_plans(context, disrupted, segments, durations):
    baseline_flow = _baseline_twoway_flow(TOY, cores=1)
    flow_scores = score_flow(context, baseline_flow, durations)
    demand_scores = score_demand(context, baseline_flow, durations)
    plans = {}
    for name, scores in (("flow", flow_scores), ("demand", demand_scores)):
        order = sorted(segments, key=lambda edge: (-scores[edge], edge))
        starts = schedule_from_permutation(order, durations, access=context["access"])
        completions = {
            edge: int(starts[edge] + durations[edge]) for edge in segments
        }
        plans[name] = {
            "order": order,
            "starts": starts,
            "completions": completions,
        }
    return plans


def _case_frame(
    intervals,
    context,
    multipliers,
    external_demands,
    reference_states,
    actual_states,
    solutions,
):
    rows = []
    for index, interval in enumerate(intervals):
        demand, times, reference_diag, fixed_point_diag = solutions[index]
        denominator = float(np.sum(demand * context["baseline_u"]))
        degradation = (
            float(np.sum(demand * times) / denominator)
            if denominator > 0.0
            else 1.0
        )
        rows.append(
            {
                "time_interval": int(interval),
                "hours_after_disaster": float(interval * P.DELTA_T_H),
                "mobility_relative_to_normal": float(multipliers[index]),
                "reference_damaged_roads": int(len(reference_states[index])),
                "actual_damaged_roads": int(len(actual_states[index])),
                "external_total_od_demand": float(external_demands[index].sum()),
                "adjusted_total_od_demand": float(demand.sum()),
                "accessibility_degradation": degradation,
                **reference_diag,
                **fixed_point_diag,
            }
        )
    frame = pd.DataFrame(rows)
    frame["accessibility_degradation_cumulative_mean"] = frame[
        "accessibility_degradation"
    ].expanding().mean()
    return frame


def run_environment_behavior(n=None, render=True):
    """Run the four reference-policy cases and optionally render their figures."""
    started = time.perf_counter()
    n = int(P.N_DISRUPTED_ORACLE if n is None else n)
    disrupted = select_oracle_instance(TOY, n=n)
    segments = sorted(int(edge) for edge in disrupted["edge_id"])
    scenarios = sample_scenarios(disrupted, P.M_SCENARIOS, P.SEED)
    horizon = int(compute_horizon(segments, scenarios))
    context = build_context(TOY, disrupted, ue_cores=1)
    durations = nominal_durations(disrupted, segments)
    severity = {
        int(row.edge_id): int(row.severity)
        for row in disrupted.itertuples(index=False)
    }
    plans = _repair_plans(context, disrupted, segments, durations)

    ensure_gravity_model(DEFAULT_MODEL)
    gravity_model = load_gravity_model(DEFAULT_MODEL)
    gamma = -float(gravity_model["coefficients"]["travel_cost_minutes"])

    intervals = np.arange(1, horizon + 1, dtype=int)
    recovery = interval_recovery_multiplier(intervals)
    recovery_demands = [context["H0"] * float(level) for level in recovery]
    normal_demands = [context["H0"].copy() for _ in intervals]
    flow_states = _damaged_states(
        intervals, segments, severity, plans["flow"]["completions"]
    )
    demand_states = _damaged_states(
        intervals, segments, severity, plans["demand"]["completions"]
    )
    no_repair_state = dict(severity)
    no_repair_states = [no_repair_state.copy() for _ in intervals]
    intact_states = [{} for _ in intervals]

    reference_paths = {}
    reference_paths["instantaneous"] = (
        [context["baseline_u"].copy() for _ in intervals],
        [
            {
                "reference_ue_iterations": 0,
                "reference_ue_relative_gap": 0.0,
            }
            for _ in intervals
        ],
    )
    for name, states in (
        ("flow", flow_states),
        ("demand", demand_states),
        ("none", no_repair_states),
    ):
        reference_paths[name] = solve_reference_path(
            context, recovery_demands, states
        )

    cases = {
        "instantaneous": {
            "external": normal_demands,
            "multipliers": np.ones_like(recovery),
            "reference_states": intact_states,
        },
        "flow": {
            "external": recovery_demands,
            "multipliers": recovery,
            "reference_states": flow_states,
        },
        "demand": {
            "external": recovery_demands,
            "multipliers": recovery,
            "reference_states": demand_states,
        },
        "none": {
            "external": recovery_demands,
            "multipliers": recovery,
            "reference_states": no_repair_states,
        },
    }

    RESULTS.mkdir(parents=True, exist_ok=True)
    CONFIG.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "time_interval": intervals,
            "hours_after_disaster": intervals * float(P.DELTA_T_H),
            "mobility_relative_to_normal": recovery,
        }
    ).to_csv(RESULTS / "exogenous_interval_recovery_curve.csv", index=False)

    frames = {}
    case_metadata = {}
    for name, case in cases.items():
        reference_times, reference_diagnostics = reference_paths[name]
        solutions = solve_actual_path(
            context,
            gamma,
            case["external"],
            reference_times,
            reference_diagnostics,
            case["reference_states"],
            flow_states,
        )
        frame = _case_frame(
            intervals,
            context,
            case["multipliers"],
            case["external"],
            case["reference_states"],
            flow_states,
            solutions,
        )
        frame.to_csv(RESULTS / CASE_FILES[name], index=False)
        frames[name] = frame
        case_metadata[name] = {
            "maximum_fixed_point_residual": float(
                frame["fixed_point_residual"].max()
            ),
            "maximum_adjusted_demand_relative_to_normal": float(
                frame["adjusted_total_od_demand"].max() / context["H0"].sum()
            ),
            "final_adjusted_demand_relative_to_normal": float(
                frame["adjusted_total_od_demand"].iloc[-1]
                / context["H0"].sum()
            ),
            "maximum_accessibility_degradation": float(
                frame["accessibility_degradation"].max()
            ),
            "final_accessibility_degradation": float(
                frame["accessibility_degradation"].iloc[-1]
            ),
        }

    metadata = {
        "pipeline": "experimental fixed-point OD-demand environment path",
        "production_simulation_unchanged": True,
        "instance_size": n,
        "horizon_intervals": horizon,
        "hours_per_interval": float(P.DELTA_T_H),
        "external_recovery_settling_intervals": RECOVERY_SETTLING_INTERVALS,
        "external_recovery_settling_fraction": RECOVERY_SETTLING_FRACTION,
        "fixed_point_tolerance": FIXED_POINT_TOLERANCE,
        "gravity_cost_sensitivity_per_minute": gamma,
        "actual_repair_rule": "descending normal-period two-way user-equilibrium flow",
        "flow_priority_order": plans["flow"]["order"],
        "demand_priority_order": plans["demand"]["order"],
        "flow_priority_completion_intervals": plans["flow"]["completions"],
        "demand_priority_completion_intervals": plans["demand"]["completions"],
        "case_results": case_metadata,
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    (CONFIG / "run_meta.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    if render:
        from viz.environment_behavior_viz import render_environment_behavior

        render_environment_behavior(frames, normal_total=float(context["H0"].sum()))
    print(json.dumps(metadata, indent=2), flush=True)
    return frames, metadata


if __name__ == "__main__":
    # Keep this older comparison implementation available for direct imports,
    # but make the documented module command reproduce the current daily study.
    from util.environment_behavior_daily import main

    main()
