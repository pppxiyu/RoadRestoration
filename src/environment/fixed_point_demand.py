"""Jointly solve traffic assignment and the gravity-demand fixed point.

This module is an experimental alternative simulation path.  The repository's
existing objective evaluator and reinforcement-learning environments do not
import it.  Callers must supply an external OD forecast and reference travel
times explicitly, which keeps the reference-policy assumption visible.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.sparse import csc_matrix, csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.special import xlogy

from src.environment.evaluate import _matrix_from_H
from src.environment.ue import _Network


def _shortest_paths(network, costs, od_pairs):
    graph = csr_matrix(
        (costs, (network.ti, network.hi)),
        shape=(network.n, network.n),
    )
    by_origin = {}
    paths = [None] * len(od_pairs)
    travel_times = np.full(len(od_pairs), np.inf)
    for pair_index, (origin_node, destination_node) in enumerate(od_pairs):
        origin = network.pos[int(origin_node)]
        destination = network.pos[int(destination_node)]
        if origin not in by_origin:
            by_origin[origin] = dijkstra(
                graph,
                directed=True,
                indices=origin,
                return_predecessors=True,
            )
        distances, predecessors = by_origin[origin]
        travel_times[pair_index] = distances[destination]
        if not np.isfinite(distances[destination]):
            continue
        current = destination
        reversed_path = []
        while current != origin:
            previous = int(predecessors[current])
            if previous < 0:
                reversed_path = []
                break
            reversed_path.append(network._arc[(previous, current)])
            current = previous
        paths[pair_index] = tuple(reversed(reversed_path))
    return paths, travel_times


def _incidence_matrix(paths, n_arcs):
    rows = []
    columns = []
    for column, path in enumerate(paths):
        rows.extend(path)
        columns.extend([column] * len(path))
    return csc_matrix(
        (np.ones(len(rows)), (rows, columns)),
        shape=(n_arcs, len(paths)),
    )


def solve_joint_elastic_assignment(
    ctx,
    edges,
    external_demand,
    reference_times,
    gamma,
    tolerance=0.01,
    max_column_rounds=15,
    precision_profile="strict",
    warm_start=None,
    initial_demand=None,
):
    """Solve road flows and OD demand together under the same gravity fixed point.

    The objective is the Beckmann traffic-assignment potential plus the integral of the
    inverse exponential demand function.  Column generation adds shortest paths until no
    omitted path can improve the solution.  The returned residual is evaluated directly
    against q = q_external exp[-gamma(c(q)-c_reference)].
    """
    network = _Network(
        edges,
        _matrix_from_H(external_demand, ctx),
        ctx["zone_ids"],
    )
    warm_start_used = warm_start is not None
    if warm_start_used:
        reachable = np.asarray(warm_start["reachable"], dtype=bool).copy()
        path_arcs = [tuple(path) for path in warm_start["path_arcs"]]
        path_od = np.asarray(warm_start["path_od"], dtype=int).copy()
        previous_path_flow = np.asarray(warm_start["path_flow"], dtype=float)
        path_keys = {
            (int(pair_index), path)
            for pair_index, path in zip(path_od, path_arcs)
        }
        desired_initial_demand = (
            np.asarray(initial_demand, dtype=float)
            if initial_demand is not None
            else np.asarray(external_demand, dtype=float)
        )
        previous_demand = np.bincount(
            path_od,
            weights=previous_path_flow,
            minlength=len(external_demand),
        )
        scale = np.divide(
            desired_initial_demand,
            previous_demand,
            out=np.ones_like(desired_initial_demand),
            where=previous_demand > 1e-12,
        )
        path_flow = previous_path_flow * scale[path_od]
    else:
        initial_paths, free_flow_times = _shortest_paths(
            network,
            network.t0,
            ctx["od_pairs"],
        )
        reachable = np.isfinite(free_flow_times)
        path_arcs = []
        path_od = []
        path_keys = set()
        initial_flow = []
        for pair_index, path in enumerate(initial_paths):
            if path is None:
                continue
            path_arcs.append(path)
            path_od.append(pair_index)
            path_keys.add((pair_index, path))
            initial_flow.append(float(external_demand[pair_index]))
        path_od = np.asarray(path_od, dtype=int)
        path_flow = np.asarray(initial_flow, dtype=float)
    if not path_arcs:
        penalized_times = np.full_like(external_demand, ctx["u_pen"])
        demand = external_demand * np.exp(
            -gamma * (penalized_times - reference_times)
        )
        warm_state = {
            "reachable": reachable,
            "path_arcs": [],
            "path_od": np.asarray([], dtype=int),
            "path_flow": np.asarray([], dtype=float),
        }
        return demand, penalized_times, {
            "fixed_point_iterations": 0,
            "fixed_point_residual": 0.0,
            "joint_optimizer_iterations": 0,
            "joint_path_count": 0,
            "cross_interval_warm_start_used": warm_start_used,
            "actual_ue_iterations_last_solve": 0,
            "actual_ue_relative_gap_last_solve": 0.0,
        }, [], warm_state

    trace = []
    total_optimizer_iterations = 0
    one_percent_profile = precision_profile == "one_percent"
    path_cost_tolerance = (
        min(0.05, np.log1p(tolerance) / gamma)
        if one_percent_profile
        else 1e-5
    )
    refinement_round = False
    path_tightening_rounds = 0

    for column_round in range(1, max_column_rounds + 1):
        incidence = _incidence_matrix(path_arcs, network.m)

        def objective_and_gradient(candidate_path_flow):
            link_flow = np.asarray(incidence @ candidate_path_flow).ravel()
            demand = np.bincount(
                path_od,
                weights=candidate_path_flow,
                minlength=len(external_demand),
            )
            network_objective = np.sum(
                network.t0
                * (
                    link_flow
                    + network.alpha
                    * np.power(link_flow, network.beta + 1.0)
                    / (
                        (network.beta + 1.0)
                        * np.power(network.cap, network.beta)
                    )
                )
            )
            q = demand[reachable]
            q_external = external_demand[reachable]
            demand_objective = np.sum(
                (xlogy(q, q) - q * np.log(q_external) - q) / gamma
                - reference_times[reachable] * q
            )
            link_cost = network.cost(link_flow)
            demand_gradient = np.zeros_like(external_demand)
            demand_gradient[reachable] = (
                np.log(np.maximum(q, 1e-300) / q_external) / gamma
                - reference_times[reachable]
            )
            gradient = np.asarray(incidence.T @ link_cost).ravel()
            gradient += demand_gradient[path_od]
            return float(network_objective + demand_objective), gradient

        if one_percent_profile and not refinement_round:
            optimizer_ftol = 1e-9
            optimizer_gtol = 1e-4
            optimizer_maxiter = 500
        else:
            optimizer_ftol = 1e-13 if not one_percent_profile else 1e-12
            optimizer_gtol = 1e-7 if not one_percent_profile else 1e-6
            optimizer_maxiter = 1000
        optimization = minimize(
            objective_and_gradient,
            path_flow,
            method="L-BFGS-B",
            jac=True,
            bounds=[(0.0, None)] * len(path_flow),
            options={
                "maxiter": optimizer_maxiter,
                "ftol": optimizer_ftol,
                "gtol": optimizer_gtol,
                "maxls": 50,
                "maxcor": 20,
            },
        )
        total_optimizer_iterations += int(optimization.nit)
        path_flow = np.maximum(optimization.x, 0.0)
        link_flow = np.asarray(incidence @ path_flow).ravel()
        demand = np.bincount(
            path_od,
            weights=path_flow,
            minlength=len(external_demand),
        )
        link_cost = network.cost(link_flow)
        shortest_paths, raw_travel_times = _shortest_paths(
            network,
            link_cost,
            ctx["od_pairs"],
        )
        actual_times = np.where(
            np.isfinite(raw_travel_times),
            raw_travel_times,
            ctx["u_pen"],
        )
        target = external_demand * np.exp(
            -gamma * (actual_times - reference_times)
        )
        demand[~reachable] = target[~reachable]
        residual = float(
            np.max(
                np.abs(target - demand)
                / np.maximum(external_demand, 1.0)
            )
        )
        demand_gradient = np.zeros_like(external_demand)
        demand_gradient[reachable] = (
            np.log(
                np.maximum(demand[reachable], 1e-300)
                / external_demand[reachable]
            )
            / gamma
            - reference_times[reachable]
        )
        new_paths = []
        minimum_reduced_cost = 0.0
        for pair_index, path in enumerate(shortest_paths):
            if path is None:
                continue
            reduced_cost = float(
                raw_travel_times[pair_index]
                + demand_gradient[pair_index]
            )
            minimum_reduced_cost = min(minimum_reduced_cost, reduced_cost)
            key = (pair_index, path)
            if reduced_cost < -path_cost_tolerance and key not in path_keys:
                new_paths.append((pair_index, path))
                path_keys.add(key)
        trace.append(
            {
                "fixed_point_iteration": column_round,
                "solver_phase": "joint_elastic_assignment",
                "phase_iteration": column_round,
                "fixed_point_residual": residual,
                "adjusted_total_od_demand": float(demand.sum()),
                "relaxation_fraction": np.nan,
                "traffic_assignment_iterations": int(optimization.nit),
                "traffic_assignment_relative_gap": np.nan,
                "traffic_assignment_target_gap": np.nan,
                "fixed_point_update_method": "convex optimization with path column generation",
                "path_count": len(path_arcs),
                "new_paths": len(new_paths),
                "minimum_reduced_cost": minimum_reduced_cost,
                "path_cost_tolerance": path_cost_tolerance,
                "optimizer_ftol": optimizer_ftol,
                "optimizer_gtol": optimizer_gtol,
            }
        )
        if residual <= tolerance and not new_paths:
            warm_state = {
                "reachable": reachable.copy(),
                "path_arcs": tuple(path_arcs),
                "path_od": path_od.copy(),
                "path_flow": path_flow.copy(),
            }
            return demand, actual_times, {
                "fixed_point_iterations": column_round,
                "fixed_point_residual": residual,
                "joint_optimizer_iterations": total_optimizer_iterations,
                "joint_path_count": len(path_arcs),
                "cross_interval_warm_start_used": warm_start_used,
                "actual_ue_iterations_last_solve": 0,
                "actual_ue_relative_gap_last_solve": 0.0,
            }, trace, warm_state
        if not new_paths:
            if one_percent_profile and not refinement_round:
                refinement_round = True
                continue
            if (
                one_percent_profile
                and residual > tolerance
                and path_cost_tolerance > 1e-4
                and path_tightening_rounds < 3
            ):
                path_cost_tolerance = max(1e-4, 0.1 * path_cost_tolerance)
                path_tightening_rounds += 1
                refinement_round = True
                continue
            break
        refinement_round = False
        path_arcs.extend(path for _, path in new_paths)
        path_od = np.concatenate(
            [
                path_od,
                np.asarray([pair_index for pair_index, _ in new_paths], dtype=int),
            ]
        )
        path_flow = np.concatenate(
            [path_flow, np.zeros(len(new_paths), dtype=float)]
        )

    raise RuntimeError(
        "Joint elastic-demand assignment did not reach the fixed-point tolerance; "
        f"final residual={residual:.3e}, paths={len(path_arcs)}"
    )
