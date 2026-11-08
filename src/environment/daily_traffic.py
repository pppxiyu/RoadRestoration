"""Solve one day's demand and traffic; no repair scheduling or policy access."""
import hashlib
import json
import time
import numpy as np
import pandas as pd
from src import config as P
from src.environment.traffic_network import build_damaged_edges, od_travel_times, _matrix_from_H
from src.environment.gravity_model import DEFAULT_MODEL, load_gravity_model
from src.environment.ue import solve_ue, warm_start_seed
from src.environment.shifted_demand import solve_shifted
from src.environment.daily_context import context_from_problem, simulator_fingerprint
from src.environment.daily_contracts import TrafficState
from src.environment.daily_demand import DemandResponse, DemandEquation
from src.environment.daily_cache import TrafficCache, pack_state, unpack_state


def _feasible_daily_seed(edges, demand, context, warm):
    """A feasible seed even when some OD demands decrease and others increase.

    Scale every previously routed path by r = min(1, min(q_new/q_old)). The
    retained flow serves r*q_old; a new shortest-path loading serves the
    nonnegative remainder. Their sum serves EXACTLY q_new. Roads only reopen
    during restoration, so retained paths stay usable. This changes numerical
    initialization only, not the demand equation or equilibrium tolerance.
    """
    if not P.UE_WARM_START or warm is None:
        return None
    previous_links, previous_routed = warm
    positive = previous_routed > 0
    fraction = min(1.0, float(np.min(demand[positive] / previous_routed[positive]))) if positive.any() else 1.0
    if fraction < 0:
        raise ValueError("negative demand cannot seed an equilibrium")
    retained = previous_links.copy()
    retained["volume"] = retained["volume"] * fraction
    increment = demand - fraction * previous_routed
    if increment.min() < -1e-8:
        raise AssertionError("scaled warm start left a negative OD increment")
    return warm_start_seed(edges, _matrix_from_H(np.maximum(increment, 0), context),
                           context["zone_ids"], retained)


class DailyTraffic:
    def __init__(self, problem, cache_dir=None):
        self.problem = problem
        self.rules = problem["rules"]
        self.ctx = context_from_problem(problem)
        model = load_gravity_model(DEFAULT_MODEL)
        self.gamma = -float(model["coefficients"]["travel_cost_minutes"])
        if self.gamma <= 0:
            raise ValueError("human-response cost sensitivity must be positive")
        self.response_spec = problem.get("human_response")
        self.demand_model = DemandResponse(self.ctx["H0"], self.ctx["od_pairs"], self.rules,
                                           self.gamma, self.response_spec)
        self.od_function_weights = self.demand_model.weights
        self.weight = self.demand_model.weight
        self.fingerprint = simulator_fingerprint(problem, self.gamma, DEFAULT_MODEL)
        self.cache = TrafficCache(self.fingerprint, cache_dir,
                                  512 * 1024**2 if self.response_spec is not None else None)
        self.references = {}
        self.solve_calls = self.cache_hits = self.fixed_point_iterations = 0
        self.solve_seconds = 0.0

    # Compatibility accessors for existing diagnostics and cache checks.
    @property
    def memo(self):
        return self.cache.memo

    @property
    def connection(self):
        return self.cache.connection

    @property
    def cache_limit_bytes(self):
        return self.cache.limit_bytes

    @cache_limit_bytes.setter
    def cache_limit_bytes(self, value):
        self.cache.limit_bytes = value

    _pack = staticmethod(pack_state)
    _unpack = staticmethod(unpack_state)


    def _links(self, edges, demand, warm=None, response=False, restored=False):
        rules = self.rules
        seed = _feasible_daily_seed(edges, demand, self.ctx, warm)
        target = (rules.restored_ue_gap if restored else
                  rules.response_ue_gap if response else rules.ue_gap)
        limit = (rules.restored_ue_max_iter if restored else
                 rules.response_ue_max_iter if response else rules.ue_max_iter)
        links, convergence = solve_ue(
            edges, _matrix_from_H(demand, self.ctx), self.ctx["zone_ids"], x0=seed,
            rgap=target, max_iter=limit,
            quiet=True)
        if convergence.rgap > target:
            raise RuntimeError(f"daily traffic UE did not converge: {convergence.rgap:g} > {target:g}")
        self.solve_calls += 1
        raw = od_travel_times(links, self.ctx)
        return links, raw, np.where(np.isfinite(raw), raw, self.ctx["u_pen"]), convergence

    def reference(self, day):
        d = min(int(day), self.rules.recovery_days)
        if d not in self.references:
            q = self.demand_model.external(d)
            if d == self.rules.recovery_days:
                self.references[d] = (q, self.problem["baseline_links"], self.ctx["baseline_u"])
            else:
                links, _, costs, _ = self._links(self.ctx["edges"], q)
                self.references[d] = (q, links, costs)
        return self.references[d]

    def _nested_response(self, day, edges, forecast, reference_costs, history, warm, *, restored=False, gamma=None):
        """Retained numerical fallback and independent equivalence check.

        This uses the SAME demand equation and tolerances as the joint solver;
        failure is never accepted as a scored trajectory.
        """
        q = forecast + history
        gamma = self.gamma if gamma is None else gamma
        tolerance = (self.rules.restored_fixed_point_tolerance if restored
                     else self.rules.fixed_point_tolerance)
        equation = DemandEquation(forecast, reference_costs, history, self.weight, gamma)
        for iterations in range(1, self.rules.response_max_iter + 1):
            links, raw, costs, convergence = self._links(edges, q, warm, response=True, restored=restored)
            warm = (links, np.where(np.isfinite(raw), q, 0.0))
            target = equation.target(costs)
            if float(target.min()) <= 0:
                raise RuntimeError(f"nonpositive adjusted demand on day {day}")
            residual = float(np.max(np.abs(target-q) / np.maximum(forecast, 1.0)))
            if residual <= tolerance:
                return q, links, raw, costs, iterations, residual, float(convergence.rgap)
            w = self.rules.response_relaxation
            q = w*q + (1-w)*target
        raise RuntimeError(f"seven-day fixed point did not converge on day {day}: {residual:g}")

    def solve(self, day, damaged, behavior, previous=None):
        d = int(day)
        memory = self.demand_model.prepare(d, behavior, previous)
        unmet_before, gamma = memory.unmet_before, memory.gamma
        identity = (behavior, d if behavior == "response7d" else min(d, self.rules.recovery_days),
                    tuple(sorted(damaged.items())), previous.key if previous and behavior == "response7d" else None)
        key = hashlib.sha256(json.dumps(identity).encode()).hexdigest()
        cached = self.cache.get(key, d)
        if cached is not None:
            self.cache_hits += 1
            return cached
        started = time.perf_counter()
        forecast, reference_links, reference_costs = self.reference(d)
        edges = build_damaged_edges(self.ctx, damaged)
        warm = None if previous is None else (previous.links, previous.served_demand)
        iterations, residual, gap = 0, 0.0, 0.0
        solver_state, extra_diagnostic = None, {}
        if behavior == "natural":
            q = forecast.copy()
            if not damaged:
                links, costs = reference_links, reference_costs
                raw = costs
            else:
                # Natural-only demand has no memory. A deterministic cold solve
                # keeps this cache value independent of which repair history or
                # parallel worker first visits the same (day, damage) state.
                links, raw, costs, convergence = self._links(edges, q)
                gap = float(convergence.rgap)
        else:
            # q = a + b*exp[-gamma*(c(q)-c_reference)] with yesterday's
            # completed state held fixed. No immediate-response approximation.
            equation = self.demand_model.equation(forecast, reference_costs, previous, gamma)
            history = equation.history
            # Approved numerical refinement after every damaged road is repaired.
            # The behavior equation, observed state and physical stopping rule
            # are unchanged; only solution accuracy is increased near recovery.
            restored = not damaged
            fp_tolerance = (self.rules.restored_fixed_point_tolerance if restored
                            else self.rules.fixed_point_tolerance)
            ue_tolerance = self.rules.restored_ue_gap if restored else self.rules.response_ue_gap
            damage_record = [list(pair) for pair in sorted(damaged.items())]
            path_warm = None if previous is None else previous.solver_state
            if path_warm is not None and path_warm["damage"] != damage_record:
                path_warm = None
            if self.rules.response_solver == "joint":
                try:
                    q, costs, joint, solver_state, _ = solve_shifted(
                        self.ctx, edges, forecast, reference_costs, gamma,
                        equation.base, equation.scale,
                        tolerance=fp_tolerance,
                        ue_tolerance=ue_tolerance, warm_start=path_warm)
                    solver_state["damage"] = damage_record
                    self.solve_calls += 1
                    links = pd.DataFrame({"from": np.r_[edges.u, edges.v],
                                          "to": np.r_[edges.v, edges.u],
                                          "volume": solver_state["flow"], "cost": solver_state["link_cost"]})
                    raw = np.where(solver_state["reachable"], costs, np.inf)
                    iterations, residual, gap = joint["rounds"], joint["fixed_point_residual"], joint["ue_gap"]
                    extra_diagnostic = dict(solver="joint", optimizer_iterations=joint["iterations"],
                                            solver_fallback=False)
                except RuntimeError as error:
                    extra_diagnostic = dict(solver="nested", solver_fallback=True, fallback_reason=str(error))
                    solver_state = None
            elif self.rules.response_solver == "nested":
                extra_diagnostic = dict(solver="nested", solver_fallback=False)
            else:
                raise ValueError("unknown seven-day numerical solver")
            if solver_state is None:
                q, links, raw, costs, iterations, residual, gap = self._nested_response(
                    d, edges, forecast, reference_costs, history, warm, restored=restored, gamma=gamma)
            extra_diagnostic.update(refined_after_repairs=restored,
                                    fixed_point_tolerance=fp_tolerance, ue_tolerance=ue_tolerance)
        self.fixed_point_iterations += iterations
        flows = np.zeros(len(self.ctx["edges"]))
        congestion = np.zeros_like(flows)
        pairs = {tuple(sorted((int(r.u), int(r.v)))): (i, float(r.free_flow_time))
                 for i, r in enumerate(self.ctx["edges"].itertuples(index=False))}
        for a, b, flow, cost in links[["from", "to", "volume", "cost"]].itertuples(index=False, name=None):
            i, base_time = pairs[tuple(sorted((int(a), int(b))))]
            flows[i] += flow
            excess = max(float(cost) / base_time - 1, 0)
            congestion[i] = max(congestion[i], excess / (1 + excess))
        diagnostic = dict(fixed_point_iterations=iterations, fixed_point_residual=residual,
                          ue_relative_gap=gap, solve_seconds=time.perf_counter() - started, **extra_diagnostic)
        self.solve_seconds += diagnostic["solve_seconds"]
        state = TrafficState(key, d, q.copy(), forecast.copy(), costs.copy(), np.isfinite(raw),
                             flows, congestion, links, diagnostic, solver_state)
        if unmet_before is not None:
            state.gamma_od = np.asarray(gamma).copy()
            state.cumulative_unmet = self.demand_model.advance_unmet(
                state.served_demand, unmet_before)
            diagnostic.update(gamma_min=float(state.gamma_od.min()),
                              gamma_max=float(state.gamma_od.max()),
                              cumulative_unmet_mean=float(state.cumulative_unmet.mean()))
        return self.cache.put(key, state)

    def close(self):
        self.cache.close()
