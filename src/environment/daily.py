"""Daily traffic observations and seven-day human-response fixed points.

The experiment supplies the behavior option. Historical damage_shortfall and
the independent human-behavior study remain available in their original files.
Delayed-response cache keys include the previous state key: a damaged-road set
alone is insufficient because demand remembers the entire previous trajectory.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass
import hashlib
import io
import json
import math
from pathlib import Path
import sqlite3
import time

import networkx as nx
import numpy as np
import pandas as pd

from src import config as P
from src.environment.evaluate import build_damaged_edges, od_travel_times, _matrix_from_H
from src.environment.gravity import DEFAULT_MODEL, load_gravity_model
from src.environment.ue import solve_ue, warm_start_seed
from src.environment.shifted_demand import solve_shifted


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


def recovery_fraction(day, rules):
    """Approved rescaling of the old logistic curve, exactly normal from day 35."""
    t = np.asarray(day, dtype=float)
    if np.any(t < 0):
        raise ValueError("day must be nonnegative")
    initial, plateau = P.RECOVERY_INITIAL_LEVEL, P.RECOVERY_PLATEAU_LEVEL
    ratio = plateau / initial - 1.0
    rate = math.log(ratio / (1 / 0.99 - 1)) / rules.recovery_days
    old = lambda d: plateau / (1 + ratio * np.exp(-rate * d))
    result = initial + (1 - initial) * (old(t) - old(0)) / (old(rules.recovery_days) - old(0))
    return np.where(t >= rules.recovery_days, 1.0, result)


def context_from_problem(problem):
    edges, od, zones = problem["edges"], problem["od"], problem["zone_ids"]
    positions = {int(z): i for i, z in enumerate(zones)}
    pairs = [(int(r.origin), int(r.destination)) for r in od.itertuples(index=False)]
    context = dict(edges=edges, zone_ids=zones, od_pairs=pairs,
                   H0=od.h0.to_numpy(dtype=float), nz=len(zones),
                   oi=np.array([positions[o] for o, _ in pairs]),
                   di=np.array([positions[d] for _, d in pairs]),
                   origins_unique=sorted({o for o, _ in pairs}),
                   edge_row={int(r.edge_id): i for i, r in enumerate(edges.itertuples(index=False))})
    context["baseline_u"] = od_travel_times(problem["baseline_links"], context)
    if not np.isfinite(context["baseline_u"]).all():
        raise ValueError("normal-period network must connect all positive-demand OD pairs")
    context["u_pen"] = P.UPEN_FACTOR * float(context["baseline_u"].max())
    # Public structural incidence, based only on the intact free-flow network.
    graph = nx.Graph()
    for r in edges.itertuples(index=False):
        graph.add_edge(int(r.u), int(r.v), weight=float(r.free_flow_time), edge=int(r.edge_id))
    incidence = np.zeros((len(pairs), len(edges)))
    for i, (o, d) in enumerate(pairs):
        path = nx.shortest_path(graph, o, d, weight="weight")
        for a, b in zip(path, path[1:]):
            incidence[i, context["edge_row"][graph[a][b]["edge"]]] = 1.0
    context["incidence"] = incidence
    return context


@dataclass
class TrafficState:
    key: str
    day: int
    demand: np.ndarray
    external: np.ndarray
    costs: np.ndarray
    reachable: np.ndarray
    road_flow: np.ndarray
    road_congestion: np.ndarray
    links: pd.DataFrame
    diagnostics: dict
    solver_state: dict | None = None

    @property
    def served_demand(self):
        return np.where(self.reachable, self.demand, 0.0)


class DailyTraffic:
    def __init__(self, problem, cache_dir=None):
        self.problem = problem
        self.rules = problem["rules"]
        self.ctx = context_from_problem(problem)
        model = load_gravity_model(DEFAULT_MODEL)
        self.gamma = -float(model["coefficients"]["travel_cost_minutes"])
        if self.gamma <= 0:
            raise ValueError("human-response cost sensitivity must be positive")
        self.weight = 1 - math.exp(math.log(0.05) / self.rules.response_95_days)
        sources = {str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest()
                   for p in [Path(__file__), Path(__file__).with_name("ue.py"),
                             Path(__file__).with_name("evaluate.py"),
                             Path(__file__).with_name("shifted_demand.py"),
                             Path(__file__).with_name("fixed_point_demand.py"),
                             Path(__file__).with_name("environment_behavior.py"), DEFAULT_MODEL]}
        identity = dict(rules=asdict(self.rules), gamma=self.gamma,
                        initial=P.RECOVERY_INITIAL_LEVEL, plateau=P.RECOVERY_PLATEAU_LEVEL,
                        cap=P.CAP_RETAIN, speed=P.SPEED_RETAIN, sever=P.SEVER_SEVERITY,
                        penalty=P.UPEN_FACTOR, warm_start=P.UE_WARM_START, sources=sources,
                        network=hashlib.sha256(problem["edges"].to_csv(index=False).encode()).hexdigest(),
                        od=hashlib.sha256(problem["od"].to_csv(index=False).encode()).hexdigest())
        self.fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        self.memo = OrderedDict()
        self.references = {}
        self.solve_calls = self.cache_hits = self.fixed_point_iterations = 0
        self.solve_seconds = 0.0
        self.connection = None
        if cache_dir is not None:
            folder = Path(cache_dir)
            folder.mkdir(parents=True, exist_ok=True)
            self.connection = sqlite3.connect(folder / (self.fingerprint[:16] + ".sqlite3"), timeout=120)
            self.connection.execute("PRAGMA journal_mode=WAL")
            self.connection.execute("CREATE TABLE IF NOT EXISTS traffic (key TEXT PRIMARY KEY, payload BLOB)")

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
            q = self.ctx["H0"] * float(recovery_fraction(d, self.rules))
            if d == self.rules.recovery_days:
                self.references[d] = (q, self.problem["baseline_links"], self.ctx["baseline_u"])
            else:
                links, _, costs, _ = self._links(self.ctx["edges"], q)
                self.references[d] = (q, links, costs)
        return self.references[d]

    def _remember(self, key, state):
        self.memo[key] = state
        self.memo.move_to_end(key)
        while len(self.memo) > 2048:
            self.memo.popitem(last=False)
        return state

    def _pack(self, state):
        stream = io.BytesIO()
        saved_solver = {}
        if state.solver_state is not None:
            warm = state.solver_state
            saved_solver = dict(path_lengths=np.array([len(p) for p in warm["path_arcs"]], dtype=int),
                                path_arcs=np.array([a for p in warm["path_arcs"] for a in p], dtype=int),
                                path_od=warm["path_od"], path_flow=warm["path_flow"],
                                solver_damage=np.asarray(warm["damage"], dtype=int).reshape(-1, 2))
        np.savez_compressed(stream, demand=state.demand, external=state.external,
                            costs=state.costs, reachable=state.reachable,
                            road_flow=state.road_flow, road_congestion=state.road_congestion,
                            links=state.links[["from", "to", "volume", "cost"]].to_numpy(),
                            diagnostics=json.dumps(state.diagnostics), **saved_solver)
        return stream.getvalue()

    def _unpack(self, key, day, payload):
        with np.load(io.BytesIO(payload), allow_pickle=False) as data:
            links = pd.DataFrame(data["links"], columns=["from", "to", "volume", "cost"])
            links[["from", "to"]] = links[["from", "to"]].astype(int)
            solver_state = None
            if "path_lengths" in data:
                ends = np.r_[0, np.cumsum(data["path_lengths"])]
                solver_state = dict(path_arcs=[tuple(data["path_arcs"][a:b].tolist())
                                              for a, b in zip(ends, ends[1:])],
                                    path_od=data["path_od"], path_flow=data["path_flow"],
                                    costs=data["costs"], reachable=data["reachable"],
                                    damage=data["solver_damage"].tolist())
            return TrafficState(key, day, data["demand"], data["external"], data["costs"],
                                data["reachable"], data["road_flow"], data["road_congestion"],
                                links, json.loads(str(data["diagnostics"])), solver_state)

    def _nested_response(self, day, edges, forecast, reference_costs, history, warm, *, restored=False):
        """Retained numerical fallback and independent equivalence check.

        This uses the SAME demand equation and tolerances as the joint solver;
        failure is never accepted as a scored trajectory.
        """
        q = forecast + history
        tolerance = (self.rules.restored_fixed_point_tolerance if restored
                     else self.rules.fixed_point_tolerance)
        for iterations in range(1, self.rules.response_max_iter + 1):
            links, raw, costs, convergence = self._links(edges, q, warm, response=True, restored=restored)
            warm = (links, np.where(np.isfinite(raw), q, 0.0))
            target = forecast + history + self.weight * forecast * np.expm1(
                -self.gamma * (costs - reference_costs))
            if float(target.min()) <= 0:
                raise RuntimeError(f"nonpositive adjusted demand on day {day}")
            residual = float(np.max(np.abs(target-q) / np.maximum(forecast, 1.0)))
            if residual <= tolerance:
                return q, links, raw, costs, iterations, residual, float(convergence.rgap)
            w = self.rules.response_relaxation
            q = w*q + (1-w)*target
        raise RuntimeError(f"seven-day fixed point did not converge on day {day}: {residual:g}")

    def solve(self, day, damaged, behavior, previous=None):
        if behavior not in ("natural", "response7d"):
            raise ValueError(f"unsupported daily behavior {behavior}")
        d = int(day)
        if behavior == "response7d" and previous is not None and previous.day != d - 1:
            raise ValueError("seven-day response requires the immediately previous day")
        identity = (behavior, d if behavior == "response7d" else min(d, self.rules.recovery_days),
                    tuple(sorted(damaged.items())), previous.key if previous and behavior == "response7d" else None)
        key = hashlib.sha256(json.dumps(identity).encode()).hexdigest()
        if key in self.memo:
            self.cache_hits += 1
            cached = self.memo[key]
            # Natural-recovery traffic is stationary after day 35; keep the caller's date.
            return TrafficState(key, d, cached.demand, cached.external, cached.costs, cached.reachable,
                                cached.road_flow, cached.road_congestion, cached.links, cached.diagnostics,
                                cached.solver_state)
        if self.connection is not None:
            row = self.connection.execute("SELECT payload FROM traffic WHERE key=?", (key,)).fetchone()
            if row:
                self.cache_hits += 1
                return self._remember(key, self._unpack(key, d, row[0]))
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
            previous_deviation = (np.zeros_like(forecast) if previous is None
                                  else previous.demand - previous.external)
            history = (1 - self.weight) * previous_deviation
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
                        self.ctx, edges, forecast, reference_costs, self.gamma,
                        (1-self.weight)*forecast+history, self.weight*forecast,
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
                    d, edges, forecast, reference_costs, history, warm, restored=restored)
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
        if self.connection is not None:
            with self.connection:
                self.connection.execute("INSERT OR IGNORE INTO traffic VALUES (?, ?)",
                                        (key, self._pack(state)))
        return self._remember(key, state)

    def close(self):
        if self.connection is not None:
            self.connection.close()


def daily_losses(traffic, context, *, disconnected_flow):
    """Rules (8)-(9): actual trips only, per-OD shortfalls, no horizon division."""
    if disconnected_flow != "unserved":
        raise ValueError("the finalized problem requires disconnected_flow='unserved'")
    positive = context["H0"] > 0
    if not positive.any():
        raise ValueError("the problem requires positive normal-period OD demand")
    q0 = context["H0"][positive]
    q = traffic.served_demand[positive]
    reference = float(q @ context["baseline_u"][positive])
    # A disconnected OD can have infinite observed time. Zero actual trips must
    # contribute zero, never 0 * infinity or an artificial penalty time.
    travelling = q > 0
    actual_time = float(q[travelling] @ traffic.costs[positive][travelling])
    time_ratio = actual_time / reference if reference > 0 else 1.0
    flow_loss = float(np.maximum(q0 - q, 0).sum() / q0.sum())
    return dict(time_loss=max(time_ratio - 1.0, 0.0), flow_loss=flow_loss,
                time_ratio=time_ratio, min_flow_fraction=float(np.min(q / q0)),
                total_flow_fraction=float(q.sum() / q0.sum()))


@dataclass(frozen=True)
class LinkObservation:
    """Current directional traffic only; closed links have zero flow/infinite time."""
    origin: int
    destination: int
    flow: float
    travel_time: float


@dataclass(frozen=True)
class Observation:
    """Policy-facing data: no scenario truth, discovery schedule, or future dates."""
    day: int
    visible: tuple
    completed: tuple[tuple[int, int], ...]
    candidates: tuple[int, ...]
    demand: np.ndarray
    external: np.ndarray
    costs: np.ndarray
    reachable: np.ndarray
    road_flow: np.ndarray
    road_congestion: np.ndarray
    link_traffic: tuple[LinkObservation, ...] = ()


class DailyEpisode:
    """One nonpreemptive crew; actions only when a discovered road can be started.

    At the beginning of day t, completions/discoveries take effect, then the
    traffic state is computed once. Starting work does not itself change road
    capacity. Its traffic cost belongs to that day's action interval. The last
    action also receives all costs of the subsequent demand recovery.
    """
    def __init__(self, simulator, scenario, behavior, *, disconnected_flow):
        if disconnected_flow != "unserved":
            raise ValueError("the finalized problem requires disconnected_flow='unserved'")
        self.simulator, self.scenario, self.behavior = simulator, scenario, behavior
        self.disconnected_flow = disconnected_flow
        self._truth = {r.edge_id: r for r in scenario.roads}
        self.day = 1
        self.completed, self.starts = {}, {}
        self.order, self.records = [], []
        self.traffic = None
        self.total_loss = self.initial_loss = 0.0
        self.restored_streak = 0
        self.finished = False
        self._solve_today()
        self.initial_loss = self._advance_without_action()

    def _solve_today(self):
        damaged = {e: r.true_severity for e, r in self._truth.items() if e not in self.completed}
        self.traffic = self.simulator.solve(self.day, damaged, self.behavior, self.traffic)

    def _candidates(self):
        return tuple(sorted(r.edge_id for r in self.scenario.visible(self.day)
                            if r.edge_id not in self.starts))

    def observation(self):
        if self.finished:
            return None
        t = self.traffic
        links = {}
        if t.links is not None:
            links = {(int(a), int(b)): LinkObservation(int(a), int(b), float(flow), float(cost))
                     for a, b, flow, cost in t.links[["from", "to", "volume", "cost"]].itertuples(index=False, name=None)}
        # Enumerate the public intact network, not the hidden damaged-road list.
        # This supplies exact directional times in addition to the unchanged RL
        # flow/congestion summaries. No network input dimension is changed here.
        if "edges" in self.simulator.ctx:
            for edge in self.simulator.ctx["edges"].itertuples(index=False):
                for a, b in ((int(edge.u), int(edge.v)), (int(edge.v), int(edge.u))):
                    links.setdefault((a, b), LinkObservation(a, b, 0.0, float("inf")))
        # Only arrays permitted by the information rule cross this boundary.
        return Observation(self.day, self.scenario.visible(self.day),
                           tuple(sorted(self.completed.items())), self._candidates(),
                           t.served_demand.copy(), t.external.copy(),
                           np.where(t.reachable, t.costs, np.inf), t.reachable.copy(),
                           t.road_flow.copy(), t.road_congestion.copy(),
                           tuple(links[key] for key in sorted(links)))

    def _charge_today(self):
        value = daily_losses(self.traffic, self.simulator.ctx,
                             disconnected_flow=self.disconnected_flow)
        rules = self.simulator.rules
        cost = rules.time_weight * value["time_loss"] + rules.flow_weight * value["flow_loss"]
        self.total_loss += cost
        restored = (len(self.completed) == len(self._truth)
                    and value["min_flow_fraction"] >= rules.restored_flow_fraction
                    and value["time_ratio"] <= rules.restored_time_ratio)
        self.restored_streak = self.restored_streak + 1 if restored else 0
        self.records.append(dict(day=self.day, cost=cost, **value,
                                 repaired=len(self.completed), discovered=len(self.scenario.visible(self.day)),
                                 **self.traffic.diagnostics))
        self.finished = self.restored_streak >= rules.consecutive_restored_days
        return cost

    def _advance_without_action(self):
        cost = 0.0
        while not self.finished and not self._candidates():
            cost += self._charge_today()
            if not self.finished:
                self.day += 1
                self._solve_today()
        return cost

    def step(self, road):
        if self.finished or road not in self._candidates():
            raise ValueError("must select one currently discovered, unrepaired road; waiting is not an action")
        self.starts[road] = self.day
        self.order.append(road)
        completion_day = self.day + self._truth[road].true_duration
        cost = 0.0
        while self.day < completion_day:
            cost += self._charge_today()
            self.day += 1
            if self.day == completion_day:
                self.completed[road] = self.day
            self._solve_today()
        cost += self._advance_without_action()
        return self.observation(), -cost, self.finished

    def result(self):
        if not self.finished:
            raise RuntimeError("an unfinished episode is not an evaluation result")
        return dict(scenario=self.scenario.scenario_id, objective=self.total_loss,
                    time_loss=sum(r["time_loss"] for r in self.records),
                    flow_loss=sum(r["flow_loss"] for r in self.records),
                    recovery_day=self.day, repair_completion_day=max(self.completed.values()),
                    order=list(self.order), starts=dict(self.starts), completions=dict(self.completed),
                    daily=self.records, initial_loss=self.initial_loss)
