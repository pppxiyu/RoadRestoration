"""Daily restoration problem and reproducible scenario data.

Only the environment consumes RoadDamage truth. A policy receives VisibleRoad
records from Scenario.visible(), never this module's complete scenario objects.
The legacy fixed-road scenario sampler remains in scenarios.py.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import norm, qmc

from src import config as P
from src.problem.io import load_toy_network, od_to_matrix
from src.environment.ue import solve_ue


ROOT = Path(__file__).resolve().parents[2]
TOY = ROOT / "data" / "siouxfalls_toy"
QUOTAS = {6: (1, 4, 0, 1), 11: (2, 6, 0, 3),
          17: (2, 11, 1, 3), 23: (4, 13, 1, 5)}
ESTIMATE_PROBABILITIES = (0.25, 0.5, 0.25)
CONFUSION = {1: (0.6, 1 / 3, 1 / 15), 2: (0.2, 0.6, 0.2),
             3: (1 / 15, 1 / 3, 0.6)}


@dataclass(frozen=True)
class DailyRules:
    version: str = "daily_discovery_v3"
    estimated_duration_rule: str = "rounded_expectation_of_truncated_integer_cell"
    slot_hours: float = 24.0
    crews: int = 1
    recovery_days: int = 35
    discovery_last_day: int = 14
    time_weight: float = 0.5
    flow_weight: float = 0.5
    restored_flow_fraction: float = 0.99
    restored_time_ratio: float = 1.01
    consecutive_restored_days: int = 2
    ue_definition_gap: float = 1e-6
    ue_definition_max_iter: int = 2000
    ue_gap: float = 1e-3
    ue_max_iter: int = 2000
    fixed_point_tolerance: float = 0.002
    response_95_days: float = 7.0
    response_ue_gap: float = 1e-4
    response_ue_max_iter: int = 1000
    restored_fixed_point_tolerance: float = 1e-5
    restored_ue_gap: float = 1e-6
    restored_ue_max_iter: int = 2000
    response_max_iter: int = 100
    response_relaxation: float = 0.5
    response_solver: str = "joint"


@dataclass(frozen=True)
class VisibleRoad:
    edge_id: int
    estimated_severity: int
    estimated_duration: int


@dataclass(frozen=True)
class RoadDamage:
    edge_id: int
    estimated_severity: int
    estimated_duration: int
    true_severity: int
    true_duration: int
    discovery_day: int


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    roads: tuple[RoadDamage, ...]

    def visible(self, day: int) -> tuple[VisibleRoad, ...]:
        return tuple(VisibleRoad(r.edge_id, r.estimated_severity, r.estimated_duration)
                     for r in self.roads if r.discovery_day <= day)

    def record(self):
        return {"scenario_id": self.scenario_id,
                "roads": [asdict(r) for r in self.roads]}

    @property
    def signature(self):
        return hashlib.sha256(json.dumps([asdict(r) for r in self.roads],
                                        sort_keys=True).encode()).hexdigest()


def _duration_parameters(road_class: str, severity: int):
    """Parameters shared by the true-duration draw and its cell expectation."""
    cell = (road_class, int(severity))
    mean, sd = float(P.DUR_MEAN[cell]), float(P.DUR_SD[cell])
    variance = np.log1p((sd / mean) ** 2)
    sigma = np.sqrt(variance)
    mu = np.log(mean) - variance / 2
    upper = float(P.DUR_TRUNC_MULT) * mean
    cdf_upper = norm.cdf((np.log(upper) - mu) / sigma)
    return mu, sigma, upper, cdf_upper


def duration_draw(road_class: str, severity: int, probability: float) -> int:
    """Inverse CDF of the truncated lognormal, then nearest integer, minimum 1."""
    mu, sigma, _, cdf_upper = _duration_parameters(road_class, severity)
    p = np.clip(float(probability) * cdf_upper, 1e-12, 1 - 1e-12)
    return max(1, int(np.floor(np.exp(mu + sigma * norm.ppf(p)) + 0.5)))


def duration_expectation(road_class: str, severity: int) -> float:
    """Exact mean AFTER truncation, half-up integer rounding and the one-day floor.

    Enumerate integer-day probabilities from lognormal CDF differences. This is
    the specified severity cell, not a mixture over the severity-confusion law.
    No simulation, random seed or realized repair duration enters the estimate.
    """
    mu, sigma, upper, cdf_upper = _duration_parameters(road_class, severity)
    days = np.arange(1, max(1, int(np.floor(upper + 0.5))) + 1)
    cutoffs = np.minimum(days + 0.5, upper)
    cumulative = norm.cdf((np.log(cutoffs) - mu) / sigma) / cdf_upper
    probabilities = np.diff(np.r_[0.0, cumulative])
    return float(days @ probabilities)


def estimated_duration(road_class: str, estimated_severity: int) -> int:
    """Public point estimate: round the actual cell mean half up, minimum one day."""
    return max(1, int(np.floor(duration_expectation(road_class, estimated_severity) + 0.5)))


def build_problem(n: int = 11, seed: int = 42, toy_dir=TOY):
    """The public road subset is sampled once per scale, before any split."""
    if int(n) not in QUOTAS:
        raise ValueError(f"daily scales must be one of {tuple(QUOTAS)}")
    rules = DailyRules()
    edges, od, zones = load_toy_network(toy_dir)
    baseline, convergence = solve_ue(
        edges, od_to_matrix(od, zones), zones, rgap=rules.ue_definition_gap,
        max_iter=rules.ue_definition_max_iter, quiet=True)
    if convergence.rgap > rules.ue_definition_gap:
        raise RuntimeError("normal-period traffic definition did not converge")
    by_pair = {}
    for row in baseline.itertuples(index=False):
        key = tuple(sorted((int(row[0]), int(row[1]))))
        by_pair[key] = by_pair.get(key, 0.0) + float(row.volume)
    flows = {int(r.edge_id): by_pair[tuple(sorted((int(r.u), int(r.v))))]
             for r in edges.itertuples(index=False)}
    ranked = sorted(flows, key=lambda e: (-flows[e], e))
    bins = (tuple(sorted(ranked[:8])), tuple(sorted(ranked[8:29])),
            tuple(sorted(ranked[29:])))
    rng = np.random.default_rng(np.random.SeedSequence([seed, n, 100]))
    ph, pm, hh, hm = QUOTAS[int(n)]
    public = tuple(sorted([*rng.choice(bins[0], ph, replace=False),
                           *rng.choice(bins[1], pm, replace=False)]))
    public = tuple(int(e) for e in public)
    return {"n": int(n), "seed": int(seed), "rules": rules,
            "toy_dir": str(toy_dir), "edges": edges, "od": od, "zone_ids": zones,
            "baseline_links": baseline, "baseline_gap": float(convergence.rgap),
            "baseline_flow": flows, "ranked": ranked, "bins": bins,
            "public_roads": public, "hidden_quotas": (hh, hm)}


def sample_split(problem, count: int, seed: int, name: str) -> list[Scenario]:
    """Independent hidden identities; independent LHS columns for every random label.

    Keep the historical five-column LHS allocation per road: severity estimate,
    true severity, UNUSED former duration-estimate column, true duration, discovery.
    Public duration is now a rounded cell expectation. Retaining the unused column
    keeps all other random labels identical for the same seed and sample count.
    No realized duration determines a public estimate, normalization or horizon.
    """
    ids = sorted(problem["baseline_flow"])
    columns = {e: 5 * i for i, e in enumerate(ids)}
    uniform = qmc.LatinHypercube(d=5 * len(ids), seed=seed).random(int(count))
    rng = np.random.default_rng(np.random.SeedSequence([seed, 701]))
    classes = {int(r.edge_id): r.road_class
               for r in problem["edges"].itertuples(index=False)}
    duration_estimates = {(c, s): estimated_duration(c, s)
                          for c in set(classes.values()) for s in (1, 2, 3)}
    estimate_cumulative = np.cumsum(ESTIMATE_PROBABILITIES)
    public = set(problem["public_roads"])
    candidates = [sorted(set(b) - public) for b in problem["bins"][:2]]
    output = []
    for i in range(int(count)):
        hidden = set()
        for pool, size in zip(candidates, problem["hidden_quotas"]):
            hidden.update(int(e) for e in rng.choice(pool, size, replace=False))
        roads = []
        for edge in sorted(public | hidden):
            u = uniform[i, columns[edge]:columns[edge] + 5]
            estimate = int(np.searchsorted(estimate_cumulative, u[0]) + 1)
            truth = int(np.searchsorted(np.cumsum(CONFUSION[estimate]), u[1]) + 1)
            roads.append(RoadDamage(
                edge, estimate, duration_estimates[classes[edge], estimate],
                truth, duration_draw(classes[edge], truth, u[3]),
                0 if edge in public else int(u[4] * problem["rules"].discovery_last_day) + 1))
        output.append(Scenario(f"{name}_{i:03d}", tuple(roads)))
    return output


def make_splits(problem, pool_n=64, n_val=11, n_test=50):
    seed = problem["seed"]
    splits = {"train": sample_split(problem, pool_n, seed * 1000 + 21, "train"),
              "validation": sample_split(problem, n_val, seed * 1000 + 15, "validation"),
              "test": sample_split(problem, n_test, seed * 1000 + 7, "test")}
    signatures = [s.signature for values in splits.values() for s in values]
    if len(set(signatures)) != len(signatures):
        raise AssertionError("duplicate scenarios across experiment splits")
    return splits


def public_spec(problem):
    return {"n_disrupted": problem["n"], "scenario_seed": problem["seed"],
            "rules": asdict(problem["rules"]), "public_roads": problem["public_roads"],
            "flow_bins": problem["bins"], "hidden_quotas": problem["hidden_quotas"],
            "severity_estimate_probabilities": list(ESTIMATE_PROBABILITIES),
            "severity_truth_given_estimate": CONFUSION,
            "damage_physics": {"capacity_retained": P.CAP_RETAIN,
                               "free_flow_speed_retained": P.SPEED_RETAIN,
                               "removed_at_severity": P.SEVER_SEVERITY},
            "duration_mean": {str(k): v for k, v in P.DUR_MEAN.items()},
            "duration_sd": {str(k): v for k, v in P.DUR_SD.items()},
            "duration_truncation_multiple": P.DUR_TRUNC_MULT,
            "estimated_duration_days": {str(k): estimated_duration(*k) for k in P.DUR_MEAN},
            "estimated_duration_cell_expectation_days": {
                str(k): duration_expectation(*k) for k in P.DUR_MEAN},
            "sampling": "independent identity draws and independent LHS random-label columns; "
                        "estimated duration is deterministic; historical estimate column reserved"}
