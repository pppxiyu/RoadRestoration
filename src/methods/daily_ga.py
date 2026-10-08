"""The retained GA with a single public priority policy and resumable fitness.

The original genetic operators, population, seed RNG and stopping logic are
imported unchanged. Persistent scores are revealed to that algorithm only when
it requests them, so replay after interruption reconstructs the same search.
"""
from __future__ import annotations

import random
import numpy as np

from src.methods.metaheuristic import _ga, GA_PARAMS, BUDGET_CAP


class PriorityPolicy:
    def __init__(self, order):
        self.order = tuple(int(e) for e in order)
        if len(set(self.order)) != len(self.order):
            raise ValueError("priority order contains duplicate roads")
        self.rank = {e: i for i, e in enumerate(self.order)}

    def choose(self, candidates):
        """Only currently legal road identities enter the fixed-order policy."""
        if not candidates or not set(candidates) <= self.rank.keys():
            raise ValueError("empty or unknown candidate set")
        return min(candidates, key=self.rank.__getitem__)


def possible_roads(problem):
    roads = set(problem["public_roads"])
    for flow_bin, count in zip(problem["bins"], problem["hidden_quotas"]):
        if count:
            roads.update(flow_bin)
    return tuple(sorted(roads))


def initial_orders(problem, context, training_scenarios):
    """Retain flow / estimated exposure / exposure-per-duration initial seeds.

    For this scenario-varying problem, use conditional means of the publicly
    estimated labels in the fixed training sample. True labels and test data
    never enter initialization. These are only initial candidates, not a
    substitute nominal environment for fitness evaluation.
    """
    roads = possible_roads(problem)
    estimates = {e: [] for e in roads}
    durations = {e: [] for e in roads}
    for scenario in training_scenarios:
        for road in scenario.roads:
            estimates[road.edge_id].append(road.estimated_severity)
            durations[road.edge_id].append(road.estimated_duration)
    # A road absent from the finite training sample uses its public prior.
    from src.problem.daily import estimated_duration
    classes = problem["edges"].set_index("edge_id").road_class.to_dict()
    exposure = context["incidence"].T @ context["H0"] / 3.0
    demand, ratio = {}, {}
    for road in roads:
        severity = float(np.mean(estimates[road])) if estimates[road] else 2.0
        duration = (float(np.mean(durations[road])) if durations[road] else
                    sum(p * estimated_duration(classes[road], s)
                        for s, p in zip((1, 2, 3), (0.25, 0.5, 0.25))))
        demand[road] = severity * exposure[context["edge_row"][road]]
        ratio[road] = demand[road] / max(1.0, duration)
    scores = (problem["baseline_flow"], demand, ratio)
    return [tuple(sorted(roads, key=lambda e: (-score[e], e))) for score in scores]


class ReplayFitness:
    """Active cache is rebuilt in original query order, not prefilled on resume."""
    def __init__(self, score_batch, budget=BUDGET_CAP):
        self.score_batch, self.budget = score_batch, int(budget)
        self.cache = {}

    @property
    def n_evals(self):
        return len(self.cache)

    def evaluate(self, permutations):
        todo = []
        for order in permutations:
            if order not in self.cache and order not in todo and self.n_evals + len(todo) < self.budget:
                todo.append(order)
        if todo:
            scores = self.score_batch(todo)
            for order in todo:
                score = tuple(float(v) for v in scores[order])
                if len(score) != 3 or not np.isfinite(score).all():
                    raise ValueError("GA requires three finite aggregate scores")
                self.cache[order] = score
        return {p: self.cache[p][0] if p in self.cache else float("inf") for p in permutations}

    def best(self):
        order = min(self.cache, key=lambda p: self.cache[p][0])
        return order, self.cache[order]


def search(roads, seeds, score_batch, seed=42, on_generation=None):
    fitness = ReplayFitness(score_batch)
    stop, trace = _ga(fitness, roads, seeds, random.Random(int(seed) * 1000),
                      on_gen=on_generation, **GA_PARAMS)
    return fitness.best(), stop, trace
