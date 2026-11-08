"""Demand equations and memory updates, independent of repair and traffic solvers."""
from dataclasses import dataclass
import math
import numpy as np
from src import config as P
from src.environment.poi_response import destination_weights, sensitivity


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


@dataclass(frozen=True)
class DemandMemory:
    gamma: float | np.ndarray
    unmet_before: np.ndarray | None


@dataclass(frozen=True)
class DemandEquation:
    forecast: np.ndarray
    reference_costs: np.ndarray
    history: np.ndarray
    weight: float
    gamma: float | np.ndarray

    @property
    def base(self):
        return (1 - self.weight) * self.forecast + self.history

    @property
    def scale(self):
        return self.weight * self.forecast

    def target(self, costs):
        return self.forecast + self.history + self.weight * self.forecast * np.expm1(
            -self.gamma * (costs - self.reference_costs))


class DemandResponse:
    """Immutable configuration; each caller supplies its own previous-day state."""
    def __init__(self, normal_demand, od_pairs, rules, gamma, specification):
        self.normal = normal_demand
        self.rules = rules
        self.gamma = gamma
        self.specification = specification
        self.weights = (None if specification is None else
                        destination_weights(specification, od_pairs))
        self.weight = 1 - math.exp(math.log(0.05) / rules.response_95_days)

    def external(self, day):
        return self.normal * float(recovery_fraction(day, self.rules))

    def prepare(self, day, behavior, previous):
        if behavior not in ("natural", "response7d"):
            raise ValueError(f"unsupported daily behavior {behavior}")
        if behavior == "response7d" and previous is not None and previous.day != day - 1:
            raise ValueError("seven-day response requires the immediately previous day")
        unmet, gamma = None, self.gamma
        if self.specification is not None and behavior == "response7d":
            if previous is not None and previous.cumulative_unmet is None:
                raise ValueError("POI response requires the previous day's unmet-demand history")
            unmet = np.zeros_like(self.normal) if previous is None else previous.cumulative_unmet
            gamma = sensitivity(self.gamma, unmet, self.weights, self.specification)
        return DemandMemory(gamma, unmet)

    def equation(self, forecast, reference_costs, previous, gamma):
        deviation = (np.zeros_like(forecast) if previous is None
                     else previous.demand - previous.external)
        return DemandEquation(forecast, reference_costs, (1-self.weight)*deviation, self.weight, gamma)

    def advance_unmet(self, actual_trips, unmet_before):
        return unmet_before + np.maximum(0.0, 1.0 - actual_trips / self.normal)
