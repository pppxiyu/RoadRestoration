"""Optional policy observation of past OD shortfalls, without changing physics.

History is reconstructed from observed actual trips on every calendar day,
including days when the crew is busy. No damage truth or future value is read.
The existing daily simulator and its cache/scoring identity remain unchanged.
"""
from dataclasses import dataclass

import numpy as np

from src.environment.daily import DailyEpisode, Observation


@dataclass(frozen=True)
class HistoryObservation(Observation):
    cumulative_unmet: np.ndarray | None = None


class HistoryObservedEpisode(DailyEpisode):
    def _solve_today(self):
        normal = self.simulator.ctx["H0"]
        if self.traffic is None:
            self.observed_unmet = np.zeros_like(normal, dtype=float)
        else:
            self.observed_unmet += np.maximum(0.0, 1.0 - self.traffic.served_demand / normal)
        super()._solve_today()

    def observation(self):
        observed = super().observation()
        if observed is None:
            return None
        return HistoryObservation(**vars(observed), cumulative_unmet=self.observed_unmet.copy())


@dataclass(frozen=True)
class DemandMemoryObservation(HistoryObservation):
    previous_observed_deviation: np.ndarray | None = None
    previous_reachable: np.ndarray | None = None


class DemandMemoryObservedEpisode(HistoryObservedEpisode):
    """Record yesterday's public trips, reference demand and reachability.

    This never exposes unserved latent demand. Day one uses the environment's
    known zero-history initialization, marked valid; it is not a simulated day 0.
    """
    def _solve_today(self):
        if self.traffic is None:
            self.previous_observed_deviation = np.zeros_like(self.simulator.ctx["H0"])
            self.previous_reachable = np.ones_like(self.simulator.ctx["H0"], dtype=bool)
        else:
            self.previous_observed_deviation = (
                self.traffic.served_demand - self.traffic.external).copy()
            self.previous_reachable = self.traffic.reachable.copy()
        super()._solve_today()

    def observation(self):
        observed = super().observation()
        if observed is None:
            return None
        return DemandMemoryObservation(**vars(observed),
            previous_observed_deviation=self.previous_observed_deviation.copy(),
            previous_reachable=self.previous_reachable.copy())
