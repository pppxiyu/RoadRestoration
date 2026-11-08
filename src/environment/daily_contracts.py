"""Data exchanged by daily environment components; no orchestration dependencies."""
from dataclasses import dataclass
import numpy as np
import pandas as pd


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
    cumulative_unmet: np.ndarray | None = None
    gamma_od: np.ndarray | None = None

    @property
    def served_demand(self):
        return np.where(self.reachable, self.demand, 0.0)


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
