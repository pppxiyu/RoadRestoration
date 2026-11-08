"""Project current public information without receiving the hidden scenario."""
import numpy as np
from src.environment.daily_contracts import LinkObservation, Observation


def build_observation(day, visible, completed, candidates, t, edges=None):
    """Visible reports are already filtered by the repair/report component."""
    links = {}
    if t.links is not None:
        links = {(int(a), int(b)): LinkObservation(int(a), int(b), float(flow), float(cost))
                 for a, b, flow, cost in t.links[["from", "to", "volume", "cost"]].itertuples(index=False, name=None)}
    # Enumerate the public intact network, not the hidden damaged-road list.
    # This supplies exact directional times in addition to the unchanged RL
    # flow/congestion summaries. No network input dimension is changed here.
    if edges is not None:
        for edge in edges.itertuples(index=False):
            for a, b in ((int(edge.u), int(edge.v)), (int(edge.v), int(edge.u))):
                links.setdefault((a, b), LinkObservation(a, b, 0.0, float("inf")))
    # Only arrays permitted by the information rule cross this boundary.
    return Observation(day, visible,
                       tuple(sorted(completed.items())), candidates,
                       t.served_demand.copy(), t.external.copy(),
                       np.where(t.reachable, t.costs, np.inf), t.reachable.copy(),
                       t.road_flow.copy(), t.road_congestion.copy(),
                       tuple(links[key] for key in sorted(links)))
