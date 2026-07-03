"""
Sample restoration-duration scenarios (Table 1 base x crew-efficiency eta).

Durations are drawn per LEVEL (road_class, severity); all disrupted segments of the same
level share one duration within a scenario (paper: d_e(ω) = d_{ℓ(e)}(ω)). Each scenario is
a dict {edge_id: duration_in_slots}.
"""

import numpy as np

from util import params as P


def sample_scenarios(disrupted, M=P.M_SCENARIOS, seed=P.SEED):
    """`disrupted`: DataFrame with columns edge_id, road_class, severity.
    Returns a list of M dicts: scenario[m][edge_id] = realized duration (slots)."""
    rng = np.random.default_rng(seed)
    rows = list(disrupted.itertuples(index=False))
    levels = sorted({(r.road_class, int(r.severity)) for r in rows})
    scenarios = []
    for _ in range(M):
        lvl_dur = {}
        for lvl in levels:
            base = int(rng.choice(P.DURATION_SUPPORT[lvl]))
            eta = float(rng.choice(P.ETA))
            lvl_dur[lvl] = max(1, int(round(base * eta)))
        scenarios.append({int(r.edge_id): lvl_dur[(r.road_class, int(r.severity))] for r in rows})
    return scenarios
