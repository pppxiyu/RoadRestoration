"""
Monte Carlo sampling of road-restoration duration scenarios.

Repair time is uncertain, so instead of committing to a single estimate we draw
many equally plausible "scenarios" and later optimize against the whole set. A
scenario is one joint realization of how long every disrupted road segment takes
to repair, measured in discrete time slots, returned as a dict
{edge_id: duration_in_slots}.

Durations are sampled per LEVEL, where a level is the pair (road_class,
severity): segments that share the same road class and damage severity are
treated as interchangeable, so all of them receive the same drawn duration
within a given scenario. Each per-level draw combines a base repair time (from a
per-level support set, i.e. the set of plausible base durations for that road
class and severity) with a crew-efficiency factor eta (a multiplier for how
fast the repair crew works relative to nominal).
"""

import numpy as np

import config as P


def sample_scenarios(disrupted, M=P.M_SCENARIOS, seed=P.SEED):
    """Draw M restoration-duration scenarios for the disrupted segments.

    `disrupted` is a DataFrame with columns edge_id, road_class, and severity.
    Returns a list of M dicts; scenario[m][edge_id] is that segment's realized
    repair duration in time slots. Because every segment of the same
    (road_class, severity) level shares one duration within a scenario, a
    duration is drawn once per distinct level and then broadcast to all of that
    level's segments. A fixed seed makes the whole sample reproducible.
    """
    rng = np.random.default_rng(seed)
    rows = list(disrupted.itertuples(index=False))
    # Enumerate the distinct (road_class, severity) levels present; sorting keeps
    # the draw order deterministic so a given seed always yields the same sample.
    levels = sorted({(r.road_class, int(r.severity)) for r in rows})
    scenarios = []
    for _ in range(M):
        # Draw one duration per level, then assign it to every segment of that level.
        lvl_dur = {}
        for lvl in levels:
            # Pick a plausible base repair time and an independent crew-efficiency
            # multiplier for this level in this scenario.
            base = int(rng.choice(P.DURATION_SUPPORT[lvl]))
            eta = float(rng.choice(P.ETA))
            # Scale the base time by crew efficiency and round to whole slots,
            # flooring at one so no repair is treated as instantaneous.
            lvl_dur[lvl] = max(1, int(round(base * eta)))
        scenarios.append({int(r.edge_id): lvl_dur[(r.road_class, int(r.severity))] for r in rows})
    return scenarios
