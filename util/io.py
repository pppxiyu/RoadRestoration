"""Data loading for the core logic: the Sioux Falls toy network/OD, and the
open-source reference UE solution used as ground truth."""

from pathlib import Path

import numpy as np
import pandas as pd


def load_toy_network(toy_dir):
    """Load the toy instance. Returns (edges_df, od_df, zone_ids).

    edges_df: one row per undirected edge, columns
              edge_id, u, v, capacity, length, free_flow_time, bpr_alpha, bpr_beta, road_class
    od_df:    od_id, origin, destination, h0
    zone_ids: 1-D int array of node ids (every node is an OD zone in Sioux Falls)
    """
    toy = Path(toy_dir)
    edges = pd.read_csv(toy / "network" / "edges.csv")
    od = pd.read_csv(toy / "network" / "od_pairs.csv")
    nodes = pd.read_csv(toy / "network" / "nodes.csv")
    zone_ids = nodes["node_id"].to_numpy(dtype=np.int64)
    return edges, od, zone_ids


def od_to_matrix(od, zone_ids):
    """Dense OD demand matrix M[i, j] (trips from zone_ids[i] to zone_ids[j])."""
    pos = {int(z): i for i, z in enumerate(zone_ids)}
    M = np.zeros((len(zone_ids), len(zone_ids)), dtype=np.float64)
    for r in od.itertuples(index=False):
        M[pos[int(r.origin)], pos[int(r.destination)]] = float(r.h0)
    return M


def directed_link_params(edges):
    """Expand undirected edges to a directed link-parameter table
    (from, to, capacity, free_flow_time, alpha, beta) — both directions per edge.
    Sioux Falls links are symmetric, so each direction inherits the same values."""
    rows = []
    for r in edges.itertuples(index=False):
        for a, b in ((r.u, r.v), (r.v, r.u)):
            rows.append(dict(**{"from": int(a), "to": int(b)},
                             capacity=float(r.capacity),
                             free_flow_time=float(r.free_flow_time),
                             alpha=float(r.bpr_alpha), beta=float(r.bpr_beta)))
    return pd.DataFrame(rows)


def load_reference_flows(flow_tntp):
    """Parse SiouxFalls_flow.tntp -> DataFrame[from, to, volume, cost].
    `volume` = equilibrium link flow, `cost` = congested travel time (the BPR value)."""
    rows = []
    for line in Path(flow_tntp).read_text().splitlines():
        s = line.strip()
        if not s or s.lower().startswith("from") or s.startswith("~") or s.startswith("<"):
            continue
        t = s.rstrip(";").split()
        if len(t) < 4:
            continue
        try:
            rows.append(dict(**{"from": int(t[0]), "to": int(t[1])},
                             volume=float(t[2]), cost=float(t[3])))
        except ValueError:
            continue
    return pd.DataFrame(rows)
