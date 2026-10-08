"""Input readers for the traffic-assignment experiments.

This module loads the Sioux Falls toy road network together with its
origin-destination (OD) travel demand — the number of trips each zone sends to
each other zone — and the published reference solution used as ground truth.
The reference is a user-equilibrium (UE) flow pattern: the traffic state in
which no driver can lower their own travel time by switching to a different
route. Comparing our solver against this known UE solution is how we check that
the assignment code is correct."""

from pathlib import Path

import numpy as np
import pandas as pd


def load_toy_network(toy_dir):
    """Read the three CSV files that define the toy instance.

    Returns (edges_df, od_df, zone_ids):

    edges_df: one row per undirected road segment, with the physical and
              congestion parameters needed to compute travel times — columns
              edge_id, u, v, capacity, length, free_flow_time, bpr_alpha,
              bpr_beta, road_class. Here bpr_alpha and bpr_beta are the two
              shape parameters of the BPR cost function, the standard formula
              that inflates a segment's travel time as its flow approaches
              capacity.
    od_df:    the travel-demand table — od_id, origin, destination, and h0, the
              baseline number of trips requested for that origin-destination
              pair.
    zone_ids: 1-D int array of node ids. In the Sioux Falls benchmark every
              node is also a demand zone, so this doubles as the list of zones.
    """
    toy = Path(toy_dir)
    edges = pd.read_csv(toy / "network" / "edges.csv")
    od = pd.read_csv(toy / "network" / "od_pairs.csv")
    nodes = pd.read_csv(toy / "network" / "nodes.csv")
    zone_ids = nodes["node_id"].to_numpy(dtype=np.int64)
    return edges, od, zone_ids


def od_to_matrix(od, zone_ids):
    """Build the dense OD demand matrix from the sparse OD table.

    Traffic assignment expects demand as a square matrix M where entry M[i, j]
    is the number of trips from zone_ids[i] to zone_ids[j]. The input table
    lists only the non-zero pairs, so we start from an all-zeros matrix and fill
    in each listed trip volume; any pair absent from the table stays zero.
    """
    # Map each external zone id to its row/column index in the matrix.
    pos = {int(z): i for i, z in enumerate(zone_ids)}
    M = np.zeros((len(zone_ids), len(zone_ids)), dtype=np.float64)
    for r in od.itertuples(index=False):
        M[pos[int(r.origin)], pos[int(r.destination)]] = float(r.h0)
    return M


def directed_link_params(edges):
    """Turn each undirected road segment into two directed links.

    Traffic flows and travel times are directional, so the assignment step needs
    a table keyed by (from, to) rather than by undirected edge. Each input edge
    is therefore emitted twice, once per direction, carrying the parameters that
    drive the BPR travel-time formula (capacity, free_flow_time, and its two
    shape parameters alpha and beta). Sioux Falls segments are symmetric, so both
    directions simply inherit the same values.
    """
    rows = []
    for r in edges.itertuples(index=False):
        for a, b in ((r.u, r.v), (r.v, r.u)):
            rows.append(dict(**{"from": int(a), "to": int(b)},
                             capacity=float(r.capacity),
                             free_flow_time=float(r.free_flow_time),
                             alpha=float(r.bpr_alpha), beta=float(r.bpr_beta)))
    return pd.DataFrame(rows)


def load_reference_flows(flow_tntp):
    """Parse the reference UE solution file into DataFrame[from, to, volume, cost].

    The .tntp file is the published equilibrium solution for Sioux Falls, in the
    fixed-width text format used by the Transportation Networks benchmark set.
    For each directed link it gives `volume`, the equilibrium link flow (trips
    per period), and `cost`, the resulting congested travel time — i.e. the BPR
    travel time evaluated at that flow. These are the ground-truth values our
    solver is validated against.
    """
    rows = []
    for line in Path(flow_tntp).read_text().splitlines():
        s = line.strip()
        # Skip blanks, the header row, and metadata/comment lines (~, <...>).
        if not s or s.lower().startswith("from") or s.startswith("~") or s.startswith("<"):
            continue
        # Drop the trailing ";" and split the row into whitespace-separated fields.
        t = s.rstrip(";").split()
        if len(t) < 4:
            continue
        try:
            rows.append(dict(**{"from": int(t[0]), "to": int(t[1])},
                             volume=float(t[2]), cost=float(t[3])))
        except ValueError:
            # Ignore any row whose first four fields are not numeric.
            continue
    return pd.DataFrame(rows)
