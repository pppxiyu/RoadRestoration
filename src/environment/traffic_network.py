"""Network transformations shared by current and historical traffic solvers."""
import networkx as nx
import numpy as np
from src import config as P


def od_travel_times(link_df, ctx):
    """Compute each origin-destination (OD) pair's travel time as the shortest-path
    distance through the congested link-time graph.

    `link_df` has columns [from, to, cost] giving the congested travel time of each
    directed link. The returned array u_r is aligned to ctx['od_pairs']; an entry is
    np.inf when the origin cannot reach the destination (the two are disconnected)."""
    G = nx.DiGraph()
    fa = link_df["from"].to_numpy()
    ta = link_df["to"].to_numpy()
    ca = link_df["cost"].to_numpy()
    for a, b, w in zip(fa, ta, ca):
        G.add_edge(int(a), int(b), weight=float(w))
    u = np.full(len(ctx["od_pairs"]), np.inf)
    by_origin = {}
    for o in ctx["origins_unique"]:
        if o in G:
            by_origin[o] = nx.single_source_dijkstra_path_length(G, o, weight="weight")
        else:
            by_origin[o] = {}
    for i, (o, d) in enumerate(ctx["od_pairs"]):
        u[i] = by_origin.get(o, {}).get(d, np.inf)
    return u

def _matrix_from_H(H, ctx):
    M = np.zeros((ctx["nz"], ctx["nz"]), dtype=float)
    M[ctx["oi"], ctx["di"]] = H
    return M

def build_damaged_edges(ctx, damaged):
    """Build the edges DataFrame describing the network in its currently-damaged state.

    `damaged` maps each still-broken segment's edge_id to its severity. A segment whose
    severity reaches the SEVER_SEVERITY threshold is dropped from the network entirely,
    modelling a true disconnection: OD pairs that relied on it can no longer reach their
    destination and later fall back to the penalty travel time u_pen. Less severe damage
    instead degrades the link in place — its capacity is scaled down and its free-flow time
    (the uncongested travel time) is inflated. Segments that are undamaged or already
    repaired keep their original attributes."""
    edges = ctx["edges"].copy()
    if damaged:
        cap = edges["capacity"].to_numpy(dtype=float, copy=True)
        fft = edges["free_flow_time"].to_numpy(dtype=float, copy=True)
        idx = ctx["edge_row"]
        sever = []
        for eid, sev in damaged.items():
            j = idx[eid]
            if sev >= P.SEVER_SEVERITY:
                sever.append(j)                    # remove this edge entirely
            else:
                cap[j] *= P.CAP_RETAIN[sev]
                fft[j] /= P.SPEED_RETAIN[sev]
        edges["capacity"] = cap
        edges["free_flow_time"] = fft
        if sever:
            edges = edges.drop(index=edges.index[sever]).reset_index(drop=True)
    return edges
