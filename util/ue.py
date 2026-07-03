"""
User-equilibrium (UE) static traffic assignment for the toy network.

The heavy numerical loop runs inside AequilibraE, so the high-level idea of the UE
method is written out here explicitly. The single entry point is `solve_ue`.

    INPUT
      - a road network: each link has a free-flow travel time t0, a capacity c, and
        BPR parameters (alpha, beta)
      - an origin-destination (OD) demand matrix h (trips per period, zone -> zone)

    CORE LOGIC  (Frank-Wolfe user equilibrium, Wardrop's first principle)
      User equilibrium = the link-flow pattern in which no traveler can reduce their
      own travel time by unilaterally changing route. It is the minimizer of the
      Beckmann objective
            Z(x) = sum_a  integral_0^{x_a} t_a(w) dw,
      where the BPR congested link cost is
            t_a(x) = t0_a * (1 + alpha * (x_a / c_a)^beta).
      Frank-Wolfe finds it by repeating:
        1. ALL-OR-NOTHING: with the current link costs, put every OD trip entirely on
           its current shortest path  ->  an auxiliary flow y.
        2. LINE SEARCH / MOVE: shift the current flow x a step toward y so that Z
           decreases  (x <- x + step * (y - x)).
        3. UPDATE COSTS from the new flows; stop when the RELATIVE GAP
           (distance between x and the all-or-nothing target) < rgap_target.
      AequilibraE uses the bi-conjugate Frank-Wolfe variant ("bfw"), which converges
      faster than plain Frank-Wolfe.

    OUTPUT
      - per directed link: the equilibrium flow `volume` and the congested travel
        time `cost`, returned as a tidy DataFrame[from, to, volume, cost].
"""

import contextlib
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

from aequilibrae.matrix import AequilibraeMatrix
from aequilibrae.paths import Graph, TrafficAssignment, TrafficClass


def _build_graph(edges, zone_ids):
    """Build a routable AequilibraE graph from the undirected edge table.
    Each edge is a bidirectional link (direction=0); symmetric AB/BA attributes."""
    net = pd.DataFrame({
        "link_id": np.arange(1, len(edges) + 1, dtype=np.int64),
        "a_node": edges["u"].to_numpy(dtype=np.int64),
        "b_node": edges["v"].to_numpy(dtype=np.int64),
        "direction": 0,                       # 0 = bidirectional
        "distance": edges["length"].to_numpy(dtype=float),
        "modes": "c",
        "capacity_ab": edges["capacity"].to_numpy(dtype=float),
        "capacity_ba": edges["capacity"].to_numpy(dtype=float),
        "free_flow_time": edges["free_flow_time"].to_numpy(dtype=float),
        "b": edges["bpr_alpha"].to_numpy(dtype=float),      # BPR alpha (field name "b")
        "power": edges["bpr_beta"].to_numpy(dtype=float),   # BPR beta  (field name "power")
    })
    net["id"] = net["link_id"]
    g = Graph()
    g.network = net
    g.prepare_graph(np.asarray(zone_ids, dtype=np.int64))   # all nodes are zones
    g.set_graph("free_flow_time")
    g.set_blocked_centroid_flows(False)                     # allow thru-traffic at zones
    return g, net


def _build_matrix(M, zone_ids):
    """Wrap a dense OD matrix as an AequilibraE matrix."""
    mat = AequilibraeMatrix()
    mat.create_empty(memory_only=True, zones=len(zone_ids), matrix_names=["demand"])
    mat.index[:] = np.asarray(zone_ids, dtype=np.int64)
    mat.matrix["demand"][:, :] = M
    mat.computational_view(["demand"])
    return mat


def solve_ue(edges, od_matrix, zone_ids, algorithm="bfw", max_iter=1000, rgap=1e-10, quiet=False):
    """Solve static user equilibrium. Returns (flows_df, assignment).

    flows_df: DataFrame[from, to, volume, cost] (one row per directed link).
    See the module docstring for the high-level UE idea (input / core logic / output).

    edges columns required: u, v, capacity, length, free_flow_time, bpr_alpha, bpr_beta.
    """
    graph, net = _build_graph(edges, zone_ids)        # network -> routable graph
    mat = _build_matrix(od_matrix, zone_ids)          # OD demand -> matrix

    tc = TrafficClass("car", graph, mat)
    assig = TrafficAssignment()
    assig.set_classes([tc])
    assig.set_vdf("BPR")                                       # BPR congestion cost
    assig.set_vdf_parameters({"alpha": "b", "beta": "power"})  # alpha=field b, beta=field power
    assig.set_capacity_field("capacity")
    assig.set_time_field("free_flow_time")
    assig.set_algorithm(algorithm)                             # bi-conjugate Frank-Wolfe
    assig.max_iter = max_iter
    assig.rgap_target = rgap
    if quiet:                                                  # silence logging + progress bars
        logging.getLogger("aequilibrae").setLevel(logging.CRITICAL)
        with open(os.devnull, "w") as _dn, contextlib.redirect_stdout(_dn), \
                contextlib.redirect_stderr(_dn):
            assig.execute()
    else:
        assig.execute()                                        # <-- runs the FW loop

    res = assig.results()                                      # per-link AB/BA flow + cost
    # assigned-flow columns are named after the matrix core ("demand")
    link = net.set_index("link_id")
    rows = []
    for lid, r in res.iterrows():
        u = int(link.loc[lid, "a_node"])
        v = int(link.loc[lid, "b_node"])
        rows.append({"from": u, "to": v, "volume": r["demand_ab"], "cost": r["Congested_Time_AB"]})
        rows.append({"from": v, "to": u, "volume": r["demand_ba"], "cost": r["Congested_Time_BA"]})
    flows = pd.DataFrame(rows)
    flows = flows[flows["volume"].notna()].reset_index(drop=True)
    return flows, assig


def beckmann_objective(flows, linkp):
    """Beckmann objective Z = sum_a [ t0*x + t0*alpha/(beta+1) * x^(beta+1)/cap^beta ],
    the convex function user equilibrium minimizes. `flows` has columns from,to,volume;
    `linkp` (from util.io.directed_link_params) carries capacity/free_flow_time/alpha/beta."""
    m = flows.merge(linkp, on=["from", "to"], how="left")
    x = m["volume"].to_numpy()
    t0 = m["free_flow_time"].to_numpy()
    cap = m["capacity"].to_numpy()
    a = m["alpha"].to_numpy()
    b = m["beta"].to_numpy()
    return float(np.sum(t0 * x + t0 * (a / (b + 1.0)) * np.power(x, b + 1.0) / np.power(cap, b)))


# --------------------------------------------------------------------------- #
# Self-check: run this module to verify the UE engine reproduces the open-source
# Sioux Falls reference solution.   ->   python -m util.ue
# --------------------------------------------------------------------------- #
def _validate():
    import sys
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from util.io import (directed_link_params, load_reference_flows,
                         load_toy_network, od_to_matrix)
    from viz.ue_val import save_validation

    toy = root / "data" / "siouxfalls_toy"
    out = root / "outputs" / "UE_val"
    obj_rel_tol, mean_rel_flow_tol, rgap_target = 1e-3, 0.01, 1e-12

    edges, od, zone_ids = load_toy_network(toy)
    M = od_to_matrix(od, zone_ids)
    linkp = directed_link_params(edges)
    print(f"Toy network: {len(zone_ids)} zones, {len(edges)} undirected edges, "
          f"{int(M.sum())} total trips")
    print("Running user-equilibrium assignment (AequilibraE, bi-conjugate Frank-Wolfe)...")
    ue, assig = solve_ue(edges, M, zone_ids, algorithm="bfw", max_iter=2000, rgap=rgap_target)

    ref = load_reference_flows(toy / "raw" / "SiouxFalls_flow.tntp")
    cmp = ue.merge(ref, on=["from", "to"], suffixes=("_ue", "_ref"))
    err = (cmp["volume_ue"] - cmp["volume_ref"]).abs()
    mean_flow, mean_abs_err, max_abs_err = cmp["volume_ref"].mean(), err.mean(), err.max()
    corr = float(np.corrcoef(cmp["volume_ue"], cmp["volume_ref"])[0, 1])
    z_ue, z_ref = beckmann_objective(ue, linkp), beckmann_objective(ref, linkp)
    obj_gap = abs(z_ue / z_ref - 1.0)
    report = assig.report()
    try:
        rgap = float(report["rgap"].iloc[-1])
    except Exception:
        rgap = float("nan")

    print("\n================ UE vs. ground truth ================")
    print(f"links compared        : {len(cmp)} (of {len(ref)})")
    print(f"final rgap            : {rgap:.2e}")
    print(f"mean |flow error|     : {mean_abs_err:,.2f}  ({mean_abs_err/mean_flow*100:.3f}% of mean)")
    print(f"max  |flow error|     : {max_abs_err:,.2f}")
    print(f"flow correlation      : {corr:.6f}")
    print(f"Beckmann obj UE / ref : {z_ue:,.1f} / {z_ref:,.1f}  (rel gap {obj_gap:.2e})")

    status = "PASS" if (obj_gap < obj_rel_tol and (mean_abs_err / mean_flow) < mean_rel_flow_tol) else "FAIL"
    print(f"\nMILESTONE: {status}")

    metrics = dict(corr=corr, mean_abs_err=mean_abs_err, max_abs_err=max_abs_err,
                   mean_flow=mean_flow, z_ue=z_ue, z_ref=z_ref, obj_gap=obj_gap,
                   rgap=rgap, rgap_target=rgap_target, status=status)
    saved = save_validation(out, toy, cmp, report, metrics)
    print(f"Visual validation written to: {saved}")
    return status == "PASS"


if __name__ == "__main__":
    _validate()
