"""
Static user-equilibrium (UE) traffic assignment for the toy network.

User equilibrium (UE) is the traffic state in which no driver can lower their own
travel time by unilaterally switching route; this is Wardrop's first principle of
route choice. The heavy numerical work runs inside the AequilibraE library, so this
module spells out the high-level method in plain terms. `solve_ue` is the single
entry point.

    INPUT
      - A road network in which each link has a free-flow travel time t0 (the time to
        traverse it when empty), a capacity c, and BPR parameters (alpha, beta).
      - An origin-destination (OD) demand matrix h giving the number of trips per
        period from each origin zone to each destination zone.

    CORE LOGIC  (Frank-Wolfe user equilibrium)
      The UE flow pattern is the unique minimizer of the Beckmann objective
            Z(x) = sum_a  integral_0^{x_a} t_a(w) dw,
      where each link's cost rises with congestion through the BPR (Bureau of Public
      Roads) travel-time function
            t_a(x) = t0_a * (1 + alpha * (x_a / c_a)^beta).
      Frank-Wolfe is an iterative descent method that minimizes Z by repeating:
        1. ALL-OR-NOTHING loading: holding the current link costs fixed, assign every
           OD trip entirely to its current shortest path, yielding an auxiliary
           target flow y.
        2. MOVE: shift the current flow x a fractional step toward y so that Z
           decreases (x <- x + step * (y - x)); the step size is picked by a line
           search along the segment from x to y.
        3. RECOMPUTE link costs from the new flows, and stop once the RELATIVE GAP
           (how far x still sits from its all-or-nothing target, i.e. the remaining
           room for improvement) drops below rgap_target.
      AequilibraE uses the bi-conjugate Frank-Wolfe variant ("bfw"), which reuses
      information from earlier iterations to converge faster than plain Frank-Wolfe.

    OUTPUT
      - For each directed link, the equilibrium flow `volume` and the resulting
        congested travel time `cost`, returned as a tidy
        DataFrame[from, to, volume, cost].
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

    Each row of `edges` is one physical road treated as a bidirectional link
    (direction=0), so its capacity and BPR attributes are copied identically to the
    forward (AB) and reverse (BA) directions. Every network node is also registered
    as a zone (a place where trips can start or end) so demand can load anywhere.
    """
    net = pd.DataFrame({
        "link_id": np.arange(1, len(edges) + 1, dtype=np.int64),
        "a_node": edges["u"].to_numpy(dtype=np.int64),
        "b_node": edges["v"].to_numpy(dtype=np.int64),
        "direction": 0,                       # direction 0 = link usable both ways
        "distance": edges["length"].to_numpy(dtype=float),
        "modes": "c",
        "capacity_ab": edges["capacity"].to_numpy(dtype=float),
        "capacity_ba": edges["capacity"].to_numpy(dtype=float),
        "free_flow_time": edges["free_flow_time"].to_numpy(dtype=float),
        "b": edges["bpr_alpha"].to_numpy(dtype=float),      # BPR alpha coefficient (AequilibraE names this column "b")
        "power": edges["bpr_beta"].to_numpy(dtype=float),   # BPR beta exponent (AequilibraE names this column "power")
    })
    net["id"] = net["link_id"]
    g = Graph()
    g.network = net
    g.prepare_graph(np.asarray(zone_ids, dtype=np.int64))   # register every node as a trip origin/destination
    g.set_graph("free_flow_time")
    g.set_blocked_centroid_flows(False)                     # let routes pass through zones rather than terminate at them
    return g, net


def _build_matrix(M, zone_ids):
    """Wrap a dense NumPy OD demand matrix in the in-memory AequilibraE matrix
    container the assignment engine expects, indexed by zone id."""
    mat = AequilibraeMatrix()
    mat.create_empty(memory_only=True, zones=len(zone_ids), matrix_names=["demand"])
    mat.index[:] = np.asarray(zone_ids, dtype=np.int64)
    mat.matrix["demand"][:, :] = M
    mat.computational_view(["demand"])
    return mat


def solve_ue(edges, od_matrix, zone_ids, algorithm="bfw", max_iter=1000, rgap=1e-10, quiet=False):
    """Solve static user equilibrium and return (flows_df, assignment).

    `flows_df` is a DataFrame[from, to, volume, cost] with one row per directed link,
    where `volume` is the equilibrium flow on that link and `cost` its congested
    travel time. The second value is the AequilibraE assignment object, kept so the
    caller can inspect the convergence report. The module docstring describes the
    underlying UE method.

    `edges` must supply the columns u, v, capacity, length, free_flow_time,
    bpr_alpha, bpr_beta. `algorithm` names the assignment algorithm ("bfw" =
    bi-conjugate Frank-Wolfe), `max_iter` caps the number of iterations, and `rgap`
    is the relative-gap tolerance at which the solver is considered converged. Set
    `quiet=True` to suppress AequilibraE's logging and progress output.
    """
    graph, net = _build_graph(edges, zone_ids)        # turn the network table into a routable graph
    mat = _build_matrix(od_matrix, zone_ids)          # wrap the OD demand as an AequilibraE matrix

    tc = TrafficClass("car", graph, mat)
    assig = TrafficAssignment()
    assig.set_classes([tc])
    assig.set_vdf("BPR")                                       # let flow raise travel time via the BPR function
    assig.set_vdf_parameters({"alpha": "b", "beta": "power"})  # read BPR alpha/beta from graph columns "b"/"power"
    assig.set_capacity_field("capacity")
    assig.set_time_field("free_flow_time")
    assig.set_algorithm(algorithm)                             # e.g. "bfw" = bi-conjugate Frank-Wolfe
    assig.max_iter = max_iter
    assig.rgap_target = rgap
    if quiet:                                                  # route logging and progress bars to null
        logging.getLogger("aequilibrae").setLevel(logging.CRITICAL)
        with open(os.devnull, "w") as _dn, contextlib.redirect_stdout(_dn), \
                contextlib.redirect_stderr(_dn):
            assig.execute()
    else:
        assig.execute()                                        # run the Frank-Wolfe iterations

    res = assig.results()                                      # equilibrium flow + congested time per link, both directions
    # the assigned-flow columns take their name from the matrix core, here "demand"
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
    """Evaluate the Beckmann objective Z(x) in closed form for a given flow pattern.

    Integrating the BPR link cost from zero flow up to x gives the per-link term
        t0*x + t0*alpha/(beta+1) * x^(beta+1) / cap^beta,
    and Z is the sum of these terms over all links. User equilibrium is the flow that
    minimizes this convex Z, so comparing Z between two flow patterns measures how
    close each is to equilibrium. `flows` has columns from, to, volume; `linkp` (from
    util.io.directed_link_params) supplies each link's capacity, free_flow_time,
    alpha, and beta.
    """
    m = flows.merge(linkp, on=["from", "to"], how="left")
    x = m["volume"].to_numpy()
    t0 = m["free_flow_time"].to_numpy()
    cap = m["capacity"].to_numpy()
    a = m["alpha"].to_numpy()
    b = m["beta"].to_numpy()
    return float(np.sum(t0 * x + t0 * (a / (b + 1.0)) * np.power(x, b + 1.0) / np.power(cap, b)))


# --------------------------------------------------------------------------- #
# Self-check: running this module as a script assigns UE on the Sioux Falls toy
# network and checks the result against its published reference solution, so a
# regression in the engine surfaces immediately.   ->   python -m util.ue
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
