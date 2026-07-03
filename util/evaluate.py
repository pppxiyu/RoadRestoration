"""
Figure-1 evaluator: the exact objective F(x | ω) for the road-recovery toy.

`evaluate_schedule(...)` implements the paper's §1 / Figure 1 pipeline. The TWO objectives
are computed very differently:
  - F2 (restoration efficiency)  : pure schedule arithmetic, NO UE.
  - F1 (accessibility degradation): a per-step loop with ONE UE solve per step.

Time is in integer slots k = 0,1,2,...; slot 0 = disaster onset t0; Δt hours/slot.
A schedule is start slots {s_e}, s_e >= 1 (start strictly after onset). Segment e is under
repair on slots [s_e, s_e+d_e); damaged (v_e*) for k < s_e+d_e, restored (0) for k >= s_e+d_e
(Eq. 2). Building blocks are exposed so main.py can replay the steps with explicit labels.
"""

import networkx as nx
import numpy as np
import pandas as pd

import config as P
from util.io import load_toy_network
from util.ue import solve_ue


# --------------------------------------------------------------------------- #
# OD travel times from per-link (congested) times, via shortest paths
# --------------------------------------------------------------------------- #
def od_travel_times(link_df, ctx):
    """link_df: DataFrame[from, to, cost] (directed congested link times).
    Returns u_r aligned to ctx['od_pairs'] (np.inf if O and D are disconnected)."""
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
    """Return an edges DataFrame for the currently-damaged network. `damaged`: {edge_id: severity}.
    Segments with severity >= SEVER_SEVERITY are REMOVED from the network (true disconnection -> the
    affected OD pairs get u_pen); milder ones keep reduced capacity/free_flow_time. Completed/undamaged
    links are unchanged."""
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


# --------------------------------------------------------------------------- #
# Schedule construction (work-conserving) + F2
# --------------------------------------------------------------------------- #
def schedule_from_permutation(perm, durations, c_max=P.C_MAX):
    """perm: list of disrupted edge_ids in priority order. Work-conserving list scheduling
    with c_max identical crews; earliest start slot = 1. Returns {edge_id: start_slot}."""
    crew_free = [1] * c_max
    start = {}
    for e in perm:
        c = int(np.argmin(crew_free))
        start[e] = crew_free[c]
        crew_free[c] = start[e] + durations[e]   # crew busy until completion slot
    return start


def makespan_slot(start, durations):
    return max(start[e] + durations[e] for e in start)


def f2_value(start, durations):
    """F2 = (makespan - t0) / Σ_e d_e·Δt = comp_slot / Σ_e d_e  (Δt cancels)."""
    return makespan_slot(start, durations) / sum(durations.values())


# --------------------------------------------------------------------------- #
# Static context (network, OD, baseline u^{t0}, B, u_pen) — built once
# --------------------------------------------------------------------------- #
def build_context(toy_dir, disrupted):
    """disrupted: DataFrame with columns edge_id, u, v, severity (the ℰ being scheduled)."""
    edges, od, zone_ids = load_toy_network(toy_dir)
    zone_pos = {int(z): i for i, z in enumerate(zone_ids)}
    od_pairs = [(int(r.origin), int(r.destination)) for r in od.itertuples(index=False)]
    H0 = od["h0"].to_numpy(dtype=float)
    oi = np.array([zone_pos[o] for o, _ in od_pairs])
    di = np.array([zone_pos[d] for _, d in od_pairs])
    edge_row = {int(r.edge_id): i for i, r in enumerate(edges.itertuples(index=False))}
    eid_of = {tuple(sorted((int(r.u), int(r.v)))): int(r.edge_id)
              for r in edges.itertuples(index=False)}

    ctx = dict(edges=edges, zone_ids=zone_ids, od_pairs=od_pairs, H0=H0,
               oi=oi, di=di, nz=len(zone_ids), edge_row=edge_row,
               origins_unique=sorted({o for o, _ in od_pairs}))

    # baseline u_r^{t0}: UE on the UNDAMAGED network with the normal demand H0
    base_links, _ = solve_ue(edges, _matrix_from_H(H0, ctx), zone_ids,
                             rgap=P.UE_RGAP, max_iter=P.UE_MAX_ITER, quiet=True)
    ctx["baseline_u"] = od_travel_times(base_links, ctx)
    ctx["u_pen"] = P.UPEN_FACTOR * float(np.nanmax(ctx["baseline_u"][np.isfinite(ctx["baseline_u"])]))

    # disrupted columns + B(Φ): B[r, j] = KAPPA*(h_r0/3) if disrupted_j on r's free-flow shortest path
    dis = [(int(r.edge_id), int(r.u), int(r.v), int(r.severity))
           for r in disrupted.itertuples(index=False)]
    ctx["disrupted"] = dis
    Gff = nx.DiGraph()
    for r in edges.itertuples(index=False):
        Gff.add_edge(int(r.u), int(r.v), weight=float(r.free_flow_time), eid=int(r.edge_id))
        Gff.add_edge(int(r.v), int(r.u), weight=float(r.free_flow_time), eid=int(r.edge_id))
    B = np.zeros((len(od_pairs), len(dis)))
    col = {eid: j for j, (eid, *_ ) in enumerate(dis)}
    for i, (o, d) in enumerate(od_pairs):
        try:
            path = nx.shortest_path(Gff, o, d, weight="weight")
        except nx.NetworkXNoPath:
            continue
        on_path = {tuple(sorted((path[t], path[t + 1]))) for t in range(len(path) - 1)}
        for (a, b) in on_path:
            eid = eid_of.get((a, b))
            if eid in col:
                B[i, col[eid]] = P.KAPPA * (H0[i] / 3.0)
    ctx["B"] = B
    ctx["severity_vec"] = np.array([sev for (_, _, _, sev) in dis], dtype=float)
    return ctx


# --------------------------------------------------------------------------- #
# THE Figure-1 pipeline
# --------------------------------------------------------------------------- #
def evaluate_schedule(start, durations, T, ctx, collect_traces=False, return_u=False):
    """F(x|ω) for one schedule (start slots) under one scenario (durations), horizon T slots.
    Returns {F, F1, F2[, traces]}.  See module docstring for the slot conventions."""
    dis = ctx["disrupted"]
    H0, B = ctx["H0"], ctx["B"]
    base_u = ctx["baseline_u"]
    sev = ctx["severity_vec"]

    # --- Figure 1, Step 2: F2 (no UE) ---
    F2 = f2_value(start, durations)

    # --- Figure 1, Step 3: per-step F1 loop ---
    D = np.zeros(len(H0))
    terms, active, traces, u_rows = [], [], [], []
    for k in range(1, T + 1):
        # Step 1 (per step): damage state v^{t_k}  (Eq. 2)
        damaged = {eid: s for (eid, _, _, s) in dis if k < start[eid] + durations[eid]}
        v_vec = np.array([s if (eid in damaged) else 0.0 for (eid, _, _, s) in dis])
        # Step 3a: demand shortfall -> H_t  (sharp drop to current shortfall, recover at rate RHO)
        target = B @ v_vec
        D = np.maximum(target, P.RHO * D)
        H = np.clip(H0 - D, 0.0, None)
        # Step 3b: damaged network
        dmg_edges = build_damaged_edges(ctx, damaged)
        # Step 3c: UE -> congested link times -> OD travel times u_r
        links, _ = solve_ue(dmg_edges, _matrix_from_H(H, ctx), ctx["zone_ids"],
                            rgap=P.UE_RGAP, max_iter=P.UE_MAX_ITER, quiet=True)
        u = od_travel_times(links, ctx)
        u_tilde = np.where(np.isfinite(u), u, ctx["u_pen"])
        u_rows.append(u_tilde)
        # Step 3d: demand-weighted ratio of realized to baseline travel time
        den = float(np.sum(H * base_u))
        term = float(np.sum(H * u_tilde) / den) if den > 0 else 1.0
        terms.append(term)
        active.append(len(damaged) > 0)
        if collect_traces:
            traces.append(dict(k=k, n_damaged=len(damaged), total_demand=float(H.sum()),
                               f1_term=term))

    terms = np.asarray(terms)
    if P.F1_ACTIVE_ONLY:                      # lever 5: only the active-recovery window
        mask = np.asarray(active, dtype=bool)
        F1 = float(terms[mask].mean()) if mask.any() else float(terms.mean())
    else:
        F1 = float(terms.mean())
    F = P.MU * F1 + (1.0 - P.MU) * F2
    out = dict(F=F, F1=F1, F2=F2)
    if return_u:
        out["u_tilde"] = np.asarray(u_rows)      # (T, |R|) fixed travel times for c_e^k
    if collect_traces:
        out["traces"] = pd.DataFrame(traces)
    return out
