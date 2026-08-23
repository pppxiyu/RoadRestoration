"""
Objective evaluator for the road-recovery toy problem: computes the exact objective
F(x | ω), where x is a repair schedule and ω is a realized scenario (the repair durations).

`evaluate_schedule(...)` combines two objectives that are computed in very different ways:
  - F2 (restoration efficiency)  : pure schedule arithmetic; no traffic model is needed.
  - F1 (accessibility degradation): a loop over time steps, each step solving one user
    equilibrium (UE) — the traffic state in which no driver can lower their own travel
    time by unilaterally switching routes.

Time is discretized into integer slots k = 0, 1, 2, ...; slot 0 is the disaster onset t0,
and each slot spans Δt hours. A schedule assigns each segment a start slot s_e >= 1, so
repair always begins strictly after onset. Segment e is under repair on the slots
[s_e, s_e+d_e): it stays damaged (severity v_e*) while k < s_e+d_e and becomes fully
restored (severity 0) once k >= s_e+d_e. The stages of this pipeline are exposed as separate
functions so the driver script can replay them one at a time with explicit labels.
"""

import networkx as nx
import numpy as np
import pandas as pd

import config as P
from util.io import load_toy_network
from util.ue import solve_ue, warm_start_seed
from util import sim_cache as _sim_cache


# --------------------------------------------------------------------------- #
# Origin-destination travel times: shortest paths over the congested link graph
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# Schedule construction (work-conserving) and the F2 objective
# --------------------------------------------------------------------------- #
def schedule_from_permutation(perm, durations, c_max=P.C_MAX):
    """Turn a priority ordering of segments into concrete start slots by greedy scheduling.

    `perm` lists the disrupted edge_ids in the order they should be repaired. Work is shared
    among c_max interchangeable crews under a work-conserving rule (no crew sits idle while
    repairs remain): each segment is handed to whichever crew frees up earliest, and the
    first repair may begin at slot 1. Returns {edge_id: start_slot}."""
    crew_free = [1] * c_max
    start = {}
    for e in perm:
        c = int(np.argmin(crew_free))
        start[e] = crew_free[c]
        crew_free[c] = start[e] + durations[e]   # this crew stays busy until the repair completes
    return start


def makespan_slot(start, durations):
    return max(start[e] + durations[e] for e in start)


def f2_value(start, durations):
    """Restoration-efficiency objective F2: the schedule's makespan (the completion slot of
    the last repair, counted from the onset slot t0) divided by the total repair work
    Σ_e d_e. Because every slot spans the same Δt hours, that factor cancels between elapsed
    time and total work, leaving completion_slot / Σ_e d_e. Lower is better — it rewards
    finishing all repairs sooner relative to the fixed amount of work they require."""
    return makespan_slot(start, durations) / sum(durations.values())


# --------------------------------------------------------------------------- #
# Static context, built once: network, OD structure, baseline times, B, u_pen
# --------------------------------------------------------------------------- #
def build_context(toy_dir, disrupted, ue_cores=None):
    """Assemble the context that stays fixed across every schedule and scenario.

    This bundles the network, the OD structure, the baseline (pre-disaster) travel times
    that F1 measures degradation against, the demand-shortfall matrix B, and the
    disconnection penalty u_pen. `disrupted` is a DataFrame with columns edge_id, u, v,
    severity listing the damaged segments to be scheduled, where u and v are a segment's
    two endpoint nodes. `ue_cores` is forwarded to the baseline UE solve's `cores` argument,
    a retained no-op kept so existing call sites keep working -- the in-house solver is
    single-threaded (see util.ue.solve_ue)."""
    edges, od, zone_ids = load_toy_network(toy_dir)
    zone_pos = {int(z): i for i, z in enumerate(zone_ids)}
    od_pairs = [(int(r.origin), int(r.destination)) for r in od.itertuples(index=False)]
    H0 = od["h0"].to_numpy(dtype=float)
    oi = np.array([zone_pos[o] for o, _ in od_pairs])
    di = np.array([zone_pos[d] for _, d in od_pairs])
    edge_row = {int(r.edge_id): i for i, r in enumerate(edges.itertuples(index=False))}
    eid_of = {tuple(sorted((int(r.u), int(r.v)))): int(r.edge_id)
              for r in edges.itertuples(index=False)}

    ctx = dict(toy_dir=str(toy_dir), edges=edges, zone_ids=zone_ids, od_pairs=od_pairs, H0=H0,
               oi=oi, di=di, nz=len(zone_ids), edge_row=edge_row,
               origins_unique=sorted({o for o, _ in od_pairs}))

    # Baseline OD travel times at onset: solve UE on the intact network under normal demand
    # H0. These serve as the reference times against which F1 later scores degradation, so they
    # use the tight DEFINITION tolerance -- the reference must be stable even while the per-slot
    # evaluation solves run loose (see config UE_RGAP_DEF vs UE_RGAP).
    base_links, _ = solve_ue(edges, _matrix_from_H(H0, ctx), zone_ids,
                             rgap=P.UE_RGAP_DEF, max_iter=P.UE_MAX_ITER_DEF, quiet=True,
                             cores=ue_cores)
    ctx["baseline_u"] = od_travel_times(base_links, ctx)
    # Penalty travel time for disconnected OD pairs: a multiple of the worst finite baseline
    # time, so a lost connection is charged a large but bounded cost rather than infinity.
    ctx["u_pen"] = P.UPEN_FACTOR * float(np.nanmax(ctx["baseline_u"][np.isfinite(ctx["baseline_u"])]))

    # Disrupted-segment records, plus the demand-shortfall matrix B. Entry B[r, j] captures
    # how much OD pair r's demand drops per unit severity of disrupted segment j, and it is
    # non-zero only when segment j lies on r's free-flow (uncongested) shortest path — the
    # premise being that damage on a route a traveler would normally take suppresses that
    # trip. The coefficient KAPPA * (H0[r]/3) scales the drop by the pair's baseline demand.
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
# The objective-evaluation pipeline
# --------------------------------------------------------------------------- #
def evaluate_schedule(start, durations, T, ctx, collect_traces=False, return_u=False):
    """Evaluate F(x | ω) for a single schedule (the start slots) under a single scenario
    (the repair durations), over a horizon of T slots. Returns a dict holding the combined
    objective F alongside its two components F1 and F2, plus optional per-step traces. The
    module docstring explains the slot conventions used here."""
    dis = ctx["disrupted"]
    H0, B = ctx["H0"], ctx["B"]
    base_u = ctx["baseline_u"]

    # F2 is pure schedule arithmetic and needs no traffic model.
    F2 = f2_value(start, durations)

    # F1 is accumulated over the horizon: for every slot k, reconstruct the traffic state the
    # schedule produces and record one accessibility term for that slot.
    D = np.zeros(len(H0))
    terms, active, traces, u_rows = [], [], [], []
    # Persistent slot-term cache (config.SIM_CACHE): the key is each segment's completion slot
    # capped at k+1, exactly as util.rl._evaluate_prefix_cached memoizes -- see util/sim_cache.py
    # for why that prefix is the correct invariant. Disabled when the caller needs the per-slot
    # travel-time vectors (return_u): a cached term cannot reconstruct them.
    comp = tuple(start[eid] + durations[eid] for (eid, _, _, _) in dis)
    psc = None if return_u else _sim_cache.for_ctx(ctx)
    # Warm-start chain (config.UE_WARM_START): the last SOLVED slot's equilibrium links and its
    # routed demand (demand zeroed on OD pairs that were disconnected, whose trips went unrouted),
    # from which the next slot's feasible seed is built. The chain premise -- per-OD demand never
    # decreases while damage clears -- holds within a schedule because the shortfall D only decays
    # after slot 1, and it is asserted rather than trusted. A slot served from the cache yields no
    # links, so the chain breaks there and the next solved slot starts cold; the term it produces
    # differs from the fully-chained one only within the solver tolerance, which is the accuracy
    # class the cache already promises (config.UE_RGAP is part of the cache fingerprint).
    warm = None
    for k in range(1, T + 1):
        # Damage state at slot k: a segment is still broken while k has not yet reached its
        # completion slot start+duration; v_vec carries the severity, or 0 once restored.
        damaged = {eid: s for (eid, _, _, s) in dis if k < start[eid] + durations[eid]}
        v_vec = np.array([s if (eid in damaged) else 0.0 for (eid, _, _, s) in dis])
        # Demand shortfall D and the demand H that survives it. `target` is the shortfall the
        # current damage would cause; D jumps up to it at once when damage worsens but only
        # decays geometrically (retaining fraction RHO each slot) as damage clears, capturing
        # trips that return gradually. H is the baseline demand minus that shortfall.
        target = B @ v_vec
        D = np.maximum(target, P.RHO * D)
        H = np.clip(H0 - D, 0.0, None)
        if psc is not None:
            hit = psc.get((k, tuple(min(c, k + 1) for c in comp)))
            if hit is not None:
                warm = None                       # no links from a cached term: chain breaks
                terms.append(hit)
                active.append(len(damaged) > 0)
                if collect_traces:
                    traces.append(dict(k=k, n_damaged=len(damaged),
                                       total_demand=float(H.sum()), f1_term=hit))
                continue
        # Network as it stands at slot k, with severed and degraded links applied.
        dmg_edges = build_damaged_edges(ctx, damaged)
        # Solve UE on the damaged network under demand H to obtain congested link times, then
        # collapse them into OD travel times. Disconnected pairs (infinite time) are charged
        # the finite penalty u_pen so the objective stays well defined.
        # cores=1: a retained no-op under the single-threaded in-house solver (see
        # util.ue.solve_ue). It dates from the retired multi-threaded engine, where single-core
        # was faster in WALL time on this 24-node network and was measured to change the slot
        # term g by at most 3.8e-15 vs the multi-core result (the ue-tolerance probe, 2026-08-11).
        x0 = None
        if P.UE_WARM_START and warm is not None:
            links_prev, H_routed_prev = warm
            dH = H - H_routed_prev
            if float(dH.min()) < -1e-9:
                raise RuntimeError(f"warm-start premise violated at slot {k}: per-OD demand "
                                   f"decreased by {-float(dH.min()):.3e} while damage was clearing")
            x0 = warm_start_seed(dmg_edges, _matrix_from_H(np.clip(dH, 0.0, None), ctx),
                                 ctx["zone_ids"], links_prev)
        links, _ = solve_ue(dmg_edges, _matrix_from_H(H, ctx), ctx["zone_ids"],
                            rgap=P.UE_RGAP, max_iter=P.UE_MAX_ITER, quiet=True, cores=1, x0=x0)
        u = od_travel_times(links, ctx)
        if P.UE_WARM_START:
            # Routed demand: a pair with no route (infinite u) had its trips left unloaded, so
            # the NEXT slot's increment must carry that pair's whole demand, not just its growth.
            warm = (links, np.where(np.isfinite(u), H, 0.0))
        u_tilde = np.where(np.isfinite(u), u, ctx["u_pen"])
        u_rows.append(u_tilde)
        # Accessibility term for slot k: realized travel time relative to the baseline, both
        # weighted by demand H so busier OD pairs count for more. A value of 1.0 means no
        # degradation, which is also the fallback when there is no demand to weight.
        den = float(np.sum(H * base_u))
        term = float(np.sum(H * u_tilde) / den) if den > 0 else 1.0
        if psc is not None:
            psc.put((k, tuple(min(c, k + 1) for c in comp)), term)
        terms.append(term)
        active.append(len(damaged) > 0)
        if collect_traces:
            traces.append(dict(k=k, n_damaged=len(damaged), total_demand=float(H.sum()),
                               f1_term=term))

    terms = np.asarray(terms)
    if P.F1_ACTIVE_ONLY:                      # average only over slots that still have damage
        mask = np.asarray(active, dtype=bool)
        F1 = float(terms[mask].mean()) if mask.any() else float(terms.mean())
    else:
        F1 = float(terms.mean())
    # Blend the two objectives; MU is the weight given to accessibility over efficiency.
    F = P.MU * F1 + (1.0 - P.MU) * F2
    out = dict(F=F, F1=F1, F2=F2)
    if return_u:
        # (T, |R|) matrix of per-slot OD travel times, reused downstream as fixed surrogate
        # cost coefficients — a surrogate being a cheap stand-in that avoids re-solving UE.
        out["u_tilde"] = np.asarray(u_rows)
    if collect_traces:
        out["traces"] = pd.DataFrame(traces)
    return out
