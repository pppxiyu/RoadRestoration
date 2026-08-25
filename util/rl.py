"""Shared evaluation and feature primitives for the restoration-ordering RL solvers.

Four functions every solver still relies on: the EXACT objective with a per-slot prefix cache
(_evaluate_prefix_cached, numerically cross-checked against util.evaluate.evaluate_schedule and
backed by the persistent cross-run store in util.sim_cache), the per-scenario statics the feature
builder reads (_scenario_statics), the decision-point feature matrix (_decision_features), and the
completion-attributed reward (_completion_rewards).

HISTORY. This module used to carry the original value-loss DQN trainer. That solver was
superseded by util.rl_rank -- its TD-regression objective optimizes value CALIBRATION while the
delivered decision argmax_e Q(s, e) needs only the value RANKING, and under function approximation
a warm-start probe showed it drifting off the known optimum, which no value-stability knob fixed
-- and its training loop, tuned defaults and legacy variant names (rl_nominal_legacy /
rl_stoch_legacy) were removed on 2026-08-10 per the project owner's decision. The run records of
that era remain the evidence of the comparison; this module keeps only what is shared.
"""
import numpy as np
import pandas as pd

import config as P
from util import sim_cache as _sim_cache
from util.evaluate import (_matrix_from_H, build_damaged_edges, f2_value,
                           od_travel_times)
from util.ue import solve_ue, warm_start_seed

_N_FEAT = 8              # feature-vector length of x(s, e); see _decision_features


def _evaluate_prefix_cached(start, durations, T, ctx, slot_cache, collect_traces=False):
    """Exactly util.evaluate.evaluate_schedule, restated so each slot's accessibility term g_t
    can be memoized on the damage-state trajectory that produced it.

    g_t is a deterministic function of the sequence of damage states over slots 1..t (the demand
    shortfall D_t carries memory, so the whole prefix matters, not just slot t's state). That
    prefix is encoded by each segment's completion slot capped at t+1 -- capping at t+1 rather
    than t keeps "completed exactly at t" distinct from "still damaged at t". Two schedules with
    equal capped vectors share g_1..g_t, so eps-greedy episodes that keep the best-so-far head and
    explore the tail reuse most of the horizon's UE solves.

    Returns dict(F, F1, F2, terms, n_ue): `terms` are the per-slot g_t the reward needs, `n_ue`
    counts the UE solves actually run (cache misses). With collect_traces the return also carries
    a `traces` frame in the same schema util.evaluate.evaluate_schedule produces, so a delivered
    schedule's recovery curve can be stored without paying for a second evaluation. Cross-checked against evaluate_schedule in
    the validation suite; any numerical edit here must keep that equivalence."""
    dis = ctx["disrupted"]
    H0, B = ctx["H0"], ctx["B"]
    base_u = ctx["baseline_u"]
    # TRUE severities for this world, exactly as util.evaluate.evaluate_schedule resolves them: a
    # util.scenarios.Scenario carries them, a plain duration dict (the nominal world) falls back
    # to ctx["severity_vec"], the instance's reported estimates. The equivalence this function
    # promises against evaluate_schedule holds only if both read the same severities.
    sev = (np.array([float(durations.sev[int(eid)]) for (eid, _, _, _) in dis])
           if hasattr(durations, "sev") else ctx["severity_vec"])

    F2 = f2_value(start, durations)
    comp = tuple(start[eid] + durations[eid] for (eid, _, _, _) in dis)   # completion slot per segment, in dis order
    sev_key = tuple(int(x) for x in sev)          # severities are part of the slot term's identity

    D = np.zeros(len(H0))
    terms, active, traces = [], [], []
    n_ue = 0
    # Warm-start chain, mirroring util.evaluate.evaluate_schedule exactly (same seed construction,
    # same premise assert, same chain-breaks-on-cache-hit rule) -- the numerical equivalence this
    # docstring promises only survives if the two evaluators solve each slot identically.
    warm = None
    for k in range(1, T + 1):
        still = np.array([k < c for c in comp])
        v_vec = np.where(still, sev, 0.0)
        target = B @ v_vec
        D = np.maximum(target, P.RHO * D)         # shortfall: jumps with damage, decays at RHO
        H = np.clip(H0 - D, 0.0, None)
        key = (k, tuple(min(c, k + 1) for c in comp), sev_key)
        term = slot_cache.get(key)
        if term is None:
            # L2: the persistent cross-run store (config.SIM_CACHE). Same key, shared by every
            # method; a hit is promoted into this run's in-memory cache.
            psc = _sim_cache.for_ctx(ctx)
            if psc is not None:
                term = psc.get(key)
                if term is not None:
                    slot_cache[key] = term
        if term is None:
            damaged = {eid: int(sv) for (eid, _, _, _), sv, st in zip(dis, sev, still) if st}
            dmg_edges = build_damaged_edges(ctx, damaged)
            x0 = None
            if P.UE_WARM_START and warm is not None:
                links_prev, H_routed_prev = warm
                dH = H - H_routed_prev
                if float(dH.min()) < -1e-9:
                    raise RuntimeError(f"warm-start premise violated at slot {k}: per-OD demand "
                                       f"decreased by {-float(dH.min()):.3e} while damage was "
                                       f"clearing")
                x0 = warm_start_seed(dmg_edges, _matrix_from_H(np.clip(dH, 0.0, None), ctx),
                                     ctx["zone_ids"], links_prev)
            links, _ = solve_ue(dmg_edges, _matrix_from_H(H, ctx),
                                ctx["zone_ids"], rgap=P.UE_RGAP, max_iter=P.UE_MAX_ITER,
                                quiet=True, cores=1, x0=x0)     # cores=1: bit-reproducible (see util.ue.solve_ue)
            u = od_travel_times(links, ctx)
            if P.UE_WARM_START:
                # Routed demand: disconnected pairs' trips went unrouted, so the next slot's
                # increment must carry their whole demand, not just its growth.
                warm = (links, np.where(np.isfinite(u), H, 0.0))
            u_tilde = np.where(np.isfinite(u), u, ctx["u_pen"])
            den = float(np.sum(H * base_u))
            term = float(np.sum(H * u_tilde) / den) if den > 0 else 1.0
            slot_cache[key] = term
            if psc is not None:
                psc.put(key, term)
            n_ue += 1
        else:
            warm = None                           # no links from a cached term: chain breaks
        terms.append(term)
        active.append(bool(still.any()))
        if collect_traces:
            traces.append(dict(k=k, n_damaged=int(still.sum()), total_demand=float(H.sum()),
                               f1_term=term))

    terms = np.asarray(terms)
    if P.F1_ACTIVE_ONLY:
        mask = np.asarray(active, dtype=bool)
        F1 = float(terms[mask].mean()) if mask.any() else float(terms.mean())
    else:
        F1 = float(terms.mean())
    F = P.MU * F1 + (1.0 - P.MU) * F2
    out = dict(F=F, F1=F1, F2=F2, terms=terms, n_ue=n_ue)
    if collect_traces:
        out["traces"] = pd.DataFrame(traces)
    return out


# --------------------------------------------------------------------------- #
# Feature and reward primitives shared by the rl_rank solvers


def _scenario_statics(ctx, segments, durations, phi):
    """Per-scenario constants the feature builder needs, precomputed once. All per-segment
    scores are normalized to [0, 1] by their max over the disrupted set so the MLP sees
    commensurate inputs; phi_hat's ordering (all that the flow prior uses) is scale-invariant."""
    dis, sev, B = ctx["disrupted"], ctx["severity_vec"], ctx["B"]
    dem = {int(dis[j][0]): float(sev[j] * B[:, j].sum()) for j in range(len(dis))}
    sv = {int(eid): float(s) for (eid, _, _, s) in dis}
    d_max = max(int(durations[e]) for e in segments)
    dem_max = max(dem.values()) or 1.0
    phi_max = max(phi.values()) or 1.0
    total_work = float(sum(int(durations[e]) for e in segments))
    return dict(
        phi_hat={e: phi[e] / phi_max for e in segments},
        sev_hat={e: sv[e] / 3.0 for e in segments},
        dur_hat={e: int(durations[e]) / d_max for e in segments},
        dem_hat={e: dem[e] / dem_max for e in segments},
        dur_raw={e: int(durations[e]) for e in segments},   # planning durations, unnormalized
        d_max=float(d_max),
        total_work=total_work, sum_H0=float(ctx["H0"].sum()),
    )


def _decision_features(cands, t_now, crew_free, D_now, st, T, dur_belief=None, work_set=None):
    """Feature matrix x(s, e) for every candidate segment e at one decision point. Row layout
    (all roughly [0, 1]): [phi_hat_e, sev_hat_e, dur_belief_e, dem_hat_e, t/T, crew-gap, demand-
    shortfall fraction, remaining-work fraction]. Column 0 doubles as the flow prior that the
    residual Q adds outside the MLP.

    INFORMATION STRUCTURE. Everything here is computable at the moment the decision is made.
    Duration inputs (column 2, and the remaining-work numerator) come from `dur_belief`, the
    agent's duration beliefs; since the progressive-revelation mechanism was removed on
    2026-08-23, callers pass the planning expectations (or nothing, which defaults to them) and
    realized durations are never read into the state. The state inputs (t_now, crew_free, D_now)
    may reflect realized outcomes: they encode only what has already happened."""
    if dur_belief is None:
        dur_belief = st["dur_raw"]
    glob = (t_now / T,
            (max(crew_free) - t_now) / T,                        # how much longer the other crew is busy
            float(D_now.sum()) / st["sum_H0"],                    # demand currently suppressed
            sum(dur_belief[e] for e in (cands if work_set is None else work_set))
            / st["total_work"])                                  # remaining-work fraction: under the
                                                                 # accessibility constraint the rows
                                                                 # are only the REACHABLE candidates,
                                                                 # so callers pass the full remaining
                                                                 # set here to keep this global's
                                                                 # meaning (work left, not work
                                                                 # reachable)
    X = np.empty((len(cands), _N_FEAT))
    for i, e in enumerate(cands):
        X[i, 0] = st["phi_hat"][e]
        X[i, 1] = st["sev_hat"][e]
        X[i, 2] = dur_belief[e] / st["d_max"]
        X[i, 3] = st["dem_hat"][e]
        X[i, 4:] = glob
    return X


def _completion_rewards(perm, start, durations, terms, T, phi):
    """Per-decision rewards: the F-ALIGNED owned-window credit.

    Segment e finishes repair at slot tau_e = start_e + d_e, its first usable slot. Each distinct
    completion slot OWNS the slots from itself up to the next completion (the earliest completion
    also owns the head slots before it), and is credited the accessibility level (1 - g_t) summed
    over that window. The windows TILE 1..T, so the episode return equals
    sum_t (1 - g_t) = T*(1 - F) EXACTLY -- the objective's own decomposition, with NO
    piecewise-constant assumption, so it stays aligned even while demand rebounds WITHIN a window.
    Return and F are then affine, corr(return, F) = -1 by construction. The g values come from the
    trajectory that has already been evaluated, so attribution costs no extra UE solve.

    When several segments complete in the same slot they share that slot's window credit, split in
    proportion to their normal-period (baseline UE) flow phi_e, the same edge-criticality score the
    flow baseline ranks by.

    `terms` is the length-T array of per-slot g values (terms[k-1] is slot k); returns rewards
    aligned with `perm`, and each segment's completion slot so the caller can order transitions
    by when their reward became observable.

    (An earlier alternative, "echo" -- the completion-boundary jump delta_e = g_{tau_e-1} -
    g_{tau_e} weighted by a geometric decay at RHO -- was removed on 2026-08-10 per the project
    owner's decision: its weight saturates past ~10 slots, so the episode return degenerates to the
    endpoint quantity g0 - g_end and decouples from the integral objective F. Results committed
    under it live in their run records.)"""
    tau = {e: int(start[e]) + int(durations[e]) for e in perm}
    by_slot = {}
    for e in perm:
        by_slot.setdefault(tau[e], []).append(e)

    R = {}

    def _split(seg, total):                                        # divide one slot's credit by phi
        w_sum = sum(phi[e] for e in seg)
        for e in seg:
            R[e] = total * (phi[e] / w_sum if w_sum > 0 else 1.0 / len(seg))

    cs = sorted(by_slot)                                           # distinct completion slots, ascending
    for j, c in enumerate(cs):
        a = 1 if j == 0 else c                                     # first window absorbs the head slots
        b = cs[j + 1] if j + 1 < len(cs) else T + 1               # up to (exclusive) the next completion
        a, b = max(1, min(a, T + 1)), max(1, min(b, T + 1))
        _split(by_slot[c], float(sum(1.0 - terms[t - 1] for t in range(a, b))))
    return [R[e] for e in perm], [tau[e] for e in perm]

