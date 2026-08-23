"""
Pretraining solver for the road-restoration scheduling problem: given a fixed set of damaged road
segments, decide at which time slot repair of each segment should start.

The difficulty is circular: the objective depends on the traffic state, the traffic state depends on
which segments are already open, and that depends on the very schedule we are trying to choose. We
break the loop with ALTERNATING OPTIMIZATION under "traffic fixation" -- hold the traffic state fixed,
optimize the schedule against it, then refresh the traffic state for the new schedule, and iterate.
Concretely, for each demand scenario:
  1. FIX the whole-horizon per-slot travel times taken from the previous iteration's user-equilibrium
     (UE) run. UE is the traffic state in which no driver can lower their own travel time by switching
     route.
  2. With travel times frozen, the marginal accessibility gain of starting each segment at each slot
     becomes a constant, so precompute those analytic sensitivity coefficients c_e^k.
  3. Solve a small mixed-integer linear program (MILP) that picks the best start-time schedule for
     those coefficients.
  4. Re-run the UE pipeline under the new schedule to refresh the travel times, and repeat until the
     schedule stops changing.
Left unstabilized this loop oscillates: repairing a segment decongests exactly the routes that made it
look worth repairing, so the MILP's choice keeps flipping between mutually-best-response schedules (a
congestion-feedback cobweb) instead of settling. Two coupled stabilizers make it converge: a DIMINISHING
travel-time relaxation lambda_n = 1/(n+1) (MILP_DAMPING_MODE == "decay") that drives the travel-time
oscillation to zero, and a decaying proximal penalty gamma_n * ||y - y_prev||_1 on the MILP
(MILP_PROX_SCALE) that discourages needless schedule flips -- the first settles the continuous traffic
state, the second settles the discrete schedule. See alternating_optimize.
The chosen schedule's TRUE objective F -- re-evaluated with a real UE solve rather than the linearized
surrogate -- is then compared against a brute-force oracle that exhaustively searches all
work-conserving schedules (schedules that never leave a crew idle while repairs remain) to give the
reference optimum.

Recovery model: the demand a segment carries returns GRADUALLY after the segment is restored, not all
at once. The fraction recovered by a later slot k' after starting repair at slot k grows as
(1 - rho^{k'-k-d_e+1}), where rho is the recovery-inertia rate and d_e the repair duration, so the
gain accumulates toward the segment's full demand B[:,e] v_e* as time passes.

MILP objective (all weight on accessibility; the efficiency term carries zero weight here):
      min_y  -sum_{e,k} c_e^k y_e^k                            (equivalently: maximize total gain)
  s.t.  sum_k y_e^k = 1                                        (each segment starts exactly once)
        sum_e sum_{k'=max(1,k-d_e+1)}^{k} y_e^{k'} <= C_MAX    (crew cap: at most C_MAX active at slot k)
        y_e^k = 0  for k > T - d_e                             (repair must finish within horizon T)
        y_e^k in {0,1}
The efficiency term F2 (pure schedule arithmetic, unaffected by the linearization) is still computed
for logging but never enters the MILP objective.

Solver: scipy.optimize.milp (the HiGHS backend) -- ships with scipy, so no external license is needed.

Outputs land in outputs/pretrain_milp/n{N}/: milp_optima.csv (the delivered order scored on every
scenario), milp_trace.csv (the alternating run's own trajectory), milp_slots.csv (the per-slot
accessibility of those evaluations, so a recovery curve needs no extra UE solve) and run_meta.json
(instance, objective and solver parameters, delivered order, stopping condition, code revision).

Run inside the road_restore conda env (PYTHONPATH = project root):
  python -m util.pretrain_milp --level-a   # encoding sanity check (MILP surrogate >= best work-conserving one)
  python -m util.pretrain_milp             # full alternating run over M scenarios + comparison figure
"""

import hashlib
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp

import config as P
from util.evaluate import (build_context, evaluate_schedule,
                           schedule_from_permutation)
from util.provenance import (solver_dir, fresh_scale_dir, log_dir, results_dir, slot_rows,
                             write_run_meta)
from util.oracle import (_baseline_twoway_flow, compute_horizon, scale_dir,
                         select_oracle_instance)
from util.scenarios import nominal_durations, sample_scenarios

ROOT = Path(__file__).resolve().parent.parent
TOY = ROOT / "data" / "siouxfalls_toy"
OUT = ROOT / "outputs" / "02-baselines" / solver_dir("pretrain_milp")


# --------------------------------------------------------------------------- #
# c_e^k precompute  (pure numpy, no UE)
# --------------------------------------------------------------------------- #
def precompute_c(ctx, u_by_slot, durations, segments, T):
    """Precompute the analytic accessibility-gain coefficients c[j, k-1]: the improvement in the F1
    (accessibility) objective from starting repair of segments[j] at slot k, GIVEN the per-slot travel
    times u_by_slot (shape (T, |R|), one row of OD travel times per slot) held FIXED from the previous
    iteration. Freezing the travel times turns this gain into a constant, which is what lets the MILP
    treat it as a plain linear coefficient instead of re-solving traffic inside the loop.

    c_e^k = (1/T) sum_{k'=k+d_e}^{T} (1 - rho^{k'-k-d_e+1}) * sum_r B[r,e] v_e* alpha_r^{k'}, where
    alpha_r^{k'} = 1 - baseline_u[r] / u_by_slot[k'-1, r] measures how much worse route r's travel time
    is at slot k' than its undisrupted baseline (its "room for improvement"), the (1 - rho^...) factor
    is the fraction of segment e's demand recovered by slot k', and the sum runs only over slots at or
    after completion (k + d_e). Infeasible starts (k > T - d_e, which would overrun the horizon) are
    left at 0; the MILP bounds separately force those decisions to 0.
    Returns (c, alpha) with c of shape (|E|, T) and alpha of shape (T, |R|)."""
    B = ctx["B"]
    sev = ctx["severity_vec"]
    base_u = ctx["baseline_u"]
    rho = P.RHO
    assert B.shape[1] == len(segments), "B columns must align with segments (edge_id order)"

    alpha = 1.0 - base_u[None, :] / u_by_slot          # (T, |R|); negative where a slot beat baseline -- legal, not clipped
    c = np.zeros((len(segments), T))
    for j, e in enumerate(segments):
        d = int(durations[e])
        Bv = B[:, j] * sev[j]                           # (|R|,): per-route demand fully served once segment j is restored
        w = alpha @ Bv                                  # (T,): accessibility gain per slot if j were fully open, w[k'-1] = sum_r alpha_r^{k'} B[r,e] v*
        for k in range(1, T - d + 1):                   # feasible starts only (completion k+d <= T)
            kp = np.arange(k + d, T + 1)                # slots from completion through the horizon end
            if kp.size:
                decay = 1.0 - rho ** (kp - k - d + 1)   # recovery fraction (1 - rho^{n+1}), n = k' - k - d slots since completion
                c[j, k - 1] = float(decay @ w[kp - 1]) / T
    return c, alpha


# --------------------------------------------------------------------------- #
# MILP build + solve
# --------------------------------------------------------------------------- #
def build_and_solve_milp(c, durations, segments, T, c_max=P.C_MAX, y_prev=None, gamma=0.0):
    """Build and solve the start-time MILP for the fixed coefficients c: choose exactly one start slot
    per segment to maximize total accessibility gain, encoded as minimizing -sum c[j,k] y[j,k], subject
    to the start-once, crew-capacity, and horizon-bound constraints with y binary. Returns the chosen
    schedule as {edge_id: start_slot}.

    When y_prev (the previous iteration's schedule) and gamma > 0 are given, a proximal penalty
    gamma * ||y - y_prev||_1 is added to the objective. Under the start-once constraint this L1 distance
    equals 2 * (number of segments that changed start slot), so it stays linear; it is applied by
    rewarding each segment for keeping its previous start slot (coefficient -2*gamma there)."""
    E = len(segments)
    n = E * T

    def idx(j, k):                                      # flatten (segment j, 1-based slot k) to a decision-vector index
        return j * T + (k - 1)

    obj = (-c.reshape(-1)).astype(float).copy()         # minimize -sum c y  ==  maximize sum c y
    if y_prev is not None and gamma > 0:                # proximal penalty: reward staying at the previous start slot
        for j, e in enumerate(segments):
            kp = int(y_prev[e])
            if 1 <= kp <= T:
                obj[idx(j, kp)] += -2.0 * gamma

    ub = np.ones(n)                                     # horizon bound: force to 0 any start whose repair would overrun T
    for j, e in enumerate(segments):
        d = int(durations[e])
        for k in range(T - d + 1, T + 1):
            ub[idx(j, k)] = 0.0
    bounds = Bounds(np.zeros(n), ub)

    A_eq = np.zeros((E, n))                             # start-once: each segment gets exactly one start slot, sum_k y[j,k] = 1
    for j in range(E):
        for k in range(1, T + 1):
            A_eq[j, idx(j, k)] = 1.0
    con_eq = LinearConstraint(A_eq, 1.0, 1.0)

    A_ub = np.zeros((T, n))                             # crew cap: at slot k count repairs still in progress (started within the last d_e-1 slots)
    for k in range(1, T + 1):
        for j, e in enumerate(segments):
            d = int(durations[e])
            for kp in range(max(1, k - d + 1), k + 1):
                A_ub[k - 1, idx(j, kp)] = 1.0
    con_ub = LinearConstraint(A_ub, -np.inf, c_max)     # ... and cap the simultaneous count at c_max crews

    res = milp(c=obj, constraints=[con_eq, con_ub],
               integrality=np.ones(n), bounds=bounds)
    if not res.success:
        raise RuntimeError(f"MILP failed: {res.message}")

    y = res.x.reshape(E, T)
    return {e: int(np.argmax(y[j])) + 1 for j, e in enumerate(segments)}


def _surrogate_value(start, c, segments, T):
    """Evaluate the linearized surrogate objective sum_e c[j, start_e - 1] for a concrete schedule --
    i.e. the value the MILP is maximizing, computed here for an arbitrary set of start slots so that
    candidate schedules can be scored on the same footing as the MILP's own solution."""
    return float(sum(c[j, start[e] - 1] for j, e in enumerate(segments) if 1 <= start[e] <= T))


def _prox_gamma0(c, durations, segments, T):
    """Base strength of the proximal schedule penalty: MILP_PROX_SCALE times the median, over segments,
    of the across-slot spread (peak-to-peak) of the coefficients c on that segment's feasible start slots
    (k = 1..T-d). Putting gamma0 on the same scale as the gains the MILP trades off makes the penalty
    bite consistently. Returns 0 when the penalty is disabled (MILP_PROX_SCALE == 0)."""
    if P.MILP_PROX_SCALE <= 0:
        return 0.0
    spreads = [float(np.ptp(c[j, : max(1, T - int(durations[e]))])) for j, e in enumerate(segments)]
    return P.MILP_PROX_SCALE * float(np.median(spreads)) if spreads else 0.0


# --------------------------------------------------------------------------- #
# Alternating optimization (Steps 1-4) with guards
# --------------------------------------------------------------------------- #
def alternating_optimize(ctx, durations, segments, T, damping=None, warm_order=None):
    """Run the alternating loop for one scenario: fix travel times -> solve the MILP -> refresh the UE
    travel times, repeating until the schedule stops changing (or a cycle/iteration guard fires).
    Returns (best_start, best_result, n_iter, converged, outcome, trace). The returned best is the
    iterate with the lowest TRUE objective F across ALL iterations, not just the final one, so even a
    loop that never settles still yields the best schedule it visited. When `warm_order` (a segment
    priority order) is given, iteration 0 is seeded from that order's work-conserving schedule instead
    of the edge-id packing order; combined with the best-by-true-F return this makes the result provably
    no worse than the schedule that order encodes (see MILP_WARM_START). `outcome` is one of "fixed_point"
    (the schedule stopped changing), "cycle" (a schedule recurred MILP_CYCLE_TOL times), or "iter_cap"
    (hit MILP_MAX_ITER); `converged` is True only for "fixed_point". `trace` is a per-iteration list of
    dicts (iter, F, F1, F2, surrogate, elapsed_s, is_best, start_<e>...) kept for later analysis.

    Two coupled stabilizers keep the loop from oscillating between mutually-best-response schedules
    (each schedule's induced traffic makes a different schedule look optimal, a congestion-feedback
    cobweb). (1) The travel-time relaxation: under MILP_DAMPING_MODE == "decay" it uses a DIMINISHING
    step lambda_n = 1/(n+1), whose vanishing size drives the travel-time oscillation to zero (a constant
    step only shrinks it and leaves a persistent cycle). (2) A decaying proximal penalty
    gamma_n * ||y - y_prev||_1 on the MILP (strength MILP_PROX_SCALE, gamma_n = gamma0/(n+1)) that
    discourages needless schedule flips. The damping settles the continuous traffic state; the penalty
    settles the discrete schedule; together they take a loop that used to cycle to a fixed point. The
    `damping` argument overrides MILP_DAMPING only in the "const" mode (kept for experiments)."""
    t0 = time.perf_counter()

    def _row(it, res, surr, start):
        d = dict(iter=it, F=res["F"], F1=res["F1"], F2=res["F2"], surrogate=surr,
                 elapsed_s=time.perf_counter() - t0)
        d.update({f"start_{e}": start[e] for e in segments})
        return d

    seed_order = list(warm_order) if warm_order is not None else list(segments)  # warm start: flow-greedy order if given, else edge-id packing order
    start = schedule_from_permutation(seed_order, durations)           # initial (iteration 0) work-conserving schedule
    res = evaluate_schedule(start, durations, T, ctx, return_u=True)
    history = [(dict(start), res)]
    trace = [_row(0, res, float("nan"), start)]                       # iteration 0 is the initial evaluation, before any MILP, so it has no surrogate value
    seen = {frozenset(start.items()): 1}                              # per-schedule occurrence counts, feeding the relaxed cycle guard below
    u_tilde = res["u_tilde"]
    # Base proximal strength gamma0, fixed from the first iteration's coefficient spread (see _prox_gamma0).
    c0, _ = precompute_c(ctx, u_tilde, durations, segments, T)
    gamma0 = _prox_gamma0(c0, durations, segments, T)
    converged = False
    outcome = "iter_cap"
    n_iter = 0
    for _ in range(P.MILP_MAX_ITER):
        n_iter += 1
        c, _ = precompute_c(ctx, u_tilde, durations, segments, T)
        new_start = build_and_solve_milp(c, durations, segments, T,       # decaying proximal penalty toward the previous schedule
                                         y_prev=start, gamma=gamma0 / (n_iter + 1))
        new_res = evaluate_schedule(new_start, durations, T, ctx, return_u=True)
        history.append((dict(new_start), new_res))
        trace.append(_row(n_iter, new_res, _surrogate_value(new_start, c, segments, T), new_start))
        if new_start == start:                          # schedule identical to last iteration -> fixed point reached, converged
            converged, outcome = True, "fixed_point"
            break
        h = frozenset(new_start.items())
        seen[h] = seen.get(h, 0) + 1
        if seen[h] >= P.MILP_CYCLE_TOL:                 # cycle guard: the loop is oscillating, so stop once one schedule has recurred enough times
            outcome = "cycle"
            break
        # Travel-time relaxation. "decay" uses a diminishing step lambda_n = 1/(n+1), whose vanishing size
        # drives the oscillation of u to zero; "const" keeps the earlier fixed-weight blend.
        lam = 1.0 / (n_iter + 1) if P.MILP_DAMPING_MODE == "decay" \
            else (P.MILP_DAMPING if damping is None else damping)
        u_tilde = lam * new_res["u_tilde"] + (1.0 - lam) * u_tilde
        start = new_start

    best_idx = int(np.argmin([h[1]["F"] for h in history]))           # pick the iterate with the lowest TRUE objective F
    for i, tr in enumerate(trace):
        tr["is_best"] = (i == best_idx)
    best_start, best_res = history[best_idx]
    return best_start, best_res, n_iter, converged, outcome, trace


# --------------------------------------------------------------------------- #
# Full run over M scenarios
# --------------------------------------------------------------------------- #
def run_pretrain_milp(toy_dir=TOY, out_dir=OUT, M=P.M_SCENARIOS, seed=P.SEED):
    """Solve ONCE on a nominal instance, then evaluate the resulting priority order on all M
    sampled scenarios.

    The alternating solver is run against a single set of NOMINAL durations, each segment's
    expected repair time under the full duration model rounded to whole slots (see
    _nominal_durations). Its solution is then projected onto the repair ORDER it implies, and that
    one order is scheduled and scored separately on every sampled scenario, exactly as the static
    rankers are. The reported mean is therefore the expected performance of a single decision that
    is committed before any duration is observed.

    This replaces the earlier arrangement, in which the solver was re-run per scenario with that
    scenario's durations already known. That gave M different schedules, each optimized with
    hindsight, so its mean was an upper bound on what any implementable rule could achieve rather
    than the value of a rule. It also cost M alternating runs instead of one.

    Two things are given up by the projection. The MILP may place a segment later than a
    work-conserving schedule would, so reading only the order discards any deliberate idling it
    chose, which is where the earlier negative gaps against the work-conserving oracle came from.
    And the nominal instance is solved as if durations were certain, so the coefficients it
    linearizes around are those of an average world, not of any world that actually occurs.
    """
    out_dir = scale_dir(out_dir)                     # outputs/pretrain_milp/n{N}/, mirroring the oracle's per-scale directory layout
    out_dir.mkdir(parents=True, exist_ok=True)

    disrupted = select_oracle_instance(toy_dir, P.N_DISRUPTED_ORACLE)
    segments = sorted(int(e) for e in disrupted["edge_id"])
    ctx = build_context(toy_dir, disrupted)
    scenarios = sample_scenarios(disrupted, M, seed)
    T = compute_horizon(segments, scenarios)
    nominal = nominal_durations(disrupted, segments)
    print(f"instance: {len(segments)} segments {segments}; M={M}; horizon T={T}")
    print(f"nominal durations (expected, rounded): "
          f"{ {e: nominal[e] for e in segments} }", flush=True)

    # Warm start: order the segments by their baseline UE two-way flow (edge criticality); this is the
    # flow-greedy baseline's priority order, and seeding iteration 0 from it makes each scenario's result
    # provably no worse than that baseline (see alternating_optimize / MILP_WARM_START).
    warm_order = None
    if P.MILP_WARM_START:
        flow = _baseline_twoway_flow(toy_dir)
        fscore = {int(eid): flow.get((min(u, v), max(u, v)), 0.0) for (eid, u, v, s) in ctx["disrupted"]}
        warm_order = sorted(segments, key=lambda e: (-fscore[e], e))
        print(f"warm start from flow-greedy order: {warm_order}", flush=True)

    # --- resume support: a scenario's result is reusable only if the parameters that produced it match ---
    # fp is a fingerprint (a short hash of the solver settings). We stash it beside the checkpoints and,
    # on restart, reuse a scenario's saved result only when its fingerprint still matches the current run.
    from util.oracle import _param_fingerprint
    _, base_fp = _param_fingerprint()
    # "|nominal=1" marks the once-on-a-nominal-instance semantics: results written by the earlier
    # per-scenario arrangement carry a fingerprint without it and are therefore never mistaken for
    # reusable, even though every individual solver setting matches.
    fp = hashlib.sha1(f"{base_fp}|mode={P.MILP_DAMPING_MODE}|damp={P.MILP_DAMPING}|prox={P.MILP_PROX_SCALE}"
                      f"|maxit={P.MILP_MAX_ITER}|cyc={P.MILP_CYCLE_TOL}|warm={P.MILP_WARM_START}"
                      f"|nominal=1".encode()).hexdigest()
    opt_path = results_dir(out_dir) / "milp_optima.csv"
    trace_path = log_dir(out_dir) / "milp_trace.csv"
    prog_path = log_dir(out_dir) / "milp_progress.json"
    # There is a single alternating solve now, so there is nothing partial to resume into: either
    # the saved result was produced by these settings and is reusable whole (in which case the
    # solve and the evaluations are skipped and only the figures are re-rendered), or it is
    # discarded and everything below runs fresh.
    rows, trace_rows = [], []
    cached = False
    if opt_path.exists() and trace_path.exists() and prog_path.exists():
        if json.loads(prog_path.read_text(encoding="utf-8")).get("hash") == fp:
            saved = pd.read_csv(opt_path)
            if len(saved) == M:
                print("[cache] MILP result for these settings is already complete; reusing it "
                      "and re-rendering figures only", flush=True)
                rows = saved.to_dict("records")
                trace_rows = pd.read_csv(trace_path).to_dict("records")
                cached = True

    t_all = time.perf_counter()
    if cached:
        return _finish_pretrain_milp(out_dir, rows, trace_rows, segments, T, M, seed,
                                     time.perf_counter() - t_all)
    # Running fresh: clear the folder first so no residue of a previous run survives beside the
    # new results. Done only AFTER the cache check -- clearing first would defeat the reuse.
    fresh_scale_dir(out_dir)
    # --- one alternating solve, on the nominal instance ---
    t_solve = time.perf_counter()
    best_start, nominal_res, n_iter, converged, outcome, trace = alternating_optimize(
        ctx, nominal, segments, T, warm_order=warm_order)
    solve_s = time.perf_counter() - t_solve
    # Project the schedule onto the priority order it implies: earliest start first, ties broken by
    # edge id. The order is what survives into every scenario, because the start slots themselves
    # are only valid for the nominal durations the solver assumed.
    order = sorted(segments, key=lambda e: (best_start[e], e))
    print(f"  nominal solve: F(nominal)={nominal_res['F']:.4f}  iters={n_iter}  outcome={outcome}  "
          f"{solve_s:.1f}s\n  projected order: {'-'.join(map(str, order))}", flush=True)

    # --- evaluate that one order on every sampled scenario ---
    slots = []
    solve_ue = (n_iter + 1) * T                      # UE solves the nominal alternating run cost
    for m, dur in enumerate(scenarios):
        t_s = time.perf_counter()
        start = schedule_from_permutation(order, dur)
        res = evaluate_schedule(start, dur, T, ctx, collect_traces=True)
        slots.extend(slot_rows(m, res))                 # recovery curve, at no extra UE cost
        row = dict(scenario=m, F_milp=res["F"], F1=res["F1"], F2=res["F2"],
                   n_iter=n_iter, converged=converged, outcome=outcome,
                   time_s=time.perf_counter() - t_s,
                   # ue_solves keeps the nominal solve's own cost, so the (n_iter + 1) * T identity
                   # that util/compare.py uses to recover T from this file still holds. ue_total is
                   # the honest per-scenario compute: the one-off solve shared across the M
                   # scenarios it serves, plus this scenario's single evaluation.
                   ue_solves=solve_ue, ue_total=solve_ue / M + T,
                   order="-".join(map(str, order)),
                   durations="-".join(str(int(dur[e])) for e in segments))
        for e in segments:
            row[f"start_{e}"] = start[e]
        rows.append(row)
        print(f"  scenario {m+1}/{M}: F_milp={res['F']:.4f}", flush=True)
    for tr in trace:                                 # one trajectory now, that of the nominal solve
        trace_rows.append(dict(scenario=0, **tr))
    pd.DataFrame(rows).to_csv(opt_path, index=False)
    pd.DataFrame(slots).to_csv(log_dir(out_dir) / "milp_slots.csv", index=False)
    pd.DataFrame(trace_rows).to_csv(trace_path, index=False)
    prog_path.write_text(json.dumps({"hash": fp, "done": sorted(range(M))}), encoding="utf-8")
    return _finish_pretrain_milp(out_dir, rows, trace_rows, segments, T, M, seed,
                                 time.perf_counter() - t_all)


def _finish_pretrain_milp(out_dir, rows, trace_rows, segments, T, M, seed, total_s):
    """Shared tail of run_pretrain_milp: metadata, the process figures, and (where the oracle
    exists at this scale) the gap comparison. Runs identically whether the rows were just computed
    or reloaded whole from a fingerprint-matched cache, so a cached rerun re-renders every figure
    without re-solving anything."""
    milp_opt = pd.DataFrame(rows)
    total_ue = int(milp_opt["ue_solves"].sum())
    (results_dir(out_dir) / "milp_solution_best.json").write_text(json.dumps(dict(
        order=([int(x) for x in str(milp_opt["order"].iloc[0]).split("-")]
               if "order" in milp_opt.columns else None),
        mean_F=float(milp_opt["F_milp"].mean()),
        outcome=(str(milp_opt["outcome"].iloc[0]) if "outcome" in milp_opt.columns else None)),
        indent=2), encoding="utf-8")
    meta = write_run_meta(
        out_dir, method="milp", segments=segments, T=T, seed=seed, M=M,
        solver=dict(damping_mode=P.MILP_DAMPING_MODE, damping=P.MILP_DAMPING,
                    prox_scale=P.MILP_PROX_SCALE, max_iter=P.MILP_MAX_ITER,
                    cycle_tol=P.MILP_CYCLE_TOL, warm_start=P.MILP_WARM_START),
        # Columns added over the life of the code: a cached rerun may be finishing an optima file
        # written before they existed, and re-solving just to fill in metadata would defeat the
        # cache. Report what the file actually has.
        order=str(milp_opt["order"].iloc[0]) if "order" in milp_opt.columns else None,
        outcome=str(milp_opt["outcome"].iloc[0]) if "outcome" in milp_opt.columns else None,
        mean_iters=float(milp_opt["n_iter"].mean()),
        total_time_s=total_s, mean_scenario_time_s=float(milp_opt["time_s"].mean()),
        total_ue_solves=total_ue, s_per_ue=total_s / max(1, total_ue),
        mean_F_milp=float(milp_opt["F_milp"].mean()))
    print(f"\ntotal MILP wall-clock: {total_s/60:.1f} min  "
          f"({total_ue} UE solves, {meta['s_per_ue']*1000:.0f} ms/UE)")

    from viz.pretrain_viz import make_process_figures
    make_process_figures(out_dir, pd.DataFrame(trace_rows), milp_opt, segments, T)

    oracle_dir = scale_dir(ROOT / "outputs" / "02-baselines" / solver_dir("brute-force"))
    oracle_opt_path = results_dir(oracle_dir) / "oracle_optima.csv"
    if oracle_opt_path.exists():
        from viz.pretrain_viz import make_comparison, make_landscape
        oracle_opt = pd.read_csv(oracle_opt_path)
        merged = milp_opt.merge(oracle_opt[["scenario", "F"]].rename(columns={"F": "F_oracle"}),
                                on="scenario", how="left")
        merged["gap"] = merged["F_milp"] - merged["F_oracle"]
        merged.to_csv(out_dir / "milp_vs_oracle.csv", index=False)
        make_comparison(out_dir, merged, segments, T)
        make_landscape(out_dir, pd.read_csv(log_dir(oracle_dir) / "oracle_landscape.csv"),
                       milp_opt, oracle_opt, segments, T)
        print(f"mean F_milp={milp_opt['F_milp'].mean():.4f}  "
              f"mean F_oracle={merged['F_oracle'].mean():.4f}  "
              f"mean gap={merged['gap'].mean():+.4f}  "
              f"(negative gap = MILP beat the work-conserving oracle)")
    else:
        print("(oracle result for this scale not ready yet; comparison + landscape figures deferred "
              "-- run `python -m util.pretrain_milp --landscape` once the oracle finishes)")
    print(f"Wrote {out_dir}")
    from util.compare import refresh_comparison
    refresh_comparison()                     # every solver run leaves the comparison current


# --------------------------------------------------------------------------- #
# Level-A: encoding sanity check (no alternating loop). The MILP is free to choose any start slot per
# segment, which is a SUPERSET of the work-conserving schedules the brute-force enumeration considers,
# so the MILP's surrogate optimum must be at least as good as the best work-conserving one. If it ever
# came out lower, the constraint encoding would be wrong.
# --------------------------------------------------------------------------- #
def level_a(toy_dir=TOY, seed=P.SEED, scenario=0):
    # Verify the MILP encoding on a single scenario without running the alternating loop: with the
    # coefficients fixed, enumerate every work-conserving schedule, take the best surrogate value, and
    # confirm the MILP's surrogate is at least that high while satisfying the crew-cap and horizon
    # constraints. Returns True on pass. This catches constraint-encoding bugs, not solution quality.
    disrupted = select_oracle_instance(toy_dir, P.N_DISRUPTED_ORACLE)
    segments = sorted(int(e) for e in disrupted["edge_id"])
    ctx = build_context(toy_dir, disrupted)
    scenarios = sample_scenarios(disrupted, P.M_SCENARIOS, seed)
    dur = scenarios[scenario]
    T = compute_horizon(segments, scenarios)

    start0 = schedule_from_permutation(list(segments), dur)
    u_tilde = evaluate_schedule(start0, dur, T, ctx, return_u=True)["u_tilde"]
    c, alpha = precompute_c(ctx, u_tilde, dur, segments, T)

    best_wc, best_perm = -np.inf, None
    for perm in itertools.permutations(segments):
        s = schedule_from_permutation(list(perm), dur)
        v = _surrogate_value(s, c, segments, T)
        if v > best_wc:
            best_wc, best_perm = v, s

    milp_start = build_and_solve_milp(c, dur, segments, T)
    milp_val = _surrogate_value(milp_start, c, segments, T)

    print(f"Level-A encoding check (scenario {scenario}, T={T}, segments={segments})")
    print(f"  durations = {dict(dur)}")
    print(f"  best work-conserving surrogate = {best_wc:.6f}  at {best_perm}")
    print(f"  MILP surrogate                 = {milp_val:.6f}  at {milp_start}")
    neg = float((alpha < 0).mean())
    if neg > 0:
        print(f"  note: {neg:.1%} of alpha entries are negative (u < baseline; legal, not an error)")
    ok = milp_val >= best_wc - 1e-9
    print(f"  MILP >= best work-conserving ?  {ok}   -> {'PASS' if ok else 'FAIL (encoding bug)'}")
    for k in range(1, T + 1):                           # independently re-check that the MILP schedule honors each constraint
        active = sum(1 for e in segments if milp_start[e] <= k < milp_start[e] + int(dur[e]))
        assert active <= P.C_MAX, f"crew cap violated at slot {k}: {active} > {P.C_MAX}"
    for e in segments:
        assert milp_start[e] + int(dur[e]) <= T, f"segment {e} overruns horizon T={T}"
    print("  constraints (crew cap, horizon bound) satisfied.")
    return ok


def render_landscape(out_dir=OUT):
    """Regenerate the MILP-vs-oracle comparison figure and the cross-scenario landscape figure from the
    already-saved CSVs; only the instance selection re-solves one baseline UE, and no per-slot evaluation
    UE is re-run. Run this once the oracle for this scale has an oracle optimum to plot against."""
    out_dir = scale_dir(out_dir)
    oracle_dir = scale_dir(ROOT / "outputs" / "02-baselines" / solver_dir("brute-force"))
    disrupted = select_oracle_instance(TOY, P.N_DISRUPTED_ORACLE)
    segments = sorted(int(e) for e in disrupted["edge_id"])
    scenarios = sample_scenarios(disrupted, P.M_SCENARIOS, P.SEED)
    T = compute_horizon(segments, scenarios)
    land = pd.read_csv(log_dir(oracle_dir) / "oracle_landscape.csv")
    o_opt = pd.read_csv(results_dir(oracle_dir) / "oracle_optima.csv")
    m_opt = pd.read_csv(results_dir(out_dir) / "milp_optima.csv")
    from viz.pretrain_viz import make_comparison, make_landscape
    merged = m_opt.merge(o_opt[["scenario", "F"]].rename(columns={"F": "F_oracle"}),
                         on="scenario", how="left")
    merged["gap"] = merged["F_milp"] - merged["F_oracle"]
    merged.to_csv(out_dir / "milp_vs_oracle.csv", index=False)
    print(f"wrote {make_comparison(out_dir, merged, segments, T)}")
    print(f"wrote {make_landscape(out_dir, land, m_opt, o_opt, segments, T)}")


if __name__ == "__main__":
    if "--level-a" in sys.argv:
        level_a()
    elif "--landscape" in sys.argv:
        render_landscape()
    else:
        run_pretrain_milp()
