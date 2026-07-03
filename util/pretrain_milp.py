"""
Section 2.1.1 pretraining solver: iterative alternating optimization via traffic fixation.

For each scenario we fix the whole-horizon travel times (from the previous iteration's UE run),
precompute the analytic F1-sensitivity coefficients c_e^k, solve a small MILP for the start-time
schedule, then re-run the UE pipeline to refresh travel times, and repeat until the schedule stops
changing. The resulting schedule's TRUE F is compared against the brute-force oracle optimum.

Problem-2 fix (matches paper eqn:Q_linear shortfall model): the demand recovered by restoring
segment e at slot k GROWS as (1 - rho^{k'-k-d_e+1}) * B[:,e] v_e*  (NOT the old decaying
rho^{k'-k-d_e}). See paper/main.tex sec 2.1.1.

Objective (mu=1, F2 weight 0):   min_y  -sum_{e,k} c_e^k y_e^k
  s.t.  sum_k y_e^k = 1                                       (each segment starts exactly once)
        sum_e sum_{k'=max(1,k-d_e+1)}^{k} y_e^{k'} <= C_MAX   (crew cap at every slot k)
        y_e^k = 0  for k > T - d_e                            (finish within the horizon; the one real guard)
        y_e^k in {0,1}
F2 is still computed (a "dull" value) for logging only; it never enters the objective.

Solver: scipy.optimize.milp (HiGHS) -- ships with scipy, no license needed.

Run inside the road_restore conda env (PYTHONPATH = project root):
  python -m util.pretrain_milp --level-a   # encoding sanity check (MILP >= best work-conserving surrogate)
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
from util.oracle import compute_horizon, scale_dir, select_oracle_instance
from util.scenarios import sample_scenarios

ROOT = Path(__file__).resolve().parent.parent
TOY = ROOT / "data" / "siouxfalls_toy"
OUT = ROOT / "outputs" / "pretrain_milp"


# --------------------------------------------------------------------------- #
# c_e^k precompute  (pure numpy, no UE)
# --------------------------------------------------------------------------- #
def precompute_c(ctx, u_by_slot, durations, segments, T):
    """Analytic F1-improvement coefficients c[j, k-1] for segments[j] starting at slot k, given
    FIXED per-slot travel times u_by_slot (shape (T, |R|)) from the previous iteration.

    c_e^k = (1/T) sum_{k'=k+d_e}^{T} (1 - rho^{k'-k-d_e+1}) * sum_r B[r,e] v_e* alpha_r^{k'},
      alpha_r^{k'} = 1 - baseline_u[r] / u_by_slot[k'-1, r].
    Infeasible starts (k > T - d_e) are left at 0 (the MILP bounds force those y to 0).
    Returns (c, alpha) with c shape (|E|, T), alpha shape (T, |R|)."""
    B = ctx["B"]
    sev = ctx["severity_vec"]
    base_u = ctx["baseline_u"]
    rho = P.RHO
    assert B.shape[1] == len(segments), "B columns must align with segments (edge_id order)"

    alpha = 1.0 - base_u[None, :] / u_by_slot          # (T, |R|); can be <0 if u < baseline (allowed)
    c = np.zeros((len(segments), T))
    for j, e in enumerate(segments):
        d = int(durations[e])
        Bv = B[:, j] * sev[j]                           # (|R|,) demand this segment restores at full
        w = alpha @ Bv                                  # (T,)  w[k'-1] = sum_r alpha_r^{k'} B[r,e] v*
        for k in range(1, T - d + 1):                   # feasible starts only (completion k+d <= T)
            kp = np.arange(k + d, T + 1)                # completion slot .. horizon end
            if kp.size:
                decay = 1.0 - rho ** (kp - k - d + 1)   # (1 - rho^{n+1}),  n = k' - k - d
                c[j, k - 1] = float(decay @ w[kp - 1]) / T
    return c, alpha


# --------------------------------------------------------------------------- #
# MILP build + solve
# --------------------------------------------------------------------------- #
def build_and_solve_milp(c, durations, segments, T, c_max=P.C_MAX):
    """min -sum c[j,k] y[j,k]  s.t. start-once + crew cap + horizon bound; y binary.
    Returns {edge_id: start_slot}."""
    E = len(segments)
    n = E * T

    def idx(j, k):                                      # k is 1-based
        return j * T + (k - 1)

    obj = -c.reshape(-1)                                # minimize -sum c y  ==  maximize sum c y

    ub = np.ones(n)                                     # horizon bound: forbid starts that overrun T
    for j, e in enumerate(segments):
        d = int(durations[e])
        for k in range(T - d + 1, T + 1):
            ub[idx(j, k)] = 0.0
    bounds = Bounds(np.zeros(n), ub)

    A_eq = np.zeros((E, n))                             # start-once: sum_k y[j,k] = 1
    for j in range(E):
        for k in range(1, T + 1):
            A_eq[j, idx(j, k)] = 1.0
    con_eq = LinearConstraint(A_eq, 1.0, 1.0)

    A_ub = np.zeros((T, n))                             # crew cap at each slot
    for k in range(1, T + 1):
        for j, e in enumerate(segments):
            d = int(durations[e])
            for kp in range(max(1, k - d + 1), k + 1):
                A_ub[k - 1, idx(j, kp)] = 1.0
    con_ub = LinearConstraint(A_ub, -np.inf, c_max)

    res = milp(c=obj, constraints=[con_eq, con_ub],
               integrality=np.ones(n), bounds=bounds)
    if not res.success:
        raise RuntimeError(f"MILP failed: {res.message}")

    y = res.x.reshape(E, T)
    return {e: int(np.argmax(y[j])) + 1 for j, e in enumerate(segments)}


def _surrogate_value(start, c, segments, T):
    """sum_e c[j, start_e - 1]  -- the (sign-flipped) MILP objective for a given schedule."""
    return float(sum(c[j, start[e] - 1] for j, e in enumerate(segments) if 1 <= start[e] <= T))


# --------------------------------------------------------------------------- #
# Alternating optimization (Steps 1-4) with guards
# --------------------------------------------------------------------------- #
def alternating_optimize(ctx, durations, segments, T, damping=None):
    """Fix travel times -> solve MILP -> refresh UE, until the schedule stops changing.
    Returns (best_start, best_result, n_iter, converged, trace); best is chosen by TRUE F over ALL
    iterates (so non-convergence is harmless). `trace` is a per-iteration list of dicts
    (iter, F, F1, F2, surrogate, elapsed_s, is_best, start_<e>...) for later analysis.
    `damping` in (0,1] relaxes the travel-time update (MSA): smaller -> smoother, more monotone."""
    damping = P.MILP_DAMPING if damping is None else damping
    t0 = time.perf_counter()

    def _row(it, res, surr, start):
        d = dict(iter=it, F=res["F"], F1=res["F1"], F2=res["F2"], surrogate=surr,
                 elapsed_s=time.perf_counter() - t0)
        d.update({f"start_{e}": start[e] for e in segments})
        return d

    start = schedule_from_permutation(list(segments), durations)      # greedy-first init
    res = evaluate_schedule(start, durations, T, ctx, return_u=True)
    history = [(dict(start), res)]
    trace = [_row(0, res, float("nan"), start)]                       # init: no surrogate yet
    seen = {frozenset(start.items()): 1}                              # count occurrences (relaxed cycle guard)
    u_tilde = res["u_tilde"]
    converged = False
    n_iter = 0
    for _ in range(P.MILP_MAX_ITER):
        n_iter += 1
        c, _ = precompute_c(ctx, u_tilde, durations, segments, T)
        new_start = build_and_solve_milp(c, durations, segments, T)
        new_res = evaluate_schedule(new_start, durations, T, ctx, return_u=True)
        history.append((dict(new_start), new_res))
        trace.append(_row(n_iter, new_res, _surrogate_value(new_start, c, segments, T), new_start))
        if new_start == start:                          # Step 4: schedule unchanged -> converged
            converged = True
            break
        h = frozenset(new_start.items())
        seen[h] = seen.get(h, 0) + 1
        if seen[h] >= P.MILP_CYCLE_TOL:                 # relaxed cycle guard -> stop only after enough repeats
            break
        # MSA-style damped travel-time update: blend toward the new UE so c_e^k (hence the MILP
        # solution) shifts gradually across iterations -> smaller swings, more monotone descent
        u_tilde = damping * new_res["u_tilde"] + (1.0 - damping) * u_tilde
        start = new_start

    best_idx = int(np.argmin([h[1]["F"] for h in history]))           # best by TRUE F
    for i, tr in enumerate(trace):
        tr["is_best"] = (i == best_idx)
    best_start, best_res = history[best_idx]
    return best_start, best_res, n_iter, converged, trace


# --------------------------------------------------------------------------- #
# Full run over M scenarios
# --------------------------------------------------------------------------- #
def run_pretrain_milp(toy_dir=TOY, out_dir=OUT, M=P.M_SCENARIOS, seed=P.SEED):
    out_dir = scale_dir(out_dir)                     # outputs/pretrain_milp/n{N}/ (mirror oracle scale)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)

    disrupted = select_oracle_instance(toy_dir, P.N_DISRUPTED_ORACLE, seed)
    segments = sorted(int(e) for e in disrupted["edge_id"])
    ctx = build_context(toy_dir, disrupted)
    scenarios = sample_scenarios(disrupted, M, seed)
    T = compute_horizon(segments, scenarios)
    print(f"instance: {len(segments)} segments {segments}; M={M}; horizon T={T}")

    # --- resume support: checkpoint per scenario; skip done ones if params (incl. damping) match ---
    from util.oracle import _param_fingerprint
    _, base_fp = _param_fingerprint()
    fp = hashlib.sha1(f"{base_fp}|damp={P.MILP_DAMPING}|maxit={P.MILP_MAX_ITER}|cyc={P.MILP_CYCLE_TOL}".encode()).hexdigest()
    opt_path, trace_path = out_dir / "milp_optima.csv", out_dir / "milp_trace.csv"
    prog_path = out_dir / "milp_progress.json"
    rows, trace_rows, done = [], [], set()
    if opt_path.exists() and trace_path.exists() and prog_path.exists():
        if json.loads(prog_path.read_text(encoding="utf-8")).get("hash") == fp:
            rows = pd.read_csv(opt_path).to_dict("records")
            trace_rows = pd.read_csv(trace_path).to_dict("records")
            done = {int(r["scenario"]) for r in rows}
            print(f"[resume] {len(done)}/{M} MILP scenarios already done; continuing", flush=True)

    t_all = time.perf_counter()
    for m, dur in enumerate(scenarios):
        if m in done:
            continue
        t_s = time.perf_counter()
        best_start, best_res, n_iter, converged, trace = alternating_optimize(ctx, dur, segments, T)
        scen_s = time.perf_counter() - t_s
        row = dict(scenario=m, F_milp=best_res["F"], F1=best_res["F1"], F2=best_res["F2"],
                   n_iter=n_iter, converged=converged, time_s=scen_s, ue_solves=(n_iter + 1) * T,
                   durations="-".join(str(int(dur[e])) for e in segments))
        for e in segments:
            row[f"start_{e}"] = best_start[e]
        rows.append(row)
        for tr in trace:
            trace_rows.append(dict(scenario=m, **tr))
        done.add(m)
        pd.DataFrame(rows).to_csv(opt_path, index=False)                          # checkpoint
        pd.DataFrame(trace_rows).to_csv(trace_path, index=False)                  # checkpoint
        prog_path.write_text(json.dumps({"hash": fp, "done": sorted(done)}), encoding="utf-8")
        print(f"  scenario {m+1}/{M}: F_milp={best_res['F']:.4f}  iters={n_iter}  "
              f"converged={converged}  {scen_s:.1f}s", flush=True)
    total_s = time.perf_counter() - t_all

    milp_opt = pd.DataFrame(rows)
    milp_opt.to_csv(out_dir / "milp_optima.csv", index=False)
    pd.DataFrame(trace_rows).to_csv(out_dir / "milp_trace.csv", index=False)   # rich raw data
    total_ue = int(milp_opt["ue_solves"].sum())
    meta = dict(N=len(segments), M=M, T=T, segments=segments, seed=seed,
                total_time_s=total_s, mean_scenario_time_s=float(milp_opt["time_s"].mean()),
                total_ue_solves=total_ue, s_per_ue=total_s / max(1, total_ue),
                mean_iters=float(milp_opt["n_iter"].mean()),
                mean_F_milp=float(milp_opt["F_milp"].mean()))
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\ntotal MILP wall-clock: {total_s/60:.1f} min  "
          f"({total_ue} UE solves, {meta['s_per_ue']*1000:.0f} ms/UE)")

    from viz.pretrain_viz import make_process_figures
    make_process_figures(out_dir, pd.DataFrame(trace_rows), milp_opt, segments, T)

    oracle_dir = scale_dir(ROOT / "outputs" / "oracle")
    oracle_opt_path = oracle_dir / "oracle_optima.csv"
    if oracle_opt_path.exists():
        from viz.pretrain_viz import make_comparison, make_landscape
        oracle_opt = pd.read_csv(oracle_opt_path)
        merged = milp_opt.merge(oracle_opt[["scenario", "F"]].rename(columns={"F": "F_oracle"}),
                                on="scenario", how="left")
        merged["gap"] = merged["F_milp"] - merged["F_oracle"]
        merged.to_csv(out_dir / "milp_vs_oracle.csv", index=False)
        make_comparison(out_dir, merged, segments, T)
        make_landscape(out_dir, pd.read_csv(oracle_dir / "oracle_landscape.csv"),
                       milp_opt, oracle_opt, segments, T)
        print(f"mean F_milp={milp_opt['F_milp'].mean():.4f}  "
              f"mean F_oracle={merged['F_oracle'].mean():.4f}  "
              f"mean gap={merged['gap'].mean():+.4f}  "
              f"(negative gap = MILP beat the work-conserving oracle)")
    else:
        print("(oracle result for this scale not ready yet; comparison + landscape figures deferred "
              "-- run `python -m util.pretrain_milp --landscape` once the oracle finishes)")
    print(f"Wrote {out_dir}")


# --------------------------------------------------------------------------- #
# Level-A: encoding sanity (no alternating loop). The MILP optimises over a SUPERSET of the
# work-conserving schedules, so its surrogate optimum must be >= the best work-conserving one.
# --------------------------------------------------------------------------- #
def level_a(toy_dir=TOY, seed=P.SEED, scenario=0):
    disrupted = select_oracle_instance(toy_dir, P.N_DISRUPTED_ORACLE, seed)
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
    for k in range(1, T + 1):                           # constraint spot-checks
        active = sum(1 for e in segments if milp_start[e] <= k < milp_start[e] + int(dur[e]))
        assert active <= P.C_MAX, f"crew cap violated at slot {k}: {active} > {P.C_MAX}"
    for e in segments:
        assert milp_start[e] + int(dur[e]) <= T, f"segment {e} overruns horizon T={T}"
    print("  constraints (crew cap, horizon bound) satisfied.")
    return ok


def render_landscape(out_dir=OUT):
    """Regenerate the comparison (01) + cross-scenario landscape (02) figures from already-saved
    CSVs (no UE loop). Run this after the oracle for this scale finishes."""
    out_dir = scale_dir(out_dir)
    oracle_dir = scale_dir(ROOT / "outputs" / "oracle")
    disrupted = select_oracle_instance(TOY, P.N_DISRUPTED_ORACLE, P.SEED)
    segments = sorted(int(e) for e in disrupted["edge_id"])
    scenarios = sample_scenarios(disrupted, P.M_SCENARIOS, P.SEED)
    T = compute_horizon(segments, scenarios)
    land = pd.read_csv(oracle_dir / "oracle_landscape.csv")
    o_opt = pd.read_csv(oracle_dir / "oracle_optima.csv")
    m_opt = pd.read_csv(out_dir / "milp_optima.csv")
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
