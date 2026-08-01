"""
Brute-force oracle for the road-restoration scheduling problem.

An "oracle" here is an exhaustive ground-truth solver: for a deliberately small instance it
enumerates every possible restoration schedule and evaluates each one exactly, so the best it
reports per scenario is the true hindsight optimum with no approximation. It exists as the
reference against which the later learned/pretrained MILP solver is measured.

Concretely, it enumerates all work-conserving schedules of the disrupted road segments and, for
each of M sampled damage scenarios, evaluates the exact objective F(x|ω). A schedule x assigns a
start time to every disrupted segment. A "work-conserving" schedule never leaves a repair crew
idle while repairable work is waiting — for list scheduling this means each segment is started
exactly once and the crews stay busy up to the crew cap. A "scenario" ω is one random draw of the
uncertain repair durations. For every scenario the oracle records both the optimal schedule and
the FULL landscape (the objective value of every schedule tested), so the whole shape of the
search space, not just its minimum, is available for later analysis.

By construction every enumerated schedule is feasible: work-conserving list scheduling never runs
more than C_max crews at once and starts each segment exactly once after it is disrupted; the
global horizon T is chosen large enough that every schedule finishes within it, so all schedules
are scored over one common time window.

Run inside the road_restore conda env:
  python -m util.oracle --probe     # measure seconds per UE solve and project runtime only
  python -m util.oracle             # full enumeration + figures
"""

import hashlib
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import config as P
from util.evaluate import (build_context, evaluate_schedule, makespan_slot,
                           schedule_from_permutation)
from util.io import load_toy_network, od_to_matrix
from util.scenarios import sample_scenarios
from util.ue import solve_ue

ROOT = Path(__file__).resolve().parent.parent
TOY = ROOT / "data" / "siouxfalls_toy"
OUT = ROOT / "outputs" / "oracle"

# The exact horizon walks all N! permutations. 8! = 40320 is still a subsecond sweep, but the
# factorial makes every further segment ~10x dearer, so larger instances switch to the Graham
# bound inside compute_horizon.
HORIZON_ENUM_MAX = 8

# Every parameter that affects the objective F is listed here. A short hashed "fingerprint" of
# these values (see _param_fingerprint) is stored with each cached result and re-checked on the
# next run: the problem SIZE (N_DISRUPTED_ORACLE) alone names the cache folder — one folder per
# scale — while a fingerprint mismatch means the parameters changed and the folder must be
# recomputed rather than reused.
FINGERPRINT_PARAMS = [
    "N_DISRUPTED_ORACLE", "MU", "CAP_RETAIN", "SPEED_RETAIN", "SEVER_SEVERITY",
    "F1_ACTIVE_ONLY", "RHO", "KAPPA", "UPEN_FACTOR", "DELTA_T_H", "C_MAX",
    "M_SCENARIOS", "SEED", "UE_RGAP", "UE_MAX_ITER", "DURATION_SUPPORT", "ETA",
]


def _param_fingerprint():
    """Build the parameter fingerprint. Returns (values, sha1_hex): the dict of every
    objective-affecting parameter together with a SHA-1 digest of it. DURATION_SUPPORT is keyed by
    tuples, which JSON cannot use as object keys, so those keys are stringified first. Hashing the
    JSON with sort_keys=True makes the digest independent of dict ordering, so identical parameter
    values always produce the same fingerprint."""
    values = {}
    for name in FINGERPRINT_PARAMS:
        v = getattr(P, name)
        if isinstance(v, dict):
            v = {str(k): v[k] for k in v}
        values[name] = v
    blob = json.dumps(values, sort_keys=True)
    return values, hashlib.sha1(blob.encode("utf-8")).hexdigest()


def scale_dir(base=OUT, n=None):
    """Return the scale-specific output folder outputs/oracle/n{N}/, where N is the number of
    disrupted segments. Giving each problem size its own folder ensures a run at one scale never
    overwrites the cached result of another."""
    n = P.N_DISRUPTED_ORACLE if n is None else n
    return Path(base) / f"n{n}"


def _baseline_twoway_flow(toy_dir, cores=None):
    """Compute the baseline traffic flow on each undirected edge, used only to rank edges by
    importance when choosing which ones to disrupt.

    The flow comes from a user-equilibrium (UE) solve — the traffic state in which no driver can
    lower their own travel time by unilaterally switching route — run by our own UE engine on the
    UNDAMAGED network under the base origin-destination (OD) demand H0. The two directed volumes on
    an edge are summed into a single undirected flow. Returns {(min(u,v), max(u,v)): flow}.
    `cores` pins the solve's thread pool (None keeps the engine default; the RL solver passes 1
    for bit-reproducibility, see util.ue.solve_ue).

    Computing the flow here rather than reading a shipped reference solution keeps the pipeline
    self-contained: swapping in a different network or OD dataset needs no external reference-flow
    file. A separate self-check inside the UE engine still validates our UE against that reference
    solution, and is left untouched."""
    edges, od, zone_ids = load_toy_network(toy_dir)
    flows, _ = solve_ue(edges, od_to_matrix(od, zone_ids), zone_ids,
                        rgap=P.UE_RGAP, max_iter=P.UE_MAX_ITER, quiet=True, cores=cores)
    f = {}
    fa, ta, vol = (flows["from"].to_numpy(), flows["to"].to_numpy(), flows["volume"].to_numpy())
    for a, b, v in zip(fa, ta, vol):
        key = (min(int(a), int(b)), max(int(a), int(b)))
        f[key] = f.get(key, 0.0) + float(v)
    return f


def select_oracle_instance(toy_dir, n=P.N_DISRUPTED_ORACLE):
    """Choose which n segments to disrupt, ranking edges by importance (their baseline UE flow
    from _baseline_twoway_flow). The selection deliberately mixes heavily used and lightly used
    links so that the ORDER of restoration has a large effect on the objective: if every disrupted
    link mattered equally, any repair order would score about the same and the instance would be a
    weak test. The two highest-flow edges are marked severity 3 (fully severed, the most damaged
    state); the remaining picks are spread across lower-flow edges at severity 2 or 1
    (progressively lighter damage). The choice is deterministic. Writes
    disrupted_segments_oracle{n}.csv and returns the resulting DataFrame."""
    toy = Path(toy_dir)
    edges = pd.read_csv(toy / "network" / "edges.csv")
    flow = _baseline_twoway_flow(toy)
    edges["flow"] = [flow.get((min(int(r.u), int(r.v)), max(int(r.u), int(r.v))), 0.0)
                     for r in edges.itertuples(index=False)]
    ranked = edges.sort_values("flow", ascending=False).reset_index(drop=True)

    n_crit = min(2, n)
    picks = list(range(n_crit))                                  # highest-flow "critical" edges
    rest = n - n_crit
    if rest > 0:                                                 # spread remaining picks over lower-flow edges
        picks += [int(round(x)) for x in np.linspace(len(ranked) // 5, len(ranked) - 1, rest)]
    sub = ranked.iloc[picks].copy().reset_index(drop=True)
    sub["severity"] = [3] * n_crit + [2 if i % 2 == 0 else 1 for i in range(rest)]
    sub["level_id"] = sub["road_class"] + "-S" + sub["severity"].astype(str)
    out = (sub[["edge_id", "u", "v", "road_class", "severity", "level_id"]]
           .sort_values("edge_id").reset_index(drop=True))
    out.to_csv(toy / "disruption" / f"disrupted_segments_oracle{n}.csv", index=False)
    return out


def compute_horizon(segments, scenarios):
    """Return the global time horizon T: the largest completion time — the makespan, i.e. the
    slot at which the last segment finishes — taken over every schedule and every scenario.
    Setting T to this maximum guarantees that every enumerated schedule finishes within the
    horizon, and that all schedules are scored over one identical time window so their objectives
    are directly comparable.

    Up to HORIZON_ENUM_MAX segments this is exact: enumerate every permutation and take the
    largest makespan. Beyond that the N! enumeration is infeasible, so the classical Graham
    list-scheduling bound stands in: with m crews, any work-conserving list schedule completes
    within (sum_e d_e + (m-1) max_e d_e) / m, plus one slot because repairs here start at slot 1
    rather than time zero. The bound is never below the exact horizon, so no schedule is ever
    truncated; the shared scoring window is merely somewhat longer than strictly necessary."""
    if len(segments) > HORIZON_ENUM_MAX:
        m = P.C_MAX
        T = 0
        for dur in scenarios:
            d = [int(dur[e]) for e in segments]
            T = max(T, 1 + -(-(sum(d) + (m - 1) * max(d)) // m))   # 1 + ceil(.../m), integer-exact
        return T
    T = 0
    for perm in itertools.permutations(segments):
        for dur in scenarios:
            T = max(T, makespan_slot(schedule_from_permutation(list(perm), dur), dur))
    return T


def run_oracle(toy_dir=TOY, out_dir=OUT, M=P.M_SCENARIOS, seed=P.SEED, probe=False, force=False):
    out_dir = scale_dir(out_dir)                     # per-scale cache folder outputs/oracle/n{N}/
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)

    disrupted = select_oracle_instance(toy_dir, P.N_DISRUPTED_ORACLE)
    segments = sorted(int(e) for e in disrupted["edge_id"])
    ctx = build_context(toy_dir, disrupted)
    scenarios = sample_scenarios(disrupted, M, seed)
    perms = list(itertools.permutations(segments))
    T = compute_horizon(segments, scenarios)
    print(f"instance: {len(segments)} segments {segments}; perms={len(perms)}; M={M}; horizon T={T} slots")

    # --- cache check: if a saved result for this scale carries the same parameter fingerprint,
    # its enumeration is still valid, so skip the expensive UE work and only re-render figures
    # (unless --force is given) ---
    values, fp = _param_fingerprint()
    meta_path = out_dir / "meta.json"
    if not probe and not force and meta_path.exists():
        cached = json.loads(meta_path.read_text(encoding="utf-8"))
        if cached.get("hash") == fp:
            N = P.N_DISRUPTED_ORACLE
            print(f"[cache] reusing oracle result for N={N} (params unchanged); "
                  f"skipping UE enumeration")
            land = pd.read_csv(out_dir / "oracle_landscape.csv")
            opt = pd.read_csv(out_dir / "oracle_optima.csv")
            from viz.oracle_viz import make_figures
            make_figures(out_dir, land, opt, ctx, segments, scenarios, T, disrupted)
            print(f"Re-rendered figures in {out_dir / 'figures'} (cache hit)")
            return

    # --- time one full schedule evaluation to estimate the per-UE-solve cost, then extrapolate to
    # the whole enumeration; --probe stops here so the runtime can be sized before committing ---
    t0 = time.perf_counter()
    evaluate_schedule(schedule_from_permutation(list(perms[0]), scenarios[0]), scenarios[0], T, ctx)
    dt = time.perf_counter() - t0
    s_ue = dt / T
    total = len(perms) * M
    print(f"one schedule = {T} UE solves took {dt:.2f}s  (~{s_ue*1000:.0f} ms/UE); "
          f"projected full run {total} schedules x {T} ~= {total*T*s_ue/60:.1f} min")
    if probe:
        return

    # --- resume support: if a previous run was interrupted, reload the already-computed scenarios
    # from the on-disk checkpoint (only when its fingerprint matches) and continue from there ---
    land_path = out_dir / "oracle_landscape.csv"
    prog_path = out_dir / "landscape_progress.json"
    rows, done = [], set()
    if not force and land_path.exists() and prog_path.exists():
        prog = json.loads(prog_path.read_text(encoding="utf-8"))
        if prog.get("hash") == fp:
            prev = pd.read_csv(land_path)
            rows = prev.to_dict("records")
            done = {int(s) for s in prev["scenario"].unique()}
            print(f"[resume] {len(done)}/{M} scenarios already computed for N={P.N_DISRUPTED_ORACLE}; "
                  f"continuing with the remaining {M - len(done)}", flush=True)

    # --- full enumeration: evaluate every schedule under every remaining scenario, logging each
    # tested schedule and its objective (with per-schedule timing) into the landscape ---
    t_run = time.perf_counter()
    scen_times = []
    for m, dur in enumerate(scenarios):
        if m in done:
            continue
        t_scen = time.perf_counter()
        for perm in perms:
            start = schedule_from_permutation(list(perm), dur)
            t_ev = time.perf_counter()
            res = evaluate_schedule(start, dur, T, ctx)
            row = dict(scenario=m, perm="-".join(map(str, perm)),
                       F=res["F"], F1=res["F1"], F2=res["F2"],
                       eval_s=time.perf_counter() - t_ev)
            for e in segments:
                row[f"start_{e}"] = start[e]
            rows.append(row)
        scen_times.append(time.perf_counter() - t_scen)
        done.add(m)
        pd.DataFrame(rows).to_csv(land_path, index=False)                       # checkpoint after each scenario
        prog_path.write_text(json.dumps({"hash": fp, "done": sorted(done)}), encoding="utf-8")
        print(f"  scenario {m + 1}/{M} done  ({scen_times[-1] / 60:.1f} min)", flush=True)
    total_time = time.perf_counter() - t_run
    land = pd.DataFrame(rows).sort_values(["scenario", "perm"]).reset_index(drop=True)
    land.to_csv(land_path, index=False)

    opt = land.loc[land.groupby("scenario")["F"].idxmin()].reset_index(drop=True)
    opt.to_csv(out_dir / "oracle_optima.csv", index=False)

    lines = [
        "Brute-force oracle — summary",
        f"segments={segments}  perms={len(perms)}  scenarios={M}  horizon T={T}",
        f"~{s_ue*1000:.0f} ms/UE; total schedules evaluated={len(land)}",
        f"total eval compute = {land['eval_s'].sum()/60:.1f} min "
        f"(~{land['eval_s'].mean()*1000:.0f} ms/schedule); this session {total_time/60:.1f} min",
        f"mean F* over scenarios = {opt['F'].mean():.4f}",
        f"mean F1* = {opt['F1'].mean():.4f}   mean F2* = {opt['F2'].mean():.4f}",
        f"oracle optimum is the min over all {len(perms)} schedules per scenario (true hindsight optimum).",
    ]
    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines))

    from viz.oracle_viz import make_figures
    make_figures(out_dir, land, opt, ctx, segments, scenarios, T, disrupted)

    meta_path.write_text(json.dumps(
        {"hash": fp, "params": values,
         "timing": {"total_eval_s": float(land["eval_s"].sum()), "n_schedules": len(land),
                    "s_per_schedule": float(land["eval_s"].mean()),
                    "s_per_ue": float(land["eval_s"].sum() / (len(land) * T)),
                    "this_session_s": total_time, "scenario_s": scen_times}},
        sort_keys=True, indent=2), encoding="utf-8")
    prog_path.unlink(missing_ok=True)                # run finished: remove the resume checkpoint marker
    print(f"\nWrote {out_dir}  (meta.json hash={fp[:12]}…)")


def render_figs(toy_dir=TOY, out_dir=OUT, M=P.M_SCENARIOS, seed=P.SEED):
    """Re-render the oracle figures from the already-saved landscape and optima CSVs, without
    re-running the expensive enumeration. It only recomputes the lightweight context the plots
    need (the disrupted instance, the scenarios, and the horizon)."""
    from viz.oracle_viz import make_figures
    out_dir = scale_dir(out_dir)
    disrupted = select_oracle_instance(toy_dir, P.N_DISRUPTED_ORACLE)
    segments = sorted(int(e) for e in disrupted["edge_id"])
    ctx = build_context(toy_dir, disrupted)
    scenarios = sample_scenarios(disrupted, M, seed)
    T = compute_horizon(segments, scenarios)
    land = pd.read_csv(out_dir / "oracle_landscape.csv")
    opt = pd.read_csv(out_dir / "oracle_optima.csv")
    make_figures(out_dir, land, opt, ctx, segments, scenarios, T, disrupted)
    print(f"Re-rendered figures in {out_dir / 'figures'}")


if __name__ == "__main__":
    if "--figs" in sys.argv:
        render_figs()
    else:
        run_oracle(probe="--probe" in sys.argv, force="--force" in sys.argv)
