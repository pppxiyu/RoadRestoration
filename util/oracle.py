"""
Brute-force oracle: enumerate all work-conserving schedules (permutations of the 5 disrupted
segments) over M scenarios, evaluate the EXACT objective F(x|ω) for each (Figure 1), and record
the per-scenario optimum + the FULL landscape (every tested x). This is the ground truth the
later pretraining MILP will be tested against.

By construction every enumerated x satisfies the §1.5 constraints: work-conserving list
scheduling never runs more than C_max crews and starts each segment exactly once (start>t0);
the global horizon T is set so every schedule finishes within it.

Run inside the road_restore conda env:
  python -m util.oracle --probe     # measure s/UE + project runtime only
  python -m util.oracle             # full run + figures
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
from util.scenarios import sample_scenarios

ROOT = Path(__file__).resolve().parent.parent
TOY = ROOT / "data" / "siouxfalls_toy"
OUT = ROOT / "outputs" / "oracle"

# Parameters whose change should REFRESH a cached oracle result of the same scale.
# Only the problem SIZE (N_DISRUPTED_ORACLE) keys the cache folder; these values key its freshness.
FINGERPRINT_PARAMS = [
    "N_DISRUPTED_ORACLE", "MU", "CAP_RETAIN", "SPEED_RETAIN", "SEVER_SEVERITY",
    "F1_ACTIVE_ONLY", "RHO", "KAPPA", "UPEN_FACTOR", "DELTA_T_H", "C_MAX",
    "M_SCENARIOS", "SEED", "UE_RGAP", "UE_MAX_ITER", "DURATION_SUPPORT", "ETA",
]


def _param_fingerprint():
    """Return (values, sha1) for the F-affecting params. DURATION_SUPPORT has tuple keys, so
    stringify keys for stable JSON; sha1 over json.dumps(..., sort_keys=True)."""
    values = {}
    for name in FINGERPRINT_PARAMS:
        v = getattr(P, name)
        if isinstance(v, dict):
            v = {str(k): v[k] for k in v}
        values[name] = v
    blob = json.dumps(values, sort_keys=True)
    return values, hashlib.sha1(blob.encode("utf-8")).hexdigest()


def scale_dir(base=OUT, n=None):
    """Scale-specific subfolder outputs/oracle/n{N}/ (never overwrites a different scale)."""
    n = P.N_DISRUPTED_ORACLE if n is None else n
    return Path(base) / f"n{n}"


def _reference_twoway_flow(toy):
    """Baseline two-way UE flow per undirected edge, from raw/SiouxFalls_flow.tntp (sum of both
    directed volumes). Returns {(min(u,v), max(u,v)): flow}."""
    f = {}
    for line in (toy / "raw" / "SiouxFalls_flow.tntp").read_text().splitlines():
        s = line.strip()
        if not s or s.lower().startswith("from"):
            continue
        t = s.rstrip(";").split()
        if len(t) < 3:
            continue
        try:
            a, b, vol = int(t[0]), int(t[1]), float(t[2])
        except ValueError:
            continue
        key = (min(a, b), max(a, b))
        f[key] = f.get(key, 0.0) + vol
    return f


def select_oracle_instance(toy_dir, n=P.N_DISRUPTED_ORACLE):
    """Choose n disrupted segments by IMPORTANCE (baseline two-way UE flow), mixing critical and
    minor links so the restoration order strongly affects F1: the top-2 flow edges get severity 3
    (severed), the rest are spread over lower-flow edges at severity 2/1. Deterministic. Saves
    disruption/disrupted_segments_oracle{n}.csv and returns the DataFrame."""
    toy = Path(toy_dir)
    edges = pd.read_csv(toy / "network" / "edges.csv")
    flow = _reference_twoway_flow(toy)
    edges["flow"] = [flow.get((min(int(r.u), int(r.v)), max(int(r.u), int(r.v))), 0.0)
                     for r in edges.itertuples(index=False)]
    ranked = edges.sort_values("flow", ascending=False).reset_index(drop=True)

    n_crit = min(2, n)
    picks = list(range(n_crit))                                  # top-flow "critical" edges
    rest = n - n_crit
    if rest > 0:                                                 # spread the rest over lower flow
        picks += [int(round(x)) for x in np.linspace(len(ranked) // 5, len(ranked) - 1, rest)]
    sub = ranked.iloc[picks].copy().reset_index(drop=True)
    sub["severity"] = [3] * n_crit + [2 if i % 2 == 0 else 1 for i in range(rest)]
    sub["level_id"] = sub["road_class"] + "-S" + sub["severity"].astype(str)
    out = (sub[["edge_id", "u", "v", "road_class", "severity", "level_id"]]
           .sort_values("edge_id").reset_index(drop=True))
    out.to_csv(toy / "disruption" / f"disrupted_segments_oracle{n}.csv", index=False)
    return out


def compute_horizon(segments, scenarios):
    """Global horizon T = max completion slot over all permutations × scenarios (so every
    enumerated schedule finishes within T, and all share one comparable horizon for F1)."""
    T = 0
    for perm in itertools.permutations(segments):
        for dur in scenarios:
            T = max(T, makespan_slot(schedule_from_permutation(list(perm), dur), dur))
    return T


def run_oracle(toy_dir=TOY, out_dir=OUT, M=P.M_SCENARIOS, seed=P.SEED, probe=False, force=False):
    out_dir = scale_dir(out_dir)                     # outputs/oracle/n{N}/  (per-scale cache)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)

    disrupted = select_oracle_instance(toy_dir, P.N_DISRUPTED_ORACLE)
    segments = sorted(int(e) for e in disrupted["edge_id"])
    ctx = build_context(toy_dir, disrupted)
    scenarios = sample_scenarios(disrupted, M, seed)
    perms = list(itertools.permutations(segments))
    T = compute_horizon(segments, scenarios)
    print(f"instance: {len(segments)} segments {segments}; perms={len(perms)}; M={M}; horizon T={T} slots")

    # --- cache check: reuse an up-to-date result for this scale, unless --force ---
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

    # --- measure s/UE on one schedule, project full runtime ---
    t0 = time.perf_counter()
    evaluate_schedule(schedule_from_permutation(list(perms[0]), scenarios[0]), scenarios[0], T, ctx)
    dt = time.perf_counter() - t0
    s_ue = dt / T
    total = len(perms) * M
    print(f"one schedule = {T} UE solves took {dt:.2f}s  (~{s_ue*1000:.0f} ms/UE); "
          f"projected full run {total} schedules x {T} ~= {total*T*s_ue/60:.1f} min")
    if probe:
        return

    # --- resume from a matching partial checkpoint (interruption-safe) ---
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

    # --- full enumeration: every tested x is logged (with per-schedule timing) ---
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
        pd.DataFrame(rows).to_csv(land_path, index=False)                       # checkpoint
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
    prog_path.unlink(missing_ok=True)                # completed -> drop the resume marker
    print(f"\nWrote {out_dir}  (meta.json hash={fp[:12]}…)")


def render_figs(toy_dir=TOY, out_dir=OUT, M=P.M_SCENARIOS, seed=P.SEED):
    """Re-render figures from already-saved CSVs (no re-enumeration)."""
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
