"""
Greedy baseline solvers for the road-restoration scheduling problem.

Cheap alternatives to the pretraining MILP and the brute-force oracle. Each produces a repair
PRIORITY ORDER that a work-conserving list schedule turns into start times; the variants differ only
in how they rank the disrupted segments:

  demand   : STATIC -- rank by the demand a segment restores when repaired, sum_r B[r,e]*v_e* (a
             first-order proxy for its contribution to the accessibility objective F1). No UE during
             ranking; the order is the same across scenarios.
  ratio    : STATIC -- rank by that restored demand PER repair slot, (sum_r B[r,e]*v_e*) / d_e, i.e.
             benefit per unit of repair effort. Scenario-dependent (durations vary).
  flow     : STATIC -- rank by the segment's baseline user-equilibrium two-way flow (edge
             criticality); one shared baseline UE solve feeds every scenario.
  stepwise : FAITHFUL -- build the order front-to-back, at each step keeping the segment whose
             repair-next lowers the exact objective F the most (real UE at every candidate). Costs
             N*(N+1)/2 full evaluations per scenario, so it is far slower and is not run by default.

The three static rankers never run UE to decide the order, so each costs a single F evaluation per
scenario (one UE per slot) -- about an order of magnitude cheaper than the stepwise greedy. All
variants share the same instance / scenarios / horizon / real evaluator as the oracle and the MILP,
so their objective values are directly comparable. Each variant writes
outputs/greedy/n{N}/{variant}_optima.csv.

Run inside the road_restore conda env (PYTHONPATH = project root):
  python -m util.greedy                    # the three static variants (default)
  python -m util.greedy demand ratio flow stepwise   # choose variants explicitly
"""
import sys
import time
from pathlib import Path

import pandas as pd

import config as P
from util.evaluate import build_context, evaluate_schedule, schedule_from_permutation
from util.oracle import (_baseline_twoway_flow, compute_horizon, scale_dir,
                         select_oracle_instance)
from util.scenarios import sample_scenarios

ROOT = Path(__file__).resolve().parent.parent
TOY = ROOT / "data" / "siouxfalls_toy"
OUT = ROOT / "outputs" / "greedy"


# --------------------------------------------------------------------------- #
# Static importance scores (no UE): higher score -> repair earlier
# --------------------------------------------------------------------------- #
def _demand_restored(ctx):
    """Per-segment demand a repair restores: sum_r B[r,e] * v_e* (severity-scaled column sum of B).
    This is the first-order accessibility contribution of finishing segment e."""
    B, sev, dis = ctx["B"], ctx["severity_vec"], ctx["disrupted"]
    return {int(dis[j][0]): float(sev[j] * B[:, j].sum()) for j in range(len(dis))}


def score_demand(ctx, durations, flow):
    return _demand_restored(ctx)


def score_ratio(ctx, durations, flow):
    dem = _demand_restored(ctx)
    return {e: dem[e] / max(1, int(durations[e])) for e in dem}          # benefit per repair slot


def score_flow(ctx, durations, flow):
    return {int(eid): flow.get((min(u, v), max(u, v)), 0.0) for (eid, u, v, s) in ctx["disrupted"]}


STATIC = {"demand": score_demand, "ratio": score_ratio, "flow": score_flow}


def greedy_stepwise(ctx, durations, segments, T):
    """Faithful greedy: build the order front-to-back, each step keeping the segment whose
    repair-next gives the lowest exact F (real UE at every candidate). Returns (order, start, res)."""
    remaining = sorted(int(e) for e in segments)
    prefix, best_res, best_start = [], None, None
    while remaining:
        best_seg = best_res = best_start = None
        for seg in remaining:
            tail = [s for s in remaining if s != seg]
            start = schedule_from_permutation(prefix + [seg] + tail, durations)
            res = evaluate_schedule(start, durations, T, ctx)
            if best_res is None or res["F"] < best_res["F"]:
                best_seg, best_res, best_start = seg, res, start
        prefix.append(best_seg)
        remaining.remove(best_seg)
    return prefix, best_start, best_res


# --------------------------------------------------------------------------- #
# Run
# --------------------------------------------------------------------------- #
def run_greedy(variants=("demand", "ratio", "flow"), toy_dir=TOY, out_dir=OUT,
               M=P.M_SCENARIOS, seed=P.SEED):
    """Run the chosen greedy variants over M scenarios and write one
    outputs/greedy/n{N}/{variant}_optima.csv per variant (checkpointed each scenario)."""
    out_dir = scale_dir(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    disrupted = select_oracle_instance(toy_dir, P.N_DISRUPTED_ORACLE)
    segments = sorted(int(e) for e in disrupted["edge_id"])
    ctx = build_context(toy_dir, disrupted)
    scenarios = sample_scenarios(disrupted, M, seed)
    T = compute_horizon(segments, scenarios)
    flow = _baseline_twoway_flow(toy_dir) if "flow" in variants else {}
    print(f"instance: {len(segments)} segments {segments}; M={M}; horizon T={T}; "
          f"variants={list(variants)}", flush=True)

    rows = {v: [] for v in variants}
    t_all = time.perf_counter()
    for m, dur in enumerate(scenarios):
        for v in variants:
            t0 = time.perf_counter()                                     # time each variant honestly (its own F evaluation)
            if v == "stepwise":
                order, start, res = greedy_stepwise(ctx, dur, segments, T)
            else:
                scores = STATIC[v](ctx, dur, flow)
                order = sorted(segments, key=lambda e: (-scores[e], e))  # highest score first, edge-id tiebreak
                start = schedule_from_permutation(order, dur)
                res = evaluate_schedule(start, dur, T, ctx)
            row = dict(scenario=m, F=res["F"], F1=res["F1"], F2=res["F2"],
                       time_s=time.perf_counter() - t0,
                       order="-".join(map(str, order)),
                       durations="-".join(str(int(dur[e])) for e in segments))
            for e in segments:
                row[f"start_{e}"] = start[e]
            rows[v].append(row)
        for v in variants:
            pd.DataFrame(rows[v]).to_csv(out_dir / f"{v}_optima.csv", index=False)
        print(f"  scenario {m + 1}/{M}:  " +
              "  ".join(f"{v}={rows[v][-1]['F']:.4f}" for v in variants), flush=True)

    for v in variants:
        print(f"  mean F [{v}] = {pd.DataFrame(rows[v])['F'].mean():.4f}", flush=True)
    print(f"Wrote {out_dir}  ({(time.perf_counter() - t_all) / 60:.1f} min)", flush=True)
    return out_dir


if __name__ == "__main__":
    vs = tuple(a for a in sys.argv[1:] if not a.startswith("-")) or ("demand", "ratio", "flow")
    run_greedy(variants=vs)
