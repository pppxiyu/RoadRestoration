"""
Greedy baseline solvers for the road-restoration scheduling problem.

Cheap alternatives to the pretraining MILP and the brute-force oracle. Each produces a repair
PRIORITY ORDER that a work-conserving list schedule turns into start times; the variants differ only
in how they rank the disrupted segments:

  demand   : rank by the demand a segment restores when repaired, sum_r B[r,e]*v_e* (a first-order
             proxy for its contribution to the accessibility objective F1).
  ratio    : rank by that restored demand PER unit of repair effort, (sum_r B[r,e]*v_e*) / E[d_e],
             where E[d_e] is the segment's EXPECTED duration under the full duration model rather
             than the duration a particular scenario realizes (util.scenarios.expected_durations).
  flow     : rank by the segment's baseline user-equilibrium two-way flow (edge criticality); one
             shared baseline UE solve feeds every scenario.

None of the three reads anything that is only known after the fact, so all three produce ONE order
that is committed before any repair starts and is then evaluated on every scenario; their mean F is
therefore the expected performance of a single fixed decision rule, not the average of M different
hindsight decisions. (Ranking by a REALIZED duration, as ratio previously did, quietly gave that
one rule an information advantage the others did not have.)

The three rankers never run UE to decide the order, so each costs a single F evaluation per
scenario (one UE per slot). All variants share the same instance / scenarios / horizon / real
evaluator as the oracle and the MILP, so their objective values are directly comparable. Each
variant writes outputs/02-baselines/02-rule-based/n{N}/results/{variant}_optima.csv, alongside
log/{variant}_slots.csv (the per-slot accessibility of that same evaluation, so a recovery curve
costs no extra UE solve) and config/{variant}_run_meta.json (the instance, objective parameters
and code revision the numbers came from). The directory is shared by the three static rankers,
hence the per-variant metadata name.

Run inside the road_restore conda env (PYTHONPATH = project root):
  python -m util.greedy                    # the three static variants (default)
  python -m util.greedy demand ratio flow   # choose variants explicitly
"""
import sys
import time
from pathlib import Path

import pandas as pd

import config as P
from util.evaluate import build_context, evaluate_schedule, schedule_from_permutation
from util.provenance import (solver_dir, fresh_scale_dir, log_dir, results_dir, slot_rows,
                             write_run_meta)
from util.oracle import (_baseline_twoway_flow, compute_horizon, scale_dir,
                         select_oracle_instance)
from util.scenarios import expected_durations, sample_scenarios

ROOT = Path(__file__).resolve().parent.parent
TOY = ROOT / "data" / "siouxfalls_toy"
OUT = ROOT / "outputs" / "02-baselines" / solver_dir("rule-based")


# --------------------------------------------------------------------------- #
# Static importance scores (no UE): higher score -> repair earlier
# --------------------------------------------------------------------------- #
def _demand_restored(ctx):
    """Per-segment demand a repair restores: sum_r B[r,e] * v_e* (severity-scaled column sum of B).
    This is the first-order accessibility contribution of finishing segment e."""
    B, sev, dis = ctx["B"], ctx["severity_vec"], ctx["disrupted"]
    return {int(dis[j][0]): float(sev[j] * B[:, j].sum()) for j in range(len(dis))}


def score_demand(ctx, flow, exp_dur):
    return _demand_restored(ctx)


def score_ratio(ctx, flow, exp_dur):
    """Restored demand per unit of repair effort. The divisor is the segment's EXPECTED duration
    under the full duration model (util.scenarios.expected_durations), not the duration the
    scenario at hand happens to realize, so the ranking can be committed to before any repair
    starts. Dividing by a realized duration would make the rule read a quantity that is only
    known after the fact, which would rate it against the other baselines unfairly."""
    dem = _demand_restored(ctx)
    return {e: dem[e] / exp_dur[e] for e in dem}


def score_flow(ctx, flow, exp_dur):
    return {int(eid): flow.get((min(u, v), max(u, v)), 0.0) for (eid, u, v, s) in ctx["disrupted"]}


STATIC = {"demand": score_demand, "ratio": score_ratio, "flow": score_flow}


# --------------------------------------------------------------------------- #
# Run
# --------------------------------------------------------------------------- #
def run_greedy(variants=("demand", "ratio", "flow"), toy_dir=TOY, out_dir=OUT,
               M=P.M_SCENARIOS, seed=P.SEED):
    """Run the chosen greedy variants over M scenarios and write one
    outputs/02-baselines/02-rule-based/n{N}/results/{variant}_optima.csv per variant
    (checkpointed each scenario)."""
    out_dir = scale_dir(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # This folder is SHARED by the static rankers, so clear only the variants being rerun --
    # a fresh run must leave no residue of its own previous run, and no trace of it may survive
    # beside the new results, but the siblings' results are not this run's to erase.
    fresh_scale_dir(out_dir, methods=list(variants))
    disrupted = select_oracle_instance(toy_dir, P.N_DISRUPTED_ORACLE)
    segments = sorted(int(e) for e in disrupted["edge_id"])
    ctx = build_context(toy_dir, disrupted)
    scenarios = sample_scenarios(disrupted, M, seed)
    T = compute_horizon(segments, scenarios)
    flow = _baseline_twoway_flow(toy_dir) if "flow" in variants else {}
    print(f"instance: {len(segments)} segments {segments}; M={M}; horizon T={T}; "
          f"variants={list(variants)}", flush=True)

    # Every static ranker is scenario-independent, so each priority order is computed ONCE and the
    # same order is then evaluated on all M scenarios. This is what makes these rules committable
    # before any duration is observed, and it is why their mean F estimates the performance of one
    # fixed decision rather than of M different hindsight decisions.
    exp_dur = expected_durations(disrupted)
    orders = {}
    for v in variants:
        scores = STATIC[v](ctx, flow, exp_dur)
        orders[v] = sorted(segments, key=lambda e: (-scores[e], e))      # highest score first, edge-id tiebreak

    rows = {v: [] for v in variants}
    slots = {v: [] for v in variants}
    t_all = time.perf_counter()
    for m, dur in enumerate(scenarios):
        for v in variants:
            t0 = time.perf_counter()                                     # time each variant honestly (its own F evaluation)
            order = orders[v]
            start = schedule_from_permutation(order, dur, access=ctx["access"])
            res = evaluate_schedule(start, dur, T, ctx, collect_traces=True)
            slots[v].extend(slot_rows(m, res))          # recovery curve, at no extra UE cost
            row = dict(scenario=m, F=res["F"], F1=res["F1"], F2=res["F2"],
                       time_s=time.perf_counter() - t0,
                       order="-".join(map(str, order)),
                       durations="-".join(str(int(dur[e])) for e in segments))
            for e in segments:
                row[f"start_{e}"] = start[e]
            rows[v].append(row)
        for v in variants:
            pd.DataFrame(rows[v]).to_csv(results_dir(out_dir) / f"{v}_optima.csv", index=False)
            pd.DataFrame(slots[v]).to_csv(log_dir(out_dir) / f"{v}_slots.csv", index=False)
        print(f"  scenario {m + 1}/{M}:  " +
              "  ".join(f"{v}={rows[v][-1]['F']:.4f}" for v in variants), flush=True)

    for v in variants:
        print(f"  mean F [{v}] = {pd.DataFrame(rows[v])['F'].mean():.4f}", flush=True)
    import json as _json
    for v in variants:
        (results_dir(out_dir) / f"{v}_solution_best.json").write_text(_json.dumps(dict(
            order=[int(x) for x in orders[v]],
            mean_F=float(pd.DataFrame(rows[v])["F"].mean())), indent=2), encoding="utf-8")
        # outputs/02-baselines/02-rule-based/n{N}/ is shared by the three static rankers, so
        # each ranker's metadata carries its own name rather than overwriting a common file.
        write_run_meta(out_dir, method=v, segments=segments, T=T, seed=seed, M=M,
                       shared_dir=True,
                       order="-".join(map(str, orders[v])),
                       expected_durations={int(e): exp_dur[e] for e in segments},
                       wall_s=time.perf_counter() - t_all)
    print(f"Wrote {out_dir}  ({(time.perf_counter() - t_all) / 60:.1f} min)", flush=True)
    from util.compare import refresh_comparison
    refresh_comparison()                     # every solver run leaves the comparison current
    return out_dir


if __name__ == "__main__":
    vs = tuple(a for a in sys.argv[1:] if not a.startswith("-")) or ("demand", "ratio", "flow")
    run_greedy(variants=vs)
