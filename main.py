"""
Single entry point for the road-restoration toy study.

The study asks in what order to repair the road segments damaged by a disaster. Each candidate
repair order is scored by an objective F that balances two competing goals: restoring the
network quickly, and limiting how much travel accessibility is lost while repairs are underway.
The toy instance is small enough that the truly best order can be found by exhaustive search,
which gives a ground-truth benchmark for a faster approximate solver.

DEFAULT (`python main.py`): run the whole task at the problem size configured in config.py.
  Step 1  Brute-force ORACLE: evaluate every feasible repair order to find the true optimum
          for each scenario. This is the ground truth; the run is resumable. -> outputs/oracle/n{N}/
  Step 2  Approximate "traffic-fixation" solver: a much cheaper mixed-integer linear program
          (MILP) that estimates the same schedule, plus comparison and process figures. Its
          quality is measured against the oracle. -> outputs/pretrain_milp/n{N}/
The oracle is RESUMABLE: if a run is interrupted, rerun `python main.py` and it continues from
the last completed scenario (finished ones are skipped). If a complete oracle result already
exists for this problem size and parameter set, it is reused from cache rather than recomputed.

`python main.py --walkthrough`: evaluate ONE (schedule, scenario) pair while printing every
stage of the objective-evaluation pipeline under an explicit label, so the scoring logic can
be inspected end to end.

Run inside the road_restore conda env:
  python main.py                 # full run (oracle + MILP)
  python main.py --walkthrough   # step-by-step objective walkthrough for one example
(Equivalent module entry points: `python -m util.oracle`, `python -m util.pretrain_milp`.)
"""

import sys
from pathlib import Path

import config as P
from util.evaluate import (build_context, evaluate_schedule, f2_value, makespan_slot,
                           schedule_from_permutation)
from util.oracle import select_oracle_instance
from util.scenarios import sample_scenarios

ROOT = Path(__file__).resolve().parent
TOY = ROOT / "data" / "siouxfalls_toy"


def run_all():
    """Run the full task: the brute-force oracle first (Step 1), then the approximate
    traffic-fixation MILP whose schedule is scored against that oracle (Step 2)."""
    from util.oracle import run_oracle
    from util.pretrain_milp import run_pretrain_milp
    print("=" * 72)
    print(f"STEP 1/2  brute-force ORACLE (ground truth)   N={P.N_DISRUPTED_ORACLE}, M={P.M_SCENARIOS}")
    print("=" * 72, flush=True)
    run_oracle()
    print("\n" + "=" * 72)
    print("STEP 2/2  traffic-fixation MILP  (+ comparison & process figures)")
    print("=" * 72, flush=True)
    run_pretrain_milp()
    print("\nDONE. Raw data + figures under outputs/oracle/n{N}/ and outputs/pretrain_milp/n{N}/.")


def walkthrough():
    # Build one concrete instance to trace end to end: the disrupted-segment set the oracle
    # scores, a single sampled duration scenario (realized repair times), and one repair order.
    disrupted = select_oracle_instance(TOY, n=P.N_DISRUPTED_ORACLE)
    ctx = build_context(TOY, disrupted)
    durations = sample_scenarios(disrupted, M=1, seed=P.SEED)[0]
    segments = sorted(int(e) for e in disrupted["edge_id"])
    perm = segments                                   # example repair priority: segments in ascending edge-id order (arbitrary but reproducible)
    # A "work-conserving" schedule assigns the crews greedily so no crew sits idle while a
    # segment still waits; the makespan is the slot at which the last repair finishes (horizon T).
    start = schedule_from_permutation(perm, durations, P.C_MAX)
    T = makespan_slot(start, durations)

    print(f"Disrupted segments E = {segments};  C_max={P.C_MAX} crews;  dt={P.DELTA_T_H} h;  mu={P.MU}")
    print("Example schedule (work-conserving from the edge-id order):")
    for e in segments:
        print(f"  edge {e}: start slot k={start[e]}, duration={durations[e]} "
              f"-> completes at k={start[e] + durations[e]}")
    print(f"Horizon for this example T={T} slots\n")

    # ===================== Step 1 - damage trajectory: which segments are still under repair (damaged) in each time slot =====================
    print("# Step 1 - damage trajectory: still-damaged segments per slot")
    for k in range(1, T + 1):
        damaged = [e for e in segments if k < start[e] + durations[e]]
        print(f"    k={k:2d}: damaged = {damaged}")
    print()

    # ===================== Step 2 - F2, the restoration-efficiency objective: makespan divided by total repair work; pure schedule arithmetic, no traffic assignment =====================
    print("# Step 2 - F2 = (makespan - t0) / sum_e d_e*dt   (pure schedule math, no UE)")
    print(f"    makespan slot = {makespan_slot(start, durations)}, total work = "
          f"{sum(durations.values())} slots  ->  F2 = {f2_value(start, durations):.4f}\n")

    # ===================== Step 3 - F1, the accessibility-degradation objective =====================
    # A per-slot loop over k=1..T; each slot solves one user equilibrium (UE) -- the traffic state in
    # which no driver can lower their own travel time by switching route -- on the network as it stands.
    print("# Step 3 - per-step loop k=1..T:")
    print("#     3a  demand shortfall  D_t = max(B*v_t, rho*D_{t-1}) ;  H_t = max(0, H0 - D_t)   (sharp drop -> recover)")
    print("#     3b  damaged network   (capacity x retain, free-flow-time / retain, per severity)")
    print("#     3c  UE on (damaged net, H_t)  ->  congested link times  ->  OD travel times u_r")
    print("#     3d  per-step F1 term  =  sum_r h_r(t_k)*u_tilde_r  /  sum_r h_r(t_k)*u_r(t0)   (>=1; ->1 when restored)")
    res = evaluate_schedule(start, durations, T, ctx, collect_traces=True)
    tr = res["traces"].copy()
    tr["total_demand"] = tr["total_demand"].round(0).astype(int)
    tr["f1_term"] = tr["f1_term"].round(4)
    print(tr.to_string(index=False))
    print(f"\n    F1 = mean(per-step terms) = {res['F1']:.4f}\n")

    # ===================== Step 4 - combine into the overall objective F = mu*F1 + (1-mu)*F2 =====================
    print("# Step 4 - F = mu*F1 + (1-mu)*F2")
    print(f"    F = {P.MU}*{res['F1']:.4f} + {1 - P.MU}*{res['F2']:.4f}  =  {res['F']:.4f}")


if __name__ == "__main__":
    if "--walkthrough" in sys.argv:
        walkthrough()
    else:
        run_all()
