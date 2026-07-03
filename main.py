"""
main.py - single entry point for the toy pipeline.

DEFAULT (`python main.py`): run the whole task for the current scale in config.py --
  Step 1  brute-force oracle (ground truth, resumable)        -> outputs/oracle/n{N}/
  Step 2  section 2.1.1 traffic-fixation MILP + all figures   -> outputs/pretrain_milp/n{N}/
The oracle is RESUMABLE: if a run is interrupted, just run `python main.py` again and it
continues from the last completed scenario (skips the finished ones). If the oracle result for
this scale + params is already complete, it is reused from cache (no re-computation).

`python main.py --walkthrough`: evaluate ONE (schedule, scenario) with every Figure-1 step
printed under an explicit label, so the pipeline logic can be checked against the paper.

Run inside the road_restore conda env:
  python main.py                 # full run (oracle + MILP)
  python main.py --walkthrough   # Figure-1 walkthrough for one example
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
    """Run the full task: brute-force oracle (Step 1) then the section 2.1.1 MILP (Step 2)."""
    from util.oracle import run_oracle
    from util.pretrain_milp import run_pretrain_milp
    print("=" * 72)
    print(f"STEP 1/2  brute-force ORACLE (ground truth)   N={P.N_DISRUPTED_ORACLE}, M={P.M_SCENARIOS}")
    print("=" * 72, flush=True)
    run_oracle()
    print("\n" + "=" * 72)
    print("STEP 2/2  section 2.1.1 traffic-fixation MILP  (+ comparison & process figures)")
    print("=" * 72, flush=True)
    run_pretrain_milp()
    print("\nDONE. Raw data + figures under outputs/oracle/n{N}/ and outputs/pretrain_milp/n{N}/.")


def walkthrough():
    # --- instance: oracle subset + one scenario + one example schedule ---
    disrupted = select_oracle_instance(TOY, n=P.N_DISRUPTED_ORACLE, seed=P.SEED)
    ctx = build_context(TOY, disrupted)
    durations = sample_scenarios(disrupted, M=1, seed=P.SEED)[0]
    segments = sorted(int(e) for e in disrupted["edge_id"])
    perm = segments                                   # example priority order (by edge id)
    start = schedule_from_permutation(perm, durations, P.C_MAX)
    T = makespan_slot(start, durations)

    print(f"Disrupted segments E = {segments};  C_max={P.C_MAX} crews;  dt={P.DELTA_T_H} h;  mu={P.MU}")
    print("Example schedule (work-conserving from the edge-id order):")
    for e in segments:
        print(f"  edge {e}: start slot k={start[e]}, duration={durations[e]} "
              f"-> completes at k={start[e] + durations[e]}")
    print(f"Horizon for this example T={T} slots\n")

    # ===================== Figure 1 . Step 1 - damage trajectory v^{t_k} (Eq. 2) =====================
    print("# Figure 1 . Step 1 - damage trajectory v^(t_k) (Eq. 2): still-damaged segments per slot")
    for k in range(1, T + 1):
        damaged = [e for e in segments if k < start[e] + durations[e]]
        print(f"    k={k:2d}: damaged = {damaged}")
    print()

    # ===================== Figure 1 . Step 2 - F2 (makespan / work), NO UE =====================
    print("# Figure 1 . Step 2 - F2 = (makespan - t0) / sum_e d_e*dt   (pure schedule math, no UE)")
    print(f"    makespan slot = {makespan_slot(start, durations)}, total work = "
          f"{sum(durations.values())} slots  ->  F2 = {f2_value(start, durations):.4f}\n")

    # ===================== Figure 1 . Step 3 - per-step UE loop for F1 =====================
    print("# Figure 1 . Step 3 - per-step loop k=1..T:")
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

    # ===================== Figure 1 . Step 4 - compose F =====================
    print("# Figure 1 . Step 4 - F = mu*F1 + (1-mu)*F2")
    print(f"    F = {P.MU}*{res['F1']:.4f} + {1 - P.MU}*{res['F2']:.4f}  =  {res['F']:.4f}")


if __name__ == "__main__":
    if "--walkthrough" in sys.argv:
        walkthrough()
    else:
        run_all()
