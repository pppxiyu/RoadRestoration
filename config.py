"""
Default parameter values for the recovery-evaluation pipeline and its exact "oracle" evaluator.

This module is the single source of truth for these constants. Every value here is a MODELING
ASSUMPTION we chose, except the few flagged as "given" -- those are fixed by the problem setup
or taken directly from the open-source network data.
"""

# --- time / resources ---
DELTA_T_H = 3.0          # Wall-clock hours per discrete time slot; the recovery horizon is split into slots of this length (given by the problem setup).
C_MAX = 2               # Maximum number of road segments that can be under repair at the same time (one crew per segment).
MU = 1.0                # Blend weight in the combined objective F = MU*F1 + (1-MU)*F2, where F1 rewards
                        # network accessibility and F2 is a secondary objective term. MU = 1.0 makes the
                        # score accessibility-only for now; F2 is still computed but carries zero weight in F.

# --- Damage physics: map damage severity to the fraction of road performance retained ---
# These factors build the damaged network used for user equilibrium (UE) routing. UE is the
# traffic state in which no driver can reduce their own travel time by switching route alone.
# Severity runs 1 (light) to 3 (heavy); higher severity retains less performance.
CAP_RETAIN = {1: 0.3, 2: 0.1, 3: 0.02}    # Damaged capacity = retained_fraction * original capacity.
SPEED_RETAIN = {1: 0.5, 2: 0.3, 3: 0.2}   # Damaged free-flow time = free_flow_time / retained_fraction; dividing by a fraction < 1 inflates travel time.
SEVER_SEVERITY = 3      # At or above this severity the road is dropped entirely from the routing network
                        # (a true disconnection). Origin-destination (OD) pairs whose only path is cut then
                        # incur the large penalty travel time u_pen, so restoration order matters a lot.
F1_ACTIVE_ONLY = False  # When False, average the F1 accessibility term over the full horizon [1, T] rather
                        # than only over active repair slots. This keeps the time-averaging consistent with
                        # the (1/T) normalization used by the mixed-integer surrogate solver below.

# --- Dynamic OD demand: the "shortfall" form of the time-varying demand model ---
# The demand shortfall D_t (how far current demand has dropped below normal) evolves as:
#   D_t = max(B v_t, RHO * D_{t-1}) ;  H_t = max(0, H0 - D_t) ;  D_0 = 0
# where H_t is the demand actually served at slot t and H0 is the normal (undamaged) demand.
# Behaviour: demand drops SHARPLY at onset to the current damage-driven shortfall B v_t, then,
# once roads are restored and B v_t shrinks, RECOVERS gradually at rate RHO back toward H0.
# The mapping B[r, e] = KAPPA * (h_r0 / 3) * 1{edge e lies on a free-flow shortest path of OD r}
# ties each disrupted edge to the demand it suppresses for the OD pairs that route through it.
RHO = 0.7               # Recovery inertia of the demand shortfall: closer to 1 means slower rebound.
KAPPA = 1.0             # Sensitivity from damage to demand shortfall; larger values deepen the demand drop.

# --- Penalty travel time charged to disconnected OD pairs ---
UPEN_FACTOR = 10.0      # u_pen = UPEN_FACTOR * (largest baseline OD travel time); a large stand-in cost
                        # assigned when no route exists, so disconnection is strongly discouraged.

# --- Random scenario generation ---
M_SCENARIOS = 50        # Number of random damage scenarios every method is finally scored on.
                        # Raised from 10 on 2026-08-10: at M=10 the standard error of a method's
                        # mean F (~0.002) was the same size as the gaps BETWEEN methods, so the
                        # three solvers looked tied when a paired test over 50 scenarios separates
                        # them. Note the horizon T is derived from the sample, so changing M can
                        # change T and hence every F -- results across different M are NOT
                        # comparable, and all methods must be re-scored together after a change.
SEED = 42               # RNG seed, fixed so the scenario sampling is reproducible.
N_DISRUPTED_ORACLE = 10  # Number of disrupted roads. The instance is chosen by baseline UE flow
                        # (the most heavily used, i.e. critical, links); the brute-force oracle is
                        # only feasible at small values (n! orders).
# --- UE solver convergence ---
# The relative gap measures how far the current flow is from true equilibrium; the solver stops
# once it drops below the target rgap, or when the max-iter cap is spent, whichever comes first.
# There are TWO tolerances, because two kinds of solve serve different roles.
#
# DEFINITION solves (UE_RGAP_DEF / UE_MAX_ITER_DEF) stay tight. They fix the PROBLEM itself and
# must be stable run to run: the baseline two-way flow that ranks edges to select the instance and
# that seeds the RL flow prior (util.oracle._baseline_twoway_flow), and the baseline reference
# travel times u_0 that F measures degradation against (util.evaluate.build_context). Loosening
# these once silently reordered the edge ranking and changed WHICH segments the n=6 instance
# selects, which is why they are pinned here rather than sharing the loosened value.
#
# DECISION 2026-08-10: the two-tier split is PERMANENT (option 甲). Deleting this tier was
# considered for compute savings and measured before rejecting: it changes the n=10 instance
# itself ([1,10,12,13,15,17,18,21,25,32] -> [1,8,12,13,15,17,19,22,23,30], six segments differ),
# which would invalidate every recorded result -- while saving almost nothing, because these
# solves run once per context (seconds) and ALL per-slot evaluation already runs at the loose
# tier below. Do not merge the tiers.
UE_RGAP_DEF = 1e-6
UE_MAX_ITER_DEF = 100
#
# EVALUATION solves (UE_RGAP / UE_MAX_ITER) are the hot path: one per slot, thousands per RL run.
# EXPLORATION-STAGE LOOSENING (2026-08-06): dropped from 1e-6/100 to 1e-2/25 while the solver was
# AequilibraE, trading ~1% error in the per-slot accessibility g for ~3.7x speed.
# RETIGHTENED TO 1e-3 + WARM STARTS (2026-08-11), after the in-house solver replaced AequilibraE.
# The tolerance audit (python -m util.ue_audit; data and write-up under outputs/0-UE_val/) showed
# 1e-2 was quietly corrupting the SHAPE of the recovery curve, not just its level: the error in the
# slot-to-slot CHANGE of g reached 3.8x the true change at the 90th percentile, and 5 of 50
# adjacent slot pairs had their direction read backwards (g measured as rising where it truly
# fell). At 1e-3 with warm-started solves the change error is bounded by 0.34x the true change on
# every pair, direction flips vanish, and each solve is ~1.9x FASTER than the 1e-2 cold setting it
# replaces (median 10 iterations vs 17, because the previous slot's equilibrium is a far better
# start than free flow). Results under this setting are not tolerance-comparable to runs recorded
# before it; every method must be re-scored together before mixing numbers.
UE_RGAP = 1e-3
UE_MAX_ITER = 100       # Hard cap on solver iterations, in case the gap tolerance is never reached.
                        # Sized from the audit: cold solves at 1e-3 take ~40 iterations (each
                        # scenario's first slot is necessarily cold), warm ones ~10.
UE_WARM_START = True    # Chain consecutive slots of an evaluation: each slot's UE solve is seeded
                        # with the previous slot's equilibrium plus an all-or-nothing loading of
                        # the demand increment (util.ue.warm_start_seed -- see there for why the
                        # seed must be built feasible). Besides speed, chaining makes consecutive
                        # slots' truncation errors nearly cancel in the g differences the recovery
                        # curve is built from (audit: change error 0.043x vs 0.067x cold at the
                        # median). Off = every slot solves cold from free flow, as before.

SIM_CACHE = True        # Persistent cross-run, cross-method cache of per-slot UE accessibility
                        # terms, under cache/sim_cache/ at the project root (util/sim_cache.py).
                        # Sound because the slot term is a deterministic function of the damage-
                        # trajectory prefix; every entry lives under a fingerprint of EVERYTHING
                        # else that determines it (network data, instance, B/H0/baseline arrays,
                        # RHO/KAPPA/penalty and both UE tolerance regimes), so changing any of
                        # those starts a fresh cache rather than silently reusing stale terms.
                        # False bypasses it entirely (every slot solved live).

# --- Pretraining surrogate: a cheap approximate stand-in for the true scheduling problem,
# cast as a mixed-integer linear program (MILP) and solved with the traffic state held fixed ---
# The schedule is found by alternating: hold travel times fixed, solve the MILP for a schedule,
# then re-solve UE to refresh travel times, and repeat until the schedule settles.
MILP_MAX_ITER = 20      # Hard stop on the number of fix-travel-times / solve-MILP / refresh-UE iterations.
MILP_CYCLE_TOL = 3      # Cycle guard: stop only after the SAME schedule recurs this many times. 1 would stop
                        # on the first repeat; a larger value lets the damped loop keep exploring.
MILP_DAMPING_MODE = "decay"  # How the frozen travel times are relaxed between iterations.
                        # "decay": a DIMINISHING step lambda_n = 1/(n+1) (n = 1-based iteration). Its
                        # vanishing size drives the travel-time oscillation to zero, which is what lets the
                        # loop actually converge to a fixed point rather than settle into a limit cycle.
                        # "const": the earlier behaviour -- a FIXED step MILP_DAMPING, which only shrinks
                        # the oscillation amplitude and leaves a persistent (non-zero) cycle.
MILP_DAMPING = 0.5      # Fixed relaxation weight, used only when MILP_DAMPING_MODE == "const":
                        # u_used = damping*u_new + (1-damping)*u_prev.
MILP_PROX_SCALE = 0.1   # Strength of the proximal schedule penalty gamma_n * ||y - y_prev||_1 added to the
                        # MILP objective, with gamma_n = gamma0/(n+1) and gamma0 = MILP_PROX_SCALE * (median
                        # across-slot spread of the coefficients c). Under the start-once constraint this L1
                        # distance equals twice the number of segments that changed start slot, so it stays
                        # linear. It discourages needless schedule flips between mutually-best-response
                        # schedules, tipping a would-be cycle into a fixed point. 0 disables it; 0.1 was the
                        # smallest strength that converged every tested scenario without hurting solution
                        # quality (a larger value freezes the schedule earlier and costs more accuracy).
                        # High-damage tuning note (n=19, 50% of the network, 2026-07-23): here the decaying
                        # prox=0.1 gives MILP == flow (the frozen-traffic surrogate leaps to worse schedules
                        # every step). A per-scenario grid of STRONGER proximals -- {2,3,5} held CONSTANT
                        # (no 1/(n+1) decay) plus a decaying 3 -- keeping the best-by-true-F iterate, forces
                        # small ~1-segment steps and recovers ~4% below flow (beats flow 9/10, converges by
                        # iter 2-3). The const-proximal mode is NOT wired into the loop; this was run
                        # out-of-repo (scratchpad run_n19_tuned.py). Wire in a MILP_PROX_MODE knob to make it
                        # reproducible here.
MILP_WARM_START = True  # Seed iteration 0 of the alternating loop from the flow-greedy schedule -- the
                        # strongest static baseline, with segments ordered by baseline UE two-way flow
                        # (edge criticality) -- instead of the edge-id packing order. Because the loop
                        # returns the best-by-true-F iterate over its whole history and iteration 0 is now
                        # that baseline, the MILP result is GUARANTEED no worse than the flow baseline on
                        # every scenario; the alternating refinement can only improve on it. Without it the
                        # loop can converge to a schedule worse than the baseline (e.g. F > 1). False
                        # restores the old edge-id cold start.

# --- Base restoration-duration support sets (in slots), keyed by (road_class, severity) ---
# Each entry lists the candidate repair durations for a road of that class and damage severity;
# heavier damage and larger road classes take longer. A concrete duration is drawn from this set.
DURATION_SUPPORT = {
    ("local", 1): [1],        ("local", 2): [2, 3],     ("local", 3): [4, 5],
    ("major", 1): [2, 3],     ("major", 2): [4, 5],     ("major", 3): [6, 7, 8],
    ("highway", 1): [4, 5],   ("highway", 2): [6, 7, 8], ("highway", 3): [9, 10, 11],
}
ETA = [0.8, 1.0, 1.2]   # Crew-efficiency multipliers on the base duration: a slow, typical, or fast crew (given).
