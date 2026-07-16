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
M_SCENARIOS = 10        # Number of random damage scenarios to sample for the full toy run.
SEED = 42               # RNG seed, fixed so the scenario sampling is reproducible.
N_DISRUPTED_ORACLE = 4  # Number of disrupted roads the oracle enumerates over. It tries every repair
                        # order, so 4 disrupted roads gives 4! = 24 candidate schedules. The disrupted set
                        # is chosen by baseline UE flow (the most heavily used, i.e. critical, links).

# --- UE solver convergence for the per-slot traffic solves (loosened for speed) ---
# The relative gap measures how far the current flow is from true equilibrium; the solver stops
# once it drops below UE_RGAP. This tolerance is looser than the tight setting used for validation.
UE_RGAP = 1e-6
UE_MAX_ITER = 100       # Hard cap on solver iterations, in case the gap tolerance is never reached.

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

# --- Base restoration-duration support sets (in slots), keyed by (road_class, severity) ---
# Each entry lists the candidate repair durations for a road of that class and damage severity;
# heavier damage and larger road classes take longer. A concrete duration is drawn from this set.
DURATION_SUPPORT = {
    ("local", 1): [1],        ("local", 2): [2, 3],     ("local", 3): [4, 5],
    ("major", 1): [2, 3],     ("major", 2): [4, 5],     ("major", 3): [6, 7, 8],
    ("highway", 1): [4, 5],   ("highway", 2): [6, 7, 8], ("highway", 3): [9, 10, 11],
}
ETA = [0.8, 1.0, 1.2]   # Crew-efficiency multipliers on the base duration: a slow, typical, or fast crew (given).
