"""
Assumed default parameters for the recovery-evaluation pipeline & oracle.

Single source of truth (mirrored in TECHNICAL_NOTE/oracle_validation.md). Everything here
is a MODELING ASSUMPTION except where noted as given by the paper / open-source data.
"""

# --- time / resources ---
DELTA_T_H = 3.0          # hours per slot (paper, given)
C_MAX = 2               # max simultaneous maintenance crews
MU = 1.0                # weight in F = MU*F1 + (1-MU)*F2
                        # F2 weight = 0 (accessibility-only for now); F2 is still computed, just not in F

# --- damage physics: severity -> retained fraction (for UE on the damaged network) ---
CAP_RETAIN = {1: 0.3, 2: 0.1, 3: 0.02}    # damaged capacity      = retain * capacity (harsher still)
SPEED_RETAIN = {1: 0.5, 2: 0.3, 3: 0.2}   # damaged free_flow_time = free_flow_time / retain
SEVER_SEVERITY = 3      # severity >= this => the edge is fully removed from the UE network
                        # (true disconnection -> affected OD pairs get u_pen); makes order matter a lot
F1_ACTIVE_ONLY = False  # average F1 over the FULL horizon [1,T] (reverted): matches the c_e^k (1/T)
                        # normalization used by the section 2.1.1 MILP surrogate

# --- dynamic OD demand (shortfall form of Eq. 4) ---
# shortfall:  D_t = max(B v_t, RHO * D_{t-1}) ;  H_t = max(0, H0 - D_t) ;  D_0 = 0
#   => SHARP drop at onset to the current damage-driven shortfall B v_t, then (once roads
#      restore and B v_t shrinks) demand RECOVERS gradually at rate RHO back to normal H0.
# B[r,e] = KAPPA * (h_r0 / 3) * 1{e on a free-flow shortest path of OD r}
RHO = 0.7               # recovery inertia (closer to 1 = slower recovery)
KAPPA = 1.0             # damage -> demand-shortfall sensitivity scale (tune for drop depth)

# --- penalty travel time for disconnected OD pairs ---
UPEN_FACTOR = 10.0      # u_pen = UPEN_FACTOR * max baseline OD travel time

# --- scenarios ---
M_SCENARIOS = 10        # toy-full run (was 5 for quick-check)
SEED = 42
N_DISRUPTED_ORACLE = 4  # toy-full run (was 3 for quick-check); 4! = 24 schedules
                        # the disruption set is now chosen by baseline UE flow (critical links), see oracle.py

# --- UE convergence for the per-step solves (looser than the 1e-12 validation run, for speed) ---
UE_RGAP = 1e-6
UE_MAX_ITER = 100

# --- section 2.1.1 pretraining MILP (traffic-fixation) ---
MILP_MAX_ITER = 20      # max alternating iterations (fix travel times -> solve MILP -> refresh UE); hard stop
MILP_CYCLE_TOL = 3      # relaxed cycle guard: stop only after the SAME schedule recurs this many times
                        # (1 = stop on first repeat; larger = let the damped loop keep exploring)
MILP_DAMPING = 0.5      # MSA-style travel-time relaxation: u_used = damping*u_new + (1-damping)*u_prev
                        # 1.0 = raw (large swings); smaller = smoother, more monotone trajectory

# --- Table 1: base restoration duration support sets (slots) by (road_class, severity) ---
DURATION_SUPPORT = {
    ("local", 1): [1],        ("local", 2): [2, 3],     ("local", 3): [4, 5],
    ("major", 1): [2, 3],     ("major", 2): [4, 5],     ("major", 3): [6, 7, 8],
    ("highway", 1): [4, 5],   ("highway", 2): [6, 7, 8], ("highway", 3): [9, 10, 11],
}
ETA = [0.8, 1.0, 1.2]   # crew-efficiency multiplier (given)
