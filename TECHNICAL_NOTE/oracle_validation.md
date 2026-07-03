# Oracle validation — parked design notes

Notes the user asked to park here (to handle later), plus the assumptions and caveats behind
the brute-force oracle (`util/oracle.py`, evaluator `util/evaluate.py`). The oracle computes the
**true** hindsight optimum `F*` per scenario by enumerating all work-conserving schedules and
scoring each with the exact Figure-1 objective `F(x|ω)`.

---

## 1. Validation levels — A (surrogate) vs B (true)  [point 3, parked]
When the pretraining **MILP** is built later, it can be checked at two levels:

- **Level A — formulation correctness.** Brute-force the MILP's *own surrogate* objective (fixed
  travel times + linearized F2) and confirm the MILP returns the same minimizer. Isolates bugs in
  the MILP encoding (constraints, `c_e^k`, makespan linearization). Uses the *same fixed* travel
  times as the MILP.
- **Level B — approximation quality.** Brute-force the *true* `F` (full UE pipeline) and compare
  `F(x_milp*)` to the true optimum `F*`. Tests whether traffic-fixation + linearization actually
  yield a (near-)optimal schedule for the real problem.

**Current decision:** the oracle does **Level B only** (true objective). Level A is parked until the
MILP exists.

## 2. Comparison metrics (MILP vs oracle)  [point 5, parked]
When comparing the future MILP solution `x_milp*` against the oracle:
- **objective gap**: `F(x_milp*) − F*` (relative); the headline acceptance number.
- **rank**: where `x_milp*` falls among all enumerated schedules ("#1 of 24" / "top X%").
- **schedule match**: does `x_milp* == x*`? (If not but `F` equal → alternative optima; compare `F`,
  not the argmin.)
- **landscape shape**: is the optimum a sharp needle or a broad plateau? (from `oracle_landscape.csv`)
  → use this to set a *reasonable* acceptance tolerance for the MILP.
The data to compute all of these is saved in `outputs/oracle/oracle_landscape.csv` (every tested `x`
with its F/F1/F2).

---

## 3. Figure-1 fidelity (the milestone) and where it is incomplete
`util/evaluate.py` follows Figure 1 step-for-step (auditable via `main.py`). Deviations only where
Figure 1 is silent/incomplete:
- **F2 is computed outside the per-step loop** (it needs no UE — pure schedule arithmetic).
- **Baseline `u_r^{t0}`** = one UE on the undamaged network with normal demand `H^{t0}` (Fig. 1 uses
  it in F1 but doesn't show how it's obtained).
- **Demand specifics** (Eq. 4 is only `H=A·H+B·v`): see §4.
- **F1 can dip below 1.** Because F1 is *demand-weighted* and demand drops after the disaster, fewer
  travellers experience the (congested) normal-demand baseline `u_r^{t0}`; during the low-demand
  recovery window the demand-weighted ratio can fall below 1. The paper's "=1 when fully restored"
  holds only once *both* network and demand are back to normal. Faster restoration still lowers F more
  (more time in the restored regime), so the optimization signal is intact.

## 4. Demand model (drop → recover)
Open-source OD = normal-time demand = `H^{t0}` (also the fully-recovered level). The literal
`H_t = A·H_{t-1} + B·v_t` with `A=ρI` decays to 0 (wrong). We model the **shortfall**:
```
D_t = max(B·v_t, ρ·D_{t-1}) ,   H_t = max(0, H^{t0} − D_t) ,   D_0 = 0
B[r,e] = κ·(h_r^{t0}/3)·1{e on a free-flow shortest path of OD r}
```
→ **sharp drop at onset** (to the current damage-driven shortfall `B·v`), then **gradual recovery**
to normal at rate ρ as roads heal. With the defaults below the onset drop is ~20% of total demand;
tune κ for depth, ρ for recovery speed (inspect `outputs/oracle/figures/03_best_schedule.png`).

## 5. Caveats
- **Work-conserving reduction.** Schedules are permutations → list scheduling (no idling). Assumes
  idling never helps F1 or F2 (true under monotonicity; with dynamic demand the demand-coupling makes
  this only *approximately* guaranteed — re-check if results look off). Keeps enumeration at `|ℰ|!`.
- **No UE cache.** Dynamic demand ⇒ UE depends on the recovery path ⇒ the `2^|ℰ|` completed-set cache
  does not apply; every step is a fresh UE solve.
- **AequilibraE is ~1.2 s/UE-solve** here (per-call setup overhead on a 24-node net). That capped the
  brute force at a small instance (4 segments, M=10 ≈ 2 h). A fast in-house Frank-Wolfe would lift
  this limit if a larger instance is needed later.

## 6. Assumed-parameter defaults (mirror of `util/params.py`)
| param | default | meaning |
|---|---|---|
| A (inertia) | ρ = 0.7 | recovery rate of the demand shortfall |
| B(Φ) | `κ·(h_r^{t0}/3)·1{e on r's free-flow SP}`, κ = 1.0 | damage → demand-shortfall; tune κ for drop depth |
| severity→capacity retain | {1:0.7, 2:0.4, 3:0.1} | damaged capacity = retain × capacity |
| severity→speed retain | {1:0.8, 2:0.6, 3:0.4} | damaged free_flow_time = t0 / retain |
| μ | 0.5 | F = μ·F1 + (1−μ)·F2 |
| u_pen | 10 × max baseline OD travel time | disconnected-OD penalty |
| C_max | 2 | crews |
| Δt | 3 h | given |
| T | global max makespan over all (schedule × scenario) | horizon |
| M / seed | 10 / 42 | scenarios |
| UE rgap / max_iter | 1e-6 / 100 | per-step UE convergence (looser than the 1e-12 validation run) |
| instance | 4 disrupted segments (subset of the 8) | kept small due to AequilibraE per-call cost |
