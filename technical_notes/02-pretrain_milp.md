# 02 — `run_pretrain_milp()`: the solving logic of the traffic-fixation MILP

This note explains **how** the pretraining solver finds a start-time schedule for each repair-duration scenario, and how that result is then compared against the brute-force oracle. The entry point is `run_pretrain_milp()` in `util/pretrain_milp.py`, triggered directly by the command line `python -m util.pretrain_milp`. The note is organized in the code's true solving order: each section is one logical step, which first states the problem that step must solve, then what it does and why it is done that way, and finally which later step consumes its product, closing with path:line anchors so every claim can be checked against the current source. Third-party libraries (AequilibraE, networkx, scipy) are summarized by their role only, without descending into their internals. Project parameters are provided uniformly by the root-level `config.py` (referenced as `P` in the code), and the default values are collected in Appendix A.

---

## Overview

The problem to solve is this: a flood has broken a number of road segments, each of which occupies one crew for a consecutive run of time slots to repair; a schedule — at which slot each segment starts — must be found that minimizes the degradation of network accessibility over the recovery period (objective $`F_1`$). The value of $`F_1`$ is the average, over the whole recovery period, of the per-slot "demand-weighted ratio of each OD pair's current travel time to its pre-disaster baseline". 

The difficulty lies in a coupling: $`F_1`$ is built from each slot's user-equilibrium (UE) travel times. Those travel times depend on which roads are still broken at that slot — that is, on the schedule itself. Feeding this coupling unaltered into an optimizer would mean a nonlinear objective with repeated UE solves inside every single evaluation, at an unacceptable cost.

The pipeline instead uses alternating optimization (traffic fixation), one round of which has four phases. 
- First, the OD travel times over the whole horizon (the unified evaluation time window, spanning $`T`$ slots) are **frozen** into constants — for now it suffices to read this as "take the previous round's UE result". 
- Second, under the frozen travel times, the marginal improvement to $`F_1`$ from starting segment $`e`$ at slot $`k`$ is computed analytically as the coefficient $`c_e^k`$, without touching UE at all. Writing $`y_e^k`$ for the 0-1 decision variable "does segment $`e`$ start at slot $`k`$", what the MILP optimizes is then no longer $`F_1`$ itself but a linear surrogate of "$`F_1`$'s decrease relative to repairing nothing", $`\sum_e \sum_k c_e^k y_e^k`$; because $`c_e^k`$ measures improvement, the larger this surrogate the better, so the third phase's **maximization** of it points in the same direction as minimizing $`F_1`$. 
- Third, a very small start-time MILP is solved that maximizes the surrogate under the scheduling constraints, producing a new schedule. 
- Fourth, a true evaluation is run for this new schedule only (one UE per slot), refreshing the travel times, and the loop returns to the first phase. This iterates until the schedule stops changing or a guard fires. The starting point of the iteration is not arbitrary: by default it departs from the flow-greedy baseline's schedule (warm start, see Step 2), which — combined with the best-by-true-$`F`$ output rule (Step 7) — guarantees that the final output is no worse than that baseline.

Left unconstrained, the raw alternation **oscillates**: repairing a segment decongests exactly the routes that made it look worth repairing, so the MILP's choice keeps flipping among a few mutually-best-response schedules and never settles. The pipeline therefore adds two cooperating **diminishing stabilizers** — one acting on the continuous traffic state (diminishing damping) and one on the discrete schedule (a decaying proximal penalty) — which together force the would-be cycling iteration into a fixed point; the mechanics are in Steps 4 and 6.

---

## Step 1 · The problem definition fully shared with the oracle (see note 01)

### 1 · Problem

The MILP's results are to be compared directly against the brute-force oracle, so the two must face **exactly the same** problem — before any MILP-specific logic begins, the damaged instance, the static context, the scenario sampling, the horizon, and the true-objective evaluator must all be in place, and they must be precisely the set the oracle uses, computed by exactly the same functions; at the same time, it must be guaranteed that the oracle's already-computed "answers" can never flow back to help the MILP.

### 2 · Approach

run_pretrain_milp defines the problem with exactly the same building blocks as run_oracle. Note 01 has already traced each of them, so this note only lists them with pointers, without restating them:

- Selecting the damaged instance (select_oracle_instance: rank importance by baseline UE flow, and assign severity 3 to the two most critical edges to create true disconnections): see **note 01, Step 1 (§1a)**;
- Building the static context once — the network table, the OD pairs, the pre-disaster baseline travel times $`u_r^{t_0}`$ (the subscript $`r`$ ranges over OD pairs; the superscript $`t_0`$ denotes the pre-disaster instant), the disconnection penalty $`u_{\text{pen}}`$, and the sensitivity matrix $`B`$ that maps "which roads are broken" into "which OD demand is suppressed": see **note 01, Step 1 (§1b–1f)**;
- Sampling $`M`$ repair-duration scenarios (drawn per level, fixed seed): see **note 01, Step 2** — because the seed matches the oracle side, both ends receive identical scenarios one for one, which is one precondition for $`F`$ being comparable;
- Unifying the global horizon $`T`$: see **note 01, Step 4**;
- The true-objective evaluator evaluate_schedule (one UE per slot, computing exactly $`F=\mu F_1+(1-\mu)F_2`$; $`\mu`$ is the blending weight of the two components and defaults to 1.0, so in this project $`F`$ equals $`F_1`$, while $`F_2`$ — the component measuring repair-timeline efficiency — carries zero weight yet is still computed and recorded): see **note 01, Step 5 (5a–5f)**.

(util/pretrain_milp.py:272-276 invoking the shared building blocks; util/pretrain_milp.py:347-358 oracle solutions read only at the end; util/pretrain_milp.py:269 and util/oracle.py:77-82 per-scale output directories)

---

## Step 2 · Warm start: the flow-greedy baseline as the initial schedule, producing the first frozen travel times

### 1 · Problem

Two problems are solved here. First, the opening iteration has no UE result to freeze yet, so an initial schedule and its whole-horizon travel times must exist first. Second, the quality of the initial point is not a matter of indifference. It was observed that cold-starting from an arbitrary order (plain edge-id order), can converge to a schedule worse than the strongest static baseline. The initial point must therefore itself carry a quality floor. "Strongest" here is an empirical finding: flow-greedy is the best of the three static greedy rules.

### 2 · Approach

With MILP_WARM_START on (the default), run_pretrain_milp first sorts the damaged segments from high to low by their two-way baseline UE flow on the intact network — the very same edge-criticality measure that note 01 §1a uses to select the damaged instance, yielding an order identical to the flow-greedy baseline's priority order (note 03, Step 1) — and passes this order as warm_order to alternating_optimize; the latter hands it to work-conserving list scheduling to obtain the round-0 initial schedule. The schedule is then evaluated by the true evaluator (with return_u), yielding the initial $`F`$ and the first frozen travel times.

### 3 · Interpretation · what does warm start guarantee, and at what cost?

- the final output is in the end picked from the whole history by true $`F`$ (Step 7), and the initial schedule is part of that history, so the output is guaranteed no worse than the initial point; warm start raises that floor from "a schedule in arbitrary order" to "the strongest static baseline's schedule", so that each scenario's MILP result is **provably no worse than the flow-greedy baseline**, and the alternating refinement can only improve on it. 
- This guarantee does not depend on whether the iteration converges, which is exactly what plugs the cold-start hole of "converging to something worse than the baseline" on large instances. 
- The cost is only one extra baseline UE on the intact network per run (used to compute the flow ranking), negligible against the $`T`$ UE solves of every round's evaluation. The warm-start switch is also recorded into the resumable-checkpoint fingerprint, so that old checkpoints cannot be misused after the setting changes (pure engineering; details omitted).
(util/pretrain_milp.py:282-287 computing the flow ranking; util/pretrain_milp.py:219-228 initial schedule, initial evaluation, and the proximal strength base $`\gamma_0`$ — the latter defined in Step 4; config.py:80)

---

## Step 3 · The analytic sensitivity coefficients c_e^k: a linear surrogate that runs no UE

### 1 · Problem

The MILP needs a linear objective, but the true $`F_1`$ depends on the schedule through UE and is nonlinear. Under the premise that travel times are frozen, the question "if segment $`e`$ starts at slot $`k`$, what is its marginal improvement to the per-slot terms of $`F_1`$" must be answered analytically — and it must be answered without a single UE run.

### 2 · Approach

precompute_c first computes, from the frozen travel times and the baseline, the **congestion excess** of every slot and every OD

```math
\alpha_r^{k'} = 1 - \frac{u_r^{t_0}}{\tilde{u}_r^{k'}}
```

(the subscript $`r`$ ranges over OD pairs and the superscript $`k'`$ over slots; $`\tilde{u}_r^{k'}`$ is OD pair $`r`$'s travel time, and $`u_r^{t_0}`$ is the pre-disaster baseline put in place in Step 1), and then accumulates in closed form over every segment $`e`$ (duration $`d_e`$) and every feasible start slot $`k`$. 

A segment starting at slot $`k`$ with duration $`d_e`$ is under repair from slot $`k`$ through slot $`k + d_e - 1`$ and is open to traffic from slot $`k + d_e`$ on; a feasible start must satisfy $`k + d_e \le T`$, that is, at least one open slot must remain within the horizon. The accumulation is

```math
c_e^k = \frac{1}{T} \sum_{k'=k+d_e}^{T} \big(1 - \rho^{\,k'-k-d_e+1}\big) \sum_r B_{r,e}\, v_e^{\ast}\, \alpha_r^{k'}
```

where $`v_e^{\ast}`$ is the segment's severity (taken from the context's severity vector), so that $`B_{r,e}\, v_e^{\ast}`$ is "the OD-$`r`$ demand released back onto the network when $`e`$ is fully repaired"; $`\rho \in (0,1)`$ is the recovery-inertia coefficient (default 0.7, the same parameter shared with the demand model of note 01 §5c; its role in this formula is explained below). Coefficients of infeasible starts ($`k > T - d_e`$) are left at 0 and separately prohibited by the MILP's variable upper bounds. 

Reading the rest of the formula term by term:

- $`\sum_r B_{r,e}\, v_e^{\ast}\, \alpha_r^{k'}`$ is the payoff of having $`e`$ open at slot $`k'`$. The product $`B_{r,e}\, v_e^{\ast}`$ needs both factors because $`B_{r,e}`$ is a slope rather than a quantity: it is the demand that OD $`r`$ loses per unit of severity of $`e`$, and $`v_e^{\ast}`$ supplies the units of damage that $`e`$ actually carries. Multiplying by the congestion excess $`\alpha_r^{k'}`$ counts that demand only to the extent that OD $`r`$ is still slower than its baseline at $`k'`$; a negative $`\alpha`$ is a legitimate signal, meaning the emptier network is faster than the baseline there and the marginal payoff of repairing is then negative.
- $`1 - \rho^{\,k'-k-d_e+1}`$ is the fraction of the released demand that has actually returned by slot $`k'`$. Writing $`n = k'-k-d_e`$ for the number of slots elapsed since completion, the factor reads $`1-\rho^{\,n+1}`$; it grows from $`1-\rho`$ at the first open slot toward 1, at the same rate at which the shortfall decays by $`\rho`$ in the true demand model of note 01 §5c. (An early version of the code wrote this factor as decaying by powers of $`\rho`$, the reverse direction; the current formula is the corrected one.)
- $`\sum_{k'=k+d_e}^{T}`$ accumulates the payoff over the open slots only, because the segment contributes nothing before its completion.
- $`1/T`$ aligns the sum with $`F_1`$'s average over the whole horizon, so the coefficient lives on the same scale as $`F_1`$'s per-slot terms.

### 3 · Interpretation · how does the surrogate $`c_e^k`$ help the true $`F_1`$?

$`c_e^k`$ is presented as a segment's improvement to $`F_1`$, yet its formula does not follow from the true $`F_1`$ by any exact algebra. The question is in what conceptual sense it helps at all.

Start from what $`F_1`$ really is. Each slot contributes a demand-weighted ratio of current to pre-disaster travel time, $`\mathrm{term}_{k'} = \sum_r H_r \tilde{u}_r \big/ \sum_r H_r u_r^{t_0}`$, where $`H_r`$ is OD $`r`$'s current demand. Repairing a segment moves this ratio through two channels:

- the network channel, in which the repair decongests the network, so the travel times $`\tilde{u}_r`$ fall toward the baseline and the ratio drops. This is where the real accessibility gain lives.
- the demand channel, in which the repair lets the suppressed demand return, so the weights $`H_r`$ grow.

The useful movement is the network channel, but that is exactly the channel the freezing premise removes, because a changed $`\tilde{u}_r`$ can only be found by running UE. The surrogate's idea is to stand in for that unseen channel with one assumption, that the demand a repair releases will, once the segment is open, travel at the baseline time rather than at the frozen congested time. Under this assumption a repair is credited with pulling its released demand down to the baseline, and that credit is exactly the summand $`B_{r,e}\, v_e^{\ast}\, \alpha_r^{k'}`$ read term by term in the Approach, namely the released demand $`B_{r,e}\, v_e^{\ast}`$ times how far that OD still sits above the baseline $`\alpha_r^{k'}`$. Conceptually, then, $`c_e^k`$ is a frozen-state proxy for the true decongestion it cannot compute. In sum, $`c_e^k`$ is a heuristic direction pointer, not an exact linearization of the term. Its error is corrected every round by the fourth phase's true evaluation, and output quality is backstopped by Step 7's best-by-true-$`F`$.
(util/pretrain_milp.py:78-109)

---

## Step 4 · The start-time MILP: the surrogate plus a decaying proximal penalty

### 1 · Problem

In the frozen-travel-time world, find the start schedule that maximizes the surrogate while obeying the scheduling rules: each segment starts exactly once, the number simultaneously under repair never exceeds the crew count, and every start satisfies Step 3's feasibility convention $`k + d_e \le T`$. Inside the alternating loop the solve carries one further ingredient, a proximal penalty that discourages the new schedule from drifting from the previous round's without payoff; it executes here, at the solve, so it is described here, while the reason it exists is a convergence question answered in Step 6.

### 2 · Approach

The decision variables $`y_e^k \in \{0, 1\}`$ mean "does segment $`e`$ start at slot $`k`$" (notation matching the source), $`|E|\cdot T`$ of them in total ($`E`$ being the set of damaged segments and $`|E|`$ its size $`N`$). The objective is

```math
\max \sum_e \sum_k c_e^k\, y_e^k
```

(the implementation negates the coefficients and minimizes, because scipy.optimize.milp's interface accepts only minimization). Inside the alternating loop, build_and_solve_milp also receives the previous round's schedule $`y_{\text{prev}}`$ and a penalty strength $`\gamma`$, so the objective actually solved in round $`n`$ carries one more, likewise linear, term

```math
\max \; \sum_e \sum_k c_e^k\, y_e^k \;-\; \gamma_n \, \lVert y - y_{\text{prev}} \rVert_1, \qquad \gamma_n = \frac{\gamma_0}{n+1}
```

Term by term: the first sum is the surrogate above; $`\lVert y - y_{\text{prev}} \rVert_1`$ measures how far the new schedule $`y`$ moves from the previous round's $`y_{\text{prev}}`$, and it enters with a negative sign because the objective is maximized; $`\gamma_n`$ is the penalty strength, decaying as $`1/(n+1)`$; $`\gamma_0`$ is the base strength, `MILP_PROX_SCALE` times the median over segments of the across-slot range (max minus min) of $`c`$ on each segment's feasible slots, which puts the penalty on the same scale as $`c`$ itself; in the code it is computed once before the main loop, right after Step 2's initial evaluation from the first frozen travel times' coefficients, and never updated afterwards (util/pretrain_milp.py:226-228 for $`\gamma_0`$; util/pretrain_milp.py:235-236 for the per-round $`\gamma_n`$; config.py:72-79 for MILP_PROX_SCALE).

The penalty stays linear: writing $`\hat{s}_e`$ for segment $`e`$'s start slot in the previous round, the start-once constraint gives $`\sum_k \lvert y_e^k - y_{\text{prev},e}^k \rvert = 2\,(1 - y_e^{\hat{s}_e})`$ per segment (0 if it keeps its slot, 2 if it moves, one unit at each of the old and new positions), hence $`\lVert y - y_{\text{prev}} \rVert_1 = 2N - 2\sum_e y_e^{\hat{s}_e}`$, twice the number of segments that moved. The constant $`2N`$ does not affect the optimum, so the implementation simply adds a reward of $`2\gamma_n`$ on each previous-start-slot variable ($`-2\gamma_n`$ coefficient in the minimization form), encouraging the solver not to flip a start unless it pays (util/pretrain_milp.py:132-136). 

The constraints fall into three classes:

- start once: $`\sum_{k=1}^{T} y_e^k = 1`$ for every segment;
- crew cap: for every slot $`k`$, $`\sum_e \sum_{k'=\max(1,\,k-d_e+1)}^{k} y_e^{k'} \le C_{\max}`$, where the inner sum enumerates the start slots whose repair is still in progress at $`k`$, and $`C_{\max}`$ is the number of crews (default 2);
- horizon bound: variables with $`k > T - d_e`$ have their upper bound forced to 0. Step 3 leaves those positions' coefficients at 0, but a zero coefficient alone does not forbid such a start, because feasible positions can carry negative coefficients and zero could then be the maximum of that segment's row; the variable bound is what actually forbids it.

The problem is handed to scipy.optimize.milp (HiGHS branch-and-bound underneath, shipped with scipy, no license needed), which raises on failure; each segment then takes the slot where its row holds 1, reconstructing the mapping from edge to start slot (util/pretrain_milp.py:115-165). A small utility also computes, in reverse, the surrogate value $`\sum_e c_e^{s_e}`$ of an arbitrary given schedule, with $`s_e`$ that schedule's start slot for $`e`$; it serves the per-round trace and the encoding check (util/pretrain_milp.py:168-172).


---

## Step 5 · Three stopping conditions

The surrogate is an approximation, so the iteration guarantees neither monotonicity nor convergence, and it may fall into cycles; explicit stopping rules are needed that both recognize true convergence and bound the non-convergent cases.

Each round of the main loop computes the coefficients, solves the MILP, runs the true evaluation, and records the trace, then applies a three-tier stopping rule, recording the manner of stopping in an `outcome` field:

- fixed point: the new schedule is identical to the previous round's; set `converged=True` and `outcome="fixed_point"`, then exit.
- counting cycle guard: the occurrences of every schedule seen are counted, and once any schedule accumulates MILP_CYCLE_TOL (3, config.py:62) occurrences the loop exits with `outcome="cycle"`. The threshold is 3 rather than "stop at the first repeat" because the two diminishing stabilizers still change the proximal pull (Step 4) and the frozen travel times (Step 6) every round, so a reappearing schedule faces different coefficients, and another chance or two may break a small 2-cycle or limit cycle.
- iteration cap: if the loop completes MILP_MAX_ITER (20, config.py:61) rounds without stopping, the default `outcome="iter_cap"` is kept.

Only fixed_point sets `converged=True`; cycle and iter_cap both count as non-converged (util/pretrain_milp.py:230-247).

---

## Step 6 · The end-of-round damping update

### 1 · Problem

The raw alternation (adopt the new UE travel times wholesale, let the MILP re-pick the schedule freely every round) does not neccessarily converge. Repairing a segment decongests exactly the routes that made it look worth repairing, the frozen travel times flip accordingly, $`c_e^k`$ jumps, and the MILP solution swings among a few mutually-best-response schedules (a congestion-feedback cobweb). There are two coupled oscillation sources, the continuous traffic state and the discrete schedule, and true convergence requires pinning both. 

Also, a **fixed-weight** damping is not enough. While the schedule keeps flipping between a few solutions, the travel times it induces jump between a few fixed values by an amount that never shrinks on its own, and a constant weight only rescales each jump by the same fraction, so the frozen travel times move by a fixed step every round and settle into a steady nonzero oscillation instead of converging.

### 2 · Approach

The code answers with two **diminishing** stabilizers, one per oscillation source. The one acting on the discrete schedule, the decaying proximal penalty, already executed earlier in the round, inside Step 4's MILP solve. This step is the other one, the diminishing travel-time relaxation acting on the continuous traffic state (`MILP_DAMPING_MODE = "decay"` in config). At the end of each round the frozen travel times are not replaced outright but updated by the convex combination

```math
\tilde{u} \leftarrow \lambda_n\, \tilde{u}_{\text{new}} + (1 - \lambda_n)\, \tilde{u}_{\text{old}}, \qquad \lambda_n = \frac{1}{n+1}
```

Term by term: $`\tilde{u}_{\text{new}}`$ is the travel-time matrix from this round's true evaluation; $`\tilde{u}_{\text{old}}`$ is the frozen matrix the round just used; $`\lambda_n`$ is the weight given to the new result, with $`n`$ the 1-based iteration index, so round 1 mixes at $`1/2`$, round 2 at $`1/3`$, and the step vanishes over the rounds (util/pretrain_milp.py:248-252). The `"const"` mode (fixed weight `MILP_DAMPING`) is kept for experiments; alternating_optimize's `damping` argument takes effect only in that mode, and the main pipeline uses `"decay"` (config.py:64-70).

---

## Step 7 · best-by-true-F: a selection rule harmless whether or not the loop converged

Even once the iteration stops, the last round's schedule need not be the best seen along the way, because the surrogate-driven iteration is not monotone and can worsen late; "which one to output" must be decided.

Every round's schedule and true $`F`$ (including round 0's greedy initial) remain in the history; upon exit, the minimum by **true $`F`$** over the entire history is taken as the scenario's final output, and that round is flagged is_best in the trace (util/pretrain_milp.py:255-258).

---

## Step 8 · Solving per scenario and aggregating the results

The $`M`$ scenarios are solved independently of one another; once all are done, the whole run's results must be written to disk together with a summary.

For each scenario, run_pretrain_milp calls the optimization once (Steps 2–7) and assembles the best result into one row: the scenario number, $`F_{\text{MILP}}`$, $`F_1`$, $`F_2`$, the iteration count, converged, outcome, the elapsed time and the number of UE solves consumed, the duration combination, and every segment's start slot. The scenario's per-round trace is appended to the trace rows (util/pretrain_milp.py:307-328). After all scenarios finish, two tables are written: milp_optima.csv (each scenario's final schedule and objective values, including the converged and outcome columns) and milp_trace.csv (the full per-round trace) (util/pretrain_milp.py:331-333).

---

# Caveats

## C1 · best-by-true-F is a safety net, but at very small scale it approximates a miniature enumeration

Step 7's best-by-true-F returns the smallest-true-$`F`$ iterate over the whole history, so the output has a floor guarantee however the loop exits, and convergence becomes a question of efficiency rather than correctness. The stabilizers of Steps 4 and 6 make the loop genuinely converge, but best-by-true-F is a safety net independent of that, which is where the caveat comes in.

On a very small instance this rule quietly approaches a brute-force search. At $`n=4`$ there are only $`4! = 24`$ permutations, while one scenario's history holds up to 21 iterates (the initial schedule plus the cap of 20 rounds), so best-by-true-F is picking the best of a set comparable in size to the whole space, and a good final $`F`$ cannot then be told apart from the history simply happening to cover the optimum. On this instance the comparison of C2 should therefore be read only as "no worse than the oracle's work-conserving optimum", not as evidence of how strong the surrogate guidance is. The effect vanishes once $`N!`$ outgrows the fixed iteration cap, and only then can surrogate guidance and convergence behavior be judged on their own.
(util/pretrain_milp.py:255-258 the selection rule; the per-scenario converged, outcome, n_iter columns are in outputs/pretrain_milp/n4/milp_optima.csv)

## C2 · The post-hoc oracle comparison: how the gap is defined, and why it can be negative

This comparison measures how far the MILP is from ground truth. The oracle has enumerated every work-conserving schedule per scenario and kept its optimal $`F`$, and under Step 1's no-leakage boundary these are read only after all MILP solving is done. Merging them by scenario number defines

```math
\text{gap} = F_{\text{MILP}} - F_{\text{oracle}}
```

where $`\text{gap} \le 0`$ means the MILP matched or beat the oracle's best work-conserving schedule. milp_vs_oracle.csv is written, along with two figures:

- 01_milp_vs_oracle.png shows the two $`F`$ values and the gap per scenario;
- 02_cross_scenario_landscape.png ranks every fixed permutation (one repair-priority order held constant across scenarios) by its across-scenario mean $`F`$ as a curve with a range band, overlaid with the MILP's per-scenario-adaptive mean and the hindsight mean, that is the mean of the oracle's per-scenario optima. The hindsight line is the floor of the work-conserving space, and the MILP can fall below it because its feasible set is a strict superset of that space, which is the negative-gap point below.

If the oracle is not ready, the comparison is skipped, and `python -m util.pretrain_milp --landscape` re-renders it later from the saved CSVs. That command still runs one baseline UE to rebuild the instance, but none of the expensive per-slot evaluation UEs.

Why the gap can be negative. The oracle's optimum is only the best **work-conserving** schedule, whereas Step 4's MILP may leave crews idle, so its feasible set is a strict superset and it can find a better schedule outside that set. This is not a bug but the effect of a larger search space. The level_a check (run manually via --level-a) guards against an encoding error. With the proximal penalty off ($`\gamma = 0`$), the MILP's surrogate optimum must be at least the best surrogate value over all work-conserving schedules, and the crew-cap and horizon constraints are re-checked independently.
(util/pretrain_milp.py:347-365, util/pretrain_milp.py:375-416, util/pretrain_milp.py:419-438, viz/pretrain_viz.py:35-137; the one baseline UE inside `--landscape` is triggered by the instance selection at util/pretrain_milp.py:425, see util/oracle.py:120 and util/oracle.py:99-100)
