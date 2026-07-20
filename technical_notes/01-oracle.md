# 01 — `run_oracle()`: the solving logic of the brute-force oracle

**The overall problem the oracle solves.** For a set of damaged road segments, the oracle enumerates **all** work-conserving repair schedules (work-conserving means no repair crew is ever left idle while a segment still awaits repair; under this rule the schedule space is exactly the set of permutations of these segments) and, under $M$ randomly drawn duration scenarios (a duration scenario is one joint realization of the repair durations of all damaged segments), computes the exact objective $F(x \mid \omega)$ for every schedule, where $x$ is a schedule and $\omega$ is a scenario. The overall objective is the weighted combination

$$F = \mu F_1 + (1-\mu) F_2$$

in which $F_1$ measures how much accessibility degrades over the recovery period — computing it requires solving a traffic equilibrium at each time slot — and $F_2$ measures the efficiency of the repair timeline and is pure scheduling arithmetic. The run produces two kinds of result: for each scenario, the hindsight optimum $F^\ast$, and the complete landscape, i.e. every schedule that was evaluated together with its $F$, $F_1$, and $F_2$. Because the enumeration is exhaustive and each evaluation is exact, this result is ground truth, later used to judge the quality of the schedules produced by a cheaper approximate solver (this project's pretraining MILP, see note 02). The entry point is `run_oracle()`, defined at util/oracle.py:152.

The instance is deliberately kept small (the number of damaged segments $N=4$ and the number of scenarios $M=10$), because each traffic-equilibrium solve takes on the order of a second while the number of schedules to evaluate grows as $N!$, so exhaustive search is only feasible on a small instance. This note keeps only the steps that bear directly on the modeling logic, written out as seven steps in execution order, followed by a single appendix (the parameter-default table). Pure engineering machinery — caching, resumable checkpointing, run timing — does not affect how the objective is defined and is therefore left out.

---

## Step 1 · Establishing the problem instance and the static background

The problem this step solves is to fix down "exactly which segments are damaged, and how severely each is damaged," and to compute once all the background quantities that every later evaluation needs but that do not depend on the particular schedule. Because the amount of evaluation to follow is on the order of "number of permutations × $M$ scenarios × $T$ slots," recomputing this background inside the enumeration loop would waste a great deal of time, so it must all be prepared before the enumeration begins. In code this step is done by two calls: first `select_oracle_instance` chooses the damaged instance, then `build_context` packs the static background around that instance into a dict for later reuse. It is worth noting that these two calls **each** perform a baseline user equilibrium on the intact network — the former to obtain each edge's flow so as to rank importance, the latter to obtain the OD pairs' baseline travel times; the two solves take the same input and are therefore a duplicated computation, recorded here faithfully and left for separate handling at the code level.
(util/oracle.py:156 and util/oracle.py:158 called in turn; build_context is defined at util/evaluate.py:127-184)

### 1a · Choosing the damaged instance: which segments are damaged, and how severely

The problem here is that the set of damaged segments cannot be chosen arbitrarily. If all damaged segments were of comparable importance, then the repair order would have only a weak effect on $F_1$, the enumerated landscape would be too flat to distinguish good schedules from bad ones, and the oracle would fail to serve as ground truth. This step therefore **deliberately mixes critical segments with minor ones**, so that "which to repair first" has a substantial effect on the objective.

The approach is to first assign each undirected edge an importance score: on the intact network under normal demand, a baseline user equilibrium is solved by the project's own UE engine (UE denotes the flow pattern in which no traveler can shorten their own travel time by unilaterally switching route), and each edge's two directional equilibrium flows are summed to give that edge's importance. The edges are then sorted by flow from high to low, and the two highest-flow edges are designated "critical segments" and assigned severity 3; because severity 3 already reaches the threshold at which the segment is removed from the network entirely, leaving these two unrepaired for long creates a genuine disconnection that is heavily penalized in the objective, so that the quality of the repair order is markedly amplified. The remaining slots are distributed at even intervals over the lower-flow edges, starting from roughly the 20th percentile of the flow ranking down to the lowest-flow edge, with severity alternating between 2 and 1 so as to contrast with the critical segments. Each damaged segment is then given a level label of the form highway-S3 (a level is the combination of road_class and severity), and after sorting by edge_id the result is written as a CSV into the disruption/ subdirectory under the toy data directory, while the damaged-segment table is returned. The whole procedure is fully deterministic and uses no random numbers.

The reason baseline UE flow is used as the importance proxy is that the higher an edge's flow, the greater the impact on network-wide accessibility when it is cut. It is worth stressing that this baseline flow is now computed on the fly by the project's own UE engine, rather than read from the open-source reference-solution file raw/SiouxFalls_flow.tntp, so that switching to another network or OD dataset in the future no longer requires shipping a reference-flow file alongside it; it has been verified that switching to self-computed flow selects exactly the same instance as the old reference solution (edge_id 1, 12, 15, 17 with severity 1, 2, 3, 3, with per-edge flows differing by about 0.14%). The product of this step is that damaged-segment table, which is immediately handed to 1b–1f to build the context, to Step 2 to sample scenarios, and to Step 7 to draw figures; the canonical sorted list of edge_ids taken from it is the shared basis for the permutations, the horizon, and the landscape column names downstream.
(util/oracle.py:109-136 selecting edges and assigning severity; util/oracle.py:85-106 self-computing the two-way baseline flow)

### 1b · Loading the network and building index maps

Three CSV files are read from the toy data directory: the undirected edge table (with capacity, length, free_flow_time, the BPR parameters, and road_class), the OD-pair table (with normal-period demand), and the node table; in Sioux Falls every node is also an OD zone. The node ids, OD pairs, and edge ids are then mapped to dense, zero-based indices, and the normal demand is extracted from the OD-pair table as the vector $H^{t_0}$, which is at the same time the target level corresponding to "full recovery." The maps are built first because both the UE solve and the matrix construction operate on dense indices, so preparing the maps up front avoids repeated lookups inside the main loop. The product is the skeleton of the context: the edge table, the zone list, the OD pairs, the normal demand, and the various index maps.
(util/io.py:17-40 loading; util/evaluate.py:136-147 mapping and skeleton)

### 1c · The pre-disaster travel-time baseline: one UE on the intact network

$F_1$ is the ratio of "realized travel time to the pre-disaster baseline travel time," so it needs a denominator, namely each OD pair's **congested** travel time $u_r^{t_0}$ under normal conditions (intact network, normal demand). The approach treats the UE solver as a black box: given an edge table and an OD demand matrix, it returns each directed link's equilibrium flow and congested travel time (internally this is AequilibraE's bi-conjugate Frank-Wolfe algorithm; at the project level we depend only on this input-output contract and do not go into its internals). One such UE solve is run on the original edge table under normal demand, and then, treating the equilibrium link costs as edge weights, a single-source shortest-path computation is run from each origin to obtain each OD pair's travel time; a pair whose origin and destination are disconnected is recorded as infinite. This baseline must come from the same UE model as the post-disaster evaluation, because only travel times computed by the same model are comparable before and after the disaster; and since it depends on neither the damaged instance nor the schedule, it is computed just this once and reused throughout. The product is the baseline travel-time vector aligned to the OD pairs, which serves both as the denominator of each slot's $F_1$ term and as the magnitude source for the next step's disconnection penalty.
(util/evaluate.py:149-153 baseline UE; util/evaluate.py:31-53 converting link times to OD travel times, a subroutine that the per-slot evaluation also reuses)

### 1d · The disconnection penalty u_pen

The damaged network may contain OD pairs that are completely disconnected, with an infinite shortest-path distance that cannot enter the $F_1$ computation directly, so a finite but sufficiently large cost is needed to replace the infinity. The approach precomputes a penalty value equal to 10 times the "largest finite pre-disaster OD travel time" (the multiplier is given by the parameter UPEN_FACTOR), and substitutes it at evaluation time whenever an OD pair is disconnected. This value is chosen so because a disconnection must be worse than any "severely congested but still connected" state; otherwise "leaving a critical segment unrepaired for a long time" would not incur the heavy penalty it deserves in the objective. At the same time, taking a fixed multiple of the worst pre-disaster travel time keeps the units consistent with the denominator of $F_1$ and makes the penalty magnitude interpretable. The product is this scalar, used in Step 5 when each slot replaces infinite travel times.
(util/evaluate.py:154-156; config.py:41)

### 1e · The demand-shortfall sensitivity matrix B

Post-disaster travel demand falls, and how far it falls should be tied to "which damaged segment lies on this OD's usual route, and how severe it is," so a static sensitivity matrix is needed that, at evaluation time, maps the current damage state linearly into each OD's demand shortfall. The approach computes, on the free-flow (uncongested) graph, one shortest path for each OD pair; if a damaged segment happens to lie on that path, the corresponding matrix entry is set to

$$B_{r,e} = \kappa \cdot \frac{h_r^{t_0}}{3}$$

and left at 0 otherwise. This is one third of that OD's normal demand, scaled by the coefficient $\kappa$ (default 1). Note that $h_r^{t_0}/3$ here is only the shortfall coefficient **per unit of severity**; the actual shortfall is obtained only in Step 5, by multiplying $B$ with the severity-carrying damage vector, so that the higher a segment's severity, the more demand falls (severity 1, 2, 3 correspond to 1/3, 2/3, and the full amount of that OD's normal demand, respectively). The reason a free-flow shortest path is used rather than a shortest path under congestion is that the former is a stable proxy for "which route this OD mainly takes in normal times," one that does not drift with the congestion state, thereby keeping $B$ purely static and precomputable. The product is the matrix $B$, used in Step 5 to generate the per-slot demand shortfall.
(util/evaluate.py:158-182; config.py:38)

### 1f · The severity vector (not used by the oracle itself)

At the end, build_context also stores a severity vector in damaged-segment order. It should be made clear that neither run_oracle nor the `evaluate_schedule` it calls ever reads this vector: the real-time severity used at each slot is taken directly from the damaged-segment records (each damaged segment's record already carries its own severity). This vector remains in the context only because the context is the same static background shared by two pipelines — this oracle and the pretraining MILP — and the latter does use it; for the oracle logic traced in this note, it is a side product that has no effect.
(util/evaluate.py:183; the actual consumer is at util/pretrain_milp.py:93)

---

## Step 2 · Sampling duration scenarios

How many slots it takes to repair a segment is uncertain: for the same set of damaged segments, differing crew efficiency and actual damage extent yield different durations. The oracle must evaluate over a batch of duration scenarios in order to reveal whether a schedule is robust and how the hindsight optimum varies across scenarios.

The approach draws $M$ scenarios with a fixed-seed random number generator. Durations are drawn not per segment but per **level** (the combination of road_class and severity): for each level, a base duration is first drawn from its support set and then multiplied by a crew-efficiency factor $\eta$ drawn from $\{0.8,\ 1.0,\ 1.2\}$, rounded to the nearest integer and floored at 1; within one scenario, all damaged segments of the same level share this single duration. Durations are drawn per level rather than per segment because the randomness of duration attaches to the "road class × severity" layer, and segments of the same class and severity have no reason to draw different durations. The seed is fixed because a determined draw order together with a fixed seed makes the whole batch of scenarios fully reproducible. The product is $M$ "segment-to-duration" maps, handed to Step 4 to fix the horizon, to the Step 5 main loop for evaluation one by one, and to Step 7 to draw figures.
(util/scenarios.py:24-52; util/oracle.py:159 the call; config.py:71-76 the support sets and η)

---

## Step 3 · Enumerating all permutations

This step exhaustively lists "all possible repair orders" as the full set of schedules to be evaluated. The approach simply takes all permutations of the sorted list of damaged-segment ids; for $N=4$ there are $4! = 24$ of them, each permutation representing a repair **priority order** that is later turned into concrete start times by the work-conserving list scheduling described in Step 4.

Enumerating permutations alone suffices on the basis of a work-conserving reduction: assuming that "leaving a crew idle" never improves the objective, the optimal schedule must correspond to some priority permutation, because once the permutation is given, the idle-free greedy assignment uniquely determines all start times; the search space thus collapses from a continuum of start-time combinations to $N!$ discrete points. It should be recognized that this assumption holds only **approximately** under the present model: because the demand trajectory carries inertia and is coupled with the damage trajectory (see 5c), one cannot in theory rule out the case where "deliberately delaying a certain repair actually improves $F_1$." The benefit accepted in exchange for this approximation is that the enumeration is held at $N!$ scale; should an anomalous result appear later, this assumption should be the first thing re-examined. The product is the permutation list, handed to Step 4 to compute the horizon and to the Step 5 main loop for evaluation one by one.
(util/oracle.py:160)

---

## Step 4 · Unifying the horizon T

$F_1$ is the average of the per-slot terms over $[1, T]$. If each schedule used a different time window, the resulting $F_1$ values would not be comparable; the enumerated schedules must therefore share one horizon, within which each of them completes. The approach is, for every permutation × scenario combination, to turn the permutation into a concrete schedule, take its last completion slot, and then take the maximum over all of them as the global $T$. The rule that turns a permutation into a schedule is exactly work-conserving list scheduling: the $C_{\max}$ (default 2) identical crews are all idle from slot 1 onward (strictly after the disaster-onset slot 0); following the priority order, each damaged segment is assigned to the earliest-idle crew, which is then occupied until that segment completes; the moment any crew becomes idle it immediately picks up the next segment, never sitting idle. The last completion slot is then the maximum, over all segments, of "start time plus duration." The maximum over all combinations is taken, rather than each schedule taking its own completion slot, because unifying $T$ is what guarantees that the "average" operation lands on the same denominator for every schedule, while taking the maximum guarantees that no schedule is truncated by the time window. The product is the integer $T$, handed to Step 5 as the upper bound of the per-slot loop and to Step 7 as the horizontal-axis range of the figures.
(util/oracle.py:139-149 taking the maximum completion slot; util/evaluate.py:95-108 list scheduling; util/evaluate.py:111-112 the completion slot; util/oracle.py:161-162 the call and the printed instance summary)

---

## Step 5 · The main enumeration loop and evaluate_schedule: the full evaluation of the exact objective

This is the body of the oracle: for every permutation under every scenario, it computes the exact $F(x \mid \omega)$. The outer loop iterates over scenarios and the inner loop over permutations; each permutation is first turned into start times by the Step 4 list scheduling, then `evaluate_schedule` is called to obtain $F$ and its components $F_1$ and $F_2$, which — together with the scenario number, the dash-joined permutation string, and each segment's start slot — are stored as one row of the landscape. Each row records every segment's start time so that the landscape not only allows the objective values to be compared but also allows any schedule to be reconstructed afterward from it (Step 7's figure 03 does exactly this).
(util/oracle.py:212-225 the outer loop)

The following 5a–5f are the internal logic of `evaluate_schedule`: given a schedule (each segment's start slot $s_e$), a scenario (each segment's duration $d_e$), and the horizon $T$, it computes the exact objective $F$ for that combination.
(util/evaluate.py:190-254)

### 5a · F2: pure scheduling arithmetic, computed before everything, needing no UE

$F_2$ measures the efficiency of the repair timeline relative to the total workload, defined as

$$F_2 = \frac{\mathrm{makespan}}{\sum_e d_e}$$

that is, the last completion slot (the makespan) divided by the total-workload slot count; the physical slot length $\Delta t$ appears in both numerator and denominator and therefore cancels. It is placed outside the per-slot loop because it depends only on the schedule's timing and not at all on the traffic state, so it can be computed in a single step. The product is the $F_2$ value, held for weighted combination with $F_1$ at the end.
(util/evaluate.py:200; util/evaluate.py:115-121)

### 5b · Per-slot loop, part one: the damage state

For each slot $k = 1, \dots, T$, one first determines which segments are still damaged at that moment. The decision rule is: segment $e$ is still damaged at slot $k$ if and only if $k < s_e + d_e$, recovering the instant it reaches its completion slot. This yields the real-time damage vector

$$v_e^{t_k} = v_e^{\ast} \cdot \mathbf{1}\{k < s_e + d_e\}$$

where $v_e^{\ast}$ is that segment's severity; a repaired segment's component is 0. This simplification is admissible because the repair process is modeled as a single step function that "holds the original severity from start to completion, then recovers fully at the instant of completion," and this is precisely the sole channel through which the schedule affects every subsequent quantity. The product is the current damaged set and damage vector, which drive 5d's network construction and 5c's demand dynamics respectively.
(util/evaluate.py:209-210)

### 5c · Per-slot loop, part two: the sharp demand drop and inertial recovery

Post-disaster travel demand first drops sharply and then climbs back gradually as the roads are repaired, so a demand trajectory is needed that is driven by the damage state and carries recovery inertia. The approach maintains a demand-shortfall vector $D$, initialized to zero and updated per slot:

$$D^{t_k} = \max\!\big(B\,v^{t_k},\ \rho\,D^{t_{k-1}}\big), \qquad H^{t_k} = \max\!\big(0,\ H^{t_0} - D^{t_k}\big), \qquad D^{t_0} = 0$$

where the max is taken componentwise. That is, the current shortfall is at least "the level the current damage maps to through $B$," while the previous slot's shortfall, decayed by the coefficient $\rho$ (default 0.7; closer to 1 means slower recovery), serves as a lower bound; the actual demand is then the normal demand minus the shortfall, floored at zero. This shortfall form is used rather than a direct recursion of the type "previous slot's demand times a decay coefficient plus a damage term," because the direct recursion decays the demand level itself — the demand would be multiplied by the decay coefficient over and over and decay steadily to zero, regardless of whether the roads are repaired — whereas the shortfall form decays "the gap to the normal level," which is exactly what gives the correct shape of "a sharp drop at onset, followed by gradual return to normal as repairs proceed." The product is the current demand vector $H^{t_k}$, which is both the travel demand for this slot's UE and the weight for this slot's $F_1$ term.
(util/evaluate.py:215-217; config.py:37 and 38)

### 5d · Per-slot loop, part three: constructing the current damaged network

This slot's UE must be solved on "the network as it stands right now," so an edge table is generated temporarily according to the current damage. The approach copies the original edge table and handles each still-damaged segment in one of two cases: a segment whose severity reaches the removal threshold (default 3) has its whole row deleted from the edge table, creating a genuine disconnection whose affected OD pairs are later counted at the penalty value; a less severe segment instead has its capacity multiplied by the capacity-retention factor and its free_flow_time divided by the speed-retention factor (both sets of factors are graded by severity, see Appendix A). Completed and undamaged segments are left as they are. This "temporarily generate an edge table per slot" approach is used rather than maintaining one continuously mutating graph, because the UE black box's entry point accepts only an edge table, so temporary generation is simplest and also keeps every solve stateless. The product is the edge table reflecting the current damage, handed to this slot's UE solve.
(util/evaluate.py:219 the call; util/evaluate.py:62-89 the construction logic)

### 5e · Per-slot loop, part four: the UE solve and this slot's F1 term

On the current damaged edge table and the reduced demand, one UE is solved (the same black box and same convergence settings as in 1c), and then each OD pair's realized travel time is obtained from the equilibrium link costs; disconnected OD pairs are substituted with the disconnection penalty, giving the corrected travel time $\tilde{u}_r^{t_k}$. This slot's $F_1$ term is the demand-weighted "realized-to-baseline" travel-time ratio:

$$\mathrm{term}_k = \frac{\sum_r H_r^{t_k}\, \tilde{u}_r^{t_k}}{\sum_r H_r^{t_k}\, u_r^{t_0}}$$

if the denominator is zero (the extreme case where demand has vanished entirely), the term is set to 1. Both numerator and denominator are weighted by the **current** demand because only by weighting the same set of travelers is this ratio a clean measure of "how many times longer these travelers now spend than before the disaster"; were the denominator instead weighted by the fixed normal demand, the demand drop itself would distort the ratio.

Two points are worth noting here. First, this ratio can fall below 1: both numerator and denominator are weighted by the current (reduced) demand, while the baseline $u_r^{t_0}$ corresponds to "the congestion level at the normal demand scale," so when there are fewer travelers within the recovery window the network is emptier than normal, the realized travel time can be shorter than the pre-disaster baseline, and the term drops below 1; only after both network and demand return to normal does it settle at 1. This does not affect the direction of the optimization signal: the faster the recovery, the more slots sit in the recovered low-value state and the lower $F_1$ becomes. Second, the per-slot UE results cannot be cached and reused by "the set of repaired segments": because the demand shortfall carries inertia (see 5c), the traffic state at a given slot depends not only on which segments are repaired at that moment but also on the history by which that state was reached, so the same "set of repaired segments" corresponds to different demand under different schedules, and every slot must therefore be a fresh UE solve.
(util/evaluate.py:223-232)

### 5f · Aggregating F1 and composing F

Averaging the $T$ per-slot terms gives $F_1$. By default the average is taken over the whole horizon $[1, T]$:

$$F_1 = \frac{1}{T} \sum_{k=1}^{T} \mathrm{term}_k$$

config keeps a switch (off by default) that, when turned on, averages only over the slots that still have damage. The whole-horizon average is the default in order to stay consistent with the $1/T$ normalization used by the pretraining MILP surrogate for its coefficients, so that the two $F_1$ values are directly comparable. Finally the composition

$$F = \mu F_1 + (1-\mu) F_2$$

is formed, with default $\mu = 1.0$, so that $F$ equals $F_1$ (accessibility-only); $F_2$ is still computed and recorded but carries zero weight. The function returns $F$, $F_1$, and $F_2$, and can optionally attach the per-slot OD travel-time matrix (which the pretraining MILP uses to construct fixed surrogate cost coefficients) and a trace table (used by figure 03).
(util/evaluate.py:239-246 aggregation and composition; util/evaluate.py:247-254 the return value)

---

## Step 6 · Sorting the landscape and the per-scenario optimum

All the evaluation rows accumulated by the main loop need to be organized into two products: the complete landscape, and each scenario's hindsight optimum. The approach sorts all rows by (scenario, permutation) and writes them as oracle_landscape.csv, then takes, within each scenario, the row with the minimum $F$ and gathers them into oracle_optima.csv — the latter being each scenario's true hindsight-optimal schedule and its $F^\ast$. Both products are kept because the optimal value serves the core check of "how far the approximate solver is off," while the complete landscape supports finer comparisons: for example, where a given approximate solution ranks among all permutations; whether multiple tied optima exist — if the approximate solution and the hindsight optimum are not the same schedule but have equal objective values, one should compare objective values rather than whether the schedules match; and whether the landscape is peaked or a broad plateau, whose shape dictates the acceptance tolerance to apply to an approximate solution. The product is these two CSVs, handed to Step 7's summary and figures and also left for later comparison against the approximate solver.
(util/oracle.py:232-236)

---

## Step 7 · The summary and figures

Finally a concise run summary is produced, along with three figures that explain "what the landscape looks like, and why the optimal schedule is better." For the summary, summary.txt is assembled and written, containing the instance size, the total number of schedules evaluated, the mean $F^\ast$, $F_1^\ast$, and $F_2^\ast$ of the per-scenario optima, and a note that the oracle optimum is the minimum over all permutations within each scenario, i.e. the true hindsight optimum.
(util/oracle.py:238-249)

For the figures, `make_figures` produces three plots. Figure 01 sorts each scenario's $F$ from best to worst and overlays them, using a filled band to show the across-scenario range and an annotation to mark the position of the oracle optimum $F^\ast$, intuitively answering "how much improvement choosing the right repair order brings"; figure 02 places all schedules of a representative scenario in the $(F_2, F_1)$ plane, colored by $F$, to show the two-objective structure; figure 03 takes the representative scenario's optimal row, reconstructs that schedule from the start times stored in the landscape, and re-evaluates it once to collect per-slot traces, drawing a three-panel figure — the optimal schedule's Gantt chart (colored by severity), the per-slot total demand (a sharp drop then recovery, with the dashed line marking the normal level), and the per-slot $F_1$ term (with a dashed line at 1 marking full recovery). The figure style uniformly follows viz/style.py's publication-grade settings (the shared rcParams, the severity warm-color ramp, panel letters, and saving only a 600-dpi PNG by default). Both the summary and the figures are for human reading only and feed back into no subsequent computation.
(util/oracle.py:251-252 the call; viz/oracle_viz.py:38-154 the three figures; viz/style.py:107 saving; there is also an entry point at util/oracle.py:265-279 that only re-renders the figures without recomputing)

---

# Appendix A — Parameter-default table

All values below are the **current** defaults in config.py (the single source of truth). Except where marked "given," each item is a modeling assumption.

| Parameter | Current default | Role in run_oracle | config.py line |
|---|---|---|---|
| DELTA_T_H | 3.0 h/slot (given) | the physical length of a slot; cancels in $F_2$ | 10 |
| C_MAX | 2 | the number of crews in list scheduling | 11 |
| MU | 1.0 | the weight in $F=\mu F_1+(1-\mu)F_2$; at 1.0 only accessibility is counted ($F_2$ is still computed and recorded, weight 0) | 12 |
| CAP_RETAIN | {1: 0.3, 2: 0.1, 3: 0.02} | damaged capacity = retention factor × original capacity | 20 |
| SPEED_RETAIN | {1: 0.5, 2: 0.3, 3: 0.2} | damaged free_flow_time = original ÷ retention factor | 21 |
| SEVER_SEVERITY | 3 | at this severity the segment is removed entirely (disconnection counted at the penalty value) | 22 |
| F1_ACTIVE_ONLY | False | $F_1$ averaged over the whole $[1,T]$ (aligned with the MILP's $1/T$ normalization) | 25 |
| RHO | 0.7 | recovery inertia: $D^{t_k} = \max(B v^{t_k}, \rho D^{t_{k-1}})$ | 37 |
| KAPPA | 1.0 | the damage-to-shortfall scale: $B_{r,e}=\kappa\,(h_r^{t_0}/3)$, non-zero only for segments on the free-flow shortest path | 38 |
| UPEN_FACTOR | 10.0 | disconnection penalty = 10 × the largest finite pre-disaster OD travel time | 41 |
| M_SCENARIOS | 10 | the number of duration scenarios | 45 |
| SEED | 42 | the RNG seed for scenario sampling (reproducible) | 46 |
| N_DISRUPTED_ORACLE | 4 | the damaged-set size ($4!=24$ schedules) | 47 |
| UE_RGAP | 1e-6 | the relative-gap target of the per-slot UE | 54 |
| UE_MAX_ITER | 100 | the per-slot UE iteration cap | 55 |
| DURATION_SUPPORT | by (road_class, severity) | the base-duration support sets | 92-96 |
| ETA | [0.8, 1.0, 1.2] (given) | the crew-efficiency multiplier sample set | 97 |

**config items not involved in the oracle** (used by the pretraining-MILP pipeline, see note 02): MILP_MAX_ITER=20, MILP_CYCLE_TOL=3, MILP_DAMPING_MODE="decay", MILP_DAMPING=0.5, MILP_PROX_SCALE=0.1, MILP_WARM_START=True (config.py:61-88).
