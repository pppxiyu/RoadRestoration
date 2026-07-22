# 03 — Baseline solvers: static greedy, GA, PSO
This note explains how three families of baseline solver produce a schedule for the same repair-scheduling problem, and how their results are placed on one comparison table with the pretraining MILP of note 02. It covers the three static greedy rankers in `util/greedy.py` (entry point run_greedy, command line `python -m util.greedy`), the two metaheuristics in `util/metaheuristic.py`, namely a genetic algorithm (GA) and particle swarm optimization (PSO) (entry point run_metaheuristic, command line `python -m util.metaheuristic`), and the comparison harness in `util/compare.py` that aligns every method into one table (entry points run_compare and run_baseline_figures). (util/greedy.py:71, util/metaheuristic.py:252, util/compare.py:23 and util/compare.py:73)

---

## Overview

### 1 · Problem

The value of the pretraining MILP has to be compared with a set of baselines spanning the compute spectrum. The pretraining MILP here is the alternating-optimization solver of note 02, named after its module pretrain_milp; this note does not unfold its mechanics and treats it purely as the object under test.

### 2 · Approach (Do NOT read if familiar with the problem setting in Note 01-02)

Every baseline shares one representation of a solution, a repair-priority order (a permutation) over the damaged segments. Repairs are executed by $`C_{\max} = 2`$ crews in parallel, and the order is translated into concrete start slots by work-conserving list scheduling, which hands each segment in priority order to the crew that frees up earliest, so that any crew picks up the next segment the moment it becomes idle and never sits idle while work remains (the rule is in note 01, Step 4).

A slot is the unit period into which the recovery period is discretized, the recovery period spans $`T`$ slots, and this $`T`$ is what the note later calls the horizon. UE (user equilibrium) denotes equilibrium traffic assignment, which given the current network and travel demand solves for the equilibrium flow on every edge and the travel time of every OD pair, an OD pair being an origin-destination pair between which trips are made, and equilibrium meaning that no driver can reduce their own travel time by unilaterally switching route. Throughout this note a UE solve is treated purely as one expensive black-box computation, whose input-output contract is in note 01 §1c. The true evaluator evaluate_schedule then computes the exact objective $`F`$, solving one UE per slot of the recovery period (note 01, Step 5); the qualifier "true" distinguishes this exact evaluation from the cheap internal surrogate objective that the MILP of note 02 optimizes, and every number quoted in this note is a true evaluation. $`F`$ itself is a normalized cost to be minimized. Roughly, it averages over the recovery period the demand-weighted ratio of current travel time to its pre-disaster baseline, so $`F = 1`$ means the recovery period sits on average at the pre-disaster level and lower is better; the exact definition, and the relation between $`F`$ and its component $`F_1`$, are set out in Step 1 §4.
(util/greedy.py:77-82 and util/metaheuristic.py:262-267 for the shared problem definition; util/evaluate.py:95-108 for list scheduling; config.py:11 for the crew count; util/scenarios.py:24-52 for scenario sampling)

One more premise behind the evaluator must be stated here, because two later passages rest on it. In the model, damage suppresses the travel demand of the affected OD pairs, and repair lets the suppressed demand return gradually (the shortfall dynamics are in note 01 §5c). The scoring rule of the demand variant in Step 1 and the phenomenon that $`F`$ can fall below 1 both build on this premise.

A scenario is one random draw of the repair durations of the damaged segments. Which segments are damaged is fixed; the uncertainty is only how many slots each takes to repair. A fixed seed draws $`M`$ such segment-to-duration combinations once, and every method below solves the same batch of $`M`$, one scenario at a time (note 01, Step 2).

### 3 · Interpretation · why reduce every baseline to a permutation?

It compresses every difference between the methods into a single question, namely how that order is chosen. What the comparison between baselines isolates is therefore ranking intelligence itself, rather than a difference of representation or of evaluation convention. Under this representation, static greedy produces an order in one shot from a static importance rule, while GA and PSO run budgeted population search in permutation space.

### 4 · Interpretation · how can the MILP be comparable if it does not share this representation?

The MILP does not share the permutation representation. It optimizes directly over start slots, and its feasible set permits crews to idle deliberately, making it a strict superset of the schedule set generated from work-conserving permutations by list scheduling (note 02, Step 4 and its Caveats C2). Its comparability with the baselines therefore does not come from the representation. It comes solely from the shared premise stated in the approach above, that all methods use the same damaged instance, the same batch of scenarios (same seed), the same horizon, and the same evaluator, which is what makes $`F`$ directly comparable scenario by scenario.

---

## Step 1 · Static greedy

### 1 · Problem

Provide the low end of the compute spectrum, a reference that performs no search at all and decides the repair order from a single static rule. Any more expensive method that cannot clearly beat it has no reason to exist.

### 2 · Approach

Each of the three variants gives every damaged segment an importance score and sorts by score from high to low, and that order is the solution. The relation between a segment and an edge should be stated first. A segment is one undirected road connecting a pair of nodes, occupying one row of the network table and identified uniquely by its edge id (regardless of direction). When several segments share the same score, ties are broken by ascending edge id.

The demand variant scores a segment by the demand its full repair releases, $`v_e^{\ast} \sum_r B_{r,e}`$. Here $`B_{r,e}`$ comes from the demand-shortfall sensitivity matrix of note 01 §1e and gives the demand that OD pair $`r`$ loses per unit of severity of segment $`e`$, while $`v_e^{\ast}`$ is that segment's severity, the fixed damage level assigned to it when the damaged instance was built (note 01 §1a), which does not vary across scenarios and also sets the scale of the demand released when the segment is fully repaired. The product $`B_{r,e}\, v_e^{\ast}`$ is therefore the OD-$`r`$ demand released back onto the network when $`e`$ is fully repaired, and the demand variant's score is that product summed over $`r`$.

The ratio variant divides that score by the segment's repair duration in the scenario at hand, measuring how much demand is bought back per construction slot, and it therefore varies from scenario to scenario.

The flow variant scores by two-way baseline UE flow on the intact network, that is, the sum of the equilibrium flows on the segment's two directed links, which measures the topological criticality of the road (util/oracle.py:101-105 for the two-way summation). It is the only one of the three rules that uses UE at all, but that single UE is solved on the intact network and is independent of any candidate order, so one solve feeds all scenarios and, once amortized, does not break the near-zero-compute standing of the family. The same intact-network two-way-flow measure is reused by the selection of the damaged instance (note 01 §1a) and by the MILP's warm start (note 02, Step 2).

Each variant spends exactly one true $`F`$ evaluation per scenario, because the scoring rules depend on no candidate schedule and the sorting stage needs no new UE (util/greedy.py:45-65 for the three rules, util/greedy.py:91-94 for the sort and the single evaluation).

### 3 · Interpretation · why not a stepwise true-value greedy?

A natural alternative design is to use no static score and instead, at every position, to run a true F lookahead over each candidate segment not yet scheduled, fix at that position the segment whose repair next would lower the current $`F`$ the most, and then move on to the next position and repeat the same selection. This is exactly the textbook form of greedy, the so-called stepwise true-value greedy.

The first reason it is not adopted is that its compute cost is too high. Since $`F`$ is defined only for a complete schedule, scoring any candidate requires first filling it into a complete order and then running one true $`F`$ evaluation, so every position must evaluate each of the candidates remaining at that point once. Adding up the candidate counts across the positions, one scenario costs $`N(N+1)/2`$ full true $`F`$ evaluations; intuitively, the first position tries $`N`$ candidates, the second has $`N-1`$ left, and the count shrinks position by position until only $`1`$ remains at the last, and summing this shrinking sequence gives exactly this triangular number. Moreover, each of those true $`F`$ evaluations itself solves one UE per slot of the recovery period, that is $`T`$ UE solves, so the stepwise true-value greedy is a full order of magnitude more expensive than the single evaluation of a static rule.

The second reason it is not adopted is that at that compute level it is already dominated by GA and PSO. The budgeted population search of GA and PSO spends a comparable number of evaluations, yet it puts those evaluations into a global search over the entire permutation space, whereas the stepwise true-value greedy is only a myopic greedy that looks at one position at a time and never goes back to revise a segment once it is fixed at a position. Therefore at the same compute tier, if that many evaluations are to be paid anyway, it is better to run GA or PSO for a global search directly than to use this lookahead greedy.

The product of this step is one {variant}_optima.csv per variant, holding the per-scenario $`F`$, elapsed time, order and start slots, to be discovered automatically by the comparison harness of Step 7. All three variants' orders subsequently serve as seeds for GA and PSO (Step 5), and flow's order additionally serves as the MILP's warm start (note 02, Step 2).

---

## Step 2 · The evaluation layer shared by GA and PSO: how many distinct schedules a budget-limited search truly examines

### 1 · Problem

The entire skill of GA and PSO lies in finding good orders with a limited number of probes, and one probe is one true $`F`$ evaluation, which solves one UE per slot of the recovery period, $`T`$ in total, and is the only expensive computation in the whole pipeline (the compute accounting of Caveats C1 is denominated in it). Before the two search loops run, a shared evaluation layer must therefore answer two questions first. First, how many distinct schedules each method is allowed to truly examine; this number caps the quality of the searched solution and also defines the compute tier the method sits in, and whether the cross-method comparison is fair rests entirely on it. Second, how to guarantee that the solution the search finally hands over has really been truly evaluated, rather than being a candidate that was never verified.

### 2 · Approach

Both searches proceed in rounds called generations, and each generation proposes a batch of candidate orders that is handed to the evaluation layer to compute their true $`F`$. The evaluation layer is built around a cache keyed by the order itself, and the cache stores $`(F, F_1, F_2)`$, where $`F_1`$ and $`F_2`$ are the two components of $`F`$ before the weighted blend (note 01, Step 5), both retained beside $`F`$ in the cache and on disk. An order is evaluated truly at most once, and candidates that reappear (such as the retained best individuals of Step 3) simply read back their existing scores, so what the budget below counts is exactly the number of distinct schedules truly examined (called unique evaluations below), not a count inflated by repeated evaluations (util/metaheuristic.py:86-115 for the cache and the best-ever readout).

The budget is a cap on the number of unique evaluations, imposed on each combination of scenario and method.
- The search's main loop checks the budget at the start of every generation and exits once it is exhausted; within one generation, when candidates are submitted in a batch, the new decisions beyond the remaining budget are recorded as $`+\infty`$ and never evaluated truly (util/metaheuristic.py:170 and util/metaheuristic.py:230 for the per-generation budget check; util/metaheuristic.py:101-111 for the over-quota marking).
- Beyond the budget, both loops also carry a stall guard that exits early after 40 consecutive generations produce no new unique evaluation, so the number of evaluations actually spent is decided by whichever of budget exhaustion and search stall arrives first; the threshold 40 is a hardcoded literal in the same two loop conditions rather than an exposed parameter.
- An evaluation that yields no score because its UE fails is booked with a large but finite sentinel as that order's score and spends one unit of budget as usual (util/metaheuristic.py:79-80). As an empirical fact, in the $`n=13`$ run, which is the only run in the repository with GA and PSO results on disk, not a single evaluation failed and the sentinel value does not appear in the result CSVs, so it is a layer of insurance that has never been triggered rather than an observed failure mode.

The search queries the evaluation layer for $`F`$ throughout, and after it ends the cache yields the best order evaluated truly over the whole run, which guarantees that the final output is always a genuinely evaluated solution. The product of this step is the evaluation layer FitnessCache, of which the two search loops of Step 3 and Step 4 each hold and use one instance, one per combination of scenario and method (see Step 6) (util/metaheuristic.py:64-80).


---

## Step 3 · GA: searching directly over permutations with legality-preserving operators

If a GA's recombination operator splices two parent permutations arbitrarily, it produces illegal individuals in which one segment appears twice and another disappears. A set of operators closed within permutation space is therefore required.

An individual is a permutation. Selection is by tournament, drawing a few individuals at random and taking the one with the lowest true $`F`$ as a parent. Recombination is order crossover (OX), which copies a random contiguous stretch from the first parent and then fills the remaining positions with the segments not yet used, in the relative order they appear in the second parent. The result is necessarily a legal permutation that inherits a run of absolute positions from one parent and the relative order of the rest from the other (util/metaheuristic.py:135-149). Mutation is one or two random swaps (util/metaheuristic.py:152-157). Each generation keeps a few elites by $`F`$ unchanged into the next, and the population is deduplicated before the elites are chosen so that the elite slots land on mutually distinct orders rather than several copies of one best schedule. The loop continues until the evaluation budget is exhausted or the stall guard fires (util/metaheuristic.py:160-185).

---

## Step 4 · PSO: a random-key encoding that grafts continuous dynamics onto permutations

### 1 · Problem

PSO maintains a swarm of candidate points, called particles, each carrying a position and a velocity, and it searches by repeatedly nudging every particle toward the best positions found so far. Its velocity and position updates are therefore defined on continuous vectors, whereas a permutation has no meaning for "add a little velocity", so the standard scheme cannot be applied directly.

### 2 · Approach

The random-key encoding is used. Each particle is a real vector in $`[0,1]^n`$, decoded by argsort over its components, with the smaller component repaired earlier. Velocity and position then iterate by the standard gbest update (gbest names the global-best variant of PSO, in which every particle is pulled not only toward its own best position but also toward the single best position the whole swarm has found)

```math
v \leftarrow w\,v + c_1 r_1 (p_{\text{best}} - x) + c_2 r_2 (g_{\text{best}} - x), \qquad x \leftarrow x + v
```

after which out-of-range components are clipped back into $`[0,1]`$. Term by term: $`w = 0.7`$ is the inertia weight and sets how much of the previous step's velocity is retained; $`c_1 = c_2 = 1.5`$ are the attraction coefficients pulling the particle toward its own historical best position $`p_{\text{best}}`$ and toward the swarm's historical best position $`g_{\text{best}}`$; and $`r_1`$ and $`r_2`$ are componentwise uniform random numbers, which make the two pulls stochastic in strength rather than fixed. Both best positions are maintained by the true $`F`$ of the decoded order (util/metaheuristic.py:206-241).

The initial swarm is likewise seeded with the heuristic orders (Step 5), injected by inverse encoding: the segment sitting at rank $`r`$ is given the key $`(r+0.5)/n`$, so that argsort reproduces exactly that order (util/metaheuristic.py:191-203).

### 3 · Interpretation · why a random-key encoding rather than a purpose-built discrete velocity?

The random-key encoding leaves PSO's standard continuous dynamics untouched while every particle decodes to a legal permutation at any moment, at the implementation cost of one extra argsort. The price is that the mapping from keys to orders is many-to-one, since many distinct real vectors decode to the same permutation. The evaluation layer of Step 2 absorbs exactly this redundancy, because a repeatedly decoded order hits the cache directly and spends no budget. The product is the same as in Step 3.

---

## Step 5 · Greedy seeding plus elite memory: starting the metaheuristics from the strongest heuristic

At $`n=13`$ the permutation space holds $`13! \approx 6.2 \times 10^9`$ orders while the budget is only a few dozen evaluations. A purely randomly initialized population is under such a budget almost certain to be worse than the heuristics across the board, and a comparison of that kind measures nothing.

Both populations are seeded with the three static greedy orders (flow, demand and ratio) as initial members, with the remaining members filled randomly. The scoring logic reproduces util/greedy.py exactly, so that the seeds really are those baselines' solutions (util/metaheuristic.py:121-129 for the seed orders; util/metaheuristic.py:164 and util/metaheuristic.py:215-216 for the injection; util/metaheuristic.py:287 for the per-scenario generation). Setting the switch seed_greedy to False falls back to purely random initialization.

---

## Step 6 · Per-scenario running: independent solving, independent budget accounting, reproducibility

Because the $`M`$ scenarios have different repair durations, search results do not transfer between them, so each scenario must be solved as an independent problem. At the same time, the way the runs are organized must not break two experimental premises, first that each method's budget of unique evaluations is counted independently (Step 2), and second that the whole experiment is reproducible.

The outer loop runs scenario by scenario, solving each scenario as one independent problem. Precisely because of that independence, the evaluation cache must be rebuilt when crossing scenarios, because the repair durations have changed and the $`F`$ values cached for the previous scenario are simply invalid for the new one. (util/metaheuristic.py:284-314 for the main loop and the write; util/metaheuristic.py:272-282 for resume; util/metaheuristic.py:55 for the common directory). Reproducibility comes from deterministic RNGs, as each combination of scenario and method derives an RNG whose seed is decided jointly by the global seed, the scenario number and the method, so rerunning the same code yields the same batch of results. (util/metaheuristic.py:291-292 for both the RNG and the fresh cache).

---

# Caveats

## C1 · The time_s of different methods is measured on different bases, so wall-clock cannot be compared across methods

The time_s of greedy and of the MILP is serial wall-clock, whereas GA and PSO spread their evaluations over (by default 16) worker processes, making their time_s parallel wall-clock. Drawing both on one time axis makes the metaheuristics look faster than a fair basis would show, by roughly the degree of parallelism. The fair compute measure, independent of hardware and of parallelism, is the serial-equivalent UE count: greedy spends $`T`$ per scenario, the MILP spends $`(\text{iterations}+1) \cdot T`$ (already written out as the ue_solves column), and GA and PSO spend $`\text{n\_evals} \times T`$.

## C2 · Under seeding, beating greedy is a guarantee by construction, and only the margin carries information

The seed injection of Step 5, combined with the best-ever readout from the cache in Step 2, makes the GA and PSO results no worse than the strongest greedy by construction. A reading such as "the metaheuristics beat flow across the board" therefore carries no information in itself; what carries information is the size of the margin and the compute paid for it. By the same token the results should be read as how much a budgeted search can improve on top of the strongest heuristic, not as how strong a blind metaheuristic is, since answering the latter would require a control experiment with seed_greedy set to False.
(util/metaheuristic.py:26-29 for the module's own statement, util/metaheuristic.py:287)

## C3 · The n=13 results used a horizon upper bound and a non-default budget

### The horizon upper bound

The exact horizon requires enumerating all $`N!`$ permutations and taking the largest completion time (note 01, Step 4), which is infeasible at $`n=13`$. The $`n=13`$ results therefore use the classical Graham bound for list scheduling instead, taking per scenario

```math
T = 1 + \left\lceil \frac{\sum_e d_e + (m-1)\max_e d_e}{m} \right\rceil
```

and then the maximum over scenarios, with $`m`$ being the $`C_{\max}`$ of the Overview (the bound keeps the notation of the scheduling literature). The completion time of any work-conserving list schedule is provably no greater than this bound. Term by term: $`\sum_e d_e / m`$ is the ideal completion time with the work divided perfectly among the $`m`$ crews; $`(m-1)\max_e d_e / m`$ is the slack that covers the worst tail, in which the longest segment starts last; the ceiling makes $`T`$ a whole number of slots; and the $`+1`$ accounts for this project starting the first repair at slot 1 rather than at time zero (util/evaluate.py:101-102). The bound was first implemented as a substitution by a runner script outside the repository, but it is now written into the repository, where compute_horizon switches to it automatically once the segment count exceeds 8 and still enumerates exactly at 8 or fewer, and at $`n=13`$ the new implementation gives $`T=42`$, matching the value of the original runner (util/oracle.py:144-168 for the computation, util/oracle.py:48-52 for the threshold).

### Why nothing is truncated and what the cost is

The danger to rule out is a $`T`$ below some method's completion time, with the tail of the recovery cut out of the evaluation window. This cannot happen, and there is one reason on each side. The baselines' schedules are all generated by work-conserving list scheduling, so their completion times are covered by the bound; the MILP's feasible set already forbids any start whose completion falls later than $`T`$, and enlarging $`T`$ only relaxes that constraint (note 02, Step 4). The cost is that the bound can only be too large, which lengthens the averaging window. The lengthening is identical for every method at this scale, so comparisons between methods are unaffected; only the absolute value of $`F`$ is diluted by the extra slots and cannot be compared directly with numbers from scales run under an exact horizon. At $`n \le 6`$ the bound was once checked against exact enumeration with no discrepancy, but the no-truncation guarantee comes from the two-sided argument above rather than from that spot check.

### Run parameters

The $`n=13`$ run used budget 60 with 16 workers, whereas the module's default budget is 120. Reproducing these numbers requires the same budget and setting the damaged segment count to 13; the horizon bound is now applied automatically by compute_horizon, so the substitution outside the repository is no longer needed (util/metaheuristic.py:252-253).

---

# Appendix A · Parameter defaults

None of the GA and PSO parameters live in config.py, and they come in two kinds. The rows anchored at util/metaheuristic.py:248 and 249 are module-level constants, whereas the three rows anchored at 253 (budget, workers, seed_greedy) are keyword-argument defaults of the entry point run_metaheuristic. One further constant belongs to neither kind and is therefore not a table row: the stall guard's threshold of 40 consecutive stalled generations (Step 2) is a hardcoded literal inside both loop conditions (util/metaheuristic.py:170 and util/metaheuristic.py:230).

| Parameter | Current default | Role | Location |
|---|---|---|---|
| pop_size | 16 | GA population size | util/metaheuristic.py:248 |
| elite | 4 | deduplicated elites GA keeps each generation | util/metaheuristic.py:248 |
| tour_k | 3 | individuals drawn per tournament selection | util/metaheuristic.py:248 |
| p_cross | 0.9 | OX crossover probability | util/metaheuristic.py:248 |
| p_mut | 0.5 | mutation probability | util/metaheuristic.py:248 |
| max_swaps | 2 | maximum random swaps per mutation | util/metaheuristic.py:248 |
| swarm | 16 | PSO particle count | util/metaheuristic.py:249 |
| w | 0.7 | PSO inertia weight | util/metaheuristic.py:249 |
| c1, c2 | 1.5, 1.5 | PSO personal and global attraction coefficients | util/metaheuristic.py:249 |
| budget | 120 (the $`n=13`$ experiment used 60) | unique-evaluation budget per scenario per method | util/metaheuristic.py:253 |
| workers | 16 | size of the evaluation process pool | util/metaheuristic.py:253 |
| seed_greedy | True | whether to inject the three greedy seeds (Step 5) | util/metaheuristic.py:253 |

The shared problem parameters (instance selection, scenario sampling, UE convergence settings and so on) are the same as in note 01's Appendix A and are not repeated here. The ones bearing directly on this note are M_SCENARIOS=10 and SEED=42 (config.py:45-46), which fix the scenario batch, and C_MAX=2 (config.py:11), which fixes the crew count of list scheduling.
