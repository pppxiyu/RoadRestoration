# 03 — The solving logic of the baseline solvers: static greedy, GA, PSO, and the unified comparison

This note explains how three families of baseline solver produce a schedule for the same repair-scheduling problem, and how their results are placed on one comparison table with the pretraining MILP of note 02. It covers the three static greedy rankers in `util/greedy.py` (entry point run_greedy, command line `python -m util.greedy`), the two metaheuristics in `util/metaheuristic.py`, namely a genetic algorithm (GA) and particle swarm optimization (PSO) (entry point run_metaheuristic, command line `python -m util.metaheuristic`), and the comparison harness in `util/compare.py` that aligns every method into one table (entry points run_compare and run_baseline_figures). The note follows the convention of notes 01 and 02. It is organized in solving order, each step stating first the problem it must solve, then what the code does, then any interpretation that goes beyond the literal computation, with path:line anchors attached inline so every claim can be checked against the current source. Third-party libraries (AequilibraE, scipy, multiprocessing) are summarized by their role only. Pure engineering machinery such as per-scenario checkpointing and resumable runs is mentioned only where it bears on the solving logic. Parameter defaults are collected in Appendix A.
(util/greedy.py:93, util/metaheuristic.py:252, util/compare.py:23 and util/compare.py:73)

---

## Overview · Why every baseline reduces to a repair-priority order over the damaged segments

### 1 · Problem

The value of the pretraining MILP has to be tested from two directions at once. It must be shown how much better it is than a far cheaper static heuristic, and how much worse it is than a general-purpose search given equal or greater compute. This calls for a set of baselines spanning the compute spectrum, and the scores of every method must be strictly comparable. The pretraining MILP here is the alternating-optimization solver of note 02, named after its module pretrain_milp; this note does not unfold its mechanics and treats it purely as the object under test.

### 2 · Approach

Every baseline shares one representation of a solution, a repair-priority order (a permutation) over the damaged segments. Repairs are executed by $`C_{\max} = 2`$ crews in parallel, and the order is translated into concrete start slots by work-conserving list scheduling, which hands each segment in priority order to the crew that frees up earliest, so that any crew picks up the next segment the moment it becomes idle and never sits idle while work remains (the rule is in note 01, Step 4). The true evaluator evaluate_schedule then computes the exact objective $`F`$, solving one UE per slot of the recovery period (note 01, Step 5). A slot is the unit period into which the recovery period is discretized, the recovery period spans $`T`$ slots, and this $`T`$ is what the note later calls the horizon. UE (user equilibrium) denotes equilibrium traffic assignment, which given the current network and travel demand solves for the equilibrium flow on every edge and the travel time of every OD pair, an OD pair being an origin-destination pair between which trips are made. Throughout this note a UE solve is treated purely as one expensive black-box computation, whose input-output contract is in note 01 §1c.
(util/greedy.py:99-104 and util/metaheuristic.py:262-267 for the shared problem definition; util/evaluate.py:95-108 for list scheduling; config.py:11 for the crew count; util/scenarios.py:24-52 for scenario sampling)

One more premise behind the evaluator must be stated here, because two later passages rest on it. In the model, damage suppresses the travel demand of the affected OD pairs, and repair lets the suppressed demand return gradually (the shortfall dynamics are in note 01 §5c). The scoring rule of the demand variant in Step 1 and the phenomenon that $`F`$ can fall below 1 both build on this premise.

A scenario is one random draw of the repair durations of the damaged segments. Which segments are damaged is fixed; the uncertainty is only how many slots each takes to repair. A fixed seed draws $`M`$ such segment-to-duration combinations once, and every method below solves the same batch of $`M`$, one scenario at a time (note 01, Step 2).

### 3 · Interpretation · why reduce every baseline to a permutation?

The first reason is to make the solution space coincide with the oracle's. The oracle is the brute-force enumerator of note 01, which evaluates all $`N!`$ permutations truly and yields each scenario's hindsight optimum, serving as the accuracy yardstick at small scale (Step 7's gap column is measured against it). The oracle's solution space is exactly these $`N!`$ permutations, so a baseline that adopts the permutation representation and inherits the same work-conserving reduction (note 01, Step 3) lands in a space that coincides with the oracle's, which is what makes the gap readings meaningful.

The second reason is that it compresses every difference between the methods into a single question, namely how that order is chosen. What the comparison between baselines isolates is therefore ranking intelligence itself, rather than a difference of representation or of evaluation convention. Under this representation, static greedy produces an order in one shot from a static importance rule, while GA and PSO run budgeted population search in permutation space.

### 4 · Interpretation · how can the MILP be comparable if it does not share this representation?

The MILP does not share the permutation representation. It optimizes directly over start slots, and its feasible set permits crews to idle deliberately, making it a strict superset of the schedule set generated from work-conserving permutations by list scheduling (note 02, Step 4 and its Caveats C2). Its comparability with the baselines therefore does not come from the representation. It comes solely from the shared premise stated in the approach above, that all methods use the same damaged instance, the same batch of scenarios (same seed), the same horizon, and the same evaluator, which is what makes $`F`$ directly comparable scenario by scenario.

### 5 · Interpretation · which experiment scales do the numbers in this note come from?

The figures quoted in this note come from two scales, one with 4 damaged segments, identical to notes 01 and 02, and one larger instance with 13 damaged segments whose construction is described in Caveats C4. Other scales such as $`n=5`$ and $`n=6`$ also have intermediate results on disk under outputs, but this note quotes none of their numbers. Capital $`N`$ and lower-case $`n`$ denote the same quantity, the number of damaged segments; the note writes $`n=13`$ following the n{N} naming convention of the output directories.

---

## Step 1 · Static greedy: three static importance rules, one sort, one evaluation per scenario

### 1 · Problem

Provide the low end of the compute spectrum, a reference that performs no search at all and decides the repair order from a single static rule. Any more expensive method that cannot clearly beat it has no reason to exist.

### 2 · Approach

Each of the three variants gives every damaged segment an importance score and sorts by score from high to low, and that order is the solution. The relation between a segment and an edge should be stated first. A segment is one undirected road connecting a pair of nodes, occupying one row of the network table and identified uniquely by its edge id. Ties in score are broken by ascending edge id, which keeps the sort deterministic and reproducible, and the edge id can serve as that tiebreaker precisely because it is unique. Because traffic flow and travel time both have direction, a segment is expanded into two opposite directed links when UE is solved (util/ue.py:61-80 for the two-way registration, util/ue.py:130-139 for the AB and BA outputs).

The demand variant scores a segment by the demand its full repair releases, $`v_e^{\ast} \sum_r B_{r,e}`$. Here $`B_{r,e}`$ comes from the demand-shortfall sensitivity matrix of note 01 §1e and gives the demand that OD pair $`r`$ loses per unit of severity of segment $`e`$, while $`v_e^{\ast}`$ is that segment's severity and doubles as the scale of the demand released when it is fully repaired, so the product $`B_{r,e}\, v_e^{\ast}`$ is the OD-$`r`$ demand released back onto the network when $`e`$ is fully repaired. That product is exactly the term appearing inside $`c_e^k`$, the coefficient of the decision variable "segment $`e`$ starts at slot $`k`$" in the surrogate objective of note 02, Step 3, and the demand variant's score is that product summed over $`r`$.

This score is a first-order approximation of the segment's contribution to accessibility, accessibility meaning the degree to which the network carries travel demand at close to pre-disaster travel times, measured by $`F`$'s component $`F_1`$ (the relation between $`F`$ and $`F_1`$ is set out in §4 below). It is first-order because the true contribution, that is the full expression of $`c_e^k`$, additionally multiplies the released demand by weights that vary with congestion and with recovery progress; this score drops those weights entirely and keeps only the linear released-demand part, and linear is exactly what first-order means.

The ratio variant divides that score by the segment's repair duration in the scenario at hand, measuring how much demand is bought back per construction slot, and it therefore varies from scenario to scenario. The flow variant scores by two-way baseline UE flow on the intact network, that is, the sum of the equilibrium flows on the segment's two directed links, which measures the topological criticality of the road (util/oracle.py:101-105 for the two-way summation). It is the only one of the three rules that uses UE at all, but that single UE is solved on the intact network and is independent of any candidate order, so one solve feeds all scenarios and, once amortized, does not break the near-zero-compute standing of the family. The same intact-network two-way-flow measure is reused by the selection of the damaged instance (note 01 §1a) and by the MILP's warm start (note 02, Step 2).

Each variant spends exactly one true $`F`$ evaluation per scenario, because the scoring rules depend on no candidate schedule and the sorting stage needs no new UE (util/greedy.py:49-69 for the three rules, util/greedy.py:116-119 for the sort and the single evaluation). The module also carries a stepwise variant that performs a faithful step-by-step greedy, fixing the order position by position from the front and at each step evaluating every candidate truly and keeping the one with the lowest current $`F`$. Since $`F`$ is defined only for a complete schedule, evaluating a candidate places it in the next position after the current prefix and then fills the remaining undecided segments in ascending edge-id order into a complete order, so one scenario costs $`N(N+1)/2`$ full evaluations. Being an order of magnitude more expensive, it is not run by default (util/greedy.py:72-87).

### 3 · Interpretation · why three rules rather than one?

They embody three mutually independent hypotheses about importance, namely raw payoff (demand), payoff per unit of effort (ratio), and topological criticality (flow). Which one is better is an empirical question that only data can answer. In the $`n=13`$ experiment flow is clearly the strongest (mean about 0.97), whereas demand and ratio produce $`F`$ values above 1 (mean about 1.06).

### 4 · Interpretation · why is $`F > 1`$ a meaningful failure signal rather than the norm?

The relation between $`F`$ and $`F_1`$ must be stated in full first. By the definition in note 01, Step 5, $`F_1`$ is the demand-weighted ratio of current travel time to the pre-disaster baseline, averaged per slot over the whole recovery period, and this is the component that measures accessibility. Nominally $`F`$ is a weighted sum of $`F_1`$ and $`F_2`$, where $`F_2`$ measures schedule length relative to total workload, but under the default weights $`F_2`$ carries zero weight and $`F`$ equals $`F_1`$ exactly, while $`F_2`$ is still computed and written out with the results (config.py:12-14). Hence $`F = 1`$ corresponds to a recovery period that sits, on average, at the pre-disaster level.

At first sight the network is damaged during recovery, so $`F`$ above 1 would seem inevitable. But precisely because the demand model stated in the Overview suppresses travel demand during recovery, the roads carry fewer vehicles and travel times can fall below the pre-disaster baseline, so $`F < 1`$ is entirely reachable (the mechanism is in note 01 §5e), and flow's mean does fall below 1. That is exactly why $`F > 1`$ is an informative failure signal: it means that under these two rules the recovery period is on average worse than the pre-disaster level even while enjoying the favourable condition of an emptier network.

### 5 · Interpretation · what does this contrast say about static rules in general?

It says that the success of a static rule rests almost entirely on the information source it reads. Flow's strength does not come from congestion feedback during recovery, since all it reads is the pre-disaster equilibrium flow, and the failure of demand and ratio is not enough to condemn static rules as a class. No static rule may therefore be assumed adequate in advance, and which information source is reliable can only be answered empirically per instance, which is the motivation for bringing in more expensive search methods as a comparison.

The product of this step is one {variant}_optima.csv per variant, holding the per-scenario $`F`$, elapsed time, order and start slots, to be discovered automatically by the comparison harness of Step 7. All three variants' orders subsequently serve as seeds for GA and PSO (Step 5), and flow's order additionally serves as the MILP's warm start (note 02, Step 2).

---

## Step 2 · Parallel, budgeted, deduplicated evaluation of the true objective (shared by GA and PSO)

### 1 · Problem

The recovery period spans $`T`$ slots, and one true $`F`$ evaluation solves one UE per slot, $`T`$ in total, which puts a single evaluation at the scale of minutes (the compute accounting of Caveats C1 is denominated in this same $`T`$). Population search easily needs hundreds of evaluations, which is infeasible serially. At the same time, without a budget the compute spent by different methods is not comparable.

### 2 · Approach

Three things are done.

- Parallelism. A multiprocessing process pool spreads evaluations over several workers, each of which builds the static context once at startup (including one baseline UE) and thereafter evaluates candidate orders in batches. This is legitimate because solve_ue builds its graph and solves entirely in memory, writes nothing to disk and holds no shared state, so the processes never contend (util/metaheuristic.py:64-80).
- Memoization. The cache is keyed by the order itself and stores $`(F, F_1, F_2)`$, where $`F_1`$ and $`F_2`$ are the two components of $`F`$ before the weighted blend, measuring accessibility and schedule efficiency respectively (note 01, Step 5), both retained beside $`F`$ in the cache and on disk. An order is therefore evaluated truly at most once, so elite survivors and repeated offspring cost nothing.
- Budget. Each scenario and method gets a cap on the number of unique evaluations, with a two-layer meaning. The search's main loop checks the budget at the start of every generation and exits once it is exhausted; and within one generation, when candidates are submitted in a batch, any new orders beyond the remaining budget are recorded as $`+\infty`$ and never evaluated truly (util/metaheuristic.py:86-115 for the budget, cache and best-ever readout; util/metaheuristic.py:170 and util/metaheuristic.py:230 for the per-generation budget check and stall guard).

Each worker evaluation is additionally wrapped in exception isolation, returning a large but finite sentinel on failure, so that one failed UE cannot abort a run of several hours. Both search loops carry a stall guard that exits after several consecutive generations produce no new unique evaluation, which prevents a converged population from spinning on the cache forever.

### 3 · Interpretation · why does $`+\infty`$ not pollute the results?

There are two reasons. First, the final output is selected only among orders that were truly evaluated and are in the cache, and $`+\infty`$ is never written into the cache. Second, a candidate being recorded as $`+\infty`$ means that this very batch exhausted the budget, so the main loop exits at the budget check that opens the next generation, and the overflowing candidates therefore never take part in any later iteration. The only place inside that same generation which still reads $`+\infty`$ is PSO's update of its personal and global best memories (those two memories are in Step 4), and as the worst possible fitness $`+\infty`$ cannot overwrite either (util/metaheuristic.py:236-240).

### 4 · Interpretation · why is the failure sentinel finite while overflow uses $`+\infty`$?

The two carry different meanings. A failed evaluation really did spend one unit of budget, so its sentinel is written into the cache as that order's score, which stops the same order from being submitted again and keeps the budget accounting consistent. $`+\infty`$ instead marks an over-quota placeholder that spent no budget, appears only in the temporary result handed back to the search, and never enters the cache. Finiteness against infinity is exactly what separates these two kinds of bad value, and the sentinel is in any case large enough that, like $`+\infty`$, it can never be selected as the best.

### 5 · Interpretation · why budget by unique evaluations rather than by generations or wall-clock?

Evaluation is the only expensive carrier of compute, and counting it is what licenses the claim that the methods are compared at one compute level. Wall-clock is distorted by the degree of parallelism (see Caveats C1), and generation counts decouple from real compute because cache hit rates differ. The product of this step is a FitnessCache object, which the search queries for $`F`$ throughout and which afterwards yields the best order evaluated truly over the whole run, guaranteeing that the final output is always a genuinely evaluated solution.

---

## Step 3 · GA: searching directly over permutations with legality-preserving operators

### 1 · Problem

If a GA's recombination operator splices two parent permutations arbitrarily, it produces illegal individuals in which one segment appears twice and another disappears. A set of operators closed within permutation space is therefore required.

### 2 · Approach

An individual is a permutation. Selection is by tournament, drawing a few individuals at random and taking the one with the lowest true $`F`$ as a parent. Recombination is order crossover (OX), which copies a random contiguous stretch from the first parent and then fills the remaining positions with the segments not yet used, in the relative order they appear in the second parent. The result is necessarily a legal permutation that inherits a run of absolute positions from one parent and the relative order of the rest from the other (util/metaheuristic.py:135-149). Mutation is one or two random swaps (util/metaheuristic.py:152-157). Each generation keeps a few elites by $`F`$ unchanged into the next, and the population is deduplicated before that selection so that the elite slots land on mutually distinct orders rather than several copies of one best schedule. The loop continues until the evaluation budget is exhausted or the stall guard fires (util/metaheuristic.py:160-185).

### 3 · Interpretation · why OX rather than a generic crossover?

The semantics of a scheduling problem live mainly in the relative order of who comes before whom, and OX mixes exactly that ordering information from both parents while guaranteeing legality. The small swap mutation supplies local perturbation. Elitism serves to keep the currently best individual breeding rather than being flushed out by a full generational replacement. The initial population is not purely random but is seeded with heuristic orders; the seeding mechanism, the guarantee it brings, and what that guarantee actually depends on are all handled together in Step 5. The product is a stream of candidates written continuously into the FitnessCache, from which the best solution is read after the search ends.

---

## Step 4 · PSO: a random-key encoding that grafts continuous dynamics onto permutations

### 1 · Problem

PSO's velocity and position updates are defined on continuous vectors, whereas a permutation has no meaning for "add a little velocity", so the standard scheme cannot be applied directly.

### 2 · Approach

The random-key encoding is used. Each particle is a real vector in $`[0,1]^n`$, decoded by argsort over its components, with the smaller component repaired earlier. Velocity and position then iterate by the standard gbest update

```math
v \leftarrow w\,v + c_1 r_1 (p_{\text{best}} - x) + c_2 r_2 (g_{\text{best}} - x), \qquad x \leftarrow x + v
```

after which out-of-range components are clipped back into $`[0,1]`$. Term by term: $`w = 0.7`$ is the inertia weight and sets how much of the previous step's velocity is retained; $`c_1 = c_2 = 1.5`$ are the attraction coefficients pulling the particle toward its own historical best position $`p_{\text{best}}`$ and toward the swarm's historical best position $`g_{\text{best}}`$; and $`r_1`$ and $`r_2`$ are componentwise uniform random numbers, which make the two pulls stochastic in strength rather than fixed. Both best positions are maintained by the true $`F`$ of the decoded order (util/metaheuristic.py:206-241).

The initial swarm is likewise seeded with the heuristic orders (Step 5), injected by inverse encoding: the segment sitting at rank $`r`$ is given the key $`(r+0.5)/n`$, so that argsort reproduces exactly that order (util/metaheuristic.py:191-203).

### 3 · Interpretation · why a random-key encoding rather than a purpose-built discrete velocity?

The random-key encoding leaves PSO's standard continuous dynamics untouched while every particle decodes to a legal permutation at any moment, at the implementation cost of one extra argsort. The price is that the mapping from keys to orders is many-to-one, since many distinct real vectors decode to the same permutation. The memoization of Step 2 absorbs exactly this redundancy, because a repeatedly decoded order hits the cache directly and spends no budget. The product is the same as in Step 3.

---

## Step 5 · Greedy seeding plus elite memory: starting the metaheuristics from the strongest heuristic

### 1 · Problem

At $`n=13`$ the permutation space holds $`13! \approx 6.2 \times 10^9`$ orders while the budget is only a few dozen evaluations. A purely randomly initialized population is under such a budget almost certain to be worse than the heuristics across the board, and a comparison of that kind measures nothing.

### 2 · Approach

Both populations are seeded with the three static greedy orders (flow, demand and ratio) as initial members, with the remaining members filled randomly. The scoring logic reproduces util/greedy.py exactly, so that the seeds really are those baselines' solutions (util/metaheuristic.py:121-129 for the seed orders; util/metaheuristic.py:164 and util/metaheuristic.py:215-216 for the injection; util/metaheuristic.py:287 for the per-scenario generation). Setting the switch seed_greedy to False falls back to purely random initialization.

### 3 · Interpretation · where does the "no worse than the best seed" guarantee actually come from?

It follows directly from two facts. The seeds are truly evaluated in the first generation and written into the FitnessCache, and the final output is the best-ever entry of that cache (Step 2). Hence, whatever course the later search takes, the output is provably no worse than the best of the three seeds, which in the experiments is usually flow. GA's deduplicated elitism and PSO's personal and global best memories do not carry this guarantee; their role is to let the ordering information carried by the seeds keep steering the generation of later candidates rather than merely sitting in the cache.

### 4 · Interpretation · what question does seeding let the metaheuristic baselines answer?

It changes the question from whether blind search can stumble onto a good solution into how much a budgeted search can still improve on top of the strongest heuristic, and the latter is the meaningful baseline question. Seeding is also symmetric with the MILP's warm start (note 02, Step 2), so that both advanced methods start from the same floor and the win-loss readings are fair. The constraint this construction places on interpretation is in Caveats C2.

---

## Step 6 · Per-scenario running and output: the same schema as greedy, so comparison needs no configuration

### 1 · Problem

The $`M`$ scenarios have different repair durations, so search results do not transfer between them and each scenario must be solved independently. The results must also be discoverable by the comparison harness without any registration step.

### 2 · Approach

The outer loop runs scenario by scenario. Each combination of scenario and method derives a deterministic RNG, seeded jointly from the global seed, the scenario number and the method, which keeps runs reproducible, and a fresh FitnessCache, because durations have changed and cached $`F`$ values cannot be reused. As soon as a method finishes, its row for that scenario is written back to {variant}_optima.csv with columns identical to static greedy plus one extra column n_evals recording the unique evaluations actually spent. On restart the existing CSV is read back and completed scenarios are skipped, which is resumable checkpointing and pure engineering (util/metaheuristic.py:272-282 for resume; util/metaheuristic.py:284-314 for the main loop and the write). Because both the schema and the directory match greedy, the wildcard discovery of the comparison harness (Step 7) will list GA and PSO as two new baselines on its next run, with no change to any comparison code.

---

## Step 7 · The comparison harness: automatic discovery, per-scenario alignment, two standard figures

### 1 · Problem

The methods keep multiplying, with three static greedy variants, GA, PSO, the MILP, and the oracle at small scale. If every new method required editing the comparison code, maintenance would not scale. The results are also scattered across directories and the MILP uses different column names, so they must be aligned before anything can be compared.

### 2 · Approach

run_compare scans outputs/greedy/n{N}/ by wildcard for every *_optima.csv, treats each file as one method, and merges in the MILP's milp_optima.csv. If that scale's oracle results exist, they are merged as well and a gap column against the oracle optimum is computed per method. Everything is aligned by scenario number into one comparison.csv, a summary sorted by mean is printed together with the number of scenarios in which the MILP beats each baseline, and a per-scenario bar figure is drawn (util/compare.py:23-70).

run_baseline_figures produces two further figures, process_and_final and accuracy_vs_time (util/compare.py:73-108). The former holds two panels. Panel a draws two kinds of element. One is the MILP's best-so-far true $`F`$ trajectory over its rounds, one curve per scenario. The other is the final $`F`$ of every other method, one horizontal line per method, taken as the across-scenario mean. The horizontal lines likewise come from wildcard discovery, so every method that has a same-schema CSV on disk in the greedy directory at drawing time is included, and GA and PSO therefore enter the figure without registration either (util/compare.py:87-88). The highlighted scenario in panel a is chosen automatically as the one whose best-so-far falls the furthest (viz/compare_viz.py:103-127). Panel b draws the final $`F`$ of every method per scenario as grouped bars (viz/compare_viz.py:129-156). accuracy_vs_time places each method on a plane of mean elapsed time against mean $`F`$ to display the accuracy-cost tradeoff (viz/compare_viz.py:164-189).

### 3 · Interpretation · why do the two kinds of element in panel a have deliberately different granularity?

The curves are per scenario while the horizontal lines are across-scenario means, because that panel only carries a schematic contrast between an iterative process and a solution given in one shot. It does not carry a rigorous comparison, and rigorous conclusions should be taken from the summary in comparison.csv. On the same reasoning, the highlighted scenario is the one that best displays the iteration dynamics, so the drop shown by the highlighted curve does not represent a typical magnitude; a fixed scenario number is not used because it might land on a scenario that barely moves and would show no process at all.

### 4 · Interpretation · why wildcard discovery rather than explicit registration?

It reduces the cost of adding a baseline to zero, as GA and PSO demonstrate by entering every comparison on the comparison script's next run purely by writing a same-schema CSV. Note that the discovery happens at run time: a comparison.csv or figure already on disk reflects only the methods that existed when it was generated, so run_compare and run_baseline_figures must be rerun after a new method lands before its results are taken into the comparison products. The time axis of accuracy_vs_time must be read with the parallelism distortion in mind, as set out in Caveats C1.

---

# Caveats

## C1 · The time_s of different methods is measured on different bases, so wall-clock cannot be compared across methods

The time_s of greedy and of the MILP is serial wall-clock, whereas GA and PSO spread their evaluations over (by default 16) worker processes, making their time_s parallel wall-clock. Drawing both on one time axis makes the metaheuristics look faster than a fair basis would show, by roughly the degree of parallelism. The fair compute measure, independent of hardware and of parallelism, is the serial-equivalent UE count: greedy spends $`T`$ per scenario, the MILP spends $`(\text{iterations}+1) \cdot T`$ (already written out as the ue_solves column), and GA and PSO spend $`\text{n\_evals} \times T`$.

In the $`n=13`$ experiment at budget 60, the horizon bound of C4 gives $`T = 42`$, so greedy is exactly 42. The MILP is about 836, the per-scenario average of $`(\text{iterations}+1) \cdot 42`$ over scenarios whose iteration counts differ, where the $`+1`$ counts the one full evaluation of the warm-start initial schedule before the alternating loop begins (note 02, Step 2). GA and PSO are $`60 \times 42 = 2520`$. The metaheuristics' best $`F`$ is therefore bought with the most compute of any method, and any reading of them as both fast and accurate is an artefact of parallelism.
(util/greedy.py:112, util/metaheuristic.py:293 and util/metaheuristic.py:300-301, util/pretrain_milp.py:316)

## C2 · Under seeding, beating greedy is a guarantee by construction, and only the margin carries information

The seed injection of Step 5, combined with the best-ever readout from the cache in Step 2, makes the GA and PSO results no worse than the strongest greedy by construction. A reading such as "the metaheuristics beat flow across the board" therefore carries no information in itself; what carries information is the size of the margin and the compute paid for it. By the same token the results should be read as how much a budgeted search can improve on top of the strongest heuristic, not as how strong a blind metaheuristic is, since answering the latter would require a control experiment with seed_greedy set to False.
(util/metaheuristic.py:26-29 for the module's own statement, util/metaheuristic.py:287)

## C3 · PSO may underspend its budget (a latent issue that the runs on disk do not trigger)

Once the swarm converges in key space, all particles may decode to the same already-cached order, after which each generation produces zero new evaluations and the stall guard exits while budget remains. The evidence on disk covers only the setting $`n=13`$ at budget 60, and there the issue does not arise: GA and PSO both spend exactly 60 in n_evals across all 10 scenarios, consuming the nominal budget in full. At which larger budget underspending begins there is no evidence in the repository, and this note makes no numerical claim about it.

Only the trend is certain: the closer the budget comes to the size of the permutation space, the more inevitable underspending becomes. The extreme case is applying the module's default budget of 120 to the default small instance, where config.py's default of 4 damaged segments admits only $`4! = 24`$ distinct permutations while the cache is keyed by the decoded permutation, so unique evaluations cannot exceed 24 and the stall guard must exit before the budget is spent, GA included. Should underspending occur, the compute basis should be taken from the n_evals written out rather than from the nominal budget, or diversity should be restored by re-injecting random particles on stall.
(util/metaheuristic.py:95 and util/metaheuristic.py:104 for the permutation-keyed cache and the budget test; util/metaheuristic.py:170 and util/metaheuristic.py:229-241 for the two stall guards; util/metaheuristic.py:300-301 for writing n_evals; config.py:47 for the default damaged count)

## C4 · The n=13 results on disk used a horizon upper bound and a non-default budget

### The horizon substitution

The repository's compute_horizon determines the horizon by enumerating all $`N!`$ permutations and taking the largest completion time (note 01, Step 4), which is infeasible at $`n=13`$. The $`n=13`$ results on disk were therefore produced by a runner script outside the repository, which replaces compute_horizon by the Graham list-scheduling bound, taking per scenario

```math
T = 1 + \left\lceil \frac{\sum_e d_e + (m-1)\max_e d_e}{m} \right\rceil
```

with $`m`$ the crew count, and then the maximum over scenarios. This is the classical Graham bound for list scheduling: the completion time of any work-conserving list schedule is provably no greater than $`(\sum_e d_e + (m-1)\max_e d_e)/m`$. Term by term: $`\sum_e d_e / m`$ is the ideal makespan if the total work divided perfectly among $`m`$ crews; $`(m-1)\max_e d_e / m`$ is the classical slack that covers the worst tail, in which the longest segment starts last while the other crews idle; the ceiling makes $`T`$ a whole number of slots; and the $`+1`$ corrects an offset, because the classical bound is derived with the first repair starting at time zero whereas this project's list scheduling starts the first repair no earlier than slot 1, shifting every completion time by one slot (util/evaluate.py:101-102).

### Why the substitution truncates nothing

The argument has two sides. Every baseline's schedule is generated by work-conserving list scheduling, so its completion time is covered by the bound above. The MILP side does not rely on that argument at all, because its feasible set is already bounded by $`T`$, since the variable upper bounds forbid any start whose completion falls later than $`T`$ (note 02, Step 4), and enlarging $`T`$ only relaxes that constraint, so no truncation arises there either.

The cost of the substitution is that the exact horizon is the maximum completion time over all work-conserving permutations while the bound is no smaller than any of them, so the substituted $`T`$ can only be too large and lengthens the averaging window. That lengthened window is identical for every method at this scale, so per-scenario comparisons between methods are unaffected; only the absolute value of $`F`$ is diluted by the extra post-recovery slots, and it cannot be compared directly against numbers computed under an exact horizon. At the scales where compute_horizon is still feasible, $`n \le 6`$, the $`T`$ from this bound was checked against the exact enumerated horizon and the implementation was confirmed correct, but that was a correctness spot check, and the no-truncation guarantee comes from the two-sided argument above rather than from that check.
(util/oracle.py:139-149 for the default horizon; util/pretrain_milp.py:138-143 for the MILP's horizon bound)

### Run parameters

That runner ran at budget 60 with 16 workers, whereas the module's default budget is 120. Reproducing the $`n=13`$ numbers requires the same horizon substitution and the same budget; running `python -m util.metaheuristic` directly applies only at scales where compute_horizon is feasible.
(util/metaheuristic.py:252-253 for the default budget and worker count)

---

# Appendix A · Parameter defaults

None of the GA and PSO parameters live in config.py, and they come in two kinds. The rows anchored at util/metaheuristic.py:248 and 249 are module-level constants, whereas the three rows anchored at 253 (budget, workers, seed_greedy) are keyword-argument defaults of the entry point run_metaheuristic.

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
