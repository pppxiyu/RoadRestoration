# Literature Search and Screening Report

```
WOS topic query (§1)                       1475 papers (all journal articles)
        │
        ▼
Filter A  Preliminary hard exclusion (§2.1)    −134 → 1341 papers
        │   pre-disaster only −66; non-transport lifeline only −68
        ▼
Filter B  Decision problem type (§2.2)  ┐
                                        ├─ intersection → 264 papers
Filter C  Studied network object (§2.3) ┘
        │
        ▼
Filter C refinement  operated object of interdependent papers (§2.4)   −73; reviews moved out −1
        │
        ▼
Final reference set: 190 papers ──→ Stage 2: six-dimension labeling (§3.1–3.6)
```

This report documents the systematic literature work for the post-disaster road-restoration scheduling topic. The work consists of three steps: candidate papers were first exported from Web of Science using a topic query; Stage 1 then applied sequential filters to converge on the final reference set; finally, Stage 2 labeled every paper in the final set along several dimensions. The per-paper decisions and labels are recorded in `screening_results.csv`, whose columns are distinguished by prefix (`s1_` for Stage 1; `s2_` and `final_*` for category filtering and final disposition; `s3_` for the Stage 2 dimension labels).

The final reference set contains 190 papers (`final_curated=True`). All papers were classified by human judgment at the abstract level, and key subsets were independently double-checked by a second group. The overall funnel is: 1475 papers retrieved → 1341 after Filter A → 264 in the intersection of Filter B and Filter C → 190 after the interdependent refinement and removal of reviews. Each layer takes intersections only, so the set can only shrink, never grow.

The per-paper screening table `screening_results.csv` is included in this repository (in the parent `literature_analysis/` directory); the raw WOS export `savedrecs.bib` and the working files under `_dims_work/` are kept locally and are not included.

---

## 1 Search Strategy

### 1.1 Search Framework

Keywords are organized into concept blocks, connected by OR within a block and by AND across blocks. The overall logic is: hazard or disruption, AND transportation-network object, AND restoration or recovery, AND prioritization, optimization, or learning methods. The prioritization (sequencing) terms and the optimization-method terms are merged into a single parenthesis connected by OR: many paper titles contain only one of scheduling, sequencing, or optimization, so requiring both prioritization and optimization simultaneously would cause missed retrievals. If the initial retrieval is too large, this parenthesis can be split back into two independent AND conditions to tighten the query.

| Concept block | Keywords (OR) |
|---|---|
| A Hazard or disruption | hurricane, typhoon, cyclone, flood\*, inundation, "storm surge", earthquake, seismic, tsunami, landslide, "natural hazard\*", "natural disaster\*", "extreme weather", disaster\*, postdisaster, "post-disaster", disruption\* |
| B Object (transportation network) | "road network\*", "roadway network\*", "transportation network\*", "transport network\*", "road infrastructure", "transportation infrastructure", "transportation system\*", "transport system\*", "infrastructure network\*", highway\*, roadway\*, "road segment\*", bridge\*, subway, metro, transit |
| C Restoration | restor\*, recover\*, repair\*, reconstruct\*, rehabilitat\*, reopen\*, "debris removal", "debris clearance", reconfigur\*, "service restoration" |
| D Prioritization, optimization, or learning | prioriti\*, priority, ranking, sequencing, "restoration sequence", "restoration scheduling", "recovery scheduling", scheduling, "resource allocation", "crew scheduling", criticality, optimi\*, "integer programming", "mixed-integer programming", MILP, "stochastic programming", heuristic, metaheuristic, "reinforcement learning", "decision support" |

### 1.2 WOS Topic Query

```
TS=(
  ( hurricane OR typhoon OR cyclone OR flood* OR inundation OR "storm surge" OR earthquake OR seismic OR tsunami OR landslide OR "natural hazard*" OR "natural disaster*" OR "extreme weather" OR disaster* OR postdisaster OR "post-disaster" OR disruption* )
  AND
  ( "road network*" OR "roadway network*" OR "transportation network*" OR "transport network*" OR "road infrastructure" OR "transportation infrastructure" OR "transportation system*" OR "transport system*" OR "infrastructure network*" OR highway* OR roadway* OR "road segment*" OR bridge* OR subway OR metro OR transit )
  AND
  ( restor* OR recover* OR repair* OR reconstruct* OR rehabilitat* OR reopen* OR "debris removal" OR "debris clearance" OR reconfigur* OR "service restoration")
  AND
  ( prioriti* OR priority OR ranking OR sequencing OR "restoration sequence" OR "restoration scheduling" OR "recovery scheduling" OR scheduling OR "resource allocation" OR "crew scheduling" OR criticality OR optimi* OR "integer programming" OR "mixed-integer programming" OR MILP OR "stochastic programming" OR heuristic OR metaheuristic OR "reinforcement learning" OR "decision support" )
)
```

---

## 2 Stage 1: Screening (1475 → 190)

Stage 1 converges the retrieved corpus to the final reference set through three sequential filters: Filter A performs a preliminary hard exclusion, Filter B screens by decision problem type, and Filter C screens by the studied network object. The final reference set is the intersection of all three, followed by one additional object-level refinement of the interdependent-infrastructure subclass.

### 2.1 Filter A: Preliminary Hard Exclusion (1475 → 1341)

Filter A removes papers that are definitively unrelated to this topic using two binary rules, excluding 134 papers in total; results are recorded in the CSV column `s1_status`. Both rules were independently double-checked by a second group with a keep-by-default policy, so the exclusions have high precision.

| Rule | Removed | Criterion |
|---|---|---|
| Temporal phase (remove if pre-disaster only) | 66 | A paper is removed only if it belongs entirely to the pre-disaster domain (mitigation, preparedness, planning, design, retrofitting, or pre-disaster risk and vulnerability assessment) and contains no post-disaster restoration, recovery, or emergency-response content whatsoever; any post-disaster recovery content leads to retention. |
| Lifeline (remove if non-transport lifeline only) | 68 | A paper is removed only if its studied object is entirely a non-transport lifeline (electricity, water, gas, telecommunications) and contains no road, bridge, or rail-transit component whatsoever; the presence of any transport layer (even as just one layer of a coupled system) leads to retention. |

### 2.2 Filter B: Decision Problem Type (problem_type, 13 categories)

Filter B determines which category of decision problem a paper solves and keeps only the three categories directly related to restoration scheduling. The counts in the table below are distributions over the 1341 papers that passed Filter A.

| Category | Count | Detailed meaning | Decision |
|---|---|---|---|
| restoration-scheduling/sequencing | 238 | Determines the order or timing of post-disaster restoration, i.e., which component to repair first and when, directly producing a restoration plan; the topic of this study itself belongs to this category. | ✅ Keep |
| crew/resource-routing-dispatch | 25 | Routes and schedules repair crews and maintenance resources (a vehicle-routing-type problem), tightly coupled with restoration scheduling and often jointly optimized with the repair sequence. | ✅ Keep |
| debris-clearance | 16 | Post-disaster debris clearing and road clearance, where the decision object is removing roadblocks and restoring passability. | ✅ Keep |
| resilience-quantification/assessment | 162 | Quantifies network resilience, robustness, or functionality-recovery curves, but contains no optimization decision; evaluation only. | ❌ Remove |
| network-design/retrofit/protection | 115 | Pre-disaster hardening, protection, survivability design, or network interdiction; a pre-disaster decision rather than post-disaster restoration. | ❌ Remove |
| vulnerability/risk/fragility-assessment | 90 | Vulnerability, risk, or fragility assessment, as well as disaster-loss assessment; produces no restoration plan. | ❌ Remove |
| empirical-recovery-analysis/prediction | 78 | Data-driven or descriptive analysis and prediction of recovery processes, without normative optimization. | ❌ Remove |
| hazard/damage-modeling | 34 | Models the hazard itself or the physical damage process; involves no network-level decision. | ❌ Remove |
| link-criticality/importance-ranking | 31 | Identifies and ranks critical links or nodes, but stops at importance ranking and produces no restoration plan. | ❌ Remove |
| post-disaster-relief-logistics/distribution | 30 | Relief distribution, emergency logistics, or response facility location; the decision object is supplies rather than road-network restoration. | ❌ Remove |
| evacuation | 19 | Evacuation planning, order timing, or evacuation routing; oriented toward evacuation rather than restoration. | ❌ Remove |
| pre-disaster-mitigation/preparedness/planning | 2 | Pre-disaster mitigation and preparedness planning (small residual). | ❌ Remove |
| other | 501 | Query false positives unrelated to this topic, such as medicine, ecology, reservoir "flooding" in petroleum engineering, signal or image "reconstruction", social capital, and masonry repair. | ❌ Remove |

Filter B keeps only restoration-scheduling/sequencing, crew/resource-routing-dispatch, and debris-clearance; 279 papers pass in total.

### 2.3 Filter C: Studied Network Object (object, 10 categories)

Filter C determines which kind of network or asset a paper studies and keeps only the five transport-related categories.

| Category | Count | Detailed meaning | Decision |
|---|---|---|---|
| road/highway-network | 250 | Studies road, highway, or urban street networks. | ✅ Keep |
| bridge/bridge-network | 208 | Studies bridges as the damaged and repaired assets; post-earthquake or post-flood bridge-restoration sequencing is a major branch of this field, and many such problems are equivalent forms of road-network problems. | ✅ Keep |
| railway/transit | 182 | Studies railway, metro, or bus networks. | ✅ Keep |
| interdependent-infrastructure-incl-transport | 144 | Coupled multi-system including a transport layer, i.e., transport interdependent with electricity, water, and other systems; this label requires further distinction of which object is actually restored — see the refinement in the next subsection. | ✅ Keep |
| general-transportation-network | 114 | Multi-modal or unspecified transportation networks. | ✅ Keep |
| other | 311 | Query false positives, same as the other category in Filter B. | ❌ Remove |
| building/urban-area/community | 62 | Studies buildings, urban areas, or communities rather than transportation networks. | ❌ Remove |
| port/airport/maritime/waterway | 49 | Studies ports, airports, or waterways. | ❌ Remove |
| supply-chain/facility | 18 | Studies supply chains, production systems, or standalone facilities. | ❌ Remove |
| lifeline-only-power-water-gas-telecom | 3 | Involves only non-transport lifelines; the vast majority were already removed by Filter A, and these are residuals. | ❌ Remove |

Filter C keeps only road/highway-network, bridge/bridge-network, railway/transit, interdependent-infrastructure-incl-transport, and general-transportation-network; 898 papers pass in total.

The roughly three hundred papers in the other category of each of Filter B and Filter C are the main force in shrinking the corpus, and almost all of them are query false positives: flooding matches reservoir waterflooding in petroleum engineering, reconstruction matches signal and image reconstruction, and bridge matches bridging or social capital bridging, in addition to a large amount of medical and ecological noise. The entire category can therefore be removed safely.

### 2.4 Filter C Refinement: Operated Object of Interdependent Papers (144 papers)

The label of a coupled system including transport cannot distinguish whether a paper actually restores the road network or the power grid. The 144 interdependent papers were therefore further examined for their operated object, i.e., the decision object that is actually restored or scheduled. The criterion is: if the transportation network serves merely as the travel medium for mobile power sources, electric vehicles, or repair crews, while the object actually restored is the power grid, then the operated object is recorded as the power grid. Key subsets were independently double-checked by a second group; per-paper results are in the CSV column `s2_interdep_operated`.

| Operated object | Count | Decision |
|---|---|---|
| transport-and-lifeline-jointly | 36 | ✅ Keep: transport and lifelines are jointly the decision objects and are restored together (including 5 papers originally judged uncertain, retained by the 2026-07-07 ruling that their transport layer is jointly operated). |
| road/highway-network | 1 | ✅ Keep: the object actually restored is the road. |
| power-electricity-system | 80 | ❌ Remove: the object actually restored is the power grid, with the transportation network serving only as the travel medium for mobile power sources, electric vehicles, or repair crews. |
| water-gas-telecom-lifeline | 2 | ❌ Remove: the object actually restored is the water, gas, or telecommunications lifeline. |
| uncertain | 25 | ⚠️ The operated object cannot be determined from the abstract; most of these were already removed by Filter B, those passing the intersection of Filter B and Filter C were handled case by case, and no pending rulings remain. |

Among the interdependent papers that passed the intersection of Filter B and Filter C, 73 whose operated object is the power grid or water/gas systems were removed at the object level (recorded in the CSV as `final_status = removed_filterB_refined`); those restoring transport or restoring jointly were retained.

### 2.5 Screening Results (1341 → 264 → 190)

Applying Filter B and Filter C to the 1341 papers remaining after Filter A, 264 papers pass their intersection (both Filter B and Filter C). Of the removed papers, 634 failed only the problem-type criterion, 428 failed both problem type and object, and 15 failed only the object criterion. The CSV column `final_status` retains the historical value names `removed_filterA`, `removed_filterA+B`, and `removed_filterB`, which correspond respectively to this report's Filter B, the intersection of Filter B and Filter C, and Filter C.

On top of the 264 papers, the interdependent refinement and the removal of reviews yield the final reference set of 190 papers (`final_curated=True`); the set only shrinks, never grows:

| Disposition | Count | Note |
|---|---|---|
| Final reference set | 190 | `final_curated=True` |
| removed_filterB_refined | 73 | Interdependent but actually restoring the power grid or water/gas systems; removed at the object level |
| moved_to_review | 1 | Passed the intersection of Filter B and Filter C but is itself a review (idx 20); moved to the review list (see Appendix C) |

The 264 papers passing Filter B and Filter C are captured by `final_status` (values `kept`, `removed_filterB_refined`, and `moved_to_review`), while `final_curated=True` marks the final reference set of 190; all labeling analysis in Stage 2 is based on these 190 papers.

---

## 3 Stage 2: Dimension Labels for the Final Set

Stage 2 performs no further removal; it only labels each of the 190 papers in the final set along several dimensions, for positioning during writing. The labels fall into two groups: objective type and traffic model (Sections 3.1 and 3.2), and solution method, decision architecture, demand dynamics, and uncertainty (Sections 3.3 to 3.6); all six are stored in the CSV columns prefixed `s3_`.

The latter four dimensions, as well as the re-checks of the objective and traffic dimensions, were all validated against full texts: full-text retrieval from the publishers was attempted for all 190 papers, of which 184 were successfully read in full and 6 could not be obtained due to paywalls; the latter retain the best abstract-level judgment. The overall effect of the full-text validation and the novelty cross-tabulation are given in Appendix A.

### 3.1 Objective Type (objective_type, 8 categories)

Each paper is assigned its dominant metric; a paper is placed in composite only if it genuinely balances two or more heterogeneous objectives simultaneously.

| Category | Count | Detailed meaning |
|---|---|---|
| resilience-area/functionality | 67 | Takes as objective the area under the functionality-versus-time recovery curve (the resilience triangle), the time integral of performance loss, or a composite resilience index combining robustness and rapidity; it measures the shape of the entire recovery process rather than a single end state, and is the most mainstream formulation in this field. |
| multi-criteria/composite | 44 | Explicitly balances two or more heterogeneous objectives simultaneously, in the form of weighted sums, ε-constraints, or Pareto frontiers, e.g., travel time with repair cost, or accessibility with completion time; this study belongs to this category (F1 travel time and F2 makespan). |
| travel-time/accessibility | 21 | Takes total travel time, average travel time, delay, or accessibility (arrival time or distance) as a single objective, without a resilience time integral; it looks only at the level at a given moment or after convergence, and shares the formulation of this study's F1. |
| connectivity/reachability | 14 | Takes binary connectivity, the size of the largest connected component, the number of reachable OD pairs, or path existence as the objective; it asks only whether the network is connected, not how fast it flows. |
| flow/demand-served | 15 | Takes the amount of travel demand satisfied, flow carried, supplies delivered, or population served as the objective, or minimizes the amount of unmet demand. |
| economic-cost/loss | 15 | Takes monetized metrics as the objective, including repair or reconstruction cost, budget constraints, and direct and indirect economic losses. |
| recovery-time/makespan | 12 | Takes the total restoration duration, makespan, completion time, or the time required to reach a functionality threshold as the objective; it looks only at the temporal endpoint, without integrating over the process. |
| other | 2 | Other miscellaneous objectives. |

### 3.2 Traffic Model (traffic_model, 5 categories)

A dividing line is drawn by whether the model computes any traffic speed, time, or flow on the network: the top three categories involve traffic state (108 papers), the next two involve none at all (59 papers), and a sixth `uncertain` bucket (23 papers) holds cases whose two full-text reads disagreed on this two-class question with no decisive feature to settle it (see Appendix A.3). The shortest-path/AON ↔ topological/graph-metric boundary is genuinely fuzzy — an efficiency or accessibility metric computed over shortest paths sits on the line — so rather than force a single label, those cases are marked `uncertain`.

| Category | Count | Detailed meaning | Involves traffic state? |
|---|---|---|---|
| shortest-path/AON | 37 | Computes shortest paths on fixed (flow-independent) link costs, or performs all-or-nothing (AON) assignment: travel time equals the sum of static link travel times (the shortest-path length), and OD demand is loaded onto the shortest path in one block to obtain link flows; travel times and flows exist, but link costs do not vary with flow and there is no congestion feedback. | ✅ Yes (travel time and flow, static) |
| network-flow/capacity | 31 | Uses maximum-flow, minimum-cost-flow, or capacitated (multi-commodity) flow models: flow is explicitly modeled subject to link-capacity constraints, but link costs are flow-independent and there is no equilibrium iteration; the concern is how much flow can pass or how much OD demand can be met, without computing congested travel times. | ✅ Yes (flow, capacity-constrained) |
| user-equilibrium | 40 | Uses Wardrop user equilibrium (UE) or stochastic user equilibrium (SUE): link cost is a function of flow (e.g., BPR), and equilibrium flows and link travel times are solved iteratively, capturing speed, time, and flow together with their congestion feedback; this category is closest to the core of this study's UE simulator. | ✅ Yes (speed, time, and flow, with congestion feedback) |
| topological/graph-metric | 37 | Uses only graph-theoretic performance metrics: connectivity, size of the largest connected component, betweenness centrality, network efficiency in the hop-count sense, number of reachable OD pairs, or path counts; links carry no cost, travel time, or flow, and performance is determined entirely by the topological connectivity structure. | ❌ No (purely topological) |
| none/not-modeled | 22 | No routing, flow, or travel-time model exists on the network at all: the objective is structural functionality, component-level repair, an integrated resilience curve, or another non-flow metric, and traffic state never enters the model. | ❌ No (no network flow model) |
| uncertain | 23 | Two independent full-text reads disagreed on whether the model involves traffic state and no decisive feature resolved it — chiefly the shortest-path/AON ↔ topological/graph-metric boundary (efficiency/accessibility metrics over shortest paths, gravity models, composite indices). | ⚠️ Undetermined |

### 3.3 Solution Method (solution_method, multi-label, 10 categories)

This dimension is multi-label: a paper can match multiple methods simultaneously (e.g., a matheuristic combining MILP with heuristics), so the category counts sum to more than 190.

| Category | Count | Detailed meaning |
|---|---|---|
| heuristic/metaheuristic/GA | 124 | Greedy and priority rules, genetic algorithms, particle swarm, simulated annealing, ant colony, tabu search, local search, and the heuristic part of matheuristics; because large instances are mostly NP-hard, this is the workhorse of the field. |
| exact/MILP/math-programming | 69 | Exact mixed-integer (linear or nonlinear) programming, integer programming, or linear programming solved to optimality by CPLEX, Gurobi, etc., or other exact mathematical programming. |
| other | 22 | Enumeration, ranking and scoring, multi-criteria decision making (MCDM), analytical methods, or evaluation frameworks without a formal optimizer. |
| simulation-optimization | 16 | Candidate solutions are evaluated by a simulation loop (traffic simulation or Monte Carlo), which drives the optimization. |
| reinforcement-learning/DRL | 15 | Reinforcement learning or deep reinforcement learning, i.e., learning an MDP policy (Q-learning, DQN, PPO, actor-critic, etc.); an emerging method in recent years. |
| decomposition/Benders/Lagrangian | 13 | Decomposition-type methods such as Benders decomposition, Lagrangian relaxation, column generation, or ADMM. |
| exact/dynamic-programming | 10 | Exact dynamic programming or backward induction. |
| other-ML/supervised-learning | 7 | Supervised learning or other non-RL machine-learning surrogate models. |
| graph-neural-network/GNN | 5 | Graph neural networks (GNN or GCN) as the model, state encoder, or surrogate. |
| learn+optimize-hybrid/warmstart-pretraining | 3 | Hybridizes learning with optimization, or warm-starts/pretrains the learning component with solutions obtained from optimization. |

Across the entire corpus, the learn+optimize warm-start category is the rarest, with only 3 papers (idx 47, 50, 900). Although the learning cluster (15 DRL papers and 5 GNN papers) is growing, the path of warm-starting DRL and GNN with MILP or Benders optimal solutions remains almost untouched; see Appendix A.

### 3.4 Decision Architecture (decision_architecture, 3 categories)

| Category | Count | Detailed meaning |
|---|---|---|
| a-priori/static-schedule | 152 | The entire restoration sequence or schedule is computed once and is not re-solved as information arrives. |
| sequential/adaptive/rolling-policy | 34 | Step-by-step or rolling decisions, re-planned as the state or information is revealed, such as MDP policies, rolling horizons, or adaptive rescheduling; this study belongs to this category (a reinforcement-learning policy). |
| not-applicable | 4 | Produces no temporal schedule; only static priority ranking or assessment. |

### 3.5 Demand Dynamics (demand_dynamics, 3 categories)

| Category | Count | Detailed meaning |
|---|---|---|
| static-demand | 131 | Fixed OD demand that remains unchanged throughout the recovery process (route re-choice with fixed OD magnitude counts as static). |
| not-applicable | 39 | No demand or OD is modeled (structural or purely topological objectives). |
| dynamic/time-varying-demand | 18 | OD demand magnitude varies over time, responds to the network state, or reflects adaptive traveler behavior; this study belongs to this category (dynamic OD). |
| uncertain | 2 | Two full-text reads disagreed on whether the demand is dynamic vs non-dynamic (idx 27, 133; see Appendix A.3). |

### 3.6 Uncertainty (uncertainty, 5 categories)

| Category | Count | Detailed meaning |
|---|---|---|
| deterministic | 136 | All parameters are known and fixed (running a deterministic model on fixed what-if scenarios still counts as deterministic). |
| stochastic/scenario-based | 34 | Uses probabilistic parameters, scenarios, stochastic programming, or expected values inside the optimization; this study belongs to this category (stochastic repair durations). |
| adaptive/multistage-revelation | 8 | Uncertainty is revealed in stages with recourse, or multistage stochastic or online revelation. |
| robust/distributionally-robust | 6 | Robust optimization, distributionally robust optimization (DRO), or min-max. |
| fuzzy/other | 3 | Fuzzy sets, intervals, or possibility theory. |
| uncertain | 3 | Two full-text reads disagreed on deterministic vs non-deterministic (idx 61, 205, 515; see Appendix A.3). |

Non-deterministic approaches total 51 papers (34 stochastic, 8 multistage, 6 robust, 3 fuzzy); 3 more are undetermined.

---

## Appendix A　Validation

This appendix records two independent validation checks on the screening: an external recall check against a hand-curated reference set (A.1), and the full-text re-reading of the dimension labels (A.2).

### A.1 External Validation Against the Zotero Reference Set

As an external recall check, the pipeline was compared against an independently hand-curated reference set: the author's Zotero collection "5.2 decision", restricted to items tagged `!-road_restoration` or `!-infrastructure_network`. This set contains 28 journal articles the author had already identified as relevant, independently of this screening pipeline.

Of these 28 papers, 22 are present in the final 190-paper set (recall about 79%); they are flagged in the CSV column `ref_in_zotero`. Their idx: 3, 11, 18, 35, 37, 50, 61, 63, 68, 128, 133, 515, 562, 576, 592, 682, 702, 740, 1047, 1059, 1065, 1205.

The other 6 fall into two groups, both explainable:

- 2 were retrieved into the 1475-paper corpus but removed by a filter: idx 572 (community emergency-medical-service network reconfiguration, removed by Filter B as not a restoration-scheduling problem) and idx 814 (infrastructure-network restoration whose operated object is the power grid, removed by Filter C).
- 4 were never retrieved by the WOS query at all (a search-recall gap): Yang et al. 2024 (multi-agent DRL for community post-hazard recovery), Sun et al. 2022 (AI-informed planning for hazard-impacted road networks), Li and Wu 2022 (DRL decision support for transportation infrastructure under hurricanes), and Joo et al. 2022 (multi-agent DRL road reconstruction after the 2018 western-Japan flooding).

The two filtered-out papers confirm the filters behave as designed; the four missed papers mark the known recall limit of the query (consistent with the recall gap recorded for the v2 search string) and would be candidates for citation-chasing / snowballing.

### A.2 Full-Text Validation and Dimension Cross-Tabulation

The process and results of the full-text validation are as follows. Full-text retrieval was attempted for all 190 papers: first, the papers whose four dimensions or objective/traffic labels could not be judged confidently, together with a 20% random sample, totaling 119 papers, were read in full; the remaining 71 papers were then also read; finally, for the 30 papers behind paywalls, retries were made using 7 PDFs provided by the author, a SAGE GT account, and ScienceDirect previews. In the end, 184 papers were read in full and 6 remained unobtainable behind paywalls (see Appendix D for the list). The full-text validation corrected 20 objective-type labels and 36 traffic-model labels in total; the traffic model is the easiest to misjudge from an abstract, e.g., idx 8 was corrected from shortest-path to UE. After correction, the involving-versus-not-involving ratio for traffic changed from 105:85 to 115:75, user-equilibrium rose from 23 to 35, and dynamic demand rose from 13 to 23. (These are the figures from this first full-text pass; the labels were later re-checked with two independent reads per paper and an `uncertain` category was added — see Appendix A.3 for the current distributions.)

The dimension cross-tabulation corroborates novelty. The positioning combination of this study is: restoration-scheduling ∩ UE traffic ∩ composite objective ∩ learning methods (DRL, GNN, or warm-start) ∩ adaptive architecture ∩ dynamic OD ∩ stochastic repair durations. Narrowing layer by layer within the 190 papers (all figures calibrated by full text):

| Intersection condition | Count | idx |
|---|---|---|
| UE ∩ composite objective | 5 | 19, 96, 153, 692, 1314 |
| Adaptive ∩ learning methods ∩ UE | 2 | 35, 133 |
| Adaptive ∩ learning methods ∩ UE ∩ dynamic OD | 0 confirmed | (133, but its demand is now `uncertain`) |
| opt+RL warm-start hybrid (the method core of this study) | 3 | 47, 50, 900 |

With all positioning dimensions stacked, idx 133 (which shares the Sioux Falls testbed with this study) is the nearest neighbor: its full text confirms reinforcement learning with MILP subproblems, adaptive architecture, UE traffic, and stochastic repair durations. However, the two-read re-check (Appendix A.3) found its OD-demand dynamics ambiguous — the model evolves traveler routes day-to-day but holds OD *magnitude* constant ("assumed constant within the short-term restoration process"), which under this study's definition is static, not dynamic. Its `demand_dynamics` is therefore recorded as `uncertain`. Either way no paper is a confirmed complete neighbor along all positioning dimensions: if 133's demand is read as static, no corpus paper combines UE, adaptive learning, and dynamic OD; if it is read as dynamic, 133 alone is the single neighbor. The method core (warm-starting reinforcement learning with optimization solutions) has three neighbors, idx 47, 50, and 900, but idx 900 targets all-terminal reliability (involving neither UE nor dynamic demand) and thus still does not overlap with this study. The novelty holds along the independent positioning dimensions.

### A.3 Binary Re-Check of Traffic, Demand, and Uncertainty

All 190 papers were re-checked on `s3_traffic_model`, `s3_demand_dynamics`, and `s3_uncertainty` so that every readable paper carries **two independent full-text reads**: the original full-text label and a blind re-read that recomputed the three dimensions from scratch. 176 papers obtained two valid reads; the remaining 14 have a single read and keep it, flagged unverified (6 never retrievable — idx 42, 120, 147, 325, 741, 994 — and 8 blocked by a SAGE/Cloudflare interstitial at re-read time — idx 23, 84, 604, 744, 778, 884, 1043, 1314).

The rule was deliberately conservative, to avoid the trap of treating one read as ground truth on a fuzzy label. A disagreement was **fixed directly** only when a read named an unambiguous, checkable modelling feature the other had missed (explicit Wardrop/BPR user equilibrium; capacitated max-/min-cost/multi-commodity flow; "no OD travel demand" or time-indexed demand magnitude; a single fixed scenario with no random variables versus Monte-Carlo/Latin-hypercube sampling feeding the objective). Every remaining disagreement — chiefly the genuinely fuzzy shortest-path/AON ↔ topological/graph-metric boundary — was routed to a new **`uncertain`** category rather than force-picked. Two-read agreement at the two-class level: **traffic 143/176 (81%)**, **demand 160/176 (91%)**, **uncertainty 158/176 (90%)**.

Outcome across the 67 two-class disagreements: **39 decisive fixes** and **28 routed to `uncertain`**.

- **Decisive fixes (39).** traffic → user-equilibrium: 4, 6, 9, 27, 101; traffic → network-flow/capacity: 29, 30, 500, 711, 913. demand → dynamic: 4, 99, 913, 1100, 1101; demand → not-applicable: 498, 818, 849, 926, 1142, 1151; demand → static: 127, 595, 889. uncertainty → deterministic: 16, 18, 47, 68, 136, 555, 740, 881, 896, 1030, 1065, 1085; uncertainty → stochastic: 10, 49, 537. (The uncertainty fixes are almost all fixed-scenario what-ifs that had been over-labelled stochastic, plus three genuine stochastic models — 10, 49, 537 — that had been missed. The demand fixes correct interdependent-power papers whose road network carries only repair crews, i.e. no OD travel demand.)
- **Routed to `uncertain` (28).** traffic (23): 47, 60, 97, 99, 140, 271, 498, 515, 537, 568, 595, 680, 783, 825, 867, 893, 1042, 1057, 1071, 1100, 1108, 1129, 1140; demand (2): 27, 133; uncertainty (3): 61, 205, 515.

The demand and uncertainty dimensions are reliable at the two-class level (91% and 90%): their disagreements are mostly one-directional and feature-decidable, so they were largely fixed rather than left uncertain. The traffic model is the weakest (81%): its shortest-path/AON ↔ topological/graph-metric boundary is genuinely fuzzy — an efficiency, accessibility, or gravity metric computed over shortest paths sits on the line — so 23 papers are recorded `uncertain` rather than assigned a possibly-wrong label. The distributions in Sections 3.2, 3.5, and 3.6 reflect the post-re-check labels including the `uncertain` bucket. One consequence worth flagging: idx 133, the nearest novelty neighbour, has its demand recorded `uncertain` because its day-to-day route evolution runs on a **constant** OD magnitude (Appendix A.2).

---

## Appendix B　Years, Venues, and Classics

The year distribution is as follows. Of the 190 papers, recent years (2021 to 2026) account for 127 papers, about 67%, with 28 papers in 2026 alone, reflecting the research surge in deep reinforcement learning and optimization-based recovery; the earliest paper dates to 2003.

| Year | 2003 | 2007 | 2009 | 2010 | 2011 | 2012 | 2014 | 2015 | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Papers | 1 | 2 | 3 | 1 | 2 | 6 | 5 | 3 | 8 | 7 | 7 | 8 | 10 | 20 | 20 | 19 | 20 | 20 | 28 |

The venue distribution is highly dispersed: the 190 papers span reliability, structural engineering, transportation, operations research, and other fields. The table below lists only venues with 3 or more papers; the rest are in the CSV column `venue`.

| Venue | Papers |
|---|---|
| Reliability Engineering & System Safety | 15 |
| International Journal of Disaster Risk Reduction | 10 |
| Transportation Research Part D | 9 |
| Structure and Infrastructure Engineering | 7 |
| Journal of Infrastructure Systems | 7 |
| European Journal of Operational Research | 7 |
| Transportation Research Record | 7 |
| Computers & Operations Research | 6 |
| Natural Hazards (4), Sustainability (4) | 4 each |
| Comput. & Industrial Eng., Sustainable Cities, OR Spectrum, J. Computing in Civil Eng., Applied Sciences, IEEE Access, J. Management in Eng., TR Part B, TR Part E | 3 each |

Citation counts are taken from OpenAlex (by DOI, real-time cumulative) and now cover all 190 papers (189 resolved; idx 120 is not indexed by OpenAlex). `ref_is_classic=True` marks the 39 papers cited at least 50 times, listed below in descending order of citations. Counts are biased toward older papers, so a high count signals impact rather than closeness to this study. The relevance tier (core / method-neighbor / background) was assigned only for the earlier v1 subset; more recent papers were judged by content.

| Citations | Year | idx | Title |
|---|---|---|---|
| 378 | 2012 | 886 | Measuring and maximizing resilience of freight transportation networks |
| 270 | 2018 | 1101 | Resiliency assessment of urban rail transit networks: Shanghai metro as an example |
| 229 | 2012 | 96 | Restoration of Bridge Networks after an Earthquake: Multicriteria Intervention Optimization |
| 226 | 2009 | 1047 | Optimal scheduling of emergency roadway repair and subsequent relief distribution |
| 221 | 2017 | 61 | Resilience-based post-disaster recovery strategies for road-bridge networks |
| 216 | 2012 | 153 | Optimal Resilience- and Cost-Based Postdisaster Intervention Prioritization for Bridges along a Highway Segment |
| 170 | 2014 | 421 | A hierarchical compromise model for the joint optimization of recovery operations and distribution of emergency goods in Humanitarian Logistics |
| 158 | 2014 | 827 | A mathematical model for post-disaster road restoration: Enabling accessibility and evacuation |
| 148 | 2018 | 587 | A resilience optimization model for transportation networks under disasters |
| 134 | 2019 | 682 | Resilience-based transportation network recovery strategy during emergency recovery phase under uncertainty |
| 133 | 2014 | 741 | Optimal recovery sequencing for enhanced resilience and service restoration in transportation networks |
| 127 | 2016 | 22 | Network repair crew scheduling and routing for emergency relief distribution problem |
| 112 | 2020 | 702 | A post-disaster resource allocation framework for improving resilience of interdependent infrastructure networks |
| 112 | 2017 | 905 | Multi-vehicle synchronized arc routing problem to restore post-disaster network connectivity |
| 100 | 2019 | 539 | Community resilience-driven restoration model for interdependent infrastructure networks |
| 99 | 2018 | 1035 | Multi-objective, multi-period location-routing model to distribute relief after earthquake by considering emergency roadway repair |
| 97 | 2003 | 604 | Transportation planning for disasters: an accessibility approach |
| 97 | 2022 | 889 | Resilience assessment of railway networks: Combining infrastructure restoration and transport management |
| 83 | 2018 | 102 | Determination of Near-Optimal Restoration Programs for Transportation Networks Following Natural Hazard Events Using Simulated Annealing |
| 79 | 2007 | 40 | Fund allocation for transportation network recovery following natural disasters |
| 77 | 2021 | 4 | Optimal restoration schedules of transportation network considering resilience |
| 77 | 2014 | 555 | Coordinating debris cleanup operations in post disaster road networks |
| 77 | 2016 | 803 | Arc routing problems to restore connectivity of a road network |
| 73 | 2009 | 692 | Optimizing Postdisaster Reconstruction Planning for Damaged Transportation Networks |
| 71 | 2011 | 860 | A GRASP metaheuristic to improve accessibility after a disaster |
| 69 | 2019 | 1135 | Integrated optimal scheduling of repair crew and relief vehicle after disaster |
| 62 | 2021 | 676 | An online optimization approach to post-disaster road restoration |
| 61 | 2019 | 127 | Post-disaster multi-period road network repair: work scheduling and relief logistics optimization |
| 60 | 2022 | 181 | Prioritizing transportation network recovery using a resilience measure |
| 60 | 2019 | 1045 | Minimizing latency in post-disaster road clearance operations |
| 58 | 2023 | 1030 | Sustainability and climate resilience metrics and trade-offs in transport infrastructure asset recovery |
| 56 | 2021 | 18 | Resilience-based Recovery Scheduling of Transportation Network in Mixed Traffic Environment: A Deep-Ensemble-Assisted Active Learning Approach |
| 56 | 2017 | 113 | From Component Damage to System-Level Probabilistic Restoration Functions for a Damaged Bridge |
| 55 | 2015 | 576 | Scheduling Short-Term Recovery Activities to Maximize Transportation Network Resilience |
| 54 | 2017 | 186 | Multi-vehicle prize collecting arc routing for connectivity problem |
| 52 | 2016 | 12 | Sequencing algorithm with multiple-input genetic operators: Application to disaster resilience |
| 52 | 2019 | 913 | CRISIS: Modeling the Restoration of Interdependent Civil and Social Infrastructure Systems Following an Extreme Event |
| 51 | 2020 | 84 | Postdisaster Decision Framework for Bridge Repair Prioritization to Improve Road Network Resilience |
| 51 | 2022 | 1043 | Strategies to Enhance the Resilience of an Urban Rail Transit Network |

The most-cited works are older resilience and routing papers — idx 886 (378, freight-network resilience), idx 1101 (270, metro resiliency), idx 96 (229, bridge-network restoration), and idx 1047 (226, emergency roadway repair) — which serve as foundations rather than direct competitors. The most relevant core classic remains idx 96 (Restoration of Bridge Networks after an Earthquake). Frontier does not equal classic: the competitors closest to this study are mostly from 2025 to 2026 with only 0 to 1 citations — for example idx 911, Robust Multicrew Scheduling and Routing in road restoration (Comput. & OR); idx 716, Coordinating road recovery and supply distribution (EJOR, DRO); idx 994, Prioritizing Disaster Recovery Under Budget Uncertainty (M&SOM); idx 1059 (constrained reinforcement learning for metro-bus recovery sequencing, TR-D); idx 747 (two-stage stochastic programming for road-network hardening and recovery, RESS); as well as idx 133, 35, 60, and 98 — and must be judged by content, not citations. Classics serve for foundations and method lineage, while frontier papers serve for direct competitive positioning. The core set (v1 relevance tier) contains 19 papers: idx 1, 11, 17, 18, 26, 35, 49, 55, 60, 63, 68, 76, 84, 89, 96, 98, 133, 181, 221.

---

## Appendix C　Review Papers (Listed Separately, Excluded from Screening)

The corpus contains 30 papers marked as reviews (`is_review=True`). Reviews do not participate in the screening; `final_curated` has excluded them from the 190-paper final set, and they serve only as citation references for background or methodological surveys. The vast majority (24 papers) are query-noise reviews (medicine, ecology, etc.) already removed in Stage 1. Six reviews are relevant to this topic:

| idx | Year | Note | Title |
|---|---|---|---|
| 20 | 2026 | Passed Filter B and Filter C but is itself a review; moved out of the final set | Scheduling Repair Resources for Post-Disaster Critical Infrastructure Systems: A Review of Models and Algorithms |
| 401 | 2019 | Removed in Stage 1 | Fragility of transport assets exposed to multiple hazards: State-of-the-art review |
| 234 | 2024 | Removed in Stage 1 | A comprehensive review of resilience of urban metro systems |
| 402 | 2021 | Removed in Stage 1 | Digital Infrastructure Asset Management Tools for Resilient Linear Infrastructure: Systematic Literature Review |
| 169 | 2018 | Removed in Stage 1 | Bridge Adaptation and Management under Climate Change Uncertainties: A Review |
| 400 | 2025 | Removed in Stage 1 | Simulation-Based Models for Postearthquake Response: A Survey and Research Directions |

---

## Appendix D　Papers Whose Full Text Could Not Be Obtained (6 papers)

Of the 190 papers, 184 have been read in full; 6 could not be obtained due to paywalls or anti-scraping measures (neither the NYU nor the GT account has a subscription). The six-dimension labels of these 6 papers are best abstract-level judgments; the retrieval status is recorded in the CSV column `ref_access` (paywalled or failed) and can be updated if the full texts are obtained later.

| idx | Status | Year | Venue | Title | DOI |
|---|---|---|---|---|---|
| 42 | ❌ Unobtainable (paywalled) | 2011 | International Journal of Information Technology Project Management | A Single-Objective Recovery Phase Model | 10.4018/jitpm.2011070104 |
| 120 | ❌ Unobtainable (paywalled) | 2023 | Advances in Aircraft and Spacecraft Science | Aircraft delivery vehicle with fuzzy time window for improving search algorithm | 10.12989/aas.2023.10.5.393 |
| 147 | ❌ Unobtainable (anti-scraping) | 2015 | International Journal of Emergency Management | Response and recovery after severe wind storms using hierarchical open vehicle routing | 10.1504/IJEM.2015.069515 |
| 325 | ❌ Unobtainable (paywalled) | 2021 | Quarterly Journal of Engineering Geology and Hydrogeology | Landslide costs on the national road network of Laos | 10.1144/qjegh2021-023 |
| 741 | ❌ Unobtainable (paywalled) | 2014 | International Journal of Critical Infrastructures | Optimal recovery sequencing for enhanced resilience and service restoration in transportation networks | 10.1504/IJCIS.2014.066356 |
| 994 | ❌ Unobtainable (paywalled) | 2026 | Manufacturing & Service Operations Management | Prioritizing Disaster Recovery Under Budget Uncertainty | 10.1287/msom.2024.1574 |
