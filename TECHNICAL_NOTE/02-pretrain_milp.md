# 02 — `run_pretrain_milp()` complete execution logic

**Purpose.** The Section 2.1.1 *pretraining* solver: for each duration scenario it iteratively
(1) fixes whole‑horizon travel times from the previous UE run, (2) precomputes analytic F1‑sensitivity
coefficients `c_e^k` (no UE), (3) solves a small start‑time MILP (HiGHS), (4) re‑runs the full UE
pipeline to refresh travel times, and repeats until the schedule stops changing. Each scenario's
best schedule (by the *true* objective `F`) is checkpointed, and — strictly afterwards — compared
against a pre‑computed brute‑force oracle.

**Entry point.** `util/pretrain_milp.py :: run_pretrain_milp(toy_dir=TOY, out_dir=OUT, M=P.M_SCENARIOS, seed=P.SEED)` — defined at `util/pretrain_milp.py:184`.

Module‑level constants (`util/pretrain_milp.py:44-46`):
- `ROOT` = project root (`…/Research_RoadRestoration`).
- `TOY` = `ROOT/data/siouxfalls_toy`.
- `OUT` = `ROOT/outputs/pretrain_milp`.

> **No‑leakage note (see §12).** The oracle solution is read only at the very end, *after* every
> MILP schedule has already been computed and checkpointed. It never enters the MILP algorithm. The
> only thing shared upstream with the oracle is the *problem definition* (same instance, same
> scenarios, same horizon, same objective), which is intentional so the two are comparable.

---

## 1. Call‑tree overview (whole flow at a glance)

```
run_pretrain_milp()                                   util/pretrain_milp.py:184
├─ scale_dir(out_dir)                                 util/oracle.py:57      → outputs/pretrain_milp/n{N}/
├─ (out_dir/"figures").mkdir(...)
├─ select_oracle_instance(toy_dir, N, seed)           util/oracle.py:83
│   ├─ pd.read_csv(edges.csv)
│   └─ _reference_twoway_flow(toy)                    util/oracle.py:63      ← raw/SiouxFalls_flow.tntp
├─ segments = sorted(int(e) for e in disrupted.edge_id)
├─ build_context(toy_dir, disrupted)                  util/evaluate.py:106
│   ├─ load_toy_network(toy_dir)                      util/io.py:10         ← edges/od_pairs/nodes.csv
│   ├─ solve_ue(undamaged, H0, …)  [baseline UE]      util/ue.py:81
│   │   ├─ _build_graph(edges, zone_ids)              util/ue.py:46         (AequilibraE Graph)
│   │   └─ _build_matrix(M, zone_ids)                 util/ue.py:71         (AequilibraE Matrix)
│   ├─ _matrix_from_H(H0, ctx)                        util/evaluate.py:48
│   ├─ od_travel_times(base_links, ctx)               util/evaluate.py:27   (networkx Dijkstra)
│   └─ B(Φ) build via networkx shortest_path (free-flow)
├─ sample_scenarios(disrupted, M, seed)               util/scenarios.py:14  (DURATION_SUPPORT × ETA)
├─ compute_horizon(segments, scenarios)               util/oracle.py:109
│   └─ for every perm × scenario:
│       ├─ schedule_from_permutation(perm, dur)       util/evaluate.py:82
│       └─ makespan_slot(start, dur)                  util/evaluate.py:94
├─ _param_fingerprint()                               util/oracle.py:44      + hashlib SHA1 extend (damp/maxit/cyc)
├─ RESUME: read milp_optima.csv + milp_trace.csv + milp_progress.json → build `done` set
│
├─ for m, dur in enumerate(scenarios):  (skip if m in done)
│   └─ alternating_optimize(ctx, dur, segments, T)    util/pretrain_milp.py:132
│       ├─ schedule_from_permutation(segments, dur)   util/evaluate.py:82   (greedy-first init)
│       ├─ evaluate_schedule(start, …, return_u=True) util/evaluate.py:156  [1 UE per slot → u_tilde (T×|R|)]
│       │   ├─ f2_value / makespan_slot               util/evaluate.py:98/94
│       │   ├─ build_damaged_edges(ctx, damaged)      util/evaluate.py:54
│       │   ├─ _matrix_from_H(H, ctx)                 util/evaluate.py:48
│       │   ├─ solve_ue(dmg_edges, H, …)              util/ue.py:81
│       │   └─ od_travel_times(links, ctx)            util/evaluate.py:27
│       └─ loop up to MILP_MAX_ITER:
│           ├─ precompute_c(ctx, u_tilde, dur, …)     util/pretrain_milp.py:52   (analytic c_e^k, NO UE)
│           ├─ build_and_solve_milp(c, dur, …)        util/pretrain_milp.py:83   (scipy.optimize.milp / HiGHS)
│           ├─ evaluate_schedule(new_start, …)        util/evaluate.py:156       [refresh UE]
│           ├─ _surrogate_value(new_start, c, …)      util/pretrain_milp.py:124
│           ├─ STOP tests: fixed-point / cycle-guard / max-iter
│           └─ MSA damped update: u_tilde = damp·u_new + (1-damp)·u_tilde
│       └─ best = argmin_history true F
│   └─ checkpoint: milp_optima.csv, milp_trace.csv, milp_progress.json
│
├─ run_meta.json  (timing / total_ue_solves / mean_iters / mean_F_milp)
├─ make_process_figures(...)                          viz/pretrain_viz.py:111  (figs 03/04)
└─ POST-HOC ORACLE COMPARE (only if oracle_optima.csv exists):
    ├─ read outputs/oracle/n{N}/oracle_optima.csv + oracle_landscape.csv
    ├─ merge on scenario → gap = F_milp − F_oracle → milp_vs_oracle.csv
    ├─ make_comparison(...)                           viz/pretrain_viz.py:21   (fig 01)
    └─ make_landscape(...)                            viz/pretrain_viz.py:62   (fig 02)
```

---

## 2. Shared setup — output folder (`scale_dir`)

**Step (line `util/pretrain_milp.py:185-186`).**
```python
out_dir = scale_dir(out_dir)                     # outputs/pretrain_milp/n{N}/
(out_dir / "figures").mkdir(parents=True, exist_ok=True)
```

- `scale_dir(base=OUT, n=None)` — `util/oracle.py:57-60`. Returns `Path(base)/f"n{n}"` where
  `n = P.N_DISRUPTED_ORACLE` when `n is None`. So `out_dir → outputs/pretrain_milp/n4/` (with the
  default `N_DISRUPTED_ORACLE = 4`). This mirrors the oracle's per‑scale folder scheme so a MILP run
  never overwrites a different problem size.
- **Inputs → outputs:** `OUT` path → scale subfolder Path; then creates `…/n4/figures/`.
- **Files touched:** creates directory `outputs/pretrain_milp/n4/figures/`.
- **Config used:** `N_DISRUPTED_ORACLE`.

---

## 3. Shared setup — instance selection (`select_oracle_instance` + `_reference_twoway_flow`)

**Step (line `util/pretrain_milp.py:188-189`).**
```python
disrupted = select_oracle_instance(toy_dir, P.N_DISRUPTED_ORACLE, seed)
segments = sorted(int(e) for e in disrupted["edge_id"])
```

### 3.1 `select_oracle_instance(toy_dir, n, seed)` — `util/oracle.py:83-106`
Chooses `n` disrupted segments by *importance* (baseline two‑way UE flow), deliberately mixing
critical and minor links so restoration order matters. Deterministic (uses no RNG despite the `seed`
argument — `seed` is passed only for signature symmetry).

Internal logic:
1. `edges = pd.read_csv(toy/"network"/"edges.csv")` (`oracle.py:89`).
2. `flow = _reference_twoway_flow(toy)` (`oracle.py:90`) — see §3.2.
3. Attach a per‑edge `flow` column by looking up each edge's undirected `(min(u,v),max(u,v))` key
   (`oracle.py:91-92`), defaulting to `0.0` if absent.
4. `ranked = edges.sort_values("flow", ascending=False)` (`oracle.py:93`).
5. `n_crit = min(2, n)`; `picks = list(range(n_crit))` — take the top‑`n_crit` (≤2) flow edges as
   "critical" (`oracle.py:95-96`).
6. `rest = n - n_crit`; if `rest > 0` spread the remainder over lower‑flow edges using
   `np.linspace(len(ranked)//5, len(ranked)-1, rest)` rounded to ints (`oracle.py:97-99`).
7. `sub = ranked.iloc[picks]` (`oracle.py:100`).
8. Assign severities (`oracle.py:101`): the `n_crit` critical edges get severity **3** (severed —
   fully removed from the UE network, see §7.1); the rest alternate **2 / 1** by parity
   (`2 if i%2==0 else 1`).
9. Build `level_id = road_class + "-S" + severity` (`oracle.py:102`).
10. `out` = the sub‑frame `[edge_id, u, v, road_class, severity, level_id]` sorted by `edge_id`
    (`oracle.py:103-104`).
11. **Writes** `toy/"disruption"/f"disrupted_segments_oracle{n}.csv"` (`oracle.py:105`) and returns
    `out`.

- **Inputs → outputs:** toy dir, `n=4` → DataFrame of 4 disrupted segments (edge_id/u/v/road_class/severity/level_id).
- **Files touched:** reads `data/siouxfalls_toy/network/edges.csv`; reads
  `data/siouxfalls_toy/raw/SiouxFalls_flow.tntp` (via §3.2); **writes**
  `data/siouxfalls_toy/disruption/disrupted_segments_oracle4.csv`.
- **Config used:** `N_DISRUPTED_ORACLE` (passed as `n`).

> This is the *same* selection function the oracle calls (`util/oracle.py:123`), guaranteeing MILP
> and oracle operate on an identical instance — an intended shared problem definition, not leakage.

### 3.2 `_reference_twoway_flow(toy)` — `util/oracle.py:63-80`
Parses the open‑source Sioux Falls equilibrium flow file to rank links by importance.
- Reads `toy/"raw"/"SiouxFalls_flow.tntp"` line by line (`oracle.py:67`).
- Skips blank lines and the `From`‑header line (`oracle.py:69-70`).
- For each data line splits tokens, drops trailing `;`, needs ≥3 tokens (`oracle.py:71-73`).
- Parses `a, b = ints`, `vol = float(t[2])` (column 3 = link volume) with a `ValueError` guard
  (`oracle.py:74-77`).
- Aggregates both directed volumes into an **undirected** key `(min(a,b), max(a,b))` (`oracle.py:78-79`).
- Returns `{(min,max): summed_flow}`.
- **Inputs → outputs:** flow file → dict of undirected‑edge → two‑way flow.

`segments = sorted(int(e) for e in disrupted["edge_id"])` (`pretrain_milp.py:189`) — the sorted list
of disrupted edge ids used as the canonical column order everywhere downstream (must align with
`ctx["B"]` columns and `durations` keys).

---

## 4. Shared setup — static context (`build_context`)

**Step (line `util/pretrain_milp.py:190`).** `ctx = build_context(toy_dir, disrupted)`.

### 4.1 `build_context(toy_dir, disrupted)` — `util/evaluate.py:106-150`
Builds the once‑per‑run static problem context: network, OD pairs, baseline travel times
`baseline_u`, penalty `u_pen`, demand‑shortfall matrix `B(Φ)`, and `severity_vec`.

Step‑by‑step:
1. `edges, od, zone_ids = load_toy_network(toy_dir)` (`evaluate.py:108`) — see §4.2.
2. `zone_pos = {node_id: index}` (`evaluate.py:109`).
3. `od_pairs = [(origin, destination) …]` from the OD frame (`evaluate.py:110`).
4. `H0 = od["h0"].to_numpy()` — baseline (normal‑time) OD demand vector (`evaluate.py:111`).
5. `oi`, `di` — integer positions of each OD's origin/destination in the zone list
   (`evaluate.py:112-113`); used by `_matrix_from_H` to scatter a demand vector back into a matrix.
6. `edge_row = {edge_id: row_index}` and `eid_of = {(min,max): edge_id}` (`evaluate.py:114-116`).
7. Assemble `ctx` dict with `edges, zone_ids, od_pairs, H0, oi, di, nz=len(zone_ids), edge_row,
   origins_unique` (`evaluate.py:118-120`).

8. **Baseline UE** `u_r^{t0}` — one UE solve on the **undamaged** network with **normal** demand
   `H0` (`evaluate.py:123-124`):
   ```python
   base_links, _ = solve_ue(edges, _matrix_from_H(H0, ctx), zone_ids,
                            rgap=P.UE_RGAP, max_iter=P.UE_MAX_ITER, quiet=True)
   ctx["baseline_u"] = od_travel_times(base_links, ctx)
   ```
   - `_matrix_from_H(H0, ctx)` — see §4.5; `solve_ue` — see §4.3; `od_travel_times` — see §4.4.
9. **Penalty time** `u_pen = UPEN_FACTOR × max(finite baseline_u)` (`evaluate.py:126`) — the travel
   time charged to a disconnected OD pair.

10. **Disrupted tuples** `dis = [(edge_id, u, v, severity), …]` stored as `ctx["disrupted"]`
    (`evaluate.py:129-131`). This drives the per‑slot damage state in `evaluate_schedule`.

11. **B(Φ) matrix build** (`evaluate.py:132-148`):
    - Build a **free‑flow, bidirectional** networkx `DiGraph` `Gff` weighted by `free_flow_time`
      (`evaluate.py:132-135`).
    - `col = {edge_id: column j}` maps each disrupted edge to a B column (`evaluate.py:137`).
    - For every OD pair `(o,d)`: find the free‑flow shortest path (`nx.shortest_path`,
      `evaluate.py:139-142`, `NetworkXNoPath` → skip). Collect the set of undirected edges on that
      path (`evaluate.py:143`). For each such edge that is a disrupted segment, set
      `B[i, col[eid]] = KAPPA · (H0[i] / 3.0)` (`evaluate.py:144-147`).
    - So **B[r,e]** = `κ·(h_r0/3)` if disrupted `e` lies on OD `r`'s free‑flow shortest path, else 0.
    - `ctx["B"] = B` (`evaluate.py:148`).
12. `ctx["severity_vec"] = np.array([sev for (_,_,_,sev) in dis])` (`evaluate.py:149`) — the segment
    severities in `dis` order (aligned to B columns).
13. Return `ctx`.

- **Inputs → outputs:** toy dir + disrupted frame → `ctx` dict with keys
  `edges, zone_ids, od_pairs, H0, oi, di, nz, edge_row, origins_unique, baseline_u, u_pen, disrupted,
  B, severity_vec`.
- **Files touched:** reads `network/edges.csv`, `network/od_pairs.csv`, `network/nodes.csv`
  (via §4.2). One AequilibraE UE solve (in memory only).
- **Config used:** `UE_RGAP`, `UE_MAX_ITER`, `UPEN_FACTOR`, `KAPPA`.

### 4.2 `load_toy_network(toy_dir)` — `util/io.py:10-23`
- Reads `network/edges.csv`, `network/od_pairs.csv`, `network/nodes.csv` (`io.py:19-21`).
- `zone_ids = nodes["node_id"].to_numpy(int64)` — every node is an OD zone in Sioux Falls
  (`io.py:22`).
- Returns `(edges_df, od_df, zone_ids)`. `edges_df` cols: `edge_id,u,v,capacity,length,
  free_flow_time,bpr_alpha,bpr_beta,road_class`. `od_df` cols: `od_id,origin,destination,h0`.

### 4.3 `solve_ue(edges, od_matrix, zone_ids, algorithm="bfw", max_iter=1000, rgap=1e-10, quiet=False)` — `util/ue.py:81-121`
Static user‑equilibrium (Wardrop / Beckmann) assignment via AequilibraE's bi‑conjugate Frank‑Wolfe.
This is the numerical heavy step called once per slot inside `evaluate_schedule` and once for the
baseline in `build_context`.

Sub‑logic:
1. `graph, net = _build_graph(edges, zone_ids)` — see §4.3.1.
2. `mat = _build_matrix(od_matrix, zone_ids)` — see §4.3.2.
3. Wrap in `TrafficClass("car", graph, mat)`; create `TrafficAssignment` (`ue.py:92-93`).
4. `assig.set_vdf("BPR")` and `set_vdf_parameters({"alpha":"b","beta":"power"})` — BPR congestion
   cost `t_a(x) = t0·(1 + α·(x/c)^β)` (`ue.py:95-96`).
5. `set_capacity_field("capacity")`, `set_time_field("free_flow_time")`, `set_algorithm("bfw")`
   (`ue.py:97-99`).
6. `assig.max_iter = max_iter`; `assig.rgap_target = rgap` (`ue.py:100-101`).
7. If `quiet`: raise AequilibraE log level to CRITICAL and redirect stdout/stderr to `os.devnull`
   around `assig.execute()`; else run plainly (`ue.py:102-108`). `execute()` runs the FW loop:
   all‑or‑nothing → line search → cost update, until the relative gap < `rgap`.
8. `res = assig.results()` — per‑link AB/BA volume + congested time (`ue.py:110`).
9. Re‑emit as tidy directed rows: for each link id, one `{from:u,to:v,volume:demand_ab,
   cost:Congested_Time_AB}` and one reversed `{from:v,to:u,volume:demand_ba,cost:Congested_Time_BA}`
   (`ue.py:113-118`).
10. Drop `NaN`‑volume rows; return `(flows_df, assig)` — `flows_df` cols `from,to,volume,cost`
    (`ue.py:119-121`).

- **Config used (by callers):** always invoked with `rgap=UE_RGAP (1e-6)`, `max_iter=UE_MAX_ITER
  (100)`, `quiet=True`.

#### 4.3.1 `_build_graph(edges, zone_ids)` — `util/ue.py:46-68`
Builds a routable AequilibraE `Graph` from the undirected edge table: one bidirectional link
(`direction=0`) per edge, symmetric AB/BA capacity, `free_flow_time`, BPR `b`(=α) / `power`(=β).
`prepare_graph(zone_ids)` marks every node a zone; `set_graph("free_flow_time")`;
`set_blocked_centroid_flows(False)` allows through‑traffic. Returns `(graph, net)`.

#### 4.3.2 `_build_matrix(M, zone_ids)` — `util/ue.py:71-78`
Wraps a dense OD matrix `M` as a memory‑only `AequilibraeMatrix` with one core `"demand"`, indexes
by `zone_ids`, sets the computational view. Returns the matrix.

> `beckmann_objective(...)` (`ue.py:124-134`) and `_validate()` (`ue.py:141-195`) exist in the module
> but are **not** on the pretraining path (validation/self‑check only).

### 4.4 `od_travel_times(link_df, ctx)` — `util/evaluate.py:27-45`
Converts congested directed link times into per‑OD travel times via shortest paths.
- Build a networkx `DiGraph` from `link_df[from,to,cost]` weighting each edge by its congested
  `cost` (`evaluate.py:31-35`).
- For each unique origin, run `single_source_dijkstra_path_length` (empty dict if origin not in the
  graph — happens when its incident edges were all severed) (`evaluate.py:37-42`).
- Assemble `u` aligned to `ctx["od_pairs"]`; entries default to `np.inf` when O and D are
  disconnected (`evaluate.py:36, 43-44`).
- Returns `u` (length `|R|`, `np.inf` for disconnected pairs).

### 4.5 `_matrix_from_H(H, ctx)` — `util/evaluate.py:48-51`
Scatters a demand *vector* `H` (aligned to `od_pairs`) back into a dense `nz×nz` matrix using the
precomputed `oi`/`di` positions: `M[oi, di] = H`. Returns `M`. This is the inverse of the OD‑pair
flattening and is what `solve_ue` consumes.

---

## 5. Shared setup — scenario sampling (`sample_scenarios`)

**Step (line `util/pretrain_milp.py:191`).** `scenarios = sample_scenarios(disrupted, M, seed)`.

### 5.1 `sample_scenarios(disrupted, M, seed)` — `util/scenarios.py:14-28`
Draws `M` restoration‑duration scenarios; durations are sampled **per level** `(road_class,
severity)`, and all disrupted segments of the same level share one duration in a scenario.
- `rng = np.random.default_rng(seed)` — deterministic for fixed `seed=42` (`scenarios.py:17`).
- `levels = sorted({(road_class, severity)})` over the disrupted rows (`scenarios.py:19`).
- For each of `M` scenarios, for each level: `base = rng.choice(DURATION_SUPPORT[level])`,
  `eta = rng.choice(ETA)`, `lvl_dur[level] = max(1, round(base·eta))` (`scenarios.py:21-26`).
- Emit `scenario[m] = {edge_id: lvl_dur[(road_class,severity)]}` for every disrupted edge
  (`scenarios.py:27`).
- Returns a list of `M` dicts `{edge_id: duration_slots}`.

- **Inputs → outputs:** disrupted frame, `M=10`, `seed=42` → list of 10 duration dicts.
- **Config used:** `M_SCENARIOS`, `SEED`, `DURATION_SUPPORT` (Table 1 base support sets), `ETA`
  (crew‑efficiency multipliers `[0.8, 1.0, 1.2]`).

---

## 6. Shared setup — horizon (`compute_horizon`)

**Step (line `util/pretrain_milp.py:192-193`).**
```python
T = compute_horizon(segments, scenarios)
print(f"instance: {len(segments)} segments {segments}; M={M}; horizon T={T}")
```

### 6.1 `compute_horizon(segments, scenarios)` — `util/oracle.py:109-116`
Global horizon `T` = the maximum completion slot over **all permutations × all scenarios**, so every
work‑conserving schedule finishes within `T` and all share one comparable F1 horizon.
- Nested loop over `itertools.permutations(segments)` and `scenarios` (`oracle.py:113-114`).
- For each: `T = max(T, makespan_slot(schedule_from_permutation(list(perm), dur), dur))`
  (`oracle.py:115`).
- Returns the max `T` (an int).

#### 6.1.1 `schedule_from_permutation(perm, durations, c_max=P.C_MAX)` — `util/evaluate.py:82-91`
Work‑conserving list scheduling with `c_max` identical crews, earliest start slot = 1.
- `crew_free = [1]*c_max` (each crew's next free slot).
- For each edge `e` in priority order: pick the earliest‑free crew `c = argmin(crew_free)`; set
  `start[e] = crew_free[c]`; advance `crew_free[c] = start[e] + durations[e]` (busy until completion).
- Returns `{edge_id: start_slot}`. No idling → the schedule is fully determined by the permutation.
- **Config used:** `C_MAX`.

#### 6.1.2 `makespan_slot(start, durations)` — `util/evaluate.py:94-95`
`max_e (start[e] + durations[e])` — the completion slot of the last segment.

- **Config used (this step):** `C_MAX` (via `schedule_from_permutation`); `segments`/`scenarios`
  from prior steps.

> For the default instance `N=4` (`4! = 24` permutations) × `M=10` scenarios, this is 240 cheap
> combinatorial evaluations (no UE).

---

## 7. Full trace of `evaluate_schedule(...)` — the true objective `F(x|ω)`

`evaluate_schedule` is the exact Figure‑1 evaluator, reused unchanged from the oracle. In the
pretraining loop it is always called with `return_u=True` so it also returns the `(T×|R|)` travel‑time
matrix `u_tilde` used by `precompute_c`.

### 7.1 `evaluate_schedule(start, durations, T, ctx, collect_traces=False, return_u=False)` — `util/evaluate.py:156-207`
Unpacks `dis = ctx["disrupted"]`, `H0`, `B`, `base_u = ctx["baseline_u"]`, `sev` (`evaluate.py:159-162`).

**Step 2 — F2 (no UE):** `F2 = f2_value(start, durations)` (`evaluate.py:165`).
- `f2_value(start, durations)` — `util/evaluate.py:98-101`: `makespan_slot(start,durations) /
  Σ_e durations[e]` (the `Δt` factors cancel). F2 is a logging‑only value; with `MU=1` it does not
  enter `F`.

**Step 3 — per‑slot F1 loop** (`evaluate.py:168-193`), iterating `k = 1 … T`:
1. `D = np.zeros(|H0|)` initialized before the loop (`evaluate.py:168`).
2. **Damage state (Eq. 2):** `damaged = {eid: s for (eid,_,_,s) in dis if k < start[eid] +
   durations[eid]}` — a segment is damaged while `k` is before its completion slot
   (`evaluate.py:172`). `v_vec = [s if damaged else 0]` per disrupted segment (`evaluate.py:173`).
3. **Demand shortfall → `H`:** `target = B @ v_vec`; `D = max(target, RHO·D)` (element‑wise);
   `H = clip(H0 - D, 0, None)` (`evaluate.py:175-177`). This is the shortfall model: a sharp onset
   drop to the current damage‑driven shortfall `B·v`, then recovery at rate `RHO` as roads heal.
4. **Damaged network:** `dmg_edges = build_damaged_edges(ctx, damaged)` (`evaluate.py:179`) — §7.2.
5. **UE → OD travel times:** `links,_ = solve_ue(dmg_edges, _matrix_from_H(H,ctx), zone_ids,
   rgap=UE_RGAP, max_iter=UE_MAX_ITER, quiet=True)` (`evaluate.py:181-182`); then
   `u = od_travel_times(links, ctx)` (`evaluate.py:183`). This is the **one UE solve per slot**.
6. **Penalty substitution:** `u_tilde = where(isfinite(u), u, u_pen)` — disconnected OD pairs get
   `u_pen` (`evaluate.py:184`). Row appended to `u_rows` (`evaluate.py:185`).
7. **F1 term:** demand‑weighted realized/baseline travel‑time ratio: `den = Σ(H·base_u)`;
   `term = Σ(H·u_tilde)/den` if `den>0` else `1.0` (`evaluate.py:187-189`). Appended to `terms`;
   `active` records whether any segment is still damaged (`evaluate.py:190-191`).
8. `traces` optionally collected (`collect_traces` — unused on the MILP path) (`evaluate.py:192-193`).

**Aggregation (`evaluate.py:195-206`):**
- `terms = np.asarray(terms)` (`evaluate.py:195`).
- If `F1_ACTIVE_ONLY` (default **False**): `F1 = terms.mean()` over the full `[1,T]` horizon
  (`evaluate.py:199-200`); else it would average only active slots. Full‑horizon averaging matches
  the `1/T` normalization used by `precompute_c`.
- `F = MU·F1 + (1-MU)·F2` (`evaluate.py:201`). With `MU=1`, `F ≡ F1`.
- `out = {F, F1, F2}`; if `return_u`: `out["u_tilde"] = np.asarray(u_rows)` — the **(T×|R|)** fixed
  travel‑time matrix (`evaluate.py:202-204`).
- Returns `out`.

- **Inputs → outputs:** one schedule + one scenario + horizon `T` + `ctx` → `{F, F1, F2, u_tilde}`.
- **UE solves:** exactly `T` per call.
- **Config used:** `MU`, `F1_ACTIVE_ONLY`, `RHO`, `UE_RGAP`, `UE_MAX_ITER` (and `u_pen`/`B` from `ctx`).

### 7.2 `build_damaged_edges(ctx, damaged)` — `util/evaluate.py:54-76`
Produces the current‑slot damaged edge table.
- Copy `ctx["edges"]` (`evaluate.py:59`).
- If `damaged`: copy the `capacity`/`free_flow_time` arrays; for each `(eid, sev)`:
  - if `sev >= SEVER_SEVERITY` (default 3): mark the row for **removal** (true disconnection)
    (`evaluate.py:67-68`);
  - else: `cap[j] *= CAP_RETAIN[sev]` and `fft[j] /= SPEED_RETAIN[sev]` (degraded but present)
    (`evaluate.py:70-72`).
  - Write back arrays; drop severed rows and reset the index (`evaluate.py:73-76`).
- Undamaged / completed links are unchanged. Returns an edges DataFrame.
- **Config used:** `SEVER_SEVERITY`, `CAP_RETAIN`, `SPEED_RETAIN`.

---

## 8. `precompute_c` — analytic F1‑sensitivity coefficients `c_e^k` (NO UE)

**Called at** `util/pretrain_milp.py:157` inside `alternating_optimize`.

### 8.1 `precompute_c(ctx, u_by_slot, durations, segments, T)` — `util/pretrain_milp.py:52-77`
Computes the coefficient matrix `c[j, k-1]` = the analytic F1 improvement of starting `segments[j]`
at slot `k`, given the **fixed** per‑slot travel times `u_by_slot` (shape `(T, |R|)`) from the
previous UE iteration. No UE is solved — pure numpy.

The closed form (Problem‑2 shortfall fix, `pretrain_milp.py:56-58`):
```
c_e^k = (1/T) · Σ_{k'=k+d_e}^{T} (1 − ρ^{k'−k−d_e+1}) · Σ_r B[r,e]·v_e*·α_r^{k'}
α_r^{k'} = 1 − baseline_u[r] / u_by_slot[k'−1, r]
```

Implementation:
1. Unpack `B = ctx["B"]`, `sev = ctx["severity_vec"]`, `base_u = ctx["baseline_u"]`, `rho = P.RHO`
   (`pretrain_milp.py:60-63`).
2. Assert `B.shape[1] == len(segments)` — B columns must align with the segment order
   (`pretrain_milp.py:64`).
3. **α matrix:** `alpha = 1.0 - base_u[None,:] / u_by_slot` → shape `(T, |R|)` (`pretrain_milp.py:66`).
   `α_r^{k'}` = fractional travel‑time excess over baseline; can be **< 0** when a slot's congested
   time is below baseline (legal, e.g. under low recovery‑window demand).
4. `c = zeros((|segments|, T))` (`pretrain_milp.py:67`).
5. For each segment `j` (edge `e`) with duration `d = durations[e]` (`pretrain_milp.py:68-69`):
   - `Bv = B[:, j] * sev[j]` — demand this segment restores at full recovery (a `|R|` vector)
     (`pretrain_milp.py:70`). (Here `sev[j]` = `v_e*`, the segment's severity used as the recovered
     demand scale.)
   - `w = alpha @ Bv` — a `(T,)` vector, `w[k'-1] = Σ_r α_r^{k'}·B[r,e]·v*` (`pretrain_milp.py:71`).
   - For each **feasible** start `k` in `1 … T-d` (completion `k+d ≤ T`) (`pretrain_milp.py:72`):
     - `kp = arange(k+d, T+1)` — completion slot … horizon end (`pretrain_milp.py:73`).
     - `decay = 1.0 - rho**(kp - k - d + 1)` — the growing recovery factor `(1 − ρ^{n+1})`,
       `n = k' − k − d` (`pretrain_milp.py:75`). *This is the Problem‑2 fix: recovered demand GROWS
       over time rather than decaying.*
     - `c[j, k-1] = (decay @ w[kp-1]) / T` (`pretrain_milp.py:76`).
6. Infeasible starts (`k > T-d`) are left at `0` — the MILP bounds (§9) force those `y` to 0.
7. Returns `(c, alpha)`; `c` shape `(|E|, T)`, `alpha` shape `(T, |R|)`.

- **Inputs → outputs:** `ctx` (B, severity, baseline_u), fixed `u_by_slot (T×|R|)`, durations,
  segments, `T` → coefficient matrix `c (|E|×T)` and `alpha (T×|R|)`.
- **UE solves:** none.
- **Config used:** `RHO`.

---

## 9. `build_and_solve_milp` — the start‑time MILP (scipy / HiGHS)

**Called at** `util/pretrain_milp.py:158`.

### 9.1 `build_and_solve_milp(c, durations, segments, T, c_max=P.C_MAX)` — `util/pretrain_milp.py:83-121`
Solves `min_y −Σ_{e,k} c[j,k]·y[j,k]` (= maximize the F1 surrogate) subject to start‑once, crew‑cap,
and horizon‑bound constraints, `y` binary.

Variable layout: `E = len(segments)`, `n = E·T` decision vars; index helper `idx(j,k) = j·T + (k-1)`
for 1‑based slot `k` (`pretrain_milp.py:85-90`).

1. **Objective:** `obj = -c.reshape(-1)` — flatten and negate so HiGHS's *minimization* maximizes the
   surrogate (`pretrain_milp.py:92`).
2. **Horizon bound (via Bounds):** `ub = ones(n)`, then for each segment set `ub[idx(j,k)] = 0` for
   `k in T-d+1 … T` — forbidding any start that would overrun the horizon; this is the one real
   feasibility guard (`pretrain_milp.py:94-99`). `bounds = Bounds(zeros(n), ub)` (`pretrain_milp.py:99`).
3. **Start‑once equality:** `A_eq (E×n)` with `A_eq[j, idx(j,k)] = 1` for all `k`; `con_eq =
   LinearConstraint(A_eq, 1, 1)` → `Σ_k y[j,k] = 1` (each segment starts exactly once)
   (`pretrain_milp.py:101-105`).
4. **Crew cap:** `A_ub (T×n)`; for each slot `k` and segment `j` with duration `d`, mark all starts
   `kp in max(1,k-d+1) … k` (i.e. segments still in repair at slot `k`) (`pretrain_milp.py:107-112`);
   `con_ub = LinearConstraint(A_ub, -inf, c_max)` → at most `c_max` active repairs per slot
   (`pretrain_milp.py:113`).
5. **Solve:** `res = milp(c=obj, constraints=[con_eq, con_ub], integrality=ones(n), bounds=bounds)`
   — all vars integer (binary via bounds) (`pretrain_milp.py:115-116`). On failure raise
   `RuntimeError(res.message)` (`pretrain_milp.py:117-118`).
6. **Decode:** reshape `res.x` to `(E, T)`; return `{edge_id: argmax_k y[j] + 1}` — the chosen 1‑based
   start slot per segment (`pretrain_milp.py:120-121`).

- **Inputs → outputs:** `c (|E|×T)`, durations, segments, `T` → `{edge_id: start_slot}` dict.
- **Solver:** `scipy.optimize.milp` (HiGHS branch‑and‑bound); no external license.
- **Config used:** `C_MAX` (crew cap).

### 9.2 `_surrogate_value(start, c, segments, T)` — `util/pretrain_milp.py:124-126`
Sign‑flipped MILP objective for a given schedule: `Σ_j c[j, start[e]-1]` over segments whose start is
in `[1,T]`. Used both to log the per‑iteration surrogate in the trace and (in `level_a`) as the
brute‑force objective. Returns a float.

---

## 10. `alternating_optimize` — the fix‑MILP‑refresh loop (Steps 1‑4)

**Called at** `util/pretrain_milp.py:214` (once per scenario).

### 10.1 `alternating_optimize(ctx, durations, segments, T, damping=None)` — `util/pretrain_milp.py:132-178`
Iterates: fix travel times → precompute `c` → solve MILP → refresh UE, until the schedule stops
changing (or a guard fires). Returns `(best_start, best_result, n_iter, converged, trace)`, where the
best iterate is chosen by the **true** `F` over the whole history (so non‑convergence is harmless).

Setup:
- `damping = P.MILP_DAMPING if damping is None else damping` (default 0.5) (`pretrain_milp.py:138`).
- `t0 = perf_counter()` (`pretrain_milp.py:139`).
- Local `_row(it, res, surr, start)` builds a trace dict `{iter, F, F1, F2, surrogate, elapsed_s,
  start_<e>…}` (`pretrain_milp.py:141-145`).

Initialization (`pretrain_milp.py:147-154`):
1. **Greedy‑first init:** `start = schedule_from_permutation(list(segments), durations)` — the
   identity‑order work‑conserving schedule (§6.1.1).
2. `res = evaluate_schedule(start, …, return_u=True)` — true `F` + `u_tilde` (§7).
3. `history = [(dict(start), res)]`; `trace = [_row(0, res, nan, start)]` (iter 0 has no surrogate).
4. `seen = {frozenset(start.items()): 1}` — occurrence counter for the relaxed cycle guard.
5. `u_tilde = res["u_tilde"]`; `converged=False`; `n_iter=0`.

Main loop `for _ in range(P.MILP_MAX_ITER)` (`pretrain_milp.py:155-172`):
1. `n_iter += 1` (`pretrain_milp.py:156`).
2. `c, _ = precompute_c(ctx, u_tilde, durations, segments, T)` — §8, using the *current fixed*
   travel times (`pretrain_milp.py:157`).
3. `new_start = build_and_solve_milp(c, durations, segments, T)` — §9 (`pretrain_milp.py:158`).
4. `new_res = evaluate_schedule(new_start, …, return_u=True)` — **refresh UE** for the new schedule
   (`pretrain_milp.py:159`).
5. Append `(dict(new_start), new_res)` to `history`; append a trace row with surrogate
   `_surrogate_value(new_start, c, …)` (`pretrain_milp.py:160-161`).
6. **Stop test ① — fixed point:** if `new_start == start` → `converged=True; break`
   (`pretrain_milp.py:162-164`).
7. **Stop test ② — relaxed cycle guard:** `h = frozenset(new_start.items())`; `seen[h] += 1`; if
   `seen[h] >= P.MILP_CYCLE_TOL` (default 3) → `break` (schedule recurred enough times)
   (`pretrain_milp.py:165-168`).
8. **MSA damped travel‑time update:** `u_tilde = damping·new_res["u_tilde"] + (1-damping)·u_tilde`
   — blend toward the new UE so `c_e^k` (and thus the MILP solution) shifts gradually across
   iterations, damping oscillation (`pretrain_milp.py:171`).
9. `start = new_start` (`pretrain_milp.py:172`).
10. **Stop test ③ — max‑iter cap:** the `for` range itself caps at `P.MILP_MAX_ITER` (default 20).

Selection (`pretrain_milp.py:174-178`):
- `best_idx = argmin([h[1]["F"] for h in history])` — best by **true** `F` over all iterates
  (`pretrain_milp.py:174`).
- Mark `trace[i]["is_best"] = (i == best_idx)` (`pretrain_milp.py:175-176`).
- `best_start, best_res = history[best_idx]` (`pretrain_milp.py:177`).
- Return `(best_start, best_res, n_iter, converged, trace)`.

- **Inputs → outputs:** `ctx`, one scenario `durations`, segments, `T` → best schedule + its result
  (F/F1/F2/u_tilde) + iteration count + convergence flag + per‑iteration trace.
- **UE solves:** `1` (init) + `1` per iteration ⇒ `(n_iter + 1)` calls to `evaluate_schedule`, each
  costing `T` UE solves ⇒ `(n_iter + 1)·T` UE solves per scenario (matches `ue_solves` in §11).
- **Config used:** `MILP_DAMPING`, `MILP_MAX_ITER`, `MILP_CYCLE_TOL`, `C_MAX` (via
  `schedule_from_permutation` and the MILP), `RHO`, `MU`, `F1_ACTIVE_ONLY`, `UE_RGAP`, `UE_MAX_ITER`
  (via `evaluate_schedule`/`precompute_c`).

---

## 11. Resume support, per‑scenario loop, and checkpoint writes

### 11.1 Fingerprint (`_param_fingerprint` + MILP extension)
**Step (line `util/pretrain_milp.py:196-198`).**
```python
from util.oracle import _param_fingerprint
_, base_fp = _param_fingerprint()
fp = hashlib.sha1(f"{base_fp}|damp={P.MILP_DAMPING}|maxit={P.MILP_MAX_ITER}|cyc={P.MILP_CYCLE_TOL}".encode()).hexdigest()
```
- `_param_fingerprint()` — `util/oracle.py:44-54`: reads the `FINGERPRINT_PARAMS` list
  (`oracle.py:37-41` — `N_DISRUPTED_ORACLE, MU, CAP_RETAIN, SPEED_RETAIN, SEVER_SEVERITY,
  F1_ACTIVE_ONLY, RHO, KAPPA, UPEN_FACTOR, DELTA_T_H, C_MAX, M_SCENARIOS, SEED, UE_RGAP, UE_MAX_ITER,
  DURATION_SUPPORT, ETA`), stringifies dict keys (DURATION_SUPPORT has tuple keys) for stable JSON,
  and returns `(values, sha1(json.dumps(..., sort_keys=True)))`.
- The MILP layer **extends** the base fingerprint with the three MILP‑specific params
  (`MILP_DAMPING`, `MILP_MAX_ITER`, `MILP_CYCLE_TOL`) and re‑hashes with SHA1 → `fp`
  (`pretrain_milp.py:198`). So a change to *any* F‑affecting param or MILP loop param invalidates the
  cached run.

### 11.2 Resume logic (build the done‑set)
**Step (line `util/pretrain_milp.py:199-207`).**
- Paths: `opt_path = milp_optima.csv`, `trace_path = milp_trace.csv`, `prog_path =
  milp_progress.json` (`pretrain_milp.py:199-200`).
- `rows, trace_rows, done = [], [], set()` (`pretrain_milp.py:201`).
- **Only if** all three files exist **and** `json.load(milp_progress.json)["hash"] == fp`
  (`pretrain_milp.py:202-203`):
  - `rows = pd.read_csv(milp_optima.csv).to_dict("records")` (`pretrain_milp.py:204`).
  - `trace_rows = pd.read_csv(milp_trace.csv).to_dict("records")` (`pretrain_milp.py:205`).
  - `done = {int(r["scenario"]) for r in rows}` — the set of finished scenarios
    (`pretrain_milp.py:206`).
  - Print a `[resume] k/M …` line (`pretrain_milp.py:207`).
- If the hash mismatches (params changed) the checkpoint is ignored → fresh run.

### 11.3 Per‑scenario loop + checkpoints
**Step (line `util/pretrain_milp.py:209-230`).**
For `m, dur in enumerate(scenarios)`:
1. `if m in done: continue` — skip finished scenarios (`pretrain_milp.py:211-212`).
2. `best_start, best_res, n_iter, converged, trace = alternating_optimize(ctx, dur, segments, T)`
   (`pretrain_milp.py:214`) — §10.
3. `scen_s = perf_counter() - t_s` (`pretrain_milp.py:215`).
4. Build the optima `row` (`pretrain_milp.py:216-220`): `scenario=m, F_milp=best_res["F"], F1, F2,
   n_iter, converged, time_s=scen_s, ue_solves=(n_iter+1)*T,
   durations="-".join(durations in segment order)`, plus `start_<e>` for every segment.
5. Append to `rows`; append each `trace` dict (with `scenario=m` prepended) to `trace_rows`; add `m`
   to `done` (`pretrain_milp.py:221-224`).
6. **Checkpoint after every scenario** (`pretrain_milp.py:225-227`):
   - `pd.DataFrame(rows).to_csv(milp_optima.csv)`,
   - `pd.DataFrame(trace_rows).to_csv(milp_trace.csv)`,
   - `milp_progress.json.write_text(json.dumps({"hash": fp, "done": sorted(done)}))`.
7. Print a per‑scenario summary line (`pretrain_milp.py:228-229`).

- **Files written (checkpointing, every scenario):** `outputs/pretrain_milp/n{N}/milp_optima.csv`,
  `…/milp_trace.csv`, `…/milp_progress.json`.
- **Config used:** `M_SCENARIOS` (loop length via `scenarios`), `T` (for `ue_solves`).

### 11.4 Final aggregation + `run_meta.json`
**Step (line `util/pretrain_milp.py:230-243`).**
- `total_s = perf_counter() - t_all` (`pretrain_milp.py:230`).
- Re‑write `milp_optima.csv` and `milp_trace.csv` from the full `rows`/`trace_rows`
  (`pretrain_milp.py:232-234`).
- `total_ue = int(milp_opt["ue_solves"].sum())` (`pretrain_milp.py:235`).
- `meta = {N, M, T, segments, seed, total_time_s, mean_scenario_time_s (mean of time_s),
  total_ue_solves, s_per_ue = total_s/max(1,total_ue), mean_iters (mean of n_iter),
  mean_F_milp (mean of F_milp)}` (`pretrain_milp.py:236-240`).
- **Writes** `outputs/pretrain_milp/n{N}/run_meta.json` (`pretrain_milp.py:241`).

---

## 12. Process figures + POST‑HOC oracle comparison

### 12.1 Process figures (figs 03/04) — `make_process_figures`
**Step (line `util/pretrain_milp.py:245-246`).**
```python
from viz.pretrain_viz import make_process_figures
make_process_figures(out_dir, pd.DataFrame(trace_rows), milp_opt, segments, T)
```
- `make_process_figures(...)` — `viz/pretrain_viz.py:111-172`. Diagnostics from the per‑iteration
  trace:
  - **03_optimization_process.png** — panel a: true `F` best‑so‑far (`F.cummin()`) vs iteration, one
    line per scenario, with the returned best iterate marked (min true F); panel b: MILP surrogate
    (`Σ c y`) vs iteration (`pretrain_viz.py:124-149`).
  - **04_runtime.png** — panel a: per‑scenario wall‑clock (`time_s`); panel b: iteration count
    (`n_iter`) (`pretrain_viz.py:152-171`).
  - Uses `viz.style` helpers `use_pub / save_pub / panel_label / C` (PNG at 600 dpi).
- **Files written:** `outputs/pretrain_milp/n{N}/figures/03_optimization_process.png`,
  `…/04_runtime.png`.

### 12.2 POST‑HOC oracle comparison (figs 01/02 + `milp_vs_oracle.csv`)
**Step (line `util/pretrain_milp.py:248-267`).** **This block runs strictly AFTER every MILP schedule
has already been computed and checkpointed (§11). The oracle result is read here for comparison only;
it never influences the MILP algorithm — there is no leakage.**

1. `oracle_dir = scale_dir(ROOT/"outputs"/"oracle")` → `outputs/oracle/n{N}/` (`pretrain_milp.py:248`).
2. `oracle_opt_path = oracle_dir/"oracle_optima.csv"` (`pretrain_milp.py:249`).
3. **If** `oracle_optima.csv` exists (`pretrain_milp.py:250`):
   - Import `make_comparison, make_landscape` (`pretrain_milp.py:251`).
   - `oracle_opt = pd.read_csv(oracle_optima.csv)` (`pretrain_milp.py:252`).
   - Merge MILP optima with the oracle's per‑scenario `F` (renamed `F_oracle`) on `scenario`
     (left join) (`pretrain_milp.py:253-254`).
   - `merged["gap"] = merged["F_milp"] - merged["F_oracle"]` — negative gap ⇒ MILP beat the
     work‑conserving oracle (`pretrain_milp.py:255`).
   - **Writes** `outputs/pretrain_milp/n{N}/milp_vs_oracle.csv` (`pretrain_milp.py:256`).
   - `make_comparison(out_dir, merged, segments, T)` — **fig 01_milp_vs_oracle.png**
     (`viz/pretrain_viz.py:21-59`; panel a: per‑scenario `F_oracle` vs `F_milp`; panel b: the gap
     bars) (`pretrain_milp.py:257`).
   - `make_landscape(out_dir, pd.read_csv(oracle_dir/"oracle_landscape.csv"), milp_opt, oracle_opt,
     segments, T)` — **fig 02_cross_scenario_landscape.png** (`viz/pretrain_viz.py:62-108`; fixed
     single‑policy schedules ranked by mean `F`, with the MILP adaptive mean and hindsight mean
     overlaid) (`pretrain_milp.py:258-259`).
   - Print mean `F_milp` / mean `F_oracle` / mean gap (`pretrain_milp.py:260-263`).
4. **Else** (oracle for this scale not ready): print a deferral note pointing at
   `python -m util.pretrain_milp --landscape` (`pretrain_milp.py:264-266`).
5. Print `Wrote {out_dir}` (`pretrain_milp.py:267`).

- **Files read (post‑hoc):** `outputs/oracle/n{N}/oracle_optima.csv`,
  `outputs/oracle/n{N}/oracle_landscape.csv`.
- **Files written:** `outputs/pretrain_milp/n{N}/milp_vs_oracle.csv`,
  `…/figures/01_milp_vs_oracle.png`, `…/figures/02_cross_scenario_landscape.png`.

> `make_comparison`/`make_landscape`/`make_process_figures` all call `viz.style.use_pub()` and
> `save_pub()` (PNG @ 600 dpi; SVG/PDF off by default — `viz/style.py:59, 78-87`). These are
> presentation only and do not feed back into any computation.

---

## 13. Complete file‑I/O map

| Direction | Path | Where | Step |
|---|---|---|---|
| read | `data/siouxfalls_toy/network/edges.csv` | `select_oracle_instance`, `load_toy_network` | §3.1, §4.2 |
| read | `data/siouxfalls_toy/network/od_pairs.csv` | `load_toy_network` | §4.2 |
| read | `data/siouxfalls_toy/network/nodes.csv` | `load_toy_network` | §4.2 |
| read | `data/siouxfalls_toy/raw/SiouxFalls_flow.tntp` | `_reference_twoway_flow` | §3.2 |
| write | `data/siouxfalls_toy/disruption/disrupted_segments_oracle{N}.csv` | `select_oracle_instance` | §3.1 |
| read (resume) | `outputs/pretrain_milp/n{N}/milp_optima.csv` | resume | §11.2 |
| read (resume) | `outputs/pretrain_milp/n{N}/milp_trace.csv` | resume | §11.2 |
| read (resume) | `outputs/pretrain_milp/n{N}/milp_progress.json` | resume | §11.2 |
| write (checkpoint) | `outputs/pretrain_milp/n{N}/milp_optima.csv` | per scenario | §11.3 |
| write (checkpoint) | `outputs/pretrain_milp/n{N}/milp_trace.csv` | per scenario | §11.3 |
| write (checkpoint) | `outputs/pretrain_milp/n{N}/milp_progress.json` | per scenario | §11.3 |
| write | `outputs/pretrain_milp/n{N}/run_meta.json` | final | §11.4 |
| write | `outputs/pretrain_milp/n{N}/figures/03_optimization_process.png` | figs | §12.1 |
| write | `outputs/pretrain_milp/n{N}/figures/04_runtime.png` | figs | §12.1 |
| read (post‑hoc) | `outputs/oracle/n{N}/oracle_optima.csv` | oracle compare | §12.2 |
| read (post‑hoc) | `outputs/oracle/n{N}/oracle_landscape.csv` | oracle compare | §12.2 |
| write | `outputs/pretrain_milp/n{N}/milp_vs_oracle.csv` | oracle compare | §12.2 |
| write | `outputs/pretrain_milp/n{N}/figures/01_milp_vs_oracle.png` | oracle compare | §12.2 |
| write | `outputs/pretrain_milp/n{N}/figures/02_cross_scenario_landscape.png` | oracle compare | §12.2 |

---

## Appendix A — parameter defaults used (from `config.py`)

| param | default | role in this pipeline |
|---|---|---|
| `DELTA_T_H` | 3.0 | hours/slot; cancels out of F2, part of the fingerprint |
| `C_MAX` | 2 | crew cap in `schedule_from_permutation` and the MILP crew‑cap constraint |
| `MU` | 1.0 | `F = MU·F1 + (1-MU)·F2` ⇒ objective is F1‑only (F2 logged only) |
| `CAP_RETAIN` | {1:0.3, 2:0.1, 3:0.02} | damaged capacity multiplier in `build_damaged_edges` |
| `SPEED_RETAIN` | {1:0.5, 2:0.3, 3:0.2} | damaged free‑flow‑time divisor in `build_damaged_edges` |
| `SEVER_SEVERITY` | 3 | severity ≥ this ⇒ edge removed (disconnection → `u_pen`) |
| `F1_ACTIVE_ONLY` | False | F1 averaged over full `[1,T]` (matches `c_e^k`'s `1/T`) |
| `RHO` | 0.7 | recovery inertia; used in both the true demand model and `c_e^k`'s `(1−ρ^{n+1})` |
| `KAPPA` | 1.0 | damage→demand‑shortfall scale in `B(Φ)` |
| `UPEN_FACTOR` | 10.0 | `u_pen = 10 × max finite baseline OD time` |
| `M_SCENARIOS` | 10 | number of duration scenarios (loop length) |
| `SEED` | 42 | RNG seed for `sample_scenarios` |
| `N_DISRUPTED_ORACLE` | 4 | instance size; keys the `n{N}` output folder |
| `UE_RGAP` | 1e-6 | per‑step UE relative‑gap target |
| `UE_MAX_ITER` | 100 | per‑step UE max Frank‑Wolfe iterations |
| `MILP_MAX_ITER` | 20 | alternating‑loop hard cap (stop test ③) |
| `MILP_CYCLE_TOL` | 3 | relaxed cycle guard (stop test ②) |
| `MILP_DAMPING` | 0.5 | MSA travel‑time relaxation `u=damp·u_new+(1-damp)·u_prev` |
| `DURATION_SUPPORT` | Table 1 (see `config.py:51-55`) | per‑(class,severity) base duration support sets |
| `ETA` | [0.8, 1.0, 1.2] | crew‑efficiency multipliers in `sample_scenarios` |

## Appendix B — shared problem definition vs post‑hoc oracle read (intentional, no leakage)

Two categories of "sharing" with the oracle exist, and only one is upstream of the algorithm:

1. **Shared problem definition (intentional, upstream).** The MILP run and the oracle run use the
   *same* functions to define the instance and objective: `select_oracle_instance` (same disrupted
   segments + severities), `sample_scenarios` (same durations for the same seed), `compute_horizon`
   (same `T`), and `evaluate_schedule` (the same true `F(x|ω)`). This is deliberate so `F_milp` and
   `F_oracle` are directly comparable. It is **not** information leakage — no oracle *solution* is
   consulted while the MILP searches.

2. **Post‑hoc oracle read (comparison only, downstream).** The oracle's optimal `F` and landscape are
   read only in §12.2, *after* every scenario's MILP schedule is already computed and checkpointed to
   `milp_optima.csv`. They feed only the comparison CSV (`milp_vs_oracle.csv`) and figures 01/02. The
   MILP objective (`−Σ c_e^k y_e^k`) and the `c_e^k` coefficients depend solely on the fixed UE travel
   times from the schedule's own iterates — never on the oracle. Hence the MILP result is what it is
   regardless of whether the oracle has been computed (if it hasn't, the run simply defers the
   comparison figures).
```
