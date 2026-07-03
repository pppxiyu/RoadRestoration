# 01 — `run_oracle()`: complete execution logic

**Purpose.** Enumerate *all* work-conserving schedules (permutations of the disrupted segments) over `M` sampled duration-scenarios, evaluate the exact Figure-1 objective `F(x|ω)` for each, and record (a) the per-scenario hindsight optimum `F*` and (b) the full landscape (every tested `x`). This is the ground-truth against which the later pretraining MILP is validated (Level B — true objective; see Appendix B).

**Entry point.** `util/oracle.py :: run_oracle()` (defined at `util/oracle.py:119`).

**Invocation.**
```
python -m util.oracle --probe     # measure s/UE + project runtime, then stop
python -m util.oracle             # full run + figures
python -m util.oracle --force     # ignore cache / partial checkpoint, recompute
python -m util.oracle --figs      # render_figs(): redraw figures from saved CSVs (separate path)
```
The `__main__` block at `util/oracle.py:242-246` routes `--figs` to `render_figs()`, otherwise calls `run_oracle(probe=…, force=…)`.

---

## Table of contents
1. [Call-tree overview](#1-call-tree-overview)
2. [Module constants & the fingerprint mechanism](#2-module-constants--fingerprint)
3. [Step 0 — output dir + figures dir](#step0)
4. [Step 1 — `select_oracle_instance` (choose the disruption set)](#step1)
5. [Step 2 — `build_context` (network, UE baseline, B(Φ), u_pen, severity)](#step2)
6. [Step 3 — `sample_scenarios` (duration draws)](#step3)
7. [Step 4 — permutations](#step4)
8. [Step 5 — `compute_horizon` (global T)](#step5)
9. [Step 6 — cache check (meta.json)](#step6)
10. [Step 7 — probe timing](#step7)
11. [Step 8 — resume from checkpoint](#step8)
12. [Step 9 — main enumeration loop + `evaluate_schedule`](#step9)
13. [Step 10 — landscape sort + per-scenario optima](#step10)
14. [Step 11 — summary.txt](#step11)
15. [Step 12 — figures (`make_figures`)](#step12)
16. [Step 13 — meta.json write + checkpoint cleanup](#step13)
17. [Appendix A — parameter defaults table](#appA)
18. [Appendix B — validation levels & comparison metrics (preserved)](#appB)
19. [Appendix C — Figure-1 fidelity & modeling caveats (preserved)](#appC)

---

## 1. Call-tree overview

```
run_oracle()                                    util/oracle.py:119
├─ scale_dir(out_dir)                            util/oracle.py:57      → outputs/oracle/n{N}/
├─ (out_dir/"figures").mkdir(...)
├─ select_oracle_instance(toy, N, seed)          util/oracle.py:83
│   └─ _reference_twoway_flow(toy)               util/oracle.py:63      reads raw/SiouxFalls_flow.tntp
│      · reads network/edges.csv                 (pd.read_csv)
│      · writes disruption/disrupted_segments_oracle{n}.csv
├─ segments = sorted(edge_id)
├─ build_context(toy, disrupted)                 util/evaluate.py:106
│   ├─ load_toy_network(toy)                      util/io.py:10          reads edges.csv, od_pairs.csv, nodes.csv
│   ├─ solve_ue(edges, M(H0), zone_ids)          util/ue.py:81          UE on UNDAMAGED net → baseline
│   │   ├─ _build_graph(edges, zone_ids)          util/ue.py:46          AequilibraE Graph
│   │   ├─ _build_matrix(M, zone_ids)             util/ue.py:71          AequilibraeMatrix
│   │   └─ TrafficAssignment.execute()            (AequilibraE bi-conjugate Frank-Wolfe)
│   ├─ _matrix_from_H(H0, ctx)                    util/evaluate.py:48
│   ├─ od_travel_times(base_links, ctx)           util/evaluate.py:27    networkx single-source dijkstra
│   └─ builds B(Φ), u_pen, severity_vec           (networkx shortest_path over free-flow graph)
├─ sample_scenarios(disrupted, M, seed)          util/scenarios.py:14   rng draws over DURATION_SUPPORT × ETA
├─ perms = itertools.permutations(segments)
├─ compute_horizon(segments, scenarios)          util/oracle.py:109
│   └─ for each perm × scenario:
│       schedule_from_permutation(perm, dur)      util/evaluate.py:82
│       makespan_slot(start, dur)                 util/evaluate.py:94
├─ _param_fingerprint()                          util/oracle.py:44      sha1 over FINGERPRINT_PARAMS
├─ [cache] if meta.json hash matches → read CSVs, make_figures, RETURN     util/oracle.py:134-145
├─ [probe] evaluate one schedule, project runtime; if --probe RETURN       util/oracle.py:148-156
├─ [resume] read landscape_progress.json + partial oracle_landscape.csv    util/oracle.py:159-169
├─ MAIN LOOP  for m,dur in scenarios: for perm in perms:                    util/oracle.py:174-192
│   ├─ schedule_from_permutation(perm, dur)       util/evaluate.py:82
│   ├─ evaluate_schedule(start, dur, T, ctx)      util/evaluate.py:156   ← the exact objective
│   │   ├─ f2_value(start, dur)                    util/evaluate.py:98    (makespan_slot util/evaluate.py:94)
│   │   └─ per-slot loop k=1..T:
│   │       ├─ demand shortfall  D=max(B·v, ρD); H=clip(H0−D)
│   │       ├─ build_damaged_edges(ctx, damaged)  util/evaluate.py:54
│   │       ├─ _matrix_from_H(H, ctx)             util/evaluate.py:48
│   │       ├─ solve_ue(dmg_edges, M(H), ...)      util/ue.py:81          ONE UE per slot
│   │       └─ od_travel_times(links, ctx)         util/evaluate.py:27
│   ├─ append row(scenario,perm,F,F1,F2,eval_s,start_e…)
│   └─ per-scenario checkpoint: write oracle_landscape.csv + landscape_progress.json
├─ land.sort_values(["scenario","perm"]); write oracle_landscape.csv
├─ opt = land.groupby("scenario")["F"].idxmin(); write oracle_optima.csv    util/oracle.py:197-198
├─ write summary.txt                              util/oracle.py:200-211
├─ make_figures(...)                              viz/oracle_viz.py:23   figs 01/02/03
├─ write meta.json (hash+params+timing)           util/oracle.py:216-222
└─ prog_path.unlink()                             util/oracle.py:223    drop resume marker
```

Two AequilibraE UE solve sites: (1) one baseline UE inside `build_context`; (2) one UE **per slot** inside `evaluate_schedule`. Everything else is arithmetic + networkx shortest paths. Total UE solves in a full run ≈ `1 + (perms × M × T)` (plus 1 for the probe and 1 baseline).

---

## 2. Module constants & the fingerprint mechanism <a id="2-module-constants--fingerprint"></a>

**Paths** (`util/oracle.py:31-33`):
- `ROOT` = repo root (parent of `util/`).
- `TOY = ROOT/data/siouxfalls_toy` — default `toy_dir`.
- `OUT = ROOT/outputs/oracle` — default `out_dir` (before scale suffix).

**`FINGERPRINT_PARAMS`** (`util/oracle.py:37-41`): the list of config attributes whose change should invalidate a cached result of the same scale:
```
N_DISRUPTED_ORACLE, MU, CAP_RETAIN, SPEED_RETAIN, SEVER_SEVERITY,
F1_ACTIVE_ONLY, RHO, KAPPA, UPEN_FACTOR, DELTA_T_H, C_MAX,
M_SCENARIOS, SEED, UE_RGAP, UE_MAX_ITER, DURATION_SUPPORT, ETA
```
Note the split: **`N_DISRUPTED_ORACLE` keys the cache folder** (`n{N}/`, via `scale_dir`), while the *whole* list (including `N_DISRUPTED_ORACLE`) keys the **freshness hash** inside that folder.

### `_param_fingerprint()` — `util/oracle.py:44-54`
- Loops over `FINGERPRINT_PARAMS`, reads each via `getattr(P, name)`.
- For any `dict`-valued param (`CAP_RETAIN`, `SPEED_RETAIN`, `DURATION_SUPPORT`), stringifies its keys: `{str(k): v[k] …}`. This matters because `DURATION_SUPPORT` has **tuple keys** like `("local", 1)` which JSON cannot serialize as keys.
- `blob = json.dumps(values, sort_keys=True)`.
- Returns `(values, hashlib.sha1(blob.encode("utf-8")).hexdigest())`.
- **Inputs → outputs:** config module `P` → `(values dict, sha1 hex string fp)`. Used for both the cache check (`meta.json["hash"]`) and the resume marker (`landscape_progress.json["hash"]`).

### `scale_dir(base=OUT, n=None)` — `util/oracle.py:57-60`
- `n` defaults to `P.N_DISRUPTED_ORACLE`.
- Returns `Path(base)/f"n{n}"` → e.g. `outputs/oracle/n4/`. Guarantees a different `N` never overwrites another scale's results. **First thing `run_oracle` does** (`util/oracle.py:120`).

---

## Step 0 — output dir + figures dir <a id="step0"></a>
`util/oracle.py:120-121`
```python
out_dir = scale_dir(out_dir)                 # outputs/oracle/n{N}/
(out_dir / "figures").mkdir(parents=True, exist_ok=True)
```
- **Files touched:** creates `outputs/oracle/n{N}/figures/` (and parents).
- **Config used:** `N_DISRUPTED_ORACLE` (via `scale_dir`).

---

## Step 1 — `select_oracle_instance` (choose the disruption set) <a id="step1"></a>
`util/oracle.py:123` → `select_oracle_instance(toy_dir, P.N_DISRUPTED_ORACLE, seed)` defined at `util/oracle.py:83-106`.

**What it does.** Deterministically picks `n` disrupted segments by *importance* (baseline two-way UE flow), mixing critical and minor links so restoration order strongly affects F1. It then assigns severities and writes the instance CSV.

**Sub-logic, step by step:**
1. `edges = pd.read_csv(toy/"network"/"edges.csv")` (`util/oracle.py:89`). Columns: `edge_id,u,v,capacity,length,free_flow_time,bpr_alpha,bpr_beta,road_class`.
2. `flow = _reference_twoway_flow(toy)` — see below.
3. Attach a `flow` column: for each edge row, look up `flow[(min(u,v), max(u,v))]`, default `0.0` (`util/oracle.py:91-92`).
4. `ranked = edges.sort_values("flow", ascending=False).reset_index(drop=True)` — highest-flow edge first (`util/oracle.py:93`).
5. **Pick indices** (`util/oracle.py:95-100`):
   - `n_crit = min(2, n)` → the top-2 highest-flow edges are "critical."
   - `picks = [0, 1, …, n_crit-1]` (top flow ranks).
   - `rest = n - n_crit`. If `rest>0`, spread the remaining picks over lower-flow edges via `np.linspace(len(ranked)//5, len(ranked)-1, rest)` rounded to int indices. So the extra picks are drawn from the 20%-mark down to the least-used edge — deliberately low-flow "minor" links.
   - `sub = ranked.iloc[picks]`.
6. **Severity assignment** (`util/oracle.py:101`):
   - `sub["severity"] = [3]*n_crit + [2 if i%2==0 else 1 for i in range(rest)]`.
   - So: the **top-2 flow edges get severity 3** (which, since `SEVER_SEVERITY=3`, are fully severed in the damaged network); the remaining edges alternate severity **2, 1, 2, 1, …**.
7. `sub["level_id"] = road_class + "-S" + severity` (e.g. `highway-S3`) (`util/oracle.py:102`).
8. Select/sort output columns `[edge_id,u,v,road_class,severity,level_id]` sorted by `edge_id` (`util/oracle.py:103-104`).
9. **Write** `toy/"disruption"/f"disrupted_segments_oracle{n}.csv"` (`util/oracle.py:105`), return the DataFrame.

**Inputs → outputs:** `toy_dir, n, seed` (seed is accepted but **unused** — selection is fully deterministic from flow) → DataFrame `disrupted[edge_id,u,v,road_class,severity,level_id]`; side-effect CSV `disruption/disrupted_segments_oracle{n}.csv`.

**Config used:** `N_DISRUPTED_ORACLE` (as `n`). `SEVER_SEVERITY` is not read here but governs the meaning of the severity-3 assignment downstream.

### `_reference_twoway_flow(toy)` — `util/oracle.py:63-80`
- Reads `toy/"raw"/"SiouxFalls_flow.tntp"` line by line.
- Skips blank lines and the header (line starting with `from`, case-insensitive) (`util/oracle.py:69`).
- Strips a trailing `;`, splits on whitespace; needs ≥3 tokens (`util/oracle.py:71-73`).
- Parses `a,b = int`, `vol = float` (columns From, To, Volume); on `ValueError` skips.
- `key = (min(a,b), max(a,b))`; **accumulates both directed volumes into one undirected key**: `f[key] += vol` (`util/oracle.py:78-79`).
- Returns `{(min,max): summed two-way flow}`.
- **Files touched:** reads `raw/SiouxFalls_flow.tntp` only.

Then back in `run_oracle`: `segments = sorted(int(e) for e in disrupted["edge_id"])` (`util/oracle.py:124`) — the canonical sorted list of edge ids being scheduled.

---

## Step 2 — `build_context` (network, UE baseline, B(Φ), u_pen, severity) <a id="step2"></a>
`util/oracle.py:125` → `ctx = build_context(toy_dir, disrupted)`, defined at `util/evaluate.py:106-150`. Built **once**; reused for every scenario × permutation.

### 2a. Load the static network — `load_toy_network` (`util/io.py:10-23`)
`edges, od, zone_ids = load_toy_network(toy_dir)` (`util/evaluate.py:108`):
- Reads three CSVs (`util/io.py:19-22`):
  - `network/edges.csv` → `edges` (undirected edge table, columns as above).
  - `network/od_pairs.csv` → `od` (`od_id,origin,destination,h0`).
  - `network/nodes.csv` → `nodes`; `zone_ids = nodes["node_id"]` int array (**every node is an OD zone** in Sioux Falls).
- Returns `(edges, od, zone_ids)`.

### 2b. Index maps (`util/evaluate.py:109-116`)
- `zone_pos = {node_id: dense_index}`.
- `od_pairs = [(origin, destination), …]` from `od` rows.
- `H0 = od["h0"]` — the **normal-time demand vector** aligned to `od_pairs` (also the fully-recovered level).
- `oi = [zone_pos[o]]`, `di = [zone_pos[d]]` — dense row/col indices per OD pair (used by `_matrix_from_H`).
- `edge_row = {edge_id: row_index_in_edges}`.
- `eid_of = {(min(u,v),max(u,v)): edge_id}` — undirected-key → edge id.
- Assemble base `ctx` dict: `edges, zone_ids, od_pairs, H0, oi, di, nz=len(zone_ids), edge_row, origins_unique=sorted(set of origins)` (`util/evaluate.py:118-120`).

### 2c. Baseline `u_r^{t0}` — UE on the UNDAMAGED network (`util/evaluate.py:122-125`)
```python
base_links, _ = solve_ue(edges, _matrix_from_H(H0, ctx), zone_ids,
                         rgap=P.UE_RGAP, max_iter=P.UE_MAX_ITER, quiet=True)
ctx["baseline_u"] = od_travel_times(base_links, ctx)
```
- `_matrix_from_H(H0, ctx)` (`util/evaluate.py:48-51`): builds a dense `nz×nz` zero matrix and scatters `M[oi, di] = H0` — i.e. the demand vector placed at its OD cells.
- **`solve_ue(...)`** runs a full user-equilibrium assignment on the *pristine* edge table with normal demand. See §2e for the internal UE trace. `quiet=True` silences AequilibraE logging + stdout/stderr.
- `od_travel_times(base_links, ctx)` (§2f) → `baseline_u[i]` = shortest-path congested travel time for each OD pair `i` under the equilibrium link costs. This is the denominator reference in F1.

### 2d. `u_pen` — disconnected-OD penalty (`util/evaluate.py:126`)
```python
ctx["u_pen"] = P.UPEN_FACTOR * float(np.nanmax(baseline_u[np.isfinite(baseline_u)]))
```
- `UPEN_FACTOR × (max finite baseline OD travel time)`. With `UPEN_FACTOR = 10.0`, disconnected OD pairs are charged 10× the worst normal-time trip.

### 2e. `solve_ue` internal trace (AequilibraE bi-conjugate Frank-Wolfe) — `util/ue.py:81-121`
`solve_ue(edges, od_matrix, zone_ids, algorithm="bfw", max_iter, rgap, quiet)`:
1. **`_build_graph(edges, zone_ids)`** (`util/ue.py:46-68`): builds a pandas link table with `link_id=1..n`, `a_node=u`, `b_node=v`, `direction=0` (bidirectional), `distance=length`, `modes="c"`, `capacity_ab=capacity_ba=capacity`, `free_flow_time`, `b=bpr_alpha` (BPR α), `power=bpr_beta` (BPR β). Creates `Graph()`, sets `g.network`, `g.prepare_graph(zone_ids)` (all nodes are zones), `g.set_graph("free_flow_time")` (cost field), `g.set_blocked_centroid_flows(False)` (allow through-traffic). Returns `(g, net)`.
2. **`_build_matrix(M, zone_ids)`** (`util/ue.py:71-78`): wraps the dense OD matrix in an in-memory `AequilibraeMatrix` with one core `"demand"`; `computational_view(["demand"])`.
3. Build `TrafficClass("car", graph, mat)`; `TrafficAssignment()`; `set_classes`; `set_vdf("BPR")`; `set_vdf_parameters({"alpha":"b","beta":"power"})`; `set_capacity_field("capacity")`; `set_time_field("free_flow_time")`; `set_algorithm("bfw")`; `assig.max_iter=max_iter`; `assig.rgap_target=rgap` (`util/ue.py:92-101`).
4. **`assig.execute()`** (`util/ue.py:106` under redirect, or `:108`): runs the Frank-Wolfe loop. **Core UE idea** (from the module docstring, `util/ue.py:12-28`): minimize the Beckmann objective `Z(x)=Σ_a ∫₀^{x_a} t_a(w)dw` with BPR cost `t_a(x)=t0_a·(1+α·(x_a/c_a)^β)`. Each FW iteration: (1) **all-or-nothing** — assign every OD trip to the current shortest path → auxiliary flow `y`; (2) **line search / move** `x ← x + step·(y−x)` decreasing `Z`; (3) **update costs**, stop when the **relative gap** < `rgap`. `bfw` = bi-conjugate Frank-Wolfe variant.
5. `res = assig.results()` — per-link AB/BA equilibrium flow + congested time. Emit two rows per link (AB and BA) with `from,to,volume,cost` (`cost` from `Congested_Time_AB/BA`). Drop rows with NaN volume. Return `(flows_df[from,to,volume,cost], assig)` (`util/ue.py:110-121`).

**Config used by every UE call:** `UE_RGAP` (1e-6), `UE_MAX_ITER` (100). BPR α/β and capacity come from the (possibly damaged) edge table.

### 2f. `od_travel_times` internal trace (networkx dijkstra) — `util/evaluate.py:27-45`
`od_travel_times(link_df, ctx)`:
- Builds a `nx.DiGraph` from `link_df[from,to,cost]`, adding a directed edge per row with `weight=cost` (`util/evaluate.py:31-35`).
- `u = np.full(len(od_pairs), np.inf)` (`util/evaluate.py:36`).
- For each unique origin `o` present in `G`, computes `nx.single_source_dijkstra_path_length(G, o, weight="weight")` → dict of shortest costs to all reachable nodes; caches in `by_origin[o]` (empty dict if `o` not in `G`) (`util/evaluate.py:37-42`).
- For each OD pair `i=(o,d)`: `u[i] = by_origin[o].get(d, inf)` (`util/evaluate.py:43-44`).
- Returns `u` (np.inf where O and D are disconnected).

### 2g. B(Φ) demand-shortfall matrix (`util/evaluate.py:129-148`)
- `dis = [(edge_id, u, v, severity) …]` for the disrupted set; stored as `ctx["disrupted"]`.
- Build an **undirected free-flow graph** `Gff`: for each edge add *both* directions with `weight=free_flow_time` and attribute `eid` (`util/evaluate.py:132-135`).
- `B = zeros(len(od_pairs), len(dis))`; `col = {edge_id: column j}` mapping each disrupted edge to a B column (`util/evaluate.py:136-137`).
- For each OD pair `i=(o,d)`: compute the **free-flow shortest path** `nx.shortest_path(Gff, o, d, weight="weight")`; on `NetworkXNoPath` skip (`util/evaluate.py:138-142`).
- Collect the undirected edge keys on that path (`on_path`); for each, if it maps to a disrupted `eid` in `col`, set `B[i, col[eid]] = P.KAPPA * (H0[i]/3.0)` (`util/evaluate.py:143-147`).
- Meaning: `B[r,e] = κ·(h_r^{t0}/3)·1{edge e lies on OD r's free-flow shortest path}`. It is the sensitivity of OD r's demand shortfall to damage on segment e.
- `ctx["B"] = B` (`util/evaluate.py:148`).

### 2h. `severity_vec` (`util/evaluate.py:149`)
- `ctx["severity_vec"] = np.array([sev for (_,_,_,sev) in dis], float)` — per-disrupted-edge severity, aligned to `dis` order. (Computed but note: `evaluate_schedule` reads `sev` from `ctx["severity_vec"]` but does not use it in the F computation — the live per-slot severity comes from the `dis` tuples directly.)

**`build_context` inputs → outputs:** `(toy_dir, disrupted DataFrame)` → `ctx` dict with keys `edges, zone_ids, od_pairs, H0, oi, di, nz, edge_row, origins_unique, baseline_u, u_pen, disrupted, B, severity_vec`.
**Files touched:** reads `edges.csv, od_pairs.csv, nodes.csv`. **Config used:** `UE_RGAP, UE_MAX_ITER, UPEN_FACTOR, KAPPA`.

---

## Step 3 — `sample_scenarios` (duration draws) <a id="step3"></a>
`util/oracle.py:126` → `scenarios = sample_scenarios(disrupted, M, seed)`, defined at `util/scenarios.py:14-28`.

**What it does.** Draws `M` restoration-duration scenarios. Durations are per **level** `(road_class, severity)`; all disrupted segments of the same level share one duration within a scenario (`d_e(ω)=d_{ℓ(e)}(ω)`).

**Sub-logic:**
1. `rng = np.random.default_rng(seed)` (`util/scenarios.py:17`) — seed = `P.SEED` = 42. Deterministic sequence.
2. `levels = sorted({(road_class, severity)})` over the disrupted rows (`util/scenarios.py:19`).
3. For each of `M` scenarios (`util/scenarios.py:21-26`):
   - For each level: `base = int(rng.choice(P.DURATION_SUPPORT[lvl]))` (a slot count from Table 1 for that `(class,severity)`); `eta = float(rng.choice(P.ETA))` (crew-efficiency multiplier 0.8/1.0/1.2); `lvl_dur[lvl] = max(1, int(round(base*eta)))`.
   - Map to edges: `scenario[edge_id] = lvl_dur[(road_class, severity)]` for every disrupted row (`util/scenarios.py:27`).
4. Returns `list[dict]` of length `M`: `scenario[m][edge_id] = duration_in_slots`.

**Inputs → outputs:** `(disrupted, M, seed)` → `scenarios: list of M dicts {edge_id: duration_slots}`.
**Config used:** `M_SCENARIOS`, `SEED`, `DURATION_SUPPORT`, `ETA`.
**Note on RNG ordering:** the two `rng.choice` calls per level, iterated over sorted levels and M scenarios, fully determine every scenario given the seed — so scenarios are reproducible and part of the fingerprint (via `DURATION_SUPPORT`, `ETA`, `SEED`, `M_SCENARIOS`).

---

## Step 4 — permutations <a id="step4"></a>
`util/oracle.py:127` → `perms = list(itertools.permutations(segments))`.
- Every ordering of the `N` disrupted edge ids. For `N=4` → `4! = 24` permutations. Each permutation is a restoration **priority order** fed to work-conserving list scheduling.

---

## Step 5 — `compute_horizon` (global T) <a id="step5"></a>
`util/oracle.py:128` → `T = compute_horizon(segments, scenarios)`, defined at `util/oracle.py:109-116`.

**What it does.** Global horizon `T` = the **max completion slot over all permutations × scenarios**, so every enumerated schedule finishes within `T` and all share one comparable horizon for F1.

**Sub-logic** (`util/oracle.py:112-116`):
```python
T = 0
for perm in itertools.permutations(segments):
    for dur in scenarios:
        T = max(T, makespan_slot(schedule_from_permutation(list(perm), dur), dur))
return T
```
For each `(perm, scenario)`:

### `schedule_from_permutation(perm, durations, c_max=P.C_MAX)` — `util/evaluate.py:82-91`
Work-conserving list scheduling with `c_max` identical crews:
- `crew_free = [1]*c_max` — every crew idle from slot 1 (start strictly after onset slot 0).
- For each edge `e` in priority order: pick the earliest-free crew `c = argmin(crew_free)`; `start[e] = crew_free[c]`; mark that crew busy until completion: `crew_free[c] = start[e] + durations[e]`.
- Returns `{edge_id: start_slot}`. **No idling** — each crew immediately takes the next edge.
- **Config used:** `C_MAX` (2).

### `makespan_slot(start, durations)` — `util/evaluate.py:94-95`
- `max(start[e] + durations[e] for e in start)` — the completion slot of the last-finishing edge.

**Inputs → outputs:** `(segments, scenarios)` → integer `T` (slots).
**Config used (transitively):** `C_MAX`. Then `print(...)` logs the instance summary (`util/oracle.py:129`): number of segments, `perms`, `M`, `T`.

---

## Step 6 — cache check (meta.json) <a id="step6"></a>
`util/oracle.py:132-145`.
```python
values, fp = _param_fingerprint()
meta_path = out_dir / "meta.json"
if not probe and not force and meta_path.exists():
    cached = json.loads(meta_path.read_text(...))
    if cached.get("hash") == fp:
        # cache HIT
        land = pd.read_csv(out_dir / "oracle_landscape.csv")
        opt  = pd.read_csv(out_dir / "oracle_optima.csv")
        from viz.oracle_viz import make_figures
        make_figures(out_dir, land, opt, ctx, segments, scenarios, T, disrupted)
        return
```
- Skipped entirely if `--probe` or `--force`.
- **Cache HIT condition:** `meta.json` exists AND its stored `"hash"` equals the current fingerprint `fp`. On hit: reload the saved landscape + optima CSVs, re-render figures, print a `[cache] reusing…` message, and **return without any UE enumeration**.
- **Files read:** `meta.json`, `oracle_landscape.csv`, `oracle_optima.csv`.
- **Config used:** all of `FINGERPRINT_PARAMS` (via `fp`), `N_DISRUPTED_ORACLE` (in the log message).

---

## Step 7 — probe timing <a id="step7"></a>
`util/oracle.py:148-156`.
```python
t0 = time.perf_counter()
evaluate_schedule(schedule_from_permutation(list(perms[0]), scenarios[0]), scenarios[0], T, ctx)
dt = time.perf_counter() - t0
s_ue = dt / T                     # seconds per UE solve (T UE solves in one schedule)
total = len(perms) * M
print(...projected full run...)
if probe:
    return
```
- Always runs one real `evaluate_schedule` (= `T` UE solves) on the first permutation under the first scenario to measure **`s_ue` = seconds per UE solve** and project full runtime `total × T × s_ue`.
- If `--probe`: prints the projection and **returns** (no enumeration).
- Note: the probe result is *not* stored into `rows`; the main loop re-evaluates from scratch (including the probed `(perm[0], scenario[0])`).

---

## Step 8 — resume from checkpoint <a id="step8"></a>
`util/oracle.py:159-169`.
```python
land_path = out_dir / "oracle_landscape.csv"
prog_path = out_dir / "landscape_progress.json"
rows, done = [], set()
if not force and land_path.exists() and prog_path.exists():
    prog = json.loads(prog_path.read_text(...))
    if prog.get("hash") == fp:
        prev = pd.read_csv(land_path)
        rows = prev.to_dict("records")
        done = {int(s) for s in prev["scenario"].unique()}
        print(f"[resume] {len(done)}/{M} scenarios already computed …")
```
- **Interruption-safe resume.** If a *partial* landscape (`oracle_landscape.csv`) and a progress marker (`landscape_progress.json`) both exist AND the marker's `hash` matches the current fingerprint, reload the already-computed rows into `rows` and mark those scenarios `done`. The main loop then skips them and computes only the remainder.
- Disabled by `--force`. Fingerprint mismatch → start fresh (stale partial ignored).
- **Files read:** `oracle_landscape.csv`, `landscape_progress.json`.

---

## Step 9 — main enumeration loop + `evaluate_schedule` <a id="step9"></a>
`util/oracle.py:172-192`.
```python
t_run = time.perf_counter(); scen_times = []
for m, dur in enumerate(scenarios):
    if m in done: continue                       # resume skip
    t_scen = time.perf_counter()
    for perm in perms:
        start = schedule_from_permutation(list(perm), dur)      # util/evaluate.py:82
        t_ev = time.perf_counter()
        res = evaluate_schedule(start, dur, T, ctx)             # util/evaluate.py:156
        row = dict(scenario=m, perm="-".join(map(str,perm)),
                   F=res["F"], F1=res["F1"], F2=res["F2"],
                   eval_s=time.perf_counter()-t_ev)
        for e in segments: row[f"start_{e}"] = start[e]
        rows.append(row)
    scen_times.append(time.perf_counter()-t_scen)
    done.add(m)
    pd.DataFrame(rows).to_csv(land_path, index=False)           # per-scenario checkpoint
    prog_path.write_text(json.dumps({"hash": fp, "done": sorted(done)}), ...)
    print(f"  scenario {m+1}/{M} done ...")
```
- For each scenario `m` (skipping any already `done`), iterate over all `perms`; build the schedule, evaluate it, and log one row per `(scenario, permutation)`.
- **Row columns:** `scenario, perm` (dash-joined edge ids), `F, F1, F2, eval_s` (this schedule's wall time), and one `start_{edge}` per segment.
- **Per-scenario checkpoint** (`util/oracle.py:190-191`): after finishing a scenario, rewrite the *entire* `oracle_landscape.csv` from `rows` and update `landscape_progress.json` with `{hash, done}`. This is what makes resume possible.

### `evaluate_schedule(start, durations, T, ctx, collect_traces=False, return_u=False)` — FULL trace
`util/evaluate.py:156-207`. Computes the exact `F(x|ω)` for one schedule under one scenario over horizon `T`.

Unpack context (`util/evaluate.py:159-162`): `dis=ctx["disrupted"]`, `H0, B`, `base_u=ctx["baseline_u"]`, `sev=ctx["severity_vec"]`.

**(A) F2 — no UE** (`util/evaluate.py:165`): `F2 = f2_value(start, durations)`.
- `f2_value` (`util/evaluate.py:98-100`): `makespan_slot(start,durations) / sum(durations.values())` = completion slot ÷ total work-slots. Pure schedule arithmetic (Δt cancels). Higher = the schedule stretches the makespan relative to the raw repair effort.

**(B) F1 — per-slot loop** (`util/evaluate.py:167-193`). Initialize `D = zeros(len(H0))` (demand shortfall, `D_0=0`); accumulators `terms, active, traces, u_rows`.
For each slot `k = 1..T`:
1. **Damage state `v^{t_k}`** (Eq. 2) (`util/evaluate.py:171-173`):
   - `damaged = {eid: s for (eid,_,_,s) in dis if k < start[eid] + durations[eid]}` — an edge is damaged on slots `[start, start+dur)`; restored (dropped from `damaged`) once `k ≥ completion`.
   - `v_vec = [s if eid in damaged else 0.0 for (eid,_,_,s) in dis]` — per-disrupted-edge live severity (0 when restored).
2. **Demand shortfall → `H_t`** (`util/evaluate.py:175-177`):
   - `target = B @ v_vec` — current damage-driven shortfall.
   - `D = np.maximum(target, P.RHO * D)` — **sharp drop** to `target` at onset, then, once damage heals and `target` shrinks, `D` decays at rate `RHO` (recovery inertia; closer to 1 = slower recovery).
   - `H = np.clip(H0 - D, 0.0, None)` — realized demand = normal minus shortfall, floored at 0.
3. **Damaged network** (`util/evaluate.py:179`): `dmg_edges = build_damaged_edges(ctx, damaged)` — see below.
4. **UE + OD times** (`util/evaluate.py:181-185`):
   - `links, _ = solve_ue(dmg_edges, _matrix_from_H(H, ctx), ctx["zone_ids"], rgap=UE_RGAP, max_iter=UE_MAX_ITER, quiet=True)` — one UE on the damaged network with reduced demand `H`. (Same internal trace as §2e.)
   - `u = od_travel_times(links, ctx)` — congested OD travel times (§2f); `np.inf` for disconnected pairs.
   - `u_tilde = np.where(np.isfinite(u), u, ctx["u_pen"])` — replace `inf` with the penalty `u_pen`. Appended to `u_rows`.
5. **Demand-weighted ratio** (`util/evaluate.py:187-189`):
   - `den = float(np.sum(H * base_u))` (finite baseline weighted by current demand).
   - `term = np.sum(H * u_tilde) / den` if `den>0` else `1.0` — the per-slot F1 term = ratio of realized to baseline demand-weighted travel time.
   - `terms.append(term)`; `active.append(len(damaged) > 0)`.
   - If `collect_traces`: append `{k, n_damaged, total_demand=H.sum(), f1_term=term}` (used by figure 03).

**(C) F1 aggregation** (`util/evaluate.py:195-200`):
```python
terms = np.asarray(terms)
if P.F1_ACTIVE_ONLY:                       # lever 5
    mask = np.asarray(active, bool)
    F1 = terms[mask].mean() if mask.any() else terms.mean()
else:
    F1 = terms.mean()
```
- With `F1_ACTIVE_ONLY=False` (default): average over the **full horizon `[1,T]`** — matches the `c_e^k (1/T)` normalization used by the §2.1.1 MILP surrogate. If `True`: average only over slots where something is still damaged.

**(D) Compose F** (`util/evaluate.py:201`): `F = P.MU * F1 + (1.0 - P.MU) * F2`.
- With `MU=1.0` (default): `F = F1` (accessibility-only); F2 is still computed and logged but carries zero weight.

**Return** (`util/evaluate.py:202-207`): `{F, F1, F2}`; optionally `u_tilde` (the `(T,|R|)` fixed travel times, if `return_u`) and `traces` DataFrame (if `collect_traces`).

**Config used in `evaluate_schedule`:** `RHO`, `KAPPA` (baked into `B` at build time), `SEVER_SEVERITY`/`CAP_RETAIN`/`SPEED_RETAIN` (via `build_damaged_edges`), `UE_RGAP`, `UE_MAX_ITER`, `UPEN_FACTOR` (baked into `u_pen`), `F1_ACTIVE_ONLY`, `MU`, `C_MAX`/`DELTA_T_H` (via the schedule).

### `build_damaged_edges(ctx, damaged)` — `util/evaluate.py:54-76`
Return an edges DataFrame for the currently-damaged network. `damaged`: `{edge_id: severity}`.
- Copy `ctx["edges"]`. If `damaged` non-empty (`util/evaluate.py:60-75`):
  - `cap`, `fft` = copies of the `capacity` / `free_flow_time` columns; `idx = ctx["edge_row"]`.
  - For each `(eid, sev)` in `damaged`, `j = idx[eid]`:
    - if `sev >= P.SEVER_SEVERITY` → record `j` in `sever` (this edge is **fully removed** — true disconnection, so affected OD pairs later hit `u_pen`).
    - else → `cap[j] *= P.CAP_RETAIN[sev]` (harsher capacity) and `fft[j] /= P.SPEED_RETAIN[sev]` (slower free-flow time).
  - Write modified `cap`, `fft` back into the frame; if any `sever`, drop those rows (`edges.drop(index=…).reset_index`).
- Completed/undamaged links are untouched.
- **Config used:** `SEVER_SEVERITY`, `CAP_RETAIN`, `SPEED_RETAIN`.

---

## Step 10 — landscape sort + per-scenario optima <a id="step10"></a>
`util/oracle.py:193-198`.
```python
total_time = time.perf_counter() - t_run
land = pd.DataFrame(rows).sort_values(["scenario","perm"]).reset_index(drop=True)
land.to_csv(land_path, index=False)                                     # final landscape
opt = land.loc[land.groupby("scenario")["F"].idxmin()].reset_index(drop=True)
opt.to_csv(out_dir / "oracle_optima.csv", index=False)
```
- Assemble the full landscape DataFrame, sort by `(scenario, perm)`, write final `oracle_landscape.csv`.
- **Per-scenario optimum:** `groupby("scenario")["F"].idxmin()` selects, for each scenario, the row with minimum `F` (the true hindsight optimum `x*` and `F*`). Write `oracle_optima.csv`.
- **Files written:** `oracle_landscape.csv`, `oracle_optima.csv`.

---

## Step 11 — summary.txt <a id="step11"></a>
`util/oracle.py:200-211`. Composes and writes `summary.txt` (also echoed to stdout):
- `segments`, `perms`, `scenarios (M)`, horizon `T`.
- `~{s_ue*1000:.0f} ms/UE` (from the probe); total schedules evaluated `= len(land)`.
- Total eval compute = `land["eval_s"].sum()/60` min; mean `ms/schedule = land["eval_s"].mean()*1000`; this-session minutes `total_time/60`.
- `mean F* over scenarios = opt["F"].mean()`; `mean F1* = opt["F1"].mean()`, `mean F2* = opt["F2"].mean()`.
- Note that the oracle optimum is the min over all permutations per scenario (true hindsight optimum).
- **Files written:** `summary.txt`.

---

## Step 12 — figures (`make_figures`) <a id="step12"></a>
`util/oracle.py:213-214` → `make_figures(out_dir, land, opt, ctx, segments, scenarios, T, disrupted)`, defined in `viz/oracle_viz.py:23-110`. (This is the **same** call used on a cache hit at Step 6.)

**Setup** (`viz/oracle_viz.py:24-28`): `use_pub()` (apply publication rcParams from `viz/style.py:59`); ensure `figures/` exists; `sev = {edge_id: severity}`; `drep` = landscape rows for the representative scenario `rep=0`.

**Figure 01 — `01_F_landscape`** (`viz/oracle_viz.py:31-60`): for each scenario, sort its `F` values ascending; stack into `sorted_F` (rows=scenarios, cols=schedule rank). Plot the across-scenario min–max band, per-scenario best→worst curves, annotate the oracle optimum `F* = sorted_F[:,0].mean()`, and the best→worst spread `%`. Saved via `save_pub`.

**Figure 02 — `02_F1_F2_tradeoff`** (`viz/oracle_viz.py:63-74`): scatter of every schedule in the representative scenario in `(F2, F1)` space, colored by `F` (colormap `CMAP_SEQ`), with a colorbar labeled `F (μ=…)`. Shows the two-objective structure.

**Figure 03 — `03_best_schedule`** (`viz/oracle_viz.py:77-110`): take the optimum row for `rep`; rebuild `start = {e: orow[start_e]}`; **re-run** `evaluate_schedule(start, dur, T, ctx, collect_traces=True)` to get the per-slot `traces`. Three stacked panels: (a) Gantt of the optimal schedule (bars colored by severity via `severity_color`), (b) total OD demand over slots (sharp drop → recovery, dashed line = normal `H0.sum()`), (c) per-step F1 term over slots (dashed line at 1 = fully restored). Panel letters via `panel_label`.

### `viz/style.py` helpers used
- `use_pub()` (`viz/style.py:59-61`): apply `PUB_RC` rcParams (Arial/Helvetica, editable SVG/PDF text, compact journal sizes, no top/right spines).
- `severity_color(s)` (`:64`): `{1:#F6CFCB, 2:#E59A93, 3:#B64342}` warm ramp.
- `panel_label(ax,label)` (`:72`): bold top-left panel letter.
- `save_pub(fig, stem, dpi=600, svg=False, pdf=False)` (`:78`): saves **PNG at 600 dpi** by default (SVG/PDF only if requested — `make_figures` calls it with defaults, so PNG only).
- `C` palette, `CMAP_SEQ="cividis"`.

**Files written:** `figures/01_F_landscape.png`, `figures/02_F1_F2_tradeoff.png`, `figures/03_best_schedule.png`.
**Config used:** `MU` (colorbar label).

---

## Step 13 — meta.json write + checkpoint cleanup <a id="step13"></a>
`util/oracle.py:216-224`.
```python
meta_path.write_text(json.dumps(
    {"hash": fp, "params": values,
     "timing": {"total_eval_s": land["eval_s"].sum(), "n_schedules": len(land),
                "s_per_schedule": land["eval_s"].mean(),
                "s_per_ue": land["eval_s"].sum() / (len(land)*T),
                "this_session_s": total_time, "scenario_s": scen_times}},
    sort_keys=True, indent=2), ...)
prog_path.unlink(missing_ok=True)          # completed -> drop the resume marker
print(f"\nWrote {out_dir}  (meta.json hash={fp[:12]}…)")
```
- Writes `meta.json` = the freshness `hash` (fingerprint), the full `params` values, and a timing block (total eval seconds, schedule count, s/schedule, s/UE, session seconds, per-scenario seconds).
- **Deletes** `landscape_progress.json` (`prog_path.unlink(missing_ok=True)`) — the run completed, so the resume marker is no longer needed. Its presence next time would otherwise indicate an interrupted run.
- **Files written:** `meta.json`; **removed:** `landscape_progress.json`.

---

## Data-flow summary (variable → consumer)

| Produced by | Variable | Consumed by |
|---|---|---|
| `select_oracle_instance` | `disrupted` (DataFrame) | `build_context`, `sample_scenarios`, `make_figures`; `segments` |
| `_reference_twoway_flow` | undirected flow dict | edge ranking in `select_oracle_instance` |
| `load_toy_network` | `edges, od, zone_ids` | index maps, baseline UE, `B`, all damaged-net UEs |
| baseline `solve_ue` + `od_travel_times` | `ctx["baseline_u"]` | `u_pen`, F1 denominator (`base_u`) |
| `build_context` | `ctx["B"]`, `H0`, `u_pen`, `disrupted`, `severity_vec` | `evaluate_schedule` per slot |
| `sample_scenarios` | `scenarios` (M dicts) | `compute_horizon`, probe, main loop, figures |
| `schedule_from_permutation` | `start` (dict) | `evaluate_schedule`, `makespan_slot`, landscape rows |
| `compute_horizon` | `T` | probe, `evaluate_schedule`, figures, timing |
| `_param_fingerprint` | `values, fp` | cache check, resume check, meta.json |
| main loop | `rows` → `land` | `opt`, `summary.txt`, `make_figures`, meta timing |
| `land.groupby.idxmin` | `opt` | `oracle_optima.csv`, `summary.txt`, `make_figures` (fig 03) |

### Every file touched
| File | R/W | Where |
|---|---|---|
| `data/siouxfalls_toy/raw/SiouxFalls_flow.tntp` | R | `_reference_twoway_flow` |
| `data/siouxfalls_toy/network/edges.csv` | R | `select_oracle_instance`, `load_toy_network` |
| `data/siouxfalls_toy/network/od_pairs.csv` | R | `load_toy_network` |
| `data/siouxfalls_toy/network/nodes.csv` | R | `load_toy_network` |
| `data/siouxfalls_toy/disruption/disrupted_segments_oracle{n}.csv` | **W** | `select_oracle_instance` |
| `outputs/oracle/n{N}/oracle_landscape.csv` | R/W | resume read; per-scenario + final write |
| `outputs/oracle/n{N}/landscape_progress.json` | R/W/del | resume read; checkpoint write; deleted at end |
| `outputs/oracle/n{N}/oracle_optima.csv` | R/W | cache-hit read; final write |
| `outputs/oracle/n{N}/summary.txt` | W | Step 11 |
| `outputs/oracle/n{N}/meta.json` | R/W | cache check; final write |
| `outputs/oracle/n{N}/figures/01…03.png` | W | `make_figures` |

---

# Appendix A — parameter defaults table <a id="appA"></a>

All values below are the **current** `config.py` defaults (single source of truth). Each is a modeling assumption except where noted "given."

| param | current default | role in `run_oracle` |
|---|---|---|
| `DELTA_T_H` | 3.0 h/slot (given) | slot length; cancels in F2, sets the physical horizon meaning |
| `C_MAX` | 2 | crews in `schedule_from_permutation` |
| `MU` | **1.0** | `F = μF1 + (1−μ)F2`; **1.0 ⇒ accessibility-only** (F2 computed, weight 0) |
| `CAP_RETAIN` | **{1:0.3, 2:0.1, 3:0.02}** | damaged capacity = retain × capacity (`build_damaged_edges`) |
| `SPEED_RETAIN` | **{1:0.5, 2:0.3, 3:0.2}** | damaged free_flow_time = t0 / retain (`build_damaged_edges`) |
| `SEVER_SEVERITY` | 3 | severity ≥ this ⇒ edge fully removed (disconnection → `u_pen`) |
| `F1_ACTIVE_ONLY` | False | average F1 over full `[1,T]` (matches MILP `c_e^k` 1/T norm) |
| `RHO` | 0.7 | recovery inertia: `D_t = max(B v_t, ρ D_{t-1})` |
| `KAPPA` | 1.0 | damage→shortfall scale in `B[r,e]=κ(h_r0/3)·1{on SP}` |
| `UPEN_FACTOR` | 10.0 | `u_pen = 10 × max finite baseline OD travel time` |
| `M_SCENARIOS` | 10 | number of duration scenarios |
| `SEED` | 42 | RNG seed for `sample_scenarios` (deterministic) |
| `N_DISRUPTED_ORACLE` | 4 | disruption-set size (4! = 24 schedules); keys `n{N}/` cache folder |
| `UE_RGAP` | 1e-6 | per-step UE relative-gap target (looser than 1e-12 validation) |
| `UE_MAX_ITER` | 100 | per-step UE iteration cap |
| `DURATION_SUPPORT` | Table 1 (per (class,severity)) | base duration draws in `sample_scenarios` |
| `ETA` | [0.8, 1.0, 1.2] (given) | crew-efficiency multiplier draw |

> **Discrepancy fixed vs. the old `oracle_validation.md`:** that file listed `CAP_RETAIN={1:0.7,2:0.4,3:0.1}`, `SPEED_RETAIN={1:0.8,2:0.6,3:0.4}`, and `μ=0.5`. The live `config.py` now uses the harsher retains above and `MU=1.0`. The values in this table reflect the actual code as of this note.

**Non-fingerprint config** (present in `config.py` but not read by `run_oracle`; used by the later MILP): `MILP_MAX_ITER=20`, `MILP_CYCLE_TOL=3`, `MILP_DAMPING=0.5`.

---

# Appendix B — validation levels & comparison metrics (preserved) <a id="appB"></a>

Carried over from the retired `oracle_validation.md`. The oracle computes the **true** hindsight optimum `F*` per scenario by enumerating all work-conserving schedules and scoring each with the exact Figure-1 objective `F(x|ω)`.

**Validation levels — A (surrogate) vs B (true).** When the pretraining MILP is built later, it can be checked at two levels:
- **Level A — formulation correctness.** Brute-force the MILP's *own surrogate* objective (fixed travel times + linearized F2) and confirm the MILP returns the same minimizer. Isolates bugs in the MILP encoding (constraints, `c_e^k`, makespan linearization).
- **Level B — approximation quality.** Brute-force the *true* `F` (full UE pipeline — what `run_oracle` does) and compare `F(x_milp*)` to the true optimum `F*`. Tests whether traffic-fixation + linearization actually yield a (near-)optimal schedule for the real problem.
- **Current decision:** the oracle does **Level B only**. Level A is parked until the MILP exists.

**Comparison metrics (future MILP vs oracle).** All computable from `oracle_landscape.csv` (every tested `x` with its `F/F1/F2`):
- **objective gap:** `F(x_milp*) − F*` (relative) — the headline acceptance number.
- **rank:** where `x_milp*` falls among all enumerated schedules ("#1 of 24" / "top X%").
- **schedule match:** does `x_milp* == x*`? (If not but `F` equal → alternative optima; compare `F`, not the argmin.)
- **landscape shape:** sharp needle vs broad plateau (sets a reasonable MILP acceptance tolerance).

---

# Appendix C — Figure-1 fidelity & modeling caveats (preserved) <a id="appC"></a>

Carried over from the retired `oracle_validation.md`. `util/evaluate.py` follows Figure 1 step-for-step; deviations only where Figure 1 is silent/incomplete:
- **F2 is computed outside the per-step loop** (needs no UE — pure schedule arithmetic).
- **Baseline `u_r^{t0}`** = one UE on the undamaged network with normal demand `H^{t0}` (Fig. 1 uses it in F1 but does not show how it is obtained).
- **F1 can dip below 1.** F1 is demand-weighted and demand drops after the disaster, so fewer travellers experience the congested normal-demand baseline `u_r^{t0}`; during the low-demand recovery window the demand-weighted ratio can fall below 1. The paper's "=1 when fully restored" holds only once *both* network and demand are back to normal. Faster restoration still lowers F more (more time in the restored regime), so the optimization signal is intact.

**Demand model (drop → recover).** Open-source OD = normal-time demand `H^{t0}` (also the fully-recovered level). The literal `H_t = A·H_{t-1} + B·v_t` with `A=ρI` decays to 0 (wrong). The code models the **shortfall**:
```
D_t = max(B·v_t, ρ·D_{t-1}) ,   H_t = max(0, H^{t0} − D_t) ,   D_0 = 0
B[r,e] = κ·(h_r^{t0}/3)·1{e on a free-flow shortest path of OD r}
```
→ sharp drop at onset (to the current damage-driven shortfall `B·v`), then gradual recovery to normal at rate ρ as roads heal. Tune κ for depth, ρ for recovery speed (inspect `figures/03_best_schedule.png`).

**Caveats.**
- **Work-conserving reduction.** Schedules are permutations → list scheduling (no idling), assuming idling never helps F1/F2. With dynamic demand the demand-coupling makes this only *approximately* guaranteed — re-check if results look off. Keeps enumeration at `|ℰ|!`.
- **No UE cache.** Dynamic demand ⇒ UE depends on the recovery path ⇒ the `2^|ℰ|` completed-set cache does not apply; **every slot is a fresh UE solve**.
- **AequilibraE per-call cost.** ~1+ s/UE-solve here (per-call setup overhead on a 24-node net) caps the brute force at a small instance (`N=4`, `M=10`). A fast in-house Frank-Wolfe would lift this if a larger instance is needed.
