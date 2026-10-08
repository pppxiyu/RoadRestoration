# Sioux Falls toy instance — `siouxfalls_toy`

Toy problem for testing the **pretraining solver** of the post-flood road-recovery
method ([../../paper/main.tex](../../paper/main.tex)). **Step 1** builds only the
*static network instance* + the *disruption instance* and makes them easy to check.
Duration scenarios, the demand model, UE, and training labels are **not** here
(see "Deferred" below).

## Source & license

- **Source:** Transportation Networks for Research — Sioux Falls
  (<https://github.com/bstabler/TransportationNetworks>), `SiouxFalls/` folder.
- **Commit:** `d1639b4ef218c17928ba573e806ddf8ba5e7ae6d` · **accessed:** 2026-06-17.
- **License:** data sets are *for academic research purposes only*; the source
  must be cited in any publication. **Cite as:** *Transportation Networks for
  Research Core Team. Transportation Networks for Research.
  https://github.com/bstabler/TransportationNetworks. Accessed June 17, 2026.*
- The four `raw/SiouxFalls_*.tntp` files are kept unchanged as the original source.
  `raw/SiouxFalls_flow.tntp` (reference UE solution, best-known objective ≈ 4,231,335)
  is kept for checking a UE solver in a later step; it is unused now.

## Files

| File | Rows | Columns |
|---|---|---|
| `network/nodes.csv` | 24 | `node_id, x, y` (x,y = lon/lat from the node file) |
| `network/edges.csv` | 38 | `edge_id, u, v, capacity, length, free_flow_time, bpr_alpha, bpr_beta, road_class` — undirected (76 directed links collapsed to 38 symmetric pairs) |
| `network/od_pairs.csv` | 528 | `od_id, origin, destination, h0` — all OD pairs with positive pre-disaster demand `h0` (total 360,600 trips) |
| `instances/disrupted_segments.csv` | 8 | `edge_id, u, v, road_class, severity, level_id` — the source toy instance |
| `instances/disrupted_segments_oracle{n}.csv` | `n` | the frozen damaged-road instances used by experiments |

Generated source figures are under `outputs/studies/problem_setting/source_figures/`.
Historical `disruption/` and `figures/` directories were preserved in `legacy/`.

`road_class` (capacity bands) and the disruption instance are the only
made-up parts — their logic is in [`TECHNICAL_NOTE.md`](TECHNICAL_NOTE.md).

## Regenerate

```bash
# deps: numpy pandas networkx matplotlib
python make_instance.py
```

Seeded (`SEED = 42`); re-running produces identical CSVs. Larger instances:
edit `N_DISRUPTED` / `SEED` at the top of `make_instance.py`.

## Deferred to later steps (NOT in this folder yet)

- **Duration scenarios** — sampling per-level restoration durations (Table 1 × η).
- **Demand model** — inertia matrix `A`, damage→demand sensitivity `B`, penalty
  travel time `u_pen`.
- **Severity → capacity/speed reduction** map (UE-side physics).
- **UE simulator** + baseline/fixed travel times `u_r^{t_k}`.
- **Pretraining labels** — the per-scenario MILP solve and the `(s, a*)` pairs.
