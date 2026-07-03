# Road Restoration — project notes

Building a toy problem (Sioux Falls) to test the **pretraining solver** of a post-flood
road-recovery method (paper in `paper/`), then growing toward the full pipeline.

## Layout
- `data/siouxfalls_toy/` — toy instance (network, OD, disruption) + figures (+ its own README / TECHNICAL_NOTE)
- `util/` — engine: `ue.py` (UE assignment), `io.py` (loaders), `params.py`, `scenarios.py`, `evaluate.py` (Figure-1 F(x|ω)), `oracle.py` (brute force)
- `viz/` — visualizations of core-logic results (not the toy-data figures)
- `outputs/` — generated results (`UE_val/` = UE-validation proof; `oracle/` = brute-force optima/landscape/figures)
- `TECHNICAL_NOTE/` — parked design notes (e.g. `oracle_validation.md`)
- `main.py` — Figure-1 walkthrough (`python main.py`); brute-force oracle = `python -m util.oracle`

## Validate the UE engine
```
conda run -n road_restore python -m util.ue
```
Reproduces the open-source Sioux Falls solution and writes proof to `outputs/UE_val/`.

## Notes & gotchas (accumulating)
- System Python is 3.14 — AequilibraE won't install there (no wheel + no MSVC). Use `road_restore` (3.11).
- `conda run` can't take a multi-line `python -c` snippet — run a file or `python -m <module>`.
- Sioux Falls links are symmetric; the toy stores 38 **undirected** edges, but UE runs on the **directed** network (76 arcs).
- UE `Cost` = congested travel time (BPR value), not a price. Reference UE objective ≈ 4,231,335.
- Only `road_class` (capacity bands) and the disruption set are **fabricated**; everything else is open-source Sioux Falls data.
- Keep distinct: **classification level** (known category → restoration-time distribution) vs **restoration time** (random duration).
- AequilibraE is **~1.2 s per UE-solve** here (per-call setup overhead on a tiny net) — fine for one solve, but it caps brute-force loops; the oracle uses a small instance (4 segments, M=10) for this reason.
- Post-disaster **demand model** (shortfall): `D_t = max(B·v_t, ρ·D_{t-1})`, `H_t = max(0, H0 − D_t)` → sharp drop at onset, gradual recovery to normal. F1 is demand-weighted, so the drop can pull per-step F1 below 1 (fewer travellers than the congested baseline).
- Printed console text is ASCII (Windows console can't encode `ℰ/μ/Δ/→`); Unicode is fine inside figures.
