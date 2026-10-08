"""Construct disrupted road instances and their common scoring horizon."""
from pathlib import Path

import numpy as np
import pandas as pd

from src import config as P
from src.experiment.retired import retired_problem
from src.problem.io import load_toy_network, od_to_matrix
from src.environment.ue import solve_ue

def _baseline_twoway_flow(toy_dir, cores=None):
    """Compute the baseline traffic flow on each undirected edge, used only to rank edges by
    importance when choosing which ones to disrupt.

    The flow comes from a user-equilibrium (UE) solve — the traffic state in which no driver can
    lower their own travel time by unilaterally switching route — run by our own UE engine on the
    UNDAMAGED network under the base origin-destination (OD) demand H0. The two directed volumes on
    an edge are summed into a single undirected flow. Returns {(min(u,v), max(u,v)): flow}.
    `cores` pins the solve's thread pool (None keeps the engine default; the RL solver passes 1
    for bit-reproducibility, see src.environment.ue.solve_ue).

    Computing the flow here rather than reading a shipped reference solution keeps the pipeline
    self-contained: swapping in a different network or OD dataset needs no external reference-flow
    file. A separate self-check inside the UE engine still validates our UE against that reference
    solution, and is left untouched."""
    edges, od, zone_ids = load_toy_network(toy_dir)
    # Tight DEFINITION tolerance: this flow ranks the edges that select the instance and seeds the
    # RL flow prior, so it must be stable regardless of the loosened per-slot evaluation tolerance
    # (config UE_RGAP_DEF vs UE_RGAP). Loosening it once reordered the ranking and changed which
    # segments the n=6 instance picked.
    flows, _ = solve_ue(edges, od_to_matrix(od, zone_ids), zone_ids,
                        rgap=P.UE_RGAP_DEF, max_iter=P.UE_MAX_ITER_DEF, quiet=True, cores=cores)
    f = {}
    fa, ta, vol = (flows["from"].to_numpy(), flows["to"].to_numpy(), flows["volume"].to_numpy())
    for a, b, v in zip(fa, ta, vol):
        key = (min(int(a), int(b)), max(int(a), int(b)))
        f[key] = f.get(key, 0.0) + float(v)
    return f


@retired_problem
def select_oracle_instance(toy_dir, n=None):
    """Choose which n segments to disrupt, ranking edges by importance (their baseline UE flow
    from _baseline_twoway_flow). The selection deliberately mixes heavily used and lightly used
    links so that the ORDER of restoration has a large effect on the objective: if every disrupted
    link mattered equally, any repair order would score about the same and the instance would be a
    weak test. The two highest-flow edges are marked severity 3 (fully severed, the most damaged
    state); the remaining picks are spread across lower-flow edges at severity 2 or 1
    (progressively lighter damage) -- except at n >= 16, which since 2026-08-26 uses the
    amplified-randomness recipe documented inline below (project owner's instruction).
    The choice is deterministic, and it is NOT tailored to the
    crew-accessibility constraint: that constraint is part of the scheduling model
    (src.environment.evaluate), and whether it binds on a given instance is an empirical property, not a
    design input. Writes disrupted_segments_oracle{n}.csv and returns the resulting DataFrame."""
    n = P.N_DISRUPTED_ORACLE if n is None else int(n)
    toy = Path(toy_dir)
    edges = pd.read_csv(toy / "network" / "edges.csv")
    flow = _baseline_twoway_flow(toy)
    edges["flow"] = [flow.get((min(int(r.u), int(r.v)), max(int(r.u), int(r.v))), 0.0)
                     for r in edges.itertuples(index=False)]
    ranked = edges.sort_values("flow", ascending=False).reset_index(drop=True)

    n_crit = min(2, n)
    picks = list(range(n_crit))                                  # highest-flow "critical" edges
    rest = n - n_crit
    if n >= 16:
        # AMPLIFIED-RANDOMNESS recipe for the large instance (2026-08-26, project owner's
        # instruction: "适当放大随机性的影响" -- moderately amplify what the scenario draw can
        # change). The LAW is untouched -- same duration cells, same severity confusion -- the
        # amplification is entirely in WHICH segments carry it: (a) the non-critical picks come
        # from the upper half of the flow ranking instead of reaching down to the quietest
        # edges, so a severity surprise lands on a road whose loss actually moves the
        # objective; (b) estimates skew 2:1 toward severity 2 -- the confusion matrix's
        # maximum-uncertainty row (15% truly severed, 15% milder than reported) -- instead of
        # alternating evenly. Both raise the truth-vs-estimate interaction without touching the
        # small-n recipe below, which stays exactly the pre-2026-08-26 rule. Live at n17 and n23;
        # n16, the size it was written for, was deleted 2026-08-27.
        if rest > 0:
            # The window is the upper half of the ranking, WIDENED when that half holds fewer
            # distinct positions than there are picks to place. Without the widening the rounded
            # linspace silently repeats indices and the instance comes out SMALLER than asked
            # (measured: n=23 wanted 21 non-critical picks from the 18 positions in [2, 19] and
            # produced a 20-segment instance). max() leaves every size the half already fits
            # untouched, so n17 is bit-for-bit what it was.
            hi = max(len(ranked) // 2, n_crit + rest - 1)
            picks += [int(round(x)) for x in np.linspace(n_crit, hi, rest)]
        sev_rest = [2 if i % 3 != 2 else 1 for i in range(rest)]
    else:
        if rest > 0:                                             # spread remaining picks over lower-flow edges
            picks += [int(round(x)) for x in np.linspace(len(ranked) // 5, len(ranked) - 1, rest)]
        sev_rest = [2 if i % 2 == 0 else 1 for i in range(rest)]
    # A repeated index would hand back an instance of fewer than n segments, and since every
    # runner reads n OFF the instance the run would look perfectly consistent while answering a
    # different question. Fail loudly instead.
    assert len(set(picks)) == n, (
        f"instance selection produced {len(set(picks))} distinct segments for n={n}: {picks}")
    sub = ranked.iloc[picks].copy().reset_index(drop=True)
    sub["severity"] = [3] * n_crit + sev_rest
    sub["level_id"] = sub["road_class"] + "-S" + sub["severity"].astype(str)
    out = (sub[["edge_id", "u", "v", "road_class", "severity", "level_id"]]
           .sort_values("edge_id").reset_index(drop=True))
    instance_dir = toy / "instances"
    instance_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(instance_dir / f"disrupted_segments_oracle{n}.csv", index=False)
    return out


def compute_horizon(segments, scenarios):
    """Return the global time horizon T: an upper bound on the completion slot of ANY priority
    order under any of the given scenarios, so no schedule is ever truncated and all schedules
    are scored over one identical time window.

    Under the accessibility constraint (2026-08-24 redesign) the classical Graham bound no
    longer applies: a crew can be forced to idle while the only reachable segments are already
    under repair, so a gated schedule can run longer than any work-conserving one. What always
    holds instead is full serialization -- whenever a crew idles, the frontier it is waiting on
    is under repair by another crew, so work never stops entirely and the last completion is at
    most 1 + sum_e d_e (repairs start at slot 1, not time zero). That serial bound is exact
    coverage for the pathological order and an overshoot for good ones; the overshoot costs
    tail slots in which the network is already repaired, which every method pays identically."""
    T = 0
    for dur in scenarios:
        T = max(T, 1 + sum(int(dur[e]) for e in segments))
    return T


