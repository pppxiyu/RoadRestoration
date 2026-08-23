"""
Static user-equilibrium (UE) traffic assignment for the toy network.

User equilibrium (UE) is the traffic state in which no driver can lower their own
travel time by unilaterally switching route; this is Wardrop's first principle of
route choice. The numerical work runs in this module's own bi-conjugate Frank-Wolfe
implementation (see WHY IN-HOUSE below), so the method is spelled out in plain terms
here. `solve_ue` is the single entry point.

    INPUT
      - A road network in which each link has a free-flow travel time t0 (the time to
        traverse it when empty), a capacity c, and BPR parameters (alpha, beta).
      - An origin-destination (OD) demand matrix h giving the number of trips per
        period from each origin zone to each destination zone.

    CORE LOGIC  (Frank-Wolfe user equilibrium)
      The UE flow pattern is the unique minimizer of the Beckmann objective
            Z(x) = sum_a  integral_0^{x_a} t_a(w) dw,
      where each link's cost rises with congestion through the BPR (Bureau of Public
      Roads) travel-time function
            t_a(x) = t0_a * (1 + alpha * (x_a / c_a)^beta).
      Frank-Wolfe is an iterative descent method that minimizes Z by repeating:
        1. ALL-OR-NOTHING loading: holding the current link costs fixed, assign every
           OD trip entirely to its current shortest path, yielding an auxiliary
           target flow y.
        2. MOVE: shift the current flow x a fractional step toward y so that Z
           decreases (x <- x + step * (y - x)); the step size is picked by a line
           search along the segment from x to y.
        3. RECOMPUTE link costs from the new flows, and stop once the RELATIVE GAP
           (how far x still sits from its all-or-nothing target, i.e. the remaining
           room for improvement) drops below rgap_target.
      The variant implemented here is BI-CONJUGATE Frank-Wolfe ("bfw"), which builds
      each step direction from the current all-or-nothing target and the previous TWO
      directions, chosen conjugate with respect to the Hessian diag(t'(x)). Plain
      Frank-Wolfe zig-zags near the optimum and converges sublinearly; the conjugate
      directions are what remove that.

    WHY IN-HOUSE (replaced AequilibraE on 2026-08-11)
      Two reasons, in this order:
        * SPEED. On a 24-node network AequilibraE's per-call graph and matrix container
          setup dominates the arithmetic. Measured on the published Sioux Falls instance,
          same algorithm and same tolerance: 28 ms here against 241 ms, and at rgap 1e-6,
          605 ms against 3,982 ms.
        * WARM STARTS. AequilibraE seeds every solve with a free-flow all-or-nothing
          loading and exposes no way to supply an initial flow (`add_preload` adds fixed
          background traffic, which is a different thing). Consecutive slots of a recovery
          differ by about 1% of demand, so the previous slot's equilibrium is 15-35x closer
          to the answer than free flow. `solve_ue(x0=...)` makes that available, and both
          per-slot evaluators (util.evaluate.evaluate_schedule and the RL's
          _evaluate_prefix_cached) chain consecutive slots through it when config.UE_WARM_START
          is on, seeding each solve via `warm_start_seed` below. Beyond speed, chaining makes
          the leftover truncation error CONSISTENT between consecutive slots, so it largely
          cancels in the slot-to-slot changes of the accessibility term g that the recovery
          curve is made of -- the tolerance audit (python -m util.ue_audit) measures exactly
          this and is what selected the production tolerance in config.py.
      Correctness was established against the PUBLISHED equilibrium shipped with the
      TransportationNetworks Sioux Falls instance -- an external reference, independent of
      either solver. Converged, this implementation lands a mean 0.7 vehicles per link from
      it (AequilibraE: 1.2), max 4.5 (7.8). `python -m util.ue` re-runs that check.

    OUTPUT
      - For each directed link, the equilibrium flow `volume` and the resulting
        congested travel time `cost`, returned as a tidy
        DataFrame[from, to, volume, cost], alongside a small convergence record.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

_CONJ_MAX = 0.99999      # cap on the conjugate coefficient, as in the reference bfw implementation


class _Network:
    """Directed-arc view of one network state plus its OD demand, and the BFW solver over it.

    Each row of `edges` is one physical road treated as BIDIRECTIONAL: it becomes two directed
    arcs (u->v and v->u) that each carry the row's full capacity and BPR parameters, so the two
    directions congest independently. Every network node is registered as a zone, and paths may
    pass through zones rather than terminating at them. These conventions are the ones the rest of
    the project was built and validated against, so they are fixed here rather than configurable.
    """

    def __init__(self, edges, od_matrix, zone_ids):
        u = edges["u"].to_numpy(dtype=np.int64)
        v = edges["v"].to_numpy(dtype=np.int64)
        cap = edges["capacity"].to_numpy(dtype=float)
        fft = edges["free_flow_time"].to_numpy(dtype=float)
        al = edges["bpr_alpha"].to_numpy(dtype=float)
        be = edges["bpr_beta"].to_numpy(dtype=float)
        self.tail = np.concatenate([u, v])
        self.head = np.concatenate([v, u])
        self.cap = np.concatenate([cap, cap])
        self.t0 = np.concatenate([fft, fft])
        self.alpha = np.concatenate([al, al])
        self.beta = np.concatenate([be, be])

        self.zones = np.asarray(zone_ids, dtype=np.int64)
        nodes = np.unique(np.concatenate([self.tail, self.head, self.zones]))
        self.pos = {int(n): i for i, n in enumerate(nodes)}
        self.n = len(nodes)
        self.ti = np.array([self.pos[int(x)] for x in self.tail])
        self.hi = np.array([self.pos[int(x)] for x in self.head])
        self.m = len(self.ti)

        M = np.asarray(od_matrix, dtype=float)
        zi = np.array([self.pos[int(z)] for z in self.zones])
        self.od_rows = []                       # (origin index, destination indices, demand)
        for a, z in enumerate(zi):
            nz = np.nonzero(M[a])[0]
            if len(nz):
                self.od_rows.append((z, zi[nz], M[a][nz]))
        self._arc = {}
        for idx in np.lexsort((self.t0, self.hi, self.ti)):
            self._arc.setdefault((self.ti[idx], self.hi[idx]), idx)

    def cost(self, x):
        """BPR congested travel time per directed arc."""
        return self.t0 * (1.0 + self.alpha * (x / self.cap) ** self.beta)

    def _derivative(self, x):
        """dt/dx of BPR, the diagonal Hessian the conjugate directions are taken against."""
        with np.errstate(divide="ignore", invalid="ignore"):
            d = self.t0 * self.alpha * self.beta * np.power(x, self.beta - 1.0) \
                / np.power(self.cap, self.beta)
        return np.nan_to_num(d, nan=0.0, posinf=0.0)

    def all_or_nothing(self, t):
        """Load every OD pair onto its shortest path under arc costs t.

        Loading walks the shortest-path TREE, not each path separately: put each destination's
        demand on its node, then sweep nodes in DECREASING distance from the origin, pushing every
        node's accumulated load onto the arc from its predecessor. A node is only reached after
        everything it feeds, so one sweep of n nodes loads all of an origin's paths exactly.
        Unreachable demand (a severed link can disconnect the network) is left unloaded, matching
        the convention that its travel time falls back to a penalty downstream.
        """
        g = csr_matrix((t, (self.ti, self.hi)), shape=(self.n, self.n))
        y = np.zeros(self.m)
        for origin, dests, dem in self.od_rows:
            dist, pred = dijkstra(g, directed=True, indices=origin, return_predecessors=True)
            load = np.zeros(self.n)
            ok = np.isfinite(dist[dests])
            np.add.at(load, dests[ok], dem[ok])
            finite = np.where(np.isfinite(dist))[0]
            for jj in finite[np.argsort(-dist[finite])]:
                q = load[jj]
                if q <= 0 or jj == origin:
                    continue
                pp = pred[jj]
                if pp < 0:
                    continue
                y[self._arc[(pp, jj)]] += q
                load[pp] += q
        return y

    def _line_search(self, x, dx):
        """Exact minimizing step along x + a*dx for a in [0, 1], from the sign of the objective's
        directional derivative phi(a) = dx . t(x + a*dx).

        phi is NON-DECREASING in a: its own derivative is sum(dx^2 * t'(.)) and BPR has t' >= 0.
        Monotonicity means the two endpoints alone decide the segment minimizer -- no descent at all
        if phi(0) >= 0, the far end if phi(1) <= 0, otherwise the unique interior root. Choosing
        between the endpoints by MAGNITUDE, as this once did, returns a zero step whenever
        |phi(0)| < |phi(1)| with both negative, abandoning a direction that was still descending.
        """
        phi = lambda a_: float(np.dot(dx, self.cost(x + a_ * dx)))
        if phi(0.0) >= 0.0:
            return 0.0
        if phi(1.0) <= 0.0:
            return 1.0
        return float(brentq(phi, 0.0, 1.0, xtol=1e-10))

    def solve(self, rgap, max_iter, x0=None):
        """Bi-conjugate Frank-Wolfe. Returns (flows, achieved relative gap, iterations).

        Direction schedule: a fresh history takes a plain FW step, then a conjugate (CFW) step,
        then bi-conjugate (BFW) steps; a line search that lands at step 1.0 has replaced the whole
        iterate, so the direction history is void and the cycle restarts. Convergence is the
        standard relative gap (t.x - t.y) / t.x, the excess of the current assignment's cost over
        putting every trip on its currently shortest path -- zero exactly at equilibrium.

        The conjugate coefficients are derived under a quadratic model of the objective, and where
        that model is poor (large mu, nu) the direction they build can point UPHILL. Such a
        direction voids the history and the iteration is RETRIED as a plain FW step, which always
        descends while the gap is positive -- its directional derivative is exactly -(gap * t.x).
        Without that fallback an uphill conjugate direction ended the solve and returned a
        non-equilibrium flow pattern indistinguishable from a converged one at the call site: on an
        n=10 recovery it hit 1 slot in 52, leaving the gap at 1.8e-2 against a 1e-6 target, and the
        same slot returned bit-identical flows whether asked for 1e-2, 1e-3 or 1e-6.

        How the loop exited stays recoverable from the return values, which `solve_ue` uses to
        raise: converged iff gap <= rgap, iteration-capped iff iterations == max_iter, stalled
        otherwise. A stall is unreachable while the fallback holds and the gap is positive.
        """
        x = self.all_or_nothing(self.t0) if x0 is None else np.asarray(x0, dtype=float).copy()
        sdr = psd = None
        step, do_fw, do_cfw = 1.0, True, False
        gap, it = np.inf, 0
        for it in range(1, max_iter + 1):
            t = self.cost(x)
            y = self.all_or_nothing(t)
            tx = float(np.dot(t, x))
            gap = abs(tx - float(np.dot(t, y))) / tx if tx > 0 else 0.0
            if gap <= rgap:
                break
            for attempt in (0, 1):
                if attempt == 1:                 # the conjugate direction pointed uphill: void the
                    sdr = psd = None             # history, retry the iteration as a plain FW step
                    do_fw, do_cfw = True, False
                if do_fw or sdr is None:
                    d = y
                    sdr, psd = y, None
                    do_fw, do_cfw = False, True
                elif do_cfw or psd is None:
                    der = self._derivative(x)
                    num = float(np.sum((sdr - x) * (y - x) * der))
                    den = float(np.sum((sdr - x) * (y - sdr) * der))
                    a = 0.0 if den == 0.0 else min(max(num / den, 0.0), _CONJ_MAX)
                    d = a * sdr + (1.0 - a) * y
                    sdr, psd, do_cfw = d, sdr, False
                else:
                    der = self._derivative(x)
                    x_ = sdr * step + psd * (1.0 - step) - x
                    y_, z_, w_ = y - x, sdr - x, psd - sdr
                    mu_den = float(np.sum(x_ * w_ * der))
                    mu = 0.0 if mu_den == 0.0 else max(0.0, -float(np.sum(x_ * y_ * der)) / mu_den)
                    nu_den = float(np.sum(z_ * z_ * der))
                    nu = 0.0 if (nu_den == 0.0 or step >= 1.0) else max(
                        0.0, -float(np.sum(z_ * y_ * der)) / nu_den + mu * step / (1.0 - step))
                    b0 = 1.0 / (1.0 + mu + nu)
                    d = b0 * y + (nu * b0) * sdr + (mu * b0) * psd
                    sdr, psd = d, sdr
                dx = d - x
                step = self._line_search(x, dx)
                if step > 0.0:
                    break
            if step <= 0.0:
                break                      # plain FW offers no descent either: numerically optimal
            x = x + step * dx
            if step >= 1.0:
                do_fw = True
        return x, gap, it

    def map_flows(self, links_prev):
        """A previous solution's [from, to, volume] aligned onto THIS state's arc vector, for
        warm starts. Arcs absent from the previous state start at zero flow."""
        x0 = np.zeros(self.m)
        idx = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(self.tail, self.head))}
        for f, t, vol in zip(links_prev["from"], links_prev["to"], links_prev["volume"]):
            i = idx.get((int(f), int(t)))
            if i is not None and np.isfinite(vol):
                x0[i] += float(vol)
        return x0


class _Convergence:
    """How a solve converged. Exposes `.report()` returning a DataFrame with iteration and rgap,
    the same accessor the previous AequilibraE assignment object offered, so any caller keeping
    the second return value keeps working."""

    def __init__(self, rgap, iterations, target):
        self.rgap, self.iterations, self.rgap_target = float(rgap), int(iterations), float(target)

    def report(self):
        return pd.DataFrame({"iteration": [self.iterations], "rgap": [self.rgap]})


def solve_ue(edges, od_matrix, zone_ids, algorithm="bfw", max_iter=1000, rgap=1e-10, quiet=False,
             cores=None, x0=None):
    """Solve static user equilibrium and return (flows_df, convergence).

    `flows_df` is a DataFrame[from, to, volume, cost] with one row per directed link, where
    `volume` is the equilibrium flow on that link and `cost` its congested travel time. The second
    value is a `_Convergence` record whose `.report()` gives the achieved relative gap and
    iteration count. The module docstring describes the underlying UE method.

    `edges` must supply the columns u, v, capacity, length, free_flow_time, bpr_alpha, bpr_beta.
    `max_iter` caps the iterations and `rgap` is the relative-gap tolerance at which the solve is
    considered converged.

    `x0` is the WARM-START hook: an initial arc-flow vector, normally a previous solution mapped
    through `_Network.map_flows`. None starts from the free-flow all-or-nothing loading -- the
    cold start used for a chain's first slot and whenever config.UE_WARM_START is off; with warm
    starting on, the per-slot evaluators seed every later slot. Note that a seed not feasible for
    THIS OD table makes the relative gap an invalid optimality bound -- it can report convergence
    early -- so a warm seed must be built to be feasible for the demand it is solved against.

    `algorithm` accepts only "bfw"; `quiet` and `cores` are retained so existing call sites keep
    working and are now no-ops (this solver prints nothing and is single-threaded, which is also
    what made it bit-reproducible for the RL argmax).
    """
    if algorithm != "bfw":
        raise ValueError(f"unknown algorithm {algorithm!r}; this solver implements 'bfw' only")
    net = _Network(edges, od_matrix, zone_ids)
    x, gap, iters = net.solve(rgap=rgap, max_iter=max_iter, x0=x0)
    # Only two ways out of the loop are legitimate: the gap reached the tolerance, or the
    # iteration cap -- an explicit configuration choice -- was spent. Anything else means the
    # solver gave up early, and returning that flow pattern as if converged is exactly the kind
    # of silent degradation no call site would ever notice (none of them inspect the convergence
    # record). It happened once: an uphill bi-conjugate direction ended solves 1.8e-2 short of
    # target until _Network.solve grew its plain-FW fallback. Unreachable now, enforced anyway.
    if gap > rgap and iters < max_iter:
        raise RuntimeError(
            f"UE solve stalled: no descent direction left at relative gap {gap:.3e} "
            f"(target {rgap:g}) after {iters} of {max_iter} iterations")
    flows = pd.DataFrame({"from": net.tail, "to": net.head, "volume": x, "cost": net.cost(x)})
    return flows, _Convergence(gap, iters, rgap)


def warm_start_seed(edges, od_delta, zone_ids, links_prev):
    """Feasible warm-start seed for `solve_ue(x0=...)`: the previous slot's equilibrium mapped
    onto THIS network's arcs, plus an all-or-nothing loading of the demand INCREMENT.

        x0  =  map(x_prev)  +  AON( t(map(x_prev)),  od_delta )

    Feasibility is the whole point. The relative gap is an optimality bound only on flows that
    actually route the demand being solved; seeding with a flow for yesterday's demand alone
    under-routes today's and lets rgap report convergence early (measured: a naive seed left g
    3.3e-3 wrong at a 1e-6 tolerance). The sum above IS feasible for the current demand because
    each part routes its own share: x_prev routes the previous slot's ROUTED demand, the
    all-or-nothing load routes the increment.

    `od_delta` must therefore be current demand minus the previous slot's routed demand -- for an
    OD pair that was disconnected last slot (its demand went unrouted), the pair's entire current
    demand belongs in the increment. The recovery guarantees the increment is non-negative
    (demand is monotone non-decreasing while damage clears) and that arcs only reopen, never
    vanish; callers assert the demand premise, and the flow-conservation check here catches the
    arc premise, so a violation raises instead of quietly producing an invalid bound."""
    net = _Network(edges, od_delta, zone_ids)
    x0 = net.map_flows(links_prev)
    vol = pd.to_numeric(links_prev["volume"], errors="coerce").to_numpy(dtype=float)
    prev_total = float(np.nansum(vol))
    if float(x0.sum()) < prev_total - 1e-6:
        raise RuntimeError(
            f"warm-start premise violated: {prev_total - float(x0.sum()):.3f} vehicles of the "
            f"previous equilibrium have no arc on the current network (arcs must only reopen)")
    return x0 + net.all_or_nothing(net.cost(x0))


def beckmann_objective(flows, linkp):
    """Evaluate the Beckmann objective Z(x) in closed form for a given flow pattern.

    Integrating the BPR link cost from zero flow up to x gives the per-link term
        t0*x + t0*alpha/(beta+1) * x^(beta+1) / cap^beta,
    and Z is the sum of these terms over all links. User equilibrium is the flow that
    minimizes this convex Z, so comparing Z between two flow patterns measures how
    close each is to equilibrium. `flows` has columns from, to, volume; `linkp` (from
    util.io.directed_link_params) supplies each link's capacity, free_flow_time,
    alpha, and beta.
    """
    m = flows.merge(linkp, on=["from", "to"], how="left")
    x = m["volume"].to_numpy()
    t0 = m["free_flow_time"].to_numpy()
    cap = m["capacity"].to_numpy()
    a = m["alpha"].to_numpy()
    b = m["beta"].to_numpy()
    return float(np.sum(t0 * x + t0 * (a / (b + 1.0)) * np.power(x, b + 1.0) / np.power(cap, b)))


# --------------------------------------------------------------------------- #
# Self-check: running this module as a script assigns UE on the Sioux Falls toy
# network and checks the result against its published reference solution, so a
# regression in the engine surfaces immediately.   ->   python -m util.ue
# --------------------------------------------------------------------------- #
def _validate():
    import sys
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from util.io import (directed_link_params, load_reference_flows,
                         load_toy_network, od_to_matrix)
    from viz.ue_val import save_validation

    toy = root / "data" / "siouxfalls_toy"
    out = root / "outputs" / "01-sim_val_n_problem_setting"
    obj_rel_tol, mean_rel_flow_tol, rgap_target = 1e-3, 0.01, 1e-12

    edges, od, zone_ids = load_toy_network(toy)
    M = od_to_matrix(od, zone_ids)
    linkp = directed_link_params(edges)
    print(f"Toy network: {len(zone_ids)} zones, {len(edges)} undirected edges, "
          f"{int(M.sum())} total trips")
    print("Running user-equilibrium assignment (in-house bi-conjugate Frank-Wolfe)...")
    ue, assig = solve_ue(edges, M, zone_ids, algorithm="bfw", max_iter=2000, rgap=rgap_target)

    ref = load_reference_flows(toy / "raw" / "SiouxFalls_flow.tntp")
    cmp = ue.merge(ref, on=["from", "to"], suffixes=("_ue", "_ref"))
    err = (cmp["volume_ue"] - cmp["volume_ref"]).abs()
    mean_flow, mean_abs_err, max_abs_err = cmp["volume_ref"].mean(), err.mean(), err.max()
    corr = float(np.corrcoef(cmp["volume_ue"], cmp["volume_ref"])[0, 1])
    z_ue, z_ref = beckmann_objective(ue, linkp), beckmann_objective(ref, linkp)
    obj_gap = abs(z_ue / z_ref - 1.0)
    report = assig.report()
    try:
        rgap = float(report["rgap"].iloc[-1])
    except Exception:
        rgap = float("nan")

    print("\n================ UE vs. ground truth ================")
    print(f"links compared        : {len(cmp)} (of {len(ref)})")
    print(f"final rgap            : {rgap:.2e}")
    print(f"mean |flow error|     : {mean_abs_err:,.2f}  ({mean_abs_err/mean_flow*100:.3f}% of mean)")
    print(f"max  |flow error|     : {max_abs_err:,.2f}")
    print(f"flow correlation      : {corr:.6f}")
    print(f"Beckmann obj UE / ref : {z_ue:,.1f} / {z_ref:,.1f}  (rel gap {obj_gap:.2e})")

    status = "PASS" if (obj_gap < obj_rel_tol and (mean_abs_err / mean_flow) < mean_rel_flow_tol) else "FAIL"
    print(f"\nMILESTONE: {status}")

    metrics = dict(corr=corr, mean_abs_err=mean_abs_err, max_abs_err=max_abs_err,
                   mean_flow=mean_flow, z_ue=z_ue, z_ref=z_ref, obj_gap=obj_gap,
                   rgap=rgap, rgap_target=rgap_target, status=status)
    saved = save_validation(out, toy, cmp, report, metrics)
    print(f"Visual validation written to: {saved}")
    return status == "PASS"


if __name__ == "__main__":
    _validate()
