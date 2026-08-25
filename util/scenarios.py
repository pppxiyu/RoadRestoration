"""
Monte Carlo sampling of road-restoration duration scenarios.

Repair time is uncertain, so instead of committing to a single estimate we draw many equally
plausible "scenarios" and later optimize against the whole set. A scenario is one joint
realization of how long every disrupted road segment takes to repair, measured in discrete time
slots, returned as a dict {edge_id: duration_in_slots}.

THE LAW IS TWO-STAGE (2026-08-24, config.py's duration and severity blocks). Drawing one
scenario for one segment e is exactly:

    1. TRUE SEVERITY.   s_e ~ SEVERITY_CONFUSION[s_hat_e]        (s_hat_e = the instance's
                                                                  reported estimate)
    2. DURATION.        d_e = max(1, round(X_e)),
                        X_e ~ LogNormal(nu, tau^2) of the cell (road_class_e, s_e), truncated
                        above at DUR_TRUNC_MULT * mean and renormalized

Severity is drawn FIRST and the duration comes from the TRUE severity's cell, so the two stay
coupled -- a road that turns out to be badly damaged really does take longer. Every segment
draws independently, so two segments sharing an estimate realize different truths.

WHY SEVERITY IS RANDOM. It is not a second source of duration noise; it is what makes a
segment's IMPORTANCE scenario-dependent. The true severity sets the capacity and free-flow
retention of the damaged link, whether the link is severed outright (config.SEVER_SEVERITY, at
which point it leaves the routing network and can disconnect a zone), and the demand shortfall
the segment drives. With only durations random, every scenario agrees on which segment matters
most and adapting to the scenario is worth almost nothing -- measured at 0.0011 across
near-optimal orders. The true severity is NEVER revealed during execution: a planner sees the
estimate and this law, and still commits to a single repair order.

The estimate is what PLANNING reads (expected_durations and nominal_durations marginalize over
the confusion, so the nominal world is the planner's honest point estimate) and the truth is
what SCORING reads (a Scenario carries its severities; util.evaluate applies them). Everything
downstream -- the exact per-cell PMF, expectations, worst cases, the LHS stratification -- is
derived from _cell_pmf and _true_severity_pmf, so each law has exactly one implementation.
Setting every SEVERITY_CONFUSION row to a one-hot vector recovers the deterministic-severity
law exactly. Rationale in technical_notes/05-problem_redefinition.md.
"""

import math
from functools import lru_cache

import numpy as np

import config as P


def _cell_lognorm(cell):
    """The cell's lognormal parameters and truncation point: (nu, tau, upper).
    tau^2 = ln(1 + sd^2/mean^2), nu = ln(mean) - tau^2/2 gives the UNTRUNCATED distribution the
    target mean and sd; the upper truncation at DUR_TRUNC_MULT * mean then cuts ~0.5-1.5% of
    mass, so the realized mean sits slightly below the target (expected_durations reports the
    truth, computed from the PMF, never the target)."""
    m, sd = float(P.DUR_MEAN[cell]), float(P.DUR_SD[cell])
    tau2 = math.log(1.0 + (sd / m) ** 2)
    tau = math.sqrt(tau2)
    nu = math.log(m) - tau2 / 2.0
    return nu, tau, P.DUR_TRUNC_MULT * m


def _Phi(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


@lru_cache(maxsize=None)
def _cell_pmf(cell):
    """The EXACT probability mass function of a cell's rounded, truncated duration, as a tuple
    of (duration_in_slots, probability) pairs -- the single authority every derived quantity
    (sampling, expectations, worst cases, LHS strata, figures) reads the law through.

    d = k collects the truncated lognormal's mass on (k-0.5, k+0.5], with the k = 1 bin
    absorbing everything below 1.5 (the max(1, .) floor). Masses come from CDF differences of
    the lognormal, renormalized by the truncation point's CDF, so the PMF carries no sampling
    error and no seed dependence."""
    nu, tau, upper = _cell_lognorm(cell)
    cdf = lambda x: _Phi((math.log(x) - nu) / tau) if x > 0 else 0.0
    z_up = cdf(upper)
    if z_up <= 0.0:
        raise ValueError(f"degenerate duration cell {cell}: truncation point {upper} has zero mass")
    kmax = int(math.floor(upper + 0.5))
    out = []
    for k in range(1, kmax + 1):
        lo = 0.0 if k == 1 else cdf(k - 0.5)
        hi = cdf(min(k + 0.5, upper))
        p = (hi - lo) / z_up
        if p > 0.0:
            out.append((k, p))
    return tuple(out)


class Scenario(dict):
    """One realized world: the segments' repair DURATIONS (this object's own dict items, so every
    existing `dur[e]` call site keeps working unchanged) plus the realized TRUE SEVERITIES on
    `.sev`.

    A dict subclass rather than a tuple because a scenario is passed through dozens of call sites
    that index it as a duration map, and through a multiprocessing Pool. `__reduce__` is explicit
    so the severities survive pickling -- the default dict-subclass reduction would rebuild the
    object by calling the class with no arguments and silently drop them, which would score every
    worker's evaluations against the ESTIMATED severities instead of the true ones."""

    def __init__(self, durations, severities):
        super().__init__(durations)
        self.sev = {int(k): int(v) for k, v in severities.items()}

    def __reduce__(self):
        return (Scenario, (dict(self), self.sev))


@lru_cache(maxsize=None)
def _true_severity_pmf(s_hat):
    """P(true severity | reported estimate) as ((severity, probability), ...) -- the single
    authority for the confusion law, validated on read so a mistyped config row cannot silently
    skew every scenario."""
    row = P.SEVERITY_CONFUSION[int(s_hat)]
    if len(row) != 3 or any(p < 0 for p in row) or abs(sum(row) - 1.0) > 1e-9:
        raise ValueError(f"SEVERITY_CONFUSION[{s_hat}] = {row} is not a probability vector over "
                         f"severities 1, 2, 3")
    return tuple((k, float(p)) for k, p in zip((1, 2, 3), row) if p > 0.0)


def _estimates_of(disrupted):
    """{edge_id: (road_class, reported severity estimate)} -- what the instance table declares and
    what PLANNING is allowed to read."""
    return {int(r.edge_id): (r.road_class, int(r.severity))
            for r in disrupted.itertuples(index=False)}


def _draw_world(disrupted, u_sev, u_dur):
    """One segment-wise world from two uniform maps: severity first, then the duration from the
    TRUE severity's cell. Both samplers (i.i.d. and LHS) funnel through here, so the two-stage
    law has one implementation and the stages cannot drift apart."""
    est = _estimates_of(disrupted)
    sev = {e: _inverse_pmf(_true_severity_pmf(s_hat), u_sev[e]) for e, (_, s_hat) in est.items()}
    dur = {e: _inverse_pmf(_cell_pmf((cls, sev[e])), u_dur[e]) for e, (cls, _) in est.items()}
    return Scenario(dur, sev)


def _cells_of(disrupted):
    """{edge_id: (road_class, severity)} for the disrupted table -- the one place the cell of a
    segment is read off the instance."""
    return {int(r.edge_id): (r.road_class, int(r.severity))
            for r in disrupted.itertuples(index=False)}


def _inverse_pmf(pmf, u):
    """Discrete inverse CDF: the smallest duration whose cumulative probability exceeds u."""
    acc = 0.0
    for k, p in pmf:
        acc += p
        if u < acc:
            return k
    return pmf[-1][0]


def sample_scenarios(disrupted, M=P.M_SCENARIOS, seed=P.SEED):
    """Draw M restoration-duration scenarios for the disrupted segments.

    `disrupted` is a DataFrame with columns edge_id, road_class, and severity (the REPORTED
    estimate). Returns a list of M Scenario objects; scenario[m][edge_id] is that segment's
    realized repair duration in time slots and scenario[m].sev[edge_id] its realized TRUE
    severity, both drawn independently per segment under the two-stage law above. A fixed seed
    makes the whole sample reproducible.

    When config.EVAL_SAMPLING == "lhs" the draw is delegated to saa_lhs_sample: the SAME
    per-segment law, sampled by probability-stratified Latin Hypercube instead of i.i.d., so
    M=50 covers the uncertainty space evenly and estimates E[F] with lower variance. Flipping
    the flag is a change of ruler (see the config note): every method must be re-scored after.
    """
    if getattr(P, "EVAL_SAMPLING", "iid") == "lhs":
        return saa_lhs_sample(disrupted, M, np.random.RandomState(seed))
    rng = np.random.default_rng(seed)
    order = sorted(_estimates_of(disrupted))
    # Two independent uniforms per segment per scenario: one for the true severity, one for the
    # duration within that severity's cell. Iteration order is the (sorted) edge order, so a given
    # seed always yields the same sample.
    return [_draw_world(disrupted, {e: rng.random() for e in order},
                        {e: rng.random() for e in order}) for _ in range(M)]


def expected_durations(disrupted):
    """Expected repair duration of every disrupted segment under the FULL model, MARGINALIZED
    over the severity confusion: the segment's estimate names a row of SEVERITY_CONFUSION, and
    each possible true severity contributes its own cell's mean weighted by that probability.
    This is the PLANNING quantity -- what a here-and-now ranker may divide by and what the
    nominal world is built from -- and it is deliberately not the raw DUR_MEAN target (truncation
    and the floor shift the two apart) nor the estimate's own cell mean (that would pretend the
    assessment is exact). Computed by enumerating both PMFs, so it carries no sampling error and
    no seed dependence.

    Returns {edge_id: expected duration in slots} as floats. A ranker that divides by these
    values stays a here-and-now rule, because nothing about it depends on which realization
    occurs."""
    cell_mean = {}

    def _mean(cell):
        if cell not in cell_mean:
            cell_mean[cell] = sum(k * p for k, p in _cell_pmf(cell))
        return cell_mean[cell]

    # Marginalize over the severity the estimate might be hiding: the planner knows the confusion
    # law, so its honest point estimate averages each cell's mean against P(true severity | s_hat).
    return {e: sum(ps * _mean((cls, s_true))
                   for s_true, ps in _true_severity_pmf(s_hat))
            for e, (cls, s_hat) in _estimates_of(disrupted).items()}


def nominal_durations(disrupted, segments=None):
    """The single duration vector the searching solvers optimize against: each segment's expected
    repair time, rounded to whole slots.

    expected_durations is exact but generally fractional, whereas every solver that reasons about
    WHEN a crew is occupied needs whole slots. Exact halves are rounded UP rather than by Python's
    round-half-to-even, so a repair is never modelled as faster than it is expected to be.

    Every solver that searches on a nominal instance draws it from HERE, so the MILP and the
    metaheuristics are guaranteed to be optimizing against the same world; a second definition
    elsewhere would silently make their results incomparable. `segments` optionally restricts the
    result to the segments actually being scheduled.
    """
    exp_dur = expected_durations(disrupted)
    keys = exp_dur if segments is None else segments
    return {e: max(1, int(math.floor(exp_dur[e] + 0.5))) for e in keys}


def draw_durations(disrupted, rng):
    """One realized world (a Scenario: durations plus true severities) under the SAME two-stage
    law as sample_scenarios, independent per segment. Exists for
    training-time world sampling (the stochastically-trained RL draws realizations from the
    CALLER's rng); sample_scenarios stays untouched so the frozen evaluation sample can never
    drift with a trainer's draw order. Keep the two laws identical: both invert the same
    _cell_pmf, so a divergence is impossible without editing that single function."""
    order = sorted(_estimates_of(disrupted))
    return _draw_world(disrupted, {e: float(rng.uniform()) for e in order},
                       {e: float(rng.uniform()) for e in order})


def saa_lhs_sample(disrupted, n, rng):
    """n worlds by PROBABILITY-STRATIFIED Latin Hypercube Sampling -- a lower-variance stand-in
    for n i.i.d. draw_durations calls, for the fixed SAA training sample and (with
    config.EVAL_SAMPLING == "lhs", the current ruler) the frozen evaluation sample.

    BOTH stages of the law are stratified, and independently: per segment, the severity's [0,1)
    probability axis is split into n equal slices and the duration's likewise, one draw taken per
    slice, and each column PERMUTED on its own before the columns are paired into worlds.
    Stratifying the severity axis is what the two-stage law makes necessary -- an i.i.d. severity
    draw at n=50 routinely gives a rare-but-decisive outcome (a segment reported light turning
    out severed) two or three times more or less often than its probability, and since that
    outcome moves the objective by far more than any duration draw does, the resulting sample
    would misstate E[F] badly. The independent permutations keep segments and stages
    decorrelated, matching the law's independence.

    Drawn ONCE from the caller's rng and then held fixed by the trainer, so this stays SAA.
    """
    order = sorted(_estimates_of(disrupted))
    strat = {}
    for e in order:
        cols = []
        for _ in range(2):                       # one stratified column per stage
            u = (np.arange(n) + rng.rand(n)) / n
            rng.shuffle(u)                       # decorrelate this column from every other
            cols.append(u)
        strat[e] = cols
    return [_draw_world(disrupted,
                        {e: float(strat[e][0][i]) for e in order},
                        {e: float(strat[e][1][i]) for e in order})
            for i in range(n)]


def worst_case_durations(disrupted):
    """The longest duration each segment can realize under the full model: the top of the PMF
    support of the WORST severity its estimate can hide, so a segment reported light but truly
    severed is covered. Used to size horizons that no sampled world can overrun."""
    return {e: max(max(k for k, _ in _cell_pmf((cls, s_true)))
                   for s_true, _ in _true_severity_pmf(s_hat))
            for e, (cls, s_hat) in _estimates_of(disrupted).items()}
