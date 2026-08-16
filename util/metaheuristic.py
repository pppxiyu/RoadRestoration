"""
Metaheuristic baseline solvers for the road-restoration scheduling problem: a genetic algorithm (GA).
It searches the same space the greedy and MILP solvers do -- a
repair PRIORITY ORDER over the disrupted segments, turned into start times by the work-conserving list
schedule -- and are scored by the identical true objective F (a real user-equilibrium solve per slot),
so their results are directly comparable to the other methods.

Each search runs ONCE, against a nominal instance whose durations are the expected repair times
rounded to whole slots, and the single order it returns is then evaluated on every sampled
scenario. The decision is therefore committed before any duration is observed, which is what makes
the reported mean the value of a rule rather than an average of hindsight-optimal answers.

  ga  : order-based genetic algorithm. Individuals are permutations of the segments; selection is
        tournament, recombination is order crossover (OX, which preserves relative order and produces a
        valid permutation), mutation is a small number of random swaps, and the best few individuals
        survive unchanged each generation (elitism).

Because one F evaluation is expensive (T user-equilibrium solves, one per slot, and congested schedules drive
the solver to its iteration cap), two things make the search affordable:
  * PARALLEL evaluation across a process pool -- each worker builds the fixed context once and evaluates
    whole batches of candidate orders concurrently (UE is solved fully in memory, so there is no shared
    state to contend on);
  * per-scenario MEMOIZATION -- a candidate order is evaluated at most once; elitist survivors and
    repeated offspring cost nothing.
The search is run under a fixed budget of UNIQUE evaluations, so every method is compared at a
controlled compute level.

Both populations are SEEDED with the three static greedy orders (flow / demand / ratio) plus random
fills; with elitism this guarantees each metaheuristic finishes no worse than the best greedy baseline,
making it a meaningful "can search improve on the heuristic?" baseline rather than a blind sample of a
13!-sized space. Set seed_greedy=False for pure random initialization.

Each variant writes outputs/greedy/n{N}/{variant}_optima.csv, the same schema and directory the static
greedy solvers use, so util/compare.py discovers them automatically, beside {variant}_slots.csv
holding the per-slot accessibility of those same evaluations, so a recovery curve costs no extra UE
solve. The search's own record stays in outputs/{variant}/n{N}/: {variant}_trace.csv, the
search-process figure drawn from it, and a run_meta.json naming the instance, hyperparameters,
delivered order and stopping condition.

Run inside the road_restore conda env (PYTHONPATH = project root); above 8 segments compute_horizon
switches to the Graham bound by itself, so a large-n run only needs the N_DISRUPTED_ORACLE override:
  python -m util.metaheuristic                 # ga at the configured scale
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import config as P
from util.evaluate import build_context, evaluate_schedule, schedule_from_permutation
from util.oracle import (_baseline_twoway_flow, compute_horizon, scale_dir,
                         select_oracle_instance)
from util.provenance import (fresh_scale_dir, log_dir, results_dir, slot_rows,
                             write_run_meta)
from util.scenarios import nominal_durations, sample_scenarios

ROOT = Path(__file__).resolve().parent.parent
TOY = ROOT / "data" / "siouxfalls_toy"
OUT = ROOT / "outputs" / "1-baselines" / "rule-based"   # legacy out_dir default (instance print only)
OUT_DIAG = ROOT / "outputs" / "1-baselines"   # each variant owns outputs/1-baselines/{variant}/n{N}/


# --------------------------------------------------------------------------- #
# Parallel evaluation worker (module-level so Windows "spawn" can import it)
# --------------------------------------------------------------------------- #
_W = {}


def _worker_init(toy_dir, disrupted):
    """Pool-worker setup: build the fixed evaluation context once per process (one baseline UE solve),
    then reuse it for every candidate this worker scores."""
    _W["ctx"] = build_context(toy_dir, disrupted)


def _worker_eval(task):
    """Evaluate one candidate order in a worker: (perm, durations, T) -> (F, F1, F2). The order is
    turned into start slots by the same work-conserving schedule the other solvers use, then scored by
    the exact objective."""
    perm, durations, T = task
    try:
        start = schedule_from_permutation(list(perm), durations)
        res = evaluate_schedule(start, durations, T, _W["ctx"])
        return float(res["F"]), float(res["F1"]), float(res["F2"])
    except Exception:
        return 1e6, 1e6, 1e6         # a single failed evaluation must not abort a multi-hour run; a sentinel this large is never selected as best


# --------------------------------------------------------------------------- #
# Budgeted, memoized, parallel fitness cache (shared by GA and PSO)
# --------------------------------------------------------------------------- #
class FitnessCache:
    """Evaluate candidate orders under a fixed budget of UNIQUE evaluations, in parallel, once each.

    `evaluate(perms)` returns each order's F, submitting only orders not seen before to the pool.
    Once `n_evals` reaches `budget`, novel orders are no longer evaluated -- their fitness falls back
    to +inf so the search stops improving through them, which bounds total compute deterministically."""

    def __init__(self, pool, durations, T, budget):
        self.pool, self.durations, self.T, self.budget = pool, durations, T, budget
        self.cache = {}                                     # perm tuple -> (F, F1, F2)

    @property
    def n_evals(self):
        return len(self.cache)

    def evaluate(self, perms):
        todo, budget_left = [], self.budget - len(self.cache)
        for p in perms:
            if p not in self.cache and p not in todo and budget_left > 0:
                todo.append(p)
                budget_left -= 1
        if todo:
            tasks = [(p, self.durations, self.T) for p in todo]
            for p, r in zip(todo, self.pool.map(_worker_eval, tasks)):
                self.cache[p] = r
        return {p: (self.cache[p][0] if p in self.cache else float("inf")) for p in perms}

    def best(self):
        p = min(self.cache, key=lambda k: self.cache[k][0])
        return p, self.cache[p]                             # (perm, (F, F1, F2))


# --------------------------------------------------------------------------- #
# Plateau-based stopping (shared by GA and PSO)
# --------------------------------------------------------------------------- #
class PlateauStop:
    """Stop a population search once it has visibly levelled off, rather than when a preset amount
    of compute is spent. After a warm-up floor (gen >= gen_min), EITHER plateau signal ends it:

      (a) since_improve >= patience_P -- patience_P consecutive generations without a MEANINGFUL
                                 improvement to the best F found so far (an improvement counts
                                 only if
                                 it exceeds a relative tolerance `tol`, so a trickle of negligible
                                 gains does not keep the search alive);
      (b) stall >= stall_K    -- stall_K consecutive generations produced NO candidate the search
                                 had not already evaluated, i.e. the population has collapsed onto
                                 orders it already knows.

    These are OR-ed, not AND-ed. An earlier version required BOTH, which never stopped a mutating
    GA: crossover/mutation almost always emits at least one unseen order per generation, so `stall`
    resets to 0 forever and the run only ended at the evaluation cap -- on the n=10 GA the best F
    converged by generation ~50 but the search ground on to generation 309 (since_improve=259)
    before it was killed. The best-F plateau signal (a) is the true convergence test; patience_P
    is set well above the largest real gap between improvements so a slow search is not cut short.

    The counters are in GENERATIONS, which is the natural iteration of these methods; note one
    generation costs up to pop_size evaluations early on but almost none after convergence, since
    the fitness cache serves repeated orders for free. The evaluation budget remains as a hard
    ceiling for the case where nothing ever settles."""

    def __init__(self, gen_min, stall_K, patience_P, tol=1e-3):
        self.gen_min, self.stall_K, self.patience_P = gen_min, stall_K, patience_P
        self.tol = tol                       # relative size an improvement must exceed to reset patience
        self.gen = self.stall = self.since_improve = 0
        self.best = float("inf")

    def update(self, evals_before, evals_after, best_F):
        """Advance one generation. `evals_before/after` are the cache's unique-evaluation counts
        around this generation; `best_F` is the best objective found so far, after it."""
        self.gen += 1
        self.stall = 0 if evals_after > evals_before else self.stall + 1
        # Only a MEANINGFUL improvement (beyond a relative tolerance) resets patience, so a run
        # creeping by negligible amounts is still recognized as converged.
        if self.best == float("inf") or best_F < self.best - self.tol * abs(self.best):
            self.best, self.since_improve = best_F, 0
        else:
            self.since_improve += 1

    def done(self):
        return (self.gen >= self.gen_min
                and (self.since_improve >= self.patience_P or self.stall >= self.stall_K))


def _gen_row(gen, fit, pop, F, stop):
    """One row of the optimization trace, describing where a generation stood.

    Both halves of the stopping rule are recorded alongside the objective, because the two things
    a reader wants from a search trace are whether the best solution so far is still improving
    and whether
    the population is still producing anything new. `cum_evals` flattening is precisely what the
    novelty half of PlateauStop is counting, and the spread between the generation's best and
    worst shows whether the population has collapsed onto one order.

    Candidates that arrive after the budget is spent score +inf, so the population statistics are
    taken over the finite entries only; `n_scored` says how many that was."""
    vals = [F[p] for p in pop if np.isfinite(F[p])]
    best_perm, (best_F, _, _) = fit.best()
    return dict(generation=gen, cum_evals=fit.n_evals, best_F=best_F,
                gen_best_F=min(vals) if vals else float("nan"),
                gen_mean_F=float(np.mean(vals)) if vals else float("nan"),
                gen_worst_F=max(vals) if vals else float("nan"),
                n_scored=len(vals), distinct=len(set(pop)),
                stall=stop.stall, since_improve=stop.since_improve,
                best_order="-".join(map(str, best_perm)))


# --------------------------------------------------------------------------- #
# Seeds: the static greedy priority orders (shared initialization)
# --------------------------------------------------------------------------- #
def _greedy_seed_orders(ctx, segments, durations, flow):
    """Three static priority orders (flow / demand / ratio) as permutations, used to seed both
    metaheuristics.

    All three match the util/greedy.py baselines, because the caller passes the NOMINAL durations
    the search itself optimizes against, and those are the same expected durations the ratio
    baseline divides by (up to the rounding to whole slots that the scheduler requires). Seeding
    from the static rankers is what makes each metaheuristic finish no worse than the best of
    them."""
    B, sev, dis = ctx["B"], ctx["severity_vec"], ctx["disrupted"]
    demand = {int(dis[j][0]): float(sev[j] * B[:, j].sum()) for j in range(len(dis))}
    ratio = {e: demand[e] / max(1, int(durations[e])) for e in demand}
    fscore = {int(eid): flow.get((min(u, v), max(u, v)), 0.0) for (eid, u, v, s) in dis}
    return [tuple(sorted(segments, key=lambda e: (-score[e], e)))
            for score in (fscore, demand, ratio)]


# --------------------------------------------------------------------------- #
# Genetic algorithm (order-based)
# --------------------------------------------------------------------------- #
def _ox(p1, p2, rng):
    """Order crossover (OX): copy a random contiguous slice of parent p1, then fill the remaining
    positions with the other segments in the order they appear in p2. Always yields a valid permutation
    that inherits absolute positions from one parent and relative order from the other."""
    n = len(p1)
    a, b = sorted(rng.sample(range(n), 2))
    child = [None] * n
    child[a:b + 1] = p1[a:b + 1]
    taken = set(p1[a:b + 1])
    fill = [x for x in p2 if x not in taken]
    it = iter(fill)
    for i in range(n):
        if child[i] is None:
            child[i] = next(it)
    return tuple(child)


def _swap_mutation(perm, rng, n_swaps):
    p = list(perm)
    for _ in range(n_swaps):
        i, j = rng.sample(range(len(p)), 2)
        p[i], p[j] = p[j], p[i]
    return tuple(p)


def _ga(fit, segments, seeds, rng, pop_size, elite, tour_k, p_cross, p_mut, max_swaps,
        gen_min, stall_K, patience_P, on_gen=None):
    """Run the order-based GA against the fitness cache `fit` until the search levels off (see
    PlateauStop) or the evaluation budget runs out, whichever comes first. Returns (stop, trace):
    the stopping state, so the caller can record which condition ended the run, and one row per
    generation describing how the search moved. The best order is read back from the cache."""
    seg = list(segments)
    pop = [tuple(s) for s in seeds][:pop_size]
    while len(pop) < pop_size:                                          # random fills
        pop.append(tuple(rng.sample(seg, len(seg))))
    F = fit.evaluate(pop)

    stop = PlateauStop(gen_min, stall_K, patience_P)
    trace = [_gen_row(0, fit, pop, F, stop)]                            # generation 0 = the seeded population
    while fit.n_evals < fit.budget and not stop.done():
        prev = fit.n_evals
        ranked = sorted(set(pop), key=lambda p: F[p])                 # dedup so elitism keeps DISTINCT best orders, not copies of one
        nxt = list(ranked[:elite])                                     # elitism: carry the best forward unchanged
        while len(nxt) < pop_size:
            def pick():                                                # tournament selection (lower F wins)
                cs = rng.sample(pop, min(tour_k, len(pop)))
                return min(cs, key=lambda p: F[p])
            a, b = pick(), pick()
            child = _ox(a, b, rng) if rng.random() < p_cross else a
            if rng.random() < p_mut:
                child = _swap_mutation(child, rng, rng.randint(1, max_swaps))
            nxt.append(child)
        pop = nxt
        F = fit.evaluate(pop)                                          # cached survivors are free; only novel children cost budget
        stop.update(prev, fit.n_evals, fit.best()[1][0])
        trace.append(_gen_row(stop.gen, fit, pop, F, stop))
        if on_gen is not None:                                          # live progress, optional
            on_gen(stop.gen, trace)
    return stop, trace


# --------------------------------------------------------------------------- #
# Run over M scenarios
# --------------------------------------------------------------------------- #
# Default hyperparameters (tuned lightly on the n=13 instance under the eval budget; see the writeup).
# The three stopping counters are in GENERATIONS; PlateauStop
# documents why both the novelty and the improvement signal are required.
_STOP_PARAMS = dict(gen_min=10, stall_K=12, patience_P=40)
GA_PARAMS = dict(pop_size=16, elite=4, tour_k=3, p_cross=0.9, p_mut=0.5, max_swaps=2, **_STOP_PARAMS)

# Ceiling on unique true evaluations per variant. Effectively OPEN: the plateau rule is what ends
# a run, and this cap is only the runaway guard behind it. It was briefly 60 (a shared-compute
# ceiling matched to the RL solver), but the recorded n=10 GA baseline (mean F 0.9389, run_meta
# budget=100000) plateau-stopped at 383 unique evaluations -- under a cap of 60 the same call
# would stop at ~6 generations and silently fail to reproduce the result on disk. The default
# must reproduce the recorded baseline; a tighter budget is a per-call override, not a default.
BUDGET_CAP = 100_000


# Generations between redraws of the search-process figure in the variant's n{N} folder, so a long
# open-budget search is readable as a figure while it runs instead of only at the end. Costs a
# fraction of a second and overwrites the same file the end-of-run render produces. Nothing else is
# written mid-run: the deliverable is written once, at the end, so results/ never holds a
# half-finished search.
FIGURE_REFRESH_EVERY_GEN = 3


def _rescore_committed(v, ctx, segments, scenarios, T, M, seed, nominal):
    """Re-evaluate a variant's ALREADY-COMMITTED order under the CURRENT objective and UE settings,
    without running the search again.

    This exists for a change of ruler: when the evaluation tolerance or the UE engine changes, every
    F on disk was measured with the old one and none of the methods are comparable until all are
    re-scored. For a search whose answer is a single priority order, re-running the search is not
    required to restore comparability -- the order is the deliverable, and re-scoring it is a few
    seconds against a search's minutes.

    WHAT IT REFUSES TO PRETEND. The search TRAJECTORY cannot be re-scored: {v}_trace.csv holds the
    per-generation best F the search actually saw, under the old ruler, and the search-process
    figure drawn from it is that same old measurement. Both are left untouched rather than silently
    reinterpreted, and run_meta records the previous run's objective block under `rescored_from` so
    the mixed provenance is visible in the file instead of living only in someone's memory. What IS
    refreshed: the delivered per-scenario evaluations, the per-slot recovery curves, and F_nominal
    (recomputed here, so the meta does not mix a new mean F with an old nominal one).

    The per-scenario compute accounting is carried over from the original search unchanged: the
    search really did cost that many evaluations, and a re-score does not make the answer cheaper
    to have found.
    """
    diag = scale_dir(OUT_DIAG / v)
    sol_path = results_dir(diag) / f"{v}_solution_best.json"
    prev_meta_path = diag / "config" / "run_meta.json"
    if not sol_path.exists():
        raise SystemExit(f"cannot re-score {v}: no committed solution at {sol_path} "
                         f"(run the search first)")
    sol = json.loads(sol_path.read_text(encoding="utf-8"))
    perm = [int(x) for x in sol["order"]]
    if sorted(perm) != sorted(segments):
        raise SystemExit(f"cannot re-score {v}: committed order {perm} is not a permutation of "
                         f"this instance's segments {segments} -- the solution on disk belongs to "
                         f"a different instance")
    prev = json.loads(prev_meta_path.read_text(encoding="utf-8")) if prev_meta_path.exists() else {}
    # Search provenance carried over verbatim: a re-score changes how the answer is MEASURED, never
    # what the search spent to find it.
    n_evals = int(prev.get("n_evals", sol.get("n_evals", 0)))
    generations = int(prev.get("generations", 0))
    outcome = str(prev.get("outcome", "unknown"))
    solve_ue = n_evals * T

    # F_nominal under the current ruler, so this file holds one ruler throughout.
    F_nom = evaluate_schedule(schedule_from_permutation(list(perm), nominal), nominal, T, ctx)["F"]

    rows, slots = [], []
    for m, dur in enumerate(scenarios):
        t_s = time.perf_counter()
        start = schedule_from_permutation(list(perm), dur)
        res = evaluate_schedule(start, dur, T, ctx, collect_traces=True)
        slots.extend(slot_rows(m, res))
        row = dict(scenario=m, F=res["F"], F1=res["F1"], F2=res["F2"],
                   time_s=time.perf_counter() - t_s, n_evals=n_evals, generations=generations,
                   outcome=outcome, ue_solves=solve_ue, ue_total=solve_ue / M + T,
                   order="-".join(map(str, perm)),
                   durations="-".join(str(int(dur[e])) for e in segments))
        for e in segments:
            row[f"start_{e}"] = start[e]
        rows.append(row)

    pd.DataFrame(rows).to_csv(results_dir(diag) / f"{v}_optima.csv", index=False)
    pd.DataFrame(slots).to_csv(log_dir(diag) / f"{v}_slots.csv", index=False)
    sol["F_nominal"] = float(F_nom)
    sol_path.write_text(json.dumps(sol, indent=2), encoding="utf-8")
    write_run_meta(diag, method=v, segments=segments, T=T, seed=seed, M=M, hp=GA_PARAMS,
                   budget=int(prev.get("budget", 0)), order="-".join(map(str, perm)),
                   n_evals=n_evals, generations=generations, outcome=outcome,
                   nominal_durations={int(e): int(nominal[e]) for e in segments},
                   F_nominal=float(F_nom), solve_s=float(prev.get("solve_s", float("nan"))),
                   rescored=dict(
                       note="delivered evaluations re-measured under the objective/UE settings in "
                            "this file; the search itself was NOT re-run",
                       search_from=prev.get("written_at"),
                       stale_artifacts=[f"log/{v}_trace.csv", f"{v}_search_process.png"],
                       rescored_from=prev.get("objective")))
    mean_F = float(pd.DataFrame(rows)["F"].mean())
    print(f"  [{v}] RE-SCORED committed order (search untouched: {n_evals} evals, "
          f"{generations} generations, {outcome})\n       order: {'-'.join(map(str, perm))}\n"
          f"       F(nominal)={F_nom:.6f}   mean F over {M} scenarios = {mean_F:.6f}", flush=True)
    return mean_F


def run_metaheuristic(variants=("ga",), toy_dir=TOY, out_dir=OUT, M=P.M_SCENARIOS,
                      seed=P.SEED, budget=BUDGET_CAP, workers=16, seed_greedy=True,
                      rescore=False, search_seed=None):
    """Search ONCE on a nominal instance, then evaluate the resulting priority order on all M
    sampled scenarios, writing outputs/greedy/n{N}/{variant}_optima.csv (schema shared with the
    static greedy solvers).

    The search optimizes against a single set of NOMINAL durations, each segment's expected repair
    time rounded to whole slots (util.scenarios.nominal_durations, the same vector the MILP uses,
    so the two are optimizing against the same world). The order it returns is then scheduled and
    scored separately on every sampled scenario, exactly as the static rankers are, so the reported
    mean is the expected performance of a single decision committed before any duration is
    observed.

    This replaces the earlier arrangement, in which the search was re-run per scenario with that
    scenario's durations already known. That produced M different orders, each chosen with
    hindsight, so its mean was an upper bound on what any implementable rule could reach rather
    than the value of a rule; it also cost M searches instead of one.

    A search ends when it levels off (PlateauStop) or when `budget` unique F evaluations on the
    nominal instance have been spent, whichever comes first. `workers` sizes the evaluation pool.

    `rescore=True` skips the search entirely and only re-measures each variant's already-committed
    order under the current objective/UE settings -- see _rescore_committed for what that does and
    does not refresh. Use it to restore comparability after a change of ruler without paying for a
    search whose answer would not change.

    TWO SEPARATE SEEDS, and they must stay separate. `seed` draws the M evaluation scenarios -- the
    frozen ruler every method in the study is scored on -- and is pinned to config.SEED; moving it
    silently rescores the whole comparison against a different sample. `search_seed` (default: the
    same value, i.e. the historical single-seed behavior) drives ONLY the search's own randomness,
    so repeated runs can measure how much of a result is the method and how much is the draw. They
    shared one parameter until 2026-08-12, which meant a multi-seed sweep would have compared five
    runs against five different rulers and called the spread "seed variance".
    """
    from multiprocessing import Pool
    import random as _random

    out_dir = scale_dir(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    disrupted = select_oracle_instance(toy_dir, P.N_DISRUPTED_ORACLE)
    segments = sorted(int(e) for e in disrupted["edge_id"])
    ctx = build_context(toy_dir, disrupted)
    scenarios = sample_scenarios(disrupted, M, seed)
    T = compute_horizon(segments, scenarios)

    if rescore:
        # No search, so no evaluation pool and no fresh_scale_dir: the search trace and its figure
        # are the only record of a run that is deliberately not being repeated.
        nominal = nominal_durations(disrupted, segments)
        print(f"instance: {len(segments)} segments {segments}; M={M}; horizon T={T}; "
              f"RE-SCORE ONLY (no search) for variants={list(variants)}", flush=True)
        t_all = time.perf_counter()
        for v in variants:
            _rescore_committed(v, ctx, segments, scenarios, T, M, seed, nominal)
        print(f"Re-scored in {(time.perf_counter() - t_all) / 60:.1f} min", flush=True)
        from util.compare import refresh_comparison
        refresh_comparison()
        return scale_dir(OUT_DIAG / variants[0])
    flow = _baseline_twoway_flow(toy_dir)
    nominal = nominal_durations(disrupted, segments)
    print(f"instance: {len(segments)} segments {segments}; M={M}; horizon T={T}; "
          f"variants={list(variants)}; budget={budget} evals; workers={workers}; "
          f"seed_greedy={seed_greedy}", flush=True)
    print(f"nominal durations (expected, rounded): { {e: nominal[e] for e in segments} }", flush=True)

    t_all = time.perf_counter()
    with Pool(workers, initializer=_worker_init, initargs=(toy_dir, disrupted)) as pool:
        seeds = _greedy_seed_orders(ctx, segments, nominal, flow) if seed_greedy else []
        for v in variants:
            # search randomness only -- the evaluation scenarios above were drawn from `seed`
            rng = _random.Random((seed if search_seed is None else search_seed) * 1000
                                 + (0 if v == "ga" else 1))
            fit = FitnessCache(pool, nominal, T, budget)
            t0 = time.perf_counter()
            # Live per-generation progress, so a long open-budget search is watchable rather than
            # only judged at the end (the trace is written when the run finishes). Overwrites a
            # one-line status file each generation; costs nothing.
            diag = scale_dir(OUT_DIAG / v)
            fresh_scale_dir(diag)            # a rerun replaces this variant's folder, no residue
            prog = log_dir(diag) / f"{v}_progress.txt"

            def _on_gen(gn, tr, _prog=prog, _v=v, _diag=diag):
                r = tr[-1]
                _prog.write_text(
                    f"generation {r['generation']}  best_F={r['best_F']:.4f}  "
                    f"gen_best={r['gen_best_F']:.4f}  evals={r['cum_evals']}  "
                    f"stall={r['stall']}  since_improve={r['since_improve']}\n", encoding="utf-8")
                if gn % FIGURE_REFRESH_EVERY_GEN == 0:
                    from viz.meta_viz import make_search_process
                    make_search_process(_diag, pd.DataFrame(tr), prefix=_v)

            if v != "ga":
                raise ValueError(f"unknown metaheuristic variant {v!r} (only 'ga' exists)")
            stop, trace = _ga(fit, segments, seeds, rng, on_gen=_on_gen, **GA_PARAMS)
            perm, (F_nom, _, _) = fit.best()
            solve_s, solve_ue = time.perf_counter() - t0, fit.n_evals * T
            print(f"  [{v}] nominal search: F(nominal)={F_nom:.4f}  {fit.n_evals} evals, "
                  f"{stop.gen} generations, {stop.done() and 'plateau' or 'budget_cap'}, "
                  f"{solve_s / 60:.1f} min\n       order: {'-'.join(map(str, perm))}", flush=True)

            # --- evaluate that one order on every sampled scenario ---
            rows, slots = [], []
            for m, dur in enumerate(scenarios):
                t_s = time.perf_counter()
                start = schedule_from_permutation(list(perm), dur)
                res = evaluate_schedule(start, dur, T, ctx, collect_traces=True)
                slots.extend(slot_rows(m, res))         # recovery curve, at no extra UE cost
                row = dict(scenario=m, F=res["F"], F1=res["F1"], F2=res["F2"],
                           time_s=time.perf_counter() - t_s,
                           n_evals=fit.n_evals, generations=stop.gen,
                           outcome="plateau" if stop.done() else "budget_cap",
                           # ue_total is the honest per-scenario compute: the one-off nominal
                           # search shared across the M scenarios it serves, plus this scenario's
                           # single evaluation.
                           ue_solves=solve_ue, ue_total=solve_ue / M + T,
                           order="-".join(map(str, perm)),
                           durations="-".join(str(int(dur[e])) for e in segments))
                for e in segments:
                    row[f"start_{e}"] = start[e]
                rows.append(row)
            # Optima + per-slot slots now live in the variant's OWN tree, outputs/{v}/n{N}/ (+ raw/),
            # beside its diagnostics -- mirroring outputs/pretrain_milp, outputs/rl and
            # outputs/rl_saa. compare.py discovers each method's optima from its own folder, so
            # nothing has to sit in the shared greedy pool any more.
            diag = scale_dir(OUT_DIAG / v)
            diag.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(results_dir(diag) / f"{v}_optima.csv", index=False)
            pd.DataFrame(slots).to_csv(log_dir(diag) / f"{v}_slots.csv", index=False)
            tr = pd.DataFrame(trace)
            tr.to_csv(log_dir(diag) / f"{v}_trace.csv", index=False)
            from viz.meta_viz import make_search_process
            make_search_process(diag, tr, prefix=v)
            # The committed solution as one small self-describing file: enough to reproduce
            # every delivered schedule (order + realized durations -> list schedule) without
            # touching the search again.
            (results_dir(diag) / f"{v}_solution_best.json").write_text(json.dumps(dict(
                order=[int(x) for x in perm], F_nominal=float(F_nom),
                found_generation=int(tr.loc[tr["best_F"] <= tr["best_F"].min() + 1e-12,
                                            "generation"].iloc[0]),
                n_evals=int(fit.n_evals)), indent=2), encoding="utf-8")
            write_run_meta(diag, method=v, segments=segments, T=T, seed=seed, M=M,
                           hp=GA_PARAMS, search_seed=int(seed if search_seed is None
                                                         else search_seed),
                           budget=budget, order="-".join(map(str, perm)),
                           n_evals=fit.n_evals, generations=stop.gen,
                           outcome="plateau" if stop.done() else "budget_cap",
                           nominal_durations={int(e): int(nominal[e]) for e in segments},
                           F_nominal=float(F_nom), solve_s=solve_s)
            print(f"  mean F [{v}] = {pd.DataFrame(rows)['F'].mean():.4f}", flush=True)

    print(f"Wrote {out_dir}  ({(time.perf_counter() - t_all) / 60:.1f} min)", flush=True)
    from util.compare import refresh_comparison
    refresh_comparison()                     # every solver run leaves the comparison current
    return out_dir


if __name__ == "__main__":
    vs = tuple(a for a in sys.argv[1:] if not a.startswith("-")) or ("ga",)
    run_metaheuristic(variants=vs)
