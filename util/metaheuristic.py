"""
Metaheuristic baseline solvers for the road-restoration scheduling problem: a genetic algorithm (GA)
and particle-swarm optimization (PSO). Both search the same space the greedy and MILP solvers do -- a
repair PRIORITY ORDER over the disrupted segments, turned into start times by the work-conserving list
schedule -- and are scored by the identical true objective F (a real user-equilibrium solve per slot),
so their results are directly comparable to the other methods.

  ga  : order-based genetic algorithm. Individuals are permutations of the segments; selection is
        tournament, recombination is order crossover (OX, which preserves relative order and produces a
        valid permutation), mutation is a small number of random swaps, and the best few individuals
        survive unchanged each generation (elitism).
  pso : particle-swarm optimization with a random-key encoding. Each particle is a real vector in
        [0,1]^n; its permutation is read off by argsort (smallest key repaired first), which keeps the
        standard continuous velocity/position update valid while always decoding to a legal order.

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
greedy solvers use, so util/compare.py discovers them automatically.

Run inside the road_restore conda env (PYTHONPATH = project root); above 8 segments compute_horizon
switches to the Graham bound by itself, so a large-n run only needs the N_DISRUPTED_ORACLE override:
  python -m util.metaheuristic                 # ga and pso at the configured scale
  python -m util.metaheuristic ga pso
"""
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
from util.scenarios import sample_scenarios

ROOT = Path(__file__).resolve().parent.parent
TOY = ROOT / "data" / "siouxfalls_toy"
OUT = ROOT / "outputs" / "greedy"


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
    of compute is spent. Two plateau signals must hold together, plus a warm-up floor:

      (a) gen >= gen_min      -- do not judge a plateau in the first few generations;
      (b) stall >= stall_K    -- stall_K consecutive generations produced NO candidate the search
                                 had not already evaluated, i.e. the population has collapsed onto
                                 orders it already knows;
      (c) since_improve >= patience_P -- patience_P consecutive generations without improving the
                                 incumbent best F.

    Either signal alone misreads the search. Novelty can dry up for a stretch while the incumbent
    is still being improved by recombining known material, and the incumbent can sit unchanged for
    a long stretch while the population is still exploring productively -- on the n=10 RL run the
    longest gap between two incumbent improvements reached 55 iterations. Requiring both keeps the
    search alive exactly while either the population or the incumbent is still moving.

    The counters are in GENERATIONS, which is the natural iteration of these methods; note one
    generation costs up to pop_size evaluations early on but almost none after convergence, since
    the fitness cache serves repeated orders for free. The evaluation budget remains as a hard
    ceiling for the case where nothing ever settles."""

    def __init__(self, gen_min, stall_K, patience_P):
        self.gen_min, self.stall_K, self.patience_P = gen_min, stall_K, patience_P
        self.gen = self.stall = self.since_improve = 0
        self.best = float("inf")

    def update(self, evals_before, evals_after, best_F):
        """Advance one generation. `evals_before/after` are the cache's unique-evaluation counts
        around this generation; `best_F` is the incumbent objective after it."""
        self.gen += 1
        self.stall = 0 if evals_after > evals_before else self.stall + 1
        if best_F < self.best - 1e-12:
            self.best, self.since_improve = best_F, 0
        else:
            self.since_improve += 1

    def done(self):
        return (self.gen >= self.gen_min and self.stall >= self.stall_K
                and self.since_improve >= self.patience_P)


# --------------------------------------------------------------------------- #
# Seeds: the static greedy priority orders (shared initialization)
# --------------------------------------------------------------------------- #
def _greedy_seed_orders(ctx, segments, durations, flow):
    """The three static greedy priority orders (flow / demand / ratio) as permutations, used to seed
    both metaheuristics. Mirrors util/greedy.py's scoring so the seeds match those baselines exactly."""
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
        gen_min, stall_K, patience_P):
    """Run the order-based GA against the fitness cache `fit` until the search levels off (see
    PlateauStop) or the evaluation budget runs out, whichever comes first. Returns the stopping
    state so the caller can record which condition ended the run; the best order is read back from
    the cache afterwards."""
    seg = list(segments)
    pop = [tuple(s) for s in seeds][:pop_size]
    while len(pop) < pop_size:                                          # random fills
        pop.append(tuple(rng.sample(seg, len(seg))))
    F = fit.evaluate(pop)

    stop = PlateauStop(gen_min, stall_K, patience_P)
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
    return stop


# --------------------------------------------------------------------------- #
# Particle-swarm optimization (random-key encoding)
# --------------------------------------------------------------------------- #
def _decode(keys, segments):
    return tuple(segments[i] for i in np.argsort(keys, kind="stable"))


def _keys_for_order(order, segments):
    """Random keys whose argsort reproduces `order`: give the segment repaired at rank r the key
    (r+0.5)/n, so smaller key = earlier repair."""
    n = len(segments)
    pos = {e: i for i, e in enumerate(segments)}
    keys = np.zeros(n)
    for r, e in enumerate(order):
        keys[pos[e]] = (r + 0.5) / n
    return keys


def _pso(fit, segments, seeds, rng, swarm, w, c1, c2, gen_min, stall_K, patience_P):
    """Run random-key PSO against the fitness cache `fit` until the search levels off (see
    PlateauStop) or the evaluation budget runs out, whichever comes first. Particles are real
    vectors in [0,1]^n decoded to orders by argsort; velocity/position use the standard gbest
    update. Fitness is memoized on the DECODED order, so distinct keys mapping to the same order
    are free. Returns the stopping state."""
    seg = list(segments)
    n = len(seg)
    npr = np.random.RandomState(rng.randint(0, 2 ** 31 - 1))
    X = npr.rand(swarm, n)
    for i, s in enumerate(seeds[:swarm]):                              # seed particles decode to the greedy orders
        X[i] = _keys_for_order(s, seg)
    V = 0.1 * (npr.rand(swarm, n) - 0.5)

    def eval_rows(rows):
        perms = [_decode(rows[i], seg) for i in range(rows.shape[0])]
        Fmap = fit.evaluate(perms)
        return np.array([Fmap[p] for p in perms]), perms

    Fx, _ = eval_rows(X)
    pbest, pbestF = X.copy(), Fx.copy()
    g = int(np.argmin(pbestF))
    gbest, gbestF = pbest[g].copy(), float(pbestF[g])

    stop = PlateauStop(gen_min, stall_K, patience_P)
    while fit.n_evals < fit.budget and not stop.done():
        prev = fit.n_evals
        r1, r2 = npr.rand(swarm, n), npr.rand(swarm, n)
        V = w * V + c1 * r1 * (pbest - X) + c2 * r2 * (gbest[None, :] - X)
        X = np.clip(X + V, 0.0, 1.0)
        Fx, _ = eval_rows(X)
        improved = Fx < pbestF
        pbest[improved], pbestF[improved] = X[improved], Fx[improved]
        g = int(np.argmin(pbestF))
        if pbestF[g] < gbestF:
            gbest, gbestF = pbest[g].copy(), float(pbestF[g])
        stop.update(prev, fit.n_evals, gbestF)
    return stop


# --------------------------------------------------------------------------- #
# Run over M scenarios
# --------------------------------------------------------------------------- #
# Default hyperparameters (tuned lightly on the n=13 instance under the eval budget; see the writeup).
# The three stopping counters are shared by both methods and are in GENERATIONS; PlateauStop
# documents why both the novelty and the improvement signal are required.
_STOP_PARAMS = dict(gen_min=10, stall_K=12, patience_P=15)
GA_PARAMS = dict(pop_size=16, elite=4, tour_k=3, p_cross=0.9, p_mut=0.5, max_swaps=2, **_STOP_PARAMS)
PSO_PARAMS = dict(swarm=16, w=0.7, c1=1.5, c2=1.5, **_STOP_PARAMS)

# Ceiling on unique true evaluations per scenario per variant, kept identical to the RL solver's
# cap so all three searches are held to the same compute. The plateau rule can end a run earlier;
# whichever condition comes first decides.
BUDGET_CAP = 60


def run_metaheuristic(variants=("ga", "pso"), toy_dir=TOY, out_dir=OUT, M=P.M_SCENARIOS,
                      seed=P.SEED, budget=BUDGET_CAP, workers=16, seed_greedy=True):
    """Run the chosen metaheuristics over M scenarios and write outputs/greedy/n{N}/{variant}_optima.csv
    (schema shared with the static greedy solvers; checkpointed each scenario).

    Each run ends when its search levels off (PlateauStop) or when `budget` unique F evaluations
    have been spent, whichever comes first, so compute is normally an OUTCOME of the run rather
    than an input to it. `workers` sizes the evaluation pool. The recorded `n_evals` and `outcome`
    columns say how much each scenario actually cost and which condition ended it."""
    from multiprocessing import Pool
    import random as _random

    out_dir = scale_dir(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    disrupted = select_oracle_instance(toy_dir, P.N_DISRUPTED_ORACLE)
    segments = sorted(int(e) for e in disrupted["edge_id"])
    ctx = build_context(toy_dir, disrupted)
    scenarios = sample_scenarios(disrupted, M, seed)
    T = compute_horizon(segments, scenarios)
    flow = _baseline_twoway_flow(toy_dir)
    print(f"instance: {len(segments)} segments {segments}; M={M}; horizon T={T}; "
          f"variants={list(variants)}; budget={budget} evals; workers={workers}; "
          f"seed_greedy={seed_greedy}", flush=True)

    # Resume support: a scenario already present in {variant}_optima.csv is not recomputed, and its rows
    # are carried forward, so a relaunch after an interruption continues instead of restarting from zero.
    rows, done = {v: [] for v in variants}, {v: set() for v in variants}
    for v in variants:
        p = out_dir / f"{v}_optima.csv"
        if p.exists():
            rows[v] = pd.read_csv(p).to_dict("records")
            done[v] = {int(r["scenario"]) for r in rows[v]}
    if any(done[v] for v in variants):
        print("[resume] already done -> " +
              ", ".join(f"{v}:{sorted(done[v])}" for v in variants), flush=True)

    t_all = time.perf_counter()
    with Pool(workers, initializer=_worker_init, initargs=(toy_dir, disrupted)) as pool:
        for m, dur in enumerate(scenarios):
            seeds = _greedy_seed_orders(ctx, segments, dur, flow) if seed_greedy else []
            for v in variants:
                if m in done[v]:
                    continue
                rng = _random.Random(seed * 1000 + m * 10 + (0 if v == "ga" else 1))
                fit = FitnessCache(pool, dur, T, budget)
                t0 = time.perf_counter()
                if v == "ga":
                    stop = _ga(fit, segments, seeds, rng, **GA_PARAMS)
                else:
                    stop = _pso(fit, segments, seeds, rng, **PSO_PARAMS)
                perm, (F, F1, F2) = fit.best()
                start = schedule_from_permutation(list(perm), dur)
                row = dict(scenario=m, F=F, F1=F1, F2=F2, time_s=time.perf_counter() - t0,
                           n_evals=fit.n_evals, generations=stop.gen,
                           outcome="plateau" if stop.done() else "budget_cap",
                           order="-".join(map(str, perm)),
                           durations="-".join(str(int(dur[e])) for e in segments))
                for e in segments:
                    row[f"start_{e}"] = start[e]
                rows[v].append(row)
                done[v].add(m)
                pd.DataFrame(rows[v]).to_csv(out_dir / f"{v}_optima.csv", index=False)
            msg = []
            for v in variants:
                r = next((x for x in rows[v] if int(x["scenario"]) == m), None)
                if r is not None:
                    msg.append(f"{v}={r['F']:.4f}({r['n_evals']}ev)")
            if msg:
                print(f"  scenario {m + 1}/{M}:  " + "  ".join(msg), flush=True)

    for v in variants:
        print(f"  mean F [{v}] = {pd.DataFrame(rows[v])['F'].mean():.4f}", flush=True)
    print(f"Wrote {out_dir}  ({(time.perf_counter() - t_all) / 60:.1f} min)", flush=True)
    return out_dir


if __name__ == "__main__":
    vs = tuple(a for a in sys.argv[1:] if not a.startswith("-")) or ("ga", "pso")
    run_metaheuristic(variants=vs)
