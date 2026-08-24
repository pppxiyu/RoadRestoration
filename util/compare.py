"""
Compare the road-restoration solvers on a common instance, each discovered by NAME: the static
rule-based rankers under outputs/02-baselines/02-rule-based/n{N}/, the GA and both RL variants
each in its own numbered folder (02-baselines/04-ga, 03-RL/01-rl_nominal, 03-RL/02-rl_saa), the
pretraining MILP, and -- when its results exist for this scale -- the brute-force oracle. Aligns
them by scenario, computes gaps to the oracle, and writes to outputs/04-comparison/n{N}/
(comparison.csv in its results/ subfolder).

The oracle is included only if oracle_optima.csv is present for this scale (it is infeasible at
larger sizes). Greedy-vs-MILP is always reported.

Everything a figure here is drawn from is written next to it: comparison.csv holds the per-scenario
objectives, compute_accounting.csv the serial-equivalent compute axis (an accounting, not a
measurement, so it is stored rather than re-derived), and run_meta.json names the source files and
the run behind each of them. Any of these figures can therefore be redrawn, or audited, without
re-running a solver.

Run inside the road_restore conda env, after greedy and pretrain_milp have been run:
  python -m util.compare
"""
from pathlib import Path

import pandas as pd

import config as P
from util.oracle import scale_dir, select_oracle_instance
from util.provenance import (solver_dir, fresh_scale_dir, results_dir, source_meta,
                             write_run_meta)

ROOT = Path(__file__).resolve().parent.parent


# The three static rankers, named rather than globbed. outputs/02-baselines/02-rule-based/n{N}/
# used to be the SHARED pool every method wrote into, so at scales last touched before the per-method
# reorganisation it still holds ga_optima.csv / rl_optima.csv / rl_stoch_optima.csv. A glob picks
# those up and silently re-admits a superseded run under a legacy name -- which is exactly how
# `greedy_rl_stoch` was still appearing in the n6 comparison.
RULE_BASED = ("flow", "demand", "ratio")

# The searched/learned methods, each owning its own folder. rl_saa is the SAA variant (nominal DQN
# config + sample-average-approximation training on an LHS-covered world sample + a quantile-
# regression distributional head); its 5-seed current-config re-run is what admits it here.
SEARCHED = (("ga", "02-baselines"), ("rl_nominal", "03-RL"), ("rl_saa", "03-RL"),
            ("rl_s2v", "03-RL"))   # rl_s2v: EXPERIMENTAL; remove with util/rl_s2v.py


def _discover(base, gdir, N):
    """Every optima file the comparison is allowed to read, by NAME. Missing files are skipped;
    nothing is admitted that is not on one of the two lists above."""
    files = [gdir / "results" / f"{m}_optima.csv" for m in RULE_BASED]
    files = [f for f in files if f.exists()]
    for m, grp in SEARCHED:
        p = scale_dir(base / grp / solver_dir(m), N) / "results" / f"{m}_optima.csv"
        if p.exists():
            files.append(p)
    return files


def run_compare(N=None):
    """Align every baseline + MILP + (optional) oracle for scale N, write
    outputs/04-comparison/n{N}/ (results/comparison.csv + figures), and print a summary sorted
    best-first."""
    N = P.N_DISRUPTED_ORACLE if N is None else N
    base = ROOT / "outputs"
    gdir = scale_dir(base / "02-baselines" / solver_dir("rule-based"), N)
    m_path = scale_dir(base / "02-baselines" / solver_dir("pretrain_milp"), N) / "results" / "milp_optima.csv"
    o_path = scale_dir(base / "02-baselines" / solver_dir("brute-force"), N) / "results" / "oracle_optima.csv"
    files = _discover(base, gdir, N)
    if not files:
        raise FileNotFoundError(f"no optima under {base}/*/n{N}")
    if not m_path.exists():
        raise FileNotFoundError(f"no MILP results ({m_path}); run `python -m util.pretrain_milp` first")

    # Every method must be scored on the SAME scenario set, and the check has to come before the
    # merge. Merging on "scenario" is an inner join, so one method left at an older, smaller M
    # silently drags every other method down to that intersection: the table still builds, the
    # figures still redraw, and every number quietly reverts to the old sample size. A stale
    # result must announce itself, so a mismatch is refused here rather than absorbed.
    counts = {f.stem[: -len("_optima")]: len(pd.read_csv(f)) for f in files}
    counts["milp"] = len(pd.read_csv(m_path))
    if len(set(counts.values())) > 1:
        want = max(set(counts.values()), key=list(counts.values()).count)
        stale = {k: v for k, v in counts.items() if v != want}
        raise ValueError(
            f"scenario counts disagree: {counts}. Most methods have M={want}; {stale} "
            f"differ and were produced under a different M. Re-run those methods (or drop them "
            f"from the comparison) -- merging would silently score EVERY method on the "
            f"{min(counts.values())}-scenario intersection.")

    # each method -> a column named by its bare stem (flow/demand/ratio/ga/rl/...); then MILP; then oracle if present
    methods, df = [], None
    for f in files:
        col = f.stem[: -len("_optima")]
        d = pd.read_csv(f)[["scenario", "F"]].rename(columns={"F": col})
        df = d if df is None else df.merge(d, on="scenario")
        methods.append(col)
    df = df.merge(pd.read_csv(m_path)[["scenario", "F_milp"]].rename(columns={"F_milp": "milp"}), on="scenario")
    methods.append("milp")
    have_oracle = o_path.exists()
    if have_oracle:
        df = df.merge(pd.read_csv(o_path)[["scenario", "F"]].rename(columns={"F": "oracle"}), on="scenario")
        for c in methods:
            df[f"gap_{c}"] = df[c] - df["oracle"]                        # shortfall vs the true optimum

    out = scale_dir(base / "04-comparison", N)
    out.mkdir(parents=True, exist_ok=True)
    fresh_scale_dir(out)                     # a refresh replaces the comparison, no residue
    df.to_csv(results_dir(out) / "comparison.csv", index=False)

    print(f"=== solver comparison  n={N}  (M={len(df)}) ===", flush=True)
    ranked = sorted(methods + (["oracle"] if have_oracle else []), key=lambda c: df[c].mean())
    for c in ranked:
        line = f"  mean {c:16s} = {df[c].mean():.4f}"
        if have_oracle and c != "oracle":
            line += f"    gap vs oracle {df[f'gap_{c}'].mean():+.4f}"
        print(line, flush=True)
    for c in [m for m in methods if m != "milp"]:
        wins = int((df[c] > df["milp"] + 1e-9).sum())
        print(f"  MILP beats {c} in {wins}/{len(df)} scenarios", flush=True)

    # The instance the comparison is over, read from the same selection every solver used, so a
    # comparison file names its own instance rather than inheriting whatever config says today.
    segments = sorted(int(e) for e in select_oracle_instance(ROOT / "data" / "siouxfalls_toy", N)["edge_id"])
    write_run_meta(out, method="comparison", segments=segments, T=0, seed=P.SEED, M=len(df),
                   methods=methods + (["oracle"] if have_oracle else []),
                   mean_F={c: float(df[c].mean()) for c in methods},
                   sources=source_meta(files + [m_path] + ([o_path] if have_oracle else [])))

    from viz.compare_viz import (make_final_performance_all, make_gap_to_oracle,
                                 make_performance_distribution)
    # Every figure that belongs in the comparison is drawn HERE, not by hand: run_compare clears
    # this folder before rewriting it, so a figure produced outside the refresh is deleted by the
    # next one.
    make_final_performance_all(out, df, methods)                    # per-scenario final F, all methods
    make_performance_distribution(out, df, methods)                 # how far those F values spread
    if have_oracle:                                                 # gap to the true optimum, small scales only
        make_gap_to_oracle(out, df, methods)

    # Accuracy against compute belongs to EVERY refresh, not to a separate entry point. It used to
    # be produced only by run_baseline_figures, so whether a scale had the figure depended on how
    # that scale happened to be invoked last, and the n{N} folders drifted apart. Every solver's
    # end-of-run refresh calls run_compare, so putting it here is what makes the folders uniform.
    from viz.compare_viz import make_accuracy_compute
    T, stats = _compute_accounting({f.stem[: -len("_optima")]: pd.read_csv(f) for f in files},
                                   pd.read_csv(m_path), M=len(df))
    pd.DataFrame(stats).to_csv(results_dir(out) / "compute_accounting.csv", index=False)
    make_accuracy_compute(out, stats)
    print(f"  compute axis (T={T} UE/evaluation, serial-equivalent UE per scenario):", flush=True)
    for st in sorted(stats, key=lambda x: x["mean_F"]):
        print(f"    {st['method']:8s} F={st['mean_F']:.4f}  {st['mean_ue']:.0f} UE/scenario",
              flush=True)
    print(f"Wrote {results_dir(out) / 'comparison.csv'} and figures", flush=True)
    return df


def refresh_comparison(N=None):
    """Refresh the cross-method comparison after a solver run, tolerating an incomplete result set.

    Every runner calls this when it finishes writing its own outputs, so the comparison table and
    figures always reflect the latest run of every method without a separate manual step. Early in
    a scale's life some inputs are legitimately missing (no MILP yet, no optima at all); that is a
    reason to skip quietly, not to fail the solver run that just completed. Any other error is
    REPORTED but still not raised: a finished multi-hour run must never be lost to a plotting
    problem in the comparison step."""
    try:
        run_compare(N)
    except FileNotFoundError as exc:
        print(f"[compare] skipped: {exc}", flush=True)
    except Exception as exc:
        print(f"[compare] FAILED ({type(exc).__name__}: {exc}) -- solver outputs are unaffected",
              flush=True)


def _compute_accounting(greedy_finals, milp_opt, M):
    """Every method placed on ONE compute axis: serial-equivalent UE solves per scenario.

    THE COMPUTE A METHOD REQUIRES, NOT THE COMPUTE IT HAPPENED TO SPEND. One full true-F evaluation
    of one order costs T user-equilibrium solves, one per slot of the horizon, and that is what a
    method is charged for every distinct order it scores. The persistent slot cache makes many of
    those solves free in practice, but a cache is an implementation detail of this codebase, not a
    property of the algorithm -- charging the cached price would make a method look cheaper on a
    machine that has run it before than on one that has not, and would flatter whichever method
    happens to revisit similar damage trajectories. Counting UE solves rather than wall-clock also
    keeps the axis neutral to how many worker processes a method was parallelised over.

    T is recovered exactly from the MILP, whose ue_solves column equals (n_iter + 1) * T.
    A static rule spends T: one evaluation. A searched or learned method spends n_evals * T for its
    one-off search on the nominal instance, amortised over the M scenarios that search serves, plus
    T for this scenario's own evaluation. The MILP spends its recorded ue_solves, likewise
    amortised, because its cost is the alternating optimisation's own solves rather than a count of
    scored orders.

    The per-method `ue_total` column is deliberately NOT read here. Each writer chose its own
    convention for it, and one of them (util.rl_rank, before 2026-08-11) divided the ORDER count
    rather than the UE count by M -- reporting rl_nominal at 36 UE/scenario when the honest figure is
    140. Deriving the axis in one place from n_evals and T removes that whole class of drift.
    """
    T = int(round((milp_opt["ue_solves"] / (milp_opt["n_iter"] + 1)).median()))
    stats = []
    for v, d in greedy_finals.items():
        if "n_evals" not in d.columns:                                 # a static ranking rule
            stats.append(dict(method=v, mean_F=float(d["F"].mean()), mean_ue=float(T),
                              kind="greedy"))
            continue
        n_evals = float(d["n_evals"].mean())
        kind = ("rl" if v.startswith("rl_nominal") else "rl_saa" if v.startswith("rl_saa")
                else "rl_s2v" if v.startswith("rl_s2v") else "meta")
        stats.append(dict(method=v, mean_F=float(d["F"].mean()), n_evals=n_evals,
                          mean_ue=n_evals * T / M + T, kind=kind))
    stats.append(dict(method="milp", mean_F=float(milp_opt["F_milp"].mean()),
                      mean_ue=float(milp_opt["ue_solves"].mean()) / M + T, kind="milp"))
    return T, stats


def run_baseline_figures(N=None):
    """Backwards-compatible entry point (`python main.py --solve compare`). Everything it used to
    add on top of run_compare -- the compute accounting and the accuracy-vs-compute figure -- is
    now part of run_compare itself, so that every scale gets the same set of files no matter which
    runner last touched it. Kept as a name so existing invocations keep working."""
    return run_compare(N)


if __name__ == "__main__":
    run_compare()
