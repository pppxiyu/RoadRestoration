"""
Compare the road-restoration solvers on a common instance: every baseline found under
outputs/greedy/n{N}/ (the static greedy variants plus the GA/PSO metaheuristics, which share that
directory), the pretraining MILP, and -- when its results exist for this scale -- the brute-force
oracle. Aligns them by scenario, computes gaps to the oracle, and writes to outputs/comparison/n{N}/.

The oracle is included only if oracle_optima.csv is present for this scale (it is infeasible at
larger sizes); greedy-vs-MILP is always reported.

Run inside the road_restore conda env, after greedy and pretrain_milp have been run:
  python -m util.compare
"""
from pathlib import Path

import pandas as pd

import config as P
from util.oracle import scale_dir

ROOT = Path(__file__).resolve().parent.parent


def run_compare(N=None):
    """Align every baseline + MILP + (optional) oracle for scale N, write
    outputs/comparison/n{N}/comparison.csv + figures, and print a summary sorted best-first."""
    N = P.N_DISRUPTED_ORACLE if N is None else N
    base = ROOT / "outputs"
    gdir = scale_dir(base / "greedy", N)
    m_path = scale_dir(base / "pretrain_milp", N) / "milp_optima.csv"
    o_path = scale_dir(base / "oracle", N) / "oracle_optima.csv"
    gfiles = sorted(gdir.glob("*_optima.csv"))
    if not gfiles:
        raise FileNotFoundError(f"no greedy results in {gdir}; run `python -m util.greedy` first")
    if not m_path.exists():
        raise FileNotFoundError(f"no MILP results ({m_path}); run `python -m util.pretrain_milp` first")

    # each greedy variant -> a column "greedy_<variant>"; then MILP; then oracle if present
    methods, df = [], None
    for f in gfiles:
        col = "greedy_" + f.stem[: -len("_optima")]
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

    out = scale_dir(base / "comparison", N)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "comparison.csv", index=False)

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

    from viz.compare_viz import make_final_performance_all, make_gap_to_oracle
    make_final_performance_all(out, df, methods)                    # per-scenario final F, all methods
    if have_oracle:                                                 # gap to the true optimum, small scales only
        make_gap_to_oracle(out, df, methods)
    print(f"Wrote {out / 'comparison.csv'} and figures", flush=True)
    return df


def run_baseline_figures(N=None):
    """Generate the baselines-vs-MILP figures into outputs/comparison/n{N}/figures/:
    'optimization_process' (each scenario's best-so-far MILP trajectory, no baseline levels) and
    'accuracy_vs_compute' (mean final F vs serial-equivalent UE solves per scenario, a
    parallelism-neutral compute axis). Then calls run_compare, which adds 'final_performance_all'
    (per-scenario final F for every method) and, where the oracle exists, 'gap_to_oracle'. Reads
    each greedy variant's {variant}_optima.csv and the MILP's milp_optima.csv + milp_trace.csv."""
    N = P.N_DISRUPTED_ORACLE if N is None else N
    base = ROOT / "outputs"
    gdir = scale_dir(base / "greedy", N)
    mdir = scale_dir(base / "pretrain_milp", N)
    out = scale_dir(base / "comparison", N)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    greedy_finals = {}
    for f in sorted(gdir.glob("*_optima.csv")):
        greedy_finals[f.stem[: -len("_optima")]] = pd.read_csv(f)
    milp_opt = pd.read_csv(mdir / "milp_optima.csv")
    milp_trace = pd.read_csv(mdir / "milp_trace.csv")

    from viz.compare_viz import make_accuracy_compute, make_process
    make_process(out, milp_trace)

    # Serial-equivalent compute: one full true-F evaluation costs T UE solves (one per slot of the
    # horizon), so this axis is parallelism-neutral. Recover T exactly from the MILP, whose ue_solves
    # column equals (n_iter + 1) * T. A static greedy rule then spends T per scenario (a single
    # evaluation), a metaheuristic spends n_evals * T (its unique-evaluation budget), and the MILP
    # spends its own ue_solves.
    T = int(round((milp_opt["ue_solves"] / (milp_opt["n_iter"] + 1)).median()))
    stats = []
    for v, d in greedy_finals.items():
        if "n_evals" in d.columns:                                     # a metaheuristic (GA / PSO)
            stats.append(dict(method=v, mean_F=float(d["F"].mean()),
                              mean_ue=float(d["n_evals"].mean()) * T, kind="meta"))
        else:                                                          # a static greedy rule
            stats.append(dict(method=v, mean_F=float(d["F"].mean()), mean_ue=float(T), kind="greedy"))
    stats.append(dict(method="milp", mean_F=float(milp_opt["F_milp"].mean()),
                      mean_ue=float(milp_opt["ue_solves"].mean()), kind="milp"))
    make_accuracy_compute(out, stats)

    print(f"=== baselines vs MILP, n={N} (M={len(milp_opt)}, T={T}) ===", flush=True)
    for s in sorted(stats, key=lambda s: s["mean_F"]):
        print(f"  {s['method']:8s}  mean F = {s['mean_F']:.4f}   serial-equiv compute = {s['mean_ue']:.0f} UE/scenario",
              flush=True)
    try:
        run_compare(N)                                              # also the per-scenario bar figure
    except FileNotFoundError:
        pass
    print(f"Wrote baseline figures to {out / 'figures'}", flush=True)


if __name__ == "__main__":
    run_compare()
