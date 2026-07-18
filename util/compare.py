"""
Compare the road-restoration solvers on a common instance: every greedy variant found under
outputs/greedy/n{N}/, the pretraining MILP, and -- when its results exist for this scale -- the
brute-force oracle. Aligns them by scenario, computes each method's gap to the oracle (the true
hindsight optimum), and writes a comparison table + figure to outputs/comparison/n{N}/.

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
    """Align all greedy variants + MILP + (optional) oracle for scale N, write
    outputs/comparison/n{N}/comparison.csv + figure, and print a summary sorted best-first."""
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

    from viz.compare_viz import make_comparison_figure
    make_comparison_figure(out, df, N, methods, have_oracle)
    print(f"Wrote {out / 'comparison.csv'} and its figure", flush=True)
    return df


def run_baseline_figures(N=None):
    """Generate the two baselines-vs-MILP figures into outputs/comparison/n{N}/figures/:
    'process_and_final' (MILP iteration trajectory + each greedy variant's final level, plus a
    per-scenario final-F bar panel) and 'accuracy_vs_time' (mean final F vs mean wall-clock per
    method). Also (re)makes the plain bar comparison if the inputs are present. Reads each greedy
    variant's {variant}_optima.csv (with time_s) and the MILP's milp_optima.csv + milp_trace.csv."""
    N = P.N_DISRUPTED_ORACLE if N is None else N
    base = ROOT / "outputs"
    gdir = scale_dir(base / "greedy", N)
    mdir = scale_dir(base / "pretrain_milp", N)
    out = scale_dir(base / "comparison", N)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    greedy_finals = {}
    for f in sorted(gdir.glob("*_optima.csv")):
        greedy_finals[f.stem[: -len("_optima")]] = pd.read_csv(f)[["scenario", "F", "time_s"]]
    milp_opt = pd.read_csv(mdir / "milp_optima.csv")
    milp_trace = pd.read_csv(mdir / "milp_trace.csv")

    from viz.compare_viz import make_accuracy_time, make_process_and_final
    make_process_and_final(out, N, milp_trace, greedy_finals)

    stats = [dict(method=v, mean_F=float(d["F"].mean()), mean_time_s=float(d["time_s"].mean()),
                  kind="greedy") for v, d in greedy_finals.items()]
    stats.append(dict(method="milp", mean_F=float(milp_opt["F_milp"].mean()),
                      mean_time_s=float(milp_opt["time_s"].mean()), kind="milp"))
    make_accuracy_time(out, N, stats)

    print(f"=== baselines vs MILP, n={N} (M={len(milp_opt)}) ===", flush=True)
    for s in sorted(stats, key=lambda s: s["mean_F"]):
        print(f"  {s['method']:8s}  mean F = {s['mean_F']:.4f}   mean time = {s['mean_time_s']:.1f}s", flush=True)
    try:
        run_compare(N)                                              # also the per-scenario bar figure
    except FileNotFoundError:
        pass
    print(f"Wrote baseline figures to {out / 'figures'}", flush=True)


if __name__ == "__main__":
    run_compare()
