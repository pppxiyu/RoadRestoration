"""Repeat GA across search seeds while keeping the evaluation sample fixed.

Every seed writes its own complete method run; no checkpoint or provenance file is
copied over another. The lowest mean F is selected for comparison in
``data/methods/selected_runs.json`` and all values are retained in
``data/analysis/seed_sweeps/ga/seeds_summary.csv``. Selecting the best seed is
optimistically biased, so the summary retains the complete distribution.
"""
import json
import time
import argparse

import pandas as pd

from src import config as P
from src.experiment.layout import analysis_dir, experiment_data_dir, method_dir

# The default seed set. config.SEED leads so the sweep always contains the run every earlier
# result was produced with, making the new spread directly comparable to what is already on disk.
DEFAULT_SEEDS = (P.SEED, 1, 2, 3, 4)

# Where each method's scale folder lives, and how its runner is invoked.
_METHODS = {
    "ga": dict(optima="ga_optima.csv"),
}


def _run_once(method, s, N, M):
    """Invoke the method's own runner for one search seed. Each runner clears and rewrites its
    scale folder exactly as in a normal single run -- nothing here is a special training path."""
    if method == "ga":
        from src.methods.metaheuristic import run_metaheuristic
        run_metaheuristic(variants=("ga",), M=M, search_seed=s)
    else:
        raise SystemExit(f"seed sweep does not know method {method!r}")


def _archive_complete(arch, optima_name):
    """Whether a seed's run can be trusted as finished for `resume`.

    Deliberately stricter than "the optima file is there": a run is complete only if its
    deliverable, provenance and trace are all required. A half-written run with
    only its optima table must be retried.
    """
    return ((arch / "results" / optima_name).exists()
            and (arch / "config" / "run_meta.json").exists()
            and any((arch / "log").glob("*_trace.csv")))


def _mean_F(run_dir, optima_name):
    """The delivered mean F over the M frozen scenarios: the number the comparison reports and the
    one this sweep selects on."""
    p = run_dir / "results" / optima_name
    if not p.exists():
        raise SystemExit(f"run produced no {optima_name} at {p}")
    return float(pd.read_csv(p)["F"].mean())


def run_seed_sweep(method, seeds=DEFAULT_SEEDS, N=None, M=P.M_SCENARIOS, resume=False):
    """Run each seed independently and record which seed comparison should select."""
    if method not in _METHODS:
        raise SystemExit(f"unknown method {method!r}; choose from {sorted(_METHODS)}")
    N = P.N_DISRUPTED_ORACLE if N is None else int(N)
    if N != P.N_DISRUPTED_ORACLE or M != P.M_SCENARIOS:
        raise ValueError("seed sweep must use the configured scale and common evaluation sample")
    spec = _METHODS[method]
    seeds = [int(s) for s in seeds]
    if len(seeds) != len(set(seeds)) or not seeds:
        raise ValueError("seed sweep requires distinct nonempty seeds")

    print("=" * 72)
    print(f"SEED SWEEP  {method}  seeds={seeds}  N={P.N_DISRUPTED_ORACLE if N is None else N}, M={M}")
    print(f"  evaluation scenarios stay pinned to config.SEED={P.SEED} in every run")
    print("=" * 72, flush=True)

    rows = []
    t_all = time.perf_counter()
    for i, s in enumerate(seeds, 1):
        run_dir = method_dir(method, N, s)
        if resume and _archive_complete(run_dir, spec["optima"]):
            mF = _mean_F(run_dir, spec["optima"])
            el = float("nan")
            print(f"\n--- seed {s}  ({i}/{len(seeds)}) ALREADY COMPLETE, mean F = {mF:.6f}",
                  flush=True)
        else:
            print(f"\n--- seed {s}  ({i}/{len(seeds)}) ---", flush=True)
            t0 = time.perf_counter()
            _run_once(method, s, N, M)
            el = time.perf_counter() - t0
            mF = _mean_F(run_dir, spec["optima"])
        meta_p = run_dir / "config" / "run_meta.json"
        meta = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.exists() else {}
        # n_evals -- distinct orders scored, the search-effort column -- sits in run_meta for the
        # GA and only in the optima table for the RL runner, so fall back rather than deliver a
        # summary with an empty column.
        n_ev = meta.get("n_evals")
        if n_ev is None:
            od = pd.read_csv(run_dir / "results" / spec["optima"])
            n_ev = int(od["n_evals"].iloc[0]) if "n_evals" in od.columns else ""
        rows.append(dict(seed=s, mean_F=mF, minutes=el / 60.0,
                         # GA delivers one static order under `order`; the RL delivers per-scenario
                         # and records its nominal-world SUMMARY under `order_nominal_summary`, so
                         # this column means "the run's representative order", not always its decision.
                         order=meta.get("order", meta.get("order_nominal_summary", "")),
                         episodes=meta.get("episodes", meta.get("generations", "")),
                         n_evals=n_ev, outcome=meta.get("outcome", "")))
        print(f"--- seed {s}: mean F = {mF:.6f}  ({el/60:.1f} min) -> {run_dir.name}", flush=True)

    summary = pd.DataFrame(rows).sort_values("mean_F").reset_index(drop=True)
    best = summary.iloc[0]
    summary.insert(0, "delivered", ["<- delivered"] + [""] * (len(summary) - 1))
    report_dir = analysis_dir(N) / "seed_sweeps" / method
    report_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(report_dir / "seeds_summary.csv", index=False)
    selection = experiment_data_dir(N) / "methods" / "selected_runs.json"
    choices = json.loads(selection.read_text(encoding="utf-8")) if selection.exists() else {}
    choices[method] = int(best["seed"])
    selection.write_text(json.dumps(choices, indent=2), encoding="utf-8")

    spread = summary["mean_F"].max() - summary["mean_F"].min()
    print(f"\n=== {method}: {len(seeds)} seeds, {(time.perf_counter()-t_all)/60:.1f} min ===")
    print(summary.to_string(index=False))
    print(f"\nbest seed {int(best['seed'])} selected at {method_dir(method, N, int(best['seed']))}")
    print(f"spread across seeds: {spread:.6f}  (best-of-{len(seeds)} is optimistically biased; "
          f"report the spread with the headline)")
    print(f"all runs kept in their own seed folders; summary at {report_dir}", flush=True)

    from src.analysis.compare import refresh_comparison
    refresh_comparison()
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method", nargs="?", default="ga", choices=tuple(_METHODS))
    parser.add_argument("--seeds", help="comma-separated search seeds, or a count from the default set")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    sd = DEFAULT_SEEDS
    if args.seeds:
        v = args.seeds
        sd = tuple(int(x) for x in v.split(",")) if "," in v else tuple(DEFAULT_SEEDS[:int(v)])
    run_seed_sweep(args.method, seeds=sd, resume=args.resume)
