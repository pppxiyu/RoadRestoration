"""
Repeat a stochastic solver across several random seeds, keep every run, deliver the best one.

WHY THIS EXISTS. Both searched solvers here are randomized: the RL agent's exploration, replay
sampling and weight initialization all read one seed, and the genetic algorithm's crossover,
mutation and tournament draws read another. A single run therefore reports one draw from a
distribution, and at n=10 that distribution is wide -- two seeds of the same RL configuration have
landed 0.019 apart in mean F, which is hundreds of times the gap the study is trying to resolve
between methods. Reporting one run as "the method's performance" is then not a small imprecision,
it is reading noise as signal.

WHAT IT DOES. Runs the solver once per seed, ARCHIVES each complete run under
outputs/.../n{N}/history/seed{s}/, then restores the best one to n{N}/ so the delivered folder
still holds exactly one run with its normal layout -- every downstream consumer (compare.py, the
figures, the provenance files) keeps working unchanged and simply describes the best seed. The
spread across seeds is not thrown away: history/seeds_summary.csv records every seed's outcome, and
the RL objective figure draws the across-seed band from the archived traces, so the delivered
figure shows the variability rather than hiding it behind its own best case.

WHICH SEED MOVES. Only the SEARCH seed. The M evaluation scenarios are the frozen ruler the whole
study is compared on and stay pinned to config.SEED in every run -- util.rl_rank draws them from
P.SEED directly, and util.metaheuristic takes a separate `search_seed` for exactly this reason.
A sweep that moved the scenario sample would produce five runs measured against five different
rulers and mislabel the result as seed variance.

BEST BY WHAT. The lowest mean F over the M frozen scenarios, which is the number the comparison
reports. Selecting the best of several runs is a real optimistic bias and it is stated in the
summary file rather than left implicit: the honest reading of a best-of-5 is "what this method can
reach when the draw goes well", with the full spread beside it.

  python -m util.seed_sweep rl_dqn            # 5 seeds, the default set
  python -m util.seed_sweep ga --seeds 1,2,3
  python main.py --solve rl_dqn --seeds 5
"""
import json
import shutil
import sys
import time
from pathlib import Path

import pandas as pd

import config as P
from util.provenance import solver_dir
from util.oracle import scale_dir

ROOT = Path(__file__).resolve().parent.parent

# The default seed set. config.SEED leads so the sweep always contains the run every earlier
# result was produced with, making the new spread directly comparable to what is already on disk.
DEFAULT_SEEDS = (P.SEED, 1, 2, 3, 4)

# Where each method's scale folder lives, and how its runner is invoked.
_METHODS = {
    "rl_dqn": dict(base=ROOT / "outputs" / "03-rl" / solver_dir("rl_dqn"), optima="rl_dqn_optima.csv"),
    "ga": dict(base=ROOT / "outputs" / "02-baselines" / solver_dir("ga"), optima="ga_optima.csv"),
}
# Subtrees copied into (and back out of) an archive: everything a run produces.
_PARTS = ("results", "log", "config")


def _run_once(method, s, N, M):
    """Invoke the method's own runner for one search seed. Each runner clears and rewrites its
    scale folder exactly as in a normal single run -- nothing here is a special training path."""
    if method == "rl_dqn":
        from util.rl_rank import run_rank
        run_rank(variants=(method,), N=N, M=M, seed=s)
    elif method == "ga":
        from util.metaheuristic import run_metaheuristic
        run_metaheuristic(variants=("ga",), M=M, search_seed=s)
    else:
        raise SystemExit(f"seed sweep does not know method {method!r}")


def _clear_files(d):
    """Delete every FILE under `d`, leaving the directory itself in place."""
    for f in sorted(d.rglob("*")):
        if f.is_file():
            for k in range(4):
                try:
                    f.unlink()
                    break
                except (PermissionError, OSError):
                    if k == 3:
                        raise
                    time.sleep(0.3 * (k + 1))


def _copy_run(src, dst):
    """Copy one complete run (its three subtrees plus its top-level figures). `history/` is never
    copied into an archive -- an archive holds one run, not a nest of them.

    Overwrites IN PLACE and never removes a directory. Windows refuses `rmdir` on a directory whose
    files are in the "delete pending" state -- unlinked, but still held by some handle -- and raises
    Access Denied from inside shutil.rmtree, after it has already removed part of the tree. Retrying
    does not help because the handle outlives the retry window. Deleting the files and copying over
    the surviving directory reaches the same end state without ever asking for the one operation
    that fails, and it cannot leave a partly deleted archive behind if it does fail.
    """
    dst.mkdir(parents=True, exist_ok=True)
    for part in _PARTS:
        s = src / part
        if s.exists():
            d = dst / part
            d.mkdir(parents=True, exist_ok=True)
            _clear_files(d)
            shutil.copytree(s, d, dirs_exist_ok=True)
    for png in src.glob("*.png"):
        shutil.copy2(png, dst / png.name)


def _archive_complete(arch, optima_name):
    """Whether an archived seed can be trusted as a finished run, for `resume`.

    Deliberately stricter than "the optima file is there": a run is complete only if its
    DELIVERABLE, its PROVENANCE and its TRACE all survived. A half-written archive that still has
    its optima looks finished, so resume would skip it and the summary would carry blank fields for
    a run nobody re-made. That is not hypothetical -- an interrupted cleanup left exactly this
    shape here, an archive with results and figures but no run_meta.json.
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
    """Run `method` once per seed, archive them all under n{N}/history/, deliver the best.

    `resume=True` keeps any archive already present and re-runs only the missing seeds. A sweep is
    a long chain of independent runs, and losing finished ones to a failure in a later link is pure
    waste. The default is False so an ordinary sweep starts clean: a stale archive from an earlier
    configuration silently joins the summary and the across-seed band otherwise.
    """
    if method not in _METHODS:
        raise SystemExit(f"unknown method {method!r}; choose from {sorted(_METHODS)}")
    spec = _METHODS[method]
    vdir = scale_dir(spec["base"], N)
    hist = vdir / "history"
    seeds = [int(s) for s in seeds]

    if hist.exists() and not resume:
        _clear_files(hist)          # files only; see _copy_run for why directories are left alone
    hist.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(f"SEED SWEEP  {method}  seeds={seeds}  N={P.N_DISRUPTED_ORACLE if N is None else N}, M={M}")
    print(f"  evaluation scenarios stay pinned to config.SEED={P.SEED} in every run")
    print("=" * 72, flush=True)

    rows = []
    t_all = time.perf_counter()
    for i, s in enumerate(seeds, 1):
        arch = hist / f"seed{s}"
        if resume and _archive_complete(arch, spec["optima"]):
            mF = _mean_F(arch, spec["optima"])
            el = float("nan")
            print(f"\n--- seed {s}  ({i}/{len(seeds)}) ALREADY ARCHIVED, mean F = {mF:.6f}",
                  flush=True)
        else:
            print(f"\n--- seed {s}  ({i}/{len(seeds)}) ---", flush=True)
            t0 = time.perf_counter()
            _run_once(method, s, N, M)
            el = time.perf_counter() - t0
            _copy_run(vdir, arch)
            mF = _mean_F(arch, spec["optima"])
        meta_p = arch / "config" / "run_meta.json"
        meta = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.exists() else {}
        # n_evals -- distinct orders scored, the search-effort column -- sits in run_meta for the
        # GA and only in the optima table for the RL runner, so fall back rather than deliver a
        # summary with an empty column.
        n_ev = meta.get("n_evals")
        if n_ev is None:
            od = pd.read_csv(arch / "results" / spec["optima"])
            n_ev = int(od["n_evals"].iloc[0]) if "n_evals" in od.columns else ""
        rows.append(dict(seed=s, mean_F=mF, minutes=el / 60.0,
                         # GA delivers one static order under `order`; the RL delivers per-scenario
                         # and records its nominal-world SUMMARY under `order_nominal_summary`, so
                         # this column means "the run's representative order", not always its decision.
                         order=meta.get("order", meta.get("order_nominal_summary", "")),
                         episodes=meta.get("episodes", meta.get("generations", "")),
                         n_evals=n_ev, outcome=meta.get("outcome", "")))
        print(f"--- seed {s}: mean F = {mF:.6f}  ({el/60:.1f} min) -> {arch.name}", flush=True)

    summary = pd.DataFrame(rows).sort_values("mean_F").reset_index(drop=True)
    best = summary.iloc[0]
    # Restore the best run to the scale folder, so the delivered layout is a normal single run.
    _copy_run(hist / f"seed{int(best['seed'])}", vdir)
    summary.insert(0, "delivered", ["<- delivered"] + [""] * (len(summary) - 1))
    summary.to_csv(hist / "seeds_summary.csv", index=False)

    if method == "rl_dqn":                    # redraw with the across-seed band now available
        from viz.rank_viz import make_rank_figures
        make_rank_figures(vdir, method)

    spread = summary["mean_F"].max() - summary["mean_F"].min()
    print(f"\n=== {method}: {len(seeds)} seeds, {(time.perf_counter()-t_all)/60:.1f} min ===")
    print(summary.to_string(index=False))
    print(f"\nbest seed {int(best['seed'])} delivered to {vdir}")
    print(f"spread across seeds: {spread:.6f}  (best-of-{len(seeds)} is optimistically biased; "
          f"report the spread with the headline)")
    print(f"all runs kept under {hist}", flush=True)

    from util.compare import refresh_comparison
    refresh_comparison()
    return summary


if __name__ == "__main__":
    args = [a for a in sys.argv[1:]]
    m = next((a for a in args if not a.startswith("-")), "rl_dqn")
    sd = DEFAULT_SEEDS
    if "--seeds" in args:
        v = args[args.index("--seeds") + 1]
        sd = tuple(int(x) for x in v.split(",")) if "," in v else tuple(DEFAULT_SEEDS[:int(v)])
    run_seed_sweep(m, seeds=sd, resume="--resume" in args)
