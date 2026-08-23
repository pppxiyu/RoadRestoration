"""Recording helpers shared by every solver, so that any figure can be redrawn from disk without
re-running a solve.

Two things are recorded that no single solver owns:

  * RUN METADATA. A results file on its own is ambiguous: the same rl_trace.csv could have come
    from a different instance size, a different seed, a different hyperparameter set or a
    different revision of the code, and nothing in the CSV says which. write_run_meta drops a
    run_meta.json into the run's config/ folder naming all of that, including the git
    revision and whether the tree was dirty, so a figure can always be traced back to the run
    that produced it. results_dir / log_dir / config_dir name the three subfolders every method
    folder carries (the layout is documented above their definitions), and fresh_scale_dir is
    what guarantees a rerun replaces a folder's contents instead of leaving residue.

  * PER-SLOT ACCESSIBILITY. The optima files record only the objective of the delivered schedule,
    which is enough for a bar chart and for nothing else. The recovery curve that this literature
    reports -- how accessibility evolves across the horizon while repairs complete -- needs the
    per-slot terms the evaluator already computes internally and then throws away. slot_rows
    flattens them for storage, so that curve costs no extra UE solve: the evaluation that
    produced the reported objective is the same one that produced the curve.
"""
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import config as P

# Numbered on-disk folder for each solver, in the fixed presentation order (baselines:
# brute-force, rule-based, MILP, GA; RL: nominal, SAA). Every path that reaches a solver's
# output folder is built through solver_dir, so the numbering lives in exactly this one table.
SOLVER_DIR = {
    "brute-force": "01-brute-force",
    "rule-based": "02-rule-based",
    "pretrain_milp": "03-pretrain_milp",
    "ga": "04-ga",
    "rl_nominal": "01-rl_nominal",
    "rl_saa": "02-rl_saa",
}


def solver_dir(name):
    """The numbered folder a solver's outputs live under (bare name if it is not in the table)."""
    return SOLVER_DIR.get(name, name)



def _git_revision(root):
    """Current commit and whether the working tree had uncommitted changes, or None outside a
    repository. A figure produced from a dirty tree cannot be reproduced from the commit alone,
    so the flag is worth as much as the hash."""
    try:
        rev = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        if rev.returncode:
            return None
        st = subprocess.run(["git", "-C", str(root), "status", "--porcelain"],
                            capture_output=True, text=True, timeout=10)
        return {"commit": rev.stdout.strip(),
                "dirty": bool(st.stdout.strip()) if st.returncode == 0 else None}
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# The output layout. Every n{N} scale folder of every method has exactly this shape (method
# folders sit under a numbered group and take their numbered names from SOLVER_DIR above,
# e.g. outputs/02-baselines/03-pretrain_milp/n{N}/, outputs/03-RL/01-rl_nominal/n{N}/):
#
#   outputs/{group}/{NN-method}/n{N}/
#     results/    what the run DELIVERS: the optima table, the delivered solution/order, the
#                 trained model weights. The one folder whose contents may be irreplaceable
#                 (a stochastic policy's weights cannot be recomputed without retraining).
#     log/        the PROCESS RECORDS: per-episode/per-generation traces, per-slot accessibility,
#                 diagnostics (Q evolution, delivered-policy analyses), progress files. Their
#                 purpose is that ANY figure can be drawn from here without re-running a solve.
#     config/     what IDENTIFIES the run: run_meta.json (instance, hyperparameters, objective
#                 parameters, git revision). Answers "which run was this, how do I get it again".
#     *.png       the figures, at the top level beside the three folders.
#
# These helpers are the layout's single authority -- every writer takes its directories from
# here, so the shape cannot drift per module.
# --------------------------------------------------------------------------- #
def results_dir(out_dir):
    """`results/`: the run's deliverables -- optima table, delivered solution, model weights."""
    d = Path(out_dir) / "results"
    d.mkdir(parents=True, exist_ok=True)
    return d


def log_dir(out_dir):
    """`log/`: process records and diagnostics, sufficient to draw any figure without re-running
    the solve that produced them."""
    d = Path(out_dir) / "log"
    d.mkdir(parents=True, exist_ok=True)
    return d


def config_dir(out_dir):
    """`config/`: the run's identity -- run_meta.json and any other metadata."""
    d = Path(out_dir) / "config"
    d.mkdir(parents=True, exist_ok=True)
    return d


def fresh_scale_dir(out_dir, methods=None, subdirs=("results", "log", "config"), figures=True):
    """Clear a scale folder before a run writes into it, so a rerun leaves NO residue of the
    previous run -- a stale trace or figure beside fresh results silently misdescribes the run.

    `methods=None` clears the whole folder (the normal case: the folder belongs to one method).
    A list of method prefixes clears only files starting with `{method}` in the given subdirs
    and the top-level figures -- for the shared static-greedy folder, where rerunning one ranker
    must not erase its siblings' results.

    `subdirs`/`figures` let a long-running solver clear in TWO stages: diagnostics (log/ and the
    figures, which training rewrites progressively anyway) at run start, but the deliverables
    (results/ and config/) only at the moment the new ones are written -- so a rerun that dies
    mid-training has not destroyed the previous run's delivered solution and model, which for a
    stochastic policy cannot be recomputed without retraining."""
    out_dir = Path(out_dir)
    if not out_dir.exists():
        return out_dir
    if methods is None:
        for sub in subdirs:
            d = out_dir / sub
            if d.exists():
                for f in d.iterdir():
                    if f.is_file():
                        f.unlink()
        if figures:
            for f in out_dir.glob("*.png"):
                f.unlink()
    else:
        for m in methods:
            for sub in subdirs:
                d = out_dir / sub
                if d.exists():
                    for f in d.glob(f"{m}*"):
                        if f.is_file():
                            f.unlink()
            if figures:
                for f in out_dir.glob(f"{m}*.png"):
                    f.unlink()
    return out_dir


def meta_name(method, shared_dir):
    """File name for a method's run metadata: `run_meta.json` when the method owns its directory,
    `{method}_run_meta.json` when several methods write into one. The shared case is
    outputs/02-baselines/02-rule-based/n{N}/, where the three static rankers (flow/demand/ratio)
    all drop their optima -- a single run_meta.json there would describe only whichever ranker
    happened to finish last. The GA, the MILP and the RL variants each own their own numbered
    directory and use the plain name."""
    return f"{method}_run_meta.json" if shared_dir else "run_meta.json"


def write_run_meta(out_dir, method, segments, T, seed, M=None, shared_dir=False, **extra):
    """Write run_meta.json into `out_dir`, describing the run that produced the files beside it.

    `extra` carries whatever the caller considers part of its own identity, typically the
    hyperparameters, the evaluation budget, the nominal durations it optimized against and the
    condition that ended it. Everything else is common to all solvers and is filled in here: when
    the run happened, which instance and horizon it used, the objective-defining parameters, and
    the code revision.

    The instance size and scenario count are read off the RUN -- from the segment list it actually
    solved and the M it was actually called with -- not from config. Both are routinely overridden
    at call time, so a file reporting the config value would confidently name the wrong instance,
    which is worse than recording nothing. The config values are kept alongside under
    `config_declares`, so an override is visible rather than silently absorbed.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    segments = list(map(int, segments))
    M = P.M_SCENARIOS if M is None else int(M)
    meta = {
        "method": method,
        "written_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "instance": {"n_disrupted": len(segments), "segments": segments,
                     "M_scenarios": M, "horizon_T": int(T), "seed": int(seed),
                     "config_declares": {"N_DISRUPTED_ORACLE": P.N_DISRUPTED_ORACLE,
                                         "M_SCENARIOS": P.M_SCENARIOS}},
        # The parameters that define the objective itself; a figure comparing two runs is only
        # meaningful when these agree.
        "objective": {"MU": P.MU, "RHO": P.RHO, "KAPPA": P.KAPPA, "C_MAX": P.C_MAX,
                      "UPEN_FACTOR": P.UPEN_FACTOR, "F1_ACTIVE_ONLY": P.F1_ACTIVE_ONLY,
                      "DELTA_T_H": P.DELTA_T_H, "UE_RGAP": P.UE_RGAP,
                      "UE_MAX_ITER": P.UE_MAX_ITER, "UE_WARM_START": P.UE_WARM_START},
        "code": _git_revision(Path(__file__).resolve().parent.parent),
    }
    meta.update(extra)
    (config_dir(out_dir) / meta_name(method, shared_dir)).write_text(
        json.dumps(meta, indent=2, default=str), encoding="utf-8")
    return meta


def source_meta(paths):
    """Describe the files a derived result was built from: each one's path relative to the
    repository, and the run_meta.json sitting beside it if there is one.

    A comparison figure is only as reproducible as its inputs. Recording which files fed it, and
    under which commit each of those was produced, is what makes a mismatch visible -- two methods
    compared across a code change look exactly like two methods that genuinely differ.
    """
    root = Path(__file__).resolve().parent.parent
    out = []
    for p in map(Path, paths):
        if not p.exists():
            continue
        try:
            rel = str(p.relative_to(root)).replace("\\", "/")
        except ValueError:
            rel = str(p)
        entry = {"path": rel}
        # Where the metadata for this file lives. Optima now sit in results/, so the metadata is
        # in the SIBLING config/ folder: under the method's own name (the shared greedy case) or
        # the plain name (a method that owns its directory). The legacy cache/ location is kept
        # as a fallback so archived pre-reorganization results remain attributable. A candidate
        # counts only if it NAMES this method, so a wrong neighbour is skipped rather than
        # silently attributed.
        method = p.stem[: -len("_optima")] if p.stem.endswith("_optima") else p.stem
        scale_root = p.parent.parent if p.parent.name == "results" else p.parent
        scale = scale_root.name
        cands = []
        for sub in ("config", "cache"):
            cands += [scale_root / sub / f"{method}_run_meta.json", scale_root / sub / "run_meta.json"]
        for pat in (f"*/{scale}/config/run_meta.json", f"*/*/{scale}/config/run_meta.json"):
            cands += sorted((root / "outputs").glob(pat))
        for sib in cands:
            if not sib.exists():
                continue
            try:
                src = json.loads(sib.read_text(encoding="utf-8"))
            except Exception:
                continue
            if src.get("method") != method:
                continue
            entry["produced_by"] = {"method": src.get("method"),
                                    "written_at": src.get("written_at"),
                                    "code": src.get("code")}
            break
        out.append(entry)
    return out


def slot_rows(scenario, res, **tag):
    """Flatten one evaluation's per-slot trace into rows for a {method}_slots.csv.

    `res` must come from an evaluator called with collect_traces=True. Each row is one slot of one
    scenario: the accessibility term the objective averages, how many segments were still damaged,
    and how much demand was being served. Those three together are what a recovery curve needs.
    Any `tag` keywords are copied onto every row, which is how a shared file distinguishes
    variants.

    The rows also reconstitute the reported objective rather than merely illustrating it: F1 is the
    mean of `g` over the rows of a scenario, restricted to rows with n_damaged > 0 when
    P.F1_ACTIVE_ONLY is set. A recovery curve drawn from this file therefore cannot silently
    disagree with the F in the optima file beside it.
    """
    tr = res.get("traces")
    if tr is None:
        return []
    return [dict(scenario=int(scenario), slot=int(r.k), g=float(r.f1_term),
                 n_damaged=int(r.n_damaged), demand=float(r.total_demand), **tag)
            for r in tr.itertuples(index=False)]
