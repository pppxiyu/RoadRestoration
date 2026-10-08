"""One authority for experiment, method, analysis, and independent-study output paths.

An experiment directory is a shared scoring ruler. All method runs under it use the
same public problem specification and frozen evaluation sample. Method training seeds
belong below data/methods/, while canonical comparisons and figures belong below
data/analysis/. Copies of selected figures are placed directly in the experiment
folder for browsing.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from src import config as P
from src.environment.behavior_models import select_behavior_model

ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = ROOT / "outputs"

_RULE_KEYS = (
    "CAP_RETAIN", "SPEED_RETAIN", "SEVER_SEVERITY", "F1_ACTIVE_ONLY",
    "RHO", "KAPPA", "UPEN_FACTOR", "C_MAX", "MU", "DELTA_T_H",
    "M_SCENARIOS", "SEED", "EVAL_SAMPLING", "UE_RGAP_DEF",
    "UE_MAX_ITER_DEF", "UE_RGAP", "UE_MAX_ITER", "UE_WARM_START",
    "DUR_MEAN", "DUR_SD", "DUR_TRUNC_MULT", "SEVERITY_CONFUSION",
    "ACCESS_DEPOT",
)


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def experiment_identity(n: int) -> tuple[str, dict]:
    """Human-readable name plus exact fingerprint of the common scoring rules."""
    n = int(n)
    model = select_behavior_model(P.BEHAVIOR_MODEL, P.DELTA_T_H, for_methods=True)
    network = ROOT / "data" / "siouxfalls_toy" / "network"
    sources = {}
    for name in ("edges.csv", "od_pairs.csv", "nodes.csv"):
        path = network / name
        sources[name] = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
    rules = {name: _jsonable(getattr(P, name)) for name in _RULE_KEYS}
    rules["damage_selection"] = "flow_amplified_v1" if n >= 16 else "flow_spread_v1"
    identity = {"n_disrupted": n, "rules": rules, "network_sha256": sources}
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:8]
    hours = f"{float(P.DELTA_T_H):g}h"
    name = (f"n{n}_flow-{model.name}_{hours}_"
            f"{P.EVAL_SAMPLING}{P.M_SCENARIOS}_s{P.SEED}_{digest}")
    return name, identity


def experiment_dir(n: int | None = None) -> Path:
    n = P.N_DISRUPTED_ORACLE if n is None else int(n)
    return OUTPUTS / "experiments" / experiment_identity(n)[0]


def experiment_data_dir(n: int | None = None) -> Path:
    """Machine-readable records are hidden one level below the experiment gallery."""
    return experiment_dir(n) / "data"


def method_dir(method: str, n: int | None = None, seed: int | None = None) -> Path:
    seed = P.SEED if seed is None else int(seed)
    return experiment_data_dir(n) / "methods" / f"{method}_seed{seed}"


def selected_method_dir(method: str, n: int | None = None) -> Path:
    """Return the run selected for comparison, defaulting to the standard seed."""
    selection = experiment_data_dir(n) / "methods" / "selected_runs.json"
    choices = json.loads(selection.read_text(encoding="utf-8")) if selection.exists() else {}
    return method_dir(method, n, choices.get(method, P.SEED))


def analysis_dir(n: int | None = None) -> Path:
    return experiment_data_dir(n) / "analysis"


def study_dir(study: str) -> Path:
    return OUTPUTS / "studies" / study


def publish_experiment_gallery(n: int | None = None, *, directory=None) -> dict:
    """Copy current figures to a visual-first view without moving canonical run files.

    The gallery index records exactly which copies this function owns. A manually
    edited figure is never overwritten or removed silently.
    """
    exp = experiment_dir(n) if directory is None else Path(directory)
    data = exp / "data"
    exp.mkdir(parents=True, exist_ok=True)
    sources = {}
    for src in sorted((data / "analysis").glob("*.png")):
        sources[src.name] = src
    for src in sorted((data / "methods").glob("*/*.png")):
        sources[f"method_figures/{src.parent.name}__{src.name}"] = src
    for src in sorted((data / "analysis" / "behavior").glob("*/*.png")):
        sources[f"behavior_figures/{src.parent.name}__{src.name}"] = src
    index_path = data / "gallery_index.json"
    previous = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else {}
    current = {}
    for rel, src in sources.items():
        dest = exp / rel
        if dest.is_symlink() or not dest.resolve().is_relative_to(exp.resolve()):
            raise ValueError(f"gallery target is outside the experiment: {dest}")
        digest = hashlib.sha256(src.read_bytes()).hexdigest()
        if dest.exists():
            existing = hashlib.sha256(dest.read_bytes()).hexdigest()
            prior = previous.get(rel, {}).get("sha256")
            if existing != digest and existing != prior:
                raise FileExistsError(f"gallery figure was edited outside the publisher: {dest}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists() or hashlib.sha256(dest.read_bytes()).hexdigest() != digest:
            shutil.copy2(src, dest)
        current[rel] = {"source": str(src.relative_to(exp)).replace("\\", "/"),
                        "sha256": digest}
    for rel, record in previous.items():
        if rel in current:
            continue
        old = exp / rel
        if old.is_symlink() or not old.resolve().is_relative_to(exp.resolve()):
            raise ValueError(f"indexed gallery target is outside the experiment: {old}")
        if old.is_file() and hashlib.sha256(old.read_bytes()).hexdigest() == record["sha256"]:
            old.unlink()  # only a byte-identical, indexed copy; canonical source is untouched
    data.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(current, indent=2), encoding="utf-8")
    return {"experiment": str(exp), "figures": len(current)}


def daily_experiment_identity(problem, simulator, splits):
    """Keep both training mechanisms under their ONE complete testing environment."""
    from src.problem.daily import public_spec
    identity = dict(problem=public_spec(problem), traffic_fingerprint=simulator.fingerprint,
                    evaluation_behavior="response7d", disconnected_flow="unserved",
                    objective="sum_days(0.5*time_loss+0.5*flow_loss); then equal scenario mean",
                    scenario_signatures={name: [s.signature for s in sample] for name, sample in splits.items()})
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:8]
    name = (f"n{problem['n']}_discovery_response7d_1day_"
            f"lhs{len(splits['test'])}_s{problem['seed']}_{digest}")
    return OUTPUTS / "experiments" / name, identity


_OLD_METHODS = {
    "01-brute-force": "oracle",
    "02-rule-based": "rule-based",
    "03-pretrain_milp": "milp",
    "04-ga": "ga",
    "01-rl_s2v": "rl_s2v",
}


def resolve_output_dir(base, n: int | None = None, seed: int | None = None) -> Path:
    """Map historical output bases to the shared experiment layout.

    Kept for older runner signatures and command-line entry points. New code should use
    method_dir or analysis_dir directly. An arbitrary non-output base keeps its old nN
    suffix, so caller-provided scratch directories remain useful.
    """
    base = Path(base)
    n = P.N_DISRUPTED_ORACLE if n is None else int(n)
    try:
        rel = base.resolve().relative_to(OUTPUTS.resolve()).parts
    except ValueError:
        return base / f"n{n}"
    if len(rel) >= 2 and rel[0] == "experiments":
        return base
    if rel and rel[0] == "04-comparison":
        return analysis_dir(n)
    if len(rel) >= 2 and rel[0] in ("02-baselines", "03-rl"):
        if rel[1] == "02-rl_s2v_saa" and len(rel) >= 3:
            return method_dir(f"rl_s2v_saa{rel[2].removeprefix('pool')}", n, seed)
        if rel[1] == "03-rl_s2v_saa_adaptive" and len(rel) >= 3:
            return method_dir(f"rl_s2v_saa{rel[2].removeprefix('pool')}_adaptive", n, seed)
        if rel[1] in _OLD_METHODS:
            return method_dir(_OLD_METHODS[rel[1]], n, seed)
    return base / f"n{n}"
