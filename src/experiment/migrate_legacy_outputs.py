"""Copy historical results into experiment-first folders without solving again.

The original tree is never edited. Destination collisions with different bytes stop
the migration. Run with ``python -m src.experiment.migrate_legacy_outputs SOURCE``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path

from src.experiment.layout import (ROOT, analysis_dir, experiment_data_dir, experiment_dir,
                         experiment_identity, method_dir, study_dir)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _copy_file(src: Path, dest: Path, inventory: list[dict]) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    digest = _sha(src)
    if (dest.exists() and _sha(dest) != digest and dest.name == "run_meta.json"
            and dest.parent.parent.name == "analysis"):
        # A comparison refresh replaces provenance without replacing historical
        # solver outputs. Keep the pre-refactor record explicitly alongside it.
        dest = dest.with_name("historical_run_meta.json")
    if dest.exists():
        if _sha(dest) != digest:
            raise FileExistsError(f"destination has different content: {dest}")
    else:
        shutil.copy2(src, dest)
    inventory.append({"source": str(src.relative_to(ROOT)).replace("\\", "/"),
                      "destination": str(dest.relative_to(ROOT)).replace("\\", "/"),
                      "sha256": digest})


def _copy_tree(src: Path, dest: Path, inventory: list[dict]) -> None:
    if not src.exists():
        return
    for path in sorted(src.rglob("*")):
        if path.is_file():
            _copy_file(path, dest / path.relative_to(src), inventory)


def _optima_rows(exp: Path):
    for path in sorted((exp / "data" / "methods").glob("*/results/*_optima.csv")):
        # The rule-based run folder contains three distinct methods. The file,
        # rather than the shared folder, is the identity of one delivered policy.
        method = path.stem.removesuffix("_optima")
        with path.open(newline="", encoding="utf-8-sig") as stream:
            for row in csv.DictReader(stream):
                yield method, path, row


def _write_evaluation(exp: Path) -> dict:
    """Normalize saved objectives and verify scenario/duration pairing across methods."""
    rows = []
    reference = {}
    methods = defaultdict(set)
    for method, source, row in _optima_rows(exp):
        scenario = int(row["scenario"])
        duration = row.get("durations", "")
        if duration:
            previous = reference.setdefault(scenario, duration)
            if previous != duration:
                raise ValueError(f"recorded scenario {scenario} has inconsistent durations: {source}")
        methods[method].add(scenario)
        rows.append({
            "method": method, "scenario": scenario,
            "F": row.get("F", row.get("F_milp", "")),
            "F1": row.get("F1", ""), "F2": row.get("F2", ""),
            "time_s": row.get("time_s", ""), "order": row.get("order", ""),
            "durations": duration,
            "source": str(source.relative_to(exp)).replace("\\", "/"),
        })
    if not rows:
        return {"method_count": 0, "scenario_count": 0}
    exp.mkdir(parents=True, exist_ok=True)
    target = exp / "data" / "evaluation.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    scenarios = exp / "data" / "scenarios"
    scenarios.mkdir(exist_ok=True)
    with (scenarios / "recorded_test_durations.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("scenario", "durations_in_edge_id_order"))
        writer.writerows(sorted(reference.items()))
    return {"method_count": len(methods), "scenario_count": len(reference),
            "method_scenario_counts": {name: len(ids) for name, ids in sorted(methods.items())},
            "recorded_duration_alignment": True}


def migrate(source: Path) -> dict:
    source = source.resolve()
    if not source.is_dir() or source == (ROOT / "outputs").resolve():
        raise ValueError("pass a separate, verified snapshot directory as the migration source")
    if ROOT.resolve() not in source.parents:
        raise ValueError("migration source must stay inside the project workspace")
    inventory: list[dict] = []
    groups = (
        ("02-baselines/01-brute-force", "oracle"),
        ("02-baselines/02-rule-based", "rule-based"),
        ("02-baselines/03-pretrain_milp", "milp"),
        ("02-baselines/04-ga", "ga"),
        ("03-rl/01-rl_s2v", "rl_s2v"),
        ("03-rl/02-rl_s2v_saa/pool64", "rl_s2v_saa64"),
        ("03-rl/02-rl_s2v_saa/pool128", "rl_s2v_saa128"),
        ("03-rl/03-rl_s2v_saa_adaptive/pool64", "rl_s2v_saa64_adaptive"),
        ("03-rl/03-rl_s2v_saa_adaptive/pool128", "rl_s2v_saa128_adaptive"),
    )
    scales = set()
    for rel, method in groups:
        for old in sorted((source / rel).glob("n*")):
            if not old.is_dir() or not old.name[1:].isdigit():
                continue
            n = int(old.name[1:])
            scales.add(n)
            meta = old / "config" / "run_meta.json"
            if not meta.exists() and method == "rule-based":
                meta = old / "config" / "flow_run_meta.json"
            seed = int(json.loads(meta.read_text(encoding="utf-8"))["instance"]["seed"]) if meta.exists() else 42
            _copy_tree(old, method_dir(method, n, seed), inventory)
    for old in sorted((source / "04-comparison").glob("n*")):
        if old.is_dir() and old.name[1:].isdigit():
            n = int(old.name[1:])
            scales.add(n)
            _copy_tree(old, analysis_dir(n), inventory)

    setting = source / "01-sim_val_n_problem_setting"
    for old in sorted((setting / "03-problem_setting").glob("n*")):
        if old.is_dir():
            _copy_tree(old, study_dir("problem_setting") / "figures" / old.name, inventory)
    _copy_tree(setting / "02-tolerance_audit", study_dir("traffic_solver") / "tolerance_audit", inventory)
    old_daily = setting / "04-env_behavior"
    daily = study_dir("human_behavior") / "n11_daily"
    for path in sorted(old_daily.glob("*.png")):
        _copy_file(path, daily / path.name, inventory)
    _copy_tree(old_daily / "results", daily / "data" / "results", inventory)
    _copy_tree(old_daily / "config", daily / "data" / "config", inventory)
    for path in sorted((setting / "01-benchmark").glob("*")):
        if path.is_file():
            study = "traffic_solver" if path.name[:2] in ("01", "02", "03") else "human_behavior"
            sub = "benchmark" if study == "traffic_solver" else "gravity"
            dest = study_dir(study) / sub
            if sub == "gravity" and path.suffix.lower() != ".png":
                dest /= "data"
            _copy_file(path, dest / path.name, inventory)
    for path in sorted((setting / "raw").glob("*")):
        if not path.is_file():
            continue
        if path.name.startswith("problem_setting_"):
            dest = study_dir("problem_setting") / "data" / path.name
        elif path.name.startswith("exogenous_"):
            dest = study_dir("human_behavior") / "recovery" / "data" / path.name
        else:
            dest = study_dir("traffic_solver") / "data" / path.name
        _copy_file(path, dest, inventory)

    reports = {}
    for n in sorted(scales):
        exp = experiment_dir(n)
        name, identity = experiment_identity(n)
        score_report = _write_evaluation(exp)
        method_meta = list((exp / "data" / "methods").glob("*/config/*run_meta.json"))
        horizons = set()
        revisions = set()
        for path in method_meta:
            meta = json.loads(path.read_text(encoding="utf-8"))
            if meta.get("instance", {}).get("horizon_T"):
                horizons.add(int(meta["instance"]["horizon_T"]))
            revision = meta.get("code", {}).get("commit")
            if revision:
                revisions.add(revision)
        if len(horizons) > 1:
            raise ValueError(f"method horizons disagree in {name}: {sorted(horizons)}")
        manifest = {
            "experiment_id": name,
            "migration": "copied from verified pre-refactor local snapshot; no solver rerun",
            "source_snapshot": str(source.relative_to(ROOT)).replace("\\", "/"),
            "problem_identity": identity,
            "horizon_slots": next(iter(horizons)) if horizons else None,
            "time_unit_hours": identity["rules"]["DELTA_T_H"],
            "behavior_model": "damage_shortfall",
            "historical_code_revisions": sorted(revisions),
            "historical_run_metadata_preserved": True,
            "validation_scenarios_recorded": False,
            "evaluation": score_report,
        }
        (experiment_data_dir(n) / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8")
        reports[name] = score_report
    report_path = ROOT / "outputs" / "data" / "migration_inventory.csv"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("source", "destination", "sha256"))
        writer.writeheader()
        writer.writerows(inventory)
    return {"files_copied_or_verified": len(inventory), "experiments": reports,
            "inventory": str(report_path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    args = parser.parse_args()
    print(json.dumps(migrate(args.source), indent=2))


if __name__ == "__main__":
    main()
