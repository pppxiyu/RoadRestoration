"""Read-only checks for the local, no-solver-rerun repository reorganization.

Run ``python -m src.experiment.verify_reorganization`` from the repository root.
The inventory proves that migrated historical files, including model weights,
still match their pre-refactor snapshot byte for byte. Comparison CSVs are
checked separately because their provenance files can be refreshed later.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from src import config as P
from src.environment.behavior_models import select_behavior_model
from src.experiment.layout import ROOT, experiment_dir, study_dir


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def verify() -> dict:
    inventory = _rows(ROOT / "outputs" / "data" / "migration_inventory.csv")
    if not inventory:
        raise ValueError("migration inventory is empty")
    for row in inventory:
        source, dest = ROOT / row["source"], ROOT / row["destination"]
        expected = row["sha256"]
        if not source.is_file() or not dest.is_file():
            raise FileNotFoundError(f"missing historical or migrated file: {row}")
        if _sha(source) != expected or _sha(dest) != expected:
            raise ValueError(f"historical byte mismatch: {row}")

    snapshot = ROOT / ".legacy" / "data_before_modular_reorg_20261006" / "siouxfalls_toy"
    current = ROOT / "data" / "siouxfalls_toy"
    data_files = 0
    for old in sorted(snapshot.rglob("*")):
        if not old.is_file():
            continue
        rel = old.relative_to(snapshot)
        if rel.parts[0] not in {"disruption", "figures", "network", "raw"}:
            # Source-code and README edits are part of the refactor, not immutable data.
            continue
        if rel.parts[0] == "disruption":
            dest = current / "instances" / Path(*rel.parts[1:])
        elif rel.parts[0] == "figures":
            dest = study_dir("problem_setting") / "source_figures" / Path(*rel.parts[1:])
        else:
            dest = current / rel
        if not dest.is_file() or _sha(dest) != _sha(old):
            raise ValueError(f"source data mismatch: {rel}")
        data_files += 1

    scales = {}
    gallery_figures = 0
    original = ROOT / ".legacy" / "outputs_before_modular_reorg_20261006"
    for n in (6, 11, 17, 23):
        # Historical evidence is located by its FROZEN manifest, never by today's
        # duration parameters or the now-retired problem's execution route.
        matches = []
        for path in (ROOT / "outputs/experiments").glob("*/data/manifest.json"):
            saved = json.loads(path.read_text(encoding="utf-8"))
            if (saved.get("source_snapshot") == ".legacy/outputs_before_modular_reorg_20261006"
                    and saved.get("problem_identity", {}).get("n_disrupted") == n):
                matches.append((path.parents[1], saved))
        if len(matches) != 1:
            raise ValueError(f"expected one frozen migrated experiment at n={n}, found {len(matches)}")
        exp, manifest = matches[0]
        if manifest["behavior_model"] != "damage_shortfall" or manifest["time_unit_hours"] != 3.0:
            raise ValueError(f"environment identity mismatch at n={n}")
        old = _rows(original / "04-comparison" / f"n{n}" / "results" / "comparison.csv")
        new = _rows(exp / "data" / "analysis" / "results" / "comparison.csv")
        if old != new:
            raise ValueError(f"saved comparison values changed at n={n}")
        durations = _rows(exp / "data" / "scenarios" / "recorded_test_durations.csv")
        if len(durations) != 50 or len(old) != 50:
            raise ValueError(f"incomplete evaluation sample at n={n}")
        gallery = json.loads((exp / "data" / "gallery_index.json").read_text(encoding="utf-8"))
        if not gallery or "performance_distribution.png" not in gallery:
            raise ValueError(f"experiment gallery is incomplete at n={n}")
        for relative, record in gallery.items():
            figure = exp / relative
            source = exp / record["source"]
            if not figure.is_file() or not source.is_file():
                raise FileNotFoundError(f"gallery figure or canonical source missing: {relative}")
            if _sha(figure) != record["sha256"] or _sha(source) != record["sha256"]:
                raise ValueError(f"gallery figure differs from canonical source: {relative}")
            gallery_figures += 1
        scales[n] = {"scenarios": len(old), "methods": manifest["evaluation"]["method_count"],
                     "horizon_slots": manifest["horizon_slots"],
                     "gallery_figures": len(gallery)}
    for path in (ROOT / "outputs").rglob("*"):
        if path.is_file() and path.suffix.lower() in {
                ".csv", ".json", ".npz", ".npy", ".pkl", ".pt", ".txt", ".log"}:
            if "data" not in path.relative_to(ROOT / "outputs").parts:
                raise ValueError(f"machine-readable file outside a data subfolder: {path}")
    assert select_behavior_model("elastic_daily", 24.0, for_methods=False).name == "elastic_daily"
    previous = P.BEHAVIOR_MODEL
    try:
        P.BEHAVIOR_MODEL = "elastic_daily"
        try:
            experiment_dir(11)
        except ValueError:
            pass
        else:
            raise ValueError("unconnected daily behavior option was admitted to method comparison")
    finally:
        P.BEHAVIOR_MODEL = previous
    return {"verified_historical_output_files": len(inventory),
            "verified_source_data_files": data_files, "experiments": scales,
            "comparison_values_unchanged": True, "behavior_option_guard": True,
            "verified_gallery_figures": gallery_figures,
            "machine_readable_files_confined_to_data": True}


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
