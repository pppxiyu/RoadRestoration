"""Standard full-input RL pipeline, from random base training to final testing.

The base reward and network reproduce the complete-model protocol. Improvement
uses the later all-position, scheduler and best-so-far protocol. Historical
models remain historical and are never loaded implicitly by this entry point.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform
import time
import zipfile

import numpy as np
import torch

from src.experiment.daily import prepare, save_problem, _source_digest, _source_files, _write_json, _save_evaluation
from src.experiment import daily_rl_workers as workers_api
from src.methods.daily_junction import JunctionEncoder
from src.methods.daily_s2v import DailyQ, train
from src.methods.daily_improvement import (ImprovementSettings, improve, freeze_json,
                                          atomic_pickle, read_pickle, atomic_torch, weights_digest)
from src.problem.daily import improvement_splits, CURRENT_PROBLEM


BASE_REWARD = ("cumulative no-further-repair system loss minus selected-single-repair system loss, "
               "from opening day through disaster day 60; no subsequent counterfactual repairs")
STANDARD_VERSION = "dual_od_memory_horizon_all_positions_plateau_best_so_far_v1"


@contextmanager
def runner_lock(path):
    """One writer, on Windows locally and POSIX on the HPC. OS releases on exit."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def run(n=11, training_behavior="response7d", seed=42, workers=4, *, scenario_seed=42,
        problem_variant=CURRENT_PROBLEM, improvement_settings=ImprovementSettings()):
    if training_behavior not in ("natural", "response7d", "both"):
        raise ValueError("choose natural, response7d, or both")
    torch.set_num_threads(1)
    problem, splits, simulator, _, folder, identity = prepare(
        n, scenario_seed, problem_variant=problem_variant)
    try:
        encoder = JunctionEncoder(problem, simulator.ctx, simulator.gamma,
                                  demand_memory=True, bidirectional=True)
    finally:
        simulator.close()
    save_problem(folder, problem, splits, identity, encoder)
    expanded = improvement_splits(problem, splits)
    source_digest, hashes = _source_digest()
    hp = dict(DailyQ(encoder).hp, lr=0.0005)
    if hp["lam"] != 0 or hp["gamma"] != 1:
        raise ValueError("standard base RL requires no teacher loss and undiscounted returns")
    workers = max(1, int(workers))
    behaviors = ("natural", "response7d") if training_behavior == "both" else (training_behavior,)
    for behavior in behaviors:
        name = f"rl_s2v_dual_od16_memory_horizon_paired_train_{behavior}_seed{seed}"
        root = folder / "data/methods" / name
        run_identity = dict(experiment=identity, standard=STANDARD_VERSION,
            training_behavior=behavior, input_variant="dual_od16_memory", code_sha256=source_digest)
        with runner_lock(root / "log/runner.lock"):
            freeze_json(root / "config/source_hashes.json", hashes)
            freeze_json(root / "config/protocol.json", dict(identity=run_identity,
                seed=seed, base_hp=hp, base_reward=BASE_REWARD,
                improvement=asdict(improvement_settings), feature_schema=encoder.specification(),
                improvement_training=[s.record() for s in expanded["train"]],
                validation=[s.record() for s in splits["validation"]],
                test=[s.record() for s in splits["test"]], evaluation_behavior="response7d"))
            # Keep the existing comparison reader's contract at the method root.
            freeze_json(root / "config/training.json", dict(identity=run_identity, seed=seed,
                hyperparameters=hp, base_reward=BASE_REWARD,
                improvement=asdict(improvement_settings), feature_schema=encoder.specification()))
            if (root / "log/complete.json").exists():
                delivery = json.loads((root / "log/complete.json").read_text())
                for rel, digest in delivery["sha256"].items():
                    if hashlib.sha256((root / rel).read_bytes()).hexdigest() != digest:
                        raise ValueError(f"completed deliverable changed: {rel}")
                print("Reusing completed standard RL:", root, flush=True)
                continue
            archive = root / "config/source_snapshot.zip"
            if not archive.exists():
                with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
                    for path in _source_files():
                        bundle.write(path, path.relative_to(Path(__file__).resolve().parents[2]))
            _write_json(root / "log/execution.json", dict(workers=workers,
                python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
                platform=platform.platform(), slurm_job=os.environ.get("SLURM_JOB_ID")))
            try:
                with ProcessPoolExecutor(max_workers=workers, initializer=workers_api.initialize,
                        initargs=(problem, hp, behavior, root / "cache/rewards")) as pool:
                    def evaluate(weights):
                        trajectories = list(pool.map(workers_api.run_world,
                            [(s, weights, 0.0, 0, False) for s in splits["validation"]]))
                        value = float(np.mean([t["result"]["objective"] for t in trajectories]))
                        if not np.isfinite(value):
                            raise FloatingPointError("nonfinite validation mean")
                        return value

                    def batch(split, indices, weights, epsilon, seeds):
                        if split not in ("train", "validation"):
                            raise ValueError("training cannot request test data")
                        return list(pool.map(workers_api.run_world, [
                            (splits[split][i], weights, epsilon, s, split == "train")
                            for i, s in zip(indices, seeds)]))

                    base_dir = root / "base"
                    _write_json(root / "log/status.json", dict(stage="base_training"))
                    base_selection = base_dir / "results/policy_selection.json"
                    if (base_dir / "complete.json").exists():
                        completion = json.loads((base_dir / "complete.json").read_text())
                        for rel, digest in completion["sha256"].items():
                            if hashlib.sha256((base_dir / rel).read_bytes()).hexdigest() != digest:
                                raise ValueError("base completion artifact changed")
                        initial = torch.load(base_dir / "results/model_best_validation.pt", weights_only=True)
                        initial_value = json.loads(base_selection.read_text())["validation_objective"]
                    else:
                        model, _ = train(encoder, batch, train_count=len(splits["train"]),
                            validation_count=len(splits["validation"]), seed=seed, run_dir=base_dir,
                            identity=run_identity, hp=hp, reward_description=BASE_REWARD)
                        initial = model.net.state_dict()
                        initial_value = json.loads(base_selection.read_text())["validation_objective"]
                        files = ("results/model_best_validation.pt", "results/model_final.pt",
                                 "results/training_latest.pt", "results/policy_selection.json",
                                 "results/training_summary.json", "config/training.json")
                        _write_json(base_dir / "complete.json", dict(sha256={
                            rel: hashlib.sha256((base_dir / rel).read_bytes()).hexdigest() for rel in files}))

                    def collect(weights, directory, positions):
                        _write_json(root / "log/status.json", dict(stage="improvement_labels", round=directory.name))
                        digest = weights_digest(weights)
                        records = list(pool.map(workers_api.collect, [
                            (i, s, weights, digest, directory / "labels" / s.scenario_id, positions)
                            for i, s in enumerate(expanded["train"])]))
                        return [row for record in records for row in record["rows"]]

                    best, selection = improve(encoder, hp, initial, initial_value, collect, evaluate,
                        root / "improvement", identity=run_identity, settings=improvement_settings, seed=seed)
                # Only this final handoff opens test worlds. Test traffic always
                # uses the complete environment, including for a natural ablation.
                if not selection["complete"]:
                    raise RuntimeError("incomplete improvement cannot proceed to test")
                results = root / "results"
                results.mkdir(exist_ok=True)
                atomic_torch(results / "model_best_validation.pt", best)
                _write_json(results / "policy_selection.json", dict(selection,
                    model_file="model_best_validation.pt", model_stage="base" if selection["selected_round"] == 0 else "improvement"))
                _write_json(root / "log/status.json", dict(stage="testing"))
                started = time.perf_counter()
                with ProcessPoolExecutor(max_workers=workers, initializer=workers_api.initialize,
                        initargs=(problem, hp, "response7d", None)) as pool:
                    test_path = results / "test_trajectories.pkl"
                    if test_path.exists():
                        saved = read_pickle(test_path)
                        if saved["model_sha256"] != selection["model_sha256"]:
                            raise ValueError("test policy differs from validation selection")
                        trajectories = saved["trajectories"]
                    else:
                        trajectories = list(pool.map(workers_api.run_world,
                            [(s, best, 0.0, 0, False) for s in splits["test"]]))
                        atomic_pickle(test_path, dict(model_sha256=selection["model_sha256"], trajectories=trajectories))
                summary = _save_evaluation(root, trajectories, time.perf_counter() - started)
                base_summary = json.loads((base_dir / "results/training_summary.json").read_text())
                _write_json(results / "training_summary.json", dict(base=base_summary,
                    improvement_rounds=len(selection["rounds"]), selected_round=selection["selected_round"],
                    best_validation_objective=selection["validation_objective"]))
                freeze_json(results / "selection_provenance.json", dict(identity=run_identity, selection=selection))
                complete_files = [p for p in results.iterdir() if p.is_file()]
                _write_json(root / "log/complete.json", dict(sha256={
                    str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in complete_files}))
                _write_json(root / "log/status.json", dict(stage="complete", **summary))
            except BaseException as exc:
                _write_json(root / "log/status.json", dict(stage="failed_or_interrupted",
                    error_type=type(exc).__name__, message=str(exc)))
                raise
    return folder
