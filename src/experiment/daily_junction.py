"""Train the approved junction-OD Q-network; reuse the completed GA unchanged.

Run: python -X utf8 -u -m src.experiment.daily_junction
"""
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import time
import zipfile

import numpy as np
import pandas as pd
import torch

from src.environment.daily import DailyTraffic, recovery_fraction
from src.environment.daily_history import HistoryObservedEpisode, DemandMemoryObservedEpisode
from src.experiment.daily import (prepare, save_problem, _write_json, _source_digest,
                                  _source_files, _save_evaluation)
from src.experiment.daily_ga import single_runner
from src.experiment.daily_ga_saa10 import compare as compare_saved_ga
from src.experiment.daily_od_history import baseline_hashes, check_contract, BASE_RL, GA
from src.experiment.layout import ROOT, publish_experiment_gallery
from src.methods.daily_junction import JunctionEncoder
from src.methods.daily_s2v import DailyQ, rollout, train, POLICY_DELIVERY


METHOD = "rl_s2v_saa64_adaptive_train_response7d_roadclass_junction_od8_seed42"
MEMORY_METHOD = "rl_s2v_saa64_adaptive_train_response7d_roadclass_junction_od16_demand_memory_seed42"
_WORKER = None


def protected_hashes(folder, demand_memory=False):
    hashes = baseline_hashes(folder)
    if demand_memory:
        root = folder / "data/methods" / METHOD
        for path in root.rglob("*"):
            if path.is_file():
                hashes[str(path.relative_to(folder))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def initialize_worker(problem, cache, hp, demand_memory=False, bidirectional=False):
    global _WORKER
    torch.set_num_threads(1)
    simulator = DailyTraffic(problem, cache)
    encoder = JunctionEncoder(problem, simulator.ctx, simulator.gamma, demand_memory=demand_memory,
                              bidirectional=bidirectional)
    _WORKER = simulator, encoder, DailyQ(encoder, hp)


def run_world(request):
    scenario, weights, epsilon, seed = request
    simulator, encoder, model = _WORKER
    model.net.load_state_dict(weights, strict=True)
    model.net.eval()
    calls, hits, started = simulator.solve_calls, simulator.cache_hits, time.perf_counter()
    episode_class = DemandMemoryObservedEpisode if encoder.demand_memory else HistoryObservedEpisode
    episode = episode_class(simulator, scenario, "response7d", disconnected_flow="unserved")
    result = rollout(episode, encoder, model, epsilon=epsilon, rng=np.random.RandomState(seed))
    result["runtime"] = dict(seconds=time.perf_counter() - started,
        ue_solves=simulator.solve_calls - calls, cached_days=simulator.cache_hits - hits)
    return result


def verify_saved_inputs(folder, root, encoder):
    with np.load(root / "results/test_junction_inputs.npz") as extra, \
            np.load(root / "results/test_policy_inputs.npz") as original:
        daily = pd.read_csv(root / "results/test_daily.csv")
        actions = pd.read_csv(root / "results/test_actions.csv")
        scenarios = pd.read_csv(root / "results/test_results.csv").scenario.tolist()
        for i in range(50):
            od, x, g = extra[f"scenario_{i}_od"], original[f"scenario_{i}_x"], original[f"scenario_{i}_g"]
            if od.shape != (len(x), len(encoder.zone_ids), len(encoder.zone_ids), encoder.raw_od_dimension) or x.shape[1:] != (38, 16):
                raise ValueError("saved raw test input dimensions differ from the approved design")
            if g.shape != (len(x), 7) or not all(np.isfinite(a).all() for a in (od, x, g)):
                raise ValueError("invalid saved global/OD inputs")
            np.testing.assert_array_equal(g[:, 6], np.float32(encoder.base_gamma))
            np.testing.assert_array_equal(od[:, ~encoder.valid], 0)
            np.testing.assert_allclose(od[:, :, :, 1:6],
                np.broadcast_to(encoder.template[None, :, :, 1:6], od[:, :, :, 1:6].shape), rtol=0, atol=0)
            if encoder.demand_memory:
                assert (od[:, :, :, 6] >= 0).all()
                assert np.isin(od[:, :, :, 8], [0, 1]).all()
                np.testing.assert_array_equal(od[0, :, :, 7], 0)
                np.testing.assert_array_equal(od[0, :, :, 8], encoder.valid)
                days = actions.loc[actions.scenario == scenarios[i], "day"].to_numpy()
                assert len(days) == len(od) and days[0] == 1
                oi, di = encoder.od_indices.T
                expected_reference = (recovery_fraction(days, encoder.rules)[:, None]
                                      * encoder.base.H0[None, :] / encoder.demand_scale)
                np.testing.assert_allclose(od[:, oi, di, 6], expected_reference, rtol=1e-6, atol=1e-7)
                previous_reference = (recovery_fraction(np.maximum(1, days - 1), encoder.rules)[:, None]
                                      * encoder.base.H0[None, :] / encoder.demand_scale)
                disconnected = od[:, oi, di, 8] == 0
                np.testing.assert_allclose(od[:, oi, di, 7][disconnected],
                                           -previous_reference[disconnected], rtol=1e-6, atol=1e-7)
            np.testing.assert_array_equal(od[0, :, :, 0], 0)
            if (np.diff(od[:, :, :, 0], axis=0) < -1e-7).any():
                raise ValueError("cumulative history cannot decrease")
        if len(daily.scenario.unique()) != 50:
            raise ValueError("saved test daily trajectories are incomplete")


def summarize(folder, encoder, hashes):
    methods = folder / "data/methods"
    suffix = "junction_memory" if encoder.demand_memory else "junction"
    root = methods / (MEMORY_METHOD if encoder.demand_memory else METHOD)
    old = check_contract(folder, encoder, json.loads((folder / "data/manifest.json").read_text())["identity"])
    current = json.loads((root / "config/training.json").read_text())
    for key in ("hyperparameters", "stop", "ep_cap", "seed", "reward", "discount"):
        if current[key] != old[key]:
            raise ValueError(f"algorithm contract differs: {key}")
    if current["delivery"] != POLICY_DELIVERY:
        raise ValueError("preserve historical final-policy results; new evaluations require validation selection")
    if protected_hashes(folder, encoder.demand_memory) != hashes:
        raise ValueError("saved GA or original Full RL changed")
    verify_saved_inputs(folder, root, encoder)
    comparison = compare_saved_ga(folder, methods / GA, root,
        output_stem=f"ga_saa10_vs_{suffix}_full_rl", rl_label=("Full RL\nwith demand memory" if encoder.demand_memory else "Full RL\nwith junction OD encoding"))
    history = pd.read_csv(root / "log/training.csv")
    result = dict(**comparison, baseline_files_unchanged=True, saved_inputs_verified=True,
        training=json.loads((root / "results/training_summary.json").read_text()),
        evaluation=json.loads((root / "results/evaluation_summary.json").read_text()),
        final_validation_objective=float(history.validation_objective.dropna().iloc[-1]),
        best_validation_objective=float(history.best_validation_objective.iloc[-1]),
        parameter_count=sum(p.numel() for p in DailyQ(encoder).net.parameters()),
        limitation="one training/search seed; encoder architecture and selected inputs change together")
    _write_json(folder / f"data/analysis/results/{suffix}_rl_comparison.json", result)
    return result


from src.experiment.retired import retired_problem


@retired_problem
def main(demand_memory=False):
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "1"
    torch.set_num_threads(1)
    problem, splits, simulator, plain, folder, identity = prepare(problem_variant="poi_unmet")
    encoder = JunctionEncoder(problem, simulator.ctx, simulator.gamma, demand_memory=demand_memory)
    method = MEMORY_METHOD if demand_memory else METHOD
    suffix = "junction_memory" if demand_memory else "junction"
    simulator.close()
    save_problem(folder, problem, splits, identity, plain)
    old = check_contract(folder, encoder, identity)
    source, source_hashes = _source_digest()
    root = folder / "data/methods" / method
    for part in ("config", "log", "results"):
        (root / part).mkdir(parents=True, exist_ok=True)
    plan = dict(method=method, source_sha256=source, experiment_identity=identity,
        feature_schema=encoder.specification(), baseline_hashes=protected_hashes(folder, demand_memory),
        hyperparameters=old["hyperparameters"], stop=old["stop"], ep_cap=old["ep_cap"], seed=42,
        training_scenarios=64, validation_scenarios=11, test_scenarios=50, workers=4,
        initialization="random from scratch, no GA demonstrations", ga_recomputed=False,
        model_selection=POLICY_DELIVERY,
        figure=dict(backend="Python/matplotlib", format="PNG", archetype="quantitative comparison",
            purpose="compare saved GA and new RL without presuming improvement", panels=1,
            evidence="same 50 test worlds; boxplots and all points", box="quartiles", whiskers="1.5 IQR",
            annotation="previously requested mean and median", title=None))
    plan_path = folder / f"data/analysis/config/{suffix}_input_plan.json"
    status = root / "log/status.json"
    overall = folder / f"data/log/{suffix}_rl_status.json"
    with single_runner(root / "log/runner.lock"):
        if plan_path.exists() and json.loads(plan_path.read_text(encoding="utf-8")) != json.loads(json.dumps(plan)):
            raise ValueError("frozen source, baseline or configuration changed; do not overwrite the run")
        _write_json(plan_path, plan)
        _write_json(root / "config/source_hashes.json", source_hashes)
        archive = root / "config/source_snapshot.zip"
        if not archive.exists():
            with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
                for path in _source_files():
                    bundle.write(path, path.relative_to(ROOT))
        try:
            complete = status.exists() and json.loads(status.read_text()).get("status") == "complete"
            if not complete:
                _write_json(overall, dict(status="training", method=method, pid=os.getpid()))
                _write_json(status, dict(status="training", pid=os.getpid()))
                from src.analysis.viz.daily_viz import make_training_figures
                with ProcessPoolExecutor(max_workers=4, initializer=initialize_worker,
                        initargs=(problem, ROOT / ".cache/daily_traffic", old["hyperparameters"], demand_memory)) as pool:
                    def batch(split, indices, weights, epsilon, seeds):
                        if split not in ("train", "validation"):
                            raise ValueError("training may not request test data")
                        return list(pool.map(run_world, [(splits[split][i], weights, epsilon, seed)
                                                        for i, seed in zip(indices, seeds)]))

                    policy, training = train(encoder, batch, train_count=64, validation_count=11,
                        seed=42, run_dir=root, hp=old["hyperparameters"], stop_params=old["stop"],
                        ep_cap=old["ep_cap"], redraw=make_training_figures,
                        identity=dict(experiment=identity, training_behavior="response7d",
                                      input_variant=("junction_od16_demand_memory" if demand_memory else "junction_od8"), code_sha256=source))
                    _write_json(status, dict(status="testing", **training))
                    started = time.perf_counter()
                    trajectories = list(pool.map(run_world, [(s, policy.net.state_dict(), 0., 0)
                                                             for s in splits["test"]]))
                    seconds = time.perf_counter() - started
                    evaluation = _save_evaluation(root, trajectories, seconds)
                    np.savez_compressed(root / "results/test_junction_inputs.npz",
                        **{f"scenario_{i}_od": np.stack([s["od"] for s in t["states"]])
                           for i, t in enumerate(trajectories)})
                    _write_json(status, dict(status="complete", **training, **evaluation))
            result = summarize(folder, encoder, plan["baseline_hashes"])
            publish_experiment_gallery(directory=folder)
            _write_json(overall, dict(status="complete", **result))
            print(json.dumps(result, indent=2), flush=True)
        except BaseException as error:
            _write_json(overall, dict(status="interrupted_or_failed", error_type=type(error).__name__,
                                    message=str(error), pid=os.getpid()))
            raise


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--demand-memory", action="store_true")
    main(demand_memory=parser.parse_args().demand_memory)
