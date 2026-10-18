"""Input-only checks, using training worlds, before the independent RL rerun."""
from dataclasses import replace
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import tempfile

import numpy as np
import torch

from src.environment.daily import DailyEpisode
from src.environment.daily_history import HistoryObservedEpisode
from src.environment.poi_response import sensitivity
from src.experiment.daily import (prepare, save_problem, _write_json, _source_digest,
                                  _initialize_worker, _run_world)
from src.experiment.daily_od_history import check_contract, baseline_hashes, BASE_RL
from src.methods.daily_s2v import DailyEncoder, DailyQ, train


def main():
    torch.set_num_threads(1)
    problem, splits, sim, encoder, folder, identity = prepare(
        problem_variant="poi_unmet", od_history_input=True)
    try:
        save_problem(folder, problem, splits, identity, encoder)
        old_contract = check_contract(folder, encoder, identity)
        baseline = baseline_hashes(folder)
        plain = DailyEncoder(problem, sim.ctx, road_class_input=True)
        assert json.loads(json.dumps(plain.specification())) == old_contract["feature_schema"]
        assert encoder.global_dimension == 6 + len(sim.ctx["od_pairs"]) * 6
        normal_episode = DailyEpisode(sim, splits["train"][0], "response7d", disconnected_flow="unserved")
        episode = HistoryObservedEpisode(sim, splits["train"][0], "response7d", disconnected_flow="unserved")
        first = encoder.encode(episode.observation())
        snapshots, picks, rewards = [], [], []
        while not episode.finished:
            observation = episode.observation()
            original_state, enriched = plain.encode(observation), encoder.encode(observation)
            np.testing.assert_array_equal(original_state["x"], enriched["x"])
            np.testing.assert_array_equal(original_state["g"], enriched["g"][:6])
            block = enriched["g"][6:].reshape(len(encoder.od_pairs), 6)
            np.testing.assert_allclose(block[:, 1:] * encoder.poi_count_scale,
                                       encoder.od_poi_counts, rtol=1e-6, atol=1e-5)
            # Independent observer history agrees with the environment's prior-day
            # state, and reconstructs the sensitivity used for the current day.
            current_shortfall = np.maximum(0, 1 - observation.demand / encoder.H0)
            np.testing.assert_allclose(observation.cumulative_unmet + current_shortfall,
                                       episode.traffic.cumulative_unmet, rtol=0, atol=1e-12)
            np.testing.assert_array_equal(sensitivity(sim.gamma, observation.cumulative_unmet,
                sim.od_function_weights, sim.response_spec), episode.traffic.gamma_od)
            road = min(observation.candidates, key=lambda e: (-problem["baseline_flow"][e], e))
            snapshots.append(enriched)
            picks.append(enriched["roads"].index(road))
            _, reward, _ = episode.step(road)
            rewards.append(reward)
            _, original_reward, _ = normal_episode.step(road)
            assert original_reward == reward
            assert episode.day == normal_episode.day and episode.finished == normal_episode.finished
        assert episode.result() == normal_episode.result()
        np.testing.assert_array_equal(first["g"][6:].reshape(-1, 6)[:, 0], 0)
        assert snapshots[1]["day"] > 2
        assert (snapshots[1]["g"][6:].reshape(-1, 6)[:, 0] > 0).any()
        # Same destination, distinct origins retain independently addressable histories.
        fresh = HistoryObservedEpisode(sim, splits["train"][0], "response7d", disconnected_flow="unserved")
        observed = fresh.observation()
        custom_history = np.zeros_like(encoder.H0)
        same_destination = [i for i, (_, d) in enumerate(encoder.od_pairs) if d == encoder.od_pairs[0][1]]
        custom_history[same_destination[0]] = 3.5
        modified = encoder.encode(replace(observed, cumulative_unmet=custom_history))
        changed = np.flatnonzero(modified["g"] != first["g"])
        np.testing.assert_array_equal(changed, [6 + 6 * same_destination[0]])
        assert modified["g"][6 + 6 * same_destination[1]] == 0
        # Every added history channel can receive a gradient; POI count channels
        # keep magnitude, e.g. double facility counts produce double inputs.
        torch.manual_seed(42)
        model = DailyQ(encoder)
        model(modified).square().sum().backward()
        assert model.net.th8.weight.grad[:, 6 + 6 * same_destination[0]].abs().sum() > 0
        assert model.net.th8.in_features == encoder.global_dimension
        # Windows workers must construct the enriched observer and network, not
        # silently use the old six-global-input path in another process.
        with ProcessPoolExecutor(max_workers=1, initializer=_initialize_worker,
                                 initargs=(problem, None, None, True, True, True)) as pool:
            worker_result = pool.submit(_run_world, (splits["train"][1], "response7d",
                model.net.state_dict(), 0.0, 0)).result()
        assert all(s["g"].shape == (encoder.global_dimension,) for s in worker_result["states"])
        assert np.isfinite(worker_result["result"]["objective"])
        # Exercise actual optimizer/replay checkpoint continuation with the new
        # shape; no extra traffic simulation or test-world model selection.
        trajectory = dict(states=snapshots, picks=picks, rewards=rewards, result=episode.result())
        def batch(split, indices, weights, epsilon, seeds):
            assert split in ("train", "validation")
            return [trajectory for _ in indices]
        options = dict(train_count=2, validation_count=1, seed=42, identity={"input_check": True},
            hp=dict(pool_n=2, batch_worlds=1, updates_per_ep=1, batch=2),
            stop_params=dict(n_val=1, ep_min=99, probe_every=1), ep_cap=3)
        with tempfile.TemporaryDirectory(prefix="od_history_check_") as temporary:
            direct, _ = train(encoder, batch, run_dir=Path(temporary) / "direct", **options)
            train(encoder, batch, run_dir=Path(temporary) / "resumed", pause_after=1, **options)
            resumed, _ = train(encoder, batch, run_dir=Path(temporary) / "resumed", **options)
            for key, value in direct.net.state_dict().items():
                assert torch.equal(value, resumed.net.state_dict()[key]), key
            a = torch.load(Path(temporary) / "direct/results/training_latest.pt", weights_only=False)
            b = torch.load(Path(temporary) / "resumed/results/training_latest.pt", weights_only=False)
            for ra, rb in zip(a["records"], b["records"]):
                for key in ra:
                    if key not in ("seconds", "cumulative_seconds"):
                        assert ra[key] == rb[key], key
        assert baseline_hashes(folder) == baseline
        audit = dict(source_sha256=_source_digest()[0], experiment=str(folder),
            original_full_rl=BASE_RL, unchanged_scoring_identity=True, frozen_scenarios_verified=True,
            unchanged_original_feature_schema=True, unchanged_old_channels=True,
            exact_same_schedule_rewards_and_objective=True, history_reconstructs_daily_gamma=True,
            history_includes_busy_days=True, distinct_od_histories=True, first_day_history_zero=True,
            destination_counts_retained=True, resume_weights_and_non_time_metrics_exact=True,
            subprocess_rollout_passed=True,
            baseline_files_unchanged=True, node_dimension=encoder.node_dimension,
            global_dimension=encoder.global_dimension, od_count=len(encoder.od_pairs),
            poi_count_scale=encoder.poi_count_scale, history_scale_days=encoder.time_scale,
            old_parameter_count=sum(p.numel() for p in DailyQ(plain).net.parameters()),
            new_parameter_count=sum(p.numel() for p in model.net.parameters()))
        _write_json(folder / "data/analysis/config/od_history_input_audit.json", audit)
        print(json.dumps(audit, indent=2), flush=True)
    finally:
        sim.close()


if __name__ == "__main__":
    main()
