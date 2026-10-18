"""Check the approved input architecture on training worlds, not test outcomes."""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from src.environment.daily import DailyEpisode
from src.environment.daily_history import HistoryObservedEpisode, DemandMemoryObservedEpisode
from src.environment.poi_response import sensitivity
from src.experiment.daily import prepare, save_problem, _write_json, _source_digest
from src.experiment.daily_junction import initialize_worker, run_world, protected_hashes, METHOD
from src.experiment.daily_od_history import baseline_hashes, check_contract
from src.methods.daily_junction import JunctionEncoder
from src.methods.daily_s2v import DailyQ, train


def equal_tree(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            equal_tree(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            equal_tree(x, y)
    else:
        assert a == b


def main(demand_memory=False, bidirectional=False):
    torch.set_num_threads(1)
    problem, splits, sim, plain, folder, identity = prepare(problem_variant="poi_unmet")
    encoder = JunctionEncoder(problem, sim.ctx, sim.gamma, demand_memory=demand_memory,
                              bidirectional=bidirectional)
    hashes = protected_hashes(folder, demand_memory)
    save_problem(folder, problem, splits, identity, plain)
    old = check_contract(folder, encoder, identity)
    assert json.loads(json.dumps(plain.specification())) == old["feature_schema"]
    if demand_memory:
        legacy_encoder = JunctionEncoder(problem, sim.ctx, sim.gamma)
        legacy_root = folder / "data/methods" / METHOD
        legacy_config = json.loads((legacy_root / "config/training.json").read_text())
        assert json.loads(json.dumps(legacy_encoder.specification())) == legacy_config["feature_schema"]
        legacy_model = DailyQ(legacy_encoder)
        legacy_model.net.load_state_dict(torch.load(legacy_root / "results/model_final.pt", weights_only=False))
        assert sum(p.numel() for p in legacy_model.net.parameters()) == 55312
        if bidirectional:
            from src.experiment.daily_junction import MEMORY_METHOD
            old_memory = JunctionEncoder(problem, sim.ctx, sim.gamma, demand_memory=True)
            memory_root = folder / "data/methods" / MEMORY_METHOD
            memory_config = json.loads((memory_root / "config/training.json").read_text())
            assert json.loads(json.dumps(old_memory.specification())) == memory_config["feature_schema"]
            old_memory_model = DailyQ(old_memory)
            old_memory_model.net.load_state_dict(torch.load(memory_root / "results/model_final.pt", weights_only=False))
            assert sum(p.numel() for p in old_memory_model.net.parameters()) == 66560
        # A disconnected OD's latent demand must never enter the new observation.
        synthetic = DemandMemoryObservedEpisode.__new__(DemandMemoryObservedEpisode)
        synthetic.simulator = SimpleNamespace(ctx={"H0": np.array([100., 200.])})
        synthetic.traffic = SimpleNamespace(served_demand=np.array([0., 70.]),
            external=np.array([40., 80.]), reachable=np.array([False, True]),
            demand=np.array([999999., 70.]))
        with patch.object(HistoryObservedEpisode, "_solve_today", return_value=None):
            synthetic._solve_today()
        np.testing.assert_array_equal(synthetic.previous_observed_deviation, [-40., -10.])
        np.testing.assert_array_equal(synthetic.previous_reachable, [False, True])
    episode_class = DemandMemoryObservedEpisode if demand_memory else HistoryObservedEpisode
    if demand_memory:
        class AuditedEpisode(episode_class):
            def _solve_today(self):
                previous = self.traffic
                super()._solve_today()
                if previous is not None:
                    np.testing.assert_array_equal(self.previous_observed_deviation,
                                                  previous.served_demand - previous.external)
                    np.testing.assert_array_equal(self.previous_reachable, previous.reachable)
                    disconnected = ~previous.reachable
                    np.testing.assert_array_equal(self.previous_observed_deviation[disconnected],
                                                  -previous.external[disconnected])
        episode_class = AuditedEpisode
    episode = episode_class(sim, splits["train"][0], "response7d", disconnected_flow="unserved")
    reference = DailyEpisode(sim, splits["train"][0], "response7d", disconnected_flow="unserved")
    initial_observation = episode.observation()
    states, picks, rewards = [], [], []
    while not episode.finished:
        obs = episode.observation()
        state, previous = encoder.encode(obs), plain.encode(obs)
        np.testing.assert_array_equal(state["x"], np.delete(previous["x"], 9, axis=1))
        np.testing.assert_array_equal(state["g"][:6], previous["g"])
        assert state["g"][6] == np.float32(sim.gamma)
        for k, (o, d) in enumerate(encoder.od_indices):
            np.testing.assert_allclose(state["od"][o, d, 0], obs.cumulative_unmet[k] / 35., rtol=1e-6)
            if demand_memory:
                np.testing.assert_allclose(state["od"][o, d, 6], obs.external[k] / encoder.demand_scale, rtol=1e-6)
                np.testing.assert_allclose(state["od"][o, d, 7], obs.previous_observed_deviation[k] / encoder.demand_scale, rtol=1e-6)
                assert state["od"][o, d, 8] == obs.previous_reachable[k]
        if demand_memory:
            legacy_encoder = JunctionEncoder(problem, sim.ctx, sim.gamma)
            legacy_state = legacy_encoder.encode(obs)
            for key in ("x", "g"):
                np.testing.assert_array_equal(state[key], legacy_state[key])
            np.testing.assert_array_equal(state["od"][:, :, :6], legacy_state["od"])
        np.testing.assert_array_equal(state["od"][~encoder.valid], 0)
        np.testing.assert_array_equal(sensitivity(sim.gamma, obs.cumulative_unmet,
            sim.od_function_weights, sim.response_spec), episode.traffic.gamma_od)
        road = min(obs.candidates, key=lambda e: (-problem["baseline_flow"][e], e))
        states.append(state)
        picks.append(state["roads"].index(road))
        _, reward, _ = episode.step(road)
        _, reference_reward, _ = reference.step(road)
        assert reward == reference_reward
        rewards.append(reward)
    assert episode.result() == reference.result()
    np.testing.assert_array_equal(states[0]["od"][:, :, 0], 0)
    assert (states[1]["od"][:, :, 0] > 0).any()
    # One observed OD changes one fixed slot, never a destination's identity.
    custom = np.zeros(len(encoder.od_pairs))
    custom[0] = 3.5
    changed = encoder.encode(replace(initial_observation, cumulative_unmet=custom))
    locations = np.argwhere(changed["od"] != states[0]["od"])
    np.testing.assert_array_equal(locations, [[*encoder.od_indices[0], 0]])
    for oi in range(24):
        for di in range(24):
            if encoder.valid[oi, di]:
                np.testing.assert_array_equal(changed["od"][oi, di, 1:6], encoder.template[oi, di, 1:6])
    if demand_memory:
        np.testing.assert_array_equal(states[0]["od"][:, :, 7], 0)
        np.testing.assert_array_equal(states[0]["od"][:, :, 8], encoder.valid)

    torch.manual_seed(42)
    model = DailyQ(encoder)
    state = states[1]
    raw_od = torch.from_numpy(state["od"].copy()).requires_grad_()
    x = torch.from_numpy(state["x"])
    road_features = model.net.road_features(x, raw_od)
    assert road_features.shape == (38, encoder.node_dimension)
    original_endpoints = model.net.endpoints.clone()
    with torch.no_grad():
        model.net.endpoints.copy_(original_endpoints.flip(1))
    torch.testing.assert_close(model.net.road_features(x, raw_od), road_features, rtol=0, atol=0)
    with torch.no_grad():
        model.net.endpoints.copy_(original_endpoints)
    # Invalid slots cannot leak, even if their raw inputs or encoder biases are nonzero.
    malformed = raw_od.detach().clone()
    malformed[~model.net.valid_od] = 999
    torch.testing.assert_close(model.net.junction_embeddings(malformed),
                               model.net.junction_embeddings(raw_od), rtol=0, atol=0)
    captured = []
    handle = model.net.junction_encoder[0].register_forward_pre_hook(
        lambda module, args: captured.append(args[0].detach().clone()))
    model.net.junction_embeddings(torch.from_numpy(states[0]["od"]))
    handle.remove()
    encoded_slots = captured[0].reshape(24, 24, encoder.latent_dimension)
    assert torch.count_nonzero(encoded_slots[~model.net.valid_od]) == 0
    assert torch.count_nonzero(encoded_slots[model.net.valid_od]) > 0
    if bidirectional:
        # Capture the two role inputs. Incoming [d,o] must be the exact same
        # encoded OD as outgoing [o,d], not a separately relabelled OD vector.
        role_inputs = {}
        hooks = [module[0].register_forward_pre_hook(
            lambda module, args, key=key: role_inputs.update({key: args[0].detach().clone()}))
            for key, module in (("outgoing", model.net.junction_encoder),
                                ("incoming", model.net.incoming_encoder))]
        model.net.junction_embeddings(raw_od)
        for hook in hooks:
            hook.remove()
        outgoing_slots = role_inputs["outgoing"].reshape(24, 24, 16)
        incoming_slots = role_inputs["incoming"].reshape(24, 24, 16)
        torch.testing.assert_close(incoming_slots, outgoing_slots.transpose(0, 1), rtol=0, atol=0)
        # No road traffic features enter the OD/junction network at all.
        assert model.net.od_encoder[0].in_features == 9
        assert model.net.role_fusion[0].in_features == 32
        assert not encoder.specification()["traffic_conditioned_od_encoding"]
    model.net.junction_embeddings(raw_od).square().sum().backward()
    assert raw_od.grad[~model.net.valid_od].abs().sum() == 0
    assert raw_od.grad[model.net.valid_od][:, 0].abs().sum() > 0
    if demand_memory:
        assert all(raw_od.grad[model.net.valid_od][:, k].abs().sum() > 0 for k in (6, 7, 8))
    optimizer = torch.optim.Adam(model.net.parameters(), lr=model.hp["lr"])
    optimizer.zero_grad()
    before = copy.deepcopy(model.net.state_dict())
    model(state).square().sum().backward()
    gradients = {}
    prefixes = ("od_encoder", "junction_encoder", "core")
    if bidirectional:
        prefixes += ("incoming_encoder", "role_fusion")
    for prefix in prefixes:
        gradients[prefix] = sum(float(p.grad.abs().sum()) for n, p in model.net.named_parameters()
                                if n.startswith(prefix) and p.grad is not None)
        assert gradients[prefix] > 0
    optimizer.step()
    for prefix in gradients:
        assert any(not torch.equal(value, model.net.state_dict()[name])
                   for name, value in before.items() if name.startswith(prefix))
    target = DailyQ(encoder)
    target.net.load_state_dict(model.net.state_dict())
    equal_tree(target.net.state_dict(), model.net.state_dict())
    # Actual Windows subprocess uses the new observer and jointly learned network.
    with ProcessPoolExecutor(max_workers=1, initializer=initialize_worker,
            initargs=(problem, None, old["hyperparameters"], demand_memory, bidirectional)) as pool:
        worker = pool.submit(run_world, (splits["train"][1], model.net.state_dict(), 0., 0)).result()
    assert all(s["od"].shape == (24, 24, encoder.raw_od_dimension) and s["x"].shape == (38, 16) for s in worker["states"])
    assert np.isfinite(worker["result"]["objective"])

    trajectory = dict(states=states, picks=picks, rewards=rewards, result=episode.result())
    def batch(split, indices, weights, epsilon, seeds):
        assert split in ("train", "validation")
        return [trajectory for _ in indices]
    options = dict(train_count=2, validation_count=1, seed=42, identity={"junction_check": True},
        hp=dict(pool_n=2, batch_worlds=1, updates_per_ep=1, batch=2, target_sync=1),
        stop_params=dict(n_val=1, ep_min=99, probe_every=1), ep_cap=3)
    with tempfile.TemporaryDirectory(prefix="junction_check_") as temporary:
        direct = Path(temporary) / "direct"
        resumed = Path(temporary) / "resumed"
        train(encoder, batch, run_dir=direct, **options)
        train(encoder, batch, run_dir=resumed, pause_after=1, **options)
        train(encoder, batch, run_dir=resumed, **options)
        a = torch.load(direct / "results/training_latest.pt", weights_only=False)
        b = torch.load(resumed / "results/training_latest.pt", weights_only=False)
        for key in ("online", "target", "optimizer", "rng", "torch_rng", "replay", "priorities", "exemplars", "stop_state"):
            equal_tree(a[key], b[key])
        equal_tree(a["online"], a["target"])
        assert a["replay"][0][0]["od"].shape == (24, 24, encoder.raw_od_dimension)
        for ra, rb in zip(a["records"], b["records"]):
            for key in ra:
                if key not in ("seconds", "cumulative_seconds"):
                    assert ra[key] == rb[key], key
    assert protected_hashes(folder, demand_memory) == hashes
    result = dict(source_sha256=_source_digest()[0], frozen_scenarios_verified=True,
        same_physics_rewards_and_objective=True, original_schema_unchanged=True,
        original_inputs_unchanged_except_removed_accessibility=True, observed_history_reconstructs_gamma=True,
        fixed_destination_slots=True, destination_count_pairing=True, invalid_slot_masking_before_and_after=True,
        valid_zero_history_retained=True, endpoint_symmetry=True,
        end_to_end_gradient_sums=gradients, optimizer_updates_all_parts=True, target_sync_all_parts=True,
        resume_weights_optimizer_replay_rng_and_non_time_metrics_exact=True, subprocess_passed=True,
        raw_replay=True, baseline_files_unchanged=True, test_worlds_used=0,
        old_parameter_count=sum(p.numel() for p in DailyQ(plain).net.parameters()),
        new_parameter_count=sum(p.numel() for p in model.net.parameters()), feature_schema=encoder.specification())
    if demand_memory:
        result.update(daily_previous_observation_verified=True, no_disconnected_latent_demand=True,
                      current_reference_verified=True, new_channels_receive_gradients=True,
                      previous_six_od_inputs_unchanged=True, initial_memory_verified=True,
                      old_junction_schema_and_saved_weights_compatible=True)
    suffix = "junction_memory" if demand_memory else "junction"
    if bidirectional:
        from src.experiment.daily_dual_junction import validate_discovery_only
        from src.experiment.daily_ga_saa10 import training_subset
        from src.problem.daily import build_problem, make_splits
        from src.methods.metaheuristic import GA_PARAMS, BUDGET_CAP
        variant = build_problem(11, 42, variant="poi_hidden6")
        validate_discovery_only(problem, variant)
        variant_splits = make_splits(variant)
        assert [len(variant_splits[k]) for k in ("train", "validation", "test")] == [64, 11, 50]
        for worlds in variant_splits.values():
            for world in worlds:
                assert len(world.roads) == 11 and len({r.edge_id for r in world.roads}) == 11
                assert sum(r.discovery_day == 0 for r in world.roads) == 5
                assert sum(1 <= r.discovery_day <= 14 for r in world.roads) == 6
                assert sum(r.edge_id in variant["bins"][0] for r in world.roads) == 2
                assert sum(r.edge_id in variant["bins"][1] for r in world.roads) == 9
        indices, _ = training_subset(variant_splits["train"], 42)
        ga = json.loads((folder / "data/methods/ga_saa10_flow_insert_seed42/config/training.json").read_text())
        assert ga["selected_training_indices"] == indices == [4, 5, 6, 12, 25, 37, 43, 51, 58, 61]
        assert ga["hyperparameters"] == GA_PARAMS and ga["budget"] == BUDGET_CAP
        suffix = "dual_junction"
        result.update(incoming_slots_exact_transpose=True, no_traffic_conditioned_od_encoder=True,
                      prior_memory_weights_and_schema_compatible=True,
                      new_public5_hidden6_only_quota_change=True, original_ga_settings_unchanged=True,
                      second_public_roads=variant["public_roads"], second_hidden_quotas=variant["hidden_quotas"])
    _write_json(folder / f"data/analysis/config/{suffix}_input_audit.json", result)
    sim.close()
    print(json.dumps({k: v for k, v in result.items() if k != "feature_schema"}, indent=2))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--demand-memory", action="store_true")
    parser.add_argument("--bidirectional", action="store_true")
    args = parser.parse_args()
    main(demand_memory=args.demand_memory or args.bidirectional, bidirectional=args.bidirectional)
