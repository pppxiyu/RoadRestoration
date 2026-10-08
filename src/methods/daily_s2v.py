"""The existing adaptive S2V Q-network connected to daily public observations.

The experiment default uses 17 road inputs (the original 14 plus three public
road-class indicators) and six global inputs. The explicit no-road-class ablation
retains 14 columns. The graph/readout/training rules remain unchanged.
This module never inspects Scenario or RoadDamage truth when encoding a state.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from src.methods.rl_s2v import _build_s2v_net
from src.methods.rl_s2v_saa import S2V_SAA_PARAMS
from src.methods.rl_rank import EP_CAP, STOP_PARAMS, RankPlateauStop, _nstep_rows


NODE_INPUTS = (
    "discovered and pending repair", "under repair", "repair completed",
    "normal-period two-way traffic / normal-period maximum",
    "estimated severity / 3", "estimated duration / 35 days",
    "estimated demand exposure / maximum among discovered roads",
    "projected natural-recovery demand deficit",
    "repair-completion history", "crew accessibility restriction (always zero)",
    "observed two-way traffic / normal-period maximum", "observed congestion excess / (1 + excess)",
    "projected disconnected normal demand share", "projected observed demand deficit",
)
GLOBAL_INPUTS = (
    "current day / 35 days", "crew busy time (zero at single-crew decision times)",
    "natural-recovery demand deficit / total normal demand",
    "pending estimated work / all discovered estimated work",
    "observed demand deficit / total normal demand", "disconnected normal demand share",
)
ROAD_CLASSES = ("highway", "major", "local")
TRAFFIC_NODE_COLUMNS = (10, 11, 12, 13)
TRAFFIC_GLOBAL_COLUMNS = (4, 5)


class DailyEncoder:
    def __init__(self, problem, context, *, road_class_input=False, traffic_input=True):
        # User-approved retention of the previous RL input. This is an encoding
        # of completion history, NOT a parameter of the new demand dynamics.
        self.recency_encoding = "legacy_decay_0.7_per_day"
        self.time_scale = float(problem["rules"].recovery_days)
        if self.time_scale != 35:
            raise ValueError("this reviewed input schema uses the public 35-day scale")
        self.ids = tuple(sorted(int(e) for e in problem["edges"].edge_id))
        self.road_class_input = bool(road_class_input)
        self.traffic_input = bool(traffic_input)
        self.node_dimension = 17 if self.road_class_input else 14
        if self.road_class_input:
            classes = {int(r.edge_id): r.road_class for r in problem["edges"].itertuples(index=False)}
            self.class_features = np.asarray(
                [[float(classes[e] == name) for name in ROAD_CLASSES] for e in self.ids], dtype=np.float32)
            if not np.all(self.class_features.sum(axis=1) == 1):
                raise ValueError("unknown public road class")
        self.idx = {e: i for i, e in enumerate(self.ids)}
        rows = context["edge_row"]
        self.row_indices = [rows[e] for e in self.ids]
        self.H0 = context["H0"].copy()
        self.total_demand = float(self.H0.sum())
        self.baseline_flow = np.array([problem["baseline_flow"][e] for e in self.ids])
        self.flow_scale = max(float(self.baseline_flow.max()), 1.0)
        # Preserve the old projection weights: normal OD demand / 3 times
        # intact free-flow path incidence. All this information is public.
        self.B = context["incidence"][:, self.row_indices] * self.H0[:, None] / 3.0
        self.denominator = self.B.T @ self.H0
        self.exposure = self.B.sum(axis=0)
        endpoints = {int(r.edge_id): {int(r.u), int(r.v)}
                     for r in problem["edges"].itertuples(index=False)}
        self.A = np.array([[float(a != b and bool(endpoints[a] & endpoints[b]))
                            for b in self.ids] for a in self.ids], dtype=np.float32)
        self.deg = self.A.sum(axis=1)
        self.history_retention = 0.7

    def specification(self):
        spec = dict(node_inputs=NODE_INPUTS, global_inputs=GLOBAL_INPUTS,
                    node_count=len(self.ids), node_dimension=self.node_dimension, global_dimension=6,
                    road_ids=self.ids, time_scale_days=self.time_scale,
                    recency_encoding=self.recency_encoding,
                    history_retention_per_day=self.history_retention,
                    information_boundary="Observation only; no scenario truth or future discoveries",
                    discovered_only_columns=[0, 1, 2, 4, 5, 6, 7, 8, 12, 13])
        if self.road_class_input:
            spec.update(node_inputs=NODE_INPUTS + tuple("public road class: " + c for c in ROAD_CLASSES),
                        road_class_encoding="one-hot; all roads, independent of damage discovery",
                        road_class_columns=dict(zip(ROAD_CLASSES, (14, 15, 16))))
        if not self.traffic_input:
            spec.update(traffic_input=False, traffic_ablation="constant zero from the first decision in all splits",
                        zeroed_node_columns=list(TRAFFIC_NODE_COLUMNS),
                        zeroed_global_columns=list(TRAFFIC_GLOBAL_COLUMNS),
                        node_inputs=tuple("disabled (constant zero): " + name if i in TRAFFIC_NODE_COLUMNS else name
                                          for i, name in enumerate(spec["node_inputs"])),
                        global_inputs=tuple("disabled (constant zero): " + name if i in TRAFFIC_GLOBAL_COLUMNS else name
                                            for i, name in enumerate(GLOBAL_INPUTS)))
        return spec

    def _project(self, quantity):
        return np.divide(self.B.T @ quantity, self.denominator,
                         out=np.zeros(len(self.ids)), where=self.denominator > 0)

    def encode(self, observation):
        """Freeze exactly the currently observed input for later replay."""
        x = np.zeros((len(self.ids), self.node_dimension), dtype=np.float32)
        if self.road_class_input:
            x[:, 14:] = self.class_features
        x[:, 3] = self.baseline_flow / self.flow_scale
        if self.traffic_input:
            x[:, 10] = observation.road_flow[self.row_indices] / self.flow_scale
            x[:, 11] = observation.road_congestion[self.row_indices]
        completed = dict(observation.completed)
        pending = set(observation.candidates)
        natural_deficit = np.maximum(self.H0 - observation.external, 0.0)
        # Do not even read the omitted traffic observations. This prevents them
        # from influencing a remaining column through a summary or normalization.
        observed_deficit = (np.maximum(self.H0 - observation.demand, 0.0)
                            if self.traffic_input else np.zeros_like(self.H0))
        disconnected = (self.H0 * ~observation.reachable
                        if self.traffic_input else np.zeros_like(self.H0))
        projections = (self._project(natural_deficit), self._project(disconnected),
                       self._project(observed_deficit))
        exposure_scale = max((r.estimated_severity * self.exposure[self.idx[r.edge_id]]
                              for r in observation.visible), default=1.0) or 1.0
        all_work = sum(r.estimated_duration for r in observation.visible)
        remaining_work = sum(r.estimated_duration for r in observation.visible if r.edge_id in pending)
        for road in observation.visible:
            e, i = road.edge_id, self.idx[road.edge_id]
            x[i, 0] = e in pending
            x[i, 2] = e in completed
            x[i, 4] = road.estimated_severity / 3.0
            x[i, 5] = road.estimated_duration / self.time_scale
            x[i, 6] = road.estimated_severity * self.exposure[i] / exposure_scale
            x[i, 7], x[i, 12], x[i, 13] = (values[i] for values in projections)
            x[i, 8] = self.history_retention ** (observation.day - completed[e]) if e in completed else 1.0
        g = np.array([observation.day / self.time_scale, 0.0,
                      natural_deficit.sum() / self.total_demand,
                      remaining_work / all_work if all_work else 0.0,
                      observed_deficit.sum() / self.total_demand,
                      disconnected.sum() / self.total_demand], dtype=np.float32)
        if not np.isfinite(x).all() or not np.isfinite(g).all():
            raise ValueError("non-finite public input")
        if not pending:
            raise ValueError("Q-network should only be called when an action is available")
        return dict(x=x, g=g, cand=tuple(self.idx[e] for e in observation.candidates),
                    roads=observation.candidates, day=observation.day)


class DailyQ:
    """Unmodified network builder and legal-action dueling aggregation."""
    def __init__(self, encoder, hp=None):
        import torch
        from torch import nn
        self.torch = torch
        self.hp = dict(S2V_SAA_PARAMS, adaptive=True, feat_obs_traffic=True,
                       feat_obs_disc=True, feat_obs_trueD=True)
        if hp:
            unknown = set(hp) - set(self.hp)
            if unknown:
                raise ValueError(f"unknown S2V parameters: {sorted(unknown)}")
            self.hp.update(hp)
        h = self.hp
        self.net = _build_s2v_net(h["p"], h["t_emb"], h["use_g"], torch, nn,
                                  dueling=h["dueling"], readout_hidden=h["readout_hidden"],
                                  in_dim=encoder.node_dimension, hop_untied=h["hop_untied"], g_dim=6)
        self.A = torch.from_numpy(encoder.A)
        self.deg = torch.from_numpy(encoder.deg)

    def __call__(self, state):
        out = self.net(self.torch.from_numpy(state["x"]), self.A, self.deg,
                       self.torch.from_numpy(state["g"]))
        indices = list(state["cand"])
        if not self.hp["dueling"]:
            return out[indices]
        value, advantages = out
        legal = advantages[indices]
        return value + legal - legal.mean()


def rollout(episode, encoder, q_function, *, epsilon=0.0, rng=None):
    """Interact only through observation/action; charge all post-repair recovery."""
    states, picks, rewards = [], [], []
    rng = np.random.RandomState(0) if rng is None else rng
    while not episode.finished:
        state = encoder.encode(episode.observation())
        if rng.rand() < epsilon:
            action = int(rng.randint(len(state["cand"])))
        else:
            with q_function.torch.no_grad():
                action = int(q_function(state).argmax())
        _, reward, _ = episode.step(state["roads"][action])
        states.append(state)
        picks.append(action)
        rewards.append(float(reward))
    result = episode.result()
    if abs(sum(rewards) + result["objective"] - result["initial_loss"]) > 1e-8:
        raise AssertionError("reward sum disagrees with cumulative objective")
    if result["initial_loss"] != 0:
        raise AssertionError("reviewed public-road counts require an initial action")
    return dict(states=states, picks=picks, rewards=rewards, result=result)


def train(encoder, run_batch, *, train_count, validation_count, seed, run_dir,
          identity, hp=None, stop_params=None, ep_cap=EP_CAP, redraw=None):
    """Daily SAA trainer, preserving existing network, loss and stop defaults.

    run_batch receives a split name, world indices and frozen network parameters.
    The experiment owns the simulator/process pool and NEVER supplies test
    rollouts during training. Validation uses the same environment as training.
    Delivered weights remain the FINAL policy, matching the previous SAA method;
    the best-validation policy is separately saved and never selected by test.
    """
    import torch
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed * 7 + 1)
    online = DailyQ(encoder, hp)
    target = DailyQ(encoder, hp)
    target.net.load_state_dict(online.net.state_dict())
    h = online.hp
    if train_count != int(h["pool_n"]):
        raise ValueError("training sample count does not match SAA pool size")
    stop_config = dict(STOP_PARAMS, **(stop_params or {}))
    if validation_count != stop_config["n_val"]:
        raise ValueError("validation sample count does not match stopping settings")
    optimizer = torch.optim.Adam(online.net.parameters(), lr=h["lr"])
    stop = RankPlateauStop(stop_config["ep_min"], stop_config["patience_P"],
                           stop_config["stable_K"], stop_config["tol"])
    root = Path(run_dir)
    for child in ("config", "results", "log"):
        (root / child).mkdir(parents=True, exist_ok=True)
    checkpoint = root / "results" / "training_latest.pt"
    contract = dict(identity=identity, seed=seed, hyperparameters=h, stop=stop_config,
                    ep_cap=ep_cap, feature_schema=encoder.specification(),
                    delivery="final policy; best validation saved separately",
                    reward="negative cumulative daily loss until next action, including terminal recovery",
                    discount="existing gamma per decision, not per calendar day")
    # JSON round-trip gives a stable comparison for tuple/list fields on resume.
    contract = json.loads(json.dumps(contract, sort_keys=True))
    replay, priorities, exemplars, records = [], [], {}, []
    best_validation, best_weights = float("inf"), None
    episode = 0
    elapsed_before = 0.0
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if saved["contract"] != contract:
            raise ValueError("resume settings differ; preserve this run and use a new experiment identity")
        online.net.load_state_dict(saved["online"])
        target.net.load_state_dict(saved["target"])
        optimizer.load_state_dict(saved["optimizer"])
        rng.set_state(saved["rng"])
        torch.set_rng_state(saved["torch_rng"])
        replay, priorities, exemplars = saved["replay"], saved["priorities"], saved["exemplars"]
        records, episode = saved["records"], saved["next_episode"]
        best_validation, best_weights = saved["best_validation"], saved["best_weights"]
        stop.__dict__.update(saved["stop_state"])
        elapsed_before = saved["training_seconds"]
    (root / "config" / "training.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")
    started = time.perf_counter()

    def save_checkpoint():
        # Replace only this runner's own latest checkpoint, after the temporary
        # file is complete. A crash cannot destroy the previous resumable state.
        temporary = checkpoint.with_suffix(".tmp")
        torch.save(dict(contract=contract, online=online.net.state_dict(), target=target.net.state_dict(),
                        optimizer=optimizer.state_dict(), rng=rng.get_state(), torch_rng=torch.get_rng_state(),
                        replay=replay, priorities=priorities, exemplars=exemplars, records=records,
                        next_episode=episode, best_validation=best_validation, best_weights=best_weights,
                        stop_state=stop.__dict__,
                        training_seconds=elapsed_before + time.perf_counter() - started), temporary)
        deadline = time.monotonic() + 120
        while True:
            try:
                temporary.replace(checkpoint)
                break
            except OSError as exc:
                if getattr(exc, "winerror", None) not in (5, 32, 33) or time.monotonic() >= deadline:
                    raise
                time.sleep(1)
        pd.DataFrame(records).to_csv(root / "log" / "training.csv", index=False)

    while episode < ep_cap and not stop.done():
        ep_started = time.perf_counter()
        epsilon = max(h["eps_min"], h["eps0"] - (h["eps0"] - h["eps_min"])
                      * episode / max(1, h["eps_anneal"]))
        lam = (h["lam"] if h.get("lam_growth") is None else
               min(h["lam"], h["lam0"] * h["lam_growth"] ** max(0, episode - 1)))
        indices = [int(i) for i in rng.choice(train_count, h["batch_worlds"], replace=False)]
        rollout_seeds = [int(x) for x in rng.randint(0, 2**31 - 1, len(indices))]
        fresh = run_batch("train", indices, online.net.state_dict(), epsilon, rollout_seeds)
        for world, trajectory in zip(indices, fresh):
            rows = _nstep_rows(trajectory["states"], trajectory["picks"], trajectory["rewards"],
                               h["n_step"], h["gamma"])
            replay.extend(rows)
            priorities.extend([max(priorities, default=1.0)] * len(rows))
            objective = trajectory["result"]["objective"]
            if world not in exemplars or objective < exemplars[world][2] - 1e-12:
                exemplars[world] = (trajectory["states"], trajectory["picks"], objective)
        replay, priorities = replay[-h["replay_cap"]:], priorities[-h["replay_cap"]:]
        beta = min(1.0, h["per_beta0"] + (1 - h["per_beta0"]) * episode / h["per_beta_eps"])
        metrics = []
        for update in range(h["updates_per_ep"]):
            optimizer.zero_grad()
            count = min(h["batch"], len(replay))
            if h["prioritized"]:
                probability = np.asarray(priorities) ** h["per_alpha"]
                probability /= probability.sum()
                batch = rng.choice(len(replay), count, replace=False, p=probability)
                weights = (len(replay) * probability[batch]) ** (-beta)
                weights = torch.tensor(weights / weights.max(), dtype=torch.float32)
            else:
                batch = rng.choice(len(replay), count, replace=False)
                weights = torch.ones(count)
            predictions = torch.stack([online(replay[i][0])[replay[i][1]] for i in batch])
            targets = []
            with torch.no_grad():
                for i in batch:
                    _, _, reward, bootstrap, discount = replay[i]
                    if bootstrap is None:
                        targets.append(reward)
                    else:
                        evaluator = target if h["target_sync"] > 0 else online
                        bootstrap_value = (evaluator(bootstrap)[int(online(bootstrap).argmax())]
                                           if h["double_dqn"] else evaluator(bootstrap).max())
                        targets.append(reward + discount * float(bootstrap_value))
            targets = torch.tensor(targets, dtype=torch.float32)
            errors = predictions - targets
            td_loss = (weights * errors.square()).mean()
            if h["prioritized"]:
                for i, error in zip(batch, errors.detach().abs().tolist()):
                    priorities[i] = error + h["per_eps"]
            hinge = torch.tensor(0.0)
            if h["lam"] > 0 and exemplars:
                world = list(exemplars)[rng.randint(len(exemplars))]
                states, picks, _ = exemplars[world]
                for state, pick in zip(states, picks):
                    values = online(state)
                    violations = torch.relu(h["margin"] - (values[pick] - values)).clone()
                    violations[pick] = 0.0
                    hinge = hinge + violations.sum()
                hinge = hinge / len(states)
            loss = td_loss + lam * hinge
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite training loss in episode {episode}")
            loss.backward()
            gradient = float(torch.sqrt(sum(p.grad.detach().square().sum() for p in online.net.parameters()
                                           if p.grad is not None)))
            if not np.isfinite(gradient):
                raise FloatingPointError(f"non-finite gradient in episode {episode}")
            optimizer.step()
            metrics.append([float(loss.detach()), float(td_loss.detach()), float(hinge.detach()),
                            float(predictions.detach().mean()), float(targets.mean()),
                            float(errors.detach().abs().mean()), gradient])
            if h["target_sync"] > 0 and (update + 1 + episode * h["updates_per_ep"]) % h["target_sync"] == 0:
                target.net.load_state_dict(online.net.state_dict())
        validation_mean, orders, validation = None, None, []
        if episode % stop_config["probe_every"] == 0:
            validation = run_batch("validation", list(range(validation_count)), online.net.state_dict(),
                                   0.0, [0] * validation_count)
            validation_mean = float(np.mean([t["result"]["objective"] for t in validation]))
            orders = tuple(tuple(t["result"]["order"]) for t in validation)
            if validation_mean < best_validation - 1e-12:
                best_validation = validation_mean
                best_weights = copy.deepcopy(online.net.state_dict())
        # With variable identities there is no legitimate single nominal world.
        # Stability therefore compares all fixed validation orders together.
        if validation_mean is not None:
            stop.update(episode, validation_mean, orders)
        else:
            stop.episode = episode
        metric_names = ("loss", "td_loss", "hinge_loss", "q_mean", "target_mean", "absolute_td_error", "gradient_norm")
        record = dict(episode=episode, epsilon=float(epsilon), hinge_weight=float(lam),
                      training_objective=float(np.mean([t["result"]["objective"] for t in fresh])),
                      validation_objective=validation_mean, best_validation_objective=best_validation,
                      seconds=time.perf_counter() - ep_started,
                      cumulative_seconds=elapsed_before + time.perf_counter() - started,
                      training_ue_solves=sum(t.get("runtime", {}).get("ue_solves", 0) for t in fresh),
                      validation_ue_solves=sum(t.get("runtime", {}).get("ue_solves", 0) for t in validation),
                      cached_days=sum(t.get("runtime", {}).get("cached_days", 0) for t in fresh + validation),
                      worker_seconds=sum(t.get("runtime", {}).get("seconds", 0) for t in fresh + validation),
                      **dict(zip(metric_names, np.mean(metrics, axis=0).tolist())))
        records.append(record)
        print(f"episode={episode} train={record['training_objective']:.5f} "
              f"validation={validation_mean} best_validation={best_validation:.5f} "
              f"seconds={record['seconds']:.1f}", flush=True)
        episode += 1
        # Episodes are expensive; retain a complete restart point after each one.
        save_checkpoint()
        if redraw is not None and (episode % 20 == 0 or stop.done()):
            redraw(root)
    outcome = stop.reason or "episode_cap"
    torch.save(online.net.state_dict(), root / "results" / "model_final.pt")
    if best_weights is not None:
        torch.save(best_weights, root / "results" / "model_best_validation.pt")
    result = dict(episodes=episode, outcome=outcome, best_validation_objective=best_validation,
                  training_seconds=elapsed_before + time.perf_counter() - started,
                  training_ue_solves=sum(r["training_ue_solves"] for r in records),
                  validation_ue_solves=sum(r["validation_ue_solves"] for r in records),
                  cached_days=sum(r["cached_days"] for r in records),
                  training_worker_seconds=sum(r["worker_seconds"] for r in records))
    (root / "results" / "training_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    if redraw is not None:
        redraw(root)
    return online, result
