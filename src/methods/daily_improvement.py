"""Pairwise policy improvement with independent candidate and best-policy state.

The experiment supplies training labels and validation-only callbacks. This
module never requests test data. A round starts a fresh Adam optimizer but
inherits the learning rate and plateau scheduler from the preceding round.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np
import torch

from src.methods.daily_s2v import DailyQ


@dataclass(frozen=True)
class ImprovementSettings:
    learning_rate: float = 0.001
    batch_states: int = 8
    outer_patience: int = 5
    tolerance: float = 1e-12
    validation_updates: tuple = (300, 1000, 3000)
    validation_interval: int = 1000
    inner_patience: int = 3
    scheduler_factor: float = 0.5
    scheduler_patience: int = 1
    minimum_learning_rate: float = 0.00001
    # None means every nontrivial decision position in every training world.
    decision_positions: tuple | None = None

    def __post_init__(self):
        if (self.learning_rate <= 0 or self.batch_states < 1 or self.outer_patience < 1
                or self.inner_patience < 1 or self.validation_interval < 1
                or self.tolerance < 0 or self.minimum_learning_rate <= 0
                or self.minimum_learning_rate > self.learning_rate
                or not 0 < self.scheduler_factor < 1 or self.scheduler_patience < 0):
            raise ValueError("invalid improvement settings")
        if not self.validation_updates or tuple(sorted(set(self.validation_updates))) != self.validation_updates:
            raise ValueError("validation milestones must be strictly increasing")
        if self.validation_updates[0] < 1:
            raise ValueError("validation milestones must be positive")
        if self.decision_positions is not None and (
                not self.decision_positions or min(self.decision_positions) < 0
                or tuple(sorted(set(self.decision_positions))) != self.decision_positions):
            raise ValueError("decision positions must be unique and increasing")


def weights_digest(weights):
    digest = hashlib.sha256()
    for name, tensor in sorted(weights.items()):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str((array.dtype, array.shape)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def atomic_torch(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def atomic_pickle(path, value):
    """Integrity-check completed label artifacts before reuse."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = pickle.dumps(value, protocol=5)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(pickle.dumps((hashlib.sha256(payload).hexdigest(), payload), protocol=5))
    temporary.replace(path)


def read_pickle(path):
    checksum, payload = pickle.loads(Path(path).read_bytes())
    if hashlib.sha256(payload).hexdigest() != checksum:
        raise ValueError(f"label artifact hash changed: {path}")
    return pickle.loads(payload)


def freeze_json(path, value):
    path = Path(path)
    normalized = json.loads(json.dumps(value, sort_keys=True))
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != normalized:
            raise ValueError(f"resume contract changed: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(normalized, indent=2), encoding="utf-8")
        temporary.replace(path)
    return normalized


def pair_loss(scores, costs):
    target = torch.as_tensor(costs, dtype=scores.dtype, device=scores.device)
    if len(scores) < 2 or len(scores) != len(target):
        raise ValueError("each training state needs all of its candidate costs")
    i, j = torch.triu_indices(len(scores), len(scores), 1, device=scores.device)
    return ((scores[i] - scores[j]) - (target[j] - target[i])).square().mean()


def _scheduler(optimizer, settings):
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=settings.scheduler_factor,
        patience=settings.scheduler_patience, threshold=settings.tolerance,
        threshold_mode="abs", cooldown=0, min_lr=settings.minimum_learning_rate, eps=0.0)


def _finite_validation(value):
    value = float(value)
    if not np.isfinite(value):
        raise FloatingPointError("nonfinite validation objective")
    return value


def fit_round(encoder, hp, initial, rows, evaluate, directory, *, identity,
              settings=ImprovementSettings(), seed=42, schedule=None,
              initial_validation=None, pause_after=None):
    """Resume after each update; commit each validation and scheduler step together.

    evaluate accepts weights and returns the fixed validation mean loss. The
    candidate is the best fitted checkpoint, not the last update or the start
    model. pause_after is a test/execution boundary, not a stopping rule.
    """
    if not rows or any(not np.isfinite(row["costs"]).all() for row in rows):
        raise ValueError("finite, nonempty training labels required")
    directory = Path(directory)
    contract = freeze_json(directory / "contract.json", dict(
        identity=identity, settings=asdict(settings), hp=hp, seed=seed,
        start_sha256=weights_digest(initial),
        labels_sha256=hashlib.sha256(pickle.dumps(rows, protocol=5)).hexdigest(),
        incoming_schedule=schedule, initial_validation=initial_validation,
        loss="all unordered action pairs, equal weight per sampled state",
        candidate="best fitted validation checkpoint, no fallback to starting weights"))
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    model = DailyQ(encoder, hp)
    model.net.load_state_dict(initial, strict=True)
    optimizer = torch.optim.Adam(model.net.parameters(), lr=(
        settings.learning_rate if schedule is None else schedule["lr"]))
    scheduler = _scheduler(optimizer, settings)
    if schedule is None:
        scheduler.step(_finite_validation(initial_validation))
    else:
        scheduler.load_state_dict(copy.deepcopy(schedule["scheduler"]))
    latest = directory / "latest.pt"
    update, bad, history, losses, best_weights, best_update = 0, 0, [], [], None, None
    best, done = float("inf"), False
    if latest.exists():
        saved = torch.load(latest, map_location="cpu", weights_only=False)
        if saved["contract"] != contract:
            raise ValueError("fitting checkpoint contract changed")
        model.net.load_state_dict(saved["weights"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        rng.set_state(saved["rng"])
        torch.set_rng_state(saved["torch_rng"])
        update, bad, history, losses = (saved[k] for k in ("update", "bad", "history", "losses"))
        best, best_weights, best_update, done = (saved[k] for k in (
            "best", "best_weights", "best_update", "done"))

    def persist():
        atomic_torch(latest, dict(contract=contract, weights=model.net.state_dict(),
            optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
            rng=rng.get_state(), torch_rng=torch.get_rng_state(), update=update,
            bad=bad, history=history, losses=losses, best=best,
            best_weights=best_weights, best_update=best_update, done=done))

    while not done and (pause_after is None or update < pause_after):
        model.net.train()
        optimizer.zero_grad()
        chosen = rng.choice(len(rows), min(settings.batch_states, len(rows)), replace=False)
        loss = torch.stack([pair_loss(model(rows[i]["state"]), rows[i]["costs"])
                            for i in chosen]).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite pairwise loss")
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.net.parameters()):
            raise FloatingPointError("nonfinite pairwise gradient")
        optimizer.step()
        if any(not torch.isfinite(p).all() for p in model.net.parameters()):
            raise FloatingPointError("nonfinite fitted weights")
        update += 1
        losses.append(dict(update=update, loss=float(loss.detach()), lr=optimizer.param_groups[0]["lr"]))
        milestone = (update in settings.validation_updates or
            (update > settings.validation_updates[-1] and
             (update - settings.validation_updates[-1]) % settings.validation_interval == 0))
        if milestone:
            model.net.eval()
            value = _finite_validation(evaluate(model.net.state_dict()))
            if value < best - settings.tolerance:
                best, best_weights, best_update, bad = value, copy.deepcopy(model.net.state_dict()), update, 0
            elif update > settings.validation_updates[-1]:
                bad += 1
            lr_before = optimizer.param_groups[0]["lr"]
            scheduler.step(value)
            history.append(dict(update=update, validation=value, best=best,
                consecutive_no_improvement=bad, lr_before=lr_before,
                lr_after=optimizer.param_groups[0]["lr"]))
            done = bad >= settings.inner_patience
        persist()
    result = dict(done=done, update=update, validation=best, candidate_update=best_update,
        history=history, schedule=dict(lr=optimizer.param_groups[0]["lr"], scheduler=scheduler.state_dict()))
    if done:
        # Repeated finish calls leave the resumable checkpoint untouched.
        model.net.load_state_dict(best_weights)
        if not (directory / "candidate.pt").exists():
            atomic_torch(directory / "candidate.pt", best_weights)
        elif weights_digest(torch.load(directory / "candidate.pt", weights_only=True)) != weights_digest(best_weights):
            raise ValueError("candidate artifact changed")
        freeze_json(directory / "summary.json", dict(result, candidate_sha256=weights_digest(best_weights)))
    return model, result


def improve(encoder, hp, initial, initial_validation, collect, evaluate, directory, *,
            identity, settings=ImprovementSettings(), seed=42, pause_after_round=None):
    """Continue each candidate while selecting and stopping against best-so-far."""
    directory = Path(directory)
    contract = freeze_json(directory / "contract.json", dict(identity=identity,
        settings=asdict(settings), hp=hp, seed=seed, base_sha256=weights_digest(initial),
        base_validation=_finite_validation(initial_validation), test_used_for_selection=False))
    checkpoint = directory / "chain.pt"
    chain = dict(contract=contract, current=copy.deepcopy(initial), best=copy.deepcopy(initial),
        best_validation=float(initial_validation), best_round=0, rounds=[], bad=0, schedule=None)
    if checkpoint.exists():
        chain = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if chain["contract"] != contract:
            raise ValueError("chain contract changed")
    while chain["bad"] < settings.outer_patience:
        number = len(chain["rounds"]) + 1
        if pause_after_round is not None and number > pause_after_round:
            break
        start_sha = weights_digest(chain["current"])
        folder = directory / f"round_{number:03d}"
        rows = collect(chain["current"], folder, settings.decision_positions)
        model, result = fit_round(encoder, hp, chain["current"], rows, evaluate, folder,
            identity=identity, settings=settings, seed=seed, schedule=chain["schedule"],
            initial_validation=float(initial_validation) if chain["schedule"] is None else None)
        if not result["done"]:
            raise RuntimeError("incomplete round cannot advance the chain")
        candidate = copy.deepcopy(model.net.state_dict())
        value = _finite_validation(result["validation"])
        better = value < chain["best_validation"] - settings.tolerance
        chain["bad"] = 0 if better else chain["bad"] + 1
        if better:
            chain["best"], chain["best_validation"], chain["best_round"] = candidate, value, number
        chain["current"], chain["schedule"] = candidate, result["schedule"]
        chain["rounds"].append(dict(round=number, start_sha256=start_sha,
            candidate_sha256=weights_digest(candidate), candidate_update=result["candidate_update"],
            validation=value, improved_global_best=better, best_validation=chain["best_validation"],
            consecutive_no_global_improvement=chain["bad"], next_lr=chain["schedule"]["lr"]))
        atomic_torch(checkpoint, chain)
        print("improvement", chain["rounds"][-1], flush=True)
    complete = chain["bad"] >= settings.outer_patience
    selection = dict(complete=complete, selected_round=chain["best_round"],
        validation_objective=chain["best_validation"], model_sha256=weights_digest(chain["best"]),
        rule="minimum validation loss over base and all round candidates",
        stopping="consecutive rounds without improving global best", test_used_for_selection=False,
        rounds=chain["rounds"], consecutive_no_global_improvement=chain["bad"])
    if complete:
        atomic_torch(directory / "model_best_validation.pt", chain["best"])
        freeze_json(directory / "selection.json", selection)
    return chain["best"], selection
