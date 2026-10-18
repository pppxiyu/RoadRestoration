"""Offline imitation of a saved legal GA policy using training observations only."""
from __future__ import annotations

import copy
import time
import numpy as np

from src.methods.daily_s2v import DailyQ


def demonstration_rows(trajectories, gamma):
    rows = []
    for trajectory in trajectories:
        returns, total = [], 0.0
        for reward in reversed(trajectory["rewards"]):
            total = float(reward) + gamma * total
            returns.append(total)
        returns.reverse()
        rows.extend(zip(trajectory["states"], trajectory["picks"], returns))
    return rows


def pretrain(encoder, trajectories, hp, *, seed, evaluate, record_probe,
             checkpoints=(200, 600, 1200), batch_size=32):
    """Clone the chosen action and calibrate its Q to the observed teacher return.

    Cross entropy on legal Q differences / 0.2 teaches action preference. A
    0.1-weight smooth-L1 term on the chosen action teaches discounted return
    units without making unobserved counterfactual Q labels. Only validation
    mean chooses among three predeclared checkpoints. Adam is discarded after
    this stage, so online learning starts with a fresh optimizer.
    """
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed + 700)
    policy = DailyQ(encoder, hp)
    optimizer = torch.optim.Adam(policy.net.parameters(), lr=hp["lr"])
    rows = demonstration_rows(trajectories, hp["gamma"])
    started = time.perf_counter()
    best_score, best_weights, best_step = float("inf"), None, None
    history = []
    for step in range(1, max(checkpoints) + 1):
        sampled = rng.choice(len(rows), min(batch_size, len(rows)), replace=False)
        optimizer.zero_grad()
        classification, regression = [], []
        for i in sampled:
            state, pick, target = rows[i]
            q = policy(state)
            classification.append(-F.log_softmax(q / 0.2, dim=0)[pick])
            regression.append(F.smooth_l1_loss(q[pick], q.new_tensor(target)))
        loss = torch.stack(classification).mean() + 0.1 * torch.stack(regression).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite imitation loss")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(policy.net.parameters(), 10.0)
        if not torch.isfinite(norm):
            raise FloatingPointError("nonfinite imitation gradient")
        optimizer.step()
        if step % 100 == 0:
            print(f"imitation step={step} loss={float(loss.detach()):.5f}", flush=True)
        if step not in checkpoints:
            continue
        with torch.no_grad():
            comparable = [(state, pick) for state, pick, _ in rows if len(state["cand"]) > 1]
            agreement = float(np.mean([int(policy(s).argmax()) == a for s, a in comparable]))
            return_error = float(np.mean([abs(float(policy(s)[a]) - r) for s, a, r in rows]))
        score = float(evaluate(policy.net.state_dict()))
        item = dict(updates=step, validation_objective=score,
                    teacher_action_agreement_nontrivial=agreement,
                    training_return_mae=return_error, seconds=time.perf_counter() - started)
        history.append(item)
        record_probe(item, policy.net.state_dict())
        if score < best_score:
            best_score, best_step = score, step
            best_weights = copy.deepcopy(policy.net.state_dict())
    return best_weights, dict(selected_updates=best_step, validation_objective=best_score,
                              pretraining_seconds=time.perf_counter() - started, probes=history)
