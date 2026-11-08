"""Daily mobility loss and sustained recovery, independent of scheduling and policies."""
import numpy as np


def daily_losses(traffic, context, *, disconnected_flow):
    """Rules (8)-(9): actual trips only, per-OD shortfalls, no horizon division."""
    if disconnected_flow != "unserved":
        raise ValueError("the finalized problem requires disconnected_flow='unserved'")
    positive = context["H0"] > 0
    if not positive.any():
        raise ValueError("the problem requires positive normal-period OD demand")
    q0 = context["H0"][positive]
    q = traffic.served_demand[positive]
    reference = float(q @ context["baseline_u"][positive])
    # A disconnected OD can have infinite observed time. Zero actual trips must
    # contribute zero, never 0 * infinity or an artificial penalty time.
    travelling = q > 0
    actual_time = float(q[travelling] @ traffic.costs[positive][travelling])
    time_ratio = actual_time / reference if reference > 0 else 1.0
    flow_loss = float(np.maximum(q0 - q, 0).sum() / q0.sum())
    return dict(time_loss=max(time_ratio - 1.0, 0.0), flow_loss=flow_loss,
                time_ratio=time_ratio, min_flow_fraction=float(np.min(q / q0)),
                total_flow_fraction=float(q.sum() / q0.sum()))


class RecoveryMonitor:
    def __init__(self, rules, context, disconnected_flow):
        if disconnected_flow != "unserved":
            raise ValueError("the finalized problem requires disconnected_flow='unserved'")
        self.rules, self.context = rules, context
        self.disconnected_flow = disconnected_flow
        self.total_loss = 0.0
        self.records = []
        self.restored_streak = 0
        self.finished = False

    def record(self, day, traffic, *, repaired, total_damaged, discovered):
        value = daily_losses(traffic, self.context, disconnected_flow=self.disconnected_flow)
        cost = self.rules.time_weight * value["time_loss"] + self.rules.flow_weight * value["flow_loss"]
        self.total_loss += cost
        restored = (repaired == total_damaged
                    and value["min_flow_fraction"] >= self.rules.restored_flow_fraction
                    and value["time_ratio"] <= self.rules.restored_time_ratio)
        self.restored_streak = self.restored_streak + 1 if restored else 0
        self.records.append(dict(day=day, cost=cost, **value, repaired=repaired,
                                 discovered=discovered, **traffic.diagnostics))
        self.finished = self.restored_streak >= self.rules.consecutive_restored_days
        return cost
