"""Available demand-behavior mechanisms and their physical time units.

This module describes environment options; the experiment runner decides which
one to invoke. The daily discovery experiment connects the natural-only and
seven-day response mechanisms. Historical mechanisms remain available for
isolated studies, but cannot be selected for current method comparisons.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class BehaviorModel:
    name: str
    slot_hours: float
    method_comparison_ready: bool


MODELS = {
    "damage_shortfall": BehaviorModel("damage_shortfall", 3.0, False),
    "elastic_daily": BehaviorModel("elastic_daily", 24.0, False),
    "natural_daily": BehaviorModel("natural_daily", 24.0, True),
    "response7d_daily": BehaviorModel("response7d_daily", 24.0, True),
}


def select_behavior_model(name: str, slot_hours: float, *, for_methods: bool) -> BehaviorModel:
    """Reject mismatched clocks and an unconnected model/method combination."""
    if name not in MODELS:
        raise ValueError(f"unknown human-behavior model {name!r}; choose from {sorted(MODELS)}")
    model = MODELS[name]
    if float(slot_hours) != model.slot_hours:
        raise ValueError(f"{name} requires {model.slot_hours:g}-hour slots, got {slot_hours:g}")
    if for_methods and not model.method_comparison_ready:
        raise ValueError(f"{name} is a historical or standalone environment option, "
                         "not part of the finalized optimization and comparison problem")
    return model
