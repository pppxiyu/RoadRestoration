"""Compare the completed n11 75/25 and 60/40 discovery experiments.

Run after both Full RL and ten-scenario GA finish in the new experiment:

    python -m src.analysis.compare_discovery_share
"""
from __future__ import annotations

import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.experiment.daily import _write_json
from src.experiment.layout import ROOT


OLD = ROOT / "outputs/experiments/n11_discovery_response7d_1day_lhs50_s42_368acec5"
NEW = ROOT / "outputs/experiments/n11_discovery_response7d_1day_lhs50_s42_d35c7028"
FULL = "rl_s2v_saa64_adaptive_train_response7d_roadclass_seed42"
GA = "ga_saa10_flow_insert_seed42"


def _record(folder: Path, method: str, expected):
    run = folder / "data/methods" / method
    config = json.loads((run / "config/training.json").read_text(encoding="utf-8"))
    status = json.loads((run / "log/status.json").read_text(encoding="utf-8"))
    if status["status"] != "complete":
        raise ValueError(f"{run} has not completed")
    frame = pd.read_csv(run / "results/test_results.csv").set_index("scenario")
    if (len(frame) != 50 or frame.index.has_duplicates or
            set(frame.index) != set(expected) or not np.isfinite(frame.F).all()):
        raise ValueError(f"{run} does not contain the frozen 50 test scenarios")
    return frame.loc[expected, "F"].to_numpy(), config, status


def compare():
    manifests = [json.loads((folder / "data/manifest.json").read_text(encoding="utf-8"))
                 for folder in (OLD, NEW)]
    old_problem, new_problem = (item["identity"]["problem"] for item in manifests)
    ignore = {"rules", "public_roads", "hidden_quotas"}
    if ({k: v for k, v in old_problem.items() if k not in ignore} !=
            {k: v for k, v in new_problem.items() if k not in ignore}):
        raise ValueError("the two experiments differ in more than discovery allocation")
    old_rules, new_rules = (dict(item["rules"]) for item in (old_problem, new_problem))
    old_rules.pop("version")
    new_rules.pop("version")
    if old_rules != new_rules:
        raise ValueError("another daily problem rule changed")
    for key in ("evaluation_behavior", "disconnected_flow", "objective"):
        if manifests[0]["identity"][key] != manifests[1]["identity"][key]:
            raise ValueError(f"different scoring contract: {key}")

    records = {}
    for label, folder in (("75% public", OLD), ("60% public", NEW)):
        expected = [row["scenario_id"] for row in json.loads(
            (folder / "data/scenarios/test.json").read_text(encoding="utf-8"))]
        for method, display in ((FULL, "Full RL"), (GA, "GA on 10 scenarios")):
            values, config, status = _record(folder, method, expected)
            if config["identity"]["experiment"] != manifests[
                    0 if folder == OLD else 1]["identity"]:
                raise ValueError("method does not belong to its experiment")
            records[label, display] = dict(values=values, config=config, status=status)

    for method, fields in ((FULL, ("seed", "feature_schema", "hyperparameters",
                                  "stop", "ep_cap", "delivery")),
                           (GA, ("seed", "hyperparameters", "budget",
                                 "training_scenarios", "selected_training_indices",
                                 "evaluation_environment", "search_environment", "policy"))):
        left = records["75% public", "Full RL" if method == FULL else "GA on 10 scenarios"]["config"]
        right = records["60% public", "Full RL" if method == FULL else "GA on 10 scenarios"]["config"]
        if any(left[field] != right[field] for field in fields):
            raise ValueError(f"{method} has changed training settings")

    outcome = {}
    paired_rows = []
    for label in ("75% public", "60% public"):
        rl = records[label, "Full RL"]["values"]
        ga = records[label, "GA on 10 scenarios"]["values"]
        difference = ga - rl
        draws = np.random.default_rng(42).choice(
            difference, size=(5000, len(difference)), replace=True).mean(axis=1)
        outcome[label] = dict(
            public_roads=len((old_problem if label == "75% public" else new_problem)["public_roads"]),
            hidden_roads=11 - len((old_problem if label == "75% public" else new_problem)["public_roads"]),
            full_rl_mean=float(rl.mean()), full_rl_median=float(np.median(rl)),
            ga_mean=float(ga.mean()), ga_median=float(np.median(ga)),
            ga_minus_rl_mean=float(difference.mean()),
            rl_wins=int((difference > 1e-9).sum()),
            paired_mean_bootstrap95=np.quantile(draws, [.025, .975]).tolist(),
            full_rl_training_seconds=records[label, "Full RL"]["status"].get("training_seconds"),
            ga_search_seconds=records[label, "GA on 10 scenarios"]["status"].get("search_seconds"))
        paired_rows.extend(dict(setting=label, scenario=i, ga_F=float(ga[j]),
                                full_rl_F=float(rl[j]), ga_minus_rl=float(difference[j]))
                           for j, i in enumerate(range(50)))

    analysis = NEW / "data/analysis"
    (analysis / "results").mkdir(parents=True, exist_ok=True)
    _write_json(analysis / "results/discovery_share_comparison.json", dict(
        comparison=outcome,
        interpretation="Within-setting GA minus Full RL is paired on each setting's own 50 test scenarios. "
                       "Cross-setting scenario truth differs; means across settings are descriptive.",
        common_setting="n11; seed 42; 64 training, 11 validation, 50 test; response7d; "
                       "same objective, model inputs, training rules, GA budget and selected training indices"))
    pd.DataFrame(paired_rows).to_csv(
        analysis / "results/discovery_share_paired.csv", index=False)

    groups = [records[setting, method]["values"]
              for setting in ("75% public", "60% public")
              for method in ("Full RL", "GA on 10 scenarios")]
    labels = ["75% public\nFull RL", "75% public\nGA on 10 scenarios",
              "60% public\nFull RL", "60% public\nGA on 10 scenarios"]
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 17,
                         "axes.labelsize": 19, "xtick.labelsize": 15,
                         "ytick.labelsize": 15, "axes.spines.top": False,
                         "axes.spines.right": False}):
        fig, ax = plt.subplots(figsize=(12.2, 6.5), layout="constrained")
        x = (1, 2, 3.4, 4.4)
        boxes = ax.boxplot(groups, positions=x, widths=.52, patch_artist=True,
                           showfliers=False,
                           medianprops=dict(color="#263444", linewidth=2))
        colors = ("#37669E", "#B98558", "#7655A3", "#D49B70")
        jitter = np.random.default_rng(0).uniform(-.09, .09, 50)
        for center, values, box, color in zip(x, groups, boxes["boxes"], colors):
            box.set_facecolor(color)
            box.set_alpha(.62)
            ax.scatter(center + jitter, values, color=color, s=22, alpha=.55, linewidths=0)
        ax.set_xticks(x, labels)
        ax.set_ylabel("Cumulative transport loss (equivalent days)")
        ax.grid(axis="y", alpha=.18)
        ax.set_axisbelow(True)
        fig_path = analysis / "performance_distribution_discovery_share.png"
        fig.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    shutil.copy2(fig_path, NEW / fig_path.name)
    return outcome


if __name__ == "__main__":
    print(json.dumps(compare(), indent=2))
