"""Compare daily policies only after checking their common scoring contract."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def refresh_comparison(folder):
    from src.analysis.viz.daily_viz import make_comparison
    folder = Path(folder)
    manifest = json.loads((folder / "data" / "manifest.json").read_text(encoding="utf-8"))
    expected = json.loads((folder / "data" / "scenarios" / "test.json").read_text(encoding="utf-8"))
    expected_ids = [s["scenario_id"] for s in expected]
    collected, run_times = [], []
    for path in sorted((folder / "data" / "methods").glob("*/results/test_results.csv")):
        run = path.parent.parent
        contract = json.loads((run / "config" / "training.json").read_text(encoding="utf-8"))
        if contract["identity"]["experiment"] != manifest["identity"]:
            raise ValueError(f"scoring identity mismatch in {run.name}")
        frame = pd.read_csv(path)
        if frame.scenario.duplicated().any() or set(frame.scenario) != set(expected_ids):
            raise ValueError(f"test scenario mismatch in {run.name}")
        if not np.isfinite(frame.F).all():
            raise ValueError(f"invalid objective in {run.name}")
        frame = frame.set_index("scenario").loc[expected_ids].reset_index()
        behavior = contract["identity"]["training_behavior"]
        frame["method"] = run.name
        frame["training_behavior"] = behavior
        frame["input_variant"] = contract["identity"].get("input_variant", "original")
        frame["training_seed"] = contract["seed"]
        collected.append(frame)
        training = json.loads((run / "results" / "training_summary.json").read_text())
        evaluation = json.loads((run / "results" / "evaluation_summary.json").read_text())
        run_times.append(dict(method=run.name, **training, **evaluation))
    if not collected:
        return None
    table = pd.concat(collected, ignore_index=True)
    analysis = folder / "data" / "analysis"
    (analysis / "results").mkdir(parents=True, exist_ok=True)
    (analysis / "config").mkdir(parents=True, exist_ok=True)
    table.to_csv(analysis / "results" / "comparison.csv", index=False)
    table.to_csv(folder / "data" / "evaluation.csv", index=False)
    pd.DataFrame(run_times).to_csv(analysis / "results" / "computation_time.csv", index=False)
    (analysis / "config" / "comparison.json").write_text(json.dumps(dict(
        evaluation_environment="response7d", test_scenarios=len(expected_ids),
        pairing="same scenario_id and complete scenario truth, checked against experiment manifest",
        objective="cumulative equally weighted time and served-flow losses",
        uncertainty="scenario bootstrap, conditional on each one trained model",
        figure=dict(points="individual test scenarios", box="25th to 75th percentiles",
                    center="median", whiskers="most extreme observed values within 1.5 IQR",
                    outliers="all scenarios retained as overlaid points",
                    training_replication="one model per training seed; scenarios are not model-training replicates",
                    significance_annotations=False, output_format="PNG")), indent=2), encoding="utf-8")
    summaries = []
    for seed, frame in table.groupby("training_seed"):
        frame = frame[frame.input_variant == "original"]
        wide = frame.pivot(index="scenario", columns="training_behavior", values="F")
        if {"natural", "response7d"} <= set(wide.columns):
            difference = (wide.natural - wide.response7d).to_numpy()
            rng = np.random.default_rng(42)
            draws = rng.choice(difference, size=(5000, len(difference)), replace=True).mean(axis=1)
            summaries.append(dict(training_seed=int(seed),
                                  difference="natural-trained objective minus response-trained objective",
                                  mean_difference=float(difference.mean()),
                                  median_difference=float(np.median(difference)),
                                  relative_mean_improvement=float(difference.mean() / wide.natural.mean()),
                                  response_wins=int(np.sum(difference > 1e-9)),
                                  ties=int(np.sum(np.abs(difference) <= 1e-9)),
                                  bootstrap_mean_95_interval=np.quantile(draws, [0.025, 0.975]).tolist()))
    (analysis / "results" / "paired_comparison.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    # Keep the established RL comparison above, and add every actual method pair.
    from itertools import combinations
    pairs = []
    wide_methods = table.pivot(index="scenario", columns="method", values="F")
    for first, second in combinations(wide_methods.columns, 2):
        difference = (wide_methods[first] - wide_methods[second]).to_numpy()
        rng = np.random.default_rng(42)
        draws = rng.choice(difference, size=(5000, len(difference)), replace=True).mean(axis=1)
        pairs.append(dict(first=first, second=second, difference="first minus second; positive favors second",
                          mean_difference=float(difference.mean()),
                          relative_mean_improvement=float(difference.mean() / wide_methods[first].mean()),
                          second_wins=int(np.sum(difference > 1e-9)),
                          ties=int(np.sum(np.abs(difference) <= 1e-9)),
                          bootstrap_mean_95_interval=np.quantile(draws, [0.025, 0.975]).tolist(),
                          scope="scenario variability conditional on one trained/search-seeded policy per method"))
    (analysis / "results" / "all_method_pairs.json").write_text(json.dumps(pairs, indent=2), encoding="utf-8")
    comparison_plan = analysis / "config/rl_ablation_plan.json"
    if comparison_plan.exists():
        refresh_controlled_comparisons(table, analysis, json.loads(comparison_plan.read_text(encoding="utf-8")))
    else:
        make_comparison(table, analysis)
    ablation = table[table.training_behavior == "response7d"]
    if {"original", "roadclass"} <= set(ablation.input_variant):
        make_comparison(ablation, analysis, filename="performance_distribution_roadclass.png")
        comparisons = []
        for seed, frame in ablation.groupby("training_seed"):
            wide = frame.pivot(index="scenario", columns="input_variant", values="F")
            if not {"original", "roadclass"} <= set(wide.columns):
                continue
            differences = (wide.original - wide.roadclass).to_numpy()
            rng = np.random.default_rng(42)
            draws = rng.choice(differences, (5000, len(differences)), replace=True).mean(axis=1)
            comparisons.append(dict(seed=int(seed), test_scenarios=len(differences),
                original_mean=float(wide.original.mean()), roadclass_mean=float(wide.roadclass.mean()),
                difference="original minus roadclass; positive favors roadclass",
                mean_difference=float(differences.mean()),
                relative_mean_improvement=float(differences.mean() / wide.original.mean()),
                roadclass_wins=int((differences > 1e-9).sum()), ties=int((np.abs(differences) <= 1e-9).sum()),
                bootstrap_mean_95_interval=np.quantile(draws, [0.025, 0.975]).tolist(),
                scope="paired scenario uncertainty, not training-seed uncertainty"))
        (analysis / "results/roadclass_comparison.json").write_text(json.dumps(comparisons, indent=2), encoding="utf-8")
    return table


def refresh_controlled_comparisons(table, analysis, plan):
    """Deliver only the two prespecified contrasts for the three-RL experiment."""
    from src.analysis.viz.daily_viz import make_controlled_pair
    analysis = Path(analysis)
    wanted = {method for pair in plan["comparisons"] for method in pair["methods"]}
    available = set(table.method)
    active = table[table.method.isin(wanted)]
    # Both files use the same vertical range, updated as additional runs finish.
    y_max = max(float(active.F.max()) * 1.08, 1.) if not active.empty else 1.
    results = []
    for pair in plan["comparisons"]:
        first, second = pair["methods"]
        if not {first, second} <= available:
            results.append(dict(comparison=pair["name"], status="waiting_for_methods",
                                missing_methods=sorted({first, second} - available)))
            continue
        data = table[table.method.isin((first, second))].copy()
        wide = data.pivot(index="scenario", columns="method", values="F").sort_index()
        if wide.isna().any().any() or len(wide) != plan["test_scenarios"]:
            raise ValueError("controlled comparison needs the identical complete test sample")
        # Sort identically in each group so equal jitter offsets denote paired worlds.
        data = data.sort_values(["method", "scenario"])
        difference = (wide[first] - wide[second]).to_numpy()
        draws = np.random.default_rng(42).choice(difference, (5000, len(difference)), replace=True).mean(axis=1)
        result = dict(comparison=pair["name"], status="complete", first=first, second=second,
                      test_scenarios=len(wide), first_mean=float(wide[first].mean()),
                      second_mean=float(wide[second].mean()), mean_difference=float(difference.mean()),
                      difference="first minus second; positive favors second",
                      relative_mean_improvement=float(difference.mean() / wide[first].mean()) if wide[first].mean() else None,
                      second_wins=int((difference > 1e-9).sum()), ties=int((abs(difference) <= 1e-9).sum()),
                      bootstrap_mean_95_interval=np.quantile(draws, [.025, .975]).tolist(),
                      scope="paired scenario variation, conditional on one training seed per policy",
                      figure=pair["filename"])
        results.append(result)
        wide.assign(first_minus_second=difference).to_csv(analysis / "results" / (pair["name"] + "_paired.csv"))
        make_controlled_pair(data, analysis, methods=pair["methods"], labels=pair["labels"],
                             colors=pair["colors"], filename=pair["filename"], y_max=y_max)
    (analysis / "results/rl_ablation_comparisons.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8")
    return results
