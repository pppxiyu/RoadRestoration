"""Four separate factor groupings using saved GA scores; no new simulations.

Descriptive in-sample group optima are distinguished from leave-one-out scheme
selection and a matched-size random-partition reference. The candidate library
was searched on all training scenarios, so leave-one-out is conditional on that
library and is NOT a new independent test experiment or causal decomposition.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import sklearn
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXPERIMENT = ROOT / "outputs/experiments/n11_discovery_response7d_1day_lhs50_s42_e4a9d3c3"
FACTORS = ("duration", "severity", "location", "discovery")
SEED, GROUPS, RESTARTS, SHUFFLES = 42, 4, 50, 2000


def features(records, public):
    public = sorted(public)
    hidden_ids = sorted({r["edge_id"] for s in records for r in s["roads"]} - set(public))
    arrays = {key: [] for key in FACTORS}
    names = {
        "duration": [f"true_duration_road_{e}" for e in public] + [f"anonymous_hidden_duration_rank_{j}" for j in range(1, 4)],
        "severity": [f"true_severity_road_{e}" for e in public] + [f"anonymous_hidden_severity_rank_{j}" for j in range(1, 4)],
        "location": [f"damaged_road_{e}" for e in hidden_ids],
        "discovery": [f"anonymous_hidden_discovery_rank_{j}" for j in range(1, 4)],
    }
    for scenario in records:
        roads = {r["edge_id"]: r for r in scenario["roads"]}
        hidden = [r for e, r in roads.items() if e not in public]
        assert len(roads) == 11 and len(public) == 8 and len(hidden) == 3
        for factor, field in (("duration", "true_duration"), ("severity", "true_severity")):
            arrays[factor].append([roads[e][field] for e in public] + sorted(r[field] for r in hidden))
        arrays["location"].append([int(e in roads) for e in hidden_ids])
        arrays["discovery"].append(sorted(r["discovery_day"] for r in hidden))
    return {key: np.asarray(value, dtype=float) for key, value in arrays.items()}, names


def group_fit(x, factor):
    # All coordinates carry equal standardized weight, except binary road
    # membership, where ordinary Euclidean distance counts identity mismatches.
    scaler = None if factor == "location" else StandardScaler().fit(x)
    transformed = x if scaler is None else scaler.transform(x)
    model = KMeans(n_clusters=GROUPS, n_init=RESTARTS, random_state=SEED, algorithm="lloyd").fit(transformed)
    return model, scaler


def best_by_group(scores, labels):
    return {int(label): int(np.argmin(scores[labels == label].mean(axis=0))) for label in np.unique(labels)}


def run(experiment):
    started = time.perf_counter()
    exp = Path(experiment)
    source = exp / "data/analysis/results/scenario_order_screening"
    paths = [source / "score_matrix.csv", source / "candidates.csv", source / "summary.json",
             exp / "data/scenarios/train.json", exp / "data/problem/public_problem.json", exp / "data/manifest.json"]
    before = {str(p.relative_to(exp)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    frame = pd.read_csv(paths[0], index_col="scenario", float_precision="round_trip")
    candidates = pd.read_csv(paths[1], float_precision="round_trip")
    previous = json.loads(paths[2].read_text())
    records_by_id = {s["scenario_id"]: s for s in json.loads(paths[3].read_text())}
    assert frame.index.is_unique and set(frame.index) == set(records_by_id)
    records = [records_by_id[i] for i in frame.index]
    public = json.loads(paths[4].read_text())["public_roads"]
    signatures = [hashlib.sha256(json.dumps(s["roads"], sort_keys=True).encode()).hexdigest() for s in records]
    manifest = json.loads(paths[5].read_text())
    assert signatures == manifest["identity"]["scenario_signatures"]["train"]
    scores = frame.to_numpy()
    assert scores.shape == (64, 461) and np.isfinite(scores).all()
    assert list(frame.columns) == [f"candidate_{j:03d}" for j in candidates.candidate]
    np.testing.assert_allclose(scores.mean(axis=0), candidates.mean_loss, rtol=0, atol=1e-10)
    fixed = int(scores.mean(axis=0).argmin())
    baseline = scores[:, fixed]
    hindsight = scores.min(axis=1)
    gap = float((baseline - hindsight).mean())
    assert abs(baseline.mean() - previous["best_common_mean"]) < 1e-10
    assert abs(gap - previous["mean_improvement"]) < 1e-10
    x_by_factor, columns = features(records, public)
    out = exp / "data/analysis/results/factor_group_screening"
    out.mkdir(parents=True, exist_ok=True)
    config = exp / "data/analysis/config/factor_group_screening.json"
    rows, groups, cases, crossfit, shuffled_rows = [], [], [], [], []
    n = len(records)
    loo_fixed_indices = np.array([int(np.argmin(np.delete(scores, i, axis=0).mean(axis=0))) for i in range(n)])
    loo_baseline = scores[np.arange(n), loo_fixed_indices]
    for factor in FACTORS:
        x = x_by_factor[factor]
        model, scaler = group_fit(x, factor)
        labels = model.labels_
        chosen = best_by_group(scores, labels)
        picks = np.array([chosen[int(label)] for label in labels])
        losses = scores[np.arange(n), picks]
        gain = float((baseline - losses).mean())
        assert -1e-10 <= gain <= gap + 1e-10
        # Match group sizes exactly, breaking their relation to scenario factors.
        rng = np.random.default_rng(SEED)
        null = []
        for repeat in range(SHUFFLES):
            random_labels = rng.permutation(labels)
            group_picks = best_by_group(scores, random_labels)
            random_loss = np.array([scores[i, group_picks[int(label)]] for i, label in enumerate(random_labels)])
            null.append(float((baseline - random_loss).mean()))
            shuffled_rows.append(dict(factor=factor, repetition=repeat, mean_gain=null[-1]))
        loo_loss = np.empty(n)
        for held in range(n):
            mask = np.arange(n) != held
            fit, scale = group_fit(x[mask], factor)
            target = x[held:held+1] if scale is None else scale.transform(x[held:held+1])
            label = int(fit.predict(target)[0])
            train_labels = fit.labels_
            pick = best_by_group(scores[mask], train_labels)[label]
            loo_loss[held] = scores[held, pick]
            crossfit.append(dict(factor=factor, scenario=frame.index[held], group=label,
                group_training_size=int((train_labels == label).sum()), chosen_candidate=pick,
                fixed_candidate=int(loo_fixed_indices[held]), fixed_loss=float(loo_baseline[held]),
                selected_loss=float(loo_loss[held]), improvement=float(loo_baseline[held]-loo_loss[held])))
        for label, pick in chosen.items():
            members = np.flatnonzero(labels == label)
            groups.append(dict(factor=factor, group=label+1, scenarios=len(members), candidate=pick,
                priority_order=candidates.iloc[pick].order, mean_fixed_loss=float(baseline[members].mean()),
                mean_selected_loss=float(losses[members].mean()),
                members=";".join(frame.index[members])))
        for i in range(n):
            cases.append(dict(factor=factor, scenario=frame.index[i], group=int(labels[i])+1,
                chosen_candidate=int(picks[i]), fixed_loss=float(baseline[i]), selected_loss=float(losses[i]),
                hindsight_loss=float(hindsight[i]), improvement=float(baseline[i]-losses[i])))
        pd.DataFrame(x, index=frame.index, columns=columns[factor]).to_csv(out / f"features_{factor}.csv", index_label="scenario")
        pd.DataFrame(model.cluster_centers_, columns=columns[factor]).to_csv(out / f"standardized_centers_{factor}.csv", index_label="group_zero_based")
        row = dict(factor=factor, group_sizes=[int((labels == label).sum()) for label in sorted(chosen)],
            group_mean_objective=float(losses.mean()), mean_gain=gain,
            improvement_percent=100*gain/float(baseline.mean()), captured_candidate_gap_percent=100*gain/gap,
            chosen_distinct_candidates=len(set(picks)),
            random_partition_gain_mean=float(np.mean(null)),
            random_partition_gain_95_range=np.quantile(null, [.025, .975]).tolist(),
            fraction_random_partitions_at_least_observed=float(np.mean(np.asarray(null) >= gain-1e-12)),
            loo_fixed_mean=float(loo_baseline.mean()), loo_group_mean=float(loo_loss.mean()),
            loo_mean_gain=float((loo_baseline-loo_loss).mean()),
            loo_improvement_percent=100*float((loo_baseline-loo_loss).mean())/float(loo_baseline.mean()))
        rows.append(row)
        print(json.dumps(row), flush=True)
    pd.DataFrame(rows).to_csv(out / "summary.csv", index=False)
    pd.DataFrame(groups).to_csv(out / "groups.csv", index=False)
    pd.DataFrame(cases).to_csv(out / "scenario_results.csv", index=False)
    pd.DataFrame(crossfit).to_csv(out / "leave_one_out.csv", index=False)
    pd.DataFrame(shuffled_rows).to_csv(out / "random_partitions.csv", index=False)
    report = dict(scenarios=n, candidates=len(candidates), fixed_candidate=fixed,
        fixed_mean=float(baseline.mean()), hindsight_mean=float(hindsight.mean()), candidate_gap=gap,
        groups_per_factor=GROUPS, factors=rows, elapsed_seconds=time.perf_counter()-started,
        new_simulations=0, new_optimization_runs=0,
        interpretation="descriptive factor-conditioned grouping benefits within the fixed GA candidate library; not causal shares or policy-achievable information value")
    (out / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    specification = dict(source_files_sha256=before, groups=GROUPS, seed=SEED,
        grouping="KMeans; 50 starts; standardized continuous/ordinal columns; unscaled binary road membership",
        rationale="four common groups for 64 scenarios; chosen before inspecting outcome; no objective-based tuning",
        feature_columns=columns, public_roads=sorted(public),
        hidden_identity_handling="three hidden values sorted anonymously, separately within each factor; no missing-value mask; loses road/value association for hidden roads",
        unseen_road_note="anonymous hidden durations may still statistically correlate with location through road class; factors are not independent",
        score_aggregation="group-specific column argmin, then scenario-weighted mean, never unweighted group mean",
        crossfit="leave one scenario out; refit scaling, groups and column selection on 63; assign nearest centroid; baseline also selected on same 63",
        crossfit_limitation="candidate library originally searched on all64; this is internal screening, not an independent test",
        random_reference=dict(repetitions=SHUFFLES, method="permute memberships preserving sizes; descriptive chance reference, not a causal test"),
        versions=dict(numpy=np.__version__, pandas=pd.__version__, sklearn=sklearn.__version__),
        excluded_scenarios=[], factor_combinations_computed=False)
    config.write_text(json.dumps(specification, indent=2), encoding="utf-8")
    assert before == {str(p.relative_to(exp)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    print(json.dumps({k: v for k, v in report.items() if k != "factors"}, indent=2), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, default=DEFAULT_EXPERIMENT)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.experiment)
