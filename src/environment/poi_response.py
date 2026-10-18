"""Frozen October 12 behavior study: OD-specific sensitivity to travel time.

POI counts weight destination functions, not measured travel-purpose shares.
Rates are illustrative agreed settings, not fitted category-specific effects.
Yesterday's cumulative unmet demand fixes today's sensitivity throughout the
traffic fixed point. This model does not repay unmet trips or decay its history.
"""
from pathlib import Path
import hashlib

import numpy as np
import pandas as pd


COUNTS_PATH = Path(__file__).resolve().parents[2] / "data/siouxfalls_toy/behavior/destination_poi_counts.csv"
CATEGORIES = ("health", "industrial", "public", "commercial", "leisure")
RATES = (0.150, 0.090, 0.085, 0.055, 0.035)


def specification(zone_ids):
    counts = pd.read_csv(COUNTS_PATH).set_index("od_node")
    if counts.index.has_duplicates or set(counts.index) != set(zone_ids):
        raise ValueError("POI count destinations must match the public OD zones")
    values = counts.loc[list(zone_ids), list(CATEGORIES)].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any() or (values.sum(axis=1) <= 0).any():
        raise ValueError("every destination needs finite nonnegative POI counts with a positive total")
    return dict(model="poi_unmet_od_v1", categories=list(CATEGORIES), rates=list(RATES),
                floor=0.5, zone_ids=[int(z) for z in zone_ids], counts=values.astype(int).tolist(),
                count_source_sha256=hashlib.sha256(COUNTS_PATH.read_bytes()).hexdigest(),
                count_source="data/siouxfalls_toy/behavior/destination_poi_counts.csv",
                provenance="October 12 OSM POI counts in synthetic OD zones; illustrative category rates",
                history="sum through yesterday of max(0, 1 - served_OD / normal_OD), separately per OD",
                within_day="sensitivity held fixed during the daily demand/traffic fixed point")


def destination_weights(spec, od_pairs):
    counts = np.asarray(spec["counts"], dtype=float)
    weights = counts / counts.sum(axis=1, keepdims=True)
    positions = {z: i for i, z in enumerate(spec["zone_ids"])}
    return weights[[positions[int(d)] for _, d in od_pairs]]


def sensitivity(base_gamma, unmet, weights, spec):
    unmet = np.asarray(unmet, dtype=float)
    if not np.isfinite(unmet).all() or (unmet < 0).any():
        raise ValueError("cumulative unmet demand must be finite and nonnegative")
    floor = spec["floor"]
    ratios = floor + (1 - floor) * np.exp(-unmet[:, None] * np.asarray(spec["rates"])[None, :])
    return base_gamma * np.sum(weights * ratios, axis=1)
