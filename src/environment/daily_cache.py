"""Bounded storage for complete daily states, without demand or scoring rules."""
from collections import OrderedDict
from dataclasses import replace
import io
import json
from pathlib import Path
import sqlite3
import numpy as np
import pandas as pd
from src.environment.daily_contracts import TrafficState


def pack_state(state):
    stream = io.BytesIO()
    saved_solver = {}
    if state.solver_state is not None:
        warm = state.solver_state
        saved_solver = dict(path_lengths=np.array([len(p) for p in warm["path_arcs"]], dtype=int),
                            path_arcs=np.array([a for p in warm["path_arcs"] for a in p], dtype=int),
                            path_od=warm["path_od"], path_flow=warm["path_flow"],
                            solver_damage=np.asarray(warm["damage"], dtype=int).reshape(-1, 2))
    saved_response = {} if state.cumulative_unmet is None else dict(
        cumulative_unmet=state.cumulative_unmet, gamma_od=state.gamma_od)
    np.savez_compressed(stream, demand=state.demand, external=state.external,
                        costs=state.costs, reachable=state.reachable,
                        road_flow=state.road_flow, road_congestion=state.road_congestion,
                        links=state.links[["from", "to", "volume", "cost"]].to_numpy(),
                        diagnostics=json.dumps(state.diagnostics), **saved_solver, **saved_response)
    return stream.getvalue()


def unpack_state(key, day, payload):
    with np.load(io.BytesIO(payload), allow_pickle=False) as data:
        links = pd.DataFrame(data["links"], columns=["from", "to", "volume", "cost"])
        links[["from", "to"]] = links[["from", "to"]].astype(int)
        solver_state = None
        if "path_lengths" in data:
            ends = np.r_[0, np.cumsum(data["path_lengths"])]
            solver_state = dict(path_arcs=[tuple(data["path_arcs"][a:b].tolist())
                                          for a, b in zip(ends, ends[1:])],
                                path_od=data["path_od"], path_flow=data["path_flow"],
                                costs=data["costs"], reachable=data["reachable"],
                                damage=data["solver_damage"].tolist())
        return TrafficState(key, day, data["demand"], data["external"], data["costs"],
                            data["reachable"], data["road_flow"], data["road_congestion"],
                            links, json.loads(str(data["diagnostics"])), solver_state,
                            data["cumulative_unmet"] if "cumulative_unmet" in data else None,
                            data["gamma_od"] if "gamma_od" in data else None)


class TrafficCache:
    def __init__(self, fingerprint, directory=None, limit_bytes=None):
        self.memo = OrderedDict()
        self.limit_bytes = limit_bytes
        self.connection = None
        if directory is not None:
            folder = Path(directory)
            folder.mkdir(parents=True, exist_ok=True)
            self.connection = sqlite3.connect(folder / (fingerprint[:16] + ".sqlite3"), timeout=120)
            self.connection.execute("PRAGMA journal_mode=WAL")
            self.connection.execute("CREATE TABLE IF NOT EXISTS traffic (key TEXT PRIMARY KEY, payload BLOB)")

    def remember(self, key, state):
        self.memo[key] = state
        self.memo.move_to_end(key)
        while len(self.memo) > 2048:
            self.memo.popitem(last=False)
        return state

    def get(self, key, day):
        if key in self.memo:
            return replace(self.memo[key], day=day)
        if self.connection is not None:
            row = self.connection.execute("SELECT payload FROM traffic WHERE key=?", (key,)).fetchone()
            if row:
                return self.remember(key, unpack_state(key, day, row[0]))
        return None

    def put(self, key, state):
        if self.connection is not None:
            used = (0 if self.limit_bytes is None else
                    self.connection.execute("PRAGMA page_count").fetchone()[0] *
                    self.connection.execute("PRAGMA page_size").fetchone()[0])
            if self.limit_bytes is None or used < self.limit_bytes:
                with self.connection:
                    self.connection.execute("INSERT OR IGNORE INTO traffic VALUES (?, ?)",
                                            (key, pack_state(state)))
        return self.remember(key, state)

    def close(self):
        if self.connection is not None:
            self.connection.close()
            self.connection = None
