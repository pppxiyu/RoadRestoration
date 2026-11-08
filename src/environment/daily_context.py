"""Public network context and reproducibility identity for the daily environment."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import networkx as nx
import numpy as np
from src import config as P
from src.environment.traffic_network import od_travel_times


def context_from_problem(problem):
    edges, od, zones = problem["edges"], problem["od"], problem["zone_ids"]
    positions = {int(z): i for i, z in enumerate(zones)}
    pairs = [(int(r.origin), int(r.destination)) for r in od.itertuples(index=False)]
    context = dict(edges=edges, zone_ids=zones, od_pairs=pairs,
                   H0=od.h0.to_numpy(dtype=float), nz=len(zones),
                   oi=np.array([positions[o] for o, _ in pairs]),
                   di=np.array([positions[d] for _, d in pairs]),
                   origins_unique=sorted({o for o, _ in pairs}),
                   edge_row={int(r.edge_id): i for i, r in enumerate(edges.itertuples(index=False))})
    context["baseline_u"] = od_travel_times(problem["baseline_links"], context)
    if not np.isfinite(context["baseline_u"]).all():
        raise ValueError("normal-period network must connect all positive-demand OD pairs")
    context["u_pen"] = P.UPEN_FACTOR * float(context["baseline_u"].max())
    # Public structural incidence, based only on the intact free-flow network.
    graph = nx.Graph()
    for r in edges.itertuples(index=False):
        graph.add_edge(int(r.u), int(r.v), weight=float(r.free_flow_time), edge=int(r.edge_id))
    incidence = np.zeros((len(pairs), len(edges)))
    for i, (o, d) in enumerate(pairs):
        path = nx.shortest_path(graph, o, d, weight="weight")
        for a, b in zip(path, path[1:]):
            incidence[i, context["edge_row"][graph[a][b]["edge"]]] = 1.0
    context["incidence"] = incidence
    return context


def simulator_fingerprint(problem, gamma, model_path):
    """Invalidate caches when any participating component changes."""
    folder = Path(__file__).parent
    names = (
        "daily.py", "daily_context.py", "daily_contracts.py", "daily_demand.py",
        "daily_traffic.py", "daily_cache.py", "daily_restoration.py",
        "daily_observation.py", "daily_performance.py", "daily_episode.py",
        "traffic_network.py", "ue.py", "shifted_demand.py", "fixed_point_demand.py",
        "poi_response.py", "gravity_model.py",
    )
    sources = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
               for p in [*(folder / name for name in names), Path(model_path)]}
    identity = dict(rules=asdict(problem["rules"]), gamma=gamma,
                    initial=P.RECOVERY_INITIAL_LEVEL, plateau=P.RECOVERY_PLATEAU_LEVEL,
                    cap=P.CAP_RETAIN, speed=P.SPEED_RETAIN, sever=P.SEVER_SEVERITY,
                    penalty=P.UPEN_FACTOR, warm_start=P.UE_WARM_START, sources=sources,
                    network=hashlib.sha256(problem["edges"].to_csv(index=False).encode()).hexdigest(),
                    od=hashlib.sha256(problem["od"].to_csv(index=False).encode()).hexdigest())
    if problem.get("human_response") is not None:
        identity["human_response"] = problem["human_response"]
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
