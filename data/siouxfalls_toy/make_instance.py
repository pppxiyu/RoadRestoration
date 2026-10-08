"""
make_instance.py  --  Step 1 of the road-recovery toy util.

Builds the STATIC network instance + the disruption instance for a small
post-flood road-recovery toy, from the public Sioux Falls benchmark
(bstabler/TransportationNetworks, academic-research-only license).

Outputs (all relative to this file):
    network/nodes.csv            node_id, x, y                              (24)
    network/edges.csv            edge_id, u, v, capacity, length,
                                 free_flow_time, bpr_alpha, bpr_beta,
                                 road_class                                 (38 undirected)
    network/od_pairs.csv         od_id, origin, destination, h0            (positive OD flows)
    disruption/disrupted_segments.csv
                                 edge_id, u, v, road_class, severity, level_id  (8 = E)
    figures/01_network.png ... 05_disruption.png

Deferred to LATER steps (NOT produced here): duration scenarios, demand-model
matrices A/B/u_pen, severity->capacity map, UE, and (s, a*) labels.

Reproducible: fixed SEED. Re-running regenerates identical CSVs.
Run:  python make_instance.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib

matplotlib.use("Agg")  # headless / deterministic
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# shared Nature-style helpers (make project root importable for `viz`)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from viz.style import (C, CMAP_DEMAND, CMAP_SEQ, panel_label, roadclass_color,
                       save_pub, severity_color, use_pub)

# --------------------------------------------------------------------------- #
# Parameters (this step only)
# --------------------------------------------------------------------------- #
SEED = 42
N_DISRUPTED = 8
ROAD_CLASSES = ["local", "major", "highway"]  # low -> high capacity

HERE = Path(__file__).resolve().parent
RAW = HERE / "raw"
NET_DIR = HERE / "network"
DIS_DIR = HERE / "instances"
FIG_DIR = Path(__file__).resolve().parents[2] / "outputs" / "studies" / "problem_setting" / "source_figures"
for d in (NET_DIR, DIS_DIR, FIG_DIR):
    d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Parsers for the TNTP format
# --------------------------------------------------------------------------- #
def parse_net(path):
    """Return a DataFrame of directed links from a *_net.tntp file."""
    rows = []
    for line in path.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("~") or s.startswith("<"):
            continue
        s = s.rstrip(";").strip()
        tok = s.split()
        if len(tok) < 10:
            continue
        rows.append(
            dict(
                init_node=int(tok[0]),
                term_node=int(tok[1]),
                capacity=float(tok[2]),
                length=float(tok[3]),
                free_flow_time=float(tok[4]),
                bpr_alpha=float(tok[5]),
                bpr_beta=float(tok[6]),
            )
        )
    return pd.DataFrame(rows)


def parse_nodes(path):
    rows = []
    for line in path.read_text().splitlines():
        s = line.strip()
        if not s or s.lower().startswith("node"):
            continue
        s = s.rstrip(";").strip()
        tok = s.split()
        if len(tok) < 3:
            continue
        rows.append(dict(node_id=int(tok[0]), x=float(tok[1]), y=float(tok[2])))
    return pd.DataFrame(rows).sort_values("node_id").reset_index(drop=True)


def parse_trips(path):
    """Parse origin-block OD format -> long DataFrame of positive flows."""
    rows = []
    origin = None
    for line in path.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("<"):
            continue
        if s.lower().startswith("origin"):
            origin = int(s.split()[1])
            continue
        if origin is None:
            continue
        for chunk in s.split(";"):
            chunk = chunk.strip()
            if not chunk:
                continue
            dest, flow = chunk.split(":")
            rows.append(dict(origin=origin, destination=int(dest), flow=float(flow)))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Build undirected edge table + road_class
# --------------------------------------------------------------------------- #
def build_edges(net):
    """Collapse directed links to undirected edges; assert symmetry."""
    canon = {}
    for r in net.itertuples(index=False):
        u, v = sorted((r.init_node, r.term_node))
        key = (u, v)
        attrs = dict(
            capacity=r.capacity,
            length=r.length,
            free_flow_time=r.free_flow_time,
            bpr_alpha=r.bpr_alpha,
            bpr_beta=r.bpr_beta,
        )
        if key in canon:
            prev = canon[key]
            for k in ("capacity", "length", "free_flow_time"):
                if not np.isclose(prev[k], attrs[k], rtol=1e-3):
                    print(f"  [warn] asymmetric {k} on edge {key}: "
                          f"{prev[k]} vs {attrs[k]} (averaging)")
                    prev[k] = 0.5 * (prev[k] + attrs[k])
        else:
            canon[key] = attrs
    edges = (
        pd.DataFrame(
            [dict(u=u, v=v, **a) for (u, v), a in canon.items()]
        )
        .sort_values(["u", "v"])
        .reset_index(drop=True)
    )
    edges.insert(0, "edge_id", np.arange(1, len(edges) + 1))

    # road_class by capacity bands at the 33rd/67th percentiles (higher capacity => higher class)
    q1, q2 = np.quantile(edges["capacity"], [1 / 3, 2 / 3])

    def cls(c):
        if c < q1:
            return "local"
        if c < q2:
            return "major"
        return "highway"

    edges["road_class"] = edges["capacity"].apply(cls)
    return edges, (q1, q2)


# --------------------------------------------------------------------------- #
# Disruption instance (seeded; one edge per road type, rest random)
# --------------------------------------------------------------------------- #
def build_disruption(edges):
    rng = np.random.default_rng(SEED)
    chosen = []
    # 1) guarantee >=1 edge of each road_class
    for rc in ROAD_CLASSES:
        pool = edges.loc[edges["road_class"] == rc, "edge_id"].to_numpy()
        chosen.append(int(rng.choice(pool)))
    # 2) fill the rest at random from what remains
    remaining = edges.loc[~edges["edge_id"].isin(chosen), "edge_id"].to_numpy()
    extra = rng.choice(remaining, size=N_DISRUPTED - len(chosen), replace=False)
    chosen.extend(int(e) for e in extra)
    chosen = sorted(chosen)

    # 3) severities cycle 1,2,3 over a seeded shuffle -> balanced, diverse levels
    order = rng.permutation(len(chosen))
    sev = np.empty(len(chosen), dtype=int)
    for rank, idx in enumerate(order):
        sev[idx] = (rank % 3) + 1

    sub = edges.set_index("edge_id").loc[chosen].reset_index()
    sub["severity"] = sev
    sub["level_id"] = sub["road_class"] + "-S" + sub["severity"].astype(str)
    return sub[["edge_id", "u", "v", "road_class", "severity", "level_id"]]


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def graph_and_pos(nodes, edges):
    G = nx.Graph()
    for r in nodes.itertuples(index=False):
        G.add_node(r.node_id)
    for r in edges.itertuples(index=False):
        G.add_edge(r.u, r.v, edge_id=r.edge_id)
    pos = {r.node_id: (r.x, r.y) for r in nodes.itertuples(index=False)}
    return G, pos


def _draw_nodes_labels(ax, G, pos):
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=70, node_color="#f0f0f0",
                           edgecolors=C["neutral_mid"], linewidths=0.5)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=4.5)


def fig_network(nodes, edges):
    G, pos = graph_and_pos(nodes, edges)
    fig, ax = plt.subplots(figsize=(3.6, 3.9))
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=C["neutral_mid"], width=1.0)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=110, node_color="#dce6f2",
                           edgecolors=C["accent"], linewidths=0.6)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=4.5)
    ax.set_title(f"Sioux Falls network: {len(nodes)} nodes, {len(edges)} undirected edges")
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
    ax.set_aspect("equal"); ax.grid(alpha=0.15, lw=0.5)
    fig.tight_layout(); save_pub(fig, FIG_DIR / "01_network"); plt.close(fig)


def fig_road_class(nodes, edges):
    G, pos = graph_and_pos(nodes, edges)
    fig, ax = plt.subplots(figsize=(3.6, 3.9))
    counts = edges["road_class"].value_counts()
    for rc in ROAD_CLASSES:
        elist = [(r.u, r.v) for r in edges.itertuples(index=False) if r.road_class == rc]
        nx.draw_networkx_edges(G, pos, ax=ax, edgelist=elist,
                               edge_color=roadclass_color(rc), width=2.4)
    _draw_nodes_labels(ax, G, pos)
    handles = [Line2D([0], [0], color=roadclass_color(rc), lw=2.4,
                      label=f"{rc} (n={int(counts.get(rc, 0))})") for rc in ROAD_CLASSES]
    ax.legend(handles=handles, loc="upper left", title="road class (capacity bands)")
    ax.set_title("Inferred road classification")
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
    ax.set_aspect("equal"); ax.grid(alpha=0.15, lw=0.5)
    fig.tight_layout(); save_pub(fig, FIG_DIR / "04_road_class"); plt.close(fig)


def fig_physical(nodes, edges):
    G, pos = graph_and_pos(nodes, edges)
    fig = plt.figure(figsize=(7.2, 3.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.7, 1])
    axN = fig.add_subplot(gs[:, 0])
    cap = edges["capacity"].to_numpy()
    widths = 0.8 + 4.5 * (cap - cap.min()) / (cap.max() - cap.min())
    elist = [(r.u, r.v) for r in edges.itertuples(index=False)]
    ec = nx.draw_networkx_edges(G, pos, ax=axN, edgelist=elist, width=widths,
                                edge_color=cap, edge_cmap=plt.get_cmap(CMAP_SEQ))
    _draw_nodes_labels(axN, G, pos)
    axN.set_title("edge width & colour = capacity"); axN.set_aspect("equal"); axN.axis("off")
    cb = fig.colorbar(ec, ax=axN, fraction=0.046, pad=0.02)
    cb.set_label("capacity (veh/h)"); cb.outline.set_linewidth(0.6)
    panel_label(axN, "a", x=0.0)
    axb = fig.add_subplot(gs[0, 1])
    axb.hist(edges["capacity"], bins=12, color=C["accent2"], edgecolor="white")
    axb.set_xlabel("capacity (veh/h)"); axb.set_ylabel("# edges")
    panel_label(axb, "b", x=-0.26)
    axc = fig.add_subplot(gs[1, 1])
    axc.hist(edges["free_flow_time"], bins=12, color=C["teal"], edgecolor="white")
    axc.set_xlabel("free-flow time"); axc.set_ylabel("# edges")
    panel_label(axc, "c", x=-0.26)
    fig.suptitle("Physical link attributes", fontsize=8.5)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_pub(fig, FIG_DIR / "02_physical_quantities"); plt.close(fig)


def fig_od(trips, n_zones):
    M = np.zeros((n_zones, n_zones))
    for r in trips.itertuples(index=False):
        M[r.origin - 1, r.destination - 1] = r.h0
    fig = plt.figure(figsize=(7.2, 3.0))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1, 1])
    ax0 = fig.add_subplot(gs[0, 0])
    im = ax0.imshow(M / 1e2, cmap=CMAP_DEMAND, origin="upper")
    ax0.set_title("OD demand matrix $H^{t_0}$")
    ax0.set_xlabel("destination zone"); ax0.set_ylabel("origin zone")
    ax0.set_xticks(range(0, n_zones, 4)); ax0.set_xticklabels(range(1, n_zones + 1, 4))
    ax0.set_yticks(range(0, n_zones, 4)); ax0.set_yticklabels(range(1, n_zones + 1, 4))
    cb = fig.colorbar(im, ax=ax0, fraction=0.046, pad=0.04)
    cb.set_label("trips ($\\times10^2$)"); cb.outline.set_linewidth(0.6)
    panel_label(ax0, "a", x=-0.28)
    z = np.arange(1, n_zones + 1)
    ax1 = fig.add_subplot(gs[0, 1]); ax1.bar(z, M.sum(1) / 1e3, color=C["accent"], width=0.75)
    ax1.set_title("production"); ax1.set_xlabel("origin zone"); ax1.set_ylabel("trips out ($\\times10^3$)")
    panel_label(ax1, "b", x=-0.3)
    ax2 = fig.add_subplot(gs[0, 2]); ax2.bar(z, M.sum(0) / 1e3, color=C["teal"], width=0.75)
    ax2.set_title("attraction"); ax2.set_xlabel("destination zone"); ax2.set_ylabel("trips in ($\\times10^3$)")
    panel_label(ax2, "c", x=-0.3)
    fig.suptitle(f"Pre-disaster OD demand (total {M.sum():,.0f} trips)", fontsize=8.5)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_pub(fig, FIG_DIR / "03_od_demand"); plt.close(fig)


def fig_disruption(nodes, edges, disr):
    G, pos = graph_and_pos(nodes, edges)
    fig, ax = plt.subplots(figsize=(3.6, 3.9))
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=C["neutral_light"], width=1.0)
    for s in (1, 2, 3):
        elist = [(r.u, r.v) for r in disr.itertuples(index=False) if r.severity == s]
        nx.draw_networkx_edges(G, pos, ax=ax, edgelist=elist,
                               edge_color=severity_color(s), width=4.0)
    _draw_nodes_labels(ax, G, pos)
    handles = [Line2D([0], [0], color=severity_color(s), lw=4, label=f"severity {s}")
               for s in (1, 2, 3)]
    ax.legend(handles=handles, loc="upper left", title=f"disrupted segments (|E|={len(disr)})")
    ax.set_title("Disruption instance")
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
    ax.set_aspect("equal"); ax.grid(alpha=0.15, lw=0.5)
    fig.tight_layout(); save_pub(fig, FIG_DIR / "05_disruption"); plt.close(fig)


# --------------------------------------------------------------------------- #
def main():
    print("Parsing raw Sioux Falls files ...")
    net = parse_net(RAW / "SiouxFalls_net.tntp")
    nodes = parse_nodes(RAW / "SiouxFalls_node.tntp")
    trips = parse_trips(RAW / "SiouxFalls_trips.tntp")
    print(f"  directed links={len(net)}, nodes={len(nodes)}, OD entries={len(trips)}")

    edges, (q1, q2) = build_edges(net)
    print(f"  undirected edges={len(edges)}")
    print(f"  capacity cut-points (33rd/67th pct): q1={q1:.1f}, q2={q2:.1f}")
    print("  road_class counts:")
    print(edges["road_class"].value_counts().to_string())

    od = trips[trips["flow"] > 0].copy().reset_index(drop=True)
    od.insert(0, "od_id", np.arange(1, len(od) + 1))
    od = od.rename(columns={"flow": "h0"})[["od_id", "origin", "destination", "h0"]]
    print(f"  positive OD pairs={len(od)}, total trips={od['h0'].sum():.0f}")

    disr = build_disruption(edges)
    print("\nDisruption instance:")
    print(disr.to_string(index=False))

    # --- write CSVs (sorted, stable -> same output every run) ---
    nodes.to_csv(NET_DIR / "nodes.csv", index=False)
    edges.to_csv(NET_DIR / "edges.csv", index=False)
    od.to_csv(NET_DIR / "od_pairs.csv", index=False)
    disr.to_csv(DIS_DIR / "disrupted_segments.csv", index=False)

    # --- figures ---
    print("\nRendering figures ...")
    use_pub()
    fig_network(nodes, edges)
    fig_physical(nodes, edges)
    fig_od(od, len(nodes))
    fig_road_class(nodes, edges)
    fig_disruption(nodes, edges, disr)
    print("Done. Wrote network/, disruption/, figures/.")


if __name__ == "__main__":
    main()
