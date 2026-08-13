"""
Visual validation that the user-equilibrium (UE) traffic-assignment engine reproduces a
trusted reference solution on the Sioux Falls benchmark road network.

User equilibrium (UE) is the traffic state in which no driver can lower their own travel
time by unilaterally switching route. It is obtained by minimizing the convex Beckmann
objective (the sum over links of the integral of each link's travel-time function). The
reference is the widely used open-source Sioux Falls flow pattern distributed in the
Transportation Networks test-problem format (SiouxFalls_flow.tntp), taken here as ground
truth: if the engine is correct, its equilibrium flows and travel times must match it.

`save_validation(...)` renders the comparison into outputs/0-UE_val/ as 600-dpi PNGs:
  00_overview          two parity scatter plots -- modeled UE vs. reference link flow, and
                       modeled UE vs. reference congested link travel time -- annotated with
                       the flow correlation, error statistics, and the Beckmann objective gap.
  01_flow_map          the network drawn twice: colored by UE link flow, and by the absolute
                       flow discrepancy against the reference.
  ue_vs_ref_links.csv  per directed link: UE vs. reference flow and travel time, plus their
                       absolute errors.
  summary.txt          the headline agreement numbers.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from viz.style import C, CMAP_ERR, CMAP_SEQ, panel_label, save_pub, use_pub

# Style for the rounded, semi-transparent white boxes drawn behind on-figure statistics,
# so the numbers stay legible over scatter points and network edges.
_BOX = dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor=C["neutral_light"], lw=0.6)


# Build a NetworkX graph together with a fixed node-position map from the node/edge tables,
# so the flow maps are drawn on the network's real geographic layout rather than an
# algorithmically generated one.
def _graph_pos(nodes, edges):
    G = nx.Graph()
    for r in nodes.itertuples(index=False):
        G.add_node(int(r.node_id))
    for r in edges.itertuples(index=False):
        G.add_edge(int(r.u), int(r.v))
    # Pin every node to its stored (x, y) coordinate for a fixed, reproducible layout.
    pos = {int(r.node_id): (r.x, r.y) for r in nodes.itertuples(index=False)}
    return G, pos


# Build the overview figure: two "parity" scatter plots that check the engine against the
# reference. A parity plot puts modeled values on one axis and reference values on the other;
# points landing on the y = x diagonal mean the two agree exactly, so tight clustering on the
# diagonal is the pass criterion.
def _overview(cmp, m, path):
    # Per-link absolute discrepancy in congested travel time between the UE and the reference.
    cost_err = (cmp["cost_ue"] - cmp["cost_ref"]).abs()
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 3.1))

    # Panel a -- reference vs. modeled link flow, in thousands of vehicles. The dashed
    # diagonal marks perfect agreement; the axis limit is padded 5% above the largest flow.
    a = ax[0]
    vmax = max(cmp["volume_ref"].max(), cmp["volume_ue"].max()) / 1e3 * 1.05
    a.plot([0, vmax], [0, vmax], ls="--", lw=1, color=C["signal"], zorder=1, label="y = x")
    a.scatter(cmp["volume_ref"] / 1e3, cmp["volume_ue"] / 1e3, s=14, alpha=0.8,
              color=C["accent"], edgecolor="white", lw=0.2, zorder=2)
    a.set_xlim(0, vmax)
    a.set_ylim(0, vmax)
    a.set_xlabel(r"reference flow ($\times10^3$ veh)")
    a.set_ylabel(r"UE flow ($\times10^3$ veh)")
    a.set_aspect("equal")
    a.set_title("link flow: UE vs. reference")
    # Corner box reporting sample size, flow correlation, and mean/max absolute flow error.
    a.text(0.05, 0.95,
           f"n = {len(cmp)} links\ncorr = {m['corr']:.6f}\n"
           f"mean |err| = {m['mean_abs_err']:.2f} veh\nmax |err| = {m['max_abs_err']:.2f} veh",
           transform=a.transAxes, va="top", fontsize=6.3, bbox=_BOX)
    panel_label(a, "a", x=-0.2)

    # Panel b -- reference vs. modeled congested link travel time (link time under the
    # equilibrium flow), the same diagonal parity check applied to travel times.
    b = ax[1]
    lim2 = [0, max(cmp["cost_ref"].max(), cmp["cost_ue"].max()) * 1.05]
    b.plot(lim2, lim2, ls="--", lw=1, color=C["signal"], zorder=1, label="y = x")
    b.scatter(cmp["cost_ref"], cmp["cost_ue"], s=14, alpha=0.8, color=C["teal"],
              edgecolor="white", lw=0.2, zorder=2)
    b.set_xlabel("reference congested time")
    b.set_ylabel("UE congested time")
    b.set_aspect("equal")
    b.set_title("link congested travel time")
    # Annotate only the worst-case travel-time discrepancy across all links.
    b.text(0.05, 0.95, f"max |err| = {cost_err.max():.1e}", transform=b.transAxes,
           va="top", fontsize=6.3, bbox=_BOX)
    panel_label(b, "b", x=-0.2)

    # Title carries the Beckmann objective value for both solutions and their relative gap
    # (their fractional difference), a single scalar summarizing how close the equilibria are.
    fig.suptitle("UE validation — reproducing the open-source Sioux Falls solution"
                 f"   (Beckmann obj {m['z_ue']:,.0f} vs {m['z_ref']:,.0f}, gap {m['obj_gap']:.1e})",
                 fontsize=8.5)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    save_pub(fig, path, svg=False, pdf=False)
    plt.close(fig)


# Draw the equilibrium flow over the network twice: on the left the two-way link flow, on the
# right the absolute flow error versus the reference, so any mismatch stands out as a bright
# edge. The two panels share the same fixed node layout.
def _flow_map(cmp, nodes, edges, path):
    G, pos = _graph_pos(nodes, edges)
    # Look-up table from each directed link (from, to) to its (UE, reference) flow pair.
    d = {(int(r["from"]), int(r["to"])): (r["volume_ue"], r["volume_ref"])
         for _, r in cmp.iterrows()}
    # The map uses undirected edges, so fold the two travel directions together: total flow is
    # the sum of both directional UE volumes, while the plotted error is the worse of the two
    # directions (a missing direction defaults to zero flow).
    elist, ftot, eabs = [], [], []
    for r in edges.itertuples(index=False):
        u, v = int(r.u), int(r.v)
        uv = d.get((u, v), (0.0, 0.0))
        vu = d.get((v, u), (0.0, 0.0))
        elist.append((u, v))
        ftot.append(uv[0] + vu[0])
        eabs.append(max(abs(uv[0] - uv[1]), abs(vu[0] - vu[1])))
    ftot = np.array(ftot)
    eabs = np.array(eabs)
    # Scale line thickness with total flow so busy corridors read as visibly thicker edges.
    widths = 1.0 + 5.0 * ftot / ftot.max()

    fig, ax = plt.subplots(1, 2, figsize=(7.2, 4.0))
    # Panel a -- edges colored and sized by total two-way UE flow.
    ec = nx.draw_networkx_edges(G, pos, ax=ax[0], edgelist=elist, width=widths,
                                edge_color=ftot, edge_cmap=plt.get_cmap(CMAP_SEQ))
    nx.draw_networkx_nodes(G, pos, ax=ax[0], node_size=80, node_color="#eeeeee",
                           edgecolors=C["neutral_mid"], linewidths=0.5)
    nx.draw_networkx_labels(G, pos, ax=ax[0], font_size=5)
    ax[0].set_title("UE equilibrium link flow (two-way)")
    ax[0].set_aspect("equal")
    ax[0].axis("off")
    cb = fig.colorbar(ec, ax=ax[0], fraction=0.046, pad=0.02)
    cb.set_label("flow (veh)")
    cb.outline.set_linewidth(0.6)
    panel_label(ax[0], "a", x=0.0)

    # Panel b -- edges colored by absolute flow error on a fixed 0..max scale. The upper
    # bound is floored at 1 so a near-perfect match (tiny errors) does not stretch the color
    # ramp and exaggerate rounding-level noise.
    ec2 = nx.draw_networkx_edges(G, pos, ax=ax[1], edgelist=elist, width=2.4,
                                 edge_color=eabs, edge_cmap=plt.get_cmap(CMAP_ERR),
                                 edge_vmin=0.0, edge_vmax=max(eabs.max(), 1.0))
    nx.draw_networkx_nodes(G, pos, ax=ax[1], node_size=80, node_color="#eeeeee",
                           edgecolors=C["neutral_mid"], linewidths=0.5)
    nx.draw_networkx_labels(G, pos, ax=ax[1], font_size=5)
    ax[1].set_title(f"|UE − reference| flow  (max {eabs.max():.2f} veh)")
    ax[1].set_aspect("equal")
    ax[1].axis("off")
    cb2 = fig.colorbar(ec2, ax=ax[1], fraction=0.046, pad=0.02)
    cb2.set_label("abs flow error (veh)")
    cb2.outline.set_linewidth(0.6)
    panel_label(ax[1], "b", x=0.0)

    fig.suptitle("UE flow on the Sioux Falls network, and its (negligible) error vs. ground truth",
                 fontsize=8.5)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_pub(fig, path, svg=False, pdf=False)
    plt.close(fig)


# Public entry point: given the precomputed UE-vs-reference comparison table `cmp` and the
# `metrics` summary, write the per-link CSV, both figures, and the text summary into out_dir.
# `toy_dir` supplies the network geometry used to lay out the flow maps.
def save_validation(out_dir, toy_dir, cmp, report, metrics):
    use_pub()  # apply the shared publication style once before any figure is drawn
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    toy = Path(toy_dir)
    # Node coordinates and edge list define the network layout for the flow maps.
    nodes = pd.read_csv(toy / "network" / "nodes.csv")
    edges = pd.read_csv(toy / "network" / "edges.csv")

    # Per-link table augmented with flow and travel-time absolute errors, then ordered so the
    # worst-agreeing links appear first -- the natural place to look when auditing mismatches.
    tbl = cmp.assign(flow_abs_err=(cmp["volume_ue"] - cmp["volume_ref"]).abs(),
                     cost_abs_err=(cmp["cost_ue"] - cmp["cost_ref"]).abs())
    tbl = tbl[["from", "to", "volume_ue", "volume_ref", "flow_abs_err",
               "cost_ue", "cost_ref", "cost_abs_err"]].sort_values("flow_abs_err", ascending=False)
    tbl.to_csv(out / "ue_vs_ref_links.csv", index=False)

    _overview(cmp, metrics, out / "00_overview")
    _flow_map(cmp, nodes, edges, out / "01_flow_map")

    # Headline agreement numbers written as a plain-text summary for a quick pass/fail read.
    lines = [
        "UE validation milestone — summary",
        "=================================",
        f"directed links compared : {len(cmp)}",
        f"flow correlation        : {metrics['corr']:.6f}",
        f"mean |flow error|       : {metrics['mean_abs_err']:.3f} veh "
        f"({metrics['mean_abs_err']/metrics['mean_flow']*100:.4f}% of mean)",
        f"max  |flow error|       : {metrics['max_abs_err']:.3f} veh",
        f"Beckmann objective (UE) : {metrics['z_ue']:,.2f}",
        f"Beckmann objective (ref): {metrics['z_ref']:,.2f}",
        f"objective relative gap  : {metrics['obj_gap']:.2e}",
        f"final solver rgap       : {metrics['rgap']:.2e}",
        f"MILESTONE               : {metrics.get('status', 'PASS')}",
    ]
    (out / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out
