"""
Visual validation that the user-equilibrium (UE) traffic-assignment engine reproduces a
trusted reference solution on the Sioux Falls benchmark road network.

User equilibrium (UE) is the traffic state in which no driver can lower their own travel
time by unilaterally switching route. It is obtained by minimizing the convex Beckmann
objective (the sum over links of the integral of each link's travel-time function). The
reference is the widely used open-source Sioux Falls flow pattern distributed in the
Transportation Networks test-problem format (SiouxFalls_flow.tntp), taken here as ground
truth: if the engine is correct, its equilibrium flows and travel times must match it.

`save_validation(...)` renders the comparison into outputs/01-sim_val_n_problem_setting/, split by role:
  01-benchmark/01_ue_vs_reference.png   ONE four-panel figure: parity scatters of link flow
                                       and congested travel time (top), and the network
                                       colored by UE flow and by the absolute flow error
                                       against the reference (bottom).
  raw/ue_vs_ref_links.csv              per directed link: UE vs. reference flow and travel
                                       time, plus their absolute errors -- the data behind
                                       the figure.
  raw/summary.txt                      the headline agreement numbers (Beckmann objectives
                                       included).

The figure is drawn at the shared slide scale (use_pub(slide=True)): every figure under
outputs/01-sim_val_n_problem_setting/ is primarily consumed in slides, so the whole folder uses one type scale.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from viz.style import C, CMAP_ERR, CMAP_SEQ, save_pub, use_pub

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


def _parity(axis, x, y, xlab, ylab, title, unit_note):
    """One parity panel: modeled values against reference values. Points on the dashed
    y = x diagonal mean exact agreement, so tight clustering on it is the pass criterion."""
    vmax = max(np.max(x), np.max(y)) * 1.05
    axis.plot([0, vmax], [0, vmax], ls="--", lw=1.4, color=C["signal"], zorder=1)
    axis.scatter(x, y, s=42, alpha=0.8, color=C["accent"], edgecolor="white", lw=0.4, zorder=2)
    axis.set_xlim(0, vmax)
    axis.set_ylim(0, vmax)
    axis.set_xlabel(xlab)
    axis.set_ylabel(ylab)
    axis.set_aspect("equal")
    axis.set_title(title)
    axis.text(0.05, 0.95, unit_note, transform=axis.transAxes, va="top", fontsize=16,
              bbox=_BOX)


# The single validation figure: two parity scatters on top (flow, congested travel time),
# the network below colored by UE flow and by the absolute flow error vs. the reference.
_RC_BIG = {"font.size": 17, "axes.titlesize": 22, "axes.labelsize": 20,
           "xtick.labelsize": 17, "ytick.labelsize": 17, "legend.fontsize": 17}


def _validation_figure(cmp, m, nodes, edges, path):
    G, pos = _graph_pos(nodes, edges)
    # Look-up table from each directed link (from, to) to its (UE, reference) flow pair.
    d = {(int(r["from"]), int(r["to"])): (r["volume_ue"], r["volume_ref"])
         for _, r in cmp.iterrows()}
    # The maps use undirected edges, so fold the two travel directions together: total flow is
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

    plt.rcParams.update(_RC_BIG)      # figure-local: save_validation re-applies use_pub after
    fig, ax = plt.subplots(2, 2, figsize=(14.2, 13.0), height_ratios=[1.0, 1.15])

    # Panel a -- link flow parity, in thousands of vehicles.
    cost_err = (cmp["cost_ue"] - cmp["cost_ref"]).abs()
    _parity(ax[0, 0], cmp["volume_ref"] / 1e3, cmp["volume_ue"] / 1e3,
            r"reference flow ($\times10^3$ veh)", r"UE flow ($\times10^3$ veh)",
            "Link flow: UE solver vs. reference",
            f"n = {len(cmp)} links\ncorr = {m['corr']:.6f}\n"
            f"max |err| = {m['max_abs_err']:.2f} veh")

    # Panel b -- congested link travel time parity (link time under the equilibrium flow).
    _parity(ax[0, 1], cmp["cost_ref"], cmp["cost_ue"],
            "reference congested time", "UE congested time",
            "Congested travel time: UE solver vs. reference",
            f"max |err| = {cost_err.max():.1e}")

    # Panel c -- edges colored and sized by total two-way UE flow.
    c_ax = ax[1, 0]
    widths = 1.6 + 8.0 * ftot / ftot.max()
    ec = nx.draw_networkx_edges(G, pos, ax=c_ax, edgelist=elist, width=widths,
                                edge_color=ftot, edge_cmap=plt.get_cmap(CMAP_SEQ))
    nx.draw_networkx_nodes(G, pos, ax=c_ax, node_size=330, node_color="#eeeeee",
                           edgecolors=C["neutral_mid"], linewidths=0.8)
    nx.draw_networkx_labels(G, pos, ax=c_ax, font_size=12)
    c_ax.set_title("UE link flow on the network (two-way)")
    c_ax.set_aspect("equal")
    c_ax.axis("off")
    cb = fig.colorbar(ec, ax=c_ax, fraction=0.046, pad=0.02)
    cb.set_label("flow (veh)", fontsize=19)
    cb.ax.tick_params(labelsize=16)
    cb.outline.set_linewidth(0.8)

    # Panel d -- edges colored by absolute flow error on a fixed 0..max scale. The upper
    # bound is floored at 1 so a near-perfect match (tiny errors) does not stretch the color
    # ramp and exaggerate rounding-level noise.
    d_ax = ax[1, 1]
    ec2 = nx.draw_networkx_edges(G, pos, ax=d_ax, edgelist=elist, width=3.6,
                                 edge_color=eabs, edge_cmap=plt.get_cmap(CMAP_ERR),
                                 edge_vmin=0.0, edge_vmax=max(eabs.max(), 1.0))
    nx.draw_networkx_nodes(G, pos, ax=d_ax, node_size=330, node_color="#eeeeee",
                           edgecolors=C["neutral_mid"], linewidths=0.8)
    nx.draw_networkx_labels(G, pos, ax=d_ax, font_size=12)
    d_ax.set_title(f"Absolute flow error vs. reference (max {eabs.max():.2f} veh)")
    d_ax.set_aspect("equal")
    d_ax.axis("off")
    cb2 = fig.colorbar(ec2, ax=d_ax, fraction=0.046, pad=0.02)
    cb2.set_label("abs flow error (veh)", fontsize=19)
    cb2.ax.tick_params(labelsize=16)
    cb2.outline.set_linewidth(0.8)

    fig.tight_layout()
    save_pub(fig, path)
    plt.close(fig)
    use_pub(slide=True)               # undo the figure-local rc bump


# Public entry point: given the precomputed UE-vs-reference comparison table `cmp` and the
# `metrics` summary, write the per-link CSV, the validation figure, and the text summary into
# out_dir. `toy_dir` supplies the network geometry used to lay out the flow maps.
def save_validation(out_dir, toy_dir, cmp, report, metrics):
    use_pub(slide=True)  # every 01-sim_val_n_problem_setting figure is slide-first; one shared type scale
    out = Path(out_dir)
    raw = out / "raw"                 # intermediate data: everything that is not a figure
    figs = out / "01-benchmark"        # the ground-truth comparison figure
    raw.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)
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
    tbl.to_csv(raw / "ue_vs_ref_links.csv", index=False)

    _validation_figure(cmp, metrics, nodes, edges, figs / "01_ue_vs_reference")

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
    (raw / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out
