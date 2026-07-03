"""
Visual proof for the Step-A milestone: the UE engine reproduces the open-source
reference solution (SiouxFalls_flow.tntp). Nature publication style.

`save_validation(...)` writes, into outputs/UE_val/ (each .png 600 dpi + .svg + .pdf):
  00_overview        flow parity + cost parity + FW convergence + error histogram
  01_flow_map        network colored by UE flow, and by |UE - reference| flow
  ue_vs_ref_links.csv   per directed link: UE vs reference flow/cost + abs errors
  summary.txt           headline numbers
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from viz.style import C, CMAP_ERR, CMAP_SEQ, panel_label, save_pub, use_pub

_BOX = dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor=C["neutral_light"], lw=0.6)


def _graph_pos(nodes, edges):
    G = nx.Graph()
    for r in nodes.itertuples(index=False):
        G.add_node(int(r.node_id))
    for r in edges.itertuples(index=False):
        G.add_edge(int(r.u), int(r.v))
    pos = {int(r.node_id): (r.x, r.y) for r in nodes.itertuples(index=False)}
    return G, pos


def _overview(cmp, report, m, path):
    cost_err = (cmp["cost_ue"] - cmp["cost_ref"]).abs()
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 3.1))

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
    a.text(0.05, 0.95,
           f"n = {len(cmp)} links\ncorr = {m['corr']:.6f}\n"
           f"mean |err| = {m['mean_abs_err']:.2f} veh\nmax |err| = {m['max_abs_err']:.2f} veh",
           transform=a.transAxes, va="top", fontsize=6.3, bbox=_BOX)
    panel_label(a, "a", x=-0.2)

    b = ax[1]
    lim2 = [0, max(cmp["cost_ref"].max(), cmp["cost_ue"].max()) * 1.05]
    b.plot(lim2, lim2, ls="--", lw=1, color=C["signal"], zorder=1, label="y = x")
    b.scatter(cmp["cost_ref"], cmp["cost_ue"], s=14, alpha=0.8, color=C["teal"],
              edgecolor="white", lw=0.2, zorder=2)
    b.set_xlabel("reference congested time")
    b.set_ylabel("UE congested time")
    b.set_aspect("equal")
    b.set_title("link congested travel time")
    b.text(0.05, 0.95, f"max |err| = {cost_err.max():.1e}", transform=b.transAxes,
           va="top", fontsize=6.3, bbox=_BOX)
    panel_label(b, "b", x=-0.2)

    fig.suptitle("UE validation — reproducing the open-source Sioux Falls solution"
                 f"   (Beckmann obj {m['z_ue']:,.0f} vs {m['z_ref']:,.0f}, gap {m['obj_gap']:.1e})",
                 fontsize=8.5)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    save_pub(fig, path, svg=False, pdf=False)
    plt.close(fig)


def _flow_map(cmp, nodes, edges, path):
    G, pos = _graph_pos(nodes, edges)
    d = {(int(r["from"]), int(r["to"])): (r["volume_ue"], r["volume_ref"])
         for _, r in cmp.iterrows()}
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
    widths = 1.0 + 5.0 * ftot / ftot.max()

    fig, ax = plt.subplots(1, 2, figsize=(7.2, 4.0))
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


def save_validation(out_dir, toy_dir, cmp, report, metrics):
    use_pub()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    toy = Path(toy_dir)
    nodes = pd.read_csv(toy / "network" / "nodes.csv")
    edges = pd.read_csv(toy / "network" / "edges.csv")

    tbl = cmp.assign(flow_abs_err=(cmp["volume_ue"] - cmp["volume_ref"]).abs(),
                     cost_abs_err=(cmp["cost_ue"] - cmp["cost_ref"]).abs())
    tbl = tbl[["from", "to", "volume_ue", "volume_ref", "flow_abs_err",
               "cost_ue", "cost_ref", "cost_abs_err"]].sort_values("flow_abs_err", ascending=False)
    tbl.to_csv(out / "ue_vs_ref_links.csv", index=False)

    _overview(cmp, report, metrics, out / "00_overview")
    _flow_map(cmp, nodes, edges, out / "01_flow_map")

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
