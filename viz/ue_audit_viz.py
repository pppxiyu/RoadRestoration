"""
Figures for the UE solver tolerance audit: what loosening the convergence tolerance and
warm-starting each slot's solve do to the numbers the study actually reads, and whether that
error is acceptable.

The data is outputs/01-sim_val_n_problem_setting/raw/tolerance_audit.csv, written by `python -m util.ue_audit`
(which needs AequilibraE as the independent referee and several minutes of reference solves).
These figures are a pure function of that csv, so they can be refreshed on their own:

    python -m viz.ue_audit_viz

Two figures into outputs/01-sim_val_n_problem_setting/02-tolerance_audit/:

  01_objective_level_error   the UE-solve error in the objective's per-slot term g itself:
                             per-slot percentage deviation from the truth, the loosest cold
                             solve vs. the production setting. Shows the error is a small,
                             SMOOTH offset under the production setting (warm starts), while
                             the cold solve wobbles slot to slot.
  02_objective_change_error  the error in the slot-to-slot CHANGE of g -- the signal method
                             comparisons and the RL reward actually read -- relative to the
                             true change. Top: each configuration's median (%) against solve
                             time per slot. Bottom: the full distribution behind the medians,
                             one dot per adjacent-slot pair on a log percent axis; above the
                             100% line the error exceeds the change itself and the measured
                             direction of a repair's effect can flip -- the dark crosses are
                             the pairs where it DID flip.

Aggregation reuses util.ue_audit's own functions (_stats_row, _changes), so a figure can never
disagree with the generated write-up in tolerance_audit.md.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

import config as P
from util.ue_audit import (AE_PROD, MEANINGFUL, OUT_DOC, OUT_RAW, REF_RGAP,
                           _changes, _stats_row)
from viz.style import C, save_pub, use_pub

# One fixed visual identity per configuration family, shared by all three figures:
# the retired AequilibraE production setting is the grey anchor, cold in-house solves are the
# light blue family, warm in-house solves the dark blue family, and the production setting
# (config.py's rgap + warm start) is the red signal color everywhere it appears.
COL_ANCHOR, COL_COLD, COL_WARM = "#8A8A8A", "#9DB8D9", "#0F4D92"
# The level/change-error figures encode the START MODE alone in color, so they need two
# unmistakably different hues rather than the light/dark blue pair: cold blue, warm red.
COL_SA_COLD, COL_SA_WARM = "#3775BA", "#B64342"


def _configs(d):
    """The audited configurations in display order: anchor first, then in-house by tolerance
    (loose to tight), cold before warm. Each entry carries the row subset, an English label,
    its family color, and whether it is the current production setting."""
    out = [dict(key="anchor", s=d[d.solver == "AequilibraE"], warm=False,
                label=f"retired @{AE_PROD[0]:g} cold",
                color=COL_ANCHOR, current=False)]
    for rg in sorted(d[d.solver == "in-house"].rgap.unique(), reverse=True):
        for mode in ("cold", "warm"):
            s = d[(d.solver == "in-house") & (d.rgap == rg) & (d["mode"] == mode)]
            if not len(s):
                continue
            cur = (rg == P.UE_RGAP) and ((mode == "warm") == bool(P.UE_WARM_START))
            out.append(dict(key=f"{rg:g}-{mode}", s=s, warm=(mode == "warm"),
                            label=f"@{rg:g} {mode}" + ("\n(production)" if cur else ""),
                            color=(C["signal"] if cur else
                                   COL_WARM if mode == "warm" else COL_COLD),
                            current=cur))
    return out


# ----------------------------------------------------------------------------------------- #
# 01  the change error of g against solve time
# ----------------------------------------------------------------------------------------- #
_RC_BIG = {"font.size": 17, "axes.titlesize": 22, "axes.labelsize": 20,
           "xtick.labelsize": 17, "ytick.labelsize": 17, "legend.fontsize": 17}


def _objective_change_error(cfgs, base_ms, path):
    """Two stacked panels, one figure: the error in the slot-to-slot CHANGE of g relative to
    the true change. Top -- each configuration's MEDIAN (in percent) against its solve time
    per slot, linear axes. Bottom -- the full distribution behind those medians: every
    adjacent-slot pair one dot, log percent axis so the sub-100% bulk stays readable while
    the extreme cold-solve outliers remain in frame; the dashed line at 100% is the hard
    threshold above which the measured direction of a repair's effect can flip, and the dark
    crosses are the pairs where it DID flip. Color encodes the start mode everywhere (cold
    blue, warm red); the retired solver is the square marker in the top panel and the first
    column in the bottom one."""
    rows = [dict(c, **_stats_row(c["label"], c["s"], c["warm"], base_ms)) for c in cfgs]
    plt.rcParams.update(_RC_BIG)      # figure-local: render_audit_figures re-applies use_pub
    fig, (axis, ax2) = plt.subplots(2, 1, figsize=(13.0, 12.4), height_ratios=[1.0, 1.15])

    # ---- top: median change error (%) vs solve time ------------------------------------- #
    for r in rows:
        col = COL_SA_WARM if r["warm"] else COL_SA_COLD
        axis.scatter(r["ms"], 100.0 * r["ratio"], s=260, color=col, zorder=3,
                     marker="s" if r["key"] == "anchor" else "o",
                     edgecolor="white", lw=1.2)
        tol = ("@0.01" if r["key"] == "anchor" else
               "@" + r["key"].rsplit("-", 1)[0])
        if r["current"]:
            tol += " (production)"
        # The production label hangs BELOW its point so the bold text cannot collide with
        # the cold @0.001 label sitting just up-right of it.
        axis.annotate(tol, (r["ms"], 100.0 * r["ratio"]), textcoords="offset points",
                      xytext=(14, -30) if r["current"] else (14, 10), ha="left",
                      fontsize=16, fontweight="bold" if r["current"] else "normal",
                      color=C["neutral_dark"])
    axis.set_ylabel("median change error /\nmedian true change (%)")
    axis.set_xlabel("median solve time per slot (ms)")
    # Headroom above for the point labels, and a nudge below zero so the points sitting
    # at y = 0 are drawn whole instead of halved by the axis line.
    top = axis.get_ylim()[1] * 1.14
    axis.set_ylim(-0.045 * top, top)
    axis.set_xlim(-22, max(r["ms"] for r in rows) * 1.14)
    axis.legend(handles=[
        Line2D([], [], color=COL_SA_COLD, marker="o", ls="", ms=15, label="cold start"),
        Line2D([], [], color=COL_SA_WARM, marker="o", ls="", ms=15, label="warm start"),
        Line2D([], [], color=COL_SA_COLD, marker="s", ls="", ms=15,
               label="retired production solver")],
        fontsize=16, loc="upper right")

    # ---- bottom: the distribution behind the medians, log percent ----------------------- #
    rng = np.random.RandomState(0)          # jitter only; fixed so the figure is reproducible
    for i, c in enumerate(cfgs):
        dv, dr = _changes(c["s"], c["warm"])
        keep = np.abs(dr) > MEANINGFUL
        pp = 100.0 * np.abs(dv - dr)[keep] / np.abs(dr[keep])
        flip = (np.sign(dv) != np.sign(dr))[keep]
        x = i + rng.uniform(-0.16, 0.16, len(pp))
        col = COL_SA_WARM if c["warm"] else COL_SA_COLD
        ax2.scatter(x[~flip], pp[~flip], s=42, color=col, alpha=0.6, lw=0, zorder=2)
        if flip.any():          # duplicate labels are collapsed when the legend is built
            ax2.scatter(x[flip], pp[flip], s=90, color=C["neutral_dark"], marker="X",
                        zorder=3, label="direction flipped")
        # The distribution's median as a horizontal tick, the summary the top panel plots.
        ax2.hlines(np.median(pp), i - 0.26, i + 0.26, color=C["neutral_dark"], lw=2.8,
                   zorder=4)
    ax2.axhline(100.0, color=C["neutral_dark"], lw=2.0, ls="--", zorder=1,
                label="error = 100% of the true change (direction can flip above)")
    ax2.set_yscale("log")
    ax2.set_xticks(range(len(cfgs)))
    ax2.set_xticklabels([c["label"] for c in cfgs], fontsize=15)
    ax2.set_ylabel("change error /\ntrue change (%)")
    handles, labels = ax2.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    ax2.legend(uniq.values(), uniq.keys(), fontsize=15, loc="lower right")
    fig.tight_layout()
    save_pub(fig, path)
    plt.close(fig)
    use_pub(slide=True)               # undo the figure-local rc bump


# ----------------------------------------------------------------------------------------- #
# 00  the per-slot level error of g
# ----------------------------------------------------------------------------------------- #
def _objective_level_error(cfgs, d, path):
    """ONE panel: the per-slot UE-solve error in the objective's per-slot term g, as a
    percentage of the truth, for the loosest cold solve (the failure mode) and the production
    setting. The retired anchor is deliberately NOT drawn -- a third overlapping line adds
    nothing here, and its error is on record in the other two figures. What the panel shows is
    the error's STRUCTURE: the cold solve wobbles from slot to slot (that wobble is what leaks
    into the slot-to-slot change of g), while the warm-started production setting sits on a
    smooth, near-constant offset that cancels in differences."""
    chosen = [c for k in ("0.01-cold",) for c in cfgs if c["key"] == k] + \
             [c for c in cfgs if c["current"]]
    styles = {"0.01-cold": dict(color=COL_SA_COLD, ls="--"),
              chosen[-1]["key"]: dict(color=COL_SA_WARM, ls="-")}
    m = int(sorted(d.scenario.unique())[0])
    ref = d[(d.scenario == m) & (d.solver == "AequilibraE")].sort_values("slot")
    plt.rcParams.update(_RC_BIG)      # figure-local: render_audit_figures re-applies use_pub
    fig, axis = plt.subplots(figsize=(13.0, 6.4))

    axis.axhline(0.0, color=C["neutral_dark"], lw=1.6, zorder=1)
    for c in chosen:
        s = c["s"]
        s = s[s.scenario == m].sort_values("slot")
        rel = 100.0 * (s.g_val.to_numpy() - ref.g_ref.to_numpy()) / ref.g_ref.to_numpy()
        axis.plot(s.slot, rel, lw=2.4, zorder=3, marker="o", ms=7,
                  label=c["label"].replace("\n", " "), **styles[c["key"]])
    axis.set_xlabel("time slot")
    axis.set_ylabel("UE-solve error in the\nobjective term $g_t$  (%)")
    axis.legend(fontsize=16, loc="lower left")
    fig.tight_layout()
    save_pub(fig, path)
    plt.close(fig)
    use_pub(slide=True)               # undo the figure-local rc bump


# ----------------------------------------------------------------------------------------- #
def render_audit_figures():
    """Render all three audit figures from raw/tolerance_audit.csv into 02-tolerance_audit/."""
    src = OUT_RAW / "tolerance_audit.csv"
    if not src.exists():
        raise FileNotFoundError(
            f"{src} not found -- run `python -m util.ue_audit` (needs AequilibraE) to produce "
            "the audit data first; the figures are a pure function of that csv.")
    use_pub(slide=True)
    d = pd.read_csv(src)
    cfgs = _configs(d)
    base_ms = float(d[d.solver == "AequilibraE"].ms.median())
    OUT_DOC.mkdir(parents=True, exist_ok=True)
    _objective_level_error(cfgs, d, OUT_DOC / "01_objective_level_error")
    _objective_change_error(cfgs, base_ms, OUT_DOC / "02_objective_change_error")
    print(f"audit figures written to {OUT_DOC}")
    return OUT_DOC


if __name__ == "__main__":
    render_audit_figures()
