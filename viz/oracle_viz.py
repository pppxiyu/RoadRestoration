"""Objective-focused figures for the brute-force oracle of the road-recovery toy problem.

The "oracle" exhaustively evaluates every feasible repair schedule and keeps the true
optimum, so it acts as the ground-truth benchmark against which any faster solver is
judged. These figures summarize what that exhaustive search reveals about the objective.

Each candidate schedule is scored by a combined objective F = mu*F1 + (1-mu)*F2, where
  F1 = accessibility degradation: the demand-weighted ratio of realized to baseline
       travel time, averaged over the recovery horizon (higher = worse access), and
  F2 = recovery efficiency: the repair makespan divided by the total repair work
       (lower is better). The "makespan" is the slot at which the last repair finishes,
       so F2 rewards crews that complete the work sooner relative to how much there is.

Writes three figures to outputs/oracle/n{N}/figures/ (PNG at 600 dpi):
  01_F_landscape    - F for every schedule sorted best->worst, showing how far the worst
                      choice sits above the optimum and how that spread varies by scenario.
  02_F1_F2_tradeoff - every schedule placed in (F2, F1) space and colored by F, exposing
                      the two-objective structure the optimum has to balance.
  03_best_schedule  - the optimal schedule as a repair Gantt chart (repair intervals on a
                      timeline) plus the demand drop/recovery and accessibility recovery it
                      produces over time.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config as P
from util.evaluate import evaluate_schedule
from viz.style import C, CMAP_SEQ, panel_label, save_pub, severity_color, use_pub


def make_figures(out_dir, land, opt, ctx, segments, scenarios, T, disrupted, rep=0):
    # Render the three oracle figures from the exhaustive-search results. Inputs:
    #   out_dir   : base directory; figures are written to its "figures" subfolder.
    #   land      : the full landscape - one row per (schedule, scenario) with columns F, F1, F2.
    #   opt       : the optimal schedule per scenario (start slots plus its F, F1, F2).
    #   ctx       : static problem context (network, OD demand, baseline travel times).
    #   segments  : the disrupted edge ids being scheduled for repair.
    #   scenarios : list of per-scenario duration dicts; scenarios[m][edge_id] = duration in slots.
    #   T         : recovery horizon in slots.
    #   disrupted : DataFrame of the disrupted edges, carrying each edge's damage severity.
    #   rep       : the representative scenario used for the single-scenario figures (02 and 03).
    use_pub()
    figs = Path(out_dir) / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    # Severity (1..3, higher = worse damage) per disrupted edge, used later to color repair bars.
    sev = {int(r.edge_id): int(r.severity) for r in disrupted.itertuples(index=False)}
    # Restrict to the representative scenario for the per-schedule scatter and best-schedule panels.
    drep = land[land["scenario"] == rep].reset_index(drop=True)

    # ---------------- 01 F landscape: how much schedule choice matters ----------------
    # For each scenario, sort every schedule's F ascending so column j is the j-th best
    # schedule; stacking the scenarios gives a rank-vs-F curve per scenario.
    scen = sorted(land["scenario"].unique())
    sorted_F = np.vstack([np.sort(land.loc[land["scenario"] == m, "F"].to_numpy()) for m in scen])
    n_sched = sorted_F.shape[1]
    ranks = np.arange(1, n_sched + 1)
    # Per-rank envelope and median across scenarios; f_star is the oracle optimum, i.e. the
    # mean best-schedule F, and spread is the best->worst gap on the median curve, as a percent.
    fmin, fmax = sorted_F.min(0), sorted_F.max(0)
    fmed = np.median(sorted_F, 0)
    f_star = float(sorted_F[:, 0].mean())
    spread = (fmed[-1] / fmed[0] - 1.0) * 100.0

    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    ax.fill_between(ranks, fmin, fmax, color=C["neutral_light"], alpha=0.6, lw=0,
                    label="across-scenario range")
    for row in sorted_F:
        ax.plot(ranks, row, color=C["neutral_mid"], lw=0.6, alpha=0.7, zorder=2)
    ax.annotate(f"oracle optimum  F* = {f_star:.3f}", xy=(1, f_star),
                xytext=(n_sched * 0.18, f_star + (fmax[-1] - fmin[0]) * 0.35),
                fontsize=6.3, va="center", ha="left",
                arrowprops=dict(arrowstyle="->", lw=0.7, color=C["neutral_dark"]))
    ax.text(0.035, 0.95, f"best $\\rightarrow$ worst: +{spread:.0f}% in F",
            transform=ax.transAxes, fontsize=6.5, va="top", color=C["neutral_dark"])
    ax.set_xlabel("schedule rank (best $\\rightarrow$ worst)")
    ax.set_ylabel(r"objective  $F=\mu F_1+(1-\mu)F_2$")
    ax.set_xlim(1, n_sched)
    ax.set_xticks([1] + list(range(5, n_sched + 1, 5)))
    ax.margins(y=0.08)
    ax.set_title(f"F across all {n_sched} feasible schedules ({len(scen)} scenarios)")
    ax.legend(loc="lower right", handlelength=1.5)
    fig.tight_layout()
    save_pub(fig, figs / "01_F_landscape")
    plt.close(fig)

    # ---------------- 02 F1-F2 trade-off: the two-objective structure ----------------
    # Place every schedule of the representative scenario in (recovery efficiency, access
    # degradation) space and color it by the combined objective F, so the shape of the
    # trade-off the optimum must balance is visible at a glance.
    fig, ax = plt.subplots(figsize=(3.6, 3.1))
    sc = ax.scatter(drep["F2"], drep["F1"], c=drep["F"], cmap=CMAP_SEQ, s=28,
                    edgecolor="white", lw=0.3, zorder=3)
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(rf"$F$ ($\mu={P.MU}$)")
    cb.outline.set_linewidth(0.6)
    ax.set_xlabel("$F_2$  (recovery efficiency: makespan / work)")
    ax.set_ylabel("$F_1$  (accessibility degradation)")
    ax.set_title(f"Two-objective structure of the {len(drep)} schedules")
    fig.tight_layout()
    save_pub(fig, figs / "02_F1_F2_tradeoff")
    plt.close(fig)

    # ---------------- 03 best schedule: what the optimum actually does ----------------
    # Recover the optimal schedule's start slots for the representative scenario, then replay
    # the objective evaluation with per-step traces so we can plot how demand and accessibility
    # evolve over the horizon under that schedule.
    orow = opt[opt["scenario"] == rep].iloc[0]
    dur = scenarios[rep]
    start = {e: int(orow[f"start_{e}"]) for e in segments}
    tr = evaluate_schedule(start, dur, T, ctx, collect_traces=True)["traces"]
    fig, ax = plt.subplots(3, 1, figsize=(4.0, 6.2))
    # Panel a: Gantt chart. One horizontal bar per edge spanning its repair window
    # [start, start+duration), colored by damage severity so worse damage stands out.
    for i, e in enumerate(segments):
        ax[0].barh(i, dur[e], left=start[e], height=0.62, color=severity_color(sev[e]),
                   edgecolor=C["neutral_dark"], lw=0.6)
        ax[0].text(start[e] + dur[e] / 2, i, f"e{e}", va="center", ha="center", fontsize=6.5)
    ax[0].set_yticks(range(len(segments)))
    ax[0].set_yticklabels([f"edge {e}" for e in segments])
    ax[0].set_xlabel("slot k  (3 h each)")
    ax[0].set_xlim(0, T)
    ax[0].set_title(f"optimal schedule  (F*={orow['F']:.3f}, $F_1$*={orow['F1']:.3f}, $F_2$*={orow['F2']:.3f})")
    panel_label(ax[0], "a")
    # Panel b: total origin-destination (OD) travel demand per slot. It collapses at onset,
    # then climbs back as repairs restore the network; the dashed line marks pre-disaster demand.
    ax[1].plot(tr["k"], tr["total_demand"] / 1000, marker="o", ms=3, color=C["accent"])
    ax[1].axhline(ctx["H0"].sum() / 1000, ls="--", lw=0.9, color=C["neutral_mid"], label="normal demand $H^{t_0}$")
    ax[1].set_xlabel("slot k")
    ax[1].set_ylabel("total OD demand (×10$^3$)")
    ax[1].set_xlim(1, T)
    ax[1].set_title("demand: sharp drop $\\rightarrow$ recovery")
    ax[1].legend(loc="lower right")
    panel_label(ax[1], "b")
    # Panel c: the per-slot F1 term (demand-weighted realized/baseline travel time). It starts
    # above 1 (degraded access) and relaxes toward 1 as the network is restored; averaging this
    # curve over the horizon yields F1.
    ax[2].plot(tr["k"], tr["f1_term"], marker="o", ms=3, color=C["teal"])
    ax[2].axhline(1.0, ls="--", lw=0.9, color=C["neutral_mid"], label="fully restored (=1)")
    ax[2].set_xlabel("slot k")
    ax[2].set_ylabel("per-step $F_1$ term")
    ax[2].set_xlim(1, T)
    ax[2].set_title("accessibility over time")
    ax[2].legend(loc="lower right")
    panel_label(ax[2], "c")
    fig.tight_layout()
    save_pub(fig, figs / "03_best_schedule")
    plt.close(fig)
