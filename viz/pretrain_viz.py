"""
Figures that benchmark the fast traffic-fixation solver against the exact brute-force oracle.

Context. The restoration problem asks in what order to repair disrupted road segments so as to
minimise an objective F (lower is better; F combines network travel time and unmet demand). Two
solvers produce a schedule:
  - Oracle: enumerates every work-conserving schedule (a schedule that never leaves a crew idle
    while repair work is waiting) and evaluates the EXACT F of each, so its per-scenario minimum
    F* is the ground-truth optimum.
  - MILP (mixed-integer linear program): a much cheaper approximation that fixes the network
    travel times from a previous solve so that F becomes linear in the start-time decisions,
    then optimises that linearised "surrogate" objective. Fixing the travel times is called
    traffic fixation.

`make_comparison(...)` writes outputs/pretrain_milp/figures/01_milp_vs_oracle.png:
  panel a: per-scenario objective F -- the oracle optimum F* against the TRUE F of the schedule
           the MILP actually returned.
  panel b: the gap F_milp - F*. A value <= 0 means the MILP found a schedule at least as good as
           the oracle's best work-conserving schedule; a value > 0 is the accuracy loss incurred
           by optimising the linear surrogate instead of the true objective.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from viz.style import C, panel_label, save_pub, use_pub


def make_comparison(out_dir, merged, segments, T):
    """Draw the two-panel MILP-vs-oracle comparison and save it as a PNG.

    `merged` is a DataFrame with one row per scenario and columns scenario, F_milp (true objective
    of the MILP's schedule), F_oracle (the exact per-scenario optimum F*), and gap (F_milp - F*).
    `segments` and `T` (the repair horizon, i.e. the number of time slots) are used only for the
    figure caption. Returns the path of the written PNG."""
    use_pub()
    figs = Path(out_dir) / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    m = merged.sort_values("scenario")
    x = m["scenario"].to_numpy()

    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.1))

    a = ax[0]
    a.plot(x, m["F_oracle"], "o-", color=C["accent"], ms=5, lw=1.2, zorder=3,
           label="oracle $F^*$ (brute force)")
    a.plot(x, m["F_milp"], "s--", color=C["signal"], ms=5, lw=1.2, zorder=2,
           label="MILP $F$ (traffic fixation)")
    a.set_xlabel("scenario")
    a.set_ylabel("objective $F$")
    a.set_title("MILP schedule vs oracle optimum")
    a.set_xticks(x)
    a.legend(fontsize=6.5, frameon=False, loc="best")
    panel_label(a, "a")

    b = ax[1]
    gap = m["gap"].to_numpy()
    colors = [C["good"] if g <= 1e-9 else C["signal"] for g in gap]
    b.bar(x, gap, color=colors, width=0.6, zorder=2)
    b.axhline(0, color=C["neutral_mid"], lw=0.8, zorder=1)
    b.set_xlabel("scenario")
    b.set_ylabel("$F_{\\mathrm{MILP}} - F^{*}$  (gap)")
    b.set_title(f"gap per scenario (mean {gap.mean():+.3f})")
    b.set_xticks(x)
    panel_label(b, "b")

    fig.suptitle("traffic-fixation MILP vs brute-force oracle "
                 f"({len(segments)} segments, horizon T={T})", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save_pub(fig, figs / "01_milp_vs_oracle")
    plt.close(fig)
    return figs / "01_milp_vs_oracle.png"


def make_landscape(out_dir, oracle_landscape, milp_opt, oracle_opt, segments, T):
    """Draw the cross-scenario landscape, which answers: if you had to commit to ONE fixed repair
    order for every scenario, how good would each choice be, and how does the scenario-adaptive
    MILP compare?

    Each candidate is a fixed single policy (one repair permutation applied unchanged across all
    scenarios). Policies are ranked by their MEAN objective F over the scenarios; for each, a band
    shows the across-scenario range (min to max F). Two reference lines are overlaid: the MILP's
    mean F (it may pick a different order per scenario, i.e. it is adaptive), and the per-scenario
    hindsight bound -- the mean of the oracle optima, the unbeatable score you would get if you
    could choose the best order separately for each scenario knowing its outcome in advance.

    `oracle_landscape` holds F for every (permutation, scenario) pair; `milp_opt` and `oracle_opt`
    supply the MILP and oracle per-scenario results. Returns the path of the written PNG."""
    use_pub()
    figs = Path(out_dir) / "figures"
    figs.mkdir(parents=True, exist_ok=True)

    piv = oracle_landscape.pivot_table(index="perm", columns="scenario", values="F")
    mean_F = piv.mean(axis=1).sort_values()
    order = mean_F.index
    sub = piv.loc[order]
    ranks = np.arange(1, len(order) + 1)
    lo = sub.min(axis=1).to_numpy()
    hi = sub.max(axis=1).to_numpy()
    meanc = mean_F.to_numpy()
    milp_mean = float(milp_opt["F_milp"].mean())
    hind_mean = float(oracle_opt["F"].mean())
    best_fixed = float(meanc[0])

    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    ax.fill_between(ranks, lo, hi, color=C["neutral_light"], alpha=0.55, lw=0,
                    label="across-scenario range")
    ax.plot(ranks, meanc, "o-", color=C["neutral_dark"], ms=5, lw=1.3,
            label="fixed policy (mean over scenarios)")
    ax.axhline(hind_mean, color=C["accent"], lw=1.1, ls=":",
               label=f"per-scenario hindsight = {hind_mean:.3f}")
    ax.axhline(milp_mean, color=C["signal"], lw=1.4, ls="--",
               label=f"MILP (adaptive) mean = {milp_mean:.3f}")
    ax.plot([ranks[0]], [milp_mean], "*", color=C["signal"], ms=16, mec="white",
            mew=0.6, zorder=6)
    ax.set_xlabel("fixed single policy, ranked best $\\rightarrow$ worst by mean $\\bar F$")
    ax.set_ylabel(r"objective $F$  (marker = mean, band = range)")
    ax.set_title(f"Cross-scenario landscape: MILP vs fixed policies "
                 f"({len(order)} schedules, {piv.shape[1]} scenarios)")
    ax.set_xticks(ranks)
    ax.legend(fontsize=6, frameon=False, loc="center left")
    ax.annotate(f"MILP $-$ hindsight = {milp_mean - hind_mean:+.4f}\n"
                f"MILP $-$ best fixed = {milp_mean - best_fixed:+.4f}",
                xy=(0.97, 0.06), xycoords="axes fraction", ha="right", va="bottom",
                fontsize=6.3,
                bbox=dict(boxstyle="round", fc="white", ec=C["neutral_light"], lw=0.6))
    fig.tight_layout()
    save_pub(fig, figs / "02_cross_scenario_landscape")
    plt.close(fig)
    return figs / "02_cross_scenario_landscape.png"


def make_process_figures(out_dir, trace_df, milp_opt, segments, T):
    """Draw diagnostics of the MILP's alternating optimisation from its per-iteration trace.

    The solver alternates two steps until the schedule stops changing: solve the linear surrogate
    for a start-time schedule, then recompute the true objective F. This function produces:
      03_optimization_process.png -- three panels, one line per scenario: (a) the returned
                                      best-so-far true F, with the actually-returned iterate marked;
                                      (b) the RAW per-iteration true F, which shows whether F
                                      settles (solid = reached a fixed point, dashed = stopped by
                                      the cycle guard and still oscillating); (c) the surrogate
                                      value the MILP maximizes.
      04_runtime.png              -- per-scenario wall-clock time and the number of iterations
                                      needed to converge.
    Returns the figures directory. `trace_df` carries the per-iteration records; `milp_opt` holds
    per-scenario timing and iteration counts."""
    use_pub()
    figs = Path(out_dir) / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    scen_ids = sorted(trace_df["scenario"].unique())
    cmap = plt.get_cmap("viridis")
    colors = {m: cmap(i / max(1, len(scen_ids) - 1)) for i, m in enumerate(scen_ids)}

    # Panel 03: the returned best-so-far F, whether the raw F actually settles, and the surrogate.
    fig, ax = plt.subplots(1, 3, figsize=(10.8, 3.2))
    a = ax[0]
    for m in scen_ids:
        g = trace_df[trace_df["scenario"] == m].sort_values("iter")
        a.plot(g["iter"], g["F"].cummin(), "-", color=colors[m], lw=1.6, alpha=0.9)      # plot the running best-so-far F, so the curve only descends
        bi = g["F"].idxmin()                                                             # the iterate with the lowest true F is the schedule the solver returns
        a.plot(g.loc[bi, "iter"], g.loc[bi, "F"], "o", color=colors[m], ms=4, mec="white", mew=0.5, zorder=5)
    a.set_xlabel("alternating iteration")
    a.set_ylabel("true objective $F$")
    a.set_title("returned value  (best-so-far $\\downarrow$)")
    panel_label(a, "a")

    # Panel b: the RAW per-iteration F (NOT best-so-far), which is what actually answers "did F
    # converge?". A scenario that reached a fixed point flattens; one stopped by the cycle guard
    # keeps oscillating -- a distinction the best-so-far curve in panel a hides by construction.
    b = ax[1]
    for m in scen_ids:
        g = trace_df[trace_df["scenario"] == m].sort_values("iter")
        conv = str(milp_opt.loc[milp_opt["scenario"] == m, "converged"].iloc[0]).strip().lower() == "true"
        b.plot(g["iter"], g["F"], "-" if conv else "--", color=colors[m], lw=1.1, alpha=0.85,
               marker="o", ms=2.4, mew=0)
    b.set_xlabel("alternating iteration")
    b.set_ylabel("true objective $F$  (raw, per iteration)")
    b.set_title("does $F$ settle?")
    b.legend(handles=[Line2D([0], [0], color=C["neutral_dark"], ls="-", lw=1.1, label="converged (fixed point)"),
                      Line2D([0], [0], color=C["neutral_dark"], ls="--", lw=1.1, label="cycle-stopped")],
             fontsize=5.6, loc="upper right", handlelength=1.8)
    panel_label(b, "b")

    c = ax[2]
    for m in scen_ids:
        g = trace_df[(trace_df["scenario"] == m) & trace_df["surrogate"].notna()].sort_values("iter")
        c.plot(g["iter"], g["surrogate"], "-", color=colors[m], lw=1.0, alpha=0.85)
    c.set_xlabel("alternating iteration")
    c.set_ylabel(r"MILP surrogate $\sum_e c_e^{k}\,y_e^{k}$")
    c.set_title("surrogate across iterations")
    panel_label(c, "c")

    fig.suptitle(f"Alternating optimization trace "
                 f"({len(scen_ids)} scenarios, {len(segments)} segments, T={T})", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save_pub(fig, figs / "03_optimization_process")
    plt.close(fig)

    # Panel 04: runtime breakdown -- solve time and iteration count per scenario.
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0))
    x = milp_opt["scenario"].to_numpy()
    a = ax[0]
    a.bar(x, milp_opt["time_s"], color=C["accent"], width=0.6)
    a.set_xlabel("scenario")
    a.set_ylabel("wall-clock (s)")
    a.set_title(f"per-scenario time (total {milp_opt['time_s'].sum() / 60:.1f} min)")
    a.set_xticks(x)
    panel_label(a, "a")
    b = ax[1]
    b.bar(x, milp_opt["n_iter"], color=C["teal"], width=0.6)
    b.set_xlabel("scenario")
    b.set_ylabel("alternating iterations")
    b.set_title(f"iterations to stop (mean {milp_opt['n_iter'].mean():.1f})")
    b.set_xticks(x)
    panel_label(b, "b")
    fig.suptitle("MILP runtime breakdown", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save_pub(fig, figs / "04_runtime")
    plt.close(fig)
    return figs
