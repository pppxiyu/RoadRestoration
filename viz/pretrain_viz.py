"""
Comparison figure for the Section 2.1.1 traffic-fixation MILP vs the brute-force oracle.

`make_comparison(...)` writes outputs/pretrain_milp/figures/01_milp_vs_oracle.png (PNG only):
  panel a: per-scenario objective F -- oracle optimum F* vs the MILP schedule's TRUE F
  panel b: the gap F_milp - F*  (<=0 means the MILP found a schedule at least as good as the
           oracle's best work-conserving schedule; >0 is the traffic-fixation surrogate cost)
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from viz.style import C, panel_label, save_pub, use_pub


def make_comparison(out_dir, merged, segments, T):
    """merged: DataFrame with columns scenario, F_milp, F_oracle, gap (one row per scenario)."""
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

    fig.suptitle("Section 2.1.1 traffic-fixation MILP vs brute-force oracle "
                 f"({len(segments)} segments, horizon T={T})", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save_pub(fig, figs / "01_milp_vs_oracle")
    plt.close(fig)
    return figs / "01_milp_vs_oracle.png"


def make_landscape(out_dir, oracle_landscape, milp_opt, oracle_opt, segments, T):
    """Cross-scenario landscape (one-decision view): the fixed single-policy schedules ranked by
    their MEAN objective F over all scenarios (band = across-scenario range), with the MILP's
    mean and the per-scenario hindsight bound overlaid -- i.e. 'if you must commit to one policy,
    where does the field sit, and where does the adaptive MILP land'."""
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
    """Diagnostics of the alternating optimization, from the per-iteration trace:
      03_optimization_process.png -- true F and surrogate value vs iteration (one line/scenario,
                                      marker = the best-by-true-F iterate that is actually returned)
      04_runtime.png              -- per-scenario wall-clock and iteration count."""
    use_pub()
    figs = Path(out_dir) / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    scen_ids = sorted(trace_df["scenario"].unique())
    cmap = plt.get_cmap("viridis")
    colors = {m: cmap(i / max(1, len(scen_ids) - 1)) for i, m in enumerate(scen_ids)}

    # ---- 03: objective + surrogate trajectories over the alternating iterations ----
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.2))
    a = ax[0]
    for m in scen_ids:
        g = trace_df[trace_df["scenario"] == m].sort_values("iter")
        a.plot(g["iter"], g["F"].cummin(), "-", color=colors[m], lw=1.6, alpha=0.9)      # damped best-so-far (descends)
        bi = g["F"].idxmin()                                                             # returned best (min true F)
        a.plot(g.loc[bi, "iter"], g.loc[bi, "F"], "o", color=colors[m], ms=4, mec="white", mew=0.5, zorder=5)
    a.set_xlabel("alternating iteration")
    a.set_ylabel("true objective $F$")
    a.set_title("objective across iterations  (damped, best-so-far $\\downarrow$)")
    panel_label(a, "a")

    b = ax[1]
    for m in scen_ids:
        g = trace_df[(trace_df["scenario"] == m) & trace_df["surrogate"].notna()].sort_values("iter")
        b.plot(g["iter"], g["surrogate"], "-", color=colors[m], lw=1.0, alpha=0.85)
    b.set_xlabel("alternating iteration")
    b.set_ylabel(r"MILP surrogate $\sum_e c_e^{k}\,y_e^{k}$")
    b.set_title("surrogate across iterations")
    panel_label(b, "b")

    fig.suptitle(f"Alternating optimization trace "
                 f"({len(scen_ids)} scenarios, {len(segments)} segments, T={T})", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save_pub(fig, figs / "03_optimization_process")
    plt.close(fig)

    # ---- 04: runtime breakdown ----
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
