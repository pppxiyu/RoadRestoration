"""Search-process diagnostics for the population metaheuristics (GA, PSO).

One figure, two panels, answering the two questions a search trace exists to settle: is the
best solution so far still improving, and is the population still producing anything the search
has not
already seen. Those are also exactly the two signals the plateau stopping rule watches, so the
figure shows why a run ended where it did rather than only that it did. Styling follows
viz/style.py, and the panels use the same conventions as the RL and MILP process figures.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from viz.style import C, panel_label, save_pub, use_pub


def make_search_process(out_dir, trace, prefix="ga"):
    """Draw the search process from the per-generation trace written by util/metaheuristic.py.

    Panel a is the objective in the NOMINAL world the search optimizes against: the population's
    spread from best to worst as a shaded band, its mean as a thin line, and the best-so-far
    best-so-far as the bold line. The band narrowing onto the bold line is the population
    collapsing onto one order, which is what ends a run together with a flat best-so-far curve.

    Panel b is the cumulative count of unique evaluations. Its slope IS the novelty the stopping
    rule counts: while the search keeps proposing orders it has not scored before the line climbs,
    and once the line goes flat the stall counter starts running. Reading the two panels together
    separates "stopped because nothing new was being tried" from "stopped because nothing new
    helped".

    Generation 0 is the seeded initial population, so panel a starts at the best of the static
    greedy orders rather than at a random level."""
    if trace.empty:
        return
    use_pub()
    figs = Path(out_dir)
    figs.mkdir(parents=True, exist_ok=True)
    g = trace.sort_values("generation")
    x = g["generation"].to_numpy()

    fig, ax = plt.subplots(2, 1, figsize=(5.0, 4.4), sharex=True,
                           gridspec_kw={"height_ratios": [2, 1]})

    ax[0].fill_between(x, g["gen_best_F"], g["gen_worst_F"], color=C["neutral_light"],
                       alpha=0.55, lw=0, zorder=1)
    ax[0].plot(x, g["gen_mean_F"], lw=0.8, color=C["neutral_mid"], zorder=2)
    ax[0].plot(x, g["best_F"], lw=1.6, color=C["signal"], zorder=3, drawstyle="steps-post")
    ax[0].axhline(1.0, ls=":", lw=0.7, color="0.8", zorder=0)          # pre-disaster level
    ax[0].set_ylabel("objective  $F$  (nominal world)")
    ax[0].text(0.98, 0.95, "band: population best-worst\nthin: population mean\nbold: best-so-far",
               transform=ax[0].transAxes, fontsize=5.5, color=C["neutral_mid"],
               ha="right", va="top", linespacing=1.5)

    ax[1].plot(x, g["cum_evals"], lw=1.3, color=C["accent"])
    ax[1].set_ylabel("unique evaluations\n(cumulative)")
    ax[1].set_xlabel("generation")

    panel_label(ax[0], "a")
    panel_label(ax[1], "b")
    fig.tight_layout()
    save_pub(fig, figs / f"{prefix}_search_process")
    plt.close(fig)
