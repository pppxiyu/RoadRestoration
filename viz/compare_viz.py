"""Comparison figures for the road-restoration solvers: the greedy baseline variants vs the
pretraining MILP vs (optional) the brute-force oracle. Each fact has one home: per-scenario final
objective F lives in make_final_performance_all, the MILP optimization trajectory in make_process,
accuracy against compute in make_accuracy_compute, and the gap to the true optimum (only where the
oracle exists) in make_gap_to_oracle. Styling follows viz/style.py so it matches the rest of the
project's figures."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from viz.style import C, CMAP_SEQ, save_pub, use_pub

# greedy variants share a darkening grey ramp (context); MILP is the emphasis blue (method under
# test); oracle is the signal red (the target the others are measured against).
_GREY_RAMP = ["#C6C6C6", "#9E9E9E", "#767676", "#4D4D4D"]


def _label(c):
    return c[len("greedy_"):] if c.startswith("greedy_") else c


def _method_colors(cols):
    """Static greedy rules share a darkening grey ramp (context); GA and PSO, being population
    metaheuristics rather than static rankers, get their own colors; the MILP is the emphasis blue
    (method under test); the oracle is the signal red (the target the others are measured against)."""
    out, gi = {}, 0
    for c in cols:
        v = _label(c)
        if v == "milp":
            out[c] = C["accent"]
        elif v == "oracle":
            out[c] = C["signal"]
        elif v == "ga":
            out[c] = C["signal"]
        elif v == "pso":
            out[c] = C["good"]
        else:                                              # a static greedy variant
            out[c] = _GREY_RAMP[gi % len(_GREY_RAMP)]
            gi += 1
    return out


def make_final_performance_all(out_dir, df, N, methods):
    """The single home of the per-scenario final-objective comparison: the final $F$ of every
    method (static greedy, MILP, GA, PSO) as grouped bars, one group per scenario, with the
    pre-disaster level $F=1$ marked. The process figure and the gap figure do not repeat it."""
    use_pub()
    figs = Path(out_dir) / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    scen = df["scenario"].to_numpy()
    x = np.arange(len(scen))
    col = _method_colors(methods)
    w = 0.85 / len(methods)

    fig, ax = plt.subplots(figsize=(8.6, 3.2))
    ax.axhline(1.0, ls=":", lw=0.7, color="0.75", zorder=1)             # pre-disaster level
    for i, c in enumerate(methods):
        ax.bar(x + (i - (len(methods) - 1) / 2) * w, df[c], width=w, label=_label(c), color=col[c])
    vals = df[methods].to_numpy()
    lo, hi = float(vals.min()), float(vals.max())
    ax.set_ylim(lo - 0.04 * (hi - lo), hi + 0.06 * (hi - lo))           # zoom: the F values sit near 1
    ax.set_xlabel("scenario")
    ax.set_ylabel("objective  $F$")
    ax.set_xticks(x)
    ax.set_xticklabels(scen)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.005), ncol=len(methods),
              fontsize=6, handlelength=1.1, columnspacing=1.2, frameon=False)
    fig.tight_layout()
    save_pub(fig, figs / "final_performance_all")
    plt.close(fig)


def make_gap_to_oracle(out_dir, df, N, methods):
    """Per-scenario gap to the brute-force oracle, $F - F^{*}$, for every method, as grouped bars.
    Only produced at scales where the oracle was enumerated, so it carries information that the
    per-scenario final-F figure cannot (the true optimum is unavailable at large scale)."""
    use_pub()
    figs = Path(out_dir) / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    scen = df["scenario"].to_numpy()
    x = np.arange(len(scen))
    col = _method_colors(methods)
    w = 0.85 / len(methods)

    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    for i, c in enumerate(methods):
        ax.bar(x + (i - (len(methods) - 1) / 2) * w, df[f"gap_{c}"], width=w,
               label=_label(c), color=col[c])
    ax.axhline(0.0, color=C["neutral_dark"], lw=0.8)
    ax.set_xlabel("scenario")
    ax.set_ylabel("gap to oracle  ($F - F^{*}$)")
    ax.set_xticks(x)
    ax.set_xticklabels(scen)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.005), ncol=len(methods),
              fontsize=6, handlelength=1.1, columnspacing=1.2, frameon=False)
    fig.tight_layout()
    save_pub(fig, figs / "gap_to_oracle")
    plt.close(fig)


def make_process(out_dir, N, milp_trace):
    """The MILP's optimization process: its best-so-far true $F$ over the alternating iterations,
    one curve per scenario, each labelled at its right end with the scenario number so the curves
    carry the same scenario numbering as make_final_performance_all. No baseline levels and no
    highlighted scenario are drawn, because comparing methods and comparing scenarios both belong to
    make_final_performance_all; this figure shows only how each scenario's MILP trajectory descends."""
    use_pub()
    figs = Path(out_dir) / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    scen_ids = sorted(milp_trace["scenario"].unique())
    cmap = plt.get_cmap(CMAP_SEQ)
    n = len(scen_ids)

    fig, a = plt.subplots(figsize=(4.8, 3.4))
    ends = []
    for i, m in enumerate(scen_ids):
        g = milp_trace[milp_trace["scenario"] == m].sort_values("iter")
        it = g["iter"].to_numpy()
        f = g["F"].cummin().to_numpy()
        color = cmap(0.05 + 0.75 * (i / max(1, n - 1)))                # keep off the palest end so numbers stay legible
        a.plot(it, f, lw=1.1, color=color, alpha=0.9, zorder=2)
        ends.append((float(it[-1]), float(f[-1]), color, int(m)))

    yspan = (max(e[1] for e in ends) - min(e[1] for e in ends)) or 1.0
    placed = []                                                        # nudge apart numbers whose curve-ends coincide
    for x_end, y_end, color, m in ends:
        k = sum(1 for px, py in placed if abs(px - x_end) <= 0.8 and abs(py - y_end) <= 0.03 * yspan)
        placed.append((x_end, y_end))
        a.annotate(str(m), (x_end, y_end), textcoords="offset points", xytext=(4, 2 + 9 * k),
                   fontsize=6, fontweight="bold", color=color, va="center", zorder=4)

    a.set_xlabel("MILP alternating iteration")
    a.set_ylabel("true objective  $F$  (best-so-far)")
    fig.tight_layout()
    save_pub(fig, figs / "optimization_process")
    plt.close(fig)


def make_accuracy_compute(out_dir, N, stats):
    """Figure 2: accuracy vs COMPUTE. One point per method at (serial-equivalent UE solves per
    scenario, mean final F). The compute axis counts UE solves rather than wall-clock, so it is
    neutral to how many worker processes a method was parallelized over and a method spread across
    many workers is not made to look cheaper than it is (see Caveats C1). Lower-left is better,
    meaning little compute together with high accuracy. `stats` is a list of dicts
    {method, mean_F, mean_ue, kind} where kind is 'greedy' (static rule, grey circle), 'meta'
    (budgeted population search, green square) or 'milp' (blue diamond)."""
    use_pub()
    figs = Path(out_dir) / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(3.6, 2.6))

    xs = [s["mean_ue"] for s in stats]
    ys = [s["mean_F"] for s in stats]
    xmax = max(xs)
    yspan = (max(ys) - min(ys)) or 1.0

    ax.axhline(1.0, ls=":", lw=0.7, color="0.75", zorder=1)     # pre-disaster level

    gi = 0
    placed = []                                                 # data coords of labels already placed
    for s in stats:
        if s["kind"] == "milp":
            color, marker = C["accent"], "D"
        elif s["kind"] == "meta":
            color, marker = C["good"], "s"
        else:
            color, marker = _GREY_RAMP[gi % len(_GREY_RAMP)], "o"
            gi += 1
        x, y = s["mean_ue"], s["mean_F"]
        ax.scatter(x, y, s=50, color=color, marker=marker, edgecolor="white", lw=0.5, zorder=3)
        # stack a label upward when its point sits on top of one already labelled (linear axis puts
        # the same-compute greedy rules, and the same-compute GA/PSO, at coincident x)
        k = sum(1 for px, py in placed if abs(px - x) <= 0.04 * xmax and abs(py - y) <= 0.05 * yspan)
        placed.append((x, y))
        left = x >= xmax - 1e-9                                 # rightmost points label leftward so text never clips
        ax.annotate(s["method"], (x, y), textcoords="offset points",
                    xytext=(-7 if left else 7, 3 + 11 * k), ha="right" if left else "left",
                    fontsize=6.5, color=C["neutral_dark"])

    ax.set_xlim(-0.04 * xmax, xmax * 1.12)
    ax.set_ylim(min(ys) - 0.06 * yspan, max(ys) + 0.16 * yspan)
    ax.set_xlabel("serial-equivalent UE solves / scenario")
    ax.set_ylabel("mean objective  $F$")
    fig.tight_layout()
    save_pub(fig, figs / "accuracy_vs_compute")
    plt.close(fig)
