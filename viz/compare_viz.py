"""Comparison figures for the road-restoration solvers: the greedy baseline variants, GA/PSO and
the RL DQN vs the pretraining MILP vs (optional) the brute-force oracle. Each fact has one home:
per-scenario final objective F lives in make_final_performance_all, its spread across scenarios in
make_performance_distribution, accuracy against compute in
make_accuracy_compute, and the gap to the true optimum (only where the oracle exists) in
make_gap_to_oracle. Every figure here compares METHODS; a single solver's own search trajectory
belongs with that solver instead -- the MILP's in viz/pretrain_viz.py, the metaheuristics' in
viz/meta_viz.py, the RL variants' in viz/rank_viz.py. Styling follows viz/style.py so it matches
the rest of the project's figures."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from viz.style import C, save_pub, use_pub

# The darkening grey ramp is reserved for the static greedy variants. The full per-method mapping
# (grey ramp / GA red / rl_nominal teal / rl_saa purple / rl_saa_per light blue / MILP dark blue /
# oracle red) lives in _method_colors below.
_GREY_RAMP = ["#C6C6C6", "#9E9E9E", "#767676", "#4D4D4D"]


def _label(c):
    return c[len("greedy_"):] if c.startswith("greedy_") else c


def _method_colors(cols):
    """Static greedy rules share a darkening grey ramp (context). GA, being a population
    metaheuristic rather than a static ranker, reuses the signal red, which cannot collide
    because the oracle is never drawn in the same figure as GA. The nominal RL DQN takes the
    teal accent, and the SAA variants take purple (rl_saa) and the lighter blue (rl_saa_per).
    The MILP is the emphasis blue (method under test), and the oracle keeps the signal red
    where it appears."""
    out, gi = {}, 0
    for c in cols:
        v = _label(c)
        if v == "milp":
            out[c] = C["accent"]
        elif v == "oracle":
            out[c] = C["signal"]
        elif v == "ga":
            out[c] = C["signal"]
        elif v == "rl_nominal":
            out[c] = C["teal"]
        elif v == "rl_saa":
            out[c] = C["purple"]
        elif v == "rl_saa_per":
            out[c] = C["accent2"]

        else:                                              # a static greedy variant
            out[c] = _GREY_RAMP[gi % len(_GREY_RAMP)]
            gi += 1
    return out


def make_final_performance_all(out_dir, df, methods):
    """The single home of the per-scenario final-objective comparison: the final $F$ of every
    method (static greedy, MILP, GA, PSO) as grouped bars, one group per scenario, with the
    pre-disaster level $F=1$ marked. The process figure and the gap figure do not repeat it."""
    use_pub()
    figs = Path(out_dir)
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


def make_performance_distribution(out_dir, df, methods):
    """The DISTRIBUTION of each method's per-scenario F, one box per method, ranked best-first.

    The bar chart beside this one shows every scenario individually and the summary table reports
    a mean; neither answers the question this figure exists for, which is how far a method's
    scenarios SPREAD. That matters twice over. A mean separating two methods by less than their
    own spread is not a difference a reader should act on. And two methods with equal means are
    not equally good if one of them occasionally lands far worse -- tail risk is a property of the
    method, and averaging is exactly the operation that hides it.

    Boxes carry the quartiles and the whiskers reach 1.5 IQR; every scenario is also drawn as a
    jittered point, because at this sample size the points are the evidence and the box is only a
    summary of them. The mean is marked separately from the median so a skewed method is visible
    as the gap between the two."""
    use_pub()
    figs = Path(out_dir)
    figs.mkdir(parents=True, exist_ok=True)
    order = sorted(methods, key=lambda c: df[c].mean())            # best (lowest F) first
    col = _method_colors(order)
    rng = np.random.RandomState(0)                                 # fixed jitter: redraws match

    fig, ax = plt.subplots(figsize=(0.85 * len(order) + 1.8, 3.2))
    bp = ax.boxplot([df[c] for c in order], widths=0.55, patch_artist=True,
                    medianprops=dict(color=C["neutral_dark"], lw=1.1),
                    whiskerprops=dict(color=C["neutral_dark"], lw=0.8),
                    capprops=dict(color=C["neutral_dark"], lw=0.8),
                    flierprops=dict(marker="", ls="none"))         # outliers shown as points below
    for patch, c in zip(bp["boxes"], order):
        patch.set_facecolor(col[c])
        patch.set_alpha(0.30)
        patch.set_edgecolor(col[c])
    for i, c in enumerate(order, start=1):
        y = df[c].to_numpy()
        ax.scatter(i + rng.uniform(-0.16, 0.16, len(y)), y, s=7, color=col[c], alpha=0.55,
                   edgecolors="none", zorder=3)
        ax.scatter([i], [y.mean()], marker="D", s=16, color=C["neutral_dark"], zorder=4,
                   edgecolors="white", linewidths=0.4)
    ax.axhline(1.0, ls=":", lw=0.7, color="0.75", zorder=1)        # pre-disaster level
    lo, hi = float(df[order].to_numpy().min()), float(df[order].to_numpy().max())
    ax.set_ylim(lo - 0.06 * (hi - lo), hi + 0.06 * (hi - lo))
    ax.set_xticks(range(1, len(order) + 1))
    ax.set_xticklabels([_label(c) for c in order], rotation=30, ha="right")
    ax.set_ylabel("objective  $F$")
    ax.set_title(f"per-scenario $F$ distribution  (M={len(df)}; diamond = mean)")
    fig.tight_layout()
    save_pub(fig, figs / "performance_distribution")
    plt.close(fig)


def make_gap_to_oracle(out_dir, df, methods):
    """Per-scenario gap to the brute-force oracle, $F - F^{*}$, for every method, as grouped bars.
    Only produced at scales where the oracle was enumerated, so it carries information that the
    per-scenario final-F figure cannot (the true optimum is unavailable at large scale)."""
    use_pub()
    figs = Path(out_dir)
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


def make_accuracy_compute(out_dir, stats):
    """Accuracy vs COMPUTE. One point per method at (serial-equivalent UE solves per
    scenario, mean final F). The compute axis counts UE solves rather than wall-clock, so it is
    neutral to how many worker processes a method was parallelized over and a method spread across
    many workers is not made to look cheaper than it is (see Caveats C1). Lower-left is better,
    meaning little compute together with high accuracy. `stats` is a list of dicts
    {method, mean_F, mean_ue, kind} where kind is 'greedy' (static rule, grey circle), 'meta'
    (budgeted population search, green square), 'rl' (budgeted nominal DQN, teal triangle),
    'rl_saa' (SAA DQN, purple down-triangle) or 'milp' (blue diamond)."""
    use_pub()
    figs = Path(out_dir)
    figs.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(3.6, 2.6))

    xs = [s["mean_ue"] for s in stats]
    ys = [s["mean_F"] for s in stats]
    xmax = max(xs)
    yspan = (max(ys) - min(ys)) or 1.0

    gi = 0
    for s in stats:                                             # markers first, all labels after
        if s["kind"] == "milp":
            s["_c"], s["_m"] = C["accent"], "D"
        elif s["kind"] == "meta":
            s["_c"], s["_m"] = C["good"], "s"
        elif s["kind"] == "rl":
            s["_c"], s["_m"] = C["teal"], "^"
        elif s["kind"] == "rl_saa":
            s["_c"], s["_m"] = C["purple"], "v"
        else:
            s["_c"], s["_m"] = _GREY_RAMP[gi % len(_GREY_RAMP)], "o"
            gi += 1
        ax.scatter(s["mean_ue"], s["mean_F"], s=50, color=s["_c"], marker=s["_m"],
                   edgecolor="white", lw=0.5, zorder=3)

    # Label placement: nudge each label off its point, then resolve residual collisions by pushing
    # the lower-F (visually higher) label of any overlapping pair further up. Points that share an x
    # (the greedy rules at T, or milp beside flow) would otherwise stack their text; comparing every
    # already-placed label rather than a fixed window is what closes the milp/flow overlap.
    dx = 0.018 * xmax
    dy = 0.05 * yspan
    lab = []                                                    # (x_text, y_text, ha, method)
    for s in sorted(stats, key=lambda z: (z["mean_ue"], z["mean_F"])):
        x, y = s["mean_ue"], s["mean_F"]
        left = x >= xmax - 1e-9                                 # rightmost labels leftward, never clip
        xt = x - dx if left else x + dx
        yt = y
        for _lx, _ly, _lha, _ in lab:                          # lift above any label too close
            if abs(_lx - xt) <= 0.16 * xmax and abs(_ly - yt) <= dy:
                yt = _ly + dy
        lab.append((xt, yt, "right" if left else "left", s["method"]))
        ax.annotate(s["method"], (x, y), xytext=(xt, yt), textcoords="data",
                    ha="right" if left else "left", va="center", fontsize=6.5,
                    color=C["neutral_dark"])

    ax.set_xlim(-0.04 * xmax, xmax * 1.14)
    ax.set_ylim(min(ys) - 0.06 * yspan, max(ys) + 0.16 * yspan)
    ax.set_xlabel("serial-equivalent UE solves / scenario")
    ax.set_ylabel("mean objective  $F$")
    fig.tight_layout()
    save_pub(fig, figs / "accuracy_vs_compute")
    plt.close(fig)
