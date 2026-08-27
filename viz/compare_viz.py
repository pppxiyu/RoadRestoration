"""Comparison figures for the road-restoration solvers: the greedy baseline variants, GA/PSO and
the RL DQN vs the pretraining MILP vs (optional) the brute-force oracle. Each fact has one home:
the spread of each method's per-scenario final objective F across scenarios lives in
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
# (grey ramp / GA red / rl_dqn teal / rl_s2v orange / rl_s2v_saa purple / MILP dark blue /
# oracle red) lives in _method_colors below.
_GREY_RAMP = ["#C6C6C6", "#9E9E9E", "#767676", "#4D4D4D"]

# The pool-SAA family shares one hue and separates by depth, so a reader sees several settings of
# one method rather than several methods. Deeper = larger training pool. The _adaptive twins
# (deviation-24 observation channels on) take their own hue with the same depth rule, so
# plain-vs-adaptive reads as two families of one method; _SAA_COLORS is the single lookup every
# consumer uses.
_SAA_RAMP = {"rl_s2v_saa64": "#B9A7D6", "rl_s2v_saa128": "#5B4383"}
_SAA_ADPT = {"rl_s2v_saa64_adaptive": "#D9A0BC", "rl_s2v_saa128_adaptive": "#A34A72"}
# Cross-scale zero-shot transfers (util/transfer.py): a hue of their own -- olive, used by no
# native family -- so a transferred policy reads as foreign at a glance. One entry per transfer.
_XFER = {"rl_s2v_saa64_adaptive_from_n10": "#8A8B3A",
         "rl_s2v_saa64_adaptive_from_n16": "#5E5F22"}
_SAA_COLORS = {**_SAA_RAMP, **_SAA_ADPT, **_XFER}


def _label(c):
    return c[len("greedy_"):] if c.startswith("greedy_") else c


def _method_colors(cols):
    """Static greedy rules share a darkening grey ramp (context). GA, being a population
    metaheuristic rather than a static ranker, reuses the signal red, which cannot collide
    because the oracle is never drawn in the same figure as GA. rl_dqn, the rank-loss DQN, takes the
    teal accent, and the experimental S2V solvers take orange (rl_s2v) and purple (rl_s2v_saa).
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
        elif v == "rl_dqn":
            out[c] = C["teal"]
        elif v == "rl_s2v":
            out[c] = "#C97B2D"     # EXPERIMENTAL S2V-DQN: orange, outside the shared palette
        elif v in _SAA_COLORS:
            out[c] = _SAA_COLORS[v]  # EXPERIMENTAL pool-SAA S2V: one color per (pool, adaptive)

        else:                                              # a static greedy variant
            out[c] = _GREY_RAMP[gi % len(_GREY_RAMP)]
            gi += 1
    return out


def make_performance_distribution(out_dir, df, methods, n_orders=None):
    """The DISTRIBUTION of each method's per-scenario F, one box per method, ranked so the BEST
    method (lowest mean F, this objective being minimized) sits at the RIGHT-hand end.

    The summary table reports a mean, which does not answer the question this figure exists
    for: how far a method's scenarios SPREAD. That matters twice over. A mean separating two methods by less than their
    own spread is not a difference a reader should act on. And two methods with equal means are
    not equally good if one of them occasionally lands far worse -- tail risk is a property of the
    method, and averaging is exactly the operation that hides it.

    Boxes carry the quartiles and the whiskers reach the full range -- NO scenario is treated as
    an outlier, because a method's extreme scenarios are part of what is being measured, not
    contamination. Every scenario is also drawn as a jittered point, because at this sample size
    the points are the evidence and the box is only a summary of them. The mean is marked separately from the median so a skewed method is visible
    as the gap between the two.

    `n_orders` maps a method to how many DISTINCT repair orders it delivered across the
    scenarios; where it is known the count joins the box annotation, since a method's spread
    reads differently when one committed order produced it than when fifty different ones did."""
    use_pub()
    figs = Path(out_dir)
    figs.mkdir(parents=True, exist_ok=True)
    # Descending mean: the worst method is leftmost and the best (lowest F) lands on the RIGHT,
    # so the eye travels toward the winner. The other comparison figures keep their own orders --
    # this one is read as a ranking, they are not.
    order = sorted(methods, key=lambda c: df[c].mean(), reverse=True)
    col = _method_colors(order)
    rng = np.random.RandomState(0)                                 # fixed jitter: redraws match

    n_orders = n_orders or {}
    # Boxes sit closer together (widths 0.72 in a unit spacing) and the canvas is scaled for
    # slide-size type rather than the old thumbnail proportions.
    fig, ax = plt.subplots(figsize=(1.25 * len(order) + 2.2, 6.4))
    # whis=(0, 100): the whiskers reach the MINIMUM and MAXIMUM, so no scenario is classified as
    # an outlier and none is excluded from the box's reach. The 1.5-IQR convention would declare a
    # tail scenario an outlier and stop the whisker short of it, which is exactly the wrong reading
    # here -- a method's bad scenarios are a property of the method, not contamination to trim.
    bp = ax.boxplot([df[c] for c in order], widths=0.72, patch_artist=True, whis=(0, 100),
                    medianprops=dict(color=C["neutral_dark"], lw=1.8),
                    whiskerprops=dict(color=C["neutral_dark"], lw=1.3),
                    capprops=dict(color=C["neutral_dark"], lw=1.3))
    for patch, c in zip(bp["boxes"], order):
        patch.set_facecolor(col[c])
        patch.set_alpha(0.30)
        patch.set_edgecolor(col[c])
    lo, hi = float(df[order].to_numpy().min()), float(df[order].to_numpy().max())
    span = (hi - lo) or 1.0
    for i, c in enumerate(order, start=1):
        y = df[c].to_numpy()
        ax.scatter(i + rng.uniform(-0.2, 0.2, len(y)), y, s=16, color=col[c], alpha=0.55,
                   edgecolors="none", zorder=3)
        ax.scatter([i], [y.mean()], marker="D", s=44, color=C["neutral_dark"], zorder=4,
                   edgecolors="white", linewidths=0.7)
        # The summary numbers on the box itself, so the reader need not cross-reference the
        # table: mean (the diamond), median (the box line), best (lowest F, this objective), and
        # -- where known -- how many distinct orders the method actually delivered.
        txt = f"mean {y.mean():.4f}\nmed  {np.median(y):.4f}\nbest {y.min():.4f}"
        if c in n_orders:
            txt += f"\norders {n_orders[c]}"
        ax.annotate(txt, (i, y.max()), textcoords="offset points", xytext=(0, 7), ha="center",
                    va="bottom", fontsize=11.5, color=C["neutral_dark"], linespacing=1.25)
    ax.set_ylim(lo - 0.06 * span, hi + 0.46 * span)               # top headroom for the annotations
    ax.set_xlim(0.4, len(order) + 0.6)                            # trim the default side padding
    ax.set_xticks(range(1, len(order) + 1))
    ax.set_xticklabels([_label(c) for c in order], rotation=30, ha="right", fontsize=15)
    ax.tick_params(axis="y", labelsize=15)
    ax.set_ylabel("objective  $F$", fontsize=18)
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
    {method, mean_F, mean_ue, kind}. SHAPE encodes the method family -- circle for the
    non-search references (static rules and the MILP), square for the population metaheuristic
    (GA), triangle for every RL variant -- and COLOR identifies the individual method; the
    legend carries one entry per method, so the points themselves stay unlabeled."""
    use_pub()
    figs = Path(out_dir)
    figs.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(3.6, 2.6))

    xs = [s["mean_ue"] for s in stats]
    ys = [s["mean_F"] for s in stats]
    xmax = max(xs)
    yspan = (max(ys) - min(ys)) or 1.0

    # Family shapes: references (rules + MILP) circle, GA square, all RL triangle.
    _SHAPE = {"greedy": "o", "milp": "o", "meta": "s", "rl": "^", "rl_s2v": "^",
              **{k: "^" for k in _SAA_COLORS}}
    _COLOR = {"milp": C["accent"], "meta": C["good"], "rl": C["teal"],
              "rl_s2v": "#C97B2D", **_SAA_COLORS}   # greedy rules take the grey ramp
    gi = 0
    handles = []
    for s in stats:
        if s["kind"] == "greedy":
            col = _GREY_RAMP[gi % len(_GREY_RAMP)]
            gi += 1
        else:
            col = _COLOR[s["kind"]]
        m = _SHAPE[s["kind"]]
        ax.scatter(s["mean_ue"], s["mean_F"], s=50, color=col, marker=m,
                   edgecolor="white", lw=0.5, zorder=3)
        handles.append(plt.Line2D([], [], color=col, marker=m, ls="", ms=6,
                                  markeredgecolor="white", markeredgewidth=0.5,
                                  label=s["method"]))
    ax.legend(handles=handles, loc="upper right", fontsize=5.5, framealpha=0.9,
              borderpad=0.4, labelspacing=0.35, handletextpad=0.4)

    ax.set_xlim(-0.04 * xmax, xmax * 1.14)
    ax.set_ylim(min(ys) - 0.06 * yspan, max(ys) + 0.16 * yspan)
    ax.set_xlabel("serial-equivalent UE solves / scenario")
    ax.set_ylabel("mean objective  $F$")
    fig.tight_layout()
    save_pub(fig, figs / "accuracy_vs_compute")
    plt.close(fig)
