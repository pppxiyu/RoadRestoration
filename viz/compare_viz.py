"""Comparison figure for the road-restoration solvers: the greedy baseline variants vs the
pretraining MILP vs (optional) the brute-force oracle. Panel a shows the per-scenario objective F
for every method; when the oracle is available, panel b shows each method's gap to that true
optimum. Styling follows viz/style.py so it matches the rest of the project's figures."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from viz.style import C, panel_label, save_pub, use_pub

# greedy variants share a darkening grey ramp (context); MILP is the emphasis blue (method under
# test); oracle is the signal red (the target the others are measured against).
_GREY_RAMP = ["#C6C6C6", "#9E9E9E", "#767676", "#4D4D4D"]


def _colors(cols):
    out, gi = {}, 0
    for c in cols:
        if c == "milp":
            out[c] = C["accent"]
        elif c == "oracle":
            out[c] = C["signal"]
        else:                                              # a greedy_<variant> column
            out[c] = _GREY_RAMP[gi % len(_GREY_RAMP)]
            gi += 1
    return out


def _label(c):
    return c[len("greedy_"):] if c.startswith("greedy_") else c


def make_comparison_figure(out_dir, df, N, methods, have_oracle):
    use_pub()
    figs = Path(out_dir) / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    scen = df["scenario"].to_numpy()
    x = np.arange(len(scen))

    plot_cols = methods + (["oracle"] if have_oracle else [])
    col = _colors(plot_cols)
    w = 0.85 / len(plot_cols)

    ncol = 2 if have_oracle else 1
    fig, ax = plt.subplots(1, ncol, figsize=(8.2 if have_oracle else 4.6, 3.3), squeeze=False)

    # ---- panel a: per-scenario objective F for each solver ----
    a = ax[0][0]
    for i, c in enumerate(plot_cols):
        a.bar(x + (i - (len(plot_cols) - 1) / 2) * w, df[c], width=w, label=_label(c), color=col[c])
    vals = df[plot_cols].to_numpy()
    lo, hi = float(vals.min()), float(vals.max())
    a.set_ylim(lo - 0.04 * (hi - lo), hi + 0.06 * (hi - lo))            # zoom: the F values sit near 1
    a.set_xlabel("scenario")
    a.set_ylabel("objective  F")
    a.set_xticks(x)
    a.set_xticklabels(scen)
    a.set_title(f"per-scenario F  (n={N})")
    a.legend(loc="upper right", ncol=2, fontsize=5.8, handlelength=1.1, columnspacing=1.0)
    panel_label(a, "a")

    # ---- panel b: gap to the true optimum (oracle only) ----
    if have_oracle:
        b = ax[0][1]
        gcols = list(methods)                                          # every non-oracle method has a gap
        wb = 0.85 / len(gcols)
        for i, c in enumerate(gcols):
            b.bar(x + (i - (len(gcols) - 1) / 2) * wb, df[f"gap_{c}"], width=wb,
                  label=_label(c), color=col[c])
        b.axhline(0.0, color=C["neutral_dark"], lw=0.8)
        b.set_xlabel("scenario")
        b.set_ylabel("gap to oracle  (F − F*)")
        b.set_xticks(x)
        b.set_xticklabels(scen)
        b.set_title("shortfall vs true optimum")
        b.legend(loc="upper right", ncol=2, fontsize=5.8, handlelength=1.1, columnspacing=1.0)
        panel_label(b, "b")

    fig.suptitle(f"Solver comparison (n={N}): greedy variants vs MILP" +
                 (" vs oracle" if have_oracle else ""), fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save_pub(fig, figs / "comparison")
    plt.close(fig)


def make_process_and_final(out_dir, N, milp_trace, greedy_finals, rep=None):
    """Figure 1: optimization PROCESS and final performance, baselines vs the MILP.
    Panel a: the MILP's best-so-far true F over its alternating iterations (one curve per scenario,
    the representative scenario `rep` highlighted), with each greedy variant's mean final F drawn as a
    horizontal line -- a one-shot static ranker has no iterative process, so its result is a level to
    reach. Panel b: the final F of every method, per scenario. `greedy_finals` maps variant -> DataFrame
    with columns [scenario, F, time_s]. When `rep` is None the highlighted scenario is chosen
    automatically as the one whose best-so-far F descends the most over the iterations (the most
    illustrative optimization trajectory); a hardcoded rep could land on a scenario that barely moves."""
    use_pub()
    figs = Path(out_dir) / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    scen_ids = sorted(milp_trace["scenario"].unique())
    if rep is None:                                                    # pick the scenario with the largest best-so-far descent
        descents = {m: (lambda s: float(s.iloc[0] - s.cummin().iloc[-1]))(
            milp_trace[milp_trace["scenario"] == m].sort_values("iter")["F"]) for m in scen_ids}
        rep = max(descents, key=descents.get)

    fig, ax = plt.subplots(1, 2, figsize=(8.2, 3.3))

    # ---- panel a: MILP best-so-far F over iterations + greedy final levels ----
    a = ax[0]
    for m in scen_ids:
        g = milp_trace[milp_trace["scenario"] == m].sort_values("iter")
        a.plot(g["iter"], g["F"].cummin(), lw=(1.8 if m == rep else 0.8),
               color=(C["accent"] if m == rep else C["neutral_light"]),
               alpha=(1.0 if m == rep else 0.8), zorder=(3 if m == rep else 1),
               label=("MILP (best-so-far)" if m == rep else None))
    gi = 0
    for v, d in greedy_finals.items():
        a.axhline(float(d["F"].mean()), ls="--", lw=1.1, color=_GREY_RAMP[gi % len(_GREY_RAMP)],
                  label=f"{v} (final)")
        gi += 1
    a.set_xlabel("MILP alternating iteration")
    a.set_ylabel("true objective  F")
    a.set_title(f"optimization process  (n={N})")
    a.legend(loc="upper right", fontsize=5.8, handlelength=1.4)
    panel_label(a, "a")

    # ---- panel b: final F per scenario, all methods ----
    b = ax[1]
    milp_final = milp_trace[milp_trace["is_best"] == True][["scenario", "F"]] if "is_best" in milp_trace \
        else milp_trace.groupby("scenario")["F"].min().reset_index()
    x = np.arange(len(scen_ids))
    cols = list(greedy_finals) + ["milp"]
    w = 0.85 / len(cols)
    for i, v in enumerate(cols):
        if v == "milp":
            vals = [float(milp_final[milp_final["scenario"] == m]["F"].min()) for m in scen_ids]
            color = C["accent"]
        else:
            d = greedy_finals[v]
            vals = [float(d[d["scenario"] == m]["F"].iloc[0]) for m in scen_ids]
            color = _GREY_RAMP[list(greedy_finals).index(v) % len(_GREY_RAMP)]
        b.bar(x + (i - (len(cols) - 1) / 2) * w, vals, width=w, label=v, color=color)
    allv = np.array([[float(greedy_finals[v][greedy_finals[v]["scenario"] == m]["F"].iloc[0])
                      for v in greedy_finals] +
                     [float(milp_final[milp_final["scenario"] == m]["F"].min())] for m in scen_ids])
    lo, hi = float(allv.min()), float(allv.max())
    b.set_ylim(lo - 0.04 * (hi - lo), hi + 0.06 * (hi - lo))
    b.set_xlabel("scenario")
    b.set_ylabel("final objective  F")
    b.set_xticks(x)
    b.set_xticklabels(scen_ids)
    b.set_title("final performance")
    b.legend(loc="upper right", ncol=2, fontsize=5.8, handlelength=1.1)
    panel_label(b, "b")

    fig.suptitle(f"Optimization process & final performance (n={N}): baselines vs MILP", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save_pub(fig, figs / "process_and_final")
    plt.close(fig)


def make_accuracy_time(out_dir, N, stats):
    """Figure 2: accuracy vs computation time. One point per method at (mean wall-clock seconds,
    mean final F). Lower-left is better (fast AND accurate). `stats` is a list of dicts
    {method, mean_F, mean_time_s, kind} where kind in {'greedy','milp'} sets the color."""
    use_pub()
    figs = Path(out_dir) / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4.4, 3.3))
    gi = 0
    for s in stats:
        color = C["accent"] if s["kind"] == "milp" else _GREY_RAMP[gi % len(_GREY_RAMP)]
        marker = "D" if s["kind"] == "milp" else "o"
        if s["kind"] == "greedy":
            gi += 1
        ax.scatter(s["mean_time_s"], s["mean_F"], s=55, color=color, marker=marker,
                   edgecolor="white", lw=0.5, zorder=3)
        ax.annotate(s["method"], (s["mean_time_s"], s["mean_F"]), textcoords="offset points",
                    xytext=(6, 3), fontsize=6.5, color=C["neutral_dark"])
    ax.set_xscale("log")
    ax.set_xlabel("mean wall-clock time per scenario  (s, log scale)")
    ax.set_ylabel("mean final objective  F  (lower = more accurate)")
    ax.set_title(f"accuracy vs computation time  (n={N})")
    ax.margins(0.18)
    fig.tight_layout()
    save_pub(fig, figs / "accuracy_vs_time")
    plt.close(fig)
