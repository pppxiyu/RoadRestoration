"""Training diagnostics for the RL (DQN) solver, one fact per figure.

`rl_learning` is the CROSS-SCENARIO view: every scenario's best-so-far true objective F on one
pair of axes, for comparing how far and how fast the ten descents get -- the same
best-so-far-vs-compute reading make_process gives the MILP. `rl_objective` is the PER-SCENARIO
view: one small panel each, showing the objective the search actually sampled at every episode
against its best-so-far envelope and that scenario's flow baseline, which is the level the
flow-initialized agent starts from and is guaranteed not to exceed. `rl_reward` is the POLICY
view: the return each episode earns from its own repair completions, which rises only if the
learned policy improves, not merely if the search stumbles on a good schedule. `rl_td_error` is the LEARNER view: the Bellman
residual the value regression still carries, which separates "the agent never learned" from "the
agent learned something that does not help" -- read it FIRST, since a flat objective means very
different things depending on whether this curve came down.

The three training-progress figures (`rl_objective`, `rl_reward`, `rl_td_error`) are drawn against
the TRAINING EPISODE, which is the natural unit of training time here: one episode is one complete
rollout --
n decisions build a full repair order, and the resulting schedule is evaluated over the whole
horizon of T slots. `rl_learning` instead uses unique true evaluations, because its job is the
equal-budget comparison against GA/PSO, whose 60-evaluation budget is counted in exactly that
unit. The two axes nearly coincide (an episode that re-proposes a cached schedule trains the
network but spends no budget; at n=10 that happened 8 times in 608 episodes), so the difference
is one of meaning rather than of shape.

All four carry the per-scenario numbering used by make_final_performance_all. Styling follows
viz/style.py."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from viz.style import C, CMAP_SEQ, save_pub, use_pub


def _scenario_color(i, n):
    """Position i of n on the sequential ramp, kept off its palest end so the end-of-curve
    scenario numbers stay legible."""
    return plt.get_cmap(CMAP_SEQ)(0.05 + 0.75 * (i / max(1, n - 1)))


def _label_ends(ax, ends, pad=0.06):
    """Write each curve's scenario number at its right end, nudging apart the ones that coincide.
    `ends` is a list of (x, y, color, scenario); `pad` is the vertical collision window as a
    fraction of the y spread of the end points."""
    yspan = (max(e[1] for e in ends) - min(e[1] for e in ends)) or 1.0
    placed = []
    for x_end, y_end, color, m in ends:
        k = sum(1 for px, py in placed if abs(px - x_end) <= 0.8 and abs(py - y_end) <= pad * yspan)
        placed.append((x_end, y_end))
        ax.annotate(str(m), (x_end, y_end), textcoords="offset points", xytext=(4, 2 + 9 * k),
                    fontsize=6, fontweight="bold", color=color, va="center", zorder=4)


def make_rl_learning(out_dir, trace):
    """Best-so-far true $F$ vs unique evaluations for every scenario, one numbered curve each.

    This is the BUDGET view, and the only one of the three drawn against unique schedule
    evaluations rather than training episodes: 60 unique evaluations is exactly the budget GA and
    PSO are given, so this axis is what makes the equal-budget claim readable. Episodes that
    replay a cached schedule train the network but spend no budget, so they add no x progress --
    hence the groupby below, which keeps one point per evaluation level. For training time as
    such, see make_rl_objective. Reads the rl_trace.csv schema written by util/rl.py."""
    if trace.empty:                      # nothing to draw (e.g. resumed run with no trace rows)
        return
    use_pub()
    figs = Path(out_dir) / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    scen_ids = sorted(trace["scenario"].unique())

    fig, a = plt.subplots(figsize=(4.8, 3.4))
    ends = []
    for i, m in enumerate(scen_ids):
        g = trace[trace["scenario"] == m].sort_values("episode")
        # One point per unique evaluation: keep the last episode row at each cum_evals level.
        g = g.groupby("cum_evals", as_index=False).last()
        x = g["cum_evals"].to_numpy()
        f = g["best_F"].to_numpy()
        color = _scenario_color(i, len(scen_ids))
        a.plot(x, f, lw=1.1, color=color, alpha=0.9, zorder=2, drawstyle="steps-post")
        ends.append((float(x[-1]), float(f[-1]), color, int(m)))
    _label_ends(a, ends)

    a.set_xlabel("unique true evaluations")
    a.set_ylabel("true objective  $F$  (best-so-far)")
    fig.tight_layout()
    save_pub(fig, figs / "rl_learning")
    plt.close(fig)


def make_rl_objective(out_dir, trace, flow_F=None, ncols=5):
    """Per-scenario trajectory of the true objective $F$, one small panel per scenario.

    Each panel carries three things against the training episode: the objective of the schedule
    the policy proposed that episode (thin line -- eps-greedy spikes included), the best-so-far
    envelope over those proposals (bold, the value the scenario finally reports), and that
    scenario's flow-greedy baseline (dashed). One episode is one full-horizon rollout, so this is
    training time, not budget; see the module docstring for why make_rl_learning uses the other
    axis.

    The baseline matters because the agent is initialized to reproduce it: episode 0 is the forced
    greedy rollout, so the bold line STARTS exactly on the dashed line (verified equal to
    flow_optima.csv to 1e-12 on all ten n=10 scenarios), and best-by-true-F selection means it can
    never end above it. The vertical distance between dashed and bold is therefore exactly what
    the RL run bought over the strongest static rule.

    This is the per-scenario counterpart to make_rl_learning, which puts all the bold envelopes on
    one pair of axes to compare scenarios; here the scenarios are separated so the raw samples and
    the baseline are legible. `flow_F` maps scenario -> flow baseline F (omit it to draw panels
    without the baseline). Reads the rl_trace.csv schema written by util/rl.py."""
    if trace.empty:
        return
    use_pub()
    figs = Path(out_dir) / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    scen_ids = sorted(trace["scenario"].unique())
    n = len(scen_ids)
    nrows = -(-n // ncols)                                       # ceil

    fig, axes = plt.subplots(nrows, ncols, figsize=(1.75 * ncols, 1.75 * nrows), sharex=True)
    axes = np.atleast_1d(axes).ravel()
    for i, m in enumerate(scen_ids):
        ax = axes[i]
        g = trace[trace["scenario"] == m].sort_values("episode")
        x = g["episode"].to_numpy()
        color = _scenario_color(i, n)
        ax.plot(x, g["F"], lw=0.6, color=color, alpha=0.55, zorder=2)          # sampled schedules
        ax.plot(x, g["F"].cummin(), lw=1.3, color=color, zorder=4,
                drawstyle="steps-post")                                        # best-so-far
        if flow_F is not None and int(m) in flow_F:
            ax.axhline(flow_F[int(m)], ls="--", lw=0.8, color=C["neutral_dark"], zorder=3)
        ax.axhline(1.0, ls=":", lw=0.6, color="0.8", zorder=1)                 # pre-disaster level
        ax.set_title(f"scenario {m}", fontsize=6.5, pad=2)
        ax.tick_params(labelsize=5.5)
    for ax in axes[n:]:                                          # blank any unused cell
        ax.axis("off")
    for i, ax in enumerate(axes[:n]):
        if i % ncols == 0:
            ax.set_ylabel("objective  $F$", fontsize=6.5)
        if i >= n - ncols:
            ax.set_xlabel("training episode", fontsize=6.5)
    fig.tight_layout(w_pad=0.6, h_pad=0.8, rect=(0, 0.045, 1, 1))
    # Key at figure level: the 2 x 5 grid is exactly filled, so an in-panel note would sit on data.
    fig.text(0.5, 0.012, "thin: schedule proposed that episode    bold: best-so-far    "
                         "dashed: flow baseline    dotted: pre-disaster level $F=1$",
             ha="center", fontsize=5.5, color=C["neutral_mid"])
    save_pub(fig, figs / "rl_objective")
    plt.close(fig)


def make_rl_td_error(out_dir, trace):
    """Mean absolute TD error per episode, one numbered curve per scenario, on a log axis.

    The TD (temporal-difference) error is the gap between what the network currently predicts and
    what one step of lookahead says it should be,

        delta = ( R_i + max_{e'} Q_target(s_{i+1}, e') ) - Q(s_i, a_i),

    with no successor term on the last decision. Each episode runs `updates_per_ep` minibatch
    updates; the value plotted is the mean |delta| over those minibatches, i.e. how far the
    regression still is from its own targets at that point in training.

    HOW TO READ IT. A fall followed by a flat floor is healthy: the value function has settled.
    A high flat line means the regression cannot fit at all (too small a network, too small a
    learning rate, or targets too noisy to chase); a rising line means the bootstrapped targets
    are diverging. Small sawtooth steps are expected rather than alarming -- the target network is
    resynced every `target_sync` gradient steps, which moves the target the residual is measured
    against.

    WHAT IT CANNOT TELL YOU. A small TD error only says the network is SELF-CONSISTENT, never that
    it is right or that its greedy ordering is any good: a value function that is uniformly wrong
    but coherent has a tiny residual. Read it first, to rule out "it never learned", and then read
    make_rl_objective and make_rl_reward for whether the schedules and the policy actually
    improved. At n=10 this curve drops roughly fourfold within the first ten episodes and then
    plateaus, so the plateau in the other figures is NOT a failure to converge.

    Silently skipped for traces written before this column existed."""
    if trace.empty or "td" not in trace.columns:
        return
    use_pub()
    figs = Path(out_dir) / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    scen_ids = sorted(trace["scenario"].unique())

    fig, a = plt.subplots(figsize=(4.8, 3.0))
    ends = []
    for i, m in enumerate(scen_ids):
        g = trace[trace["scenario"] == m].sort_values("episode")
        color = _scenario_color(i, len(scen_ids))
        a.plot(g["episode"], g["td"], lw=1.0, color=color, alpha=0.9, zorder=2)
        ends.append((float(g["episode"].iloc[-1]), float(g["td"].iloc[-1]), color, int(m)))
    _label_ends(a, ends)

    a.set_yscale("log")
    a.set_xlabel("training episode")
    a.set_ylabel("mean $|$TD error$|$ per episode")
    fig.tight_layout()
    save_pub(fig, figs / "rl_td_error")
    plt.close(fig)


def make_rl_reward(out_dir, trace, window=9):
    """Episode RETURN for every scenario, one numbered curve each.

    The return is the sum of the episode's per-decision rewards, each of which is the objective
    improvement that decision's own segment produced when it came back into service, credited
    with its RHO-decaying retention (see util/rl.py). Higher is better. It is NOT a rescaling of
    $F$: every completion jump is extrapolated over the slots that remain, so the return measures
    how much improvement the policy's choices EARNED, on a scale of its own.

    Unlike rl_learning this is not a best-so-far curve: it is what the policy actually earned,
    which falls whenever exploration takes a bad action and rises only as the learned corrections
    improve -- the two figures answer "did the search find something?" and "did the policy get
    better?" separately, and a rising curve here is the direct evidence that the agent learned
    rather than sampled its way to a good schedule. Plotted as a CENTRED rolling mean over
    `window` episodes, since single-episode returns swing several times wider than the trend. The
    x axis is the training episode, one full-horizon rollout per step, matching make_rl_objective.

    Silently skipped for traces written before the `ret` column existed."""
    if trace.empty or "ret" not in trace.columns:
        return
    use_pub()
    figs = Path(out_dir) / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    scen_ids = sorted(trace["scenario"].unique())

    fig, a = plt.subplots(figsize=(4.8, 3.4))
    ends = []
    for i, m in enumerate(scen_ids):
        g = trace[trace["scenario"] == m].sort_values("episode")
        x = g["episode"].to_numpy()
        smooth = g["ret"].rolling(window, center=True, min_periods=1).mean().to_numpy()
        color = _scenario_color(i, len(scen_ids))
        a.plot(x, smooth, lw=1.1, color=color, alpha=0.9, zorder=2)
        ends.append((float(x[-1]), float(smooth[-1]), color, int(m)))
    _label_ends(a, ends)

    a.set_xlabel("training episode")
    a.set_ylabel("episode return  $\\sum_i R_i$")
    a.text(0.02, 0.03, f"centred rolling mean, {window} episodes", transform=a.transAxes,
           fontsize=5.5, color=C["neutral_mid"], ha="left", va="bottom")
    fig.tight_layout()
    save_pub(fig, figs / "rl_reward")
    plt.close(fig)
