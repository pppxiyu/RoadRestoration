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
  02_objective_change_error  the per-slot level error e_t = g_t(loose) - g_t(truth), SPLIT
                             into the two parts that matter differently. Every decision this
                             study makes compares two alternatives evaluated under the SAME
                             solver, so the component of e_t shared by both alternatives
                             cancels and only the fluctuating remainder can do harm. Dots:
                             the de-biased jitter |e_t - mean(e)| per slot (mean taken within
                             the scenario) -- the raw material of every non-cancelling error.
                             Squares: the per-scenario bias |mean(e)| -- large but (to first
                             order) harmless, which is exactly why warm starting is usable:
                             its columns show high squares over low dots. Dashed line: the
                             smallest method-mean gap in the current comparison, read live
                             from comparison.csv; the verdict compares the DOTS to it, never
                             the squares, and is conservative -- jitter shrinks by a further
                             ~1/sqrt(104) when it aggregates into F. Grey curve (right axis):
                             what each tolerance costs in solve time.

Aggregation reuses util.ue_audit's own _changes, so a figure can never disagree with the
generated write-up in tolerance_audit.md.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

import config as P
from util.ue_audit import AE_PROD, OUT_DOC, OUT_RAW
from viz.style import C, save_pub, use_pub

# One fixed visual identity per configuration family, shared by both figures:
# the retired AequilibraE production setting is the grey anchor, cold in-house solves are the
# light blue family, warm in-house solves the dark blue family, and the production setting
# (config.py's rgap + warm start) is the red signal color everywhere it appears.
COL_ANCHOR, COL_COLD, COL_WARM = "#8A8A8A", "#9DB8D9", "#0F4D92"
# The level/change-error figures encode the START MODE alone in color, so they need two
# unmistakably different hues rather than the light/dark blue pair: cold blue, warm red.
COL_SA_COLD, COL_SA_WARM = "#3775BA", "#B64342"


def _tick_label(c):
    """A configuration's two-line tick label: tolerance above, start mode below (production
    marked on the mode line). Two lines keep the ticks horizontal at slide font sizes, where the
    one-line form had to slant to avoid collisions. _configs' single-line `label` is left alone
    -- figure 01's legend reads it."""
    mode = "warm" if c["warm"] else "cold"
    if c["key"] == "anchor":       # three lines: "retired" gets its own so it cannot collide
        return f"retired\n@{AE_PROD[0]:g}\n{mode}"
    tol = "@" + c["key"].rsplit("-", 1)[0]
    return f"{tol}\n{mode}" + ("\n(production)" if c["current"] else "")


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
# 02  the change error of g against solve time
# ----------------------------------------------------------------------------------------- #
_RC_BIG = {"font.size": 20, "axes.titlesize": 26, "axes.labelsize": 27,
           "xtick.labelsize": 23, "ytick.labelsize": 23, "legend.fontsize": 21}


def _current_horizon():
    """The scoring horizon T of the current instance, read from any solver run_meta on disk
    (the comparison's own meta runs no schedules and records horizon_T=0). Needed to fold the
    jitter criterion: per-slot noise entering the T-slot mean F shrinks by ~1/sqrt(T), so the
    reference the DOTS compare to sits sqrt(T) above the F-level gap."""
    import json
    best = 0
    for p in OUT_DOC.parents[1].rglob("run_meta.json"):
        try:
            t = json.load(open(p, encoding="utf-8")).get("instance", {}).get("horizon_T", 0)
        except (OSError, ValueError):
            continue
        best = max(best, int(t or 0))
    return best or None


def _smallest_method_gap():
    """The smallest gap between adjacent method means in the newest on-disk comparison -- the
    finest effect size the study currently claims to resolve, and therefore the level a solver
    error must stay well under to leave the method ranking untouched. Read live from
    comparison.csv so the reference moves with the study instead of fossilizing a hand-typed
    number; returns (gap, label) or None when no comparison exists yet."""
    roots = sorted((OUT_DOC.parents[1] / "04-comparison").glob("n*/results/comparison.csv"))
    pick = [p for p in roots if p.parent.parent.name == f"n{P.N_DISRUPTED_ORACLE}"] or roots
    if not pick:
        return None
    df = pd.read_csv(pick[-1]).drop(columns=["scenario"], errors="ignore")
    means = df.mean().sort_values()
    if len(means) < 2:
        return None
    gaps = means.diff().dropna()
    g, at = float(gaps.min()), gaps.idxmin()
    prev = means.index[list(means.index).index(at) - 1]
    n = int(pick[-1].parent.parent.name.lstrip("n"))    # instance size, from the folder name
    return g, f"{prev} vs {at}: {g:.4f}", n


def _objective_change_error(cfgs, path):
    """ONE panel: the per-slot level error e_t = g_val - g_ref split into BIAS and JITTER, the
    decomposition the study's decisions actually feel. Every decision compares two alternatives
    evaluated under the SAME solver, so the shared component of e_t cancels; only the
    fluctuating remainder harms. Dots -- |e_t - mean(e)| per slot, the de-biased jitter (mean
    within the scenario), the raw material of every non-cancelling error. Squares -- the
    per-scenario bias |mean(e)|: large but to first order harmless, which is why warm starting
    is usable at all (high squares over LOW dots). Dashed line -- the smallest method-mean gap
    in the current comparison, read live from disk; the verdict compares the DOTS to it, never
    the squares, and is conservative: jitter shrinks by a further ~1/sqrt(104) on its way into
    F, and by ~sqrt(window)/window relative to per-decision reward gaps ~100x larger. Honest
    caveat (also in the module docstring): bias cancellation assumes both alternatives see a
    similar bias; its state-dependence is not measurable from one schedule per scenario -- a
    multi-order differential audit would be needed. The file name keeps its historical
    'change_error' stem so downstream references stay valid; the content is the level-error
    decomposition that superseded the change-error view."""
    plt.rcParams.update(_RC_BIG)      # figure-local: render_audit_figures re-applies use_pub
    fig, ax2 = plt.subplots(figsize=(18.5, 8.6))

    rng = np.random.RandomState(0)          # jitter only; fixed so the figure is reproducible
    ms_per_cfg = []
    for i, c in enumerate(cfgs):
        # A warm chain solves its FIRST slot cold: that slot belongs to the cold error regime,
        # so it is excluded for warm configurations everywhere here (error rows AND timing),
        # exactly as util.ue_audit._stats_row excludes it.
        sw = c["s"][c["s"].slot > 1] if c["warm"] else c["s"]
        jit, bias = [], []
        for _, t in sw.groupby("scenario"):
            e = t.g_val.to_numpy() - t.g_ref.to_numpy()
            jit.append(np.abs(e - e.mean()))
            bias.append(abs(float(e.mean())))
        jit = np.concatenate(jit)
        x = i + rng.uniform(-0.16, 0.16, len(jit))
        col = COL_SA_WARM if c["warm"] else COL_SA_COLD
        ax2.scatter(x, jit, s=42, color=col, alpha=0.6, lw=0, zorder=2)
        # The jitter distribution's median as a horizontal tick.
        ax2.hlines(np.median(jit), i - 0.26, i + 0.26, color=C["neutral_dark"], lw=2.8,
                   zorder=4)
        ax2.scatter([i] * len(bias), bias, marker="s", s=150, color=C["neutral_dark"],
                    zorder=5, edgecolors="white", linewidths=1.2)
        ms_per_cfg.append(float(sw.ms.median()))
    # ONE reference line, at the TRAINING criterion (the stricter of the two consumers, chosen
    # by the project owner 2026-08-25): a reward is the sum of one completion window of
    # L = T/N slots, and training must tell apart orders whose F differs by the smallest
    # method gap -- per decision that is a return difference G = gap*T/N, while per-slot jitter
    # eta feeds the window sum as eta*sqrt(L). Safe iff eta*sqrt(L) < G, i.e. the DOTS must
    # stay under gap*sqrt(T/N). The folding lives in the line, so the dots stay raw
    # measurements and the reader folds nothing.
    ref, T_h = _smallest_method_gap(), _current_horizon()
    thr = None
    if ref is not None and T_h:
        thr = ref[0] * np.sqrt(T_h / ref[2])
        ax2.axhline(thr, color=C["neutral_dark"], lw=2.2, ls="--", zorder=1)
    # LINEAR y (2026-08-25, project owner's instruction; was log): distances now read as the
    # criterion does -- how far a cloud sits from the dashed line is proportional to how much
    # margin it has. The price is that the two @1e-06 columns flatten onto the axis, which is
    # itself the honest reading: on the scale where the criterion lives they are simply zero.
    ax2.set_ylim(bottom=0.0)
    ax2.set_xticks(range(len(cfgs)))
    # Two lines per tick (tolerance above, start mode below) so the labels stay horizontal and
    # readable at slide font sizes instead of slanting into each other.
    ax2.set_xticklabels([_tick_label(c) for c in cfgs])
    ax2.set_ylabel("UE-solve error (units of $F$)")

    # ---- second axis: what each tolerance COSTS, the trade-off the left axis cannot show ---- #
    ax_t = ax2.twinx()
    # Grey and behind the clouds: the cost curve is context for the error distributions, not a
    # competing signal, so it may cross them rather than being fenced into its own band.
    ax_t.plot(range(len(cfgs)), ms_per_cfg, color=C["neutral_mid"], lw=2.4, marker="D", ms=10,
              mec="white", mew=1.2, zorder=1)
    ax_t.set_ylabel("median solve time per slot (ms)")   # LINEAR: reads as elapsed time directly
    ax_t.set_ylim(0.0, max(ms_per_cfg) * 1.10)
    ax_t.spines["right"].set_visible(True)              # the pub style hides it by default

    # Below the axes: five entries in one row would collide with any corner of the plot, and the
    # figure is slide-shaped, so the width is there to spend.
    handles = [
        Line2D([], [], color=COL_SA_COLD, marker="o", ls="", ms=14,
               label="per-slot jitter, cold start"),
        Line2D([], [], color=COL_SA_WARM, marker="o", ls="", ms=14,
               label="per-slot jitter, warm start"),
        Line2D([], [], color=C["neutral_dark"], lw=2.8, label="median jitter"),
        Line2D([], [], color=C["neutral_dark"], marker="s", ls="", ms=13, mec="white",
               label="per-scenario bias (cancels in comparisons)"),
        Line2D([], [], color=C["neutral_mid"], lw=2.4, marker="D", ms=10, mec="white", mew=1.2,
               label="solve time (right axis)")]
    if thr is not None:
        handles.append(Line2D([], [], color=C["neutral_dark"], lw=2.2, ls="--",
                              label=(f"training criterion {thr:.4f}: smallest method gap "
                                     f"({ref[1]}) $\\times\\sqrt{{T/N}}$")))
    ax2.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2,
               frameon=False, columnspacing=1.8, handletextpad=0.5)
    fig.tight_layout()
    save_pub(fig, path)
    plt.close(fig)
    use_pub(slide=True)               # undo the figure-local rc bump


# ----------------------------------------------------------------------------------------- #
# 01  the per-slot level error of g
# ----------------------------------------------------------------------------------------- #
def _objective_level_error(cfgs, d, path):
    """ONE panel: the per-slot UE-solve error in the objective's per-slot term g, as a
    percentage of the truth, for the loosest cold solve (the failure mode) and the production
    setting. The retired anchor is deliberately NOT drawn -- a third overlapping line adds
    nothing here, and its error is on record in the change-error figure. What the panel shows is
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
    fig, axis = plt.subplots(figsize=(14.5, 6.8))

    axis.axhline(0.0, color=C["neutral_dark"], lw=1.6, zorder=1)
    for c in chosen:
        s = c["s"]
        s = s[s.scenario == m].sort_values("slot")
        rel = 100.0 * (s.g_val.to_numpy() - ref.g_ref.to_numpy()) / ref.g_ref.to_numpy()
        axis.plot(s.slot, rel, lw=2.4, zorder=3, marker="o", ms=7,
                  label=c["label"].replace("\n", " "), **styles[c["key"]])
    axis.set_xlabel("time slot")
    axis.set_ylabel("UE-solve error in the\nobjective term $g_t$  (%)")
    axis.legend(loc="lower left")
    fig.tight_layout()
    save_pub(fig, path)
    plt.close(fig)
    use_pub(slide=True)               # undo the figure-local rc bump


# ----------------------------------------------------------------------------------------- #
def render_audit_figures():
    """Render both audit figures from raw/tolerance_audit.csv into 02-tolerance_audit/."""
    src = OUT_RAW / "tolerance_audit.csv"
    if not src.exists():
        raise FileNotFoundError(
            f"{src} not found -- run `python -m util.ue_audit` (needs AequilibraE) to produce "
            "the audit data first; the figures are a pure function of that csv.")
    use_pub(slide=True)
    d = pd.read_csv(src)
    cfgs = _configs(d)
    OUT_DOC.mkdir(parents=True, exist_ok=True)
    _objective_level_error(cfgs, d, OUT_DOC / "01_objective_level_error")
    _objective_change_error(cfgs, OUT_DOC / "02_objective_change_error")
    print(f"audit figures written to {OUT_DOC}")
    return OUT_DOC


if __name__ == "__main__":
    render_audit_figures()
