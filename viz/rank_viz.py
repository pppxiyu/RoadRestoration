"""
Figures for the ranking-loss RL solver (util.rl_rank), drawn from the diagnostics that solver
records rather than from anything held in memory, so a run's figures can be redrawn -- or refreshed
mid-run -- without retraining.

They are drawn by the solver itself, on every diagnostics flush and again at delivery, which is
deliberate: an external plotting script keyed to column names drifts out of sync the moment the
recorder changes one, and fails silently if its errors are swallowed. Keeping the figures a
by-product of the same flush that writes the data makes that drift impossible to miss.

Four figures per run, all reading outputs/03-rl/{solver}/n{N}/log/:
  {v}_objective   the objective curves (for the stochastic variant this includes BOTH rulers: the
                  validation worlds the stopping rule reads, and, dashed, the frozen evaluation
                  scenarios recorded purely as a diagnostic), which differ by variant because the
                  two variants do not
                  have comparable per-episode objectives: the nominal variant's F is measured in
                  one fixed world so its best-so-far F is meaningful, while the stochastic
                  variant's
                  episode F comes from a freshly drawn world and is not comparable across episodes,
                  so its honest progress signal is the validation score instead
  {v}_learning    the training loss
  {v}_qvalue      the per-sample TD error. The batch means of the prediction and of its own target
                  are NOT plotted: the regression term fits one to the other, so they coincide by
                  construction and the resulting figure looks like agreement where none was tested
  {v}_stability   how often the greedy policy's order changes, and how long it holds between
                  changes -- read off the recorded orders themselves, not inferred from F
  {v}_qmc         (after delivery) the delivered policy's predicted Q against the realized
                  Monte-Carlo return, which is where a ranking loss is expected to rank well while
                  staying mis-calibrated in level
  {v}_value_estimate  (after delivery; nominal variant) the value estimate max_e Q(s0, e) across
                  training against the recorded scenF probes in return units -- van Hasselt et al.
                  (2016) Figure 3 top row, realized side as sparse recorded points, nothing
                  re-evaluated
  {v}_repr_tsne_states  (after delivery) two-dimensional t-SNE of the last-hidden-layer
                  representations of the states the delivered policy experiences, one point per
                  state, colored by the predicted state value V, with the decision-state
                  "screenshots" Mnih et al. (2015) Figure 4 annotates selected points with -- the
                  Sioux Falls network drawn with repaired / still-damaged / chosen-now segments.
                  Points are taken from the two ENDS of the value range, four lowest-V above the
                  scatter and four highest-V below, so the figure contrasts the states the network
                  values least against those it values most
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd

from viz.style import C, panel_label, save_pub, use_pub

_TOY_NET = Path(__file__).resolve().parent.parent / "data" / "siouxfalls_toy" / "network"


def _normalize(tr):
    """Accept the column names an earlier prototype of the recorder used, so every run ever
    produced is drawable by this one path. A second plotting script kept for the older names is
    exactly how figures drift away from the data they claim to show."""
    # NOTE `scenF` is deliberately NOT aliased. An early prototype used that name for what is now
    # the evaluation-scenario diagnostic, and it still means exactly that, so it must keep its own
    # identity -- renaming it onto F_val would plot the test-set curve under the validation label.
    alias = {"world_F": "F"}
    for old, new in alias.items():
        if old in tr.columns and new not in tr.columns:
            tr = tr.rename(columns={old: new})
    return tr


def _scen_line(ax, tr):
    """The M frozen evaluation scenarios -- the ruler every method is finally compared on --
    recorded as a DIAGNOSTIC and drawn for BOTH variants. It answers "what would this run have
    delivered at episode k?" after the fact. Dashed to mark that it drives nothing: stopping reads
    the best-so-far order (nominal variant) or the validation worlds (stochastic), never this
    curve."""
    if "scenF" not in tr or not tr["scenF"].notna().any():
        return
    q = tr.dropna(subset=["scenF"])
    ax.plot(q["episode"], q["scenF"], color=C["teal"], lw=1.4, ls="--", marker="s", ms=2.5,
            label="greedy policy, evaluation (diagnostic)")


def _seed_traces(out_dir, variant):
    """Every archived seed's trace from this run's history/ folder, newest sweep only.

    util.seed_sweep keeps each seed's complete run under n{N}/history/seed{s}/ and restores the
    best one to n{N}/ itself. So the figure at the top level describes ONE run (the delivered best
    seed) while the sibling archives hold the others -- reading them here is what lets the delivered
    figure show its own variability instead of only its own best case. Returns [] when no sweep has
    been run, and the figure then draws exactly as it always did."""
    hist = Path(out_dir) / "history"
    if not hist.is_dir():
        return []
    out = []
    for d in sorted(hist.glob("seed*")):
        p = d / "log" / f"{variant}_trace.csv"
        if p.exists():
            t = _normalize(pd.read_csv(p))
            if len(t):
                out.append((d.name, t))
    return out


def _seed_band(ax, traces, col, color, label):
    """Min-max envelope across seeds of one recorded curve, drawn behind the delivered run.

    Seeds stop at DIFFERENT episodes -- each run ends on its own plateau -- so the number of runs
    contributing to the envelope falls as the axis goes right. That is drawn rather than smoothed
    over: the band is solid only across the span where EVERY seed was still training, and fainter
    beyond it, where it summarizes a shrinking subset and a widening or narrowing envelope may be
    an artifact of which runs have already stopped. Forward-filling the stopped runs was rejected
    -- it would draw a flat continuation as though training had gone on.
    """
    series = []
    for _, t in traces:
        if col in t and t[col].notna().any():
            q = t.dropna(subset=[col])
            series.append(pd.Series(q[col].to_numpy(), index=q["episode"].to_numpy()))
    if len(series) < 2:
        return 0
    wide = pd.concat(series, axis=1).sort_index()
    lo, hi, n = wide.min(axis=1), wide.max(axis=1), wide.notna().sum(axis=1)
    full = n == len(series)                       # episodes where every seed is still running
    ax.fill_between(wide.index[full], lo[full], hi[full], color=color, alpha=0.20, lw=0,
                    zorder=0, label=f"{label} ({len(series)} seeds, min–max)")
    if (~full).any():
        part = n >= 2
        ax.fill_between(wide.index[part], lo[part], hi[part], color=color, alpha=0.08, lw=0,
                        zorder=0)
    return len(series)


def _objective(ax, tr, variant, traces=()):
    ep = tr["episode"]
    # Bands first, so the delivered run's own lines stay readable on top of them.
    if traces:
        _seed_band(ax, traces, "best_F", C["signal"], "best so far")
        _seed_band(ax, traces, "scenF", C["teal"], "evaluation")
    # WHICH curves exist is a property of the RUN, read off its trace, not of the variant's
    # name. A solver that probes on validation worlds records F_val and is judged by it; a
    # nominal-world solver records best_F. Name-matching here was a live defect: rl_s2v_saa
    # probes on validation worlds but its name does not start with the retired "rl_saa", so it
    # was drawn with the nominal curve set and its own selection signal never appeared.
    if "F_val" in tr and tr["F_val"].notna().any():
        ax.plot(ep, tr["F"], color=C["neutral_light"], lw=0.8,
                label="episode F (in its own training worlds)")
        if "buf_bestF" in tr and tr["buf_bestF"].notna().any():
            ax.plot(ep, tr["buf_bestF"], color=C["accent2"], lw=1.1,
                    label="best F in the imitation buffer")
        p = tr.dropna(subset=["F_val"])
        ax.plot(p["episode"], p["F_val"], color=C["signal"], lw=1.4, marker="o", ms=2.5,
                label="greedy policy, validation worlds")
        _scen_line(ax, tr)
    else:
        ax.plot(ep, tr["F"], color=C["neutral_light"], lw=0.6, alpha=0.45, zorder=1,
                label="behavior rollout (nominal)")
        if "best_F" in tr and tr["best_F"].notna().any():
            ax.plot(ep, tr["best_F"], color=C["signal"], lw=1.4, zorder=3,
                    label="best so far (nominal)")
        # The nominal variant needs this line MORE than the stochastic one, not less: every other
        # curve on this axis is a nominal-world F, so without it the figure never shows the number
        # the method is actually judged by.
        _scen_line(ax, tr)
    ax.set_xlabel("episode")
    ax.set_ylabel("F (lower = better)")


def _draw_state(ax, dl, disset, nodes, edges, m, k):
    """One decision-state 'screenshot': the whole Sioux Falls network, its disrupted segments
    colored by status at step k of scenario m -- repaired (teal), still damaged (gray), and the one
    the greedy policy chooses now (red, the action). Undisrupted roads are drawn faint for context.
    The state is reconstructed from the recorded candidates: the rows for (m, k) ARE the segments
    not yet repaired, so repaired = disrupted minus those, and the chosen one is is_chosen == 1."""
    sub = dl[(dl["scenario"] == m) & (dl["step"] == k)]
    remaining = set(int(x) for x in sub["seg"])
    repaired = disset - remaining
    now = int(sub[sub["is_chosen"] == 1]["seg"].iloc[0])
    for _, r in edges.iterrows():
        a, b = nodes.loc[int(r.u)], nodes.loc[int(r.v)]
        eid = int(r.edge_id)
        if eid == now:          col, lw, zo = C["signal"], 2.6, 5
        elif eid in repaired:   col, lw, zo = C["teal"], 1.8, 4
        elif eid in remaining:  col, lw, zo = "#9A9A9A", 1.8, 3
        else:                   col, lw, zo = "#E4E4E4", 0.6, 1
        ax.plot([a.x, b.x], [a.y, b.y], color=col, lw=lw, zorder=zo, solid_capstyle="round")
    ax.scatter(nodes.x, nodes.y, s=3, color="#777", zorder=6)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
    for sp in ax.spines.values():        # the publication style hides top/right; a thumbnail needs
        sp.set_visible(True)             # all four, or its "frame" is two stray lines
        sp.set_edgecolor("#999")
        sp.set_linewidth(0.5)
    return now, len(repaired)


def _repr_tsne_screenshots(out_dir, variant, xy, v, z, dl):
    """Mnih et al. (2015) Figure 4 in full: the t-SNE scatter with a selected number of points
    annotated by their decision-state screenshot, connected by a thin line. Draws nothing that was
    not already recorded -- the embedding is passed in, the states come from the deliver records,
    the geometry from the toy network files. Skips cleanly if the network files are absent."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import ConnectionPatch
        from matplotlib.colors import Normalize
        nodes = pd.read_csv(_TOY_NET / "nodes.csv").set_index("node_id")
        edges = pd.read_csv(_TOY_NET / "edges.csv")
    except Exception as exc:
        print(f"  [{variant}] repr_tsne_states SKIPPED ({type(exc).__name__}: {exc})", flush=True)
        return
    scen, step = z["scenario"].astype(int), z["step"].astype(int)
    # The disrupted set needs no oracle-file lookup: step 0's candidates are every disrupted segment.
    disset = set(int(x) for x in dl[dl["step"] == 0]["seg"])

    # Annotated points are drawn from the two ENDS of the value range -- the darkest-red states
    # (highest V) and the darkest-blue ones (lowest V) -- rather than spread evenly around the
    # cloud. Those two groups are what the figure is for: the reader should be able to see what a
    # state the network values highly looks like next to one it values least, and confirm that the
    # colour axis corresponds to how far the recovery has progressed. Within each group the points
    # are taken from distinct scenarios and spaced apart in the embedding, so the thumbnails fan
    # out instead of stacking on one cluster.
    n_side = 4
    lo_pool = np.argsort(v)[:max(20, len(v) // 8)]          # deepest blue
    hi_pool = np.argsort(v)[-max(20, len(v) // 8):]         # deepest red

    def spread(pool):
        """Greedy farthest-point pick within a pool, one point per scenario."""
        out, used = [], set()
        cand = list(pool)
        while cand and len(out) < n_side:
            if not out:                                      # start at the pool's extreme
                j = cand[0] if pool is lo_pool else cand[-1]
            else:                                            # then the point farthest from all picks
                d = np.min([np.hypot(xy[cand, 0] - xy[q, 0], xy[cand, 1] - xy[q, 1])
                            for q in out], axis=0)
                j = cand[int(np.argmax(d))]
            if int(scen[j]) not in used:
                out.append(j); used.add(int(scen[j]))
            cand.remove(j)
        return out

    # Each group is laid out left-to-right in the order its points appear left-to-right in the
    # embedding. Selection is by value and spacing, which says nothing about horizontal order, so
    # without this the leftmost thumbnail could belong to the rightmost point and the connectors
    # cross. Sorting by the point's x is what keeps them roughly parallel.
    pick = (sorted(spread(hi_pool), key=lambda j: xy[j, 0])
            + sorted(spread(lo_pool), key=lambda j: xy[j, 0]))

    norm = Normalize(vmin=float(np.nanmin(v)), vmax=float(np.nanmax(v)))
    cmap = plt.get_cmap("coolwarm")
    fig = plt.figure(figsize=(10, 6.2))
    # Two fixed rows of thumbnails -- the LOW-V group above the scatter, the high-V group below --
    # with the scatter in the band between. Low above, high below because the colour axis runs that
    # way in the reader's hand: blue sits at the bottom of the colourbar but these are the states
    # LATE in a recovery, and putting them first matches the order the eye reads the two rows.
    # Placing thumbnails on a grid rather than radially around each point is what guarantees they
    # cannot overlap; the connector, not the position, is what ties a thumbnail to its point.
    main = fig.add_axes([0.16, 0.335, 0.69, 0.295])
    sca = main.scatter(xy[:, 0], xy[:, 1], c=v, cmap="coolwarm", s=13, lw=0, alpha=0.85)
    main.set_xticks([])
    main.set_yticks([])
    for sp in main.spines.values():        # t-SNE coordinates carry no units; a frame implies they do
        sp.set_visible(False)
    fig.text(0.5, 0.978, f"{variant}: t-SNE of last-hidden-layer representations, "
             f"with decision-state screenshots", ha="center", fontsize=13)
    cax = fig.add_axes([0.885, 0.395, 0.010, 0.175])
    cb = fig.colorbar(sca, cax=cax)
    cb.set_label("state value  V = max$_e$ Q(s, e)", fontsize=11)
    cb.ax.tick_params(labelsize=10)
    cb.outline.set_linewidth(0.4)
    # No "lowest / highest state value" row captions: the connector colours and the colourbar
    # already say which end of the value range each row came from, and at these column widths the
    # captions had nowhere to sit without landing on a thumbnail title.

    # Columns nearly touch (0.216 pitch for a 0.205 width): the thumbnails are the content here,
    # and whitespace between them was only separating four drawings of the same network.
    W, H = 0.205, 0.222
    xs = [0.028, 0.244, 0.460, 0.676]
    for i, p in enumerate(pick):
        top = i >= 4                                   # first four are high-V, and go BELOW
        ins = fig.add_axes([xs[i % 4], 0.700 if top else 0.058, W, H])
        now, nrep = _draw_state(ins, dl, disset, nodes, edges, int(scen[p]), int(step[p]))
        ins.set_title(f"scen {int(scen[p])}, step {int(step[p])}\n{nrep} done, choose {now}",
                      fontsize=6, pad=2)
        # The connector carries the point's own colour and lands ON the point, with a ring there:
        # anchoring at the thumbnail centre hid the endpoint behind the network drawing, so which
        # point a screenshot belonged to had to be guessed.
        col = cmap(norm(v[p]))
        fig.add_artist(ConnectionPatch(xyA=(xy[p, 0], xy[p, 1]), coordsA=main.transData,
                                       xyB=(0.5, 0.0 if top else 1.0), coordsB=ins.transAxes,
                                       color=col, lw=0.9, alpha=0.85, zorder=1))
        main.scatter([xy[p, 0]], [xy[p, 1]], s=95, facecolors="none", edgecolors=col,
                     linewidths=1.6, zorder=5)
        main.scatter([xy[p, 0]], [xy[p, 1]], s=95, facecolors="none", edgecolors="white",
                     linewidths=0.5, zorder=6)          # thin white halo, readable over any cluster
    from matplotlib.lines import Line2D
    leg = [Line2D([0], [0], color=C["signal"], lw=2.6, label="chosen now (action)"),
           Line2D([0], [0], color=C["teal"], lw=1.8, label="repaired"),
           Line2D([0], [0], color="#9A9A9A", lw=1.8, label="still damaged"),
           Line2D([0], [0], color="#E4E4E4", lw=1.2, label="undisrupted road")]
    fig.legend(handles=leg, loc="lower center", ncol=4, fontsize=11, frameon=False,
               bbox_to_anchor=(0.5, 0.002))
    save_pub(fig, os.path.join(out_dir, f"{variant}_repr_tsne_states"))
    plt.close(fig)


def make_rank_figures(out_dir, variant, quiet=True):
    """Redraw every figure for one run from its recorded diagnostics. Safe to call mid-run: each
    figure is skipped if the record it needs is not there yet."""
    out_dir = str(out_dir)
    raw = os.path.join(out_dir, "log")
    tpath = os.path.join(raw, f"{variant}_trace.csv")
    if not os.path.exists(tpath):
        return
    use_pub()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tr = _normalize(pd.read_csv(tpath))
    if not len(tr):
        return
    ep = tr["episode"]

    traces = _seed_traces(out_dir, variant)
    fig, a = plt.subplots(figsize=(4.8, 3.2))
    _objective(a, tr, variant, traces=traces)
    a2 = a.twinx()                                        # the exploration schedule, as context
    a2.plot(ep, tr["eps"], color=C["neutral_mid"], lw=0.7, ls="--", alpha=0.6)
    a2.set_ylabel("epsilon", color=C["neutral_mid"])
    a2.tick_params(axis="y", colors=C["neutral_mid"])
    a2.spines["right"].set_visible(True)
    a2.spines["right"].set_color(C["neutral_mid"])
    a.legend(loc="upper right", fontsize=5.6, framealpha=0.88, borderpad=0.35,
             labelspacing=0.32, handlelength=1.5)
    a.set_title(f"{variant}: training objective (ranking loss)"
                + (f"   —  delivered run is the best of {len(traces)} seeds" if traces else ""))
    save_pub(fig, os.path.join(out_dir, f"{variant}_objective"))
    plt.close(fig)

    if tr["loss"].notna().any():
        fig, a = plt.subplots(figsize=(4.8, 3.0))
        a.plot(ep, tr["loss"], color=C["accent"], lw=1.1)
        a.set_xlabel("episode")
        a.set_ylabel("mean loss per update")
        a.set_title(f"{variant}: training loss")
        save_pub(fig, os.path.join(out_dir, f"{variant}_learning"))
        plt.close(fig)

    if "td_abs_mean" in tr and tr["td_abs_mean"].notna().any():
        fig, a = plt.subplots(figsize=(4.8, 3.0))
        a.plot(ep, tr["td_abs_mean"], color=C["teal"], lw=1.1)
        a.set_xlabel("episode")
        a.set_ylabel("mean |q - y| per batch sample")
        a.set_title(f"{variant}: per-sample TD error")
        save_pub(fig, os.path.join(out_dir, f"{variant}_qvalue"))
        plt.close(fig)

    # Greedy-policy stability, in the view established by the retired value-based solver's figure
    # (the old rl_viz module): panel a is the SAWTOOTH of consecutive episodes with an
    # unchanged greedy order -- every tooth one stretch of agreement, every reset one change of
    # mind -- and panel b the same information as a rolling change rate, which separates
    # "converging slowly" from "not converging". When the run's own run_meta records a stopping
    # rule, its stable_K threshold and ep_min warm-up band are drawn so the figure shows the
    # thresholds that actually governed that run; a fixed-cap run gets neither (drawing today's
    # defaults over yesterday's run would claim a rule that never applied). The change signal is
    # the recorded order itself where available, else the order's-F change proxy (blind only to
    # equal-F swaps); the annotation names which one was used.
    exact = "greedy_order" in tr and tr["greedy_order"].notna().any()
    proxy = (not exact) and "greedy_F" in tr and tr["greedy_F"].notna().any()
    if exact or proxy:
        import json
        ep_min = stable_K = None
        meta_p = os.path.join(out_dir, "config", "run_meta.json")
        if os.path.exists(meta_p):
            try:
                sp = json.load(open(meta_p, encoding="utf-8")).get("stop_params") or {}
                ep_min, stable_K = sp.get("ep_min"), sp.get("stable_K")
            except Exception:
                pass
        g = (tr["greedy_order"] if exact else tr["greedy_F"]).reset_index(drop=True)
        changed = g.ne(g.shift())
        changed.iloc[0] = False
        runlen, c = [], 0
        for ch in changed:
            c = 0 if ch else c + 1
            runlen.append(c)
        runlen = np.asarray(runlen)
        window = 15
        rate = pd.Series(changed.astype(float)).rolling(window, min_periods=1).mean().to_numpy()
        x = ep.to_numpy()

        fig, ax = plt.subplots(2, 1, figsize=(5.0, 4.2), sharex=True,
                               gridspec_kw={"height_ratios": [2, 1]})
        if ep_min:
            ax[0].axvspan(x.min(), min(ep_min, x.max()), color=C["neutral_light"], alpha=0.35,
                          lw=0, zorder=0)
        if stable_K:
            ax[0].axhline(stable_K, ls="--", lw=0.9, color=C["signal"], zorder=2)
        ax[0].step(x, runlen, where="post", lw=1.2, color=C["accent"], zorder=3)
        ax[0].set_ylabel("consecutive episodes\nwith an unchanged order")
        n_changes = int(changed.sum())
        rule = (f"stop rule: stable_K={stable_K}, ep_min={ep_min}" if stable_K
                else "fixed episode cap (no stop rule)")
        ax[0].text(0.98, 0.95, f"{n_changes} changes of mind in {len(g)} episodes\n"
                               f"longest agreement: {int(runlen.max())}\n"
                               f"signal: {'exact order' if exact else 'F-change proxy'}\n{rule}",
                   transform=ax[0].transAxes, fontsize=5.5, color=C["neutral_mid"],
                   ha="right", va="top", linespacing=1.5)
        ax[1].plot(x, rate, lw=1.3, color=C["teal"])
        ax[1].set_ylim(-0.05, 1.05)
        ax[1].set_ylabel(f"change rate\n(last {window} episodes)")
        ax[1].set_xlabel("training episode")
        panel_label(ax[0], "a")
        panel_label(ax[1], "b")
        fig.tight_layout()
        save_pub(fig, os.path.join(out_dir, f"{variant}_stability"))
        plt.close(fig)

    # Value estimate against realized performance, after van Hasselt et al. (AAAI 2016) Figure 3
    # top row: their learning curve is the value estimate during training, their straight line the
    # actual discounted value of the learned policy, and the gap between the two is the
    # over/under-estimation the figure exists to show. Adaptations: only this one method is drawn;
    # the estimate is max_e Q(s0, e) at THE initial state (every order starts from the same s0, so
    # no averaging over visited states is needed); and the realized side is the scenF probes the
    # trace already carries -- the greedy order of every 25th episode scored on the M evaluation
    # scenarios DURING the run -- converted to the same units, the episode return
    # sum_t (1 - g_t) = T(1 - F) under MU = 1. Sparse markers rather than a band: a per-episode
    # band was tried and removed, because scoring every intermediate order on the scenarios after
    # the fact is almost entirely cache-cold (measured: killed unfinished at 415 CPU-minutes).
    # Nothing here evaluates anything; both series come off the run's own records.
    # WHAT THIS FIGURE HAS TO DEFEND: under a ranking loss the value estimate stays BELOW the
    # return the policy actually earns, and keeps wandering after that return has plateaued --
    # the signature of an objective that constrains the ORDER of Q within a state and never its
    # level. The gap is therefore the subject, not a by-product of two lines that happen to differ,
    # so it is drawn as a shaded band with the two series as its edges and labelled in place.
    epath = os.path.join(raw, f"{variant}_value_est.csv")
    if os.path.exists(epath):
        ve = pd.read_csv(epath).sort_values("episode")
        T_h, q = None, None
        if "scenF" in tr and tr["scenF"].notna().any():
            meta_p = os.path.join(out_dir, "config", "run_meta.json")
            if os.path.exists(meta_p):
                try:
                    import json
                    T_h = json.load(open(meta_p, encoding="utf-8"))["instance"]["horizon_T"]
                except Exception:
                    T_h = None
            if T_h:
                q = tr.dropna(subset=["scenF"]).sort_values("episode")

        fig, a = plt.subplots(figsize=(4.6, 3.1))
        if q is not None:
            ep_r = q["episode"].to_numpy(dtype=float)
            realized = T_h * (1.0 - q["scenF"].to_numpy(dtype=float))
            ep_e = ve["episode"].to_numpy(dtype=float)
            v_e = ve["v_est"].to_numpy(dtype=float)
            # The band is built on the UNION of the two sampling grids, with both series
            # interpolated onto it. Drawing it on the probe grid alone left the estimate's own
            # vertices (it is snapshotted on a different cadence) inside a straight band segment,
            # so the red line visibly escaped the shading between probes. Both lines are drawn
            # piecewise-linear, so evaluating them on the union reproduces them exactly and the
            # band closes onto both edges.
            grid = np.union1d(ep_r, ep_e)
            grid = grid[(grid >= max(ep_r[0], ep_e[0])) & (grid <= min(ep_r[-1], ep_e[-1]))]
            a.fill_between(grid, np.interp(grid, ep_e, v_e), np.interp(grid, ep_r, realized),
                           color=C["neutral_light"], alpha=0.55, lw=0, zorder=1)
            a.plot(ep_r, realized, color=C["teal"], lw=1.6, marker="s", ms=3.2, zorder=3,
                   label="realized return, evaluation scenarios")
        a.plot(ve["episode"], ve["v_est"], color=C["signal"], lw=1.6, marker="o", ms=3.2, zorder=3,
               label="value estimate  max$_e$ Q(s$_0$, e)")
        a.set_xlabel("training episode")
        a.set_ylabel("episode return  $\\sum_t (1 - g_t)$")
        a.set_title(f"{variant}: the value estimate never catches the return it earns")
        # Legend rather than end-of-curve labels: both series converge into the top-right corner,
        # where labels collided with each other and with the title, while the lower right -- empty
        # because episode 0 starts both series at the flow order's return -- is beside the curves
        # and costs no eye travel.
        a.legend(loc="lower right", fontsize=6.5, handlelength=1.6, borderaxespad=0.4)
        a.margins(x=0.02)
        save_pub(fig, os.path.join(out_dir, f"{variant}_value_estimate"))
        plt.close(fig)

    # t-SNE of the last hidden layer, after Mnih et al. (Nature 2015) Figure 4: embed the
    # representations the network assigns to the states the DELIVERED policy experiences, one
    # point per state, colored by the predicted state value (dark red highest, dark blue lowest
    # in their rendering; a blue-to-red diverging map here). One adaptation: this Q network
    # scores (state, candidate) pairs, so "the state's representation" is the pair the greedy
    # policy actually takes -- its argmax candidate -- and V is that maximum Q. The paper's
    # reading carries over: states that are close in expected value should map near one another
    # even when they are not superficially similar (here: different scenarios, same phase of
    # the recovery).
    #
    # Only the annotated version is drawn. A bare scatter of the same embedding was produced
    # alongside it until 2026-08-11 and removed as redundant: it showed the identical points with
    # no screenshots, so it carried strictly less than the figure below.
    rpath = os.path.join(raw, f"{variant}_deliver_repr.npz")
    dpath = os.path.join(raw, f"{variant}_deliver.csv")
    if os.path.exists(rpath) and os.path.exists(dpath):
        try:
            from sklearn.manifold import TSNE
        except ImportError:
            print(f"  [{variant}] repr_tsne_states SKIPPED: scikit-learn is not installed",
                  flush=True)
        else:
            # `with`, because np.load on an .npz keeps the FILE OPEN until the archive object is
            # closed. Leaving it to the garbage collector works on POSIX and fails on Windows,
            # where the next run's fresh_scale_dir cannot unlink a file this process still holds --
            # which is exactly how a multi-seed sweep died on its fourth seed.
            with np.load(rpath) as z:
                dl = pd.read_csv(dpath)
                V = dl.groupby(["scenario", "step"])["q"].max()
                key = pd.MultiIndex.from_arrays([z["scenario"], z["step"]])
                v = V.reindex(key).to_numpy()
                # Seeded, so a redraw reproduces the same layout.
                xy = TSNE(n_components=2, perplexity=30, init="pca", learning_rate="auto",
                          random_state=0).fit_transform(np.asarray(z["repr"], dtype=np.float64))
                zz = {k: z[k] for k in ("scenario", "step")}
            _repr_tsne_screenshots(out_dir, variant, xy, v, zz, dl)

    qpath = os.path.join(raw, f"{variant}_qmc.csv")
    if os.path.exists(qpath):
        q = pd.read_csv(qpath)
        fig, a = plt.subplots(figsize=(3.6, 3.4))
        a.scatter(q["mc_return"], q["q_chosen"], s=9, alpha=0.55, color=C["accent2"],
                  edgecolors="none")
        lo = float(min(q["mc_return"].min(), q["q_chosen"].min()))
        hi = float(max(q["mc_return"].max(), q["q_chosen"].max()))
        a.plot([lo, hi], [lo, hi], color=C["neutral_mid"], lw=0.8, ls="--", label="y = x")
        r = float(np.corrcoef(q["mc_return"], q["q_chosen"])[0, 1])
        rs = float(q["mc_return"].rank().corr(q["q_chosen"].rank()))
        a.set_xlabel("Monte-Carlo return-to-go")
        a.set_ylabel("predicted Q of chosen action")
        a.set_title(f"{variant}: Q vs realized return  (r={r:.2f}, rho={rs:.2f})")
        a.legend(loc="upper left")
        save_pub(fig, os.path.join(out_dir, f"{variant}_qmc"))
        plt.close(fig)
    if not quiet:
        print(f"[{variant}] figures refreshed -> {out_dir}", flush=True)


if __name__ == "__main__":
    import sys
    make_rank_figures(sys.argv[2], sys.argv[1], quiet=False)
