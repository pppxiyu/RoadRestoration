"""
Behavior analysis of a delivered RL policy (2026-08-26, project owner's instruction): what the
agent's internal state representation is organized around, and which of its decisions actually
matter for the objective. Two studies under outputs/05-behavior/n{N}/, laid out like every
other run folder (config/ metadata, results/ recorded data, log/, figures at the top level):

  STATE EMBEDDING         Every decision state of the full M-scenario delivery, embedded with
                          the policy's own ACTION-INDEPENDENT state representation
                          [theta6*sum(mu) || theta8*g] -- the value head's input (the elementwise
                          relu keeps the state block separate from the action block; state and
                          action first mix inside the advantage MLP, so this is the cleanest
                          state summary the network owns). Recorded with a t-SNE layout and Ward
                          trajectory clustering (each trajectory = its step-ordered embeddings
                          concatenated; k by silhouette plus fixed cuts at 4 and 5). Data only:
                          its figures were removed on the project owner's instruction
                          (2026-08-26) and the embeddings / layout / cluster labels feed the
                          two studies below. Measured meaning of the axes (Spearman, n10):
                          dim 1 tracks episode progress (step -0.95), dim 2 tracks the
                          scenario's damage regime via its delivered F (+0.78).

  STATE MODES             Hierarchical Ward clustering of the SAME 500 raw 128-d embeddings
                          (states, not trajectories), cut at four nested levels k=2/4/8/16 of
                          one linkage tree, then pruned to a reportable frontier: descend from
                          a node only if every child holds >= 15 states AND is separable from
                          its siblings, inside the parent's context, by one conjunctive rule of
                          at most three thresholds with F1 >= 0.80. The description basis is
                          FAITHFUL to the network's inputs and nothing else -- the 14 node
                          feature columns each summed over all 38 vertices (the network's own
                          sum pooling, applied to raw inputs instead of learned embeddings)
                          plus the 6 global-block dimensions verbatim. No derived contrasts, no
                          candidate-set summaries, no per-segment identity indicators: the
                          clustered embedding sum-pools and never sees the candidate mask or
                          vertex identity, so describing it with those would explain the
                          clusters using information the state summary does not contain. Seven
                          of the 20 basis dimensions are constant on n10 (the four static node
                          columns sum to the same total in every state; blocked, observed
                          disconnection and disconnected demand are identically zero) and are
                          dropped from the figure, leaving 13 informative dimensions.

  COUNTERFACTUAL          One-step counterfactual deviation analysis: at every recorded
                          decision, force each legal alternative action, let the SAME policy
                          take over afterwards, score the full episode with the real evaluator.
                          Importance of a state-action pair:
                              delta_best = F(best alternative) - F(chosen)
                          positive = the value uniquely added by making exactly this choice;
                          negative = a better alternative existed (a local mistake). delta_worst
                          (same vs the worst alternative) is the state's risk exposure, q_gap
                          (Q of chosen minus second-best Q) is the agent's own belief that the
                          decision mattered. The feature table records, per pair, the policy's
                          INPUT features (node columns of the chosen segment + the global
                          block) and NOT-input features: structural (edge betweenness on the
                          intact and on the current severed-removed network, line-graph degree,
                          hops from the crew depot) and unobservable ground truth (the chosen
                          segment's true severity and duration). Data only since 2026-08-26
                          (project owner's instruction): the per-pair table and the Spearman
                          correlation table are written, the two figures that showed them are
                          not.

  ACTION PROFILE          What the agent DOES inside each state mode. For every decision and
                          every attribute of the candidate segments, the metric is the chosen
                          segment's MID-RANK PERCENTILE within the candidate set (ties
                          averaged): 1 means it held the attribute's highest value, 0 the
                          lowest, and 0.5 is what a uniformly random pick yields in expectation
                          REGARDLESS of the candidate count -- which is what makes the mean
                          comparable across modes whose candidate sets differ in size. Each
                          attribute appears once in its natural direction, so a mean above 0.5
                          reads as a preference for high values and below 0.5 for low values,
                          symmetrically. Attributes are the network's own per-segment
                          inputs: static (baseline flow, severity estimate, duration belief,
                          demand drop) and observed (live flow, congestion, realized shortfall
                          projection). Structural quantities (betweenness, line-graph degree,
                          hops from the depot) and the demand-drop-per-unit-duration heuristic
                          were carried initially and dropped on the project owner's instruction
                          (2026-08-26). The measurement behind that: across the 10 damaged
                          segments betweenness correlates 0.49 with demand drop and 0.44 with
                          baseline flow, and the composite is by construction the ratio of two
                          columns already present, so their percentiles restate other columns
                          rather than adding a dimension. Their reading was also a
                          specific-segment artifact -- the two segments the policy repairs
                          first happen to hold the lowest betweenness of the ten, which alone
                          produced the low betweenness percentiles. Reported per mode: the mean percentile and
                          the share of decisions above 0.5 -- a high mean carried by a minority
                          of decisions marks a mode that is still behaviourally mixed. Decisions
                          with fewer than three candidates are excluded from the headline (the
                          percentile is unbiased there but can only be 0 or 1) and kept in the
                          row table. Each cell carries the mean percentile and its 95%
                          SCENARIO-BLOCK bootstrap interval -- whole scenarios are resampled,
                          never individual decisions, since the ~10 decisions of one scenario
                          share its damage draw. The FIGURE prints the mean and a star for
                          cells whose mean differs from 0.5 after Benjamini-Hochberg correction
                          at FDR 5% over all cells; the intervals and p-values behind the star
                          are in results/action_percentile_stats.csv. Modes are
                          ordered EARLIER STAGE FIRST, more damage first within a stage band
                          (mode_order, shared with the state-mode figures). CAVEAT: this
                          describes what the choices LOOK like, not which attribute drives them
                          -- the attributes are collinear and often point at the same segment.

Findings recorded from the first run (n10, rl_s2v_saa64_adaptive, seed-42 delivery) live in
config/run_meta.json next to the data; headline: importance concentrates LATE in the episode
(early deviations are healed by the adaptive remainder), edge betweenness -- which the policy
never receives -- is the strongest non-temporal correlate of importance, the agent's own q_gap
is only weakly calibrated to measured importance (rho 0.13), and the OD-disconnection
observation columns were constant zero across the whole n10 delivery (severing never actually
disconnected an OD pair on Sioux Falls at n10).

USAGE. python -m util.behavior  regenerates figures from the recorded data (cheap, no UE);
run_behavior(recompute=True) recomputes everything from the delivered checkpoint (the
counterfactual pass costs ~2250 policy rollouts + evaluations, ~45 min at n10). All heavy
artifacts (embeddings, per-pair table) are kept on disk precisely so recomputation is never
needed to redraw or re-analyze.
"""
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import config as P
from util.rl import _evaluate_prefix_cached
from util.rl_rank import build_env
from util.rl_s2v import (_build_s2v_net, _graph_tensors, _resolve_variant, _s2v_rollout,
                         _state_x)
from util.oracle import scale_dir
from util.provenance import results_dir, solver_dir
from viz.style import C, save_pub, use_pub

ROOT = Path(__file__).resolve().parent.parent
OUT_BEH = ROOT / "outputs" / "05-behavior"
_RC = {"font.size": 19, "axes.labelsize": 22, "xtick.labelsize": 18, "ytick.labelsize": 18,
       "legend.fontsize": 17}

_FEATS = [
    ("phi", "baseline flow of chosen segment", "input: node"),
    ("sev_est", "severity estimate (chosen)", "input: node"),
    ("dur_belief", "duration belief (chosen)", "input: node"),
    ("dem_drop", "demand drop (chosen)", "input: node"),
    ("proj_belief", "belief shortfall projection (chosen)", "input: node"),
    ("obs_flow", "observed live flow (chosen)", "input: node"),
    ("obs_cong", "observed congestion (chosen)", "input: node"),
    ("obs_disc", "observed disconnection share (chosen)", "input: node"),
    ("obs_trueD", "observed realized shortfall (chosen)", "input: node"),
    ("t_frac", "episode time t/T", "input: global"),
    ("crew_gap", "crew busy gap", "input: global"),
    ("bel_shortfall", "belief total shortfall", "input: global"),
    ("rem_work", "remaining work fraction", "input: global"),
    ("true_shortfall", "observed total shortfall", "input: global"),
    ("disc_demand", "observed disconnected demand", "input: global"),
    ("step", "decision step index", "input: global"),
    ("bet_static", "edge betweenness, intact network", "NOT input: structural"),
    ("bet_current", "edge betweenness, current network", "NOT input: structural"),
    ("lg_degree", "line-graph degree (chosen)", "NOT input: structural"),
    ("hops_depot", "hops from crew depot (chosen)", "NOT input: structural"),
    ("true_sev", "TRUE severity (chosen, unobservable)", "NOT input: ground truth"),
    ("true_dur", "TRUE duration (chosen, unobservable)", "NOT input: ground truth"),
]
# The faithful description basis for the state-mode study: node feature columns summed over all
# 38 vertices (the network's own pooling), then the global block verbatim. Column indices follow
# util.rl_s2v's x layout; the labels are what the figures print.
_NODE_SUM = [(0, "sum of pending tags"), (1, "sum of under-repair tags"),
             (2, "sum of done tags"), (3, "sum of baseline flow"),
             (4, "sum of severity estimate"), (5, "sum of duration belief"),
             (6, "sum of demand drop"), (7, "sum of belief shortfall projection"),
             (8, "sum of completion recency"), (9, "sum of blocked flags"),
             (10, "sum of observed live flow"), (11, "sum of observed congestion"),
             (12, "sum of observed disconnection share"),
             (13, "sum of observed realized shortfall")]
_GLOB = [(0, "global: elapsed horizon t/T"), (1, "global: crew busy gap"),
         (2, "global: belief total shortfall"), (3, "global: remaining work fraction"),
         (4, "global: realized total shortfall"), (5, "global: disconnected demand")]
# DISPLAY ONLY: the same names wrapped for upright tick labels. The keys above stay the column
# names in state_modes.csv, so the recorded tables and mode_order's lookups are unaffected --
# only what the figure prints changes.
_MODE_FIG_LABEL = {
    "sum of pending tags": "sum:\npending\ntags",
    "sum of under-repair tags": "sum:\nunder-repair\ntags",
    "sum of done tags": "sum:\ndone\ntags",
    "sum of baseline flow": "sum:\nbaseline\nflow",
    "sum of severity estimate": "sum:\nseverity\nestimate",
    "sum of duration belief": "sum:\nduration\nbelief",
    "sum of demand drop": "sum:\ndemand\ndrop",
    "sum of belief shortfall projection": "sum: belief\nshortfall\nprojection",
    "sum of completion recency": "sum:\ncompletion\nrecency",
    "sum of blocked flags": "sum:\nblocked\nflags",
    "sum of observed live flow": "sum:\nobserved\nlive flow",
    "sum of observed congestion": "sum:\nobserved\ncongestion",
    "sum of observed disconnection share": "sum: observed\ndisconnection\nshare",
    "sum of observed realized shortfall": "sum: observed\nrealized\nshortfall",
    "global: elapsed horizon t/T": "global:\nelapsed\nhorizon t/T",
    "global: crew busy gap": "global:\ncrew\nbusy gap",
    "global: belief total shortfall": "global:\nbelief total\nshortfall",
    "global: remaining work fraction": "global:\nremaining\nwork fraction",
    "global: realized total shortfall": "global:\nrealized total\nshortfall",
    "global: disconnected demand": "global:\ndisconnected\ndemand",
}
# The action-profile attribute dictionary: (key, printed label). Natural direction only --
# no mirrored shortest/longest entries, since the percentile already reads symmetrically about
# 0.5. Structural entries are NOT network inputs and are labelled as such in the study docstring.
# Labels carry a newline where they would otherwise crowd the axis: the tick text is what a
# reader sees, so it wraps rather than shrinks.
_ACT_ATTRS = [("phi", "baseline\nflow"),
              ("sev_est", "severity\nestimate"),
              ("dur", "duration\nbelief"),
              ("dem_drop", "demand\ndrop"),
              ("obs_flow", "observed\nlive\nflow"),
              ("obs_cong", "observed\ncongestion"),
              ("obs_trueD", "observed\nrealized\nshortfall")]
_ACT_MIN_CAND = 3         # below this the percentile can only be 0 or 1: reported, not headlined
_ACT_BOOT = 2000          # scenario-block bootstrap resamples behind every reported interval
_ACT_STAGE_BAND = 1.0     # modes within this many decision steps count as the same stage band

_MODE_LEVELS = ["L1", "L2", "L3", "L4"]
_MODE_KS = [2, 4, 8, 16]
_MODE_MIN_N = 15          # a child too small to describe honestly stops the descent
_MODE_MIN_F1 = 0.80       # sibling-context describability gate (see the module docstring)


def _dirs(N):
    """The study's on-disk layout, matching every other run folder in outputs/: config/ for
    metadata, results/ for the recorded data (embeddings, per-pair table -- the expensive
    artifacts that make recomputation unnecessary), log/ for traces, figures directly under
    n{N}/."""
    base = OUT_BEH / f"n{int(N)}"
    d = dict(base=base, res=base / "results", cfg=base / "config", log=base / "log")
    for p in d.values():
        p.mkdir(parents=True, exist_ok=True)
    return d


def _load_policy(variant, N):
    """The delivered checkpoint of `variant` at scale N, rebuilt under the SAME resolved
    configuration it trained with, plus the greedy Q evaluator. Returns (env, gt, hp, qvec)."""
    import torch
    import torch.nn as nn
    torch.set_num_threads(1)
    hp = _resolve_variant(variant)
    env = build_env(N=N)
    gt = _graph_tensors(env, hp=hp)
    net = _build_s2v_net(int(hp["p"]), int(hp["t_emb"]), bool(hp["use_g"]), torch, nn,
                         dueling=bool(hp["dueling"]), readout_hidden=int(hp["readout_hidden"]),
                         in_dim=gt["n_feat"], hop_untied=bool(hp["hop_untied"]),
                         g_dim=gt["g_dim"])
    model = results_dir(scale_dir(ROOT / "outputs" / "03-rl" / solver_dir(variant), N)) \
        / "model_best.pt"
    net.load_state_dict(torch.load(model, weights_only=True), strict=True)
    net.eval()
    A = torch.tensor(gt["A"])
    deg = torch.tensor(gt["deg"], dtype=torch.float32)

    def qvec(state):
        with torch.no_grad():
            x = torch.tensor(_state_x(gt, state))
            idx = [gt["idx"][e] for e in state["cand"]]
            out = net(x, A, deg, torch.tensor(state["g"]))
            if not bool(hp["dueling"]):
                return out[idx].numpy()
            V, adv = out
            a = adv[idx]
            return (V + a - a.mean()).numpy()

    def semb(state):
        """The action-independent state embedding [th6*sum(mu) || th8*g] (module docstring)."""
        import torch as _t
        with _t.no_grad():
            x = _t.tensor(_state_x(gt, state))
            pre = net.th1(x) + net.th3(deg.unsqueeze(1) * _t.relu(net.th4).unsqueeze(0))
            mu = _t.zeros(x.shape[0], int(hp["p"]))
            for t in range(int(hp["t_emb"])):
                th2 = net.th2[t] if net.hop_untied else net.th2
                mu = _t.relu(pre + th2(A @ mu))
            return _t.cat([net.th6(mu.sum(dim=0)),
                           net.th8(_t.tensor(state["g"]))]).numpy()

    return env, gt, hp, qvec, semb, str(model)


# --------------------------------------------------------------------------- #
# 01  state embeddings: collect, t-SNE, trajectory clustering
# --------------------------------------------------------------------------- #
def collect_embeddings(variant, N, d):
    """Roll the delivered policy on every frozen scenario, embed every decision state, lay the
    cloud out with t-SNE and cluster the per-scenario trajectories (Ward on step-ordered
    concatenated embeddings; k by silhouette over 2..6, plus fixed cuts k=4 and k=5)."""
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.manifold import TSNE
    from sklearn.metrics import silhouette_score
    env, gt, hp, qvec, semb, model = _load_policy(variant, N)
    embs, meta = [], []
    hold = dict(m=None)

    def pick(rem, state):
        q = qvec(state)
        embs.append(semb(state))
        meta.append(dict(scenario=hold["m"],
                         step=sum(1 for r in meta if r["scenario"] == hold["m"]),
                         t_slot=float(state["g"][0]) * env["T"],
                         n_done=len(state["done"])))
        return int(np.argmax(q))

    for m, dur in enumerate(env["scen"]):
        hold["m"] = m
        _s2v_rollout(env, gt, pick, dur)
    X = np.stack(embs)
    md = pd.DataFrame(meta)
    Z = TSNE(n_components=2, random_state=0, perplexity=30, init="pca").fit_transform(X)
    md["tsne_x"], md["tsne_y"] = Z[:, 0], Z[:, 1]

    steps = md.groupby("scenario").step.max().iloc[0] + 1
    order = md.sort_values(["scenario", "step"]).index.to_numpy()
    traj = X[order].reshape(len(env["scen"]), -1)
    sil = {k: float(silhouette_score(
        traj, AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(traj)))
        for k in range(2, 7)}
    for tag, k in [("cluster", max(sil, key=sil.get)), ("cluster_k4", 4), ("cluster_k5", 5)]:
        lab = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(traj)
        sizes = pd.Series(lab).value_counts()
        remap = {old: new for new, old in enumerate(sizes.index)}
        md[tag] = md.scenario.map({m: remap[lab[m]] for m in range(len(traj))})
    np.save(d["res"] / "state_embeddings.npy", X)
    md.to_csv(d["res"] / "state_embeddings_meta.csv", index=False)
    json.dump(dict(model=model, embedding="[th6*sum(mu) || th8*g] (value-head input)",
                   n_states=int(len(md)), steps_per_scenario=int(steps), silhouette=sil,
                   tsne=dict(perplexity=30, random_state=0, init="pca")),
              open(d["cfg"] / "embedding_meta.json", "w", encoding="utf-8"), indent=1)
    return md


# --------------------------------------------------------------------------- #
# 02  state modes: hierarchical clustering + describability pruning
# --------------------------------------------------------------------------- #
def collect_state_modes(variant, N, d):
    """Replay the delivery to record the faithful description basis per state, cluster the
    recorded embeddings hierarchically, and prune to the reportable mode frontier. Writes
    state_modes.csv (one row per state: basis columns, candidate set, chosen action, the four
    nested cluster labels, the assigned mode) and modes_table.csv (one row per mode: size, step
    span, scenario count, and its sibling-context rule)."""
    from scipy.cluster.hierarchy import fcluster, linkage
    from sklearn.tree import DecisionTreeClassifier
    X = np.load(d["res"] / "state_embeddings.npy")
    base = pd.read_csv(d["res"] / "state_embeddings_meta.csv")[
        ["scenario", "step", "tsne_x", "tsne_y"]]
    env, gt, hp, qvec, semb, model = _load_policy(variant, N)
    rows, hold = [], dict(m=None)

    def pick(rem, state):
        x, g = _state_x(gt, state), state["g"]
        r = dict(scenario=hold["m"], step=sum(1 for z in rows if z["scenario"] == hold["m"]),
                 cand="-".join(map(str, state["cand"])))
        for ci, name in _NODE_SUM:
            r[name] = float(x[:, ci].sum())
        for gi, name in _GLOB:
            r[name] = float(g[gi])
        i = int(np.argmax(qvec(state)))
        r["chosen"] = state["cand"][i]
        rows.append(r)
        return i

    for m, dur in enumerate(env["scen"]):
        hold["m"] = m
        _s2v_rollout(env, gt, pick, dur)
    md = pd.DataFrame(rows).merge(base, on=["scenario", "step"], validate="one_to_one")
    VARS = [n for _, n in _NODE_SUM] + [n for _, n in _GLOB]
    F = md[VARS].to_numpy()

    Z = linkage(X, method="ward")
    lab_of = {}
    for li, k in enumerate(_MODE_KS):
        raw = fcluster(Z, t=k, criterion="maxclust")
        lv = np.empty(len(X), dtype=object)
        if li == 0:                                    # level 1 named by size: A biggest
            letters = {c: chr(ord("A") + i)
                       for i, c in enumerate(pd.Series(raw).value_counts().index)}
            for i in range(len(X)):
                lv[i] = letters[raw[i]]
        else:                                          # deeper levels extend the parent name
            prev, seen = lab_of[_MODE_LEVELS[li - 1]], {}
            for c in np.unique(raw):
                mask = raw == c
                parent = pd.Series(prev[mask]).mode()[0]
                assert (prev[mask] == parent).all(), "cuts of one linkage tree must nest"
                seen[parent] = seen.get(parent, 0) + 1
                lv[mask] = parent + (str(seen[parent]) if li % 2 == 1
                                     else chr(ord("a") + seen[parent] - 1))
        lab_of[_MODE_LEVELS[li]] = lv
        md[_MODE_LEVELS[li]] = lv

    def rule_for(mask, ctx):
        """Best conjunctive rule (<=3 thresholds) telling the cluster from its siblings, fitted
        on the parent-context rows only; precision/recall are measured in that context."""
        Fc, yc = F[ctx], mask[ctx].astype(int)
        if yc.min() == yc.max():
            return "", 1.0, 1.0
        t = DecisionTreeClassifier(max_depth=3, min_samples_leaf=8, class_weight="balanced",
                                   random_state=0).fit(Fc, yc)
        tr, best, stack = t.tree_, (None, 0.0, 0.0, -1.0), [(0, [])]
        while stack:
            node, conds = stack.pop()
            if tr.children_left[node] == -1:
                sel = np.ones(len(Fc), dtype=bool)
                for f, thr, le in conds:
                    sel &= (Fc[:, f] <= thr) if le else (Fc[:, f] > thr)
                tp = int((sel & mask[ctx]).sum())
                if tp == 0:
                    continue
                prec, rec = tp / sel.sum(), tp / mask[ctx].sum()
                f1 = 2 * prec * rec / (prec + rec)
                if f1 > best[3]:
                    best = (conds, prec, rec, f1)
                continue
            f, thr = tr.feature[node], tr.threshold[node]
            stack.append((tr.children_left[node], conds + [(f, thr, True)]))
            stack.append((tr.children_right[node], conds + [(f, thr, False)]))
        conds, prec, rec, _ = best
        tight = {}
        for f, thr, le in conds:                       # keep the tightest bound per direction
            key = (f, le)
            if key not in tight or ((thr < tight[key]) if le else (thr > tight[key])):
                tight[key] = thr
        return (" AND ".join("[" + VARS[f] + "] " + ("<=" if le else ">") + f" {thr:.3f}"
                             for (f, le), thr in tight.items()), prec, rec)

    def frontier(label, li):
        if li + 1 >= len(_MODE_LEVELS):
            return [(label, li)]
        ch = sorted(md.loc[md[_MODE_LEVELS[li]] == label, _MODE_LEVELS[li + 1]].unique())
        if len(ch) == 1:                               # a level that does not split is skipped
            return frontier(ch[0], li + 1)
        ctx = (md[_MODE_LEVELS[li]] == label).to_numpy()
        for c in ch:
            m = (md[_MODE_LEVELS[li + 1]] == c).to_numpy()
            if m.sum() < _MODE_MIN_N:
                return [(label, li)]
            _, p_, r_ = rule_for(m, ctx)
            if 2 * p_ * r_ / max(p_ + r_, 1e-9) < _MODE_MIN_F1:
                return [(label, li)]
        out = []
        for c in ch:
            out += frontier(c, li + 1)
        return out

    modes = []
    for top in sorted(md.L1.unique()):
        modes += frontier(top, 0)
    md["mode"] = ""
    recs = []
    for label, li in modes:
        m = (md[_MODE_LEVELS[li]] == label).to_numpy()
        md.loc[m, "mode"] = label
        parent, ctx = "-", np.ones(len(md), dtype=bool)
        for up in range(li - 1, -1, -1):               # nearest ancestor strictly larger
            anc = label[:len(label) - (li - up)]
            cm = (md[_MODE_LEVELS[up]] == anc).to_numpy()
            if cm.sum() > m.sum():
                parent, ctx = anc, cm
                break
        txt, p_, r_ = rule_for(m, ctx)
        grp = md[m]
        recs.append(dict(mode=label, parent=parent, n=int(m.sum()),
                         steps=f"{grp.step.min()}-{grp.step.max()}",
                         scenarios=int(grp.scenario.nunique()), rule_within_parent=txt,
                         precision=round(p_, 2), recall=round(r_, 2)))
    assert (md["mode"] != "").all()
    md.to_csv(d["res"] / "state_modes.csv", index=False)
    order = [lbl for lbl, _ in modes]                  # tree order, left to right
    pd.DataFrame(recs).set_index("mode").loc[order].reset_index().to_csv(
        d["res"] / "modes_table.csv", index=False)
    return md


def mode_order(d):
    """Reading order for the modes: EARLIER STAGE FIRST, and among modes at the same stage,
    MORE DAMAGE FIRST (project owner's instruction, 2026-08-26). Stage is the mean decision step
    of the mode and damage its mean realized total shortfall; modes whose mean step differs by
    less than _ACT_STAGE_BAND form one stage band and are ordered by damage inside it. Both
    figures read this one function, so they can never disagree about the row order."""
    md = pd.read_csv(d["res"] / "state_modes.csv")
    stat = (md.groupby("mode")
              .agg(stage=("step", "mean"),
                   damage=("global: realized total shortfall", "mean"))
              .sort_values("stage"))
    out, band = [], []
    for name, r in stat.iterrows():
        if band and r.stage - stat.loc[band[0], "stage"] >= _ACT_STAGE_BAND:
            out += sorted(band, key=lambda m: -stat.loc[m, "damage"])
            band = []
        band.append(name)
    out += sorted(band, key=lambda m: -stat.loc[m, "damage"])
    return out


def make_state_mode_figures(d):
    """Redraw the two state-mode figures from state_modes.csv (no model, no UE)."""
    md = pd.read_csv(d["res"] / "state_modes.csv")
    order = mode_order(d)
    VARS = [n for _, n in _NODE_SUM] + [n for _, n in _GLOB]
    use_pub(slide=True)
    plt.rcParams.update(_RC)

    cmap = plt.get_cmap("tab10")
    cols = {mo: cmap(i % 10) for i, mo in enumerate(order)}
    fig, ax = plt.subplots(figsize=(12.5, 9.5))
    for mo in order:
        g = md[md["mode"] == mo]
        ax.scatter(g.tsne_x, g.tsne_y, s=40, color=cols[mo], lw=0, alpha=0.9,
                   label=f"{mo}  ({len(g)} states)")
    ax.set_xlabel("t-SNE dimension 1")
    ax.set_ylabel("t-SNE dimension 2")
    ax.legend(loc="lower left", frameon=False)
    fig.tight_layout()
    save_pub(fig, d["base"] / "state_modes")
    plt.close(fig)

    mu, sd = md[VARS].mean(), md[VARS].std()
    keep = [v for v in VARS if sd[v] > 1e-12]          # constant inputs carry no contrast
    prof = pd.DataFrame([pd.Series((md[md["mode"] == mo][keep].mean() - mu[keep]) / sd[keep],
                                   name=f"{mo}  ({int((md['mode'] == mo).sum())} states)")
                         for mo in order])
    prof.to_csv(d["res"] / "mode_profiles_z.csv")
    fig, ax = plt.subplots(figsize=(23.0, 9.0))
    im = ax.imshow(np.clip(prof.to_numpy(), -2.5, 2.5), cmap="RdBu_r", vmin=-2.5, vmax=2.5,
                   aspect="auto")
    ax.set_xticks(range(len(keep)))
    ax.set_xticklabels([_MODE_FIG_LABEL[v] for v in keep], rotation=0, ha="center",
                       fontsize=17)
    ax.set_yticks(range(len(prof)))
    ax.set_yticklabels(prof.index, fontsize=20)
    ax.axvline(sum(1 for v in keep if v.startswith("sum")) - 0.5, color=C["neutral_dark"],
               lw=1.4, ls=":")
    cb = fig.colorbar(im, ax=ax, shrink=0.85)
    cb.set_label("mode mean, z-score vs all states", fontsize=20)
    cb.ax.tick_params(labelsize=18)
    fig.tight_layout()
    save_pub(fig, d["base"] / "mode_profiles")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 03  one-step counterfactual importance
# --------------------------------------------------------------------------- #
def run_counterfactual(variant, N, d):
    """The full deviation pass (module docstring). EXPENSIVE: (M * ~sum_k(cand_k - 1)) policy
    rollouts, each scored by the real evaluator (prefix-cached)."""
    t0 = time.time()
    env, gt, hp, qvec, semb, model = _load_policy(variant, N)
    ctx = env["ctx"]
    G0 = nx.Graph()
    for r in ctx["edges"].itertuples(index=False):
        G0.add_edge(int(r.u), int(r.v), eid=int(r.edge_id), weight=float(r.free_flow_time))
    eb0 = nx.edge_betweenness_centrality(G0, weight="weight")
    bet_static = {dd["eid"]: eb0.get((u, v), eb0.get((v, u)))
                  for u, v, dd in G0.edges(data=True)}
    hops = nx.single_source_shortest_path_length(G0, P.ACCESS_DEPOT)
    hops_depot = {e: min(hops[a], hops[b]) for e, (a, b) in ctx["access"]["ends"].items()}

    def bet_current(sev_true, damaged_now):
        cut = {e for e in damaged_now if sev_true[e] >= P.SEVER_SEVERITY}
        G = nx.Graph()
        for u, v, dd in G0.edges(data=True):
            if dd["eid"] not in cut:
                G.add_edge(u, v, eid=dd["eid"], weight=dd["weight"])
        eb = nx.edge_betweenness_centrality(G, weight="weight")
        return {dd["eid"]: eb.get((u, v), eb.get((v, u))) for u, v, dd in G.edges(data=True)}

    sc, rows = {}, []
    for m, dur in enumerate(env["scen"]):
        sev_true = {int(e): int(v) for e, v in dur.sev.items()}
        rec = []

        def pick_greedy(rem, state):
            q = qvec(state)
            rec.append((state, list(state["cand"]), q))
            return int(np.argmax(q))

        perm, start, _, _ = _s2v_rollout(env, gt, pick_greedy, dur)
        F_base = _evaluate_prefix_cached(start, dur, env["T"], env["ctx"], sc)["F"]
        for k, (state_k, cand_k, q_k) in enumerate(rec):
            chosen = cand_k[int(np.argmax(q_k))]
            bc = bet_current(sev_true, set(state_k["pending"]) | set(state_k["under"]))
            alts = {}
            for aprime in cand_k:
                if aprime == chosen:
                    continue
                cnt = dict(i=0)

                def pick_cf(rem, state, _k=k, _a=aprime, _c=cnt):
                    i = _c["i"]
                    _c["i"] += 1
                    return (state["cand"].index(_a) if i == _k
                            else int(np.argmax(qvec(state))))

                _, s2, _, _ = _s2v_rollout(env, gt, pick_cf, dur)
                alts[aprime] = _evaluate_prefix_cached(s2, dur, env["T"], env["ctx"], sc)["F"]
            if not alts:
                continue
            x_k = _state_x(gt, state_k)
            i_ch = gt["idx"][chosen]
            g = state_k["g"]
            rows.append(dict(
                scenario=m, step=k, chosen=chosen, n_alt=len(alts), F_base=F_base,
                F_alt_best=min(alts.values()), F_alt_worst=max(alts.values()),
                delta_best=min(alts.values()) - F_base,
                delta_worst=max(alts.values()) - F_base,
                q_gap=(float(np.sort(q_k)[-1] - np.sort(q_k)[-2])
                       if len(q_k) > 1 else np.nan),
                phi=float(x_k[i_ch, 3]), sev_est=float(x_k[i_ch, 4]),
                dur_belief=float(x_k[i_ch, 5]), dem_drop=float(x_k[i_ch, 6]),
                proj_belief=float(x_k[i_ch, 7]), obs_flow=float(x_k[i_ch, 10]),
                obs_cong=float(x_k[i_ch, 11]), obs_disc=float(x_k[i_ch, 12]),
                obs_trueD=float(x_k[i_ch, 13]),
                t_frac=float(g[0]), crew_gap=float(g[1]), bel_shortfall=float(g[2]),
                rem_work=float(g[3]), true_shortfall=float(g[4]), disc_demand=float(g[5]),
                bet_static=bet_static[chosen], bet_current=bc.get(chosen, 0.0),
                lg_degree=float(gt["deg"][i_ch]), hops_depot=hops_depot[chosen],
                true_sev=sev_true[chosen], true_dur=int(dur[chosen])))
    df = pd.DataFrame(rows)
    df.to_csv(d["res"] / "counterfactual_pairs.csv", index=False)
    json.dump(dict(model=model, n_pairs=len(df), obs_solves=int(gt["obs"]["solves"][0]),
                   minutes=round((time.time() - t0) / 60, 1)),
              open(d["cfg"] / "counterfactual_meta.json", "w", encoding="utf-8"), indent=1)
    return df


def write_counterfactual_correlations(d):
    """Spearman correlation of every recorded feature with the counterfactual importance
    delta_best, written to results/. Figures were removed on the project owner's instruction
    (2026-08-26); this keeps the table they were drawn from."""
    from scipy import stats as st
    df = pd.read_csv(d["res"] / "counterfactual_pairs.csv")
    rows = []
    for col, label, cat in _FEATS:
        rho, p = ((np.nan, np.nan) if df[col].nunique() <= 1
                  else st.spearmanr(df[col], df.delta_best))
        rows.append(dict(col=col, label=label, cat=cat, rho=rho, p=p))
    cor = pd.DataFrame(rows).sort_values("rho")
    cor.to_csv(d["res"] / "feature_importance_correlations.csv", index=False)
    return cor


# --------------------------------------------------------------------------- #
# 04  action profile: what the agent does inside each state mode
# --------------------------------------------------------------------------- #
def _mid_rank_pct(values, chosen_val):
    """Mid-rank percentile of chosen_val within values, ties averaged, mapped to [0, 1]. NaN for
    a single candidate (no ranking exists)."""
    v = np.asarray(values, dtype=float)
    if len(v) < 2:
        return np.nan
    below = int((v < chosen_val).sum())
    equal = int((v == chosen_val).sum())
    return (below + (equal - 1) / 2.0) / (len(v) - 1)


def collect_action_profile(variant, N, d):
    """Replay the delivery and record, per decision, the chosen segment's percentile within the
    candidate set on every attribute of the dictionary. Writes action_percentiles.csv (one row
    per decision) joined to the state modes."""
    env, gt, hp, qvec, semb, model = _load_policy(variant, N)
    rows, hold = [], dict(m=None)

    def pick(rem, state):
        x = _state_x(gt, state)
        cand = list(state["cand"])
        i = int(np.argmax(qvec(state)))
        chosen = cand[i]
        vals = {}
        for e in cand:
            row = x[gt["idx"][e]]
            dur = float(row[5]) or 1e-9
            vals[e] = dict(phi=float(row[3]), sev_est=float(row[4]), dur=dur,
                           dem_drop=float(row[6]), obs_flow=float(row[10]),
                           obs_cong=float(row[11]), obs_trueD=float(row[13]))
        r = dict(scenario=hold["m"],
                 step=sum(1 for z in rows if z["scenario"] == hold["m"]),
                 n_cand=len(cand), chosen=chosen)
        for key, _lab in _ACT_ATTRS:
            r[f"pct_{key}"] = _mid_rank_pct([vals[e][key] for e in cand], vals[chosen][key])
        rows.append(r)
        return i

    for m, dur in enumerate(env["scen"]):
        hold["m"] = m
        _s2v_rollout(env, gt, pick, dur)
    modes = pd.read_csv(d["res"] / "state_modes.csv")[["scenario", "step", "mode"]]
    df = pd.DataFrame(rows).merge(modes, on=["scenario", "step"], validate="one_to_one")
    df.to_csv(d["res"] / "action_percentiles.csv", index=False)
    return df


def _boot_stats(sub, key, seed=0):
    """Mean percentile with a 95% interval and a two-sided p-value against the random-pick
    reference 0.5, by SCENARIO-BLOCK bootstrap: whole scenarios are resampled with replacement,
    never individual decisions, because the ~10 decisions of one scenario share its damage draw
    and are not independent draws. The p-value is the usual bootstrap two-sided tail mass around
    0.5."""
    v = sub[["scenario", f"pct_{key}"]].dropna()
    if len(v) < 2:
        return np.nan, np.nan, np.nan, np.nan
    scen = v.scenario.unique()
    by = {sc: v.loc[v.scenario == sc, f"pct_{key}"].to_numpy() for sc in scen}
    rng = np.random.RandomState(seed)
    draws = np.array([np.concatenate([by[sc] for sc in
                                      rng.choice(scen, size=len(scen), replace=True)]).mean()
                      for _ in range(_ACT_BOOT)])
    lo, hi = np.percentile(draws, [2.5, 97.5])
    tail = min((draws <= 0.5).mean(), (draws >= 0.5).mean())
    return float(v[f"pct_{key}"].mean()), float(lo), float(hi), float(min(1.0, 2 * tail))


def _bh_reject(pvals, q=0.05):
    """Benjamini-Hochberg: which of the p-values survive at false-discovery rate q. Needed
    because the figure runs one test per (group, attribute) cell -- 72 of them here -- and at
    q=0.05 uncorrected roughly four would look significant by chance alone."""
    p = np.asarray(pvals, dtype=float)
    ok = np.isfinite(p)
    out = np.zeros(len(p), dtype=bool)
    idx = np.where(ok)[0][np.argsort(p[ok])]
    m = len(idx)
    passed = [i for r, i in enumerate(idx, start=1) if p[i] <= q * r / m]
    if passed:
        cut = max(p[i] for i in passed)
        out[ok] = p[ok] <= cut
    return out


def make_action_profile_figure(d):
    """Redraw the action-profile figure from action_percentiles.csv (no model, no UE), with the
    bootstrap statistics recomputed alongside."""
    df = pd.read_csv(d["res"] / "action_percentiles.csv")
    order = mode_order(d)
    big = df[df.n_cand >= _ACT_MIN_CAND]
    groups = [("ALL modes", big)] + [(mo, big[big["mode"] == mo]) for mo in order]
    labs = [lab for _, lab in _ACT_ATTRS]

    recs = []
    for name, sub in groups:
        for k, lab in _ACT_ATTRS:
            mean, lo, hi, pv = _boot_stats(sub, k)
            recs.append(dict(group=name, attribute=lab, n_decisions=len(sub), mean_pct=mean,
                             ci_low=lo, ci_high=hi, p_value=pv,
                             share_above_half=float((sub[f"pct_{k}"] > 0.5).mean())))
    stats = pd.DataFrame(recs)
    stats["significant_fdr05"] = _bh_reject(stats.p_value.to_numpy())
    stats.to_csv(d["res"] / "action_percentile_stats.csv", index=False)
    piv = stats.pivot(index="group", columns="attribute")
    for col, fname in [("mean_pct", "action_percentile_mean.csv"),
                       ("share_above_half", "action_percentile_share_above_half.csv")]:
        piv[col].loc[[n for n, _ in groups], labs].to_csv(d["res"] / fname)

    use_pub(slide=True)
    plt.rcParams.update(dict(_RC, **{"font.size": 20, "xtick.labelsize": 19,
                                     "ytick.labelsize": 20}))
    rows = [n for n, _ in groups]
    M = piv["mean_pct"].loc[rows, labs].to_numpy()
    SIG = piv["significant_fdr05"].loc[rows, labs].to_numpy()
    fig, ax = plt.subplots(figsize=(17.5, 8.0))
    im = ax.imshow(M, cmap="RdBu_r", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(labs)))
    ax.set_xticklabels(labs, rotation=0, ha="center")     # wrapped labels fit upright
    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels([f"{n}  ({len(sub)} decisions)" for n, sub in groups])
    ax.axhline(0.5, color=C["neutral_dark"], lw=2.0)      # pooled row above the per-mode rows
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            star = "*" if SIG[i, j] else ""
            ax.text(j, i, f"{M[i, j]:.2f}{star}", ha="center", va="center", fontsize=19.0,
                    fontweight="bold" if SIG[i, j] else "normal",
                    color="white" if abs(M[i, j] - 0.5) > 0.28 else C["neutral_dark"])
    cb = fig.colorbar(im, ax=ax, shrink=0.85)
    cb.set_label("mean percentile of the chosen segment\n(0.5 = a random pick; "
                 "* = differs from 0.5 at FDR 5%)", fontsize=19)
    cb.ax.tick_params(labelsize=18)
    fig.tight_layout()
    save_pub(fig, d["base"] / "action_profile")
    plt.close(fig)


# --------------------------------------------------------------------------- #
def run_behavior(variant="rl_s2v_saa64_adaptive", N=None, recompute=False):
    """Regenerate the behavior study for `variant` at scale N. With recompute=False (default)
    the recorded data on disk is reused and only the figures are redrawn; recompute=True
    replays the delivery and reruns the counterfactual pass from the checkpoint."""
    N = P.N_DISRUPTED_ORACLE if N is None else int(N)
    d = _dirs(N)
    if recompute or not (d["res"] / "state_embeddings_meta.csv").exists():
        collect_embeddings(variant, N, d)
    if recompute or not (d["res"] / "state_modes.csv").exists():
        collect_state_modes(variant, N, d)
    if recompute or not (d["res"] / "counterfactual_pairs.csv").exists():
        run_counterfactual(variant, N, d)
    if recompute or not (d["res"] / "action_percentiles.csv").exists():
        collect_action_profile(variant, N, d)
    make_state_mode_figures(d)
    make_action_profile_figure(d)
    write_counterfactual_correlations(d)
    print(f"behavior study written to {d['base']}")
    return d["base"]


if __name__ == "__main__":
    run_behavior()
