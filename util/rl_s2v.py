"""EXPERIMENTAL (2026-08-23): S2V-DQN (Dai et al., NIPS 2017 / arXiv 1704.01665, structure2vec
embedding) trained WITH the repo's large-margin ranking hinge, as a solver PARALLEL to
util.rl_rank and kept deliberately removable.

WHAT THIS IS. Dai et al.'s structure2vec graph embedding parameterizing Q, trained by n-step
Q-learning over an experience-replay buffer, PLUS the repo's large-margin self-imitation hinge --
the regularizer rl_nominal uses to hold the greedy argmax onto the best order the search has
found. The hinge's whole configuration (margin, weight cap, imitation ramp) is taken live from
RANK_PARAMS_NOMINAL so the two solvers regularize identically.

WHY THE HINGE IS HERE. This module first ran as the FAITHFUL paper reproduction (embedding +
n-step Q-learning, no hinge -- the `lam=0` special case, which still exists and recovers it
exactly). That run delivered scenario-mean F=0.9813 at n10/seed42 but its greedy policy NEVER
converged: the search found a nominal order at F=0.9626 that plain TD regression could not hold
onto, the greedy order still churned at the stop, and the frozen-scenario diagnostic drifted the
wrong way late in training. That is precisely the failure the repo retired a pure value-loss DQN
for, and the large-margin hinge is the mechanism that fixed it there. Adding it here tests whether
it fixes the same drift on the S2V embedding. Setting `lam=0` in the params reverts to the
faithful baseline.

Still ABSENT relative to rl_nominal (so the embedding stays the thing under test): the flow-prior
residual. Shared now: the hinge, Double DQN, prioritized replay, and the dueling head (the last
flagged as possibly reverting -- dueling=False removes it cleanly).

MAPPING ONTO THIS PROBLEM. The embedding runs on the LINE GRAPH of the road network: every road
segment is a vertex, two segments are adjacent iff they share an intersection (exactly the
adjacency build_env already computes for the mp features). Actions are the damaged, not-yet-
scheduled vertices; intact segments participate in the embedding but are never actions. The
paper's binary "in the partial solution" node tag becomes an 8-dim x_v: a 3-way one-hot
(pending / under repair / completed -- an under-repair segment still suppresses accessibility,
so merging it with "completed" would alias two physically different states), four static
normalized scores (baseline flow phi_hat over ALL segments, severity, duration BELIEF at the
planning expectation, demand drop), and (use_flow, on by default) the LIVE UE FLOW of each
segment in the current damaged network. The live-flow column is the RAW per-segment value:
structure2vec then propagates it over the line graph, so the neighbourhood aggregates that
rl_nominal hand-computes as mp*_flow are formed inside the embedding rather than precomputed.
use_flow=False drops it and recovers the 7-feature static state (whether S2V can learn the flow
signal from structure alone).

DEVIATIONS FROM THE PAPER, all deliberate and recorded in run_meta (`paper_deviations`):
  1. x_v is an 8-dim vector (7 with use_flow off), not the binary tag (the paper allows this).
  2. The carrier is the line graph (the paper is handed its graph).
  3. The readout gains a global temporal block g = (t/T, crew gap, demand-shortfall fraction,
     remaining-work fraction): Q = th5' relu([th6*sum_u mu_u, th7*mu_v, th8*g]). The embedding
     itself stays time-free; `use_g=False` restores the paper's exact readout.
  4. Reward = the repo's completion-window credit (util.rl._completion_rewards), not the
     per-action marginal objective delta. Both telescope to the episode objective; this one keeps
     the return scale identical to rl_nominal so diagnostics stay comparable.
  5. Replay tuples are inserted at EPISODE END, not per step: a decision's window credit depends
     on the NEXT completion, so it is unobservable mid-episode.
  6. gamma = 0.9995 (project owner's decision: aligned with rl_nominal; the paper's
     cumulative-reward-equals-objective reading implies 1.0).
  7. Target network ON by default (project owner's decision: the operational fitted-Q-iteration
     reading, and what the paper's released code does); target_sync=0 restores Algorithm 1's
     literal current-parameter bootstrap.
  8. Adam, not the SGD the paper's text names.
  9. Stopping by the repo's plateau rule (RankPlateauStop) + safety cap, not a fixed episode
     budget. Delivery is the FINAL policy per repo convention; the best-nominal-probe weights are
     saved alongside (results/model_best_nominal.pt) as the drift diagnostic.
 10. The eps-greedy anneal is LINEAR over a fixed episode span (an open-ended stop rule cannot
     provide the horizon a linear schedule needs).
 11. Training is on the ONE nominal world of one damage instance (rl_nominal's regime), not the
     paper's distribution-over-instances regime -- fair vs. the siblings, but not a test of the
     paper's generalization claim.
 12. Constant learning rate; the paper decays lr exponentially by 0.95 (Appendix D.5).
 13. Update cadence: Algorithm 1 interleaves ONE gradient step with every acted step; here all
     updates_per_ep steps run at episode end. Forced by deviation 5 -- no tuple exists to learn
     from until the episode's rewards become observable.
 14. The reused n-step helper discounts INSIDE the reward window (sum_k gamma^k r_{t+k}) where
     the paper sums the window undiscounted with a single gamma on the bootstrap; at
     gamma=0.9995 over <= 2 steps the difference is ~1e-3 relative, kept for the shared,
     battle-tested implementation.
 15. The large-margin ranking hinge (the WHOLE point of the current default): a regularizer the
     paper does not have, added on the project owner's instruction to counter the drift the
     faithful run exposed. lam=0 removes it and restores the paper's method.
 16. Double DQN (van Hasselt et al. 2016): online net selects the bootstrap action, target net
     evaluates it. Not in the paper; a rank-loss sibling default. double_dqn=False restores the
     paper's plain max-Q target.
 17. Prioritized experience replay (Schaul et al. 2016): proportional priorities + annealed IS
     weights, config from RANK_PARAMS_NOMINAL. The paper uses uniform replay; prioritized=False
     restores it.
 18. Dueling head (Wang et al. 2016): Q = V(s) + A(s,v) - mean_legal(A). Not in the paper; kept
     ISOLATED for possible reversion, dueling=False restores the plain single-head readout.
     (The Wang trunk-gradient 1/sqrt(2) rescale was tried on top and reverted 2026-08-24 -- it
     hurt: 0.9567 vs 0.9473.)
 19. Live UE flow as an 8th node feature (use_flow): the current-network flow enters x_v raw and
     the embedding aggregates it, in contrast to rl_nominal's precomputed mp*_flow neighbourhood
     means. Not in the paper; one memoized UE solve per decision state. use_flow=False removes it.

ISOLATION / REMOVAL. This module only READS from the rest of the repo (util.rl primitives,
util.rl_rank's build_env / RankPlateauStop / _nstep_rows / _Recorder / EP_CAP, provenance
helpers); it mutates nothing outside its own output folder. To delete the method entirely:
  * delete this file;
  * remove the "rl_s2v" dispatch line in main.py (--solve mapping + the name in its error text
    and in the module docstring's solver list);
  * remove "rl_s2v" from util/provenance.py SOLVER_DIR;
  * remove the ("rl_s2v", "03-RL") entry and the rl_s2v kind branch in util/compare.py;
  * remove the rl_s2v color / marker lines in viz/compare_viz.py;
  * delete outputs/03-RL/03-rl_s2v/ and refresh the comparison.
It is deliberately NOT registered in util.sim_cache's canonical-run scan or util.seed_sweep;
both registrations wait until the method is adopted.

Run:  python main.py --solve rl_s2v      (or  python -m util.rl_s2v)
"""
import time

import numpy as np
import pandas as pd

import config as P
from util.evaluate import schedule_from_permutation
from util.oracle import _baseline_twoway_flow, scale_dir
from util.provenance import (solver_dir, fresh_scale_dir, log_dir, results_dir, slot_rows,
                             write_run_meta)
from util.rl import _completion_rewards, _evaluate_prefix_cached
from util.rl_rank import (EP_CAP, FIGURE_REFRESH_EVERY, OUT_DIAG, RANK_PARAMS_NOMINAL,
                          STOP_PARAMS, TOY, RankPlateauStop, _current_flow, _nstep_rows,
                          _Recorder, _rollout, build_env)

# --------------------------------------------------------------------------- #
# Hyperparameters. p / t_emb / batch are the paper's main setting (Table D.9: embedding size 64,
# 3-5 propagation rounds, batch 64); n_step sits inside the paper's per-problem set. Where a value
# departs from what the paper pins (constant lr vs its 0.95 decay, end-of-episode update cadence
# vs Algorithm 1's per-step interleaving) the departure is a numbered deviation in the module
# docstring; the truly unpinned knobs (anneal span, target_sync cadence, the lr value itself) are
# fabricated choices and say so in place. All of it lands in run_meta.
S2V_PARAMS = dict(
    p=64,                # embedding dimension (paper Table D.9)
    t_emb=4,             # synchronous structure2vec propagation rounds (paper uses 3-5)
    use_g=True,          # append the 4-dim global temporal block at the readout (deviation 3)
    use_flow=True,       # LIVE UE FLOW as an 8th node feature (deviation 19). The RAW per-segment
                         # current-network flow enters x_v; structure2vec's own propagation then
                         # produces the neighbourhood aggregates by itself -- so, unlike
                         # rl_nominal's hand-computed mp*_flow columns, NO first/second-order
                         # neighbourhood mean is precomputed here. Costs one UE solve per decision
                         # state (memoized). use_flow=False restores the 7-feature static state.
    dueling=True,        # DUELING HEAD (Wang et al. 2016), deviation 18. Split the readout into a
                         # state value V(s) and a per-vertex advantage; Q = V + A - mean_legal(A).
                         # ISOLATED for easy removal (may revert) -- dueling=False restores the
                         # plain single-head readout byte-identical. See _build_s2v_net.
                         # NOTE: the Wang et al. trunk-gradient 1/sqrt(2) rescale (trunk_rescale)
                         # was tried on top of this on 2026-08-24 and REVERTED -- it delivered
                         # 0.9567 vs 0.9473 without it (early stop at 143 eps), matching
                         # rl_nominal's own finding that the Wang recipe hurts on this problem.
    n_step=2,            # n-step return horizon, inside the paper's per-problem set 1-5
                         # (Table D.9: MVC 5, MAXCUT 1, TSP 1, SCP 2)
    gamma=0.9995,        # discount, aligned with rl_nominal (deviations 6 and 14)
    lr=1e-3,             # CONSTANT Adam step; the paper decays lr by 0.95 (D.5) -- deviation 12
    batch=64,            # replay minibatch (paper Table D.9)
    updates_per_ep=10,   # ~one gradient step per decision, approximating the paper's per-step SGD
    eps0=1.0,            # eps-greedy start...
    eps_min=0.05,        # ...floor...
    eps_anneal=200,      # ...reached by LINEAR anneal over this many episodes (deviation 10)
    target_sync=25,      # gradient steps between target-network copies; 0 = Algorithm 1's
                         # literal current-parameter bootstrap (deviation 7)
    double_dqn=True,     # van Hasselt et al. 2016: the ONLINE net selects the bootstrap action,
                         # the TARGET net evaluates it (deviation 16; a rank-loss sibling default,
                         # not in the paper). Meaningful only with target_sync > 0.
    replay_cap=4000,     # transitions kept, oldest dropped (matches the sibling's buffer)
    # PRIORITIZED EXPERIENCE REPLAY (Schaul et al. 2016), deviation 17. Config taken live from
    # RANK_PARAMS_NOMINAL as with the hinge: proportional priorities on a plain list (no sum-tree
    # -- the buffer is small), importance-sampling weights annealed by beta. The paper uses UNIFORM
    # replay, so prioritized=False restores it.
    prioritized=bool(RANK_PARAMS_NOMINAL["prioritized"]),
    per_alpha=float(RANK_PARAMS_NOMINAL["per_alpha"]),       # priority exponent
    per_beta0=float(RANK_PARAMS_NOMINAL["per_beta0"]),       # IS-weight exponent at episode 0...
    per_beta_eps=float(RANK_PARAMS_NOMINAL["per_beta_eps"]), # ...annealed to 1 over ~this many eps
    per_eps=float(RANK_PARAMS_NOMINAL["per_eps"]),           # floor added to each |TD| priority
    # LARGE-MARGIN RANKING HINGE (added 2026-08-23 on the project owner's instruction). Its whole
    # configuration is TAKEN LIVE from RANK_PARAMS_NOMINAL so this solver regularizes exactly as
    # rl_nominal does ("reference the current RL"). The exemplar is the best-by-F nominal order so
    # far; the term demands every rival action fall at least `margin` below the exemplar's Q, and
    # its weight climbs a geometric ramp lam(ep)=min(lam, lam0*lam_growth^(ep-1)). lam=0 removes
    # the term and recovers the FAITHFUL paper reproduction (see the module docstring).
    lam=float(RANK_PARAMS_NOMINAL["lam"]),               # hinge weight cap / terminal weight
    margin=float(RANK_PARAMS_NOMINAL["margin"]),         # m: exemplar action vs each rival gap
    lam0=float(RANK_PARAMS_NOMINAL["lam0"]),             # imitation ramp start...
    lam_growth=float(RANK_PARAMS_NOMINAL["lam_growth"]), # ...geometric growth/episode, capped at lam
)

# The plateau rule probes every episode here (the nominal best-so-far score is free), so the
# shared STOP_PARAMS patience -- calibrated in PROBES -- must be widened exactly as rl_nominal's
# override widens it, and for the same reason (see STOP_OVERRIDES in util.rl_rank).
S2V_STOP_OVERRIDES = dict(patience_P=75, stable_K=75)


def _merge_stop_s2v(stop_params):
    sp = dict(STOP_PARAMS)
    sp.update(S2V_STOP_OVERRIDES)
    sp.update(stop_params or {})
    return sp


# --------------------------------------------------------------------------- #
# Graph carrier and state assembly
# --------------------------------------------------------------------------- #
def _graph_tensors(env, toy_dir=TOY):
    """Everything static the embedding needs, computed once per run.

    The carrier is the line graph over ALL edges: env["mp"]["adj"] (undirected shared-endpoint
    adjacency) as a dense n_all x n_all matrix, plus each vertex's degree for the paper's
    th3-term (with w=1 the term sum_{u in N(v)} relu(th4*w) collapses to deg_v * relu(th4), a
    learned connectivity signal -- the paper's own unweighted-graph setting).

    x_v's four static columns: phi_hat over ALL segments needs its own baseline-flow pass --
    env["st"] carries the damaged subset only -- and is normalized by the ALL-edge maximum so
    intact segments stay in [0, 1]. Severity / duration-belief / demand columns exist only for
    the damaged set and reuse env["st"]'s normalizations unchanged."""
    mp, st = env["mp"], env["st"]
    segs_all = sorted(mp["end_of"])
    idx = {e: i for i, e in enumerate(segs_all)}
    n = len(segs_all)
    A = np.zeros((n, n), dtype=np.float32)
    for a, nbrs in mp["adj"].items():
        for b in nbrs:
            A[idx[a], idx[b]] = 1.0
    flow = _baseline_twoway_flow(toy_dir, cores=1)     # cores=1: bit-stable, as in build_env
    phi_raw = np.array([flow.get(tuple(sorted(mp["end_of"][e])), 0.0) for e in segs_all],
                       dtype=np.float32)
    # flow_scale = the largest pre-disaster segment flow, used to normalize BOTH the static phi
    # column and (when use_flow) the live-flow feature, so the two flows share one scale. Live
    # flow can exceed it on rerouted edges -- that > 1 is meaningful and left as-is.
    flow_scale = float(phi_raw.max()) or 1.0
    xs = np.zeros((n, 7), dtype=np.float32)            # cols: pending, under, done, phi, sev, dur, dem
    xs[:, 3] = phi_raw / flow_scale
    for e in env["segs"]:
        xs[idx[e], 4] = st["sev_hat"][e]
        xs[idx[e], 5] = st["dur_hat"][e]
        xs[idx[e], 6] = st["dem_hat"][e]
    return dict(segs_all=segs_all, idx=idx, A=A, deg=A.sum(axis=1), xs=xs, flow_scale=flow_scale)


def _state_x(gt, state, use_flow=False):
    """The tagged node-feature matrix for one decision state: the static columns plus the 3-way
    tag set from what has OBSERVABLY happened (pending = damaged & unscheduled, under = crew on
    it and not yet complete, done = everything else damaged). When use_flow, an 8th column holds
    the RAW live UE flow of each segment (state['flow'], keyed by edge id), normalized by the
    baseline-flow scale -- the embedding aggregates it over the graph, so no neighbourhood mean
    is formed here."""
    x = gt["xs"].copy()
    for e in state["pending"]:
        x[gt["idx"][e], 0] = 1.0
    for e in state["under"]:
        x[gt["idx"][e], 1] = 1.0
    for e in state["done"]:
        x[gt["idx"][e], 2] = 1.0
    if not use_flow:
        return x
    flow = state["flow"]
    fcol = np.array([flow.get(e, 0.0) for e in gt["segs_all"]],
                    dtype=np.float32)[:, None] / gt["flow_scale"]
    return np.concatenate([x, fcol], axis=1)


def _build_s2v_net(p, t_emb, use_g, torch, nn, dueling=False, n_in=7):
    """The paper's parameterization, verbatim plus the optional th8 global block. All linear maps
    bias-free as in the paper's equations. Forward returns Q for EVERY vertex; the caller gathers
    the pending candidates, which is the legality mask.

    DUELING HEAD (Wang et al. 2016), added 2026-08-24 and DELIBERATELY ISOLATED for easy removal
    (the project owner flagged this step as possibly reverting). When `dueling` is set, the readout
    splits into a state-value scalar V(s) and a per-vertex advantage A(s,v), and forward returns
    the PAIR (V, A) instead of Q; the caller forms Q = V + A - mean_legal(A) over the PENDING
    vertices (see _q). EVERYTHING the dueling path adds is guarded by `dueling` -- the extra value
    head th5_val, the tuple return, the _q aggregation -- so dueling=False leaves the original
    single-head forward byte-identical and dropping the head is a one-flag revert."""

    class S2VNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.th1 = nn.Linear(n_in, p, bias=False)   # n_in = 7, or 8 with the live-flow feature
            self.th2 = nn.Linear(p, p, bias=False)
            self.th3 = nn.Linear(p, p, bias=False)
            self.th4 = nn.Parameter(torch.randn(p) * 0.1)
            self.th6 = nn.Linear(p, p, bias=False)
            self.th7 = nn.Linear(p, p, bias=False)
            self.th8 = nn.Linear(4, p, bias=False) if use_g else None
            self.th5 = nn.Linear(3 * p if use_g else 2 * p, 1, bias=False)
            self.dueling = dueling
            # DUELING-ONLY (removable with the flag): a state-value head on the pooled embedding
            # (+ g). th5 above then serves as the advantage head. None when not dueling.
            self.th5_val = (nn.Linear(2 * p if use_g else p, 1, bias=False) if dueling else None)

        def forward(self, x, A, deg, g):
            # structure2vec: mu <- relu(th1 x + th2 sum_nbr mu + th3 (deg * relu(th4)))
            pre = self.th1(x) + self.th3(deg.unsqueeze(1) * torch.relu(self.th4).unsqueeze(0))
            mu = torch.zeros(x.shape[0], p)
            for _ in range(t_emb):
                mu = torch.relu(pre + self.th2(A @ mu))
            pooled = self.th6(mu.sum(dim=0))                       # state surrogate
            per = self.th7(mu)                                     # action surrogate, per vertex
            g_block = self.th8(g) if self.th8 is not None else None
            blocks = [pooled.unsqueeze(0).expand(x.shape[0], p), per]
            if g_block is not None:
                blocks.append(g_block.unsqueeze(0).expand(x.shape[0], p))
            head = self.th5(torch.relu(torch.cat(blocks, dim=1))).squeeze(-1)
            if not self.dueling:
                return head                                       # Q per vertex (ORIGINAL path)
            # DUELING-ONLY (removable): `head` is now the advantage A(s,v); add the state value V(s)
            # from a state-only representation. The caller subtracts the legal-action mean of A.
            val_blocks = [pooled] + ([g_block] if g_block is not None else [])
            V = self.th5_val(torch.relu(torch.cat(val_blocks, dim=0))).squeeze(-1)
            return V, head                                        # (scalar V, per-vertex A)

    return S2VNet()


def _s2v_rollout(env, gt, pick, durations=None, use_flow=False):
    """One episode on the line-graph state: the same crew scheduling and demand-shortfall
    recursion as util.rl_rank._rollout (the schedule/demand block is copied from it verbatim --
    _self_check asserts the two stay in lockstep), but the recorded per-decision state is the
    tagged-graph state this solver's network consumes, not a hand-crafted feature matrix.

    Returns (perm, start, states, picks): `states` are dicts carrying the pending tuple (in the
    remaining-list order, so picks index into it), the under-repair and done tuples, the 4-dim
    global block g, and (when use_flow) `flow`, the current-network UE flow of every segment for
    this decision's damage state -- one memoized UE solve, reused by _state_x's 8th column.
    Duration BELIEFS stay at the planning expectation throughout, exactly as in _rollout."""
    ctx, st, T = env["ctx"], env["st"], env["T"]
    durations = env["nominal"] if durations is None else durations
    dis_l, sev, B = ctx["disrupted"], ctx["severity_vec"], ctx["B"]
    seg_idx = env["seg_idx"]
    remaining = list(env["segs"])
    crew = [1] * P.C_MAX
    start, perm, states, picks = {}, [], [], []
    D = np.zeros(len(ctx["H0"]))
    k, comp = 0, {}
    while remaining:
        t = min(crew)
        while k < t:                                   # advance the demand shortfall to slot t
            k += 1
            v = np.zeros(len(dis_l))
            for e in remaining:
                v[seg_idx[e]] = sev[seg_idx[e]]
            for e, c in comp.items():
                if k < c:
                    v[seg_idx[e]] = sev[seg_idx[e]]
            D = np.maximum(B @ v, P.RHO * D)
        bel = st["dur_raw"]
        g = np.array([t / T, (max(crew) - t) / T, float(D.sum()) / st["sum_H0"],
                      sum(bel[e] for e in remaining) / st["total_work"]], dtype=np.float32)
        under = tuple(e for e, c in comp.items() if c > t)
        state = dict(pending=tuple(remaining), under=under,
                     done=tuple(e for e in comp if comp[e] <= t), g=g)
        if use_flow:                                   # live UE flow of the current damage state
            state["flow"] = _current_flow(env, set(remaining) | set(under), D)
        states.append(state)
        i = pick(remaining, state)
        picks.append(i)
        e = remaining.pop(i)
        c = int(np.argmin(crew))
        start[e] = crew[c]
        crew[c] = start[e] + int(durations[e])
        comp[e] = crew[c]
        perm.append(e)
    return tuple(perm), start, states, picks


def _self_check(env, gt, Qf, torch, use_flow=False):
    """Run-start invariants, asserted every run because their failure modes are silent.

    (a) Schedule/demand parity: _s2v_rollout's copied recursion must produce, decision for
        decision, the same global features as util.rl_rank._rollout under the same fixed order --
        a divergence corrupts the state without corrupting F, the worst kind of bug.
    (b) Reward identity: the completion-window credits must sum to T - sum_t g_t (== T*(1 - F)
        under this project's MU=1, full-horizon-mean objective) -- the alignment the training
        signal is built on.
    (c) The network forward is deterministic on a fixed state."""
    order = sorted(env["segs"], key=lambda e: -env["phi"][e])          # the flow order

    def fp(rem, _state_or_X):
        for e in order:
            if e in rem:
                return rem.index(e)
    perm_a, start_a, states, _ = _s2v_rollout(env, gt, fp, use_flow=use_flow)
    perm_b, start_b, feats, _, _ = _rollout(env, fp, n_lag=0)
    assert perm_a == perm_b and start_a == start_b, "s2v rollout schedules differently than rl_rank"
    for s, X in zip(states, feats):
        assert np.allclose(s["g"], np.asarray(X)[0, 4:8], atol=1e-6), (
            "s2v rollout's global state diverged from rl_rank._rollout's -- the copied "
            "schedule/demand recursion has drifted")
    res = _evaluate_prefix_cached(start_a, env["nominal"], env["T"], env["ctx"], {})
    rew, _ = _completion_rewards(list(perm_a), start_a, env["nominal"], res["terms"], env["T"],
                                 env["phi"])
    assert abs(sum(rew) - (env["T"] - float(res["terms"].sum()))) < 1e-6, (
        "episode return != T - sum(g_t): the reward decomposition is broken")
    if P.MU == 1.0 and not P.F1_ACTIVE_ONLY:
        assert abs(sum(rew) - env["T"] * (1.0 - res["F"])) < 1e-6
    with torch.no_grad():
        q1, q2 = Qf(states[0]), Qf(states[0])
    assert torch.allclose(q1, q2), "network forward is not deterministic"


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train_s2v(env, hp=None, seed=P.SEED, ep_cap=EP_CAP, rec_dir=None, stop_params=None,
              verbose=True):
    """Train the faithful S2V-DQN to a plateau; return the same delivery dict shape as
    util.rl_rank.train so the runner can mirror run_rank's provenance behavior."""
    import torch
    import torch.nn as nn
    torch.set_num_threads(1)

    unknown = set(hp or {}) - set(S2V_PARAMS)
    if unknown:
        raise ValueError(f"unknown rl_s2v hyperparameters {sorted(unknown)}; valid keys are "
                         f"{sorted(S2V_PARAMS)}")
    hp = dict(S2V_PARAMS, **(hp or {}))
    sp = _merge_stop_s2v(stop_params)
    torch.manual_seed(seed)                            # same seed derivations as rl_rank.train,
    rng = np.random.RandomState(seed * 7 + 1)          # so same-seed runs are comparable in kind

    p, t_emb, use_g = int(hp["p"]), int(hp["t_emb"]), bool(hp["use_g"])
    n_step, gamma = int(hp["n_step"]), float(hp["gamma"])
    UPD, BATCH, sync = int(hp["updates_per_ep"]), int(hp["batch"]), int(hp["target_sync"])
    eps0, eps_min, anneal = float(hp["eps0"]), float(hp["eps_min"]), int(hp["eps_anneal"])
    cap = int(hp["replay_cap"])
    lam_cap, margin = float(hp["lam"]), float(hp["margin"])
    lam0, lam_growth = hp.get("lam0"), hp.get("lam_growth")   # None -> constant lam_cap
    use_hinge = lam_cap > 0.0
    ddqn = bool(hp["double_dqn"])
    prioritized = bool(hp["prioritized"])
    per_alpha, per_beta0 = float(hp["per_alpha"]), float(hp["per_beta0"])
    per_beta_eps, per_eps = float(hp["per_beta_eps"]), float(hp["per_eps"])
    dueling = bool(hp["dueling"])
    use_flow = bool(hp["use_flow"])
    n_in = 7 + (1 if use_flow else 0)                          # 8th node feature = live UE flow

    gt = _graph_tensors(env)
    A = torch.tensor(gt["A"])
    deg = torch.tensor(gt["deg"], dtype=torch.float32)
    net = _build_s2v_net(p, t_emb, use_g, torch, nn, dueling=dueling, n_in=n_in)
    tgt = _build_s2v_net(p, t_emb, use_g, torch, nn, dueling=dueling, n_in=n_in)
    tgt.load_state_dict(net.state_dict())
    opt = torch.optim.Adam(net.parameters(), lr=float(hp["lr"]))

    def _q(model, state):
        """Q over one state's PENDING candidates, in their remaining-list order (the gather IS
        the legality mask: a scheduled segment's vertex never reaches any consumer)."""
        x = torch.tensor(_state_x(gt, state, use_flow))
        idx = [gt["idx"][e] for e in state["pending"]]
        out = model(x, A, deg, torch.tensor(state["g"]))
        if not dueling:
            return out[idx]
        # DUELING-ONLY (removable with the flag): Q = V + A - mean_legal(A). The mean is over the
        # PENDING (legal) advantages, so the identifiability subtraction and the argmax both range
        # over exactly the live actions -- matching rl_rank's dueling aggregation.
        V, adv = out
        a = adv[idx]
        return V + a - a.mean()

    def Qf(state):
        return _q(net, state)

    def Qt(state):
        """Target-side Q: the lagged snapshot when target_sync > 0, else (Algorithm 1's literal
        form, deviation 7) the online network itself."""
        return _q(tgt if sync > 0 else net, state)

    def greedy(rem, state):
        with torch.no_grad():
            return int(torch.argmax(Qf(state)))

    def states_for(order):
        """The decision states along a FIXED order, for the hinge exemplar: roll the order out
        deterministically and return (states, picks). picks[i] indexes into states[i]['pending']
        (the remaining-list order), so Q[pick] is the exemplar action's Q -- rl_nominal's
        states_for semantics, on this solver's state objects."""
        def fp(rem, _state):
            for e in order:
                if e in rem:
                    return rem.index(e)
        _, _, sts, pk_ = _s2v_rollout(env, gt, fp, use_flow=use_flow)
        return sts, pk_

    def _hinge(ex_states, ex_picks):
        """Large-margin term over one exemplar trajectory (util.rl_rank._hinge, adapted to the
        state-dict Qf): every rival action must fall at least `margin` below the exemplar action's
        Q, and contributes nothing once it does. Qf carries gradients here, so the term trains the
        embedding to RANK the exemplar order first."""
        ml = 0.0
        for st_i, a in zip(ex_states, ex_picks):
            Q = Qf(st_i)
            viol = torch.relu(margin - (Q[a] - Q)).clone()
            viol[a] = 0.0                              # the exemplar action never penalizes itself
            ml = ml + viol.sum()
        return ml / len(ex_states)

    _self_check(env, gt, Qf, torch, use_flow=use_flow)

    rec = None
    if rec_dir is not None:
        # Trace-only recorder: the qevo/representation probes are row-architecture-specific in
        # rl_rank; the per-episode trace is architecture-agnostic and every figure drawn from it
        # survives (make_rank_figures skips what is missing).
        rec = _Recorder(rec_dir, "rl_s2v", None, None, 0.0, torch,
                        flush_every=FIGURE_REFRESH_EVERY)

    stop = RankPlateauStop(sp["ep_min"], sp["patience_P"], sp["stable_K"], sp["tol"])
    memo, sc, replay, prio = {}, {}, [], []      # prio: per-transition priorities (PER only)
    best_F, best_perm, best_sd, gstep = float("inf"), None, None, 0
    t0 = time.perf_counter()
    ep = 0
    while ep < ep_cap and not stop.done():
        # LINEAR eps anneal (deviation 10): eps-greedy from the very first episode, as in the
        # paper -- no forced-greedy episode 0 and no prior to start from.
        eps = max(eps_min, eps0 - (eps0 - eps_min) * ep / max(1, anneal))
        # Imitation ramp, exactly rl_nominal's: min(lam_cap, lam0*lam_growth^(ep-1)); constant
        # lam_cap when no ramp is configured. 0 when the hinge is off.
        lam_ep = (0.0 if not use_hinge else
                  lam_cap if lam_growth is None else
                  min(lam_cap, float(lam0) * float(lam_growth) ** max(0, ep - 1)))

        def pk(rem, state):
            if rng.rand() < eps:
                return rng.randint(len(rem))
            with torch.no_grad():
                return int(torch.argmax(Qf(state)))

        perm, start, states, picks = _s2v_rollout(env, gt, pk, use_flow=use_flow)
        if perm in memo:                               # one fixed world: a repeated order is free
            res = memo[perm]
        else:
            res = _evaluate_prefix_cached(start, env["nominal"], env["T"], env["ctx"], sc)
            memo[perm] = res
        if res["F"] < best_F:
            best_F, best_perm = res["F"], perm
            best_sd = {k2: v2.detach().clone() for k2, v2 in net.state_dict().items()}
        rew, _ = _completion_rewards(list(perm), start, env["nominal"], res["terms"], env["T"],
                                     env["phi"])
        # Episode-end insertion (deviation 5): a decision's window credit is unobservable until
        # the episode's later completions are known.
        fresh = _nstep_rows(states, picks, rew, n_step, gamma)
        replay.extend(fresh)
        if prioritized:                            # new transitions enter at the current max priority
            prio.extend([max(prio, default=1.0)] * len(fresh))
        if len(replay) > cap:
            replay = replay[-cap:]
            if prioritized:
                prio = prio[-cap:]

        # The hinge exemplar: the best-by-F nominal order so far, rolled once per episode into its
        # decision states (rl_nominal recomputes states_for(best_perm) per update; here the order
        # is fixed for the whole episode, so one rollout serves every update -- cheaper, identical
        # exemplar). None until the first order is scored, or whenever the hinge is off.
        ex = states_for(best_perm) if (use_hinge and best_perm is not None) else None
        # IS-weight exponent, annealed 0 -> 1 across training (Schaul et al. 2016); constant within
        # an episode, so computed once here rather than per update.
        beta = min(1.0, per_beta0 + (1.0 - per_beta0) * ep / per_beta_eps) if prioritized else None
        ep_q, ep_y, ep_td, ep_loss = [], [], [], []
        for _u in range(UPD):
            opt.zero_grad()
            if prioritized:
                pr = np.asarray(prio, dtype=float) ** per_alpha
                probs = pr / pr.sum()
                idx = rng.choice(len(replay), size=min(BATCH, len(replay)), replace=False, p=probs)
                w = (len(replay) * probs[idx]) ** (-beta)
                w_t = torch.tensor(w / w.max(), dtype=torch.float32)
            else:
                idx = rng.choice(len(replay), size=min(BATCH, len(replay)), replace=False)
            q = torch.stack([Qf(replay[i][0])[replay[i][1]] for i in idx])
            y = []
            for i in idx:
                _, _, R, bs, bd = replay[i]
                if bs is None:
                    y.append(R)                        # truncated tail: pure Monte-Carlo target
                else:
                    with torch.no_grad():
                        if ddqn:                       # online selects the action, target evaluates it
                            a_on = int(torch.argmax(Qf(bs)))
                            y.append(R + bd * float(Qt(bs)[a_on]))
                        else:
                            y.append(R + bd * float(Qt(bs).max()))
            yt = torch.tensor(y, dtype=torch.float32)
            td_elem = (q - yt) ** 2                     # squared TD error...
            if prioritized:                            # ...IS-weighted, then priorities refreshed
                loss = (w_t * td_elem).mean()
                with torch.no_grad():
                    for j, i in enumerate(idx):
                        prio[i] = float(abs(q[j] - yt[j])) + per_eps
            else:
                loss = td_elem.mean()
            if ex is not None:                         # ...plus the large-margin ranking hinge
                loss = loss + lam_ep * _hinge(ex[0], ex[1])
            with torch.no_grad():
                ep_q.append(float(q.mean()))
                ep_y.append(float(yt.mean()))
                ep_td.append(float((q - yt).abs().mean()))
            loss.backward()
            opt.step()
            gstep += 1
            ep_loss.append(float(loss.detach()))
            if sync > 0 and gstep % sync == 0:
                tgt.load_state_dict(net.state_dict())

        # Plateau probe, the rl_nominal wiring: score = best-so-far nominal F (free), order = the
        # greedy policy's nominal rollout; the frozen-scenario diagnostic every probe_every
        # episodes, list-scheduling the greedy order as the cheap proxy, and NEVER into the stop.
        gperm, _, _, _ = _s2v_rollout(env, gt, greedy, use_flow=use_flow)
        scen_F = None
        if ep % sp["probe_every"] == 0:
            scen_F = float(np.mean([
                _evaluate_prefix_cached(schedule_from_permutation(list(gperm), d), d,
                                        env["T"], env["ctx"], sc)["F"] for d in env["scen"]]))
        stop.update(ep, best_F, gperm)

        if rec is not None:
            _m = lambda a: (float(np.mean(a)) if a else None)
            rec.episode(episode=ep, eps=float(eps), lam=float(lam_ep), F=float(res["F"]),
                        best_F=float(best_F), scenF=scen_F,
                        greedy_order="-".join(map(str, gperm)),
                        loss=_m(ep_loss), q_pred_mean=_m(ep_q), y_target_mean=_m(ep_y),
                        td_abs_mean=_m(ep_td), since_improve=stop.since_improve,
                        stable=stop.stable)
        if verbose and ep % 25 == 0:
            print(f"  [rl_s2v] ep {ep}  eps={eps:.3f}  lam={lam_ep:.3f}  best_F={best_F:.4f}  "
                  f"since_improve={stop.since_improve}  stable={stop.stable}", flush=True)
        ep += 1

    outcome = stop.reason or "episode_cap"
    if rec is not None:
        rec.finish(net)

    # Delivery: the FINAL policy rolled inside each frozen scenario, exactly rl_nominal's
    # semantics (observed history only, beliefs at the planning expectation).
    per_scenario = []
    for d in env["scen"]:
        gp, gs, _, _ = _s2v_rollout(env, gt, greedy, d, use_flow=use_flow)
        per_scenario.append((list(gp), gs))
    order = list(_s2v_rollout(env, gt, greedy, use_flow=use_flow)[0])   # nominal-world summary only
    if verbose:
        print(f"  [rl_s2v] stopped at ep {ep} ({outcome}) in "
              f"{(time.perf_counter() - t0) / 60:.1f} min", flush=True)
    return dict(order=order, per_scenario=per_scenario, net=net, best_net_sd=best_sd,
                episodes=ep, n_evals=len(memo), outcome=outcome, best_score=best_F, hp=hp,
                seed=seed)


# --------------------------------------------------------------------------- #
# Run + provenance (mirrors util.rl_rank.run_rank's file family and two-stage clear)
# --------------------------------------------------------------------------- #
def run_s2v(toy_dir=TOY, N=None, M=P.M_SCENARIOS, seed=P.SEED, ep_cap=EP_CAP, hp=None,
            stop_params=None):
    """Train rl_s2v to a plateau and write its canonical results and diagnostics to
    outputs/03-RL/{solver_dir('rl_s2v')}/n{N}/, then refresh the comparison."""
    import torch
    N = P.N_DISRUPTED_ORACLE if N is None else N
    env = build_env(toy_dir, N=N, M=M)
    print(f"instance: {len(env['segs'])} segments {env['segs']}; M={M}; T={env['T']}; "
          f"seed={seed}; ep_cap={ep_cap}; variant=rl_s2v", flush=True)
    v = "rl_s2v"
    vdir = scale_dir(OUT_DIAG / solver_dir(v), N)
    vdir.mkdir(parents=True, exist_ok=True)
    # Stage 1 of the two-stage clear: diagnostics only, so a run that dies mid-training cannot
    # destroy a delivered model it never reproduced (see run_rank).
    fresh_scale_dir(vdir, subdirs=("log",), figures=True)
    rdir = log_dir(vdir)
    t0 = time.perf_counter()

    r = train_s2v(env, hp=hp, seed=seed, ep_cap=ep_cap, rec_dir=str(rdir),
                  stop_params=stop_params)
    rows, slots = [], []
    for m, (dur, (order_m, start)) in enumerate(zip(env["scen"], r["per_scenario"])):
        ts = time.perf_counter()
        res = _evaluate_prefix_cached(start, dur, env["T"], env["ctx"], {}, collect_traces=True)
        slots.extend(slot_rows(m, res))
        row = dict(scenario=m, F=res["F"], F1=res["F1"], F2=res["F2"],
                   time_s=time.perf_counter() - ts, n_evals=r["n_evals"],
                   episodes=r["episodes"], outcome=r["outcome"],
                   # Serial-equivalent compute, the nominal-variant convention: every distinct
                   # order the search scored costs T UE solves, amortised over the M scenarios,
                   # plus this scenario's own evaluation (see run_rank / _compute_accounting).
                   ue_total=r["n_evals"] * env["T"] / len(env["scen"]) + env["T"],
                   order="-".join(map(str, order_m)),
                   durations="-".join(str(int(dur[e])) for e in env["segs"]))
        row["policy_order_nominal"] = "-".join(map(str, r["order"]))
        for e in env["segs"]:
            row[f"start_{e}"] = start[e]
        rows.append(row)
    fresh_scale_dir(vdir, subdirs=("results", "config"), figures=False)   # stage 2: now replace
    pd.DataFrame(rows).to_csv(results_dir(vdir) / f"{v}_optima.csv", index=False)
    pd.DataFrame(slots).to_csv(rdir / f"{v}_slots.csv", index=False)
    torch.save(r["net"].state_dict(), results_dir(vdir) / "model_best.pt")
    if r["best_net_sd"] is not None:                   # the drift diagnostic (deviation 9)
        torch.save(r["best_net_sd"], results_dir(vdir) / "model_best_nominal.pt")
    meanF = float(np.mean([x["F"] for x in rows]))
    write_run_meta(vdir, method=v, segments=env["segs"], T=env["T"], seed=seed, M=M,
                   hp=dict(r["hp"]), stop_params=_merge_stop_s2v(stop_params), ep_cap=ep_cap,
                   episodes=r["episodes"], outcome=r["outcome"],
                   order_nominal_summary=r["order"], mean_F=meanF,
                   solver=("S2V-DQN (Dai et al. 2018 embedding) + large-margin ranking hinge "
                           "(lam=0 -> faithful reproduction); util.rl_s2v, EXPERIMENTAL"),
                   delivery="per-scenario adaptive policy (final), observed history only; "
                            "best-nominal-probe weights saved alongside as model_best_nominal.pt",
                   paper_deviations=[
                       "7-dim x_v instead of the binary tag",
                       "line-graph carrier over all road segments",
                       "global temporal block th8*g at the readout (use_g)",
                       "reward = completion-window credit (equal in sum to the objective)",
                       "replay tuples inserted at episode end (mid-episode unobservable)",
                       "gamma=0.9995 aligned with rl_nominal (paper reading implies 1.0)",
                       "target network on by default (paper code practice; 0 = Algorithm 1)",
                       "Adam instead of SGD",
                       "plateau stopping + final-policy delivery instead of fixed budget + "
                       "best-validation checkpoint",
                       "linear eps anneal over a fixed span",
                       "single-instance nominal-world training, not the paper's "
                       "distribution-over-instances regime",
                       "constant lr (paper D.5 decays lr by 0.95)",
                       "all gradient steps at episode end, not Algorithm 1's per-step "
                       "interleaving (forced by episode-end reward observability)",
                       "n-step window discounted internally by the shared helper (paper sums "
                       "the window undiscounted); ~1e-3 relative at gamma=0.9995, n<=2",
                       "large-margin ranking hinge from RANK_PARAMS_NOMINAL (the current "
                       "default; lam=0 removes it and restores the paper's method)",
                       "Double DQN target (online selects, target evaluates); not in the paper, "
                       "double_dqn=False restores the plain max-Q target",
                       "prioritized experience replay from RANK_PARAMS_NOMINAL; paper uses "
                       "uniform replay, prioritized=False restores it",
                       "dueling head V(s)+A(s,v)-mean_legal(A) (Wang et al. 2016); not in the "
                       "paper, isolated for reversion, dueling=False restores single head",
                       "live UE flow as an 8th node feature (use_flow), raw per-segment value "
                       "aggregated by the embedding rather than hand-computed like mp*_flow"])
    try:                                               # final redraw, now that run_meta exists
        from viz.rank_viz import make_rank_figures
        make_rank_figures(vdir, v)
    except Exception as exc:
        print(f"  [{v}] final figure draw failed ({type(exc).__name__}: {exc})", flush=True)
    print(f"[{v}] mean F = {meanF:.4f}  ({r['episodes']} eps, {r['outcome']}, "
          f"{(time.perf_counter() - t0) / 60:.1f} min)  -> {vdir}", flush=True)
    from util.compare import refresh_comparison
    refresh_comparison(N)
    return {v: meanF}


if __name__ == "__main__":
    run_s2v()
