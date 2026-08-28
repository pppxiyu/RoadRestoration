"""EXPERIMENTAL (2026-08-23): S2V-DQN (Dai et al., NIPS 2017 / arXiv 1704.01665, structure2vec
embedding) trained WITH the repo's large-margin ranking hinge, as a solver PARALLEL to
util.rl_rank and kept deliberately removable.

WHAT THIS IS. Dai et al.'s structure2vec graph embedding parameterizing Q, trained by n-step
Q-learning over an experience-replay buffer, PLUS the repo's large-margin self-imitation hinge --
the regularizer rl_dqn uses to hold the greedy argmax onto the best order the search has
found. The hinge's whole configuration (margin, weight cap, imitation ramp) is taken live from
RANK_PARAMS_DQN so the two solvers regularize identically.

WHY THE HINGE IS HERE. This module first ran as the FAITHFUL paper reproduction (embedding +
n-step Q-learning, no hinge -- the `lam=0` special case, which still exists and recovers it
exactly). That run delivered scenario-mean F=0.9813 at n10/seed42 but its greedy policy NEVER
converged: the search found a nominal order at F=0.9626 that plain TD regression could not hold
onto, the greedy order still churned at the stop, and the frozen-scenario diagnostic drifted the
wrong way late in training. That is precisely the failure the repo retired a pure value-loss DQN
for, and the large-margin hinge is the mechanism that fixed it there. Adding it here tests whether
it fixes the same drift on the S2V embedding. Setting `lam=0` in the params reverts to the
faithful baseline.

Still ABSENT relative to rl_dqn (so the embedding stays the thing under test): the flow-prior
residual. Shared now: the hinge, Double DQN, prioritized replay, and the dueling head (the last
flagged as possibly reverting -- dueling=False removes it cleanly).

MAPPING ONTO THIS PROBLEM. The embedding runs on the LINE GRAPH of the road network: every road
segment is a vertex, two segments are adjacent iff they share an intersection (exactly the
adjacency build_env already computes for the mp features). Actions are the damaged, not-yet-
scheduled vertices; intact segments participate in the embedding but are never actions. The
paper's binary "in the partial solution" node tag becomes x_v: a 3-way one-hot
(pending / under repair / completed -- an under-repair segment still suppresses accessibility,
so merging it with "completed" would alias two physically different states) plus four static
normalized scores (baseline flow phi_hat over ALL segments, severity, duration BELIEF at the
planning expectation, demand drop), plus up to three flag-guarded DYNAMIC columns (deviation 22). No live UE flow enters the state: whether S2V can learn
what the hand-crafted mp features encode is part of what this baseline tests.

DEVIATIONS FROM THE PAPER, all deliberate and recorded in run_meta (`paper_deviations`):
  1. x_v is a 7-dim vector, not the binary tag (the paper explicitly allows this).
  2. The carrier is the line graph (the paper is handed its graph).
  3. The readout gains a global temporal block g = (t/T, crew gap, demand-shortfall fraction,
     remaining-work fraction): Q = th5' relu([th6*sum_u mu_u, th7*mu_v, th8*g]). The embedding
     itself stays time-free; `use_g=False` restores the paper's exact readout.
  4. Reward = the repo's completion-window credit (util.rl._completion_rewards), not the
     per-action marginal objective delta. Both telescope to the episode objective; this one keeps
     the return scale identical to rl_dqn so diagnostics stay comparable.
  5. Replay tuples are inserted at EPISODE END, not per step: a decision's window credit depends
     on the NEXT completion, so it is unobservable mid-episode.
  6. gamma = 0.9995 (project owner's decision: aligned with rl_dqn; the paper's
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
 11. Training is on the ONE nominal world of one damage instance (rl_dqn's regime), not the
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
     weights, config from RANK_PARAMS_DQN. The paper uses uniform replay; prioritized=False
     restores it.
 18. Dueling head (Wang et al. 2016): Q = V(s) + A(s,v) - mean_legal(A). Not in the paper; kept
     ISOLATED for possible reversion, dueling=False restores the plain single-head readout.
     (The Wang trunk-gradient 1/sqrt(2) rescale was tried on top and reverted 2026-08-24 -- it
     hurt: 0.9567 vs 0.9473.)
 19. (HISTORY, tried 2026-08-24 and REMOVED outright on the project owner's instruction.) Live
     UE flow as an 8th node feature -- the raw current-network flow per segment, aggregated by
     the embedding itself: delivered 0.95266 vs 0.94727 without it, no gain (mildly worse,
     within seed noise), at one memoized UE solve per decision state. The same feature was
     independently measured useless on rl_am (its history note): live flow verified unhelpful
     on two different architectures.
 20. (HISTORY, tried 2026-08-24 and REMOVED on the project owner's instruction.) Deeper
     neighbor aggregation -- the paper's own Eq.-3 option, one relu layer per neighbor before
     the pooling sum: delivered 0.94725 vs 0.94727 without, floor-neutral, no reason to keep.
     Together with the t_emb=3 probe (0.9627, coverage DOES bind) this closed the encoder axis:
     4-round coverage is necessary, extra encoder capacity buys nothing.
 21. Deeper READOUT (2026-08-24, project owner's instruction): th5 -- and the dueling value
     head, symmetrically -- is a small MLP (Linear -> relu -> Linear, hidden width
     readout_hidden) instead of the paper's single linear on the Eq.-4 concatenation.
     readout_hidden=0 restores the paper's readout.
 22. Three DYNAMIC node columns (2026-08-24, project owner's instruction; each behind its own
     flag, each a one-flag revert, added with removal explicitly prepared for). They close the
     three provable observability gaps of the static x_v: (a) feat_shortfall -- the per-OD demand
     shortfall vector D enters only as a global sum, so states differing in WHERE the shortfall
     sits were aliased; column 7 is the B-projection sum_r B[r,e] D_r / sum_r B[r,e] H0_r.
     (b) feat_recency -- only the done SET was visible, not WHEN repairs finished, though the
     demand law's memory decays at rho per slot; column 8 is that same kernel, 1 while damaged
     and rho^(t - c_e) after completing at c_e. (c) feat_blocked -- the accessibility constraint
     restricts the action set but pending-blocked and pending-reachable segments looked
     identical; column 9 marks the blocked ones (constant 0 where the constraint never binds).
     All three are computed from OBSERVABLE quantities only (the belief-D at estimated
     severities, realized completion slots, the current reachability set) -- nothing about true
     severities or unrevealed durations leaks in.
 23. UNTIED propagation weights (2026-08-24, project owner's instruction; flag hop_untied,
     one-flag revert). The paper reuses ONE aggregation matrix th2 across all t_emb rounds --
     weight sharing is intrinsic to structure2vec's fixed-point derivation and is what makes the
     round count changeable after training. With the flag, each round gets its own th2^(t), the
     ordinary stacked-GNN parameterization, so successive rounds may transform their input
     differently (p*p extra parameters per extra round: 4096 -> 16384 at p=64, t_emb=4). It does
     NOT give a per-HOP parameter: a k-hop contribution still arrives through a PRODUCT of k
     matrices, so hop distance and round index remain entangled -- a genuinely per-hop scheme
     would aggregate A^k separately (not implemented). Prior from this project is that encoder
     CAPACITY has been consistently floor-neutral here (Eq.-3 pre-pooling relu, readout MLP,
     live-flow features) while encoder COVERAGE binds (t_emb=3 cost 0.015), and this is a
     capacity change. MEASURED n10 seed 42 (with deviation 22 on, everything else identical):
     delivered 0.9774 untied vs 0.9782 tied, a 0.0008 gain that is 1/35 of the scenario spread
     and so NOT separable on one seed. What did move: the best scenario improved markedly
     (0.9070 vs 0.9382) and the worst loosened (1.0735 vs 1.0539), and the delivered distinct
     orders fell from 10 to 5. KEPT as the default on the project owner's instruction. The cost
     of keeping it is that weight sharing is what let the round count change after training;
     untied, the network is fixed at t_emb rounds.
 24. EXECUTION-TIME OBSERVATION (2026-08-25, project owner's instruction after lifting the
     revelation ban; flags feat_obs_traffic / feat_obs_disc / feat_obs_trueD, each a one-flag
     revert). The agent observes the REALIZED world's field state at decision time: live UE
     flow and congestion per segment (solved under TRUE severities -- the state a road sensor
     would report), the OD-level disconnection pattern, and the realized demand-shortfall
     projection -- each frozen onto the state at decision time like deviation 22, and the last
     two with a global-dim companion (g grows 4 -> 6). The information rule: field state at
     slots <= now is observable; the severity/duration LABELS of unstarted segments are never
     inputs, so disconnection is fed as a B-weighted OD share, never a per-segment severed
     flag. In the NOMINAL world truth == estimate by construction, so feat_obs_trueD reproduces
     the belief columns exactly there (asserted in _self_check) and the observation channels
     only carry signal where a SCENARIO's truth deviates -- i.e. during rl_s2v_saa's pool
     training and every per-scenario delivery. NOTE the old "live-flow features measured
     useless" prior does NOT transfer: those were solved under ESTIMATED severities and carried
     no scenario information by construction. Cost: only feat_obs_traffic solves UE (one per
     decision state, cold, memoized across episodes by the same capped-completion-prefix +
     severity-tuple key the slot cache uses); disc is graph reachability, trueD is arithmetic.
     MEASURED on THIS solver, n10 seed 42: 0.9806 with the channels vs 0.9774 without, and the
     delivered distinct orders exploded 5 -> 34 -- exactly the predicted failure: nominal
     training gives the channels zero signal, so the delivery-time observations are inputs no
     gradient ever shaped and the extra adaptivity is noise. DEFAULT SPLIT since 2026-08-25
     (project owner's instruction): flags OFF in this solver, ON in rl_s2v_saa via
     S2V_SAA_PARAMS, whose pool worlds show the channels real daylight every episode.

ISOLATION / REMOVAL. This module only READS from the rest of the repo (util.rl primitives,
util.rl_rank's build_env / RankPlateauStop / _nstep_rows / _Recorder / EP_CAP, provenance
helpers); it mutates nothing outside its own output folder. To delete the method entirely:
  * delete this file;
  * remove the "rl_s2v" dispatch line in main.py (--solve mapping + the name in its error text
    and in the module docstring's solver list);
  * remove "rl_s2v" from util/provenance.py SOLVER_DIR;
  * remove the ("rl_s2v", "03-rl") entry and the rl_s2v kind branch in util/compare.py;
  * remove the rl_s2v color / marker lines in viz/compare_viz.py;
  * delete outputs/03-rl/02-rl_s2v/ and refresh the comparison.
It is deliberately NOT registered in util.sim_cache's canonical-run scan or util.seed_sweep;
both registrations wait until the method is adopted.

Run:  python main.py --solve rl_s2v      (or  python -m util.rl_s2v)
"""
import re
import time

import numpy as np
import pandas as pd

import config as P
from util.evaluate import (_matrix_from_H, accessible_segments, build_damaged_edges,
                           schedule_from_permutation)
from util.ue import solve_ue
from util.oracle import _baseline_twoway_flow, scale_dir
from util.provenance import (solver_dir, fresh_scale_dir, log_dir, results_dir, slot_rows,
                             write_run_meta)
from util.rl import _completion_rewards, _evaluate_prefix_cached
from util.rl_rank import (EP_CAP, FIGURE_REFRESH_EVERY, OUT_DIAG, RANK_PARAMS_DQN,
                          STOP_PARAMS, TOY, RankPlateauStop, _nstep_rows,
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
    t_emb=4,             # synchronous structure2vec propagation rounds (paper uses 3-5).
                         # Measured 2026-08-24: 3 rounds is NOT enough on this instance
                         # (delivered 0.9627 vs 0.94727) -- 2-hop coverage drops damaged pairs
                         # (mean distance 2.84 hops) out of each other's embedding view.
    readout_hidden=64,   # hidden width of the READOUT MLP (deviation 21): th5 (and the dueling
                         # value head, symmetrically) becomes Linear->relu->Linear instead of the
                         # paper's single linear on the Eq.-4 concat. 0 = the paper's single
                         # linear th5. (The Eq.-3 pre-pooling relu layer was tried 2026-08-24 and
                         # REMOVED: floor-neutral, delivered 0.94725 vs 0.94727 without.)
    use_g=True,          # append the 4-dim global temporal block at the readout (deviation 3)
    dueling=True,        # DUELING HEAD (Wang et al. 2016), deviation 18. Split the readout into a
                         # state value V(s) and a per-vertex advantage; Q = V + A - mean_legal(A).
                         # ISOLATED for easy removal (may revert) -- dueling=False restores the
                         # plain single-head readout byte-identical. See _build_s2v_net.
                         # NOTE: the Wang et al. trunk-gradient 1/sqrt(2) rescale (trunk_rescale)
                         # was tried on top of this on 2026-08-24 and REVERTED -- it delivered
                         # 0.9567 vs 0.9473 without it (early stop at 143 eps), matching
                         # rl_dqn's own finding that the Wang recipe hurts on this problem.
    # Deviation 22 (2026-08-24): three DYNAMIC node columns, each behind its own flag so each
    # is a one-flag revert (added on the project owner's instruction, with removal expected to be
    # cheap). They close the three observability gaps the 7-dim x_v provably has -- see the
    # deviation-22 entry in the module docstring for what each column is and why.
    feat_shortfall=True, # column 7: projected OD shortfall  sum_r B[r,e] D_r / sum_r B[r,e] H0_r
                         # -- WHERE the demand shortfall currently sits, the information the
                         # global scalar sum(D)/sum(H0) throws away
    feat_recency=True,   # column 8: residual demand pressure -- 1 while damaged, rho^(t - c_e)
                         # after completing at slot c_e (the demand law's own decay kernel), 0 for
                         # never-damaged. Makes completion HISTORY visible, not just the done set.
    feat_blocked=True,   # column 9: 1 for pending segments a crew cannot currently reach
                         # (accessibility constraint); constant 0 on instances where the
                         # constraint does not bind, a zero-risk placeholder there
    # Deviation 24 (2026-08-25): EXECUTION-TIME OBSERVATION of the realized world, added on the
    # project owner's instruction after the standing ban on execution-time revelation was lifted.
    # The information rule these implement: the agent may observe the FIELD STATE of the network
    # at any slot <= now (sensors see traffic), but the severity / duration LABELS of unstarted
    # segments are never inputs -- disconnection is therefore fed at OD level, never as a
    # per-segment severed flag. Each flag is a one-flag revert; see the deviation-24 ledger
    # entry for exactly what each column reads and what it costs.
    #
    # OFF here BY MEASUREMENT (2026-08-25, project owner's instruction): this solver trains in
    # the NOMINAL world, where truth == estimate and every observation column equals its belief
    # twin, so the channels carry zero training signal here by construction -- at delivery the
    # scenario observations are then out-of-distribution inputs no gradient ever shaped.
    # Measured n10 seed 42: 0.9806 with the flags on vs 0.9774 without, delivered distinct
    # orders 5 -> 34 (adaptive noise, not adaptation). The flags stay ON in rl_s2v_saa
    # (S2V_SAA_PARAMS overrides them): its pool worlds carry real truth-vs-estimate daylight.
    feat_obs_traffic=False,  # two columns: live two-way flow / baseline max (same normalizer as
                             # the static phi column, so their difference reads as "flow lost vs
                             # normal"), and congestion excess x/(1+x) with
                             # x = max(cost/free_flow_intact - 1, 0), max over directions.
                             # The only observation that costs UE solves (one per decision
                             # state, memoized across episodes).
    feat_obs_disc=False,     # one column + one global dim: the demand-weighted share of each
                             # segment's B-associated OD pairs currently DISCONNECTED, and the
                             # fraction of total demand disconnected. Graph reachability on the
                             # severed-removed network -- no UE solve.
    feat_obs_trueD=False,    # one column + one global dim: the REALIZED demand-shortfall
                             # projection (the evaluator's own D recursion under true
                             # severities) and its total fraction -- the observed twin of the
                             # belief columns; their gap is the "world worse/better than
                             # believed" signal. Pure arithmetic -- no UE solve.
    hop_untied=True,     # Deviation 23: give EACH propagation round its own aggregation matrix
                         # th2^(t) instead of reusing one across all t_emb rounds. False is the
                         # paper's weight-tied structure2vec, byte-identical to before the flag
                         # existed (verified: same parameters, same forward output). ADOPTED as
                         # the default 2026-08-24 on the project owner's instruction after the
                         # measurement below; see the deviation-23 entry for what it does and
                         # does not change.
    n_step=2,            # n-step return horizon, inside the paper's per-problem set 1-5
                         # (Table D.9: MVC 5, MAXCUT 1, TSP 1, SCP 2)
    gamma=0.9995,        # discount, aligned with rl_dqn (deviations 6 and 14)
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
    # RANK_PARAMS_DQN as with the hinge: proportional priorities on a plain list (no sum-tree
    # -- the buffer is small), importance-sampling weights annealed by beta. The paper uses UNIFORM
    # replay, so prioritized=False restores it.
    prioritized=bool(RANK_PARAMS_DQN["prioritized"]),
    per_alpha=float(RANK_PARAMS_DQN["per_alpha"]),       # priority exponent
    per_beta0=float(RANK_PARAMS_DQN["per_beta0"]),       # IS-weight exponent at episode 0...
    per_beta_eps=float(RANK_PARAMS_DQN["per_beta_eps"]), # ...annealed to 1 over ~this many eps
    per_eps=float(RANK_PARAMS_DQN["per_eps"]),           # floor added to each |TD| priority
    # LARGE-MARGIN RANKING HINGE (added 2026-08-23 on the project owner's instruction). Its whole
    # configuration is TAKEN LIVE from RANK_PARAMS_DQN so this solver regularizes exactly as
    # rl_dqn does ("reference the current RL"). The exemplar is the best-by-F nominal order so
    # far; the term demands every rival action fall at least `margin` below the exemplar's Q, and
    # its weight climbs a geometric ramp lam(ep)=min(lam, lam0*lam_growth^(ep-1)). lam=0 removes
    # the term and recovers the FAITHFUL paper reproduction (see the module docstring).
    lam=float(RANK_PARAMS_DQN["lam"]),               # hinge weight cap / terminal weight
    margin=float(RANK_PARAMS_DQN["margin"]),         # m: exemplar action vs each rival gap
    lam0=float(RANK_PARAMS_DQN["lam0"]),             # imitation ramp start...
    lam_growth=float(RANK_PARAMS_DQN["lam_growth"]), # ...geometric growth/episode, capped at lam
)

def _resolve_variant(variant):
    """The hyperparameters a variant name implies, resolved through the SAME authorities the
    trainers use (S2V_PARAMS / S2V_SAA_PARAMS + the adaptive-knob derivation), so a checkpoint is
    never rebuilt under a different configuration than the one it was trained with.

    Lived in util/transfer.py until that feature was removed (2026-08-27); it moved here rather
    than dying with it because util/behavior.py loads delivered policies through it."""
    if variant == "rl_s2v":
        return dict(S2V_PARAMS)
    m = re.fullmatch(r"rl_s2v_saa(\d+)(_adaptive)?", variant)
    if not m:
        raise SystemExit(f"unknown variant {variant!r}: expected rl_s2v or "
                         f"rl_s2v_saa<pool>[_adaptive]")
    # Deferred so the import direction stays one-way: rl_s2v_saa imports from rl_s2v, never back.
    from util.rl_s2v_saa import S2V_SAA_PARAMS
    hp = dict(S2V_SAA_PARAMS, pool_n=int(m.group(1)), adaptive=bool(m.group(2)))
    if hp["adaptive"]:
        # Same derivation as train_s2v_saa: the knob equips the observation channels.
        hp = dict(hp, feat_obs_traffic=True, feat_obs_disc=True, feat_obs_trueD=True)
    return hp


# The plateau rule probes every episode here (the nominal best-so-far score is free), so the
# shared STOP_PARAMS patience -- calibrated in PROBES -- must be widened. The rank-loss DQN that
# once carried the twin override was removed 2026-08-27; this one stands on its own.
S2V_STOP_OVERRIDES = dict(patience_P=100, stable_K=100)   # 2026-08-24: 75 -> 100, repo-wide


def _merge_stop_s2v(stop_params):
    sp = dict(STOP_PARAMS)
    sp.update(S2V_STOP_OVERRIDES)
    sp.update(stop_params or {})
    return sp


# --------------------------------------------------------------------------- #
# Graph carrier and state assembly
# --------------------------------------------------------------------------- #
def _graph_tensors(env, toy_dir=TOY, hp=None):
    """Everything static the embedding needs, computed once per run, plus the layout of the
    OPTIONAL dynamic columns (deviation 22): `n_feat` is the x_v width under the caller's flags,
    col_proj / col_rec / col_blocked are the column indices of the enabled dynamic features
    (None when off), and B / den / rho are the constants the rollout needs to fill them.

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
    phi_all = np.array([flow.get(tuple(sorted(mp["end_of"][e])), 0.0) for e in segs_all],
                       dtype=np.float32)
    # The raw baseline maximum is kept: the deviation-24 live-flow column divides by the SAME
    # normalizer, so static phi and observed flow stay on one scale and their difference reads
    # as "flow lost vs normal".
    phi_max = float(phi_all.max() or 1.0)
    phi_all = phi_all / phi_max
    # Dynamic-column layout (deviation 22). Enabled columns are appended after the 7 base
    # columns in a fixed order, so disabling one shifts the ones after it -- the indices here are
    # the single authority and the rollout / _state_x read them, never hard-coded positions.
    fl = dict(S2V_PARAMS, **(hp or {}))
    n_feat = 7
    col_proj = col_rec = col_blocked = None
    col_oflow = col_ocong = col_odisc = col_otrue = None
    if bool(fl.get("feat_shortfall")):
        col_proj, n_feat = n_feat, n_feat + 1
    if bool(fl.get("feat_recency")):
        col_rec, n_feat = n_feat, n_feat + 1
    if bool(fl.get("feat_blocked")):
        col_blocked, n_feat = n_feat, n_feat + 1
    # Deviation-24 observation columns, appended after deviation 22's in the same fixed order.
    if bool(fl.get("feat_obs_traffic")):
        col_oflow, n_feat = n_feat, n_feat + 1
        col_ocong, n_feat = n_feat, n_feat + 1
    if bool(fl.get("feat_obs_disc")):
        col_odisc, n_feat = n_feat, n_feat + 1
    if bool(fl.get("feat_obs_trueD")):
        col_otrue, n_feat = n_feat, n_feat + 1
    # The global block g grows with the two observation totals, in this fixed order:
    # [t/T, crew_gap, belief_shortfall, remaining_work] + [realized_shortfall?] + [disc_demand?].
    g_dim = 4 + (1 if col_otrue is not None else 0) + (1 if col_odisc is not None else 0)
    xs = np.zeros((n, n_feat), dtype=np.float32)       # cols 0..6: pending, under, done, phi, sev, dur, dem
    xs[:, 3] = phi_all
    for e in env["segs"]:
        xs[idx[e], 4] = st["sev_hat"][e]
        xs[idx[e], 5] = st["dur_hat"][e]
        xs[idx[e], 6] = st["dem_hat"][e]
    # Constants for the projected-shortfall column: B's columns follow ctx["disrupted"] order
    # (env["seg_idx"]), and the denominator is each segment's full-drop level sum_r B[r,e] H0_r.
    ctx = env["ctx"]
    den = ctx["B"].T @ ctx["H0"]
    # Deviation-24 constants: the intact per-edge free-flow time (the congestion column's
    # reference), the (u,v)->edge_id map for folding directed UE links back onto segments, and
    # the observation memos -- ue_memo keyed exactly like the slot cache (slot,
    # capped-completion-prefix, severity tuple), disc_memo keyed by the severed subset, and the
    # actual-UE-solve counter the compute accounting reports. Mutable on purpose: they persist
    # across every rollout of a training run.
    fft0 = {int(r.edge_id): float(r.free_flow_time) for r in ctx["edges"].itertuples(index=False)}
    pair_eid = {tuple(sorted(uv)): e for e, uv in ctx["access"]["ends"].items()}
    return dict(segs_all=segs_all, idx=idx, A=A, deg=A.sum(axis=1), xs=xs, n_feat=n_feat,
                col_proj=col_proj, col_rec=col_rec, col_blocked=col_blocked,
                col_oflow=col_oflow, col_ocong=col_ocong, col_odisc=col_odisc,
                col_otrue=col_otrue, g_dim=g_dim, phi_max=phi_max, fft0=fft0,
                pair_eid=pair_eid, obs=dict(ue_memo={}, disc_memo={}, solves=[0]),
                B=ctx["B"], den=den, rho=float(P.RHO))


def _state_x(gt, state):
    """The tagged node-feature matrix for one decision state: the static columns, the 3-way
    tag set from what has OBSERVABLY happened (pending = damaged & unscheduled, under = crew on
    it and not yet complete, done = everything else damaged), and -- when enabled (deviation 22)
    -- the dynamic columns the rollout froze onto the state at decision time: the projected
    shortfall map, the completion-recency map, and the blocked indicator (pending minus the
    accessible candidate set)."""
    x = gt["xs"].copy()
    for e in state["pending"]:
        x[gt["idx"][e], 0] = 1.0
    for e in state["under"]:
        x[gt["idx"][e], 1] = 1.0
    for e in state["done"]:
        x[gt["idx"][e], 2] = 1.0
    if gt["col_proj"] is not None:
        for e, v in state.get("proj", {}).items():
            x[gt["idx"][e], gt["col_proj"]] = v
    if gt["col_rec"] is not None:
        for e, v in state.get("rec", {}).items():
            x[gt["idx"][e], gt["col_rec"]] = v
    if gt["col_blocked"] is not None:
        blocked = set(state["pending"]) - set(state["cand"])
        for e in blocked:
            x[gt["idx"][e], gt["col_blocked"]] = 1.0
    # Deviation-24 observation columns, frozen onto the state at decision time exactly like
    # proj/rec above: flow and congestion cover ALL segments (network-wide sensing), the
    # disconnection and realized-shortfall projections exist for the damaged set only.
    if gt["col_oflow"] is not None:
        for e, v in state.get("oflow", {}).items():
            x[gt["idx"][e], gt["col_oflow"]] = v
        for e, v in state.get("ocong", {}).items():
            x[gt["idx"][e], gt["col_ocong"]] = v
    if gt["col_odisc"] is not None:
        for e, v in state.get("odisc", {}).items():
            x[gt["idx"][e], gt["col_odisc"]] = v
    if gt["col_otrue"] is not None:
        for e, v in state.get("otrue", {}).items():
            x[gt["idx"][e], gt["col_otrue"]] = v
    return x


# --------------------------------------------------------------------------- #
# Execution-time observation (deviation 24)
# --------------------------------------------------------------------------- #
def _sev_true_of(env, durations):
    """The severities the OBSERVATION runs under, resolved exactly as
    util.evaluate.evaluate_schedule resolves scoring: a util.scenarios.Scenario carries the truth
    on `.sev`; the nominal world (a plain dict) carries none and the instance ESTIMATES stand in.
    One resolution rule means the agent always observes the world it is scored in -- a third
    possibility is exactly the silent state/score mismatch this project treats as the worst
    failure mode."""
    dis = env["ctx"]["disrupted"]
    return ({int(e): int(v) for e, v in getattr(durations, "sev", {}).items()}
            or {int(eid): int(s) for (eid, _, _, s) in dis})


def _make_observer(env, gt, durations):
    """One episode's field-state observer: returns advance(t, comp) -> the deviation-24
    observations at slot t, given the completions recorded so far (comp maps segment ->
    completion slot; entries may exceed t -- a crew's busy-until is already state the rollout
    legally knows). Everything here is computable from the history up to t: the damage state at
    any slot j <= t is fixed by the completions <= j, so no future information can leak in.

    Maintains the evaluator's OWN realized demand-shortfall recursion (true severities), advanced
    slot by slot exactly as evaluate_schedule advances it. Per call it returns:
      D_true  -- the realized shortfall vector at slot t (feat_obs_trueD);
      disc    -- per-OD boolean: destination unreachable from origin on the severed-removed
                 network (feat_obs_disc). The network is built by the SAME build_damaged_edges
                 the evaluator uses, so what counts as severed has a single authority; memoized
                 by the severed subset (reachability ignores capacity degradation).
      flow/cong -- per-segment live two-way flow (÷ the baseline maximum, the static phi
                 column's normalizer) and congestion excess x/(1+x), x = max(cost/fft0 - 1, 0)
                 over the two directions (feat_obs_traffic). One COLD UE solve per distinct
                 (slot, capped completion prefix, severity tuple) -- the slot cache's own
                 invariant, see util/sim_cache.py -- memoized in gt["obs"]["ue_memo"]. Cold on
                 purpose: deterministic and key-cacheable, and the warm-chain difference is
                 inside the solver tolerance anyway. A severed segment reads flow 0, cong 0 --
                 a sensor on a closed road, not a severity label.
    """
    ctx = env["ctx"]
    dis = ctx["disrupted"]
    H0, B = ctx["H0"], ctx["B"]
    sev_true = _sev_true_of(env, durations)
    sev_key = tuple(sev_true[eid] for (eid, _, _, _) in dis)
    need_ue = gt["col_oflow"] is not None
    box = dict(k=0, D=np.zeros(len(H0)))

    def _damaged_at(j, comp):
        return {eid: sev_true[eid] for (eid, _, _, _) in dis
                if not (eid in comp and comp[eid] <= j)}

    def advance(t, comp):
        while box["k"] < t:                            # evaluate_schedule's recursion, verbatim
            box["k"] += 1
            dmg_j = _damaged_at(box["k"], comp)
            v = np.zeros(len(dis))
            for j, (eid, _, _, _) in enumerate(dis):
                if eid in dmg_j:
                    v[j] = sev_true[eid]
            box["D"] = np.maximum(B @ v, P.RHO * box["D"])
        dmg = _damaged_at(t, comp)
        out = dict(D_true=box["D"].copy())
        if gt["col_odisc"] is not None:
            severed = frozenset(e for e, s in dmg.items() if s >= P.SEVER_SEVERITY)
            dm = gt["obs"]["disc_memo"]
            if severed not in dm:
                edges_now = build_damaged_edges(ctx, {e: sev_true[e] for e in severed})
                adj = {}
                for r in edges_now.itertuples(index=False):
                    adj.setdefault(int(r.u), []).append(int(r.v))
                    adj.setdefault(int(r.v), []).append(int(r.u))
                reach = {}
                for o in ctx["origins_unique"]:
                    seen, stack = {o}, [o]
                    while stack:
                        nd = stack.pop()
                        for nb in adj.get(nd, ()):
                            if nb not in seen:
                                seen.add(nb)
                                stack.append(nb)
                    reach[o] = seen
                dm[severed] = np.array([d not in reach[o] for (o, d) in ctx["od_pairs"]])
            out["disc"] = dm[severed]
        if need_ue:
            key = (t, tuple(min(comp.get(eid, t + 1), t + 1) for (eid, _, _, _) in dis), sev_key)
            tm = gt["obs"]["ue_memo"]
            if key not in tm:
                H = np.clip(H0 - box["D"], 0.0, None)
                links, _ = solve_ue(build_damaged_edges(ctx, dmg), _matrix_from_H(H, ctx),
                                    ctx["zone_ids"], rgap=P.UE_RGAP, max_iter=P.UE_MAX_ITER,
                                    quiet=True, cores=1)
                gt["obs"]["solves"][0] += 1
                vol, cng = {}, {}
                for a, b, vv, cc in zip(links["from"].to_numpy(), links["to"].to_numpy(),
                                        links["volume"].to_numpy(), links["cost"].to_numpy()):
                    eid = gt["pair_eid"].get(tuple(sorted((int(a), int(b)))))
                    if eid is None:
                        continue
                    vol[eid] = vol.get(eid, 0.0) + float(vv)
                    cng[eid] = max(cng.get(eid, 0.0),
                                   max(float(cc) / gt["fft0"][eid] - 1.0, 0.0))
                tm[key] = ({e: vol.get(e, 0.0) / gt["phi_max"] for e in gt["segs_all"]},
                           {e: cng.get(e, 0.0) / (1.0 + cng.get(e, 0.0))
                            for e in gt["segs_all"]})
            out["flow"], out["cong"] = tm[key]
        return out

    return advance


def _build_s2v_net(p, t_emb, use_g, torch, nn, dueling=False, readout_hidden=0, in_dim=7,
                   hop_untied=False, g_dim=4):
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

            def _head(n_in):
                """A readout head: the paper's single bias-free linear when readout_hidden=0,
                else the deviation-21 two-layer MLP."""
                if readout_hidden <= 0:
                    return nn.Linear(n_in, 1, bias=False)
                return nn.Sequential(nn.Linear(n_in, readout_hidden, bias=False), nn.ReLU(),
                                     nn.Linear(readout_hidden, 1, bias=False))
            self.th1 = nn.Linear(in_dim, p, bias=False)   # in_dim = gt["n_feat"]: 7 base
                                                          # columns + the enabled deviation-22
                                                          # dynamic columns
            # th2: the neighbour-aggregation map. The paper REUSES one matrix across every
            # propagation round (structure2vec is a fixed-point iteration, so sharing is intrinsic
            # to its derivation). With hop_untied (deviation 23) each round gets its own, which is
            # the ordinary stacked-GNN parameterization; the ModuleList is indexed by round, so
            # dropping the flag restores the single shared matrix exactly.
            self.th2 = (nn.ModuleList([nn.Linear(p, p, bias=False) for _ in range(t_emb)])
                        if hop_untied else nn.Linear(p, p, bias=False))
            self.hop_untied = hop_untied
            self.th3 = nn.Linear(p, p, bias=False)
            self.th4 = nn.Parameter(torch.randn(p) * 0.1)
            self.th6 = nn.Linear(p, p, bias=False)
            self.th7 = nn.Linear(p, p, bias=False)
            # g_dim = gt["g_dim"]: 4 base dims plus the deviation-24 observation totals.
            self.th8 = nn.Linear(g_dim, p, bias=False) if use_g else None
            # th5: the Eq.-4 readout head. With readout_hidden > 0 it is a small MLP
            # (Linear -> relu -> Linear, deviation 21) instead of the paper's single linear.
            self.th5 = _head(3 * p if use_g else 2 * p)
            self.dueling = dueling
            # DUELING-ONLY (removable with the flag): a state-value head on the pooled embedding
            # (+ g). th5 above then serves as the advantage head. None when not dueling.
            self.th5_val = (_head(2 * p if use_g else p) if dueling else None)

        def forward(self, x, A, deg, g):
            # structure2vec: mu <- relu(th1 x + th2 sum_nbr mu + th3 (deg * relu(th4)))
            pre = self.th1(x) + self.th3(deg.unsqueeze(1) * torch.relu(self.th4).unsqueeze(0))
            mu = torch.zeros(x.shape[0], p)
            for _t in range(t_emb):
                # UNTIED (deviation 23): round _t uses its own th2. Note this does not give a
                # per-HOP parameter -- after T rounds a k-hop contribution has passed through a
                # PRODUCT of k of these matrices, so round index and hop distance stay entangled.
                # It lets successive rounds transform differently, nothing more.
                th2 = self.th2[_t] if self.hop_untied else self.th2
                mu = torch.relu(pre + th2(A @ mu))
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


def _s2v_rollout(env, gt, pick, durations=None):
    """One episode on the line-graph state: the same crew scheduling and demand-shortfall
    recursion as util.rl_rank._rollout (the schedule/demand block is copied from it verbatim --
    _self_check asserts the two stay in lockstep), but the recorded per-decision state is the
    tagged-graph state this solver's network consumes, not a hand-crafted feature matrix.

    Returns (perm, start, states, picks): `states` are dicts carrying the world tags -- the
    pending, under-repair and done tuples -- plus the global block g (4 base dims + the enabled
    deviation-24 observation totals), the frozen observation maps, and `cand`, the ACCESSIBLE
    subset of pending that forms the action set; picks index into cand. Duration BELIEFS stay at
    the planning expectation throughout, exactly as in _rollout; what the deviation-24 channels
    add is the OBSERVED field state of this episode's world, never its labels."""
    ctx, st, T = env["ctx"], env["st"], env["T"]
    durations = env["nominal"] if durations is None else durations
    dis_l, sev, B = ctx["disrupted"], ctx["severity_vec"], ctx["B"]
    seg_idx = env["seg_idx"]
    access = ctx["access"]
    # Deviation-24 field-state observer for THIS episode's world (None when every observation
    # flag is off). Fresh per episode -- its shortfall recursion starts at onset -- while its
    # expensive memos persist across episodes inside gt["obs"].
    obs_fn = (_make_observer(env, gt, durations)
              if (gt["col_oflow"] is not None or gt["col_odisc"] is not None
                  or gt["col_otrue"] is not None) else None)
    remaining = list(env["segs"])
    crew = [1] * P.C_MAX
    start, perm, states, picks = {}, [], [], []
    D = np.zeros(len(ctx["H0"]))
    k, comp = 0, {}
    while remaining:
        t = min(crew)
        # Accessibility constraint (mirrors util.rl_rank._rollout): the ACTION set is the
        # accessible remaining segments; when none is reachable the free crew waits for the next
        # completion. The graph TAGS below keep describing the whole world (a blocked segment is
        # still pending damage) -- only state["cand"] restricts what may be chosen.
        blocked = set(remaining) | {e for e, c in comp.items() if c > t}
        acc = accessible_segments(access, blocked, remaining)
        while not acc:
            busy = [c for c in crew if c > t]
            if not busy:
                raise RuntimeError(f"accessibility deadlock at slot {t}: {sorted(remaining)} "
                                   f"unreachable with no repair underway")
            nxt = min(busy)
            crew = [max(c, nxt) for c in crew]
            t = min(crew)
            blocked = set(remaining) | {e for e, c in comp.items() if c > t}
            acc = accessible_segments(access, blocked, remaining)
        cand = tuple(e for e in remaining if e in acc)
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
        under = tuple(e for e, c in comp.items() if c > t)
        # Deviation-24 observations at slot t, frozen onto the state like proj/rec below. The
        # projections reuse the SAME B-projection and denominator as the belief column, so the
        # otrue-vs-proj gap is a like-for-like "world worse/better than believed" signal; the
        # disconnection share is B-weighted demand, an OD-level quantity that never becomes a
        # per-segment severed label. g grows by the two totals in _graph_tensors' fixed order.
        oflow, ocong, odisc, otrue = {}, {}, {}, {}
        g_ext = []
        if obs_fn is not None:
            ob = obs_fn(t, comp)
            if gt["col_otrue"] is not None:
                BDt = gt["B"].T @ ob["D_true"]
                otrue = {e: (float(BDt[j] / gt["den"][j]) if gt["den"][j] > 0 else 0.0)
                         for e, j in seg_idx.items()}
                g_ext.append(float(ob["D_true"].sum()) / st["sum_H0"])
            if gt["col_odisc"] is not None:
                num = gt["B"].T @ (ctx["H0"] * ob["disc"])
                odisc = {e: (float(num[j] / gt["den"][j]) if gt["den"][j] > 0 else 0.0)
                         for e, j in seg_idx.items()}
                g_ext.append(float(ctx["H0"][ob["disc"]].sum()) / st["sum_H0"])
            if gt["col_oflow"] is not None:
                oflow, ocong = ob["flow"], ob["cong"]
        g = np.array([t / T, (max(crew) - t) / T, float(D.sum()) / st["sum_H0"],
                      sum(bel[e] for e in remaining) / st["total_work"]] + g_ext,
                     dtype=np.float32)
        # Dynamic node columns (deviation 22), FROZEN onto the state at decision time so replayed
        # states re-embed exactly what was observable when the decision was made. proj carries
        # WHERE the current shortfall sits (B-projection of the belief-D the recursion above
        # maintains, normalized by each segment's full-drop level); rec is the demand law's own
        # decay kernel -- 1 while a segment is still damaged, rho^(elapsed since completion)
        # after -- which is what makes completion HISTORY visible. The blocked indicator needs no
        # extra state: _state_x derives it from pending minus cand.
        proj, rec = {}, {}
        if gt["col_proj"] is not None:
            BD = gt["B"].T @ D
            proj = {e: (float(BD[j] / gt["den"][j]) if gt["den"][j] > 0 else 0.0)
                    for e, j in seg_idx.items()}
        if gt["col_rec"] is not None:
            rec = {e: (float(gt["rho"] ** (t - comp[e])) if (e in comp and comp[e] <= t) else 1.0)
                   for e in env["segs"]}
        state = dict(pending=tuple(remaining), under=under,
                     done=tuple(e for e in comp if comp[e] <= t), g=g, cand=cand,
                     proj=proj, rec=rec, oflow=oflow, ocong=ocong, odisc=odisc, otrue=otrue)
        states.append(state)
        i = pick(cand, state)
        picks.append(i)
        e = cand[i]
        remaining.remove(e)
        c = int(np.argmin(crew))
        start[e] = crew[c]
        crew[c] = start[e] + int(durations[e])
        comp[e] = crew[c]
        perm.append(e)
    return tuple(perm), start, states, picks


def _self_check(env, gt, Qf, torch):
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
    perm_a, start_a, states, _ = _s2v_rollout(env, gt, fp)
    perm_b, start_b, feats, _, _ = _rollout(env, fp, n_lag=0)
    assert perm_a == perm_b and start_a == start_b, "s2v rollout schedules differently than rl_rank"
    for s, X in zip(states, feats):
        assert np.allclose(s["g"][:4], np.asarray(X)[0, 4:8], atol=1e-6), (
            "s2v rollout's global state diverged from rl_rank._rollout's -- the copied "
            "schedule/demand recursion has drifted")
    # Deviation-24 invariants, on the same nominal rollout. In the NOMINAL world truth ==
    # estimate by construction, so the observed-truth channel must REPRODUCE the belief channel
    # exactly -- any daylight means the observer's recursion drifted from the rollout's (the
    # silent state corruption this check exists to catch).
    if gt["col_otrue"] is not None and gt["col_proj"] is not None:
        for s in states:
            assert all(abs(s["otrue"][e] - s["proj"].get(e, 0.0)) < 1e-9 for e in s["otrue"]), \
                "nominal-world realized-shortfall projection != belief projection"
            assert abs(float(s["g"][4]) - float(s["g"][2])) < 1e-9, \
                "nominal-world realized-shortfall total != belief total"
    if gt["col_oflow"] is not None:
        st0 = states[0]
        for eid, _, _, sv in env["ctx"]["disrupted"]:
            if sv >= P.SEVER_SEVERITY:                 # nominal world: estimate IS the truth
                assert st0["oflow"][eid] == 0.0 and st0["ocong"][eid] == 0.0, (
                    "a severed segment shows traffic -- the observation read the wrong network")
        assert 0.0 < max(st0["ocong"].values()) < 1.0, "no congestion reading anywhere at onset"
        assert all(np.isfinite(v) and v >= 0.0 for v in st0["oflow"].values()), \
            "live-flow column contains a non-finite or negative reading"
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
    the retired util.rl_rank.train so the runner keeps the provenance behaviour that solver
    established (that trainer and its runner were removed 2026-08-27)."""
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
    readout_hidden = int(hp["readout_hidden"])
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

    gt = _graph_tensors(env, hp=hp)
    A = torch.tensor(gt["A"])
    deg = torch.tensor(gt["deg"], dtype=torch.float32)
    hop_untied = bool(hp["hop_untied"])
    net = _build_s2v_net(p, t_emb, use_g, torch, nn, dueling=dueling,
                         readout_hidden=readout_hidden, in_dim=gt["n_feat"],
                         hop_untied=hop_untied, g_dim=gt["g_dim"])
    tgt = _build_s2v_net(p, t_emb, use_g, torch, nn, dueling=dueling,
                         readout_hidden=readout_hidden, in_dim=gt["n_feat"],
                         hop_untied=hop_untied, g_dim=gt["g_dim"])
    tgt.load_state_dict(net.state_dict())
    opt = torch.optim.Adam(net.parameters(), lr=float(hp["lr"]))

    def _q(model, state):
        """Q over one state's ACCESSIBLE candidates (state["cand"]), in their remaining-list
        order (the gather IS the legality mask: scheduled AND currently unreachable segments'
        vertices never reach any consumer, so the TD max, the hinge and the dueling mean all
        range over exactly the legal actions)."""
        x = torch.tensor(_state_x(gt, state))
        idx = [gt["idx"][e] for e in state["cand"]]
        out = model(x, A, deg, torch.tensor(state["g"]))
        if not dueling:
            return out[idx]
        # DUELING-ONLY (removable with the flag): Q = V + A - mean_legal(A). The mean is over
        # the ACCESSIBLE (legal) advantages, so the identifiability subtraction and the argmax
        # both range over exactly the live actions -- matching rl_rank's dueling aggregation.
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
        deterministically and return (states, picks). picks[i] indexes into states[i]['cand']
        (the accessible candidates in remaining-list order), so Q[pick] is the exemplar action's
        Q; under the accessibility constraint a fixed order rolls out with skip semantics --
        first accessible segment in the order -- exactly as the delivery scheduler does."""
        def fp(rem, _state):
            for e in order:
                if e in rem:
                    return rem.index(e)
        _, _, sts, pk_ = _s2v_rollout(env, gt, fp)
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

    _self_check(env, gt, Qf, torch)

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
        # Imitation ramp, exactly rl_dqn's: min(lam_cap, lam0*lam_growth^(ep-1)); constant
        # lam_cap when no ramp is configured. 0 when the hinge is off.
        lam_ep = (0.0 if not use_hinge else
                  lam_cap if lam_growth is None else
                  min(lam_cap, float(lam0) * float(lam_growth) ** max(0, ep - 1)))

        def pk(rem, state):
            if rng.rand() < eps:
                return rng.randint(len(rem))
            with torch.no_grad():
                return int(torch.argmax(Qf(state)))

        perm, start, states, picks = _s2v_rollout(env, gt, pk)
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
        # decision states (rl_dqn recomputes states_for(best_perm) per update; here the order
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

        # Plateau probe, the rl_dqn wiring: score = best-so-far nominal F (free), order = the
        # greedy policy's nominal rollout; the frozen-scenario diagnostic every probe_every
        # episodes, list-scheduling the greedy order as the cheap proxy, and NEVER into the stop.
        gperm, _, _, _ = _s2v_rollout(env, gt, greedy)
        scen_F = None
        if ep % sp["probe_every"] == 0:
            scen_F = float(np.mean([
                _evaluate_prefix_cached(schedule_from_permutation(list(gperm), d,
                                                                  access=env["ctx"]["access"]), d,
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

    # Delivery: the FINAL policy rolled inside each frozen scenario, exactly rl_dqn's
    # semantics (observed history only, beliefs at the planning expectation).
    per_scenario = []
    for d in env["scen"]:
        gp, gs, _, _ = _s2v_rollout(env, gt, greedy, d)
        per_scenario.append((list(gp), gs))
    order = list(_s2v_rollout(env, gt, greedy)[0])     # nominal-world summary only
    if verbose:
        print(f"  [rl_s2v] stopped at ep {ep} ({outcome}) in "
              f"{(time.perf_counter() - t0) / 60:.1f} min", flush=True)
    return dict(order=order, per_scenario=per_scenario, net=net, best_net_sd=best_sd,
                episodes=ep, n_evals=len(memo), outcome=outcome, best_score=best_F, hp=hp,
                seed=seed, obs_solves=int(gt["obs"]["solves"][0]))


# --------------------------------------------------------------------------- #
# Run + provenance (the file family and two-stage clear established by util.rl_rank.run_rank,
# which was removed with the rl_dqn solver on 2026-08-27)
# --------------------------------------------------------------------------- #
def run_s2v(toy_dir=TOY, N=None, M=P.M_SCENARIOS, seed=P.SEED, ep_cap=EP_CAP, hp=None,
            stop_params=None):
    """Train rl_s2v to a plateau and write its canonical results and diagnostics to
    outputs/03-rl/{solver_dir('rl_s2v')}/n{N}/, then refresh the comparison."""
    import torch
    N = P.N_DISRUPTED_ORACLE if N is None else N
    env = build_env(toy_dir, N=N, M=M)
    print(f"instance: {len(env['segs'])} segments {env['segs']}; M={M}; T={env['T']}; "
          f"seed={seed}; ep_cap={ep_cap}; variant=rl_s2v", flush=True)
    v = "rl_s2v"
    vdir = scale_dir(OUT_DIAG / solver_dir(v), N)
    vdir.mkdir(parents=True, exist_ok=True)
    # Stage 1 of the two-stage clear: diagnostics only, so a run that dies mid-training cannot
    # destroy a delivered model it never reproduced (the rule run_rank set).
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
                   # plus this scenario's own evaluation (see util.compare._compute_accounting).
                   # The deviation-24 live-traffic reads are REAL solves too and are charged the
                   # same amortised way (obs_solves counts actual solver calls, memo hits free).
                   ue_total=((r["n_evals"] * env["T"] + r.get("obs_solves", 0))
                             / len(env["scen"]) + env["T"]),
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
                       "gamma=0.9995 aligned with rl_dqn (paper reading implies 1.0)",
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
                       "large-margin ranking hinge from RANK_PARAMS_DQN (the current "
                       "default; lam=0 removes it and restores the paper's method)",
                       "Double DQN target (online selects, target evaluates); not in the paper, "
                       "double_dqn=False restores the plain max-Q target",
                       "prioritized experience replay from RANK_PARAMS_DQN; paper uses "
                       "uniform replay, prioritized=False restores it",
                       "dueling head V(s)+A(s,v)-mean_legal(A) (Wang et al. 2016); not in the "
                       "paper, isolated for reversion, dueling=False restores single head",
                       "deeper readout: th5 (and dueling value head) as a small MLP with "
                       "hidden width readout_hidden; 0 restores the paper's single linear",
                       "execution-time observation channels (deviation 24) exist behind the "
                       "feat_obs_* flags but are OFF in this solver by measurement (nominal "
                       "training gives them zero signal; 0.9806 on vs 0.9774 off); they are ON "
                       "in rl_s2v_saa -- the hp block records this run's actual flags"])
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
