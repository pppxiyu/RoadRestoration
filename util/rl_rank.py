"""
Ranking-loss RL solver for the restoration-ordering problem: the same DQN state/action model as
util.rl, trained against an objective that optimizes the ORDER Q induces rather than the values Q
predicts.

WHY THIS EXISTS. The value-based solver in util.rl fits Q by TD regression, so its loss measures
how well Q predicts returns. The delivered decision, however, is argmax_e Q(s, e): only the RANKING
of Q within a state matters, and its absolute calibration does not. Under function approximation
those two objectives come apart -- a parameter step that lowers mean squared TD error can, and here
routinely does, flip an argmax and make the delivered order worse. This was isolated with a
warm-start probe: initialized AT the best known order, the value-based learner drifts away from it
within a few episodes, and no value-stability knob (n-step returns, gamma < 1, Double DQN,
potential-based shaping, lower lr, Monte-Carlo targets) prevents that drift. Replacing the loss
does prevent it, which is what this module implements.

TWO RANKING LOSSES, both supervised by a SELF-IMITATION target -- a good trajectory the agent
itself produced, not an external demonstration:

  margin (the winner of both hyperparameter searches, and the only loss this module carries):

      L = mean_batch (Q - y)^2  +  lam * mean_state sum_{e != a} max(0, m - (Q(s,a) - Q(s,e)))

    a TD regression term that keeps Q calibrated, plus a large-margin hinge that forces the
    exemplar's action a to outscore every rival by at least m. The hinge is an INEQUALITY
    CONSTRAINT, not a competing objective: once every gap reaches m its gradient is exactly zero,
    so the regression term is free to calibrate in the slack. The margin (rather than merely
    requiring a positive gap) buys robustness -- a gap of 0+ is flipped by the next regression
    step. The hinge term follows DQfD (Hester et al., AAAI 2018), which introduced it for HUMAN
    demonstrations; here the exemplar is the agent's own best trajectory, after the self-imitation
    idea of Oh et al. (ICML 2018). Two deliberate departures from DQfD are worth naming: the
    penalty is summed over every violating action rather than taken at the single worst one (denser
    gradient, and the action set shrinks as repairs complete), and the exemplar MOVES during
    training rather than being a fixed demonstration set.

(A second ranking loss, sil_ce -- pure softmax cross-entropy toward the exemplar, no TD term --
also fixed the drift during the diagnosis but lost both hyperparameter searches to margin and was
removed on 2026-08-10; the historical comparison lives in the run records, not in this module.)

THE TWO VARIANTS mirror util.rl. The nominal variant ("rl_nominal") trains in the nominal world, its
exemplar is the best-by-F order found so far, and it delivers ONE committed order. The stochastic
variant ("rl_saa") draws a fresh duration realization every episode (util.scenarios.draw_durations,
the law the evaluation sample is drawn from) and delivers the POLICY, rolled out per scenario.
F values from different worlds are not comparable, so the nominal variant's single best-by-F
exemplar does not exist there; instead a top-K buffer holds the lowest-F_w trajectories seen and
the ranking loss imitates one SAMPLED from it. Sampling rather than always taking the buffer
minimum is what keeps a single lucky world -- one whose durations happened to be short -- from
becoming the training target: an action reinforced across several buffered trajectories is one that
was good in several different worlds.

STOPPING. Both variants stop on a plateau rather than at a fixed episode count (RankPlateauStop),
so a converged run does not keep burning evaluations. The nominal variant watches the best order it
has found so far, which costs nothing extra. The stochastic variant has no such cross-world best, so
it watches the greedy policy's mean F over VALIDATION worlds drawn once up front from a dedicated
RNG stream. Those worlds are deliberately disjoint from the M evaluation scenarios: stopping on the
evaluation sample would make the reported mean an optimistic best-of over the test set. Either
plateau signal ends a run -- no improvement for `patience_P` probes, or a greedy order unchanged
for `stable_K` probes -- once past an `ep_min` floor, with `ep_cap` as the outer guard. Both
variants deliver the FINAL policy at the stop.

Alongside that, and never feeding it, the stochastic variant also records the greedy policy's mean
F on the M FROZEN EVALUATION SCENARIOS at every probe. That is the ruler the methods are finally
compared on, so it must not enter any decision; recording it anyway is what lets a finished run
answer "what would this policy have delivered at episode k?" without being re-trained, which is a
question the validation curve cannot answer (different worlds, different horizon).

DIAGNOSTICS. Every run records what the later analyses need, flushed periodically so an interrupted
run keeps everything up to the last flush: per-episode objective/loss/predicted-Q-vs-TD-target
traces, a FIXED probe of decision states whose Q values and last-hidden-layer representations are
snapshotted across training, and, for the delivered policy, per-candidate features with Q and the
chosen flag plus Q against the realized Monte-Carlo return. See _Recorder.

Run inside the road_restore conda env (PYTHONPATH = project root):
  python -m util.rl_rank
"""
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

import config as P
from util.evaluate import (_matrix_from_H, build_context, build_damaged_edges,
                           schedule_from_permutation)
from util.oracle import (_baseline_twoway_flow, compute_horizon, scale_dir,
                         select_oracle_instance)
from util.provenance import (solver_dir, fresh_scale_dir, log_dir, results_dir, slot_rows,
                             write_run_meta)
from util.rl import (_completion_rewards, _decision_features, _evaluate_prefix_cached,
                     _scenario_statics)
from util.scenarios import (draw_durations, nominal_durations, saa_lhs_sample, sample_scenarios,
                            worst_case_durations)
from util.ue import solve_ue

ROOT = Path(__file__).resolve().parent.parent
TOY = ROOT / "data" / "siouxfalls_toy"
OUT_DIAG = ROOT / "outputs" / "03-RL" # each variant owns outputs/03-RL/{variant}/n{N}/
_N_FEAT = 8                          # base per-candidate features; see util.rl._decision_features

# Feature-column names, in the order _decision_features emits them. Carried here because every
# downstream analysis (policy-vs-intuition correlations, demand effects) indexes by name rather
# than by position, and a silent reordering upstream would otherwise mislabel every figure.
FEAT_COLS = ["phi", "sev", "dur", "dem_hat", "t_frac", "crew_gap", "demand_shortfall", "rem_work",
             # the six message-passing columns appended when mp_rounds=2 (see _nbr_cols):
             # round-1 neighbourhood means, then the neighbourhood means of those means
             "mp1_phi", "mp1_dem", "mp1_flow", "mp2_phi", "mp2_dem", "mp2_flow"]

# --------------------------------------------------------------------------- #
# Tuned hyperparameters
# --------------------------------------------------------------------------- #
# Both configurations come from continuous Optuna (TPE) searches over the FULL joint space --
# loss, architecture, n_lag, the exploration schedule, the prior weight, lr, update count, batch,
# and (for margin) the margin/lam weights plus the value-stability knobs -- with a DIFFERENT seed
# per trial, which makes the search an implicit multi-restart and so attacks the exploration limit
# at the same time. Objective: the delivered scenario-mean F, the same ruler the GA is scored on.
#
# What the searches agreed on, across ~100 trials each and both variants: margin beats sil_ce
# everywhere; exploration wants to be BROAD (eps0 ~0.5, far above the value-based solver's 0.2);
# Double DQN helps as a COMPONENT of margin even though it did nothing for plain TD regression;
# and the prior weight settles well below the value-based default.
#
# CAVEAT that must travel with these numbers: this solver's seed-to-seed spread is large (the
# stochastic variant delivered 0.937 at one seed and ~0.951 at another under the same
# configuration), which exceeds the gap between competing methods. Single-seed numbers from this
# module are not evidence; report a multi-seed mean and spread.
RANK_PARAMS_NOMINAL = dict(
    # ARCHITECTURE (adopted 2026-08-13, replacing the row architecture as the default).
    # Classic DQN wiring after Mnih et al. (2015): the STATE goes in and one Q comes out per
    # action, instead of scoring one (state, candidate) PAIR per network call. The action set here
    # shrinks as segments are repaired while a head has fixed width, so the head spans all n
    # segments and a LEGALITY MASK selects the live ones -- implemented by gathering, so an illegal
    # action never reaches any consumer. Dueling (Wang et al., 2016) splits the trunk into a scalar
    # V(s) and a per-segment advantage, aggregated over the LEGAL actions only; trunk_rescale is
    # that paper's 1/sqrt(2) correction for the two streams' gradients summing in the shared trunk.
    #
    # Measured on n10, five paired seeds, against the row architecture it replaces (medians):
    # row 0.955483 (spread 0.0133) -> dqn 0.952152 (0.0065) -> dqn+dueling+rescale 0.949892
    # (0.0066), against GA's 0.949234. The two dueling switches only work TOGETHER: dueling alone
    # left the median unmoved and inflated the spread to 0.0109, because the shared trunk receives
    # both streams' gradients and is effectively over-stepped without the rescale.
    #
    # The row architecture is NOT deleted -- arch="rows" still selects it, so earlier results stay
    # reproducible.
    arch="dqn",
    dueling=True,
    trunk_rescale=True,
    loss="margin",       # TD regression + large-margin hinge (the only supported loss)
    hidden=(32, 32),     # widths of the MLP hidden layers (tanh); the output layer starts at zero
    n_lag=1,             # append the base features of the last n_lag REPAIRED segments: an explicit
                         # finite memory, which both searches selected and which is direct evidence
                         # that the state as featurized is not fully observed
    prior_scale=0.5639,  # weight of the flow prior phi_hat in Q = prior*phi_hat + h(x)
    lr=9.979e-3,
    eps0=0.4980,         # eps-greedy rate at episode 1 (episode 0 is forced greedy = the flow order)
    eps_decay=0.9928,
    eps_min=0.1141,
    updates_per_ep=30,
    batch=16,
    margin=0.0689,       # m: the gap the hinge demands between the exemplar action and each rival
    lam=1.0,             # weight of the hinge relative to the TD regression term. With the ramp
                         # below this is the CAP, i.e. the terminal weight the schedule climbs to.
    # Imitation ramp: lam(ep) = min(lam, lam0 * lam_growth^(ep-1)), the MIRROR of the eps decay
    # (eps explores the action space early and follows Q late; lam lets the greedy order roam free
    # of the best-so-far order early and tightens onto it late).
    #
    # The 2026-08-10 constant-lam screen (seed 42, M=50, thresholds 75/75) put the whole useful
    # range at or below 1.0: constant 1.0 was the best of every configuration tried (0.9434),
    # constant 0.5 was next (0.9451), and constant 2.92 was worst (0.9484). The cap is therefore 1.0
    # -- the best constant -- and the ramp only decides how long the run stays BELOW it.
    #
    # The first ramp attempt (cap 2.9158, growth 1.01425, reaching 1.0 at ep 50) delivered 0.9484,
    # exactly base, and its trace is why this one is slower. Its best order improved at episodes
    # 0, 1, 9, 12, 35, 39, 50 and then never again, so all of its search happened while lam was in
    # 0.5-1.0 and the entire climb above 1.0 acted on a search that had already stopped. Two things
    # follow, both recorded because they contradict the reasoning that produced that attempt:
    # the ep-50 freeze is NOT caused by lam being strong there (this run was weaker than the 1.0
    # constant throughout and froze at the same episode), and there is no delivery gap for a late
    # clamp to close (the final greedy order equalled the delivered order exactly).
    #
    # Ramp start and rate come from the 2026-08-11 structure campaign (sandbox, seed 42, M=50),
    # which screened the start at 0.5 / 0.35 / 0.25 with the cap position held at ~ep 150:
    # 0.9431 / 0.9421 / 0.9488. The response is U-shaped and 0.35 is its floor -- freer early
    # roaming helps until the imitation is too weak to consolidate what the roaming finds. A later
    # cap (~ep 200) was also tried and lost (0.9451, delivery under-clamped at the stop).
    lam0=0.35,
    lam_growth=1.00707,   # doubles ~ep 99; hits the 1.0 cap at ~ep 150
    # Prioritized experience replay (Schaul et al. 2016), the one Rainbow component that helped:
    # on this lam shape, alpha 0.3 delivered 0.9396 over 266 episodes and 206 distinct orders --
    # statistically tied with the GA (paired per-scenario t = 0.11), where the paper-default
    # alpha 0.6 HURT (0.9480): the replay receives only ~10 fresh transitions per episode, and
    # heavy prioritization collapses sampling onto a few surprising ones. Milder priorities keep
    # the search alive longer instead (266 eps vs 159 unprioritized). Also screened on top of the
    # same shape and rejected: n-step 3 / 5 (0.9484 / 0.9434), dueling as-is (0.9447), dueling
    # with the Wang et al. recipe (0.9524). Full records: the campaign folder in the session
    # scratchpad, summarized in campaign_results.csv.
    prioritized=True,
    per_alpha=0.3,
    per_beta0=0.4,
    per_beta_eps=150,     # anneal beta to 1 over ~a run length
    per_eps=1e-3,
    # Error clipping, after Mnih et al. (Nature 2015): the TD error entering the update is bounded
    # to +/- huber_delta, which that paper notes is an absolute-value loss outside the interval.
    # The threshold is NOT the paper's 1.0 -- theirs was set for rewards clipped to [-1, 1], and
    # this problem's TD errors have median |q - y| 0.37 with a tail to ~1.3, so delta 1 clips only
    # the top few percent. Measured on 2026-08-11 (seed 42, M=50), delta 1.0 was the only
    # configuration in the whole campaign that was statistically WORSE than the baseline (0.9426,
    # paired t = 6.2 vs GA), while delta 0.5 -- scaled to bite about the top quartile -- delivered
    # 0.939563 in 207 episodes and 162 distinct orders, matching the GA to six decimals and getting
    # there in fewer episodes than the unclipped run (0.939597, 266 eps, 206 orders).
    #
    # Why it helps: the loss is (q - y)^2 + lam * hinge, and a squared term lets one large TD error
    # dominate the gradient and swamp the ranking term this solver exists for. Bounding it keeps
    # the two terms in proportion -- visible in the trace as fewer, better-aimed improvements
    # (9 to reach the final order, against 16 unclipped) rather than a wider search.
    # 2026-08-23: switched OFF (None) when the sandbox line's three changes were adopted --
    # every mp2-line result was measured without clipping, and re-enabling it would make the
    # adopted configuration one never actually run. The measured defence of 0.5 above is kept
    # for the day clipping is revisited.
    huber_delta=None,
    n_step=1,            # n-step return horizon for the TD target (1 = one-step)
    gamma=0.9995,        # discount on the bootstrap term ONLY; F itself is undiscounted
    double_dqn=True,     # decouple action selection from evaluation in the TD target
    target_sync=25,      # gradient steps between target-network copies
    # 2026-08-23: adopted from the sandbox line together with huber_delta=None after the 4-seed
    # comparison (mean F 0.9611 vs the retired 8-feature line's 0.9653, spread halved
    # 0.019 -> 0.0094). The third sandbox change -- progressive revelation OFF -- became
    # code-level removal of the whole mechanism the same day, so it no longer appears here.
    mp_rounds=2,         # two rounds of iterated neighbourhood aggregation appended to the state
                         # (8 -> 14 features per segment); 0 = the plain 8-feature state. See
                         # _nbr_cols for the exact columns and _current_flow for their UE cost.
)

# Search-coverage screen (run with `python main.py --solve tune-search`). The nominal variant
# plateaus having evaluated ~113 DISTINCT orders where the GA evaluates 423, and its shortfall
# against the GA on the scenario mean equals its shortfall on the NOMINAL objective, four
# measurements running. So the gap is search coverage, not training budget and not transfer. Each
# config below moves ONE knob off the tuned configuration, so any change in coverage is
# attributable to that knob.
#
# THE EXPLORATION SCHEDULE IS ALREADY RULED OUT (2026-08-10, seed 42, M=50, thresholds 75/75):
#   base 0.9484 / 126 eps / 113 orders     floor eps_min=0.25 -> 0.9484    floor 0.35 -> 0.9484
#   eps_decay 0.998 -> 0.9479 / 238 eps / 224 orders, ~20x the wall-clock for 0.0005 of F.
# Raising the floor changes nothing because eps(ep) = max(eps_min, eps0*eps_decay^(ep-1)) is still
# ~0.24 at episode 126, so the tuned floor of 0.114 never binds inside a run. Slowing the decay does
# buy continued search -- base freezes its best order at ep 50 while slow-decay keeps improving to
# ep 200 -- but converges to the same place. The reason is structural and no eps setting can fix it:
# one episode is ONE trajectory, hence at most one new order, so coverage is bounded by the episode
# count. eps changes where the single trajectory wanders, never how many are drawn.
#
# UPDATE COUNT IS ALSO RULED OUT, and it is the more informative negative (same conditions):
#   updates_per_ep 30 -> 10 gave 0.9520 over 189 episodes and 160 distinct orders.
# That is MORE coverage than base (160 vs 113 orders) and a WORSE delivered F. So distinct-order
# count is not the binding constraint after all: fitting less per episode does let the greedy order
# keep moving, but the extra orders it reaches are lower-quality, because the network no longer
# fits well enough to turn what it has seen into a better ranking. Coverage and ranking quality
# trade off against each other here, which is exactly what the GA does not have to trade -- each
# of its 423 evaluations is an independently proposed candidate, not a by-product of a policy that
# must also stay fitted. updates_per_ep=5 was not run: 10 already moved F the wrong way.
#
# What is left are the knobs that govern how hard each episode is pulled toward the best order so
# far, plus the state representation that decides which orders are reachable at all.
SEARCH_SWEEP = {
    "base":              {},               # control: the tuned configuration, expected 0.9484
    # Weight of the large-margin hinge, i.e. how strongly the exemplar trajectory is imitated. This
    # is the most direct suppressor of diversity in the whole loss: the hinge actively pushes every
    # rival action below the exemplar's, so a smaller lam leaves more room for the greedy order to
    # differ from the best order found so far. lam=0 is deliberately NOT tried -- it removes the
    # ranking term altogether and reverts to the value regression this module exists to replace.
    "lam-1.0":           dict(lam=1.0),
    "lam-0.5":           dict(lam=0.5),
    # Gradient steps between target-network copies. A staler target slows the propagation of the
    # best-so-far order's values through Q and so slows convergence of the induced order.
    "sync-100":          dict(target_sync=100),
    "sync-10":           dict(target_sync=10),
    # State representation: n_lag appends the features of the last n repaired segments. Both
    # searches selected n_lag=1, which is evidence the state as featurized is not fully observed.
    # More memory can separate states the policy currently aliases (reaching orders it cannot now
    # express), less memory makes the policy coarser and its rollouts more varied.
    "n_lag-0":           dict(n_lag=0),
    "n_lag-2":           dict(n_lag=2),
}

RANK_PARAMS_SAA = dict(
    # ADOPTED WHOLESALE FROM THE NOMINAL VARIANT (2026-08-12). Every value below that the two
    # variants share is the one the nominal variant's tuning campaign selected -- the loss shape,
    # the network, the exploration schedule, the imitation ramp, prioritized replay and the error
    # clipping. Those mechanisms were screened one at a time against a measured baseline, and there
    # is no reason a change of training world should overturn what they established; keeping two
    # independently drifted parameter sets was making the two variants incomparable for reasons
    # that had nothing to do with the thing they actually differ in.
    #
    # KEPT VARIANT-SPECIFIC, because they only exist for stochastic training:
    #   sil_buffer  the top-K good-trajectory buffer the ranking loss imitates from
    #   saa_n       the sample-average size (see below); the nominal variant rejects saa_n > 1
    # Same architecture as the nominal variant (see RANK_PARAMS_NOMINAL for the evidence): a
    # change of training world is no reason for a different function approximator, and letting the
    # two drift apart would make them incomparable for a reason unrelated to what they differ in.
    arch="dqn",
    dueling=True,
    trunk_rescale=True,
    loss="margin",
    hidden=(32, 32),
    n_lag=1,
    prior_scale=0.5639,
    lr=9.979e-3,
    eps0=0.4980,
    eps_decay=0.9928,
    eps_min=0.1141,
    updates_per_ep=30,
    batch=16,
    margin=0.0689,
    lam=1.0,
    lam0=0.35,
    lam_growth=1.00707,
    prioritized=True,
    per_alpha=0.3,
    per_beta0=0.4,
    per_beta_eps=150,
    per_eps=1e-3,
    huber_delta=0.5,
    n_step=1,
    gamma=0.9995,
    double_dqn=True,
    target_sync=25,
    sil_buffer=8,        # top-K good-trajectory buffer the ranking loss imitates from: too small
                         # tracks one lucky world, too large imitates mediocrity
    # SAMPLE AVERAGE APPROXIMATION: worlds drawn per episode whose F is averaged into the reward.
    # 1 reproduces the historical single-draw behaviour. Above 1 the learner is scored on a sample
    # average of its value rather than on one draw, at a proportional evaluation cost -- see the
    # episode loop in train() for why the single draw is so noisy here.
    saa_n=10,
    # How the fixed SAA sample is drawn: "lhs" = probability-stratified Latin Hypercube over the
    # per-level marginals, which covers the uncertainty space evenly and estimates E[F] with far
    # lower variance than the same number of i.i.d. draws (util.scenarios.saa_lhs_sample). Measured
    # on n10 seed 42 (at saa_n=10): LHS lifted the delivered mean F from 0.9562 (i.i.d.) to 0.9502,
    # level with GA; the delivered numbers at this saa_n come from the re-run. Drawn from
    # seed*1000+21, a separate stream from the eval ruler, so the two do not systematically coincide
    # (measured overlap <=3/50 at saa_n=20, the same order as i.i.d.).
    saa_sampling="lhs",
    # DISTRIBUTIONAL RL, quantile-regression head (QR-DQN, Dabney et al., AAAI 2018), the SAA
    # variant's headline addition: each action's head emits n_quantiles VALUES at fixed quantile
    # fractions tau_i=(i+0.5)/n instead of a single expected value, and the value regression becomes
    # the quantile-Huber (pinball) loss toward the Bellman-image quantiles. n_quantiles=1 would be the
    # ordinary scalar head. The rationale for putting it HERE and not on the nominal variant: the SAA
    # reward is a sample average over drawn worlds, so the return a state can realize is genuinely a
    # DISTRIBUTION, which a quantile head represents rather than collapses. Unlike a fixed-support
    # categorical head it needs no return range to bound or tune, and it proved both better and far
    # more stable on n10 (seeds 42/1 delivered 0.9515/0.9517 against a categorical head's 0.9581/
    # 0.9573; spread 0.0002 against the categorical head's 0.9492-0.9623). The flow prior enters by
    # ADDING prior_scale*phi to every quantile in Qf (see train()), so episode 0 still plays the flow
    # order.
    n_quantiles=51,
)

# Mechanism variants reuse the tuned bases unchanged and add ONE mechanism each, so a difference
# in the delivered F is attributable to that mechanism rather than to a different starting point.
# Light per-variant retuning belongs in hp overrides, not in edits to the bases.
#
# Rainbow ladder: PRIORITIZED EXPERIENCE REPLAY on the tuned base configs (Schaul et
# al. 2016, proportional variant). Sampling probability p_i ∝ (|td_i| + per_eps)^per_alpha;
# importance weights (N p_i)^(-beta) normalized by their max, with beta annealed linearly from
# per_beta0 to 1 over per_beta_eps episodes; a new transition enters at the current maximum
# priority so it is seen at least once. The buffer is small (4000), so plain proportional
# sampling replaces the usual sum-tree with no measurable cost.
RANK_PARAMS_SAA_PER = dict(RANK_PARAMS_SAA, prioritized=True, per_alpha=0.6,
                             per_beta0=0.4, per_beta_eps=300, per_eps=1e-3)

# Plateau stopping. patience_P and stable_K are counted in PROBES, not episodes: the nominal
# variant probes every episode (its best-so-far order is already computed, so a probe is free)
# while the
# stochastic variant probes every probe_every episodes (each probe costs n_val greedy rollouts and
# their evaluations). ep_cap is the outer guard, not the intended terminator.
STOP_PARAMS = dict(
    ep_min=60,           # floor: no run stops before this many episodes, however flat it looks
    patience_P=10,       # probes without an improvement beating `tol` -> plateau
    stable_K=25,         # probes with an unchanged greedy order -> the policy has settled
    tol=1e-3,            # relative improvement below this does not count as an improvement
    probe_every=25,      # (stochastic) episodes between validation probes
    n_val=11,            # (stochastic) validation worlds, drawn once from a dedicated stream.
                         # 11 rather than a handful because the stopping rule reads their MEAN:
                         # with 3 worlds that mean is noisy enough that a real improvement can be
                         # missed and a chance dip can pass for one, which is exactly the signal
                         # the patience counter is built on
)

# Per-variant overrides. patience_P is counted in PROBES, and the two variants probe on different
# cadences -- the nominal variant every episode (its best-so-far order is free to read), the
# stochastic one
# every `probe_every` episodes -- so one number means very different budgets. Splitting them here
# keeps a change to one variant from silently rescaling the other: patience_P=45 is 45 EPISODES for
# rl_nominal, but would be 45*25 = 1125 episodes for rl_saa.
#
# rl_nominal's 45 (raised from the shared 10 on 2026-08-10) is a search-budget decision, not a tuning
# one: at patience 10 the run stopped at episode 61 having scored only ~60 distinct orders, while
# the GA it is compared against scores 423. The nominal objective it reaches was worse than the
# GA's by exactly the amount it trails on the scenario mean, so the gap was search breadth, and
# breadth is what this buys.
# rl_nominal raised again on 2026-08-10: at patience_P=45 the run still stopped at episode 76, because
# stable_K=25 fired first -- the two are near-redundant for this variant (the margin loss imitates
# the best-so-far order, so a frozen best order freezes the greedy order a few episodes later), and
# the
# smaller threshold always wins. Raising BOTH to 50 keeps either from cutting the search off while
# the best order is still moving. It had improved as recently as episode 50 when the run
# was stopped.
STOP_OVERRIDES = {
    "rl_nominal": dict(patience_P=75, stable_K=75),
}
EP_CAP = 1000            # outer guard only; the plateau rule is what should end a run. Deliberately
                         # far above where the plateau fires, so a run that hits this cap is a
                         # signal that the plateau parameters are wrong, not a normal ending.

# Episodes between redraws of this variant's figures in its n{N} folder, so a long run is readable
# as figures while it is still going instead of only after it delivers. _Recorder.flush does the
# redraw off the records it has just written, which is what keeps a mid-run figure from disagreeing
# with the trace beside it. Nothing else is written mid-run: the deliverable is written once, at the
# end, so results/ never holds a half-finished run.
FIGURE_REFRESH_EVERY = 20


def _merge_stop(variant, stop_params):
    """The stopping rule that ACTUALLY governs a run: shared defaults, then the variant's override,
    then the caller's. Built by sequential update rather than dict(base, **a, **b) -- the latter
    raises "multiple values for keyword" whenever the override and the caller share a key (both
    carry patience_P/stable_K), which is exactly what a stop_params sweep does."""
    sp = dict(STOP_PARAMS)
    sp.update(STOP_OVERRIDES.get(variant, {}))
    sp.update(stop_params or {})
    return sp


class RankPlateauStop:
    """End a run once it has visibly levelled off, rather than at a preset episode count.

    Two independent plateau signals, either of which ends the run after an `ep_min` floor:
    (a) the watched score has not improved by more than a relative `tol` for `patience_P` probes,
    and (b) the greedy order has been unchanged for `stable_K` probes. (a) is the objective test
    and is what normally fires; (b) catches a policy that has settled onto one order while its
    score wobbles within noise. The floor exists because early training is non-monotone -- the
    exemplar is still poor, so a few flat probes mean nothing.

    `update` is called once per PROBE with the score being watched (lower is better) and the
    greedy order that produced it. The caller decides how often to probe, and therefore what the
    patience counters mean in episodes.
    """

    def __init__(self, ep_min=STOP_PARAMS["ep_min"], patience_P=STOP_PARAMS["patience_P"],
                 stable_K=STOP_PARAMS["stable_K"], tol=STOP_PARAMS["tol"]):
        self.ep_min, self.patience_P, self.stable_K, self.tol = ep_min, patience_P, stable_K, tol
        self.best = float("inf")
        self.since_improve = 0
        self.stable = 0
        self.last_order = None
        self.episode = 0
        self.reason = None

    def update(self, episode, score, order):
        self.episode = episode
        if score is not None:
            # The `best is inf` guard is load-bearing, not defensive: without it the first probe
            # compares against `inf - tol*inf`, which is NaN, every comparison with NaN is False,
            # so no probe ever counts as an improvement and the rule silently degenerates into a
            # fixed cap of ep_min + patience_P probes. util.metaheuristic.PlateauStop carries the
            # same guard for the same reason.
            if self.best == float("inf") or score < self.best - self.tol * abs(self.best):
                self.best, self.since_improve = score, 0
            else:
                self.since_improve += 1
        order = None if order is None else tuple(order)
        self.stable = self.stable + 1 if (order is not None and order == self.last_order) else 0
        self.last_order = order

    def done(self):
        if self.episode < self.ep_min:
            return False
        if self.since_improve >= self.patience_P:
            self.reason = "plateau_no_improvement"
            return True
        if self.stable >= self.stable_K:
            self.reason = "plateau_stable_order"
            return True
        return False


# --------------------------------------------------------------------------- #
# Environment, network, rollout
# --------------------------------------------------------------------------- #
def build_env(toy_dir=TOY, N=None, M=P.M_SCENARIOS, ue_cores=1):
    """Everything a run needs about the instance: context, the nominal world the deterministic
    variant optimizes against, the frozen evaluation scenarios, both horizons and the per-scenario
    statics. `T` scores every delivery, so it matches what the other solvers report; `T_train` is
    sized against the worst realization the duration model can produce, so no stochastic training
    trajectory is ever truncated."""
    N = P.N_DISRUPTED_ORACLE if N is None else N
    dis = select_oracle_instance(toy_dir, N)
    segs = sorted(int(e) for e in dis["edge_id"])
    ctx = build_context(toy_dir, dis, ue_cores=ue_cores)
    flow = _baseline_twoway_flow(toy_dir, cores=1)     # cores=1: the prior must be bit-stable
    phi = {int(e): flow.get((min(u, v), max(u, v)), 0.0) for (e, u, v, s) in ctx["disrupted"]}
    nominal = nominal_durations(dis, segs)
    # The evaluation sample is FROZEN at the project seed, never the run's training seed: every
    # method is scored on the same M scenarios and horizon or the comparison means nothing. A
    # training seed reaches only the training/validation RNG streams inside train(). (A run that
    # threaded its own seed in here was caught scoring itself on its own private scenario set --
    # the exact failure this line is worded against.)
    scen = sample_scenarios(dis, M, P.SEED)
    # Statics for the mp_rounds message-passing columns (built unconditionally -- they are cheap
    # and hp-independent): undirected adjacency by shared endpoint over ALL edges, each edge's
    # endpoints for folding directed UE volumes, the instance severities for rebuilding damaged
    # networks, and a per-env cache of current-flow solves keyed by the exact damaged state.
    end_of = {int(r.edge_id): (int(r.u), int(r.v)) for r in ctx["edges"].itertuples(index=False)}
    adj = {a: [b for b in end_of if b != a and set(end_of[a]) & set(end_of[b])] for a in end_of}
    mp = dict(end_of=end_of, adj=adj, phi_max=max(phi.values()),
              sev_of={int(eid): float(sv) for (eid, _, _, sv) in ctx["disrupted"]},
              flow_cache={})
    T = compute_horizon(segs, scen)
    T_train = max(T, compute_horizon(segs, [worst_case_durations(dis)]))
    st = _scenario_statics(ctx, segs, nominal, phi)
    return dict(dis=dis, segs=segs, ctx=ctx, phi=phi, nominal=nominal, scen=scen, T=T,
                T_train=T_train, st=st, mp=mp,
                seg_idx={int(eid): j for j, (eid, _, _, _) in enumerate(ctx["disrupted"])})


def _build_net(hidden, n_lag, torch, nn, dueling=False):
    """MLP h with a ZERO-initialized output layer, so Q starts equal to the flow prior and episode
    0 plays the flow order. The search starts at the strongest static baseline instead of at noise,
    which is also what makes the delivery no worse than flow in the nominal world.

    With dueling=True (Wang et al., ICML 2016; an EXPERIMENTAL flag, off in every tuned config)
    the same trunk splits into a value stream and an advantage stream,
        Q(s, e) = prior + V(s) + A(s, e) - mean_e' A(s, e'),
    with V(s) the candidate-mean of the value head (the trunk scores one candidate per row, so the
    state's value is read as the mean over its candidate rows) and the mean-subtraction the paper's
    identifiability aggregator. Both heads are zero-initialized, so episode 0 still plays flow."""
    dims = [_N_FEAT * (1 + n_lag)] + list(hidden)
    layers = []
    for a, b in zip(dims[:-1], dims[1:]):
        layers += [nn.Linear(a, b), nn.Tanh()]
    trunk = nn.Sequential(*layers)

    if not dueling:
        head = nn.Linear(dims[-1], 1)
        nn.init.zeros_(head.weight)
        nn.init.zeros_(head.bias)
        net = nn.Sequential(*list(trunk), head)
        return net

    class _Dueling(nn.Module):
        def __init__(self):
            super().__init__()
            self.trunk = trunk
            self.val = nn.Linear(dims[-1], 1)
            self.adv = nn.Linear(dims[-1], 1)
            for h in (self.val, self.adv):
                nn.init.zeros_(h.weight)
                nn.init.zeros_(h.bias)

        def forward(self, X):                      # X: one state's candidate matrix
            z = self.trunk(X)
            v = self.val(z).squeeze(-1)
            a = self.adv(z).squeeze(-1)
            return v.mean() + a - a.mean()
    return _Dueling()


def _noisy_linear(nn, torch):
    """Factorised-Gaussian NoisyLinear (Fortunato et al., ICLR 2018), built lazily so torch is only
    touched inside train().

    Each weight is mu + sigma * eps with mu and sigma BOTH learned, so the network decides how much
    exploration noise it wants and can decide it PER STATE -- which is the whole point, and exactly
    what an eps-greedy schedule cannot do: eps is a single number applied identically everywhere,
    while a state the network is confident about can have its noise annealed away while an
    ambiguous one keeps it.

    Factorised noise (the paper's cheaper variant) draws p + q unit samples instead of p*q and
    forms the weight noise as an outer product of f(eps) = sign(eps)*sqrt(|eps|). `reset_noise()`
    redraws; `set_noise(False)` evaluates at mu alone, which is what delivery and the plateau
    probes use so that the reported policy is the learned one rather than one noisy sample of it.
    """
    class NoisyLinear(nn.Module):
        def __init__(self, p_in, q_out, sigma0=0.5):
            super().__init__()
            self.p_in, self.q_out = p_in, q_out
            self.w_mu = nn.Parameter(torch.empty(q_out, p_in))
            self.w_sigma = nn.Parameter(torch.empty(q_out, p_in))
            self.b_mu = nn.Parameter(torch.empty(q_out))
            self.b_sigma = nn.Parameter(torch.empty(q_out))
            self.register_buffer("w_eps", torch.zeros(q_out, p_in))
            self.register_buffer("b_eps", torch.zeros(q_out))
            bound = p_in ** -0.5
            nn.init.uniform_(self.w_mu, -bound, bound)
            nn.init.uniform_(self.b_mu, -bound, bound)
            nn.init.constant_(self.w_sigma, sigma0 * bound)
            nn.init.constant_(self.b_sigma, sigma0 * bound)
            self.noisy = True
            self.reset_noise()

        @staticmethod
        def _f(x):
            return x.sign() * x.abs().sqrt()

        def reset_noise(self):
            ei = self._f(torch.randn(self.p_in))
            eo = self._f(torch.randn(self.q_out))
            self.w_eps.copy_(eo.outer(ei))
            self.b_eps.copy_(eo)

        def forward(self, x):
            if self.noisy and self.training:
                w = self.w_mu + self.w_sigma * self.w_eps
                b = self.b_mu + self.b_sigma * self.b_eps
            else:
                w, b = self.w_mu, self.b_mu
            return torch.nn.functional.linear(x, w, b)
    return NoisyLinear


def _reset_noise(model):
    """Redraw every NoisyLinear's noise. No-op on a network without any."""
    for m in model.modules():
        if hasattr(m, "reset_noise"):
            m.reset_noise()


def _build_dqn_net(hidden, n_seg, n_lag, torch, nn, dueling=False, noisy=False, n_out=1):
    """State-to-vector Q network for the classic-DQN variant: input is the fixed-order state
    vector (n_seg per-segment blocks + the lag block), output is one Q residual per segment.
    The head is zero-initialized for the same reason as the row architecture's: Q then starts
    at the flow prior exactly, and episode 0 plays the flow order.

    With dueling=True (Wang et al., ICML 2016) the trunk splits into a scalar state-value stream
    V(s) and a per-segment advantage stream A(s, e). This is the architecture the paper actually
    describes, and it is only expressible HERE: the row architecture had no state-level input, so
    its V had to be faked as the mean over candidate rows -- which is a plausible reason it lost
    there (0.9447 / 0.9524 against 0.9396).

    Returns the two streams SEPARATELY rather than a combined Q, because the paper's aggregator
    subtracts the mean advantage over the LEGAL action set and this module cannot see which
    actions are legal -- the legality mask lives in the caller's gather. Combining here would
    average over repaired segments too and shift every Q by a garbage constant. The caller
    aggregates after gathering; see Qf in train()."""
    dims = [_N_FEAT * (n_seg + n_lag)] + list(hidden)
    # Noise goes on the HIDDEN layers only; the output head stays a deterministic zero-initialized
    # Linear. That is a deliberate restriction: the head is what makes Q start exactly equal to the
    # flow prior so episode 0 plays the flow order, and a noisy head would emit sigma*eps*z at
    # init and destroy that invariant -- the property every variant in this project is anchored on.
    # Noise in the trunk still gives state-dependent exploration once the head has learned anything.
    Lin = _noisy_linear(nn, torch) if noisy else nn.Linear
    layers = []
    for a, b in zip(dims[:-1], dims[1:]):
        layers += [Lin(a, b), nn.Tanh()]
    trunk = nn.Sequential(*layers)
    # n_out>1 is the DISTRIBUTIONAL (quantile) head: every segment gets n_out values instead of one.
    # The head is always zero-initialized (zero weight and bias): the flow prior is added by Qf,
    # so Q starts at the prior for both the scalar and the quantile head and episode 0 plays the flow
    # order.
    if not dueling:
        head = nn.Linear(dims[-1], n_seg * n_out)
        nn.init.zeros_(head.weight)
        nn.init.zeros_(head.bias)
        return nn.Sequential(*list(trunk), head)

    class _DuelDQN(nn.Module):
        def __init__(self):
            super().__init__()
            self.trunk = trunk
            self.n_out = n_out
            self.val = nn.Linear(dims[-1], n_out)            # V(s)     [n_out]
            self.adv = nn.Linear(dims[-1], n_seg * n_out)    # A(s, e)  [n_seg, n_out]
            for h in (self.val, self.adv):
                nn.init.zeros_(h.weight)
                nn.init.zeros_(h.bias)

        def forward(self, sv):                     # -> (V, A) shaped for the caller to aggregate
            z = self.trunk(sv)
            v = self.val(z)
            a = self.adv(z)
            if self.n_out > 1:
                return v, a.reshape(-1, self.n_out)
            return v.squeeze(-1), a
    return _DuelDQN()


def _augment(base_X, chosen_base, n_lag):
    """Append the base features of the last n_lag repaired segments to every candidate row, zero
    padded at the start of an episode. Column 0 stays the flow prior, which the residual Q reads."""
    if n_lag <= 0:
        return base_X
    hist = np.zeros((n_lag, _N_FEAT))
    recent = chosen_base[-n_lag:]
    if recent:
        hist[n_lag - len(recent):] = np.asarray(recent)
    return np.hstack([base_X, np.tile(hist.reshape(-1), (base_X.shape[0], 1))])


def _current_flow(env, damaged_now, D_now):
    """Two-way UE flow per edge for the CURRENT traffic state: the network with `damaged_now`
    still broken and the demand surviving the shortfall D_now. One UE solve per distinct state,
    memoized on the exact (damaged set, shortfall vector) key, so repeated orders are free."""
    mp, ctx = env["mp"], env["ctx"]
    key = (tuple(sorted(damaged_now)), D_now.round(9).tobytes())
    hit = mp["flow_cache"].get(key)
    if hit is not None:
        return hit
    H = np.clip(ctx["H0"] - D_now, 0.0, None)
    links, _ = solve_ue(build_damaged_edges(ctx, {e: mp["sev_of"][e] for e in damaged_now}),
                        _matrix_from_H(H, ctx), ctx["zone_ids"], rgap=P.UE_RGAP,
                        max_iter=P.UE_MAX_ITER, quiet=True, cores=1)
    f = {}
    for a, b, v in zip(links["from"].to_numpy(), links["to"].to_numpy(),
                       links["volume"].to_numpy()):
        k2 = (min(int(a), int(b)), max(int(a), int(b)))
        f[k2] = f.get(k2, 0.0) + float(v)
    flow = {eid: f.get(tuple(sorted(pr)), 0.0) for eid, pr in mp["end_of"].items()}
    mp["flow_cache"][key] = flow
    return flow


def _nbr_cols(env, remaining, damaged_now, flow):
    """The six message-passing columns for each remaining candidate: two ROUNDS of iterated
    neighbourhood aggregation, not a fixed 2-hop window. Round 1 gives every segment the mean of
    its damaged neighbours' phi_hat and dem_hat and (over ALL neighbours) the mean current UE
    flow; round 2 aggregates those round-1 values over the neighbourhood again, so information
    reaches a candidate along every 2-edge path and can flow back through itself. phi/demand
    propagate on the still-damaged subgraph (an intact neighbour carries no damage signal), the
    current flow on the full network (intact edges carry traffic). Flow columns are normalized by
    the instance's largest pre-disaster segment flow."""
    st, mp = env["st"], env["mp"]
    adj, pm = mp["adj"], mp["phi_max"]
    ext = np.zeros((len(remaining), 6))
    m1p, m1d = {}, {}
    for e in env["segs"]:
        dn = [j for j in adj[e] if j in damaged_now]
        m1p[e] = float(np.mean([st["phi_hat"][j] for j in dn])) if dn else 0.0
        m1d[e] = float(np.mean([st["dem_hat"][j] for j in dn])) if dn else 0.0
    m1f = {a: (float(np.mean([flow[b] for b in adj[a]])) if adj[a] else 0.0) for a in adj}
    for i, e in enumerate(remaining):
        dn = [j for j in adj[e] if j in damaged_now]
        ext[i, 0] = m1p[e]
        ext[i, 1] = m1d[e]
        ext[i, 2] = m1f[e] / pm
        ext[i, 3] = float(np.mean([m1p[j] for j in dn])) if dn else 0.0
        ext[i, 4] = float(np.mean([m1d[j] for j in dn])) if dn else 0.0
        ext[i, 5] = (float(np.mean([m1f[b] for b in adj[e]])) / pm) if adj[e] else 0.0
    return ext


def _rollout(env, pick, n_lag, durations=None):
    """One episode: repeatedly hand the caller's `pick` the candidate feature matrix at the current
    decision point and schedule whichever candidate it returns, work-conserving over C_MAX crews.

    `durations` is the world the rollout EXECUTES in (the nominal world by default). Duration
    beliefs stay at the planning expectation throughout: realized durations shape the schedule as
    crews free up, but are never read into the state."""
    ctx, st, T = env["ctx"], env["st"], env["T"]
    durations = env["nominal"] if durations is None else durations
    dis_l, sev, B, H0 = ctx["disrupted"], ctx["severity_vec"], ctx["B"], ctx["H0"]
    seg_idx = env["seg_idx"]
    remaining = list(env["segs"])
    crew = [1] * P.C_MAX
    start, perm, feats, picks, rems, chosen_base = {}, [], [], [], [], []
    D = np.zeros(len(H0))
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
        # Duration beliefs are ALWAYS the planning expectation. The progressive-revelation
        # mechanism (a completion exposing its whole level's realized duration) was removed
        # outright on 2026-08-23: under per-segment-independent durations a completion reveals
        # nothing about other segments, so the level broadcast was pure leakage.
        bel = {e: st["dur_raw"][e] for e in remaining}
        base_X = _decision_features(remaining, t, crew, D, st, T, bel)
        if int(st.get("mp_rounds", 0)):
            dmg = set(remaining) | {e for e, c in comp.items() if c > t}
            base_X = np.hstack([base_X,
                                _nbr_cols(env, remaining, dmg, _current_flow(env, dmg, D))])
        X = _augment(base_X, chosen_base, n_lag)
        rems.append(list(remaining))
        i = pick(remaining, X)
        feats.append(X)
        picks.append(i)
        chosen_base.append(base_X[i])
        e = remaining.pop(i)
        c = int(np.argmin(crew))
        start[e] = crew[c]
        crew[c] = start[e] + int(durations[e])
        comp[e] = crew[c]
        perm.append(e)
    return tuple(perm), start, feats, picks, rems


# --------------------------------------------------------------------------- #
# Diagnostics recorder
# --------------------------------------------------------------------------- #
def repr_q(net, Xnp, prior_scale, torch, groups=None):
    """Last-hidden-layer activations and the residual Q = prior*phi_hat + head(activations), for a
    batch of candidate rows. Splitting the network here is what makes the representation itself
    recordable, not just the scalar it collapses to.

    For a dueling net the representation is the shared trunk's output, and Q needs the state's
    candidate set to aggregate -- `groups` labels each row's state for batches that mix states
    (the fixed probe spans every step of the flow order); rows sharing a label aggregate together.
    A single-state batch (every other call site) needs no labels."""
    Xt = torch.tensor(np.asarray(Xnp), dtype=torch.float32)
    if Xt.dim() == 1:
        Xt = Xt.unsqueeze(0)
    with torch.no_grad():
        if hasattr(net, "trunk"):                      # dueling
            rep = net.trunk(Xt)
            v = net.val(rep).squeeze(-1)
            a = net.adv(rep).squeeze(-1)
            if groups is None:
                q_res = v.mean() + a - a.mean()
            else:
                q_res = torch.empty_like(a)
                for g in np.unique(np.asarray(groups)):
                    m = torch.tensor(np.asarray(groups) == g)
                    q_res[m] = v[m].mean() + a[m] - a[m].mean()
            q = prior_scale * Xt[:, 0] + q_res
        else:
            rep = net[:-1](Xt)
            q = prior_scale * Xt[:, 0] + net[-1](rep).squeeze(-1)
    return rep.numpy(), q.numpy()


class _Recorder:
    """Persist what the later analyses need, so no figure requires retraining.

    Three records. `{v}_trace.csv` is per-episode scalars, including the batch means of the
    predicted Q and of its TD target: those two nearly coincide by construction (the regression
    term fits Q to that target), so it is their DIFFERENCE FROM THE PER-SAMPLE error that carries
    information -- a small gap alongside a large mean |q - y| says the network is unbiased overall
    while individual state-actions stay mis-calibrated, which is the signature of state aliasing.
    `{v}_probe_meta.csv` plus `{v}_qevo.npz` fix a set of decision states once and snapshot their Q
    values AND last-hidden-layer representations across training, which is what makes
    representation drift observable. Files are rewritten every `flush_every` episodes, so a killed
    run keeps everything up to the last flush."""

    def __init__(self, rec_dir, variant, probe_X, probe_meta, prior_scale, torch,
                 snap_every=20, flush_every=20):
        self.dir, self.v, self.PS = str(rec_dir), variant, prior_scale
        self.torch = torch
        self.snap_every, self.flush_every = max(1, snap_every), max(1, flush_every)
        # probe_X=None disables the representation/Q snapshots while keeping the trace. The probe
        # feeds a CANDIDATE MATRIX through the net and reads its last hidden layer, which only
        # means something for the row architecture -- the DQN architecture has one representation
        # per STATE, not one per candidate. The per-episode trace is architecture-agnostic, so it
        # is recorded either way and every figure drawn from it survives.
        self.probe_X = None if probe_X is None else np.asarray(probe_X, dtype=np.float32)
        self.probe_meta = probe_meta
        self.probe_groups = (None if probe_meta is None
                             else np.array([m["step"] for m in probe_meta]))
        self.trace, self.snaps, self.snap_eps = [], [], []
        os.makedirs(self.dir, exist_ok=True)
        if probe_meta is not None:
            pd.DataFrame(probe_meta).to_csv(os.path.join(self.dir, f"{variant}_probe_meta.csv"),
                                            index=False)

    def repr_q(self, net, Xnp):
        return repr_q(net, Xnp, self.PS, self.torch, groups=self.probe_groups)

    def episode(self, **row):
        self.trace.append(row)
        with open(os.path.join(self.dir, f"{self.v}_live_progress.txt"), "w", encoding="utf-8") as f:
            f.write(f"episode {row.get('episode', len(self.trace) - 1)}  " + "  ".join(
                f"{k}={v:.4f}" for k, v in row.items() if isinstance(v, float)) + "\n")
        if len(self.trace) % self.flush_every == 0:
            self.flush()

    def maybe_snapshot(self, ep, net):
        if self.probe_X is None:
            return
        if ep == 0 or ep % self.snap_every == 0:
            rep, q = self.repr_q(net, self.probe_X)
            self.snaps.append((rep, q))
            self.snap_eps.append(int(ep))

    def flush(self):
        pd.DataFrame(self.trace).to_csv(os.path.join(self.dir, f"{self.v}_trace.csv"), index=False)
        if self.snaps:
            np.savez_compressed(
                os.path.join(self.dir, f"{self.v}_qevo.npz"),
                episodes=np.array(self.snap_eps),
                q=np.stack([q for _, q in self.snaps]),
                repr=np.stack([r for r, _ in self.snaps]),
                seg=np.array([m["seg"] for m in self.probe_meta]),
                step=np.array([m["step"] for m in self.probe_meta]))
        # Redraw the figures from what was just written, so the run directory is readable at any
        # moment rather than only after delivery. Drawing here, off the same data the flush wrote,
        # is what keeps figures from silently drifting out of step with the records they describe.
        # A plotting failure must not kill a multi-hour training run, but it must not pass unseen
        # either -- hence caught and REPORTED rather than swallowed.
        try:
            from viz.rank_viz import make_rank_figures
            make_rank_figures(Path(self.dir).parent, self.v)
        except Exception as exc:
            print(f"  [{self.v}] figure refresh failed ({type(exc).__name__}: {exc})", flush=True)

    def finish(self, net):
        if self.probe_X is not None:
            rep, q = self.repr_q(net, self.probe_X)    # always snapshot the DELIVERED network
            self.snaps.append((rep, q))
            self.snap_eps.append(-1)                   # -1 marks the delivered net
        self.flush()


def _build_probe(env, n_lag):
    """A fixed set of decision states -- every candidate at every step of the flow order -- whose Q
    and representation are tracked across training. The flow order is used because it is the
    episode-0 policy, so the probe covers states the learner actually starts from."""
    flow_order = sorted(env["segs"], key=lambda e: -env["phi"][e])

    def fp(rem, X):
        for e in flow_order:
            if e in rem:
                return rem.index(e)
    _, _, feats, _, rems = _rollout(env, fp, n_lag)
    meta, X = [], []
    for step, (Xa, rem) in enumerate(zip(feats, rems)):
        for i, seg in enumerate(rem):
            base = np.asarray(Xa[i])
            row = dict(probe_id=len(meta), step=step, seg=int(seg))
            row.update({FEAT_COLS[c]: float(base[c]) for c in range(_N_FEAT)})
            meta.append(row)
            X.append(base)
    return np.array(X, dtype=np.float32), meta


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def _nstep_rows(feats, picks, rewards, n_step, gamma):
    """(X_i, a_i, R_i, bootstrap_state, bootstrap_discount) for every decision of one episode.

    The row carries the decision state's FULL candidate matrix X_i plus the chosen index a_i,
    not just the chosen candidate's feature row: a dueling head needs the whole candidate set to
    aggregate (its Q subtracts the state's mean advantage), and for the plain head indexing the
    matrix's Q vector at a_i is exactly the old single-row value."""
    n = len(picks)
    out = []
    for i in range(n):
        R, disc = 0.0, 1.0
        for kk in range(min(n_step, n - i)):
            R += disc * rewards[i + kk]
            disc *= gamma
        boot = feats[i + n_step] if i + n_step < n else None
        out.append((feats[i], picks[i], R, boot, gamma ** n_step if boot is not None else 0.0))
    return out


def _hinge(Qf, net, ex_feats, ex_picks, margin, torch):
    """Large-margin term over one exemplar trajectory: every rival action must fall at least
    `margin` below the exemplar's, and contributes nothing once it does."""
    ml = 0.0
    for X, a in zip(ex_feats, ex_picks):
        Q = Qf(net, X)
        viol = torch.relu(margin - (Q[a] - Q)).clone()
        viol[a] = 0.0                                  # the exemplar action never penalizes itself
        ml = ml + viol.sum()
    return ml / len(ex_feats)


def train(env, variant="rl_nominal", hp=None, seed=P.SEED, ep_cap=EP_CAP, rec_dir=None,
          stop_params=None, verbose=True):
    """Train one variant to a plateau and return the delivery plus its diagnostics.

    Returns dict(order, per_scenario, net, episodes, n_evals, outcome, best_score). For "rl_nominal",
    per_scenario is the one committed order list-scheduled into each evaluation scenario; for
    "rl_saa" it is the policy's own greedy rollout in each scenario, so each gets its own order.
    """
    import torch
    import torch.nn as nn
    torch.set_num_threads(1)

    saa = variant.startswith("rl_saa")
    base = {"rl_nominal": RANK_PARAMS_NOMINAL, "rl_saa": RANK_PARAMS_SAA,
            "rl_saa_per": RANK_PARAMS_SAA_PER}[variant]
    hp = dict(base, **(hp or {}))
    prioritized = bool(hp.get("prioritized", False))
    sp = _merge_stop(variant, stop_params)
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed * 7 + 1)
    # (the stream that used to draw a fresh training world each episode is gone: SAA fixes the
    #  sample once, at saa_rng below, so nothing consumes a per-episode world draw any more)
    val_rng = np.random.RandomState(seed * 1000 + 15)        # validation worlds, disjoint stream

    n_lag, PS = int(hp["n_lag"]), float(hp["prior_scale"])
    loss_mode, n_step, gamma = hp["loss"], int(hp["n_step"]), float(hp["gamma"])
    UPD, BATCH = int(hp["updates_per_ep"]), int(hp["batch"])
    MARGIN, LAM, ddqn = float(hp["margin"]), float(hp["lam"]), bool(hp["double_dqn"])
    lam0, lam_growth = hp.get("lam0"), hp.get("lam_growth")   # imitation ramp; None -> constant LAM
    sync = int(hp["target_sync"])
    if "reveal" in hp:
        raise ValueError("the progressive-revelation mechanism was removed from the code on "
                         "2026-08-23 -- drop the 'reveal' key (beliefs always stay at the "
                         "planning expectation now)")
    env["st"]["n_lag"] = n_lag
    # Feature width follows the hp: 8 base columns, plus 6 when the two message-passing rounds
    # are on. A module global because every consumer (net sizing, the state assembler, the lag
    # augmenter, the recorder) reads _N_FEAT; variants run sequentially, never concurrently.
    mp_rounds = int(hp.get("mp_rounds", 0))
    if mp_rounds not in (0, 2):
        raise ValueError(f"mp_rounds must be 0 or 2, got {mp_rounds} -- only the plain state and "
                         "the two-round aggregation exist (one round was measured and retired)")
    env["st"]["mp_rounds"] = mp_rounds
    global _N_FEAT
    _N_FEAT = 8 + 3 * mp_rounds
    Ttr = env["T_train"] if saa else env["T"]

    dueling = bool(hp.get("dueling", False))
    dqn_arch = str(hp.get("arch", "rows")) == "dqn"
    noisy = bool(hp.get("noisy", False))
    if noisy and not dqn_arch:
        raise ValueError("noisy nets are implemented for arch='dqn' only")
    # QR-DQN (Dabney et al., AAAI 2018): learn the return DISTRIBUTION as n_quantiles VALUES at fixed
    # quantile fractions tau_i=(i+0.5)/n, instead of the expectation. n_quantiles=1 is the ordinary
    # scalar head; >1 turns the head distributional (and switches the loss to quantile-Huber below).
    n_quantiles = int(hp.get("n_quantiles", 1) or 1)
    distributional = n_quantiles > 1
    if distributional and not dqn_arch:
        raise ValueError("the distributional (quantile) head is implemented for arch='dqn' only")
    if dqn_arch:
        n_seg_all = len(env["segs"])
        # THE FLOW PRIOR. The scalar head adds prior_scale*phi_hat to Q so episode 0 plays the flow
        # order -- the starting point every variant is anchored on. The quantile head carries the same
        # prior by ADDING prior_scale*phi to EVERY quantile in Qf (see below), so the initial per-
        # action mean is prior_scale*phi and the induced order is exactly the flow order. tau is the
        # vector of quantile fractions; None for the scalar head.
        tau = ((torch.arange(n_quantiles, dtype=torch.float32) + 0.5) / n_quantiles) \
            if distributional else None
        net = _build_dqn_net(hp["hidden"], n_seg_all, n_lag, torch, nn, dueling=dueling,
                             noisy=noisy, n_out=n_quantiles)
        tgt = _build_dqn_net(hp["hidden"], n_seg_all, n_lag, torch, nn, dueling=dueling,
                             noisy=noisy, n_out=n_quantiles)
    else:
        net = _build_net(hp["hidden"], n_lag, torch, nn, dueling=dueling)
        tgt = _build_net(hp["hidden"], n_lag, torch, nn, dueling=dueling)
    tgt.load_state_dict(net.state_dict())
    opt = torch.optim.Adam(net.parameters(), lr=float(hp["lr"]))
    # Wang et al. (2016) training recommendations, both off unless the hp asks: clip the global
    # gradient norm, and rescale the shared trunk's gradients by 1/sqrt(2) because both streams'
    # gradients sum there.
    grad_clip = hp.get("grad_clip")
    trunk_rescale = bool(hp.get("trunk_rescale", False)) and dueling
    # Mnih et al. (Nature 2015) error clipping: bound the TD error entering the update to
    # [-huber_delta, +huber_delta], which the paper notes is the same as an absolute-value loss
    # outside that interval. Written as 2*huber so the QUADRATIC region stays exactly the current
    # (q - y)^2 -- plain huber is 0.5*x^2 there, which would silently halve the TD term's weight
    # and so double lam's relative pull, confounding this with the lam experiments. None = off.
    huber_delta = hp.get("huber_delta")
    # Replay warm-up (Mnih et al. 2015 "replay start size"): hold off gradient updates until the
    # buffer holds this many transitions. Without it the first episode's 10 transitions are the
    # ENTIRE sample for that episode's updates_per_ep updates, so the network is fitted hard to one
    # trajectory before a second exists -- and the margin hinge, whose exemplar is that same
    # trajectory, pulls the same way. Collection continues during the warm-up (episode 0 is the
    # flow order and later ones are its eps-perturbations, so the buffer fills with a strong
    # baseline's neighbourhood rather than random play, unlike the paper's uniform-random warm-up),
    # and the best-so-far order still updates because it is found by EVALUATION, not by learning.
    # 0 = off.
    replay_start = int(hp.get("replay_start", 0) or 0)
    # SAA sample size: how many worlds are drawn ONCE and then fixed, their average F being the
    # reward signal every episode. 1 is a single fixed world (a degenerate sample average), not the
    # old per-episode redraw -- that redraw was resampling, not SAA, and has been removed.
    saa_n = int(hp.get("saa_n", 1) or 1)
    if saa_n > 1 and not saa:
        raise ValueError("saa_n > 1 is meaningless for the nominal variant: it trains in one fixed "
                         "world, so there is no world distribution to average over")

    if dqn_arch:
        # Everything downstream -- greedy and eps policies, the TD target, the hinge, the plateau
        # probes and the delivery rollouts -- consumes "the Q vector aligned with this state's
        # candidate rows" through this one function. The DQN architecture therefore plugs in HERE
        # and nowhere else: convert the candidate matrix to the fixed-order state vector, run the
        # single forward pass, and GATHER the head's outputs back into candidate-row order. The
        # legality mask falls out of the gathering -- a repaired segment's entry is simply never in
        # the returned vector, so no consumer can select it, bootstrap from it, or penalize it.
        #
        # Row identity comes from the phi column: column 0 of every candidate row is that segment's
        # normalized flow prior, a per-segment constant copied from one dict, so exact float
        # equality identifies the segment. Asserted unique at setup; an unknown value raises
        # KeyError rather than mis-assigning a row.
        segs_sorted = list(env["segs"])
        _phi_to_head = {float(env["st"]["phi_hat"][e]): j for j, e in enumerate(segs_sorted)}
        if len(_phi_to_head) != len(segs_sorted):
            raise ValueError("phi values are not distinct across segments; the DQN row-identity "
                             "lookup cannot work on this instance")
        _phi_vec = torch.tensor([float(env["st"]["phi_hat"][e]) for e in segs_sorted],
                                dtype=torch.float32)
        _sdim = _N_FEAT * (len(segs_sorted) + n_lag)

        def Qf(model, Xnp):
            """Q for ONE state's candidate matrix, DQN wiring: one forward on the state vector,
            outputs gathered into candidate-row order."""
            X = np.asarray(Xnp, dtype=float)
            if X.ndim == 1:
                X = X[None, :]
            sv = np.zeros(_sdim)
            heads = []
            for row in X:
                j = _phi_to_head[float(row[0])]        # KeyError = loud, never silent
                sv[j * _N_FEAT:(j + 1) * _N_FEAT] = row[:_N_FEAT]
                heads.append(j)
            if n_lag > 0:                              # lag block is state-wide: same on every row
                sv[len(segs_sorted) * _N_FEAT:] = X[0, _N_FEAT:]
            out = model(torch.tensor(sv, dtype=torch.float32))
            if distributional:
                # Per (legal action, quantile) aggregation, exactly as the scalar dueling case but per
                # quantile. The flow prior is added to EVERY quantile so the initial per-action mean is
                # prior_scale*phi (episode 0 = the flow order).
                if dueling:
                    v, a_full = out                       # v [n_quantiles], a_full [n_seg, n_quantiles]
                    a = a_full[heads]
                    agg = v.unsqueeze(0) + a - a.mean(dim=0, keepdim=True)
                else:
                    agg = out.reshape(-1, n_quantiles)[heads]
                return agg + PS * _phi_vec[heads].unsqueeze(1)       # [n_legal, n_quantiles]
            if dueling:
                # Wang et al. aggregation, applied to the LEGAL actions only: gather the advantage
                # stream first, then subtract ITS mean. Subtracting the mean over all n segments
                # would fold repaired segments' advantages into every remaining Q as a shared
                # offset that drifts as the episode proceeds.
                v, a_full = out
                a = a_full[heads]
                return PS * _phi_vec[heads] + v + a - a.mean()
            return (PS * _phi_vec + out)[heads]
    else:
        def Qf(model, Xnp):
            """Q for ONE state's candidate matrix. The dueling head aggregates across the matrix's
            rows, which is why every caller passes a full state, never rows from mixed states."""
            Xt = torch.tensor(np.asarray(Xnp), dtype=torch.float32)
            if Xt.dim() == 1:
                Xt = Xt.unsqueeze(0)
            return PS * Xt[:, 0] + model(Xt).squeeze(-1)

    if distributional:
        # Every existing consumer asks for Q. The head returns the QUANTILES, so Pf is the raw head and
        # Qf is their mean (equal-weighted by construction of tau) -- one rebind that keeps the policy,
        # the hinge, the probes and the delivery rollouts working untouched on the new representation.
        Pf = Qf

        def Qf(model, Xnp):                              # noqa: F811 - deliberate rebind
            return Pf(model, Xnp).mean(dim=1)

    def greedy(rem, X):
        """The DELIVERED policy: always evaluated at mu, never at a noise sample, so what gets
        delivered and probed is the learned policy rather than one draw from it."""
        with torch.no_grad():
            if noisy:
                net.eval()                       # NoisyLinear falls back to mu when not training
                q = Qf(net, X)
                net.train()
                return int(torch.argmax(q))
            return int(torch.argmax(Qf(net, X)))

    def states_for(order):
        def fp(rem, X):
            for e in order:
                if e in rem:
                    return rem.index(e)
        _, _, f, p, _ = _rollout(env, fp, n_lag)
        return f, p

    rec = None
    if rec_dir is not None:
        # The DQN architecture keeps the TRACE (and so every figure drawn from it) but drops the
        # probe snapshots, whose representation-per-candidate notion is row-architecture-specific.
        pX, pmeta = (None, None) if dqn_arch else _build_probe(env, n_lag)
        rec = _Recorder(rec_dir, variant, pX, pmeta, PS, torch,
                        snap_every=max(1, ep_cap // 50),
                        flush_every=FIGURE_REFRESH_EVERY)

    val_worlds = [draw_durations(env["dis"], val_rng) for _ in range(sp["n_val"])] if saa else []
    # SAMPLE AVERAGE APPROXIMATION: the sample is drawn ONCE and then held FIXED for the whole run.
    # That is what makes it SAA rather than plain resampling -- fixing the sample replaces the
    # stochastic objective E[F] with a DETERMINISTIC surrogate, the average over these particular
    # worlds, and the learner then optimizes that surrogate. Re-drawing every episode would move the
    # objective under the learner's feet: an order could score better than another purely because it
    # was measured on an easier draw, which is the noise this is meant to remove.
    #
    # Drawn from a dedicated stream so the sample does not depend on how many episodes run, and
    # never from the M frozen evaluation scenarios -- those stay the untouched ruler.
    #
    # The bias this buys is the standard SAA one and is real: the delivered policy is optimal for
    # THESE saa_n worlds, not for the distribution, and it is optimistic on them. Larger saa_n
    # shrinks that gap; the frozen 50 report what survives it.
    # `saa_resample=True` overrides the fix described above: it draws a FRESH sample of saa_n worlds
    # every episode instead of holding one fixed. That is NOT SAA -- it is the plain resampling SAA
    # was introduced to replace -- and is kept only as a measurable alternative: the objective then
    # moves under the learner, so an order can score better purely by drawing an easier world. Default
    # False = the fixed-sample SAA the delivery uses.
    saa_resample = bool(hp.get("saa_resample", False))
    # How the FIXED SAA sample is drawn: "iid" = saa_n independent draw_durations (plain Monte
    # Carlo, the historical default); "lhs" = probability-stratified Latin Hypercube over the
    # per-level marginals, which covers the uncertainty space far more evenly at n=10 and so
    # estimates E[F] with lower variance (see util.scenarios.saa_lhs_sample). Ignored when
    # saa_resample redraws every episode.
    saa_sampling = str(hp.get("saa_sampling", "iid"))
    saa_rng = np.random.RandomState(seed * 1000 + 21)
    if saa and not saa_resample:
        saa_worlds = (saa_lhs_sample(env["dis"], saa_n, saa_rng) if saa_sampling == "lhs"
                      else [draw_durations(env["dis"], saa_rng) for _ in range(saa_n)])
    else:
        saa_worlds = []
    stop = RankPlateauStop(sp["ep_min"], sp["patience_P"], sp["stable_K"], sp["tol"])
    memo, sc, replay, buf = {}, {}, [], []
    prio = []                                  # per-transition priorities (prioritized replay only)
    best_perm, best_F, gstep = None, float("inf"), 0
    t0 = time.perf_counter()
    ep = 0
    while ep < ep_cap and not stop.done():
        eps = 0.0 if ep == 0 else max(hp["eps_min"], hp["eps0"] * hp["eps_decay"] ** (ep - 1))
        # The imitation weight, scheduled as the MIRROR of eps. eps decays high->low (explore the
        # action space early, follow Q late); lam grows low->high (let the greedy order roam free
        # of the best-so-far order early, then clamp onto it late). The late clamp is not optional
        # tidiness: what is DELIVERED is the final greedy order, not the best order found, and the
        # hinge is the only force dragging the one onto the other. Geometric like eps, capped at
        # the constant LAM, which keeps its meaning as the terminal commitment weight.
        lam_ep = LAM if lam_growth is None else min(
            LAM, float(lam0) * float(lam_growth) ** max(0, ep - 1))

        def pk(rem, X):
            # BEHAVIOUR policy. With noisy nets the exploration comes from the weights: the noise
            # is redrawn at every decision, so the perturbation is state-dependent -- a state the
            # network has grown confident about (small learned sigma) is barely perturbed while an
            # ambiguous one still is. eps-greedy cannot do that: one number, applied identically
            # everywhere. Whether eps ALSO stays on is a hyperparameter; Fortunato et al. replace
            # it outright, which `eps0=0` expresses.
            if noisy:
                _reset_noise(net)
            if eps > 0 and rng.rand() < eps:
                return rng.randint(len(rem))
            with torch.no_grad():
                return int(torch.argmax(Qf(net, X)))

        if saa:
            # SAMPLE AVERAGE APPROXIMATION. The episode's reward is the AVERAGE F over saa_n worlds --
            # a far less noisy estimate of the policy's value E[F] than one draw, at saa_n times the
            # evaluation cost (on this problem the across-world spread is many times the gap between
            # the methods being compared, so a single draw wastes capacity fitting which world it hit).
            #
            # DEFAULT mode holds the sample FIXED (drawn once above): the surrogate is deterministic,
            # so a repeated (world, order) pair is memoized. `saa_resample` mode redraws the worlds
            # FRESH every episode (plain resampling, not SAA) and skips the memo, since no
            # (world, order) key recurs. Neither set is ever the M frozen evaluation scenarios.
            worlds_ep = ([draw_durations(env["dis"], saa_rng) for _ in range(saa_n)]
                         if saa_resample else saa_worlds)
            trajs = []
            for wi, dur in enumerate(worlds_ep):
                perm, start, feats, picks, _ = _rollout(env, pk, n_lag, dur)
                key = (wi, perm)
                if (not saa_resample) and key in memo:
                    res_i, rew_i = memo[key]
                else:
                    res_i = _evaluate_prefix_cached(start, dur, Ttr, env["ctx"], sc)
                    rew_i, _ = _completion_rewards(perm, start, dur, res_i["terms"], Ttr,
                                                   env["phi"])
                    if not saa_resample:
                        memo[key] = (res_i, rew_i)
                trajs.append(dict(perm=perm, feats=feats, picks=picks, res=res_i, rew=rew_i))
            F_ep = float(np.mean([t["res"]["F"] for t in trajs]))
            # Self-imitation exemplar: this episode's BEST trajectory, but filed under the SAA MEAN.
            # Ranking the buffer by the trajectory's own F would promote whichever snapshot drew the
            # easiest world; ranking by the sample average ranks the POLICY, which is what the
            # imitation term should be chasing.
            b = min(trajs, key=lambda t: t["res"]["F"])
            buf.append((b["feats"], b["picks"], F_ep))
            buf.sort(key=lambda z: z[2])
            del buf[int(hp["sil_buffer"]):]
            fresh = []
            for t in trajs:                            # every sampled world contributes experience
                fresh.extend(_nstep_rows(t["feats"], t["picks"], t["rew"], n_step, gamma))
            res = {"F": F_ep}                          # what the trace records for this episode
        else:
            # The nominal variant trains in ONE fixed world by definition, so there is nothing to
            # average and the memo makes a repeated order free.
            dur = env["nominal"]
            perm, start, feats, picks, _ = _rollout(env, pk, n_lag, dur)
            if perm in memo:
                res = memo[perm]
            else:
                res = _evaluate_prefix_cached(start, dur, Ttr, env["ctx"], sc)
                memo[perm] = res
                if res["F"] < best_F:
                    best_F, best_perm = res["F"], perm
            rew, _ = _completion_rewards(perm, start, dur, res["terms"], Ttr, env["phi"])
            fresh = _nstep_rows(feats, picks, rew, n_step, gamma)
        replay.extend(fresh)
        if prioritized:                       # a new transition enters at the current max priority
            prio.extend([max(prio, default=1.0)] * len(fresh))
        if len(replay) > 4000:
            replay = replay[-4000:]
            if prioritized:
                prio = prio[-4000:]

        ep_q, ep_y, ep_td, ep_loss = [], [], [], []
        n_upd = UPD if len(replay) >= replay_start else 0
        for _u in range(n_upd):
            if noisy:            # one fresh draw per gradient step, as in the paper
                _reset_noise(net)
                _reset_noise(tgt)
            opt.zero_grad()
            if saa:                                   # exemplar: SAMPLED from the good-trajectory buffer
                ex = buf[rng.randint(len(buf))][:2] if buf else None
            else:                                       # exemplar: the best-by-F order so far
                ex = states_for(best_perm) if best_perm is not None else None
            if loss_mode != "margin":
                raise ValueError(f"unknown loss {loss_mode!r} (only 'margin' exists; "
                                 f"sil_ce was removed 2026-08-10)")
            if prioritized:
                pr = np.asarray(prio, dtype=float) ** float(hp["per_alpha"])
                probs = pr / pr.sum()
                idx = rng.choice(len(replay), size=min(BATCH, len(replay)), replace=False,
                                 p=probs)
                beta = min(1.0, float(hp["per_beta0"])
                           + (1.0 - float(hp["per_beta0"])) * ep / float(hp["per_beta_eps"]))
                w = (len(replay) * probs[idx]) ** (-beta)
                w_t = torch.tensor(w / w.max(), dtype=torch.float32)
            else:
                idx = rng.choice(len(replay), size=min(BATCH, len(replay)), replace=False)
            # Per-transition forward on each state's full candidate matrix (a dueling head must
            # see all candidates of a state at once); indexing at the chosen action reproduces the
            # old stacked-single-rows computation exactly for the plain head.
            if distributional:
                # QR-DQN (Dabney et al., AAAI 2018): the value regression becomes the QUANTILE-HUBER loss
                # between the predicted quantiles theta and the Bellman-image quantiles R+gamma*theta'.
                # For predicted quantile tau_i the pinball asymmetry |tau_i - 1{u<0}| pulls it toward
                # the tau_i-quantile of the target. No fixed support and no error clipping here.
                kappa = 1.0
                q, td_list = [], []
                for i in idx:
                    _, a_i, R, bs, bd = replay[i]
                    theta = Pf(net, replay[i][0])[a_i]                 # [N] predicted quantiles
                    with torch.no_grad():
                        if bs is None:
                            Tq = torch.full((n_quantiles,), float(R))   # terminal: point mass at R
                        else:
                            a_on = int(torch.argmax(Qf(net, bs))) if ddqn \
                                else int(torch.argmax(Qf(tgt, bs)))
                            Tq = R + bd * Pf(tgt, bs)[a_on]            # [N] target quantiles
                    u = Tq.unsqueeze(0) - theta.unsqueeze(1)          # [N_pred, N_target]
                    hub = torch.where(u.abs() <= kappa, 0.5 * u ** 2, kappa * (u.abs() - 0.5 * kappa))
                    rho = (tau.unsqueeze(1) - (u.detach() < 0).float()).abs() * hub
                    td_list.append(rho.mean(dim=1).sum())             # mean over target, sum over pred
                    q.append(theta.detach().mean())
                td_elem = torch.stack(td_list)
                q = torch.stack(q)
                yt = q.detach()
            else:
                q = torch.stack([Qf(net, replay[i][0])[replay[i][1]] for i in idx])
                y = []
                for i in idx:
                    _, _, R, bs, bd = replay[i]
                    if bs is None:
                        y.append(R)
                    else:
                        with torch.no_grad():
                            a_on = int(torch.argmax(Qf(net, bs))) if ddqn else None
                            y.append(R + bd * float(Qf(tgt, bs)[a_on] if ddqn
                                                    else Qf(tgt, bs).max()))
                yt = torch.tensor(y, dtype=torch.float32)
                if huber_delta:
                    td_elem = 2.0 * torch.nn.functional.huber_loss(
                        q, yt, reduction="none", delta=float(huber_delta))
                else:
                    td_elem = (q - yt) ** 2
            if prioritized:                    # weighted TD loss + refresh sampled priorities
                loss = (w_t * td_elem).mean()
                with torch.no_grad():          # priority: |TD| for the scalar head, the per-transition
                    for j, i in enumerate(idx):    # quantile-Huber loss for the distributional head
                        prio[i] = (float(td_elem[j]) if distributional
                                   else float(abs(q[j] - yt[j]))) + float(hp["per_eps"])
            else:
                loss = td_elem.mean()
            if ex is not None:
                loss = loss + lam_ep * _hinge(Qf, net, ex[0], ex[1], MARGIN, torch)
            if rec is not None:
                with torch.no_grad():
                    ep_q.append(float(q.mean()))
                    ep_y.append(float(yt.mean()))
                    ep_td.append(float((q - yt).abs().mean()))
            loss.backward()
            if trunk_rescale:
                with torch.no_grad():
                    for prm in net.trunk.parameters():
                        if prm.grad is not None:
                            prm.grad.mul_(0.70710678)
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(net.parameters(), float(grad_clip))
            opt.step()
            gstep += 1
            if rec is not None:
                ep_loss.append(float(loss.detach()))
            if gstep % sync == 0:
                tgt.load_state_dict(net.state_dict())

        # --- plateau probe. The nominal variant watches its best-so-far order (free). The
        # stochastic
        # one has no cross-world best, so it scores the greedy policy on VALIDATION worlds, which
        # are disjoint from the evaluation scenarios by construction.
        gperm, _, _, _, _ = _rollout(env, greedy, n_lag)         # nominal greedy order (no UE cost)
        score, F_val, scen_F = None, None, None
        probe = (ep % sp["probe_every"] == 0)

        # DIAGNOSTIC ONLY, and recorded for BOTH variants: the greedy policy scored on the M frozen
        # EVALUATION scenarios -- the ruler every method is finally compared on. It answers "what
        # would this run have delivered at episode k?" without re-training, which no other recorded
        # curve can: the nominal variant's best-so-far F is a NOMINAL-world number and the
        # stochastic
        # variant's validation curve uses different worlds and a different horizon.
        #
        # It is deliberately kept OUT of `stop.update` for both variants. Stopping on the evaluation
        # sample would turn the reported mean into an optimistic best-of over the test set.
        #
        # Each variant is scored the way it actually DELIVERS, so the curve lands on the same number
        # run_rank finally reports: the stochastic policy is rolled out per scenario, the nominal
        # variant list-schedules its ONE committed order into every scenario. Costs M evaluations
        # per probe, most of them cache hits.
        if probe:
            if saa:
                scen_F = float(np.mean([
                    _evaluate_prefix_cached(_rollout(env, greedy, n_lag, d)[1], d, env["T"],
                                            env["ctx"], sc)["F"] for d in env["scen"]]))
            else:
                scen_F = float(np.mean([
                    _evaluate_prefix_cached(schedule_from_permutation(list(gperm), d), d,
                                            env["T"], env["ctx"], sc)["F"] for d in env["scen"]]))

        if saa:
            if probe:
                F_val = float(np.mean([
                    _evaluate_prefix_cached(_rollout(env, greedy, n_lag, dv)[1], dv, Ttr,
                                            env["ctx"], sc)["F"] for dv in val_worlds]))
                score = F_val
            stop.update(ep, score, gperm if probe else None)
        else:
            score = best_F
            stop.update(ep, score, gperm)

        if rec is not None:
            _m = lambda a: (float(np.mean(a)) if a else None)
            rec.episode(episode=ep, eps=float(eps), lam=float(lam_ep), F=float(res["F"]),
                        best_F=(None if saa else float(best_F)),
                        buf_bestF=(float(buf[0][2]) if (saa and buf) else None),
                        F_val=F_val, scenF=scen_F, greedy_order="-".join(map(str, gperm)),
                        loss=_m(ep_loss), q_pred_mean=_m(ep_q), y_target_mean=_m(ep_y),
                        td_abs_mean=_m(ep_td), since_improve=stop.since_improve, stable=stop.stable)
            rec.maybe_snapshot(ep, net)
        if verbose and ep % 25 == 0:
            print(f"  [{variant}] ep {ep}  eps={eps:.3f}  lam={lam_ep:.3f}  "
                  f"{'F_val=' + (f'{F_val:.4f}' if F_val is not None else '-') if saa else f'best_F={best_F:.4f}'}"
                  f"  since_improve={stop.since_improve}  stable={stop.stable}", flush=True)
        ep += 1

    outcome = stop.reason or "episode_cap"
    if rec is not None:
        rec.finish(net)

    # --- delivery: the FINAL policy, rolled INSIDE each scenario (both variants) ---
    # What training produces is a POLICY, not an order, so it is delivered as one: rolled inside a
    # given scenario it meets that scenario's own states and chooses differently, and the M delivered
    # decisions differ from one another. The nominal variant used to lock ONE nominal rollout and
    # apply that same order to every scenario, which threw the policy away and delivered a static
    # order it never needed to be a policy to produce.
    #
    # OBSERVED HISTORY ONLY, so this is not clairvoyance. Durations are drawn per (road_class,
    # severity) LEVEL and shared by every segment of that level (util.scenarios.draw_durations), so
    # _rollout holds each unfinished segment at its planning duration `dur_raw` and adopts a realized
    # duration only for levels that a COMPLETED segment has already exposed. No segment's own time is
    # ever read before the world has shown it.
    #
    # MEASURED COST, recorded because it argues against the change on results while the change is
    # still the right one on semantics (n10, 2026-08-12, the five saved nominal models, no
    # retraining): mean F 0.9552 -> 0.9633 and spread 0.0157 -> 0.0403, with no seed improving. The
    # nominal-trained Q has only ever seen nominal-world states, so the states adaptivity leads to
    # sit off its training manifold: the two seeds that produced ONE distinct order over 50 scenarios
    # were unchanged, while the seeds that adapted most (9 and 10 distinct orders) degraded most.
    # Adaptivity has to be TRAINED FOR to pay -- that is what rl_saa's random-scenario training is.
    per_scenario = []
    for d in env["scen"]:
        gp, gs, _, _, _ = _rollout(env, greedy, n_lag, d)
        per_scenario.append((list(gp), gs))
    # A summary order for provenance and the figures: what the policy does in the NOMINAL world. It
    # is no longer the delivered decision -- each scenario's own order in `per_scenario` is.
    order = list(_rollout(env, greedy, n_lag)[0])
    if verbose:
        print(f"  [{variant}] stopped at ep {ep} ({outcome}) in {(time.perf_counter()-t0)/60:.1f} min",
              flush=True)
    return dict(order=order, per_scenario=per_scenario, net=net, episodes=ep,
                n_evals=(ep if saa else len(memo)), outcome=outcome,
                best_score=(stop.best if saa else best_F), hp=hp, seed=seed)


# --------------------------------------------------------------------------- #
# Run + provenance
# --------------------------------------------------------------------------- #
def _deliver_analysis(env, variant, net, per_scenario, n_lag, PS, rec_dir, torch):
    """Post-hoc, no retraining: for the delivered policy record every candidate's features, its Q
    and whether it was chosen, the chosen action's representation, and Q against the realized
    Monte-Carlo return. This is what the policy-vs-intuition, representation and Q-calibration
    figures are drawn from."""
    drows, qmc, reps, rmeta = [], [], [], []
    for m, (dur, (order, _)) in enumerate(zip(env["scen"], per_scenario)):
        def fp(rem, X):
            for e in order:
                if e in rem:
                    return rem.index(e)
        perm, start, feats, picks, rems = _rollout(env, fp, n_lag, dur)
        res = _evaluate_prefix_cached(start, dur, env["T"], env["ctx"], {})
        rew, _ = _completion_rewards(perm, start, dur, res["terms"], env["T"], env["phi"])
        for step, (X, rem, a) in enumerate(zip(feats, rems, picks)):
            rp, q = repr_q(net, X, PS, torch)
            for i, seg in enumerate(rem):
                row = dict(scenario=m, step=step, seg=int(seg), is_chosen=int(i == a), q=float(q[i]))
                row.update({FEAT_COLS[c]: float(np.asarray(X[i])[c]) for c in range(_N_FEAT)})
                drows.append(row)
            reps.append(rp[a])
            rmeta.append((m, step, int(perm[step])))
            qmc.append(dict(scenario=m, step=step, seg=int(perm[step]), q_chosen=float(q[a]),
                            mc_return=float(sum(rew[step:]))))
    pd.DataFrame(drows).to_csv(os.path.join(rec_dir, f"{variant}_deliver.csv"), index=False)
    pd.DataFrame(qmc).to_csv(os.path.join(rec_dir, f"{variant}_qmc.csv"), index=False)
    rm = np.array(rmeta)
    np.savez_compressed(os.path.join(rec_dir, f"{variant}_deliver_repr.npz"),
                        repr=np.stack(reps), scenario=rm[:, 0], step=rm[:, 1], seg=rm[:, 2])


def _value_est_record(vdir, variant):
    """Record the value-estimate series for the van Hasselt-style figure: max_e Q(s0, e) at every
    Q snapshot, in log/{v}_value_est.csv. s0 is the same initial state under every order, so this
    is the network's value for the whole episode, in the same units as the episode return
    sum_t (1 - g_t) (gamma 0.9995 discounts only the bootstrap, negligible here). Free: read off
    the recorded qevo snapshots.

    The figure's realized-performance side comes from the trace's scenF probes, ALSO already
    recorded. A per-episode band across the M scenarios was tried and REMOVED on 2026-08-10: the
    intermediate greedy orders live only in the nominal world during training, so scoring all of
    them on the scenarios is almost entirely cache-cold -- the one attempt was killed at 415
    CPU-minutes, unfinished, for a single figure's shading. Anything wanting per-episode scenario
    scores must record them DURING the run, not recompute them after it."""
    tr = pd.read_csv(log_dir(vdir) / f"{variant}_trace.csv")
    # `with`: an .npz stays OPEN until closed, and on Windows a still-held file cannot be unlinked
    # by the next run's fresh_scale_dir.
    with np.load(log_dir(vdir) / f"{variant}_qevo.npz") as qevo:
        s0 = qevo["step"] == 0                     # the initial decision state's candidate rows
        eps_ = [int(tr["episode"].iloc[-1]) if e == -1 else int(e) for e in qevo["episodes"]]
        v_est = qevo["q"][:, s0].max(axis=1)
    pd.DataFrame(dict(episode=eps_, v_est=v_est)).to_csv(
        log_dir(vdir) / f"{variant}_value_est.csv", index=False)


def record_value_curve(variant="rl_nominal", N=None):
    """Post-hoc entry: write the value-estimate record for an already-delivered run from its own
    log, without retraining and without evaluating anything."""
    N = P.N_DISRUPTED_ORACLE if N is None else N
    vdir = scale_dir(OUT_DIAG / solver_dir(variant), N)
    _value_est_record(vdir, variant)
    print(f"[{variant}] value-estimate record written -> {log_dir(vdir)}", flush=True)


def run_rank(variants=("rl_nominal", "rl_saa"), toy_dir=TOY, N=None, M=P.M_SCENARIOS, seed=P.SEED,
             ep_cap=EP_CAP, hp=None, stop_params=None):
    """Train each variant to a plateau and write its canonical results and diagnostics.

    Outputs go to the variant's OWN folder (outputs/{variant}/n{N}/): {variant}_optima.csv, the
    per-slot recovery curve in raw/, the diagnostics in raw/, and run_meta.json plus the delivered
    network in cache/. A rerun REPLACES that folder: it holds the latest run, and only that.
    """
    import torch
    N = P.N_DISRUPTED_ORACLE if N is None else N
    env = build_env(toy_dir, N=N, M=M)
    print(f"instance: {len(env['segs'])} segments {env['segs']}; M={M}; T={env['T']} "
          f"(T_train={env['T_train']}); seed={seed}; ep_cap={ep_cap}; variants={list(variants)}",
          flush=True)
    out = {}
    for v in variants:
        vdir = scale_dir(OUT_DIAG / solver_dir(v), N)
        vdir.mkdir(parents=True, exist_ok=True)
        # Stage 1 of the two-stage clear: diagnostics only. The previous run's DELIVERABLES
        # (results/ + config/) survive until this run has something to replace them with, so a
        # rerun that dies mid-training cannot destroy a delivered model it never reproduced.
        fresh_scale_dir(vdir, subdirs=("log",), figures=True)
        rdir = log_dir(vdir)
        t0 = time.perf_counter()

        r = train(env, variant=v, hp=hp, seed=seed, ep_cap=ep_cap, rec_dir=str(rdir),
                  stop_params=stop_params)
        rows, slots = [], []
        for m, (dur, (order_m, start)) in enumerate(zip(env["scen"], r["per_scenario"])):
            ts = time.perf_counter()
            res = _evaluate_prefix_cached(start, dur, env["T"], env["ctx"], {}, collect_traces=True)
            slots.extend(slot_rows(m, res))
            row = dict(scenario=m, F=res["F"], F1=res["F1"], F2=res["F2"],
                       time_s=time.perf_counter() - ts, n_evals=r["n_evals"],
                       episodes=r["episodes"], outcome=r["outcome"],
                       # Serial-equivalent compute for THIS scenario: every evaluation the search
                       # performed costs a full pass (T UE solves at the horizon the training used),
                       # amortised over the M scenarios the one search serves, plus this scenario's
                       # own evaluation. The slot cache makes many of those solves free in practice
                       # and is deliberately NOT credited -- it is an engineering convenience, not a
                       # property of the method. For the SAA variant each EPISODE evaluates saa_n
                       # worlds (n_evals counts episodes), so the honest count multiplies by saa_n:
                       # at saa_n=20 the search does 20 full evaluations per episode, and hiding that
                       # behind the cache would understate its cost 20x. (Before 2026-08-11 this line
                       # divided the ORDER count by M and so understated the cost by a factor of T.)
                       ue_total=(r["n_evals"]
                                 * (int(r["hp"].get("saa_n", 1) or 1) if v.startswith("rl_saa") else 1)
                                 * (env["T_train"] if v.startswith("rl_saa") else env["T"])
                                 / len(env["scen"]) + env["T"]),
                       order="-".join(map(str, order_m)),
                       durations="-".join(str(int(dur[e])) for e in env["segs"]))
            # Recorded for BOTH variants since both now deliver per-scenario: `order` above is
            # THIS scenario's decision, this column is the policy's nominal-world summary, and
            # comparing them shows how far the policy moved off nominal for this scenario.
            row["policy_order_nominal"] = "-".join(map(str, r["order"]))
            for e in env["segs"]:
                row[f"start_{e}"] = start[e]
            rows.append(row)
        fresh_scale_dir(vdir, subdirs=("results", "config"), figures=False)   # stage 2: now replace
        pd.DataFrame(rows).to_csv(results_dir(vdir) / f"{v}_optima.csv", index=False)
        pd.DataFrame(slots).to_csv(rdir / f"{v}_slots.csv", index=False)
        torch.save(r["net"].state_dict(), results_dir(vdir) / "model_best.pt")
        if str(r["hp"].get("arch", "rows")) == "dqn":
            print(f"  [{v}] deliver-analysis skipped: repr/qmc probes are row-architecture-"
                  f"specific", flush=True)
        else:
            _deliver_analysis(env, v, r["net"], r["per_scenario"], int(r["hp"]["n_lag"]),
                              float(r["hp"]["prior_scale"]), str(rdir), torch)
            if not v.startswith("rl_saa"):
                _value_est_record(vdir, variant=v)
        meanF = float(np.mean([x["F"] for x in rows]))
        write_run_meta(vdir, method=v, segments=env["segs"], T=env["T"], seed=seed, M=M,
                       hp={k: (list(x) if isinstance(x, tuple) else x) for k, x in r["hp"].items()},
                       # the rule that ACTUALLY governed this run: the shared defaults, the
                       # variant's override, then the caller's. Recording the bare defaults would
                       # misdescribe the run, and the stability figure reads its thresholds here.
                       stop_params=_merge_stop(v, stop_params), ep_cap=ep_cap,
                       episodes=r["episodes"], outcome=r["outcome"],
                       # NOT the delivered decision any more -- the policy's nominal-world summary
                       # order. Each scenario's actual decision is its own row in {v}_optima.csv.
                       order_nominal_summary=r["order"],
                       mean_F=meanF,
                       solver="rank-loss RL (util.rl_rank)",
                       delivery="per-scenario adaptive policy (final), observed history only")

        # Figures are drawn AFTER run_meta exists. Stage 2 above deletes config/, so drawing before
        # this point left the stability figure with no stop_params to read and it annotated itself
        # "fixed episode cap (no stop rule)" on runs that did stop on a plateau -- a figure stating
        # the opposite of what governed the run.
        try:                                    # final redraw, now that every record exists
            from viz.rank_viz import make_rank_figures
            make_rank_figures(vdir, v)
        except Exception as exc:
            print(f"  [{v}] final figure draw failed ({type(exc).__name__}: {exc})", flush=True)
        print(f"[{v}] mean F = {meanF:.4f}  ({r['episodes']} eps, {r['outcome']}, "
              f"{(time.perf_counter()-t0)/60:.1f} min)  -> {vdir}", flush=True)
        out[v] = meanF
    from util.compare import refresh_comparison
    refresh_comparison(N)                    # every solver run leaves the comparison current
    return out


def run_search_sweep(configs=None, variant="rl_nominal", N=None, M=P.M_SCENARIOS, seed=P.SEED,
                  ep_cap=EP_CAP):
    """Screen search-coverage knobs (SEARCH_SWEEP), each as an ordinary full run.

    Each config is a plain run_rank call, so it writes the variant's n{N} folder exactly as any run
    does and REPLACES what was there -- when the sweep ends, that folder holds the last config, not
    a winner. What survives the sweep is the ranked table, written to log/search_sweep.csv and read
    back from each run's own run_meta before the next run overwrites it. Adopting a schedule means
    editing RANK_PARAMS_NOMINAL and rerunning, which is the same thing every other tuning decision here
    has meant.

    Screening is NOT adopting: this is one seed, and this solver's seed spread exceeds the
    between-method gaps (see the caveat on RANK_PARAMS_NOMINAL). A winner here is a candidate to confirm
    across seeds, not a result.
    """
    configs = SEARCH_SWEEP if configs is None else configs
    N = P.N_DISRUPTED_ORACLE if N is None else N
    vdir = scale_dir(OUT_DIAG / solver_dir(variant), N)
    rows = []
    for name, ov in configs.items():
        print(f"\n--- exploration config {name!r}: {ov or 'tuned defaults'} ---", flush=True)
        run_rank(variants=(variant,), N=N, M=M, seed=seed, ep_cap=ep_cap, hp=(ov or None))
        meta = pd.read_json(vdir / "config" / "run_meta.json", typ="series")
        # n_evals = DISTINCT orders the run scored. This is the coverage number the screen is about,
        # so it is read from the results table rather than inferred from the episode count.
        opt = pd.read_csv(vdir / "results" / f"{variant}_optima.csv")
        rows.append(dict(config=name, override=str(ov), mean_F=float(meta["mean_F"]),
                         episodes=int(meta["episodes"]), n_evals=int(opt["n_evals"].iloc[0]),
                         outcome=meta["outcome"],
                         order="-".join(map(str, meta["order_nominal_summary"]))))
        print(f"    -> mean F {rows[-1]['mean_F']:.4f}  ({rows[-1]['episodes']} eps, "
              f"{rows[-1]['n_evals']} distinct orders, {rows[-1]['outcome']})", flush=True)
    # Written once, at the end. It CANNOT be written per config: every run_rank clears log/ before
    # it trains, so a running config would erase the summary of the ones before it. The per-config
    # lines printed above are what an interrupted sweep leaves behind.
    df = pd.DataFrame(rows).sort_values("mean_F").reset_index(drop=True)
    df.to_csv(log_dir(vdir) / "search_sweep.csv", index=False)
    print(f"\n{df.to_string(index=False)}\n-> {log_dir(vdir) / 'search_sweep.csv'}", flush=True)
    return df
