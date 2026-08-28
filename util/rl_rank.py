"""
Shared substrate for the S2V reinforcement-learning solvers: the environment, the rollout, the
plateau stopping rule, the diagnostics recorder, the n-step helper, and the two parameter blocks
those solvers read.

WHAT THIS IS NOT, ANY MORE. Until 2026-08-27 this module also held a solver of its own, `rl_dqn`
-- a ranking-loss DQN over a flat per-candidate feature vector, trained with a large-margin term
instead of pure TD regression. That solver and its runner were removed at the project owner's
request. Nothing here trains any more; util/rl_s2v.py and util/rl_s2v_saa.py own the training
loops, and they build their own networks, their own losses and their own stop overrides.

WHY THE SUBSTRATE OUTLIVED THE SOLVER. The pieces below define what the PROBLEM looks like to a
learner, not how a particular learner attacks it:

  build_env      the instance, the frozen scenario sample, the horizon, the message-passing
                 statics and the flow prior -- one construction every solver shares, so two
                 methods can never be compared across subtly different environments.
  _rollout       one episode, mirroring util.evaluate.schedule_from_permutation decision for
                 decision under the crew-accessibility constraint, so an order rolled here scores
                 exactly as the delivery scores it. Its helpers _augment / _nbr_cols /
                 _current_flow come with it.
  RankPlateauStop  the repo-wide stopping rule: stop on a plateau, not at a fixed episode count.
  _Recorder      the per-episode diagnostics trace every RL figure is drawn from (with repr_q,
                 the representation/Q probe it snapshots through).
  _nstep_rows    the n-step return rows, shared by both S2V trainers.

RANK_PARAMS_DQN keeps its historical name because util/rl_s2v.py imports it by that name and
reads its regularization constants (prioritized replay's alpha/beta, the hinge's lam / margin /
lam0 / lam_growth) so the two solvers regularize identically. The measurements recorded in that
block were taken on the retired rl_dqn; they are kept because they are what those numbers were
tuned against, not because the solver still exists.

STOP_PARAMS is likewise shared, and is imported by both S2V trainers, each of which widens the
patience through its own override block.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd

import config as P
from util.evaluate import (_matrix_from_H, accessible_segments, build_context,
                           build_damaged_edges)
from util.oracle import _baseline_twoway_flow, compute_horizon, select_oracle_instance
from util.rl import _decision_features, _scenario_statics
from util.scenarios import nominal_durations, sample_scenarios
from util.ue import solve_ue

ROOT = Path(__file__).resolve().parent.parent
TOY = ROOT / "data" / "siouxfalls_toy"
OUT_DIAG = ROOT / "outputs" / "03-rl"   # the RL root; each solver resolves its own subfolder
                                        # under it through util.provenance.solver_dir
_N_FEAT = 8                          # base per-candidate features; see util.rl._decision_features


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
RANK_PARAMS_DQN = dict(
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


# Plateau stopping. patience_P and stable_K are counted in PROBES, not episodes: rl_dqn
# probes every episode (its best-so-far order is already computed, so a probe is free). The
# probe_every and n_val keys below cost it nothing and are kept because util/rl_s2v_saa.py, which
# probes on validation worlds, reads this same block. ep_cap is the outer guard, not the intended
# terminator.
STOP_PARAMS = dict(
    ep_min=60,           # floor: no run stops before this many episodes, however flat it looks
    patience_P=10,       # probes without an improvement beating `tol` -> plateau
    stable_K=25,         # probes with an unchanged greedy order -> the policy has settled
    tol=1e-3,            # relative improvement below this does not count as an improvement
    probe_every=25,      # episodes between validation probes (validation-world solvers only)
    n_val=11,            # validation worlds, drawn once from a dedicated stream (same solvers).
                         # 11 rather than a handful because the stopping rule reads their MEAN:
                         # with 3 worlds that mean is noisy enough that a real improvement can be
                         # missed and a chance dip can pass for one, which is exactly the signal
                         # the patience counter is built on
)

EP_CAP = 1000            # outer guard only; the plateau rule is what should end a run. Deliberately
                         # far above where the plateau fires, so a run that hits this cap is a
                         # signal that the plateau parameters are wrong, not a normal ending.

# Episodes between redraws of this variant's figures in its n{N} folder, so a long run is readable
# as figures while it is still going instead of only after it delivers. _Recorder.flush does the
# redraw off the records it has just written, which is what keeps a mid-run figure from disagreeing
# with the trace beside it. Nothing else is written mid-run: the deliverable is written once, at the
# end, so results/ never holds a half-finished run.
FIGURE_REFRESH_EVERY = 20


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
    # T_train here covers only the NOMINAL world (the one every trainer touches); a stochastic
    # trainer recomputes it over the worlds it actually draws (see train), because sizing it for
    # the distribution's worst case (the old worst_case_durations bound) would, under the serial
    # horizon the accessibility constraint requires, make every training evaluation pay for a
    # pathological world that fixed-sample training never visits.
    T_train = max(T, compute_horizon(segs, [nominal]))
    if T_train != T:
        # Several trainers evaluate nominal-world rollouts at T (their probes share the frozen
        # ruler's memo); a nominal world outrunning every frozen scenario would silently truncate
        # those evaluations, so it is refused rather than absorbed.
        raise RuntimeError(f"nominal world needs a horizon of {T_train} slots but the frozen "
                           f"evaluation sample only spans T={T}; refusing to truncate")
    st = _scenario_statics(ctx, segs, nominal, phi)
    return dict(dis=dis, segs=segs, ctx=ctx, phi=phi, nominal=nominal, scen=scen, T=T,
                T_train=T_train, st=st, mp=mp,
                seg_idx={int(eid): j for j, (eid, _, _, _) in enumerate(ctx["disrupted"])})


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
    decision point and schedule whichever candidate it returns, over C_MAX crews UNDER THE
    ACCESSIBILITY CONSTRAINT (config's accessibility block): the candidate set at each decision is
    the ACCESSIBLE remaining segments -- those a crew can reach from the depot through passable
    (undamaged or completed) roads -- and when none is accessible the free crew idles until the
    next completion opens new frontier. The action set, the feature rows, the recorded `rems`, and
    therefore every downstream consumer (TD max over next-state rows, hinge over same-state rows,
    dueling mean) range over exactly the legal actions. This recursion mirrors
    util.evaluate.schedule_from_permutation decision for decision, so a fixed order rolled here
    produces the identical gated schedule the delivery is scored with.

    `durations` is the world the rollout EXECUTES in (the nominal world by default). Duration
    beliefs stay at the planning expectation throughout: realized durations shape the schedule as
    crews free up, but are never read into the state. (The progressive-revelation mechanism this
    solver once had is still absent. Its original retirement argument -- under the 2026-08-24
    per-segment-independent lognormal law a completion reveals nothing about OTHER segments'
    durations -- remains true and still bounds how much a revelation channel can pay here. What
    changed on 2026-08-25 is only that building such a channel is no longer excluded; see
    technical_notes/05-problem_redefinition.md sec.6.5 for the three levers and their costs.)"""
    ctx, st, T = env["ctx"], env["st"], env["T"]
    durations = env["nominal"] if durations is None else durations
    dis_l, sev, B, H0 = ctx["disrupted"], ctx["severity_vec"], ctx["B"], ctx["H0"]
    seg_idx = env["seg_idx"]
    access = ctx["access"]
    remaining = list(env["segs"])
    crew = [1] * P.C_MAX
    start, perm, feats, picks, rems, chosen_base = {}, [], [], [], [], []
    D = np.zeros(len(H0))
    k, comp = 0, {}
    while remaining:
        t = min(crew)
        # Accessible candidates at slot t; if none, the free crew waits for the next completion.
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
        cand = [e for e in remaining if e in acc]      # legal actions, in remaining-list order
        while k < t:                                   # advance the demand shortfall to slot t
            k += 1
            v = np.zeros(len(dis_l))
            for e in remaining:
                v[seg_idx[e]] = sev[seg_idx[e]]
            for e, c in comp.items():
                if k < c:
                    v[seg_idx[e]] = sev[seg_idx[e]]
            D = np.maximum(B @ v, P.RHO * D)
        bel = {e: st["dur_raw"][e] for e in remaining}
        base_X = _decision_features(cand, t, crew, D, st, T, bel, work_set=remaining)
        if int(st.get("mp_rounds", 0)):
            dmg = set(remaining) | {e for e, c in comp.items() if c > t}
            base_X = np.hstack([base_X,
                                _nbr_cols(env, cand, dmg, _current_flow(env, dmg, D))])
        X = _augment(base_X, chosen_base, n_lag)
        rems.append(list(cand))
        i = pick(cand, X)
        feats.append(X)
        picks.append(i)
        chosen_base.append(base_X[i])
        e = cand[i]
        remaining.remove(e)
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


