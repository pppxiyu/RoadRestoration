"""
Reinforcement-learning baseline solver for the road-restoration scheduling problem: a DQN
(deep Q-network) whose action-value function is initialized from the flow-greedy baseline.

Like GA/PSO the agent searches repair PRIORITY ORDERS: an episode builds a permutation of the
disrupted segments one decision at a time (each decision assigns one segment to the crew that
frees up earliest, exactly the work-conserving list schedule the other solvers use), and is
scored by the identical true objective F. Each scenario is a DETERMINISTIC MDP and is trained
COMPLETELY INDEPENDENTLY -- one Q-network per scenario, no parameter sharing -- so the RL result
is a per-scenario search, directly comparable to the per-scenario GA/PSO runs.

Key design choices, in the order they matter:

  * Flow initialization, not pretraining. The Q-network is the residual form
        Q(s, e) = prior_scale * phi_hat_e + h(x(s, e)),
    where phi_hat_e is the segment's baseline-UE two-way flow normalized to [0, 1] (the same
    score the static flow greedy ranks by) and h is a small MLP whose OUTPUT layer starts at
    exactly zero. This is a weight CONSTRUCTION, not a training phase: at initialization
    h == 0, so the greedy rollout of episode 0 (forced eps = 0) reproduces the flow-greedy
    order exactly. Combined with best-by-true-F selection over every schedule evaluated, the
    RL output is guaranteed no worse than the flow baseline on every scenario.

  * Completion-attributed reward. The reward for scheduling segment e is the objective
    improvement ITS OWN completion causes, read off the already-evaluated trajectory: when e
    completes at slot tau_e (its first usable slot), the consecutive per-slot accessibility
    terms of the true evaluator give the jump
        delta_e = g_{tau_e - 1} - g_{tau_e},
    and the reward is that jump plus its retained, RHO-decaying echo over the remaining
    horizon,
        R_i = delta_e * sum_{j=0}^{T - tau_e} RHO^j,
    mirroring the demand model itself: suppressed trips flow back at rate RHO and eat part of
    the improvement, so credit decays at the same rate. Two segments completing in the same
    slot share one joint jump, split in proportion to their normal-period flow phi_e. This is
    the counterfactual reward of Babaee et al. (2026) with the re-simulated "without repair"
    tail replaced by the trajectory's own pre-completion slot -- per-road attribution at ZERO
    extra UE solves. The returns do NOT sum to the objective (each jump is extrapolated over
    the tail); the reward is purely the LEARNING signal, and the deliverable is still selected
    by best-by-true-F, which is what keeps the output aligned with the true F.

  * Budgeted like the metaheuristics. A schedule (permutation) is evaluated at most once; the
    search stops after `budget` UNIQUE true evaluations (plus a stall guard mirroring GA/PSO:
    40 consecutive episodes that only replay already-known schedules end the run). Episodes
    that rehit a cached schedule still train the network -- their rewards are free.

  * Prefix-cached evaluation. Per-slot terms g_t depend only on the damage-state trajectory
    up to t, so schedules sharing a prefix (very common: eps-greedy edits the tail of the
    incumbent order far more often than its head) reuse each other's UE solves slot by slot.
    This changes NOTHING numerically -- _evaluate_prefix_cached mirrors
    util.evaluate.evaluate_schedule term for term -- it only cuts wall-clock. The budget and
    the comparison's compute axis still count full evaluations (n_evals * T UE solves), which
    can only OVERSTATE the RL compute, never understate it.

Each run writes outputs/greedy/n{N}/rl_optima.csv (the schema the static greedy and GA/PSO
solvers use, so util/compare.py discovers it automatically) plus its own training diagnostics
under outputs/rl/n{N}/: rl_trace.csv (one row per episode) and four figures -- the learner view
(mean |TD error| per episode, i.e. did the value regression converge at all), the cross-scenario
search view (best-so-far F vs budget), the per-scenario objective panels (sampled F, its
best-so-far envelope and the flow baseline), and the policy view (per-episode return).

Reproducibility: every UE solve on the RL path (the per-slot evaluations, the worker's baseline
context, the phi prior) runs with cores=1, because AequilibraE's multi-threaded assignment is not
bit-stable run to run (~1e-10 cross-thread reduction noise) and the eps-greedy argmax amplifies
such noise into entirely different schedules. With the engine pinned, a fixed seed reproduces a
run bit for bit; the scenario-parallel pool keeps all cores busy anyway, so pinning costs nothing
(on this small network one core per solve is in fact slightly faster than threading).

Run inside the road_restore conda env (PYTHONPATH = project root); a large-n run only needs the
N_DISRUPTED_ORACLE override, as with the other solvers:
  python -m util.rl
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd

import config as P
from util.evaluate import (_matrix_from_H, build_context, build_damaged_edges,
                           f2_value, od_travel_times, schedule_from_permutation)
from util.oracle import (_baseline_twoway_flow, compute_horizon, scale_dir,
                         select_oracle_instance)
from util.scenarios import sample_scenarios
from util.ue import solve_ue

ROOT = Path(__file__).resolve().parent.parent
TOY = ROOT / "data" / "siouxfalls_toy"
OUT_OPTIMA = ROOT / "outputs" / "greedy"     # rl_optima.csv joins the baselines for auto-discovery
OUT_RL = ROOT / "outputs" / "rl"             # training diagnostics: rl_trace.csv + figures

# Default hyperparameters, re-tuned for the completion-attributed reward on the n=10 instance at
# the production budget (2026-07-31). The previous values had been chosen under the old window
# reward, whose returns were of order 0.1; the completion reward's are of order 3.3, a ~30x
# rescale of every Bellman target, so lr, the prior weight and the exploration schedule were all
# being used outside the regime they were selected in.
#
# Protocol: stage A screened 28 configurations drawn at random from the joint space (lr and
# prior_scale log-uniform, the rest categorical) on scenario 8 with one seed; stage B re-ran the
# top six on scenarios 8/2/9 with two seeds each and ranked them by mean best-true-F. Random
# rather than coordinate-wise search because the knobs interact, and a two-stage screen because
# one seed cannot separate close configurations -- indeed stage A's winner placed only second
# under the robust protocol, and the apparent stage-A pattern (small lr, fast eps decay) did NOT
# survive it. The selected configuration is the only one of the six that beat the previous
# defaults at a matched seed (0.9589 vs 0.9622 on the three tuning scenarios).
#
# CAVEAT worth carrying: the margin over the previous defaults (~0.003) is smaller than this
# solver's seed-to-seed spread (~0.012), so these values are the best available choice rather
# than a demonstrated optimum. The tuning scenarios are also among the ten finally reported --
# there is no held-out set at M = 10.
#
# gamma is NOT tunable: the finite horizon is already carried by the reward's own RHO-decaying
# retention, so the Bellman recursion is left undiscounted at 1.
RL_PARAMS = dict(
    hidden=(16,),        # widths of the MLP h's hidden layers (tanh), output layer starts at 0
    prior_scale=0.82,    # weight of the flow prior phi_hat in Q: strong enough to anchor the
                         # search at flow, small enough that learned corrections can re-rank
    lr=1.81e-2,          # Adam learning rate
    eps0=0.2,            # eps-greedy exploration rate at episode 1 (episode 0 is forced greedy)
    eps_decay=0.98,      # per-episode multiplicative decay of eps
    eps_min=0.10,        # exploration floor: the tuned schedule starts low and stays broad, the
                         # opposite of the old reward's preference for a high, fast-decaying eps
    updates_per_ep=60,   # gradient steps on the replay buffer after each episode
    batch=16,            # minibatch size per gradient step
    target_sync=25,      # gradient steps between target-network syncs
    # --- termination (see the stopping rule in _train_one_scenario) ---
    ep_min=45,           # never judge a plateau before eps has annealed to its floor (~41 episodes)
    stable_K=15,         # consecutive episodes the GREEDY order must repeat to count as settled
    patience_P=30,       # consecutive episodes without a best-so-far improvement
    ep_cap=900,          # hard safety net on episodes; the plateau rule is expected to fire first
)

# Ceiling on unique true evaluations per scenario, kept identical to util.metaheuristic.BUDGET_CAP
# so all three searches are held to the same compute. The plateau rule can end a run earlier;
# whichever condition comes first decides.
BUDGET_CAP = 60

_N_FEAT = 8              # feature-vector length of x(s, e); see _decision_features


# --------------------------------------------------------------------------- #
# Exact objective with a per-slot prefix cache
# --------------------------------------------------------------------------- #
def _evaluate_prefix_cached(start, durations, T, ctx, slot_cache):
    """Exactly util.evaluate.evaluate_schedule, restated so each slot's accessibility term g_t
    can be memoized on the damage-state trajectory that produced it.

    g_t is a deterministic function of the sequence of damage states over slots 1..t (the demand
    shortfall D_t carries memory, so the whole prefix matters, not just slot t's state). That
    prefix is encoded by each segment's completion slot capped at t+1 -- capping at t+1 rather
    than t keeps "completed exactly at t" distinct from "still damaged at t". Two schedules with
    equal capped vectors share g_1..g_t, so eps-greedy episodes that keep the incumbent head and
    explore the tail reuse most of the horizon's UE solves.

    Returns dict(F, F1, F2, terms, n_ue): `terms` are the per-slot g_t the reward needs, `n_ue`
    counts the UE solves actually run (cache misses). Cross-checked against evaluate_schedule in
    the validation suite; any numerical edit here must keep that equivalence."""
    dis = ctx["disrupted"]
    H0, B = ctx["H0"], ctx["B"]
    base_u = ctx["baseline_u"]
    sev = ctx["severity_vec"]

    F2 = f2_value(start, durations)
    comp = tuple(start[eid] + durations[eid] for (eid, _, _, _) in dis)   # completion slot per segment, in dis order

    D = np.zeros(len(H0))
    terms, active = [], []
    n_ue = 0
    for k in range(1, T + 1):
        still = np.array([k < c for c in comp])
        v_vec = np.where(still, sev, 0.0)
        target = B @ v_vec
        D = np.maximum(target, P.RHO * D)         # shortfall: jumps with damage, decays at RHO
        H = np.clip(H0 - D, 0.0, None)
        key = (k, tuple(min(c, k + 1) for c in comp))
        term = slot_cache.get(key)
        if term is None:
            damaged = {eid: s for (eid, _, _, s), st in zip(dis, still) if st}
            links, _ = solve_ue(build_damaged_edges(ctx, damaged), _matrix_from_H(H, ctx),
                                ctx["zone_ids"], rgap=P.UE_RGAP, max_iter=P.UE_MAX_ITER,
                                quiet=True, cores=1)     # cores=1: bit-reproducible (see module docstring)
            u = od_travel_times(links, ctx)
            u_tilde = np.where(np.isfinite(u), u, ctx["u_pen"])
            den = float(np.sum(H * base_u))
            term = float(np.sum(H * u_tilde) / den) if den > 0 else 1.0
            slot_cache[key] = term
            n_ue += 1
        terms.append(term)
        active.append(bool(still.any()))

    terms = np.asarray(terms)
    if P.F1_ACTIVE_ONLY:
        mask = np.asarray(active, dtype=bool)
        F1 = float(terms[mask].mean()) if mask.any() else float(terms.mean())
    else:
        F1 = float(terms.mean())
    F = P.MU * F1 + (1.0 - P.MU) * F2
    return dict(F=F, F1=F1, F2=F2, terms=terms, n_ue=n_ue)


# --------------------------------------------------------------------------- #
# Tiny numpy MLP with Adam (no framework dependency)
# --------------------------------------------------------------------------- #
class _MLP:
    """Fully-connected tanh network, linear output of size 1. The output layer is initialized to
    EXACTLY zero so the network starts as the constant 0 -- that is what makes the residual Q
    equal the flow prior at initialization. Hidden layers use Xavier-uniform init."""

    def __init__(self, sizes, rng):
        self.W, self.b = [], []
        for fi, fo in zip(sizes[:-1], sizes[1:]):
            s = np.sqrt(6.0 / (fi + fo))
            self.W.append(rng.uniform(-s, s, size=(fi, fo)))
            self.b.append(np.zeros(fo))
        self.W[-1][:] = 0.0                     # h == 0 at init: Q starts as the flow prior
        self.b[-1][:] = 0.0
        # Adam state (one slot per parameter array) and its step counter.
        self._m = [np.zeros_like(w) for w in self.W] + [np.zeros_like(b) for b in self.b]
        self._v = [np.zeros_like(w) for w in self.W] + [np.zeros_like(b) for b in self.b]
        self._t = 0

    def forward(self, X):
        """Return (out, cache): out is the (n,) network output, cache holds the layer
        activations needed by backward()."""
        A = [np.asarray(X, dtype=float)]
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            Z = A[-1] @ W + b
            A.append(np.tanh(Z) if i < len(self.W) - 1 else Z)   # hidden tanh, linear head
        return A[-1].ravel(), A

    def backward(self, cache, grad_out):
        """Gradients of a scalar loss w.r.t. every parameter, given dLoss/d(out) as (n,).
        Returns (gW, gb) lists aligned with self.W / self.b."""
        A = cache
        delta = np.asarray(grad_out, dtype=float).reshape(-1, 1)
        gW = [None] * len(self.W)
        gb = [None] * len(self.b)
        for i in range(len(self.W) - 1, -1, -1):
            gW[i] = A[i].T @ delta
            gb[i] = delta.sum(axis=0)
            if i > 0:
                delta = (delta @ self.W[i].T) * (1.0 - A[i] ** 2)   # tanh' = 1 - tanh^2
        return gW, gb

    def adam_step(self, gW, gb, lr, beta1=0.9, beta2=0.999, eps=1e-8):
        self._t += 1
        params = self.W + self.b
        grads = gW + gb
        for p, g, m, v in zip(params, grads, self._m, self._v):
            m[:] = beta1 * m + (1 - beta1) * g
            v[:] = beta2 * v + (1 - beta2) * g * g
            mh = m / (1 - beta1 ** self._t)
            vh = v / (1 - beta2 ** self._t)
            p -= lr * mh / (np.sqrt(vh) + eps)

    def copy_weights_from(self, other):
        for w, ow in zip(self.W, other.W):
            w[:] = ow
        for b, ob in zip(self.b, other.b):
            b[:] = ob


# --------------------------------------------------------------------------- #
# The per-scenario MDP: features, rollout, and DQN training
# --------------------------------------------------------------------------- #
def _scenario_statics(ctx, segments, durations, phi):
    """Per-scenario constants the feature builder needs, precomputed once. All per-segment
    scores are normalized to [0, 1] by their max over the disrupted set so the MLP sees
    commensurate inputs; phi_hat's ordering (all that the flow prior uses) is scale-invariant."""
    dis, sev, B = ctx["disrupted"], ctx["severity_vec"], ctx["B"]
    dem = {int(dis[j][0]): float(sev[j] * B[:, j].sum()) for j in range(len(dis))}
    sv = {int(eid): float(s) for (eid, _, _, s) in dis}
    d_max = max(int(durations[e]) for e in segments)
    dem_max = max(dem.values()) or 1.0
    phi_max = max(phi.values()) or 1.0
    total_work = float(sum(int(durations[e]) for e in segments))
    return dict(
        phi_hat={e: phi[e] / phi_max for e in segments},
        sev_hat={e: sv[e] / 3.0 for e in segments},
        dur_hat={e: int(durations[e]) / d_max for e in segments},
        dem_hat={e: dem[e] / dem_max for e in segments},
        total_work=total_work, sum_H0=float(ctx["H0"].sum()),
    )


def _decision_features(cands, t_now, crew_free, D_now, st, durations, T):
    """Feature matrix x(s, e) for every candidate segment e at one decision point. Row layout
    (all roughly [0, 1]): [phi_hat_e, sev_hat_e, dur_hat_e, dem_hat_e, t/T, crew-gap, demand-
    shortfall fraction, remaining-work fraction]. Column 0 doubles as the flow prior that the
    residual Q adds outside the MLP."""
    glob = (t_now / T,
            (max(crew_free) - t_now) / T,                        # how much longer the other crew is busy
            float(D_now.sum()) / st["sum_H0"],                    # demand currently suppressed
            sum(int(durations[e]) for e in cands) / st["total_work"])
    X = np.empty((len(cands), _N_FEAT))
    for i, e in enumerate(cands):
        X[i, 0] = st["phi_hat"][e]
        X[i, 1] = st["sev_hat"][e]
        X[i, 2] = st["dur_hat"][e]
        X[i, 3] = st["dem_hat"][e]
        X[i, 4:] = glob
    return X


def _q_values(net, X, prior_scale):
    """Residual action values Q(s, e) = prior_scale * phi_hat_e + h(x): the flow prior enters
    through a fixed skip on feature column 0, scaled into the value range."""
    h, _ = net.forward(X)
    return prior_scale * X[:, 0] + h


def _rollout(net, eps, rng, ctx, segments, durations, T, st, prior_scale):
    """Play one episode: build a full priority order with eps-greedy decisions over the residual
    Q. Candidates are kept in ascending edge-id order so an argmax tie resolves to the smallest
    edge id -- the same tiebreak the static greedy rankers use, which is what lets episode 0
    reproduce the flow order exactly.

    The demand-shortfall recursion is advanced slot by slot WITHOUT any UE solve (D_t depends
    only on the damage trajectory), so state features are free; UE is paid only once per novel
    schedule, at evaluation time. Returns (perm, start, feats, choices): the repair order, its
    start slots, and the per-decision candidate feature matrices with the index chosen from
    each, which is what the replay buffer stores."""
    dis, sev, B, H0 = ctx["disrupted"], ctx["severity_vec"], ctx["B"], ctx["H0"]
    seg_idx = {int(eid): j for j, (eid, _, _, _) in enumerate(dis)}
    remaining = list(segments)                                  # ascending edge ids
    crew_free = [1] * P.C_MAX
    start, perm = {}, []
    feats, choices = [], []

    D = np.zeros(len(H0))
    k_sim = 0                                                   # last slot the shortfall recursion has reached
    comp = {}                                                   # completion slot of already-assigned segments
    while remaining:
        t_now = min(crew_free)
        # Advance D up to slot t_now: unassigned segments are damaged; assigned ones are damaged
        # until their completion slot. This mirrors the evaluator's recursion exactly.
        while k_sim < t_now:
            k_sim += 1
            v_vec = np.zeros(len(dis))
            for e in remaining:
                v_vec[seg_idx[e]] = sev[seg_idx[e]]
            for e, c in comp.items():
                if k_sim < c:
                    v_vec[seg_idx[e]] = sev[seg_idx[e]]
            D = np.maximum(B @ v_vec, P.RHO * D)

        X = _decision_features(remaining, t_now, crew_free, D, st, durations, T)
        if rng.rand() < eps:
            i_pick = int(rng.randint(len(remaining)))
        else:
            i_pick = int(np.argmax(_q_values(net, X, prior_scale)))   # first max = smallest edge id
        e = remaining.pop(i_pick)

        c = int(np.argmin(crew_free))
        start[e] = crew_free[c]
        crew_free[c] = start[e] + int(durations[e])
        comp[e] = crew_free[c]
        perm.append(e)
        feats.append(X)
        choices.append(i_pick)
    return tuple(perm), start, feats, choices


def _completion_rewards(perm, start, durations, terms, T, phi):
    """Per-decision rewards, each one the objective improvement that decision's OWN segment
    causes when it comes back into service.

    Segment e finishes repair at slot tau_e = start_e + d_e, which is its first usable slot, so
    the two consecutive accessibility terms straddling that boundary bracket exactly the change
    the repair produced: delta_e = g_{tau_e - 1} - g_{tau_e} (positive when the repair helped,
    since a lower g is better). Both terms come from the trajectory that has already been
    evaluated, so attribution costs no extra UE solve.

    The improvement is not permanent. Demand that the damage had suppressed flows back at rate
    RHO, re-congesting the network and eating into the gain, so the credited total is the jump
    plus its geometrically decaying echo over the slots that remain:

        R_i = delta_e * sum_{j=0}^{T - tau_e} RHO^j = delta_e * (1 - RHO^{T - tau_e + 1})/(1 - RHO)

    Repairing early therefore earns more than repairing the same segment late, which is the
    ordering incentive the agent needs. When two segments complete in the same slot their jump
    is a single joint quantity; it is split in proportion to their normal-period (baseline UE)
    flow phi_e, which is the same edge-criticality score the flow baseline ranks by.

    `terms` is the length-T array of per-slot g values (terms[k-1] is slot k); returns rewards
    aligned with `perm`, and each segment's completion slot so the caller can order transitions
    by when their reward became observable."""
    tau = {e: int(start[e]) + int(durations[e]) for e in perm}
    by_slot = {}
    for e in perm:
        by_slot.setdefault(tau[e], []).append(e)

    R = {}
    for t_c, seg in by_slot.items():
        delta = float(terms[t_c - 2] - terms[t_c - 1])            # slot tau-1 vs slot tau
        retained = (1.0 - P.RHO ** (T - t_c + 1)) / (1.0 - P.RHO)  # jump plus its decaying echo
        w_sum = sum(phi[e] for e in seg)
        for e in seg:
            share = phi[e] / w_sum if w_sum > 0 else 1.0 / len(seg)
            R[e] = delta * share * retained
    return [R[e] for e in perm], [tau[e] for e in perm]


def _train_one_scenario(ctx, m, durations, segments, T, phi, budget, hp, seed):
    """Train one scenario's DQN from scratch and return (optima_row, trace_rows). Independent of
    every other scenario: fresh network, fresh replay, fresh caches, its own RNG stream (offset 2
    after GA's 0 and PSO's 1).

    STOPPING RULE. Training runs until the optimization has visibly levelled off, not until a
    preset amount of compute is spent. Three conditions must hold at once:

      (a) ep >= ep_min           -- eps has annealed to its floor, so exploration is no longer the
                                    reason nothing is changing;
      (b) the GREEDY order (an eps = 0 rollout, evaluated for its ORDER only, so it costs forward
          passes and no UE solve) has been identical for stable_K consecutive episodes -- the
          policy itself has settled;
      (c) no best-so-far improvement for patience_P consecutive episodes -- the search has too.

    Both plateau signals are required because either alone misreads this problem. Best-so-far
    improvements arrive in sparse bursts: on the n=10 run the longest gap BETWEEN two improvements
    reached 55 episodes while whole runs lasted 66-97, so patience alone would have cut two
    scenarios off before their final improvement. Greedy-order stability is the smoother signal --
    it is measurable every episode at no UE cost -- but on its own it would stop a policy that has
    merely stalled in a rut while exploration is still finding better schedules. Requiring both
    keeps training alive exactly while either the policy or the incumbent is still moving.

    `budget` optionally caps unique true evaluations (None = uncapped, the default). It exists for
    equal-budget experiments against GA/PSO; leaving it None is what lets the plateau rule, rather
    than an arbitrary ceiling, decide when the scenario is done. `ep_cap` is a pure safety net."""
    # The reward reads per-slot accessibility terms directly, so it tracks the reported objective
    # only while F is the accessibility term averaged over the whole horizon. Refuse loudly rather
    # than silently train against something else.
    if P.MU != 1.0 or P.F1_ACTIVE_ONLY:
        raise ValueError("RL rewards assume MU == 1 and F1_ACTIVE_ONLY == False; "
                         "adapt _completion_rewards before changing either")
    rng = np.random.RandomState(seed * 1000 + m * 10 + 2)
    st = _scenario_statics(ctx, segments, durations, phi)
    net = _MLP([_N_FEAT] + list(hp["hidden"]) + [1], rng)
    tgt = _MLP([_N_FEAT] + list(hp["hidden"]) + [1], rng)
    tgt.copy_weights_from(net)
    prior = hp["prior_scale"]

    memo = {}                    # perm -> dict(F, F1, F2, terms); unique evals = len(memo)
    slot_cache = {}              # per-slot g_t memo shared by every evaluation in this scenario
    replay = []                  # (x_i, R_i, X_next or None)
    best_perm, best_res = None, None
    trace, grad_steps, cum_ue = [], 0, 0

    t0 = time.perf_counter()
    # A dedicated RNG for the diagnostic greedy rollouts: _rollout draws once per decision even at
    # eps = 0, so sharing the training stream would shift every subsequent training draw.
    probe_rng = np.random.RandomState(seed * 1000 + m * 10 + 3)
    ep, stable, since_improve = 0, 0, 0
    last_greedy = None
    while ep < hp["ep_cap"] and (budget is None or len(memo) < budget):
        if (ep >= hp["ep_min"] and stable >= hp["stable_K"]
                and since_improve >= hp["patience_P"]):
            break                                   # policy settled AND search settled -> plateau
        # Episode 0 is forced greedy: under the zero-initialized residual it IS the flow order,
        # which seeds the memo (and best-by-true-F) with the flow baseline itself.
        eps = 0.0 if ep == 0 else max(hp["eps_min"], hp["eps0"] * hp["eps_decay"] ** (ep - 1))
        perm, start, feats, choices = _rollout(
            net, eps, rng, ctx, segments, durations, T, st, prior)

        since_improve += 1
        if perm in memo:
            res, new_eval = memo[perm], 0
        else:
            res, new_eval = _evaluate_prefix_cached(start, durations, T, ctx, slot_cache), 1
            memo[perm] = res
            cum_ue += res["n_ue"]
            if best_res is None or res["F"] < best_res["F"]:
                best_perm, best_res = perm, res
                since_improve = 0

        rewards, taus = _completion_rewards(perm, start, durations, res["terms"], T, phi)
        # Transitions enter the buffer in COMPLETION order, not decision order: a decision's
        # reward only becomes observable once its segment finishes, and completions do not follow
        # the order the decisions were made in. Ordering by tau keeps every buffer prefix a set of
        # transitions whose rewards are already revealed.
        for i in sorted(range(len(perm)), key=lambda j: (taus[j], j)):
            x_i = feats[i][choices[i]]
            X_next = feats[i + 1] if i + 1 < len(perm) else None
            replay.append((x_i, rewards[i], X_next))

        # DQN updates: minibatch Bellman regression on the replay buffer, gamma = 1, targets
        # from the periodically-synced target network.
        td_abs = []                              # |TD error| per minibatch, for the diagnostics
        for _ in range(hp["updates_per_ep"]):
            idx = rng.choice(len(replay), size=min(hp["batch"], len(replay)), replace=False)
            X_b = np.stack([replay[i][0] for i in idx])
            y = np.empty(len(idx))
            for j, i in enumerate(idx):
                _, r, X_next = replay[i]
                y[j] = r if X_next is None else r + float(np.max(_q_values(tgt, X_next, prior)))
            q, cache = net.forward(X_b)
            q = prior * X_b[:, 0] + q            # broadcast add: a NEW array, so `cache` is untouched
            td_abs.append(float(np.mean(np.abs(y - q))))    # read-only; the gradient below is unchanged
            gW, gb = net.backward(cache, 2.0 * (q - y) / len(idx))    # d/dq of mean squared error
            net.adam_step(gW, gb, hp["lr"])
            grad_steps += 1
            if grad_steps % hp["target_sync"] == 0:
                tgt.copy_weights_from(net)

        # Greedy probe AFTER this episode's updates: the deterministic order the current policy
        # would emit. Only the ORDER is compared, so no schedule is evaluated and no UE is solved.
        greedy_perm, _, _, _ = _rollout(net, 0.0, probe_rng, ctx, segments, durations, T, st, prior)
        stable = stable + 1 if greedy_perm == last_greedy else 0
        last_greedy = greedy_perm

        trace.append(dict(scenario=m, episode=ep, eps=round(eps, 4), F=res["F"],
                          best_F=best_res["F"], new_eval=new_eval, ue_new=res["n_ue"] if new_eval else 0,
                          cum_evals=len(memo), cum_ue=cum_ue,
                          ret=float(sum(rewards)),        # episode return the policy earned
                          td=float(np.mean(td_abs)),      # mean |Bellman residual| this episode
                          stable=stable,                  # consecutive episodes with an unchanged greedy order
                          since_improve=since_improve))   # episodes since the last best-so-far improvement
        ep += 1

    plateau = (ep >= hp["ep_min"] and stable >= hp["stable_K"]
               and since_improve >= hp["patience_P"])
    start = schedule_from_permutation(list(best_perm), durations)
    row = dict(scenario=m, F=best_res["F"], F1=best_res["F1"], F2=best_res["F2"],
               time_s=time.perf_counter() - t0, n_evals=len(memo), episodes=ep,
               ue_total=cum_ue,                     # UE solves actually run (prefix cache included)
               outcome="plateau" if plateau else ("budget_cap" if budget is not None
                                                  and len(memo) >= budget else "episode_cap"),
               order="-".join(map(str, best_perm)),
               durations="-".join(str(int(durations[e])) for e in segments))
    for e in segments:
        row[f"start_{e}"] = start[e]
    return row, trace


# --------------------------------------------------------------------------- #
# Parallel worker (module-level so Windows "spawn" can import it)
# --------------------------------------------------------------------------- #
_W = {}


def _rl_worker_init(toy_dir, disrupted, flow):
    """Pool-worker setup: build the fixed evaluation context once per process (one baseline UE
    solve, pinned to one core for bit-reproducibility) plus the flow scores; both are shared by
    every scenario this worker trains."""
    _W["ctx"] = build_context(toy_dir, disrupted, ue_cores=1)
    _W["phi"] = {int(eid): flow.get((min(u, v), max(u, v)), 0.0)
                 for (eid, u, v, s) in _W["ctx"]["disrupted"]}


def _rl_worker_train(task):
    """Train one scenario end-to-end in a worker; scenarios are independent, so this is the unit
    of parallelism (episodes within a scenario are inherently sequential -- each one depends on
    the network the previous episodes trained). Returns (m, row, trace); on failure (m, None,
    None) so the parent can finish the other scenarios and then fail LOUDLY listing m."""
    m, durations, segments, T, budget, hp, seed = task
    try:
        row, trace = _train_one_scenario(_W["ctx"], m, durations, segments, T, _W["phi"],
                                         budget, hp, seed)
        return m, row, trace
    except Exception:
        import traceback
        traceback.print_exc()
        return m, None, None


# --------------------------------------------------------------------------- #
# Run over M scenarios
# --------------------------------------------------------------------------- #
def run_rl(toy_dir=TOY, M=P.M_SCENARIOS, seed=P.SEED, budget=BUDGET_CAP, workers=None, hp=None):
    """Train the per-scenario DQNs and write outputs/greedy/n{N}/rl_optima.csv (schema shared
    with the other baselines; checkpointed per scenario, already-done scenarios are skipped on
    relaunch) plus outputs/rl/n{N}/rl_trace.csv and the diagnostic figures.

    Each scenario runs until its optimization levels off -- see the stopping rule in
    _train_one_scenario -- so compute is an OUTCOME of the run, not an input to it, and differs
    from scenario to scenario. `budget` caps unique true evaluations per scenario as a safety net
    only; it defaults to BUDGET_CAP, the same ceiling the GA/PSO searches use, and None removes it
    entirely. `workers` sizes the scenario-parallel pool (default: one worker per pending
    scenario)."""
    from multiprocessing import Pool

    hp = dict(RL_PARAMS, **(hp or {}))
    out_opt = scale_dir(OUT_OPTIMA)
    out_rl = scale_dir(OUT_RL)
    (out_rl / "figures").mkdir(parents=True, exist_ok=True)
    out_opt.mkdir(parents=True, exist_ok=True)

    disrupted = select_oracle_instance(toy_dir, P.N_DISRUPTED_ORACLE)
    segments = sorted(int(e) for e in disrupted["edge_id"])
    scenarios = sample_scenarios(disrupted, M, seed)
    T = compute_horizon(segments, scenarios)
    flow = _baseline_twoway_flow(toy_dir, cores=1)     # cores=1: the phi prior must be bit-stable run to run
    print(f"instance: {len(segments)} segments {segments}; M={M}; horizon T={T}; "
          f"budget={'open' if budget is None else str(budget) + ' evals'}; hp={hp}", flush=True)

    # Resume support, mirroring the metaheuristics: scenarios already in rl_optima.csv are kept
    # as-is and not retrained. `done` is keyed off the OPTIMA file alone, so resumed trace rows
    # are filtered down to it: a scenario about to be retrained must not keep its stale episode
    # rows (they would interleave with the new run's and corrupt the learning curve).
    opt_path, trace_path = out_opt / "rl_optima.csv", out_rl / "rl_trace.csv"
    rows = pd.read_csv(opt_path).to_dict("records") if opt_path.exists() else []
    trace_rows = pd.read_csv(trace_path).to_dict("records") if trace_path.exists() else []
    done = {int(r["scenario"]) for r in rows}
    trace_rows = [r for r in trace_rows if int(r["scenario"]) in done]
    pending = [m for m in range(M) if m not in done]
    if done:
        print(f"[resume] already done -> {sorted(done)}", flush=True)

    t_all = time.perf_counter()
    failed = []
    if pending:
        workers = min(len(pending), workers or len(pending))
        tasks = [(m, scenarios[m], segments, T, budget, hp, seed) for m in pending]
        with Pool(workers, initializer=_rl_worker_init,
                  initargs=(toy_dir, disrupted, flow)) as pool:
            for m, row, trace in pool.imap_unordered(_rl_worker_train, tasks):
                if row is None:
                    failed.append(m)
                    continue
                rows.append(row)
                trace_rows.extend(trace)
                # Trace first, optima second: a kill between the two writes then leaves a
                # trace-without-optima scenario, which the resume filter above simply retrains,
                # instead of an optima-without-trace scenario that would never regain its curve.
                pd.DataFrame(trace_rows).sort_values(["scenario", "episode"]).to_csv(
                    trace_path, index=False)
                pd.DataFrame(rows).sort_values("scenario").to_csv(opt_path, index=False)
                print(f"  scenario {row['scenario']}: F={row['F']:.4f}  "
                      f"({row['n_evals']}ev, {row['time_s']/60:.1f} min)", flush=True)

    # Fail LOUDLY on any dropped scenario: a silently missing row would shrink every method's
    # comparison downstream (util/compare.py inner-merges on scenario). The successes above are
    # already checkpointed, so a relaunch retrains only the failures.
    if failed:
        raise RuntimeError(f"RL training failed for scenarios {sorted(failed)}; "
                           f"successes are checkpointed in {opt_path} -- relaunch to retry the rest")
    if not rows:
        raise RuntimeError("RL produced no results (nothing to write)")

    opt = pd.DataFrame(rows).sort_values("scenario")
    print(f"  mean F [rl] = {opt['F'].mean():.4f}", flush=True)

    if trace_rows:                       # absent only when resuming with a deleted diagnostics tree
        from viz.rl_viz import (make_rl_learning, make_rl_objective, make_rl_reward,
                                make_rl_td_error)
        tr = pd.DataFrame(trace_rows)
        # The flow baseline is the level the flow-initialized agent starts from and cannot end
        # above; it is drawn as the reference in the per-scenario objective panels.
        flow_path = out_opt / "flow_optima.csv"
        flow_F = (pd.read_csv(flow_path).set_index("scenario")["F"].to_dict()
                  if flow_path.exists() else None)
        make_rl_td_error(out_rl, tr)              # learner view: did the value regression converge?
        make_rl_learning(out_rl, tr)              # cross-scenario: best-so-far F vs budget
        make_rl_objective(out_rl, tr, flow_F)     # per-scenario: sampled F, envelope, baseline
        make_rl_reward(out_rl, tr)                # policy view: per-episode return
    print(f"Wrote {opt_path} and {trace_path}  ({(time.perf_counter() - t_all)/60:.1f} min)",
          flush=True)
    return opt


if __name__ == "__main__":
    run_rl()
