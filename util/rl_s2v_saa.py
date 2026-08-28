"""EXPERIMENTAL (2026-08-24): the redesigned SAA solver -- the rl_s2v network trained on a LARGE
fixed pool of sampled duration worlds instead of the one nominal world, kept deliberately
removable. Built on the project owner's brief: training only in one fictional world is the
limitation; a training distribution that is diverse and close to the test distribution should do
better. Designed from scratch, independent of util.rl_rank's old rl_saa.

DESIGN, and the literature it stands on (the three signals are deliberately SEPARATED):

  1. LEARNING signal <- a large fixed LHS pool. `pool_n` worlds are drawn ONCE by probability-
     stratified Latin Hypercube (util.scenarios.saa_lhs_sample) from a dedicated stream; each
     episode rolls `batch_worlds` of them (sampled per episode) and their transitions mix in one
     replay buffer. Grounds: SAA theory (Kleywegt, Shapiro & Homem-de-Mello 2002 -- the fixed-
     sample optimum converges to the true optimum as the sample grows, and the sample-optimism
     bias shrinks with it; LHS reduces the sample's representation variance), and the deep-RL
     generalization line (Cobbe et al., ICML 2019/2020 -- CoinRun/Procgen: test performance
     rises monotonically with the number of training levels, small fixed sets overfit, large
     fixed sets approach full-distribution training). A large FIXED pool, rather than fresh
     worlds every episode, additionally keeps (world, order) evaluations memoizable -- the
     expensive resource here is the exact evaluator (T UE solves per new pair), not sampling.
  2. SELECTION signal <- a fixed, disjoint validation set (common-random-numbers principle:
     candidates are only comparable when measured on the SAME worlds -- the recorded project
     lesson that an order must not win by drawing an easy world). The plateau stop, the
     best-policy snapshot and the probe curves all read the `n_val` validation worlds; the M
     frozen evaluation scenarios stay untouched until delivery. This split -- sampled scenarios
     for gradients, fixed sets for validation -- is the field standard (Kool et al. ICLR 2019's
     SPCTSP; Nazari et al. NeurIPS 2018 stochastic VRP; AAAI 2025 stochastic FJSP NCO).
  3. HINGE exemplars <- PER-WORLD best trajectories. The fixed pool makes every world recur, so
     the solver keeps, for each pool world, the best trajectory seen IN THAT WORLD, and the
     large-margin hinge pulls Q toward these world-conditional exemplars (self-imitation
     learning, Oh et al. ICML 2018, in its per-environment form) -- a better fit for the
     per-scenario rolling delivery than the old rl_saa's mean-ranked buffer, and only possible
     BECAUSE the pool is fixed.

Everything else -- the S2V network (t_emb=4 coverage, readout MLP), the hinge configuration, PER,
Double DQN, dueling, the eps schedule -- is taken LIVE from util.rl_s2v.S2V_PARAMS, so the PLAIN
variants differ from rl_s2v in exactly one axis: the training worlds. The ADAPTIVE variants
(rl_s2v_saa{pool}_adaptive, hp adaptive=True) additionally switch on rl_s2v's deviation-24
observation channels -- parallel methods with their own folders and comparison columns, because
only pool-world training can teach those channels (nominal-world rl_s2v measured 0.9806 with
them vs 0.9774 without; see S2V_SAA_PARAMS' adaptive entry).

ISOLATION / REMOVAL. Self-contained trainer + runner; READ-ONLY imports. It DEPENDS ON
util.rl_s2v (network builder, graph tensors, rollout, self-check, params) -- deleting rl_s2v
breaks this module, so remove them together or promote the shared pieces first. To delete THIS
method alone:
  * delete this file;
  * remove the "rl_s2v_saa" dispatch in main.py (--solve mapping + error text + docstring list);
  * remove "rl_s2v_saa" from util/provenance.py SOLVER_DIR;
  * remove the ("rl_s2v_saa", "03-rl") entry and its kind branch in util/compare.py;
  * remove the rl_s2v_saa color/shape lines in viz/compare_viz.py;
  * delete outputs/03-rl/02-rl_s2v_saa/ and refresh the comparison.
NOT registered in util.sim_cache's canonical scan or util.seed_sweep until adopted.

Run:  python main.py --solve rl_s2v_saa      (or  python -m util.rl_s2v_saa)
"""
import time

import numpy as np
import pandas as pd

import config as P
from util.oracle import compute_horizon, scale_dir
from util.provenance import (solver_dir, fresh_scale_dir, log_dir, results_dir, slot_rows,
                             write_run_meta)
from util.rl import _completion_rewards, _evaluate_prefix_cached
from util.rl_rank import (EP_CAP, FIGURE_REFRESH_EVERY, OUT_DIAG, STOP_PARAMS, TOY,
                          RankPlateauStop, _nstep_rows, _Recorder, build_env)
from util.rl_s2v import (S2V_PARAMS, _build_s2v_net, _graph_tensors, _s2v_rollout, _self_check,
                         _state_x)
from util.scenarios import draw_durations, saa_lhs_sample

# --------------------------------------------------------------------------- #
# Hyperparameters: the FULL rl_s2v configuration taken live (one-axis-difference principle),
# plus the two knobs this design adds. n_val lives in util.rl_rank's shared STOP_PARAMS.
S2V_SAA_PARAMS = dict(
    S2V_PARAMS,
    pool_n=64,           # LHS training pool size. The design's central quantity: Procgen-style
                         # generalization rises with it, SAA optimism bias falls with it; 64 is
                         # 6.4x the old rl_saa's 10 and keeps per-run evaluation cost bounded
                         # through pool-recurrence memoization.
    batch_worlds=4,      # worlds rolled per episode (sampled from the pool). Decouples per-
                         # episode cost from pool size -- the pool can be large while an episode
                         # stays ~4x rl_s2v's single-world cost.
    # THE ADAPTIVE AXIS (2026-08-25, project owner's instruction). adaptive=True switches on
    # rl_s2v's deviation-24 observation channels (live traffic, OD disconnection, realized
    # shortfall) AND renames the variant rl_s2v_saa{pool}_adaptive with its own folders and
    # comparison columns -- a PARALLEL METHOD, never an overwrite. The observation flags are
    # DERIVED from this one knob in train_s2v_saa, so name and configuration cannot disagree.
    # adaptive=False is byte-identical to the pre-deviation-24 configuration (n_feat 10,
    # g_dim 4), which is what keeps the on-disk pool64/pool128 results valid without a rerun.
    # WHY the axis lives here and not in rl_s2v: the channels only carry signal where training
    # worlds have truth-vs-estimate daylight -- the nominal world has none by construction
    # (rl_s2v measured 0.9806 with them vs 0.9774 without, distinct orders 5 -> 34, i.e.
    # out-of-distribution noise at delivery), the pool worlds have it every episode.
    adaptive=False,
)

# The pool size is the design's central quantity, so each size is a SEPARATE METHOD rather than a
# setting one run overwrites: they get their own folders (outputs/03-rl/02-rl_s2v_saa/pool{n}/)
# and their own comparison columns, so the size-vs-quality trend is on disk instead of only in a
# chat log. Adding a size means adding it here and to util.provenance.SOLVER_DIR +
# util.compare.SEARCHED; the run itself needs no other change.
POOL_SIZES = (64, 128)


def variant_name(pool_n, adaptive=False):
    """The method name for a (pool size, adaptive) pair: rl_s2v_saa64, rl_s2v_saa128_adaptive,
    ... -- the single authority, used for the folder, the optima file, the figures and the
    comparison column."""
    return f"rl_s2v_saa{int(pool_n)}" + ("_adaptive" if adaptive else "")


def _merge_stop_saa(stop_params):
    """Shared defaults + caller. NO patience widening here: this solver probes every
    `probe_every` episodes (validation rollouts are not free), so STOP_PARAMS' probe-denominated
    patience already means patience_P * probe_every episodes."""
    sp = dict(STOP_PARAMS)
    sp.update(stop_params or {})
    return sp


# --------------------------------------------------------------------------- #
def train_s2v_saa(env, hp=None, seed=P.SEED, ep_cap=EP_CAP, rec_dir=None, stop_params=None,
                  verbose=True):
    """Train the pool-SAA S2V solver to a validation plateau; return the sibling delivery dict."""
    import torch
    import torch.nn as nn
    torch.set_num_threads(1)

    unknown = set(hp or {}) - set(S2V_SAA_PARAMS)
    if unknown:
        raise ValueError(f"unknown rl_s2v_saa hyperparameters {sorted(unknown)}; valid keys are "
                         f"{sorted(S2V_SAA_PARAMS)}")
    hp = dict(S2V_SAA_PARAMS, **(hp or {}))
    adaptive = bool(hp["adaptive"])
    if adaptive:
        # The observation flags are DERIVED from the adaptive knob, never set directly: one
        # switch names the variant and equips it, so the two cannot drift apart.
        hp = dict(hp, feat_obs_traffic=True, feat_obs_disc=True, feat_obs_trueD=True)
    sp = _merge_stop_saa(stop_params)
    torch.manual_seed(seed)                            # sibling seed derivations
    rng = np.random.RandomState(seed * 7 + 1)
    val_rng = np.random.RandomState(seed * 1000 + 15)  # validation worlds, disjoint stream
    pool_rng = np.random.RandomState(seed * 1000 + 21) # training pool, disjoint stream

    p, t_emb, use_g = int(hp["p"]), int(hp["t_emb"]), bool(hp["use_g"])
    readout_hidden = int(hp["readout_hidden"])
    n_step, gamma = int(hp["n_step"]), float(hp["gamma"])
    UPD, BATCH, sync = int(hp["updates_per_ep"]), int(hp["batch"]), int(hp["target_sync"])
    eps0, eps_min, anneal = float(hp["eps0"]), float(hp["eps_min"]), int(hp["eps_anneal"])
    cap = int(hp["replay_cap"])
    lam_cap, margin = float(hp["lam"]), float(hp["margin"])
    lam0, lam_growth = hp.get("lam0"), hp.get("lam_growth")
    use_hinge = lam_cap > 0.0
    ddqn = bool(hp["double_dqn"])
    prioritized = bool(hp["prioritized"])
    per_alpha, per_beta0 = float(hp["per_alpha"]), float(hp["per_beta0"])
    per_beta_eps, per_eps = float(hp["per_beta_eps"]), float(hp["per_eps"])
    dueling = bool(hp["dueling"])
    pool_n, K = int(hp["pool_n"]), int(hp["batch_worlds"])
    if K > pool_n:
        raise ValueError(f"batch_worlds ({K}) cannot exceed pool_n ({pool_n})")

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

    # Q plumbing, identical to util.rl_s2v.train_s2v (local closures there, so restated).
    def _q(model, state):
        x = torch.tensor(_state_x(gt, state))
        idx = [gt["idx"][e] for e in state["cand"]]     # accessible action set (see rl_s2v._q)
        out = model(x, A, deg, torch.tensor(state["g"]))
        if not dueling:
            return out[idx]
        V, adv = out
        a = adv[idx]
        return V + a - a.mean()

    def Qf(state):
        return _q(net, state)

    def Qt(state):
        return _q(tgt if sync > 0 else net, state)

    def greedy(rem, state):
        with torch.no_grad():
            return int(torch.argmax(Qf(state)))

    def _hinge(ex_states, ex_picks):
        ml = 0.0
        for st_i, a in zip(ex_states, ex_picks):
            Q = Qf(st_i)
            viol = torch.relu(margin - (Q[a] - Q)).clone()
            viol[a] = 0.0
            ml = ml + viol.sum()
        return ml / len(ex_states)

    _self_check(env, gt, Qf, torch)

    # THE TRAINING POOL: pool_n worlds, LHS-stratified, drawn once, then FIXED (design point 1).
    pool = saa_lhs_sample(env["dis"], pool_n, pool_rng)
    # THE VALIDATION WORLDS: fixed, disjoint stream, never trained on (design point 2).
    val_worlds = [draw_durations(env["dis"], val_rng) for _ in range(sp["n_val"])]
    # The training horizon covers exactly the worlds this run evaluates (pool + validation +
    # nominal): under the serial gated horizon, sizing for the distribution's worst case would
    # charge every evaluation for a pathological world the fixed sample never contains.
    Ttr = max(env["T"], compute_horizon(env["segs"], pool + val_worlds + [env["nominal"]]))
    env["T_train"] = Ttr                               # prints and UE accounting report the truth

    rec = None
    if rec_dir is not None:
        rec = _Recorder(rec_dir, variant_name(pool_n, adaptive), None, None, 0.0, torch,
                        flush_every=FIGURE_REFRESH_EVERY)

    stop = RankPlateauStop(sp["ep_min"], sp["patience_P"], sp["stable_K"], sp["tol"])
    memo, sc, replay, prio = {}, {}, [], []            # memo: (world key, perm) -> res
    exemplars = {}                                     # world idx -> (states, picks, F): per-world
    best_val, best_sd = float("inf"), None             # best (design point 3)
    t0 = time.perf_counter()
    ep = 0
    while ep < ep_cap and not stop.done():
        eps = max(eps_min, eps0 - (eps0 - eps_min) * ep / max(1, anneal))
        lam_ep = (0.0 if not use_hinge else
                  lam_cap if lam_growth is None else
                  min(lam_cap, float(lam0) * float(lam_growth) ** max(0, ep - 1)))

        def pk(rem, state):
            if rng.rand() < eps:
                return rng.randint(len(rem))
            with torch.no_grad():
                return int(torch.argmax(Qf(state)))

        # Episode: K pool worlds, one behavior rollout each; transitions mix in the replay.
        wis = rng.choice(pool_n, size=K, replace=False)
        fresh, F_ep = [], []
        for wi in wis:
            wi = int(wi)
            dur = pool[wi]
            perm, start, states, picks = _s2v_rollout(env, gt, pk, dur)
            key = (wi, perm)
            if key in memo:
                res = memo[key]
            else:
                res = _evaluate_prefix_cached(start, dur, Ttr, env["ctx"], sc)
                memo[key] = res
            rew, _ = _completion_rewards(list(perm), start, dur, res["terms"], Ttr, env["phi"])
            fresh.extend(_nstep_rows(states, picks, rew, n_step, gamma))
            F_ep.append(res["F"])
            ex = exemplars.get(wi)
            if ex is None or res["F"] < ex[2] - 1e-12:  # per-world best-so-far (self-imitation)
                exemplars[wi] = (states, picks, res["F"])
        replay.extend(fresh)
        if prioritized:
            prio.extend([max(prio, default=1.0)] * len(fresh))
        if len(replay) > cap:
            replay = replay[-cap:]
            if prioritized:
                prio = prio[-cap:]

        beta = (min(1.0, per_beta0 + (1.0 - per_beta0) * ep / per_beta_eps)
                if prioritized else None)
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
                    y.append(R)
                else:
                    with torch.no_grad():
                        if ddqn:
                            a_on = int(torch.argmax(Qf(bs)))
                            y.append(R + bd * float(Qt(bs)[a_on]))
                        else:
                            y.append(R + bd * float(Qt(bs).max()))
            yt = torch.tensor(y, dtype=torch.float32)
            td_elem = (q - yt) ** 2
            if prioritized:
                loss = (w_t * td_elem).mean()
                with torch.no_grad():
                    for j, i in enumerate(idx):
                        prio[i] = float(abs(q[j] - yt[j])) + per_eps
            else:
                loss = td_elem.mean()
            if use_hinge and exemplars:
                # One exemplar WORLD per update, drawn from the per-world best table -- the
                # world-conditional self-imitation this design exists for (design point 3).
                wi = list(exemplars)[rng.randint(len(exemplars))]
                ex_states, ex_picks, _ = exemplars[wi]
                loss = loss + lam_ep * _hinge(ex_states, ex_picks)
            with torch.no_grad():
                ep_q.append(float(q.mean()))
                ep_y.append(float(yt.mean()))
                ep_td.append(float((q - yt).abs().mean()))
            loss.backward()
            opt.step()
            ep_loss.append(float(loss.detach()))
            if sync > 0 and (_u + 1 + ep * UPD) % sync == 0:
                tgt.load_state_dict(net.state_dict())

        # Plateau probe on the VALIDATION worlds only (design point 2): every probe_every
        # episodes the greedy policy is rolled in each val world; the mean is the stop score and
        # drives the best-policy snapshot. The frozen-scenario diagnostic (what delivery would
        # score) is recorded alongside, and NEVER feeds the stop.
        probe = (ep % sp["probe_every"] == 0)
        gperm, F_val, scen_F = None, None, None
        if probe:
            gperm, _, _, _ = _s2v_rollout(env, gt, greedy)          # nominal order, stability signal
            vals = []
            for i, dv in enumerate(val_worlds):
                vp, vs, _, _ = _s2v_rollout(env, gt, greedy, dv)
                key = (f"val{i}", vp)
                if key in memo:
                    r_ = memo[key]
                else:
                    r_ = _evaluate_prefix_cached(vs, dv, Ttr, env["ctx"], sc)
                    memo[key] = r_
                vals.append(r_["F"])
            F_val = float(np.mean(vals))
            if F_val < best_val - 1e-12:
                best_val = F_val
                best_sd = {k2: v2.detach().clone() for k2, v2 in net.state_dict().items()}
            scen_F = float(np.mean([
                _evaluate_prefix_cached(_s2v_rollout(env, gt, greedy, d)[1], d, env["T"],
                                        env["ctx"], sc)["F"] for d in env["scen"]]))
        stop.update(ep, F_val if probe else None, tuple(gperm) if probe else None)

        if rec is not None:
            _m = lambda a: (float(np.mean(a)) if a else None)
            rec.episode(episode=ep, eps=float(eps), lam=float(lam_ep),
                        F=float(np.mean(F_ep)), best_F=(None if best_val == float("inf")
                                                        else float(best_val)),
                        F_val=F_val, scenF=scen_F,
                        greedy_order=("-".join(map(str, gperm)) if gperm else None),
                        loss=_m(ep_loss), q_pred_mean=_m(ep_q), y_target_mean=_m(ep_y),
                        td_abs_mean=_m(ep_td), since_improve=stop.since_improve,
                        stable=stop.stable)
        if verbose and ep % 25 == 0:
            print(f"  [{variant_name(pool_n, adaptive)}] ep {ep}  eps={eps:.3f}  lam={lam_ep:.3f}  "
                  f"F_val={('%.4f' % F_val) if F_val is not None else '-'}  "
                  f"best_val={best_val:.4f}  pool evals={len(memo)}  "
                  f"since_improve={stop.since_improve}", flush=True)
        ep += 1

    outcome = stop.reason or "episode_cap"
    if rec is not None:
        rec.finish(net)

    # Delivery: the FINAL policy rolled inside each frozen scenario (sibling semantics).
    per_scenario = []
    for d in env["scen"]:
        gp, gs, _, _ = _s2v_rollout(env, gt, greedy, d)
        per_scenario.append((list(gp), gs))
    order = list(_s2v_rollout(env, gt, greedy)[0])     # nominal-world summary only
    if verbose:
        print(f"  [{variant_name(pool_n, adaptive)}] stopped at ep {ep} ({outcome}) in "
              f"{(time.perf_counter() - t0) / 60:.1f} min", flush=True)
    return dict(order=order, per_scenario=per_scenario, net=net, best_net_sd=best_sd,
                episodes=ep, n_evals=len(memo), outcome=outcome, best_score=best_val, hp=hp,
                seed=seed, obs_solves=int(gt["obs"]["solves"][0]))


# --------------------------------------------------------------------------- #
def run_s2v_saa(toy_dir=TOY, N=None, M=P.M_SCENARIOS, seed=P.SEED, ep_cap=EP_CAP, hp=None,
                stop_params=None):
    """Train one pool-size variant to a validation plateau and write its canonical results to
    outputs/03-rl/{solver_dir(variant)}/n{N}/ (i.e. .../02-rl_s2v_saa/pool{n}/n{N}/), then
    refresh the comparison. The pool size names the variant, so sizes never overwrite each
    other."""
    import torch
    N = P.N_DISRUPTED_ORACLE if N is None else N
    merged = dict(S2V_SAA_PARAMS, **(hp or {}))
    pool_n, adaptive = int(merged["pool_n"]), bool(merged["adaptive"])
    v = variant_name(pool_n, adaptive)
    env = build_env(toy_dir, N=N, M=M)
    print(f"instance: {len(env['segs'])} segments {env['segs']}; M={M}; T={env['T']} "
          f"(T_train={env['T_train']}); seed={seed}; ep_cap={ep_cap}; variant={v}",
          flush=True)
    vdir = scale_dir(OUT_DIAG / solver_dir(v), N)
    vdir.mkdir(parents=True, exist_ok=True)
    fresh_scale_dir(vdir, subdirs=("log",), figures=True)   # stage 1: diagnostics only
    rdir = log_dir(vdir)
    t0 = time.perf_counter()

    r = train_s2v_saa(env, hp=hp, seed=seed, ep_cap=ep_cap, rec_dir=str(rdir),
                      stop_params=stop_params)
    rows, slots = [], []
    for m, (dur, (order_m, start)) in enumerate(zip(env["scen"], r["per_scenario"])):
        ts = time.perf_counter()
        res = _evaluate_prefix_cached(start, dur, env["T"], env["ctx"], {}, collect_traces=True)
        slots.extend(slot_rows(m, res))
        row = dict(scenario=m, F=res["F"], F1=res["F1"], F2=res["F2"],
                   time_s=time.perf_counter() - ts, n_evals=r["n_evals"],
                   episodes=r["episodes"], outcome=r["outcome"],
                   # n_evals counts distinct (world, order) evaluations (pool + validation), each
                   # costing ~T_train UE solves -- the honest input to the comparison's
                   # n_evals * T / M + T compute axis (T_train >= T slightly understates; the
                   # frozen-scenario diagnostic probes are deliberately not charged, matching
                   # every sibling). The deviation-24 live-traffic reads are real solves too and
                   # are charged the same amortised way (obs_solves = actual solver calls).
                   ue_total=((r["n_evals"] * env["T_train"] + r.get("obs_solves", 0))
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
    if r["best_net_sd"] is not None:                   # best-validation snapshot (drift diagnostic)
        torch.save(r["best_net_sd"], results_dir(vdir) / "model_best_val.pt")
    meanF = float(np.mean([x["F"] for x in rows]))
    write_run_meta(vdir, method=v, segments=env["segs"], T=env["T"], seed=seed, M=M,
                   hp=dict(r["hp"]), stop_params=_merge_stop_saa(stop_params), ep_cap=ep_cap,
                   episodes=r["episodes"], outcome=r["outcome"],
                   order_nominal_summary=r["order"], mean_F=meanF,
                   best_val_F=r["best_score"],
                   solver=("pool-SAA S2V-DQN: rl_s2v's network trained on a fixed LHS pool of "
                           "sampled worlds (util.rl_s2v_saa, EXPERIMENTAL)"),
                   delivery="per-scenario adaptive policy (final), observed history only; "
                            "best-validation weights saved as model_best_val.pt",
                   design_notes=[
                       "learning signal: fixed LHS pool of pool_n worlds, batch_worlds rolled "
                       "per episode (SAA convergence: Kleywegt/Shapiro/Homem-de-Mello 2002; "
                       "diversity-generalization: Cobbe et al. ICML 2019/2020)",
                       "selection signal: disjoint fixed validation worlds drive the plateau "
                       "stop and the best snapshot (common random numbers); frozen scenarios "
                       "recorded as diagnostic only",
                       "hinge exemplars: per-world best trajectories (self-imitation, Oh et al. "
                       "ICML 2018), enabled by pool recurrence",
                       "network/hinge/PER/ddqn/dueling config taken live from rl_s2v: the ONE "
                       "axis changed vs rl_s2v is the training worlds",
                       ("execution-time observation (rl_s2v deviation 24): ON -- adaptive "
                        "variant; the pool worlds carry true severities, so the channels see "
                        "truth-vs-estimate daylight during training (nominal-world rl_s2v "
                        "measured unable to use them: 0.9806 on vs 0.9774 off)"
                        if adaptive else
                        "execution-time observation (rl_s2v deviation 24): OFF -- plain "
                        "variant, byte-identical to the pre-deviation-24 configuration; the "
                        "_adaptive twin carries the channels")])
    try:
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
    run_s2v_saa()
