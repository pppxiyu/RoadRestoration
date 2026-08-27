"""Cross-scale ZERO-SHOT delivery of a trained RL policy (2026-08-26, project owner's
instruction): load the model a variant trained at instance size SRC, roll it greedily on the
frozen evaluation scenarios of instance size DST, and register the result as its own
comparison method named {variant}_from_n{SRC} -- the name carries the model's provenance, so a
transferred policy can never be mistaken for one trained at the scale it is scored on.

WHY THIS WORKS WITHOUT RETRAINING. The S2V carrier is the line graph over ALL 38 road segments
of the Sioux Falls network at EVERY instance size -- the adjacency, the feature widths and
therefore every parameter shape are size-invariant; an instance only changes WHICH vertices
carry damage (the static severity/duration/demand columns and the tags). A state dict trained
at n10 therefore loads into the n16 graph verbatim (asserted with strict loading). What the
transfer TESTS is whether the decision rule the policy distilled -- which field observations
and beliefs make a segment worth repairing next -- generalizes to a damage pattern with more
segments, heavier estimates and a longer horizon than anything it saw in training.

Delivery semantics are exactly the trainers': the FINAL policy (results/model_best.pt, the
delivered net) rolled per scenario with observed history only; the deviation-24 observation
channels are active iff the source variant trained with them (the adaptive family). Scoring,
provenance and the comparison refresh reuse the shared machinery, so a transfer row is
comparable line-for-line with the natively trained methods.

REMOVAL. Delete this file, the *_from_n* entries in util/provenance.SOLVER_DIR and
util/compare.SEARCHED, the _XFER colors in viz/compare_viz.py, and the "_from_n" dispatch
branch in main.py; on-disk results live under outputs/03-rl/05-transfer/.
"""
import re
import time

import numpy as np
import pandas as pd

import config as P
from util.oracle import scale_dir
from util.provenance import (fresh_scale_dir, log_dir, results_dir, slot_rows, solver_dir,
                             write_run_meta)
from util.rl import _evaluate_prefix_cached
from util.rl_rank import OUT_DIAG, TOY, build_env
from util.rl_s2v import S2V_PARAMS, _build_s2v_net, _graph_tensors, _s2v_rollout, _state_x


def _resolve_variant(variant):
    """The hyperparameters a variant name implies, resolved through the SAME authorities the
    trainers use (S2V_PARAMS / S2V_SAA_PARAMS + the adaptive-knob derivation), so a transfer
    can never rebuild the network under a different configuration than the checkpoint's."""
    if variant == "rl_s2v":
        return dict(S2V_PARAMS)
    m = re.fullmatch(r"rl_s2v_saa(\d+)(_adaptive)?", variant)
    if not m:
        raise SystemExit(f"unknown transfer source variant {variant!r}: expected rl_s2v or "
                         f"rl_s2v_saa<pool>[_adaptive]")
    from util.rl_s2v_saa import S2V_SAA_PARAMS
    hp = dict(S2V_SAA_PARAMS, pool_n=int(m.group(1)), adaptive=bool(m.group(2)))
    if hp["adaptive"]:
        # Same derivation as train_s2v_saa: the knob equips the observation channels.
        hp = dict(hp, feat_obs_traffic=True, feat_obs_disc=True, feat_obs_trueD=True)
    return hp


def run_transfer(variant, src_n, toy_dir=TOY, N=None, M=P.M_SCENARIOS):
    """Deliver the model `variant` trained at size src_n onto the size-N instance (default:
    config's N_DISRUPTED_ORACLE, i.e. main.py's --n override), writing a full provenance set
    under outputs/03-rl/{solver_dir(name)}/n{N}/ and refreshing that N's comparison."""
    import torch
    import torch.nn as nn
    torch.set_num_threads(1)
    N = P.N_DISRUPTED_ORACLE if N is None else int(N)
    name = f"{variant}_from_n{int(src_n)}"
    src_dir = scale_dir(OUT_DIAG / solver_dir(variant), int(src_n))
    src_model = results_dir(src_dir) / "model_best.pt"
    if not src_model.exists():
        raise SystemExit(f"no trained model at {src_model} -- train {variant} at n{src_n} first")
    if solver_dir(name) == name:
        raise SystemExit(f"{name} is not registered in util/provenance.SOLVER_DIR -- add it "
                         f"(and the util/compare.SEARCHED entry) before running the transfer")

    hp = _resolve_variant(variant)
    env = build_env(toy_dir, N=N, M=M)
    gt = _graph_tensors(env, hp=hp)
    p, t_emb, use_g = int(hp["p"]), int(hp["t_emb"]), bool(hp["use_g"])
    net = _build_s2v_net(p, t_emb, use_g, torch, nn, dueling=bool(hp["dueling"]),
                         readout_hidden=int(hp["readout_hidden"]), in_dim=gt["n_feat"],
                         hop_untied=bool(hp["hop_untied"]), g_dim=gt["g_dim"])
    sd = torch.load(src_model, weights_only=True)
    net.load_state_dict(sd, strict=True)      # size-invariant shapes: any mismatch is a bug
    net.eval()
    print(f"[{name}] loaded {src_model}  ->  delivering on n{N} "
          f"(T={env['T']}, M={len(env['scen'])})", flush=True)

    A = torch.tensor(gt["A"])
    deg = torch.tensor(gt["deg"], dtype=torch.float32)

    def greedy(rem, state):
        with torch.no_grad():
            x = torch.tensor(_state_x(gt, state))
            idx = [gt["idx"][e] for e in state["cand"]]
            out = net(x, A, deg, torch.tensor(state["g"]))
            if not bool(hp["dueling"]):
                return int(torch.argmax(out[idx]))
            V, adv = out
            a = adv[idx]
            return int(torch.argmax(V + a - a.mean()))

    vdir = scale_dir(OUT_DIAG / solver_dir(name), N)
    vdir.mkdir(parents=True, exist_ok=True)
    fresh_scale_dir(vdir, subdirs=("log",), figures=True)
    t0 = time.perf_counter()
    rows, slots = [], []
    for m, dur in enumerate(env["scen"]):
        ts = time.perf_counter()
        perm, start, _, _ = _s2v_rollout(env, gt, greedy, dur)
        res = _evaluate_prefix_cached(start, dur, env["T"], env["ctx"], {}, collect_traces=True)
        slots.extend(slot_rows(m, res))
        row = dict(scenario=m, F=res["F"], F1=res["F1"], F2=res["F2"],
                   time_s=time.perf_counter() - ts, n_evals=0, episodes=0,
                   outcome=f"zero-shot transfer of {variant} trained at n{src_n}",
                   # Delivery-only compute: this scenario's own evaluation plus the
                   # deviation-24 observation solves, amortised over the M scenarios exactly as
                   # training solves are elsewhere (n_evals=0: no search happened here).
                   ue_total=env["T"],
                   order="-".join(map(str, perm)),
                   durations="-".join(str(int(dur[e])) for e in env["segs"]))
        for e in env["segs"]:
            row[f"start_{e}"] = start[e]
        rows.append(row)
    obs_solves = int(gt["obs"]["solves"][0])
    for row in rows:
        row["ue_total"] += obs_solves / len(env["scen"])
    order = list(_s2v_rollout(env, gt, greedy)[0])      # nominal-world summary only
    for row in rows:
        row["policy_order_nominal"] = "-".join(map(str, order))

    fresh_scale_dir(vdir, subdirs=("results", "config"), figures=False)
    pd.DataFrame(rows).to_csv(results_dir(vdir) / f"{name}_optima.csv", index=False)
    pd.DataFrame(slots).to_csv(log_dir(vdir) / f"{name}_slots.csv", index=False)
    torch.save(sd, results_dir(vdir) / "model_best.pt")   # self-contained copy of the weights
    meanF = float(np.mean([r["F"] for r in rows]))
    write_run_meta(vdir, method=name, segments=env["segs"], T=env["T"], seed=P.SEED,
                   M=len(env["scen"]), hp=dict(hp),
                   mean_F=meanF, order_nominal_summary=order,
                   source_model=dict(variant=variant, trained_at_n=int(src_n),
                                     path=str(src_model)),
                   solver=(f"zero-shot cross-scale transfer: {variant} trained at n{src_n}, "
                           f"delivered on n{N} without retraining (util.transfer)"),
                   delivery="per-scenario adaptive policy (source checkpoint, final weights), "
                            "observed history only; weights copied alongside as model_best.pt",
                   obs_solves=obs_solves)
    print(f"[{name}] mean F = {meanF:.4f} on n{N}  "
          f"({(time.perf_counter() - t0) / 60:.1f} min, obs solves {obs_solves})  -> {vdir}",
          flush=True)
    from util.compare import refresh_comparison
    refresh_comparison(N)
    return {name: meanF}
