"""
Figures documenting the CURRENT problem setting, organized by world:

    python -m viz.problem_setting            # per-segment eta, the incoming setting (default)
    python -m viz.problem_setting --law repo # the scoring ruler (util.scenarios.sample_scenarios)

  01_nominal_setting    the NOMINAL scenario, everything a solver optimizing that one world
                        faces: the damaged network (map), what each severity level does to a
                        road's capacity and free-flow speed, the nominal repair time of each
                        damaged segment, and the travel demand each segment suppresses while
                        it stays broken. The damage physics and the demand-shortfall rule are
                        MODELING ASSUMPTIONS (config.py flags them as such); the network, BPR
                        parameters and OD demand are given by the open Sioux Falls data.
  02_scenario_setting   the same repair-time dimension ACROSS ALL M evaluation scenarios --
                        the only quantity the scenarios vary. Per segment, every realized
                        duration in the frozen evaluation sample, plus the total workload the
                        crews face per scenario against the scoring horizon. The damage map,
                        the severity physics and the demand rule are scenario-invariant, so
                        they appear only in 01.

THE DURATION LAW RENDERED BY DEFAULT. The default view draws per-segment INDEPENDENT
durations: segment e's realized repair time is max(1, round(E[d_e] * eta)) with its own
eta ~ U(ETA_LO, ETA_HI) per scenario, where E[d_e] is the segment's expected duration under
config.py's (road_class, severity) support sets. This is the research line's INCOMING problem
setting (adopted on the sandbox line, where the current RL configuration was selected); it is
NOT yet the ruler the committed solver results on disk are scored under, which is why its
parameters live here (ETA_LO/ETA_HI below) rather than in config.py. The scoring ruler remains
util.scenarios.sample_scenarios: level-shared durations (one draw per (road_class, severity)
level), sampled under config.EVAL_SAMPLING == "lhs" by probability-stratified Latin Hypercube
over the per-level marginals. Passing --law repo renders that ruler instead; the generated
figures and raw tables record which law they were built from, so the two can never be
silently confused.

Raw tables behind the figures (outputs/01-sim_val_n_problem_setting/raw/):
  problem_setting_segments.csv   one row per damaged segment: endpoints, class, severity,
                                 pre-disaster flow, expected + nominal duration, demand drop
  problem_setting_durations.csv  realized durations, scenario x segment
  problem_setting_meta.json      the law, its parameters, T, crews, damage physics
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

import config as P
from util.evaluate import build_context
from util.oracle import TOY, _baseline_twoway_flow, compute_horizon, select_oracle_instance
from util.scenarios import expected_durations, nominal_durations, sample_scenarios
from viz.style import C, save_pub, severity_color, use_pub

ROOT = Path(__file__).resolve().parent.parent
OUT_FIG = ROOT / "outputs" / "01-sim_val_n_problem_setting" / "03-problem_setting"
OUT_RAW = ROOT / "outputs" / "01-sim_val_n_problem_setting" / "raw"

# The per-segment duration law rendered by default (see the module docstring for its status).
ETA_LO, ETA_HI = 0.30, 1.70


def _draw_scenarios(law, dis, segs):
    """The M frozen evaluation scenarios under the requested law, at the project seed."""
    if law == "repo":
        return sample_scenarios(dis, P.M_SCENARIOS, P.SEED)
    exp = expected_durations(dis)
    rng = np.random.RandomState(P.SEED)
    return [{e: max(1, int(round(exp[e] * rng.uniform(ETA_LO, ETA_HI)))) for e in segs}
            for _ in range(P.M_SCENARIOS)]


# ----------------------------------------------------------------------------------------- #
# 01  the nominal scenario: network, damage physics, nominal repair time, demand drop
# ----------------------------------------------------------------------------------------- #
def _map_panel(a, dis, ctx, phi):
    """The damaged network. Line width carries pre-disaster flow, color the severity; the two
    severed (severity-3) edges are the two highest-flow links, so the damage hits both busy
    corridors and lighter roads whose repair can afford to wait."""
    nodes = pd.read_csv(Path(TOY) / "network" / "nodes.csv")
    pos = {int(r.node_id): (r.x, r.y) for r in nodes.itertuples(index=False)}
    sev_of = {int(r.edge_id): int(r.severity) for r in dis.itertuples(index=False)}
    fmax = max(phi.values())
    for r in ctx["edges"].itertuples(index=False):
        eid, u, v = int(r.edge_id), int(r.u), int(r.v)
        w = 1.2 + 7.0 * phi.get(eid, 0.0) / fmax
        if eid in sev_of:
            # A white casing under each damaged edge lifts even the palest severity color
            # off the grey background without changing the severity ramp itself.
            a.plot(*zip(pos[u], pos[v]), color="white", lw=w + 3.0, zorder=3,
                   solid_capstyle="round")
            a.plot(*zip(pos[u], pos[v]), color=severity_color(sev_of[eid]), lw=w, zorder=3,
                   solid_capstyle="round")
            xm, ym = (pos[u][0] + pos[v][0]) / 2, (pos[u][1] + pos[v][1]) / 2
            a.text(xm, ym, str(eid), fontsize=13, fontweight="bold", ha="center", va="center",
                   color=C["neutral_dark"], zorder=5,
                   bbox=dict(boxstyle="circle,pad=0.18", fc="white", ec=severity_color(
                       sev_of[eid]), lw=1.2, alpha=0.95))
        else:
            a.plot(*zip(pos[u], pos[v]), color=C["neutral_light"], lw=w, zorder=2,
                   solid_capstyle="round")
    xs, ys = zip(*pos.values())
    a.scatter(xs, ys, s=26, color="white", edgecolor=C["neutral_mid"], lw=0.8, zorder=4)
    a.set_aspect("equal")
    a.axis("off")
    a.legend(handles=[Line2D([], [], color=severity_color(s), lw=3.5,
                             label=f"severity {s}" + (" (severed)" if s >= P.SEVER_SEVERITY
                                                      else ""))
                      for s in (1, 2, 3)] +
                     [Line2D([], [], color=C["neutral_light"], lw=3.5, label="intact road")],
             loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=2, fontsize=15)


def _nominal_setting(dis, ctx, phi, nom, drops, path):
    """One figure, the whole nominal world: the map spans the tall left column; the right
    column stacks the damage physics, the nominal repair times, and the demand drops. The two
    per-segment panels share one segment order (ascending nominal duration) so a segment can
    be followed across them."""
    sev_of = {int(r.edge_id): int(r.severity) for r in dis.itertuples(index=False)}
    order = sorted(sev_of, key=lambda e: (nom[e], e))

    fig = plt.figure(figsize=(14.8, 12.0))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.12, 1.0], hspace=0.52, wspace=0.16)
    a = fig.add_subplot(gs[:, 0])
    _map_panel(a, dis, ctx, phi)

    # Right, top -- the damage physics, exactly the rule build_damaged_edges applies: at or
    # above SEVER_SEVERITY the road leaves the network entirely (nothing retained), below it
    # capacity is scaled by CAP_RETAIN and free-flow speed by SPEED_RETAIN.
    b = fig.add_subplot(gs[0, 1])
    sevs = [1, 2, 3]
    x = np.arange(len(sevs))
    capr = [0.0 if s >= P.SEVER_SEVERITY else P.CAP_RETAIN[s] for s in sevs]
    spdr = [0.0 if s >= P.SEVER_SEVERITY else P.SPEED_RETAIN[s] for s in sevs]
    b.bar(x - 0.19, capr, width=0.34, color=C["accent"], label="capacity retained")
    b.bar(x + 0.19, spdr, width=0.34, color=C["teal"], label="free-flow speed retained")
    b.set_xticks(x)
    b.set_xticklabels([f"severity {s}" + ("\n(severed: road removed)"
                                          if s >= P.SEVER_SEVERITY else "") for s in sevs],
                      fontsize=14)
    b.set_ylabel("fraction retained")
    b.set_ylim(0, 1.0)
    b.legend(fontsize=14)

    # Right, middle -- the nominal repair time of each segment: the single duration vector the
    # nominal world uses (expected duration rounded to whole slots).
    c = fig.add_subplot(gs[1, 1])
    c.bar(range(len(order)), [nom[e] for e in order],
          color=[severity_color(sev_of[e]) for e in order], edgecolor="white", lw=1.0)
    c.set_xticks(range(len(order)))
    c.set_xticklabels([str(e) for e in order], fontsize=14)
    c.set_xlabel("damaged segment")
    c.set_ylabel("nominal repair\nduration (slots)")

    # Right, bottom -- the demand drop each segment triggers while unrepaired: the column sum
    # of the shortfall matrix B, i.e. the per-slot demand suppressed across every OD pair
    # whose free-flow shortest path uses the segment. Same segment order as the middle panel.
    d = fig.add_subplot(gs[2, 1])
    d.bar(range(len(order)), [drops[e] for e in order],
          color=[severity_color(sev_of[e]) for e in order], edgecolor="white", lw=1.0)
    d.set_xticks(range(len(order)))
    d.set_xticklabels([str(e) for e in order], fontsize=14)
    d.set_xlabel("damaged segment")
    d.set_ylabel("demand drop\n(trips per slot)")

    save_pub(fig, path)
    plt.close(fig)


# ----------------------------------------------------------------------------------------- #
# 02  the same setting across ALL evaluation scenarios (durations are all that varies)
# ----------------------------------------------------------------------------------------- #
def _scenario_setting(dis, segs, scen, nom, T, law, path):
    """ONE panel: per segment, every scenario's realized duration as one dot, the nominal
    duration as a dark tick. The vertical spread IS the uncertainty the solvers are scored
    under; the segment order matches 01_nominal_setting. T is unused here but kept in the
    signature so render_problem_setting passes one set of arguments either way."""
    sev_of = {int(r.edge_id): int(r.severity) for r in dis.itertuples(index=False)}
    order = sorted(segs, key=lambda e: (nom[e], e))
    dur = np.array([[s[e] for e in order] for s in scen])          # scenario x segment
    rng = np.random.RandomState(0)                                 # jitter only, reproducible

    fig, a = plt.subplots(figsize=(13.0, 5.8))
    for i, e in enumerate(order):
        xj = i + rng.uniform(-0.18, 0.18, dur.shape[0])
        a.scatter(xj, dur[:, i], s=30, color=severity_color(sev_of[e]), alpha=0.55, lw=0,
                  zorder=2)
        a.hlines(nom[e], i - 0.30, i + 0.30, color=C["neutral_dark"], lw=2.6, zorder=3)
    a.set_xticks(range(len(order)))
    a.set_xticklabels([str(e) for e in order], fontsize=15)
    a.set_xlabel("damaged segment")
    a.set_ylabel("repair duration (slots)")
    a.legend(handles=[Line2D([], [], color=severity_color(s), marker="o", ls="",
                             label=f"severity {s}") for s in (1, 2, 3)] +
                     [Line2D([], [], color=C["neutral_dark"], lw=2.6, label="nominal duration")],
             fontsize=15, loc="upper left")
    fig.tight_layout()
    save_pub(fig, path)
    plt.close(fig)


# ----------------------------------------------------------------------------------------- #
def render_problem_setting(law="new"):
    """Render both problem-setting figures plus the raw tables behind them."""
    use_pub(slide=True)
    # Everything EXCEPT the panel titles goes bigger still; the titles keep the slide scale.
    plt.rcParams.update({"font.size": 17, "axes.labelsize": 20,
                         "xtick.labelsize": 17, "ytick.labelsize": 17,
                         "legend.fontsize": 16})
    dis = select_oracle_instance(TOY, P.N_DISRUPTED_ORACLE)
    segs = sorted(int(e) for e in dis["edge_id"])
    ctx = build_context(TOY, dis, ue_cores=1)
    flow = _baseline_twoway_flow(TOY, cores=1)
    end_of = {int(r.edge_id): (int(r.u), int(r.v)) for r in ctx["edges"].itertuples(index=False)}
    phi = {e: flow.get(tuple(sorted(end_of[e])), 0.0) for e in end_of}
    exp = expected_durations(dis)
    nom = nominal_durations(dis, segs)
    scen = _draw_scenarios(law, dis, segs)
    T = compute_horizon(segs, scen)
    bcol = {int(eid): j for j, (eid, *_) in enumerate(ctx["disrupted"])}
    # The ACTUAL per-slot demand drop a still-damaged segment causes: the evaluator's damage
    # vector carries the SEVERITY value (util.evaluate.evaluate_schedule builds v_vec from s,
    # not from a 0/1 indicator), so the standing shortfall target is severity * column-sum(B).
    sev_of = {int(r.edge_id): int(r.severity) for r in dis.itertuples(index=False)}
    drops = {e: sev_of[e] * float(ctx["B"][:, bcol[e]].sum()) for e in segs}

    OUT_FIG.mkdir(parents=True, exist_ok=True)
    OUT_RAW.mkdir(parents=True, exist_ok=True)
    _nominal_setting(dis, ctx, phi, nom, drops, OUT_FIG / "01_nominal_setting")
    _scenario_setting(dis, segs, scen, nom, T, law, OUT_FIG / "02_scenario_setting")

    # The raw tables behind the figures, so every plotted number has a data row.
    seg_rows = []
    for r in dis.itertuples(index=False):
        e = int(r.edge_id)
        seg_rows.append(dict(edge_id=e, u=int(r.u), v=int(r.v), road_class=str(r.road_class),
                             severity=int(r.severity), pre_disaster_flow=phi[e],
                             expected_duration=exp[e], nominal_duration=int(nom[e]),
                             demand_drop_per_unit_severity=drops[e] / int(r.severity),
                             demand_drop_per_slot=drops[e]))
    pd.DataFrame(seg_rows).to_csv(OUT_RAW / "problem_setting_segments.csv", index=False)
    pd.DataFrame([{**{"scenario": m}, **{f"seg_{e}": s[e] for e in segs}}
                  for m, s in enumerate(scen)]).to_csv(
        OUT_RAW / "problem_setting_durations.csv", index=False)
    meta = dict(law=("per-segment independent eta" if law == "new"
                     else "level-shared eta (the current scoring ruler)"),
                eta=({"lo": ETA_LO, "hi": ETA_HI} if law == "new" else {"support": P.ETA}),
                n_disrupted=P.N_DISRUPTED_ORACLE, M=P.M_SCENARIOS, seed=P.SEED,
                horizon_T=int(T), crews=P.C_MAX, rho=P.RHO, kappa=P.KAPPA,
                cap_retain=P.CAP_RETAIN, speed_retain=P.SPEED_RETAIN,
                sever_severity=P.SEVER_SEVERITY)
    (OUT_RAW / "problem_setting_meta.json").write_text(json.dumps(meta, indent=1),
                                                       encoding="utf-8")
    print(f"problem-setting figures written to {OUT_FIG}  (law: {meta['law']}, T={T})")
    return OUT_FIG


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--law", choices=("new", "repo"), default="new",
                    help="duration law to render: 'new' = per-segment independent eta "
                         "(the incoming setting, default), 'repo' = the current scoring ruler "
                         "(util.scenarios.sample_scenarios, level-shared eta)")
    render_problem_setting(ap.parse_args().law)
