"""
Figures documenting the CURRENT problem setting:

    python -m viz.problem_setting

  01a_damaged_network   line width = ROAD CLASS, color = severity (changed from flow width
                        2026-08-26 on the project owner's instruction).
                        the NOMINAL scenario as FOUR standalone figures (split 2026-08-25 on
  01b_damage_physics    the project owner's instruction; was one 4-panel 01_nominal_setting).
  01c_nominal_durations 01a is the damaged network map with the crew depot and initially
  01d_demand_drops      unreachable interior; 01b what each severity does to a road's capacity
                        and free-flow speed; 01c the nominal repair time of each damaged
                        segment; 01d the travel demand each segment suppresses while broken.
                        The damage physics and the demand-shortfall rule are MODELING
                        ASSUMPTIONS (config.py flags them as such); the network, BPR parameters
                        and OD demand are given by the open Sioux Falls data.
  02_scenario_setting   the same repair-time dimension ACROSS ALL M evaluation scenarios --
                        the only quantity the scenarios vary. Per segment, every realized
                        duration in the frozen evaluation sample, plus the nominal duration.
  03_duration_cells     THE DURATION LAW itself, as a ridge plot: the exact
                        rounded-truncated-lognormal PMF of each of the 9 (road_class, severity)
                        cells (config.DUR_MEAN / DUR_SD) stacked on ONE shared duration axis,
                        ordered by cell mean, so the designed overlap across cells and the
                        equal-mean/different-sd pairs are read directly. Hue = road class,
                        shade = severity, shared with 02.

  04_severity_law       THE SEVERITY LAW: what a reported severity ESTIMATE implies about the
                        true severity (config.SEVERITY_CONFUSION). The truth is drawn per
                        scenario; its LABEL is never an input, though since 2026-08-25 the
                        field state it produces is observable during execution (rl_s2v
                        deviation 24). What each true severity does to the road
                        (capacity/speed retention, severing) lives in config.py's damage block
                        and the run_meta -- its panel was removed 2026-08-25 on the project
                        owner's instruction.

THE DURATION LAW is config.py's per-segment truncated-lognormal law, rendered through
util.scenarios (sample_scenarios for the frozen sample, _cell_pmf for the exact per-cell
distributions) -- the SAME code path every solver is scored under, so these figures cannot
drift from the ruler. The 2026-08-24 redefinition (per-segment independence, the lognormal
cells, the crew-accessibility constraint -- a scheduling-model constraint, with the damage
instance NOT designed around it) and the second redefinition of the same day (severity is a
reported ESTIMATE, the truth drawn per scenario and not observed by any solver) are both
recorded in technical_notes/05-problem_redefinition.md. Everywhere below, a segment's severity
is its ESTIMATE -- the truth is a property of a scenario, not of the instance. Whether the truth
is observable during execution is a live design question as of 2026-08-25, not a fixed rule, so
these figures state what the CURRENT setting does and the run_meta field
`severity.revealed_during_execution` carries the same fact in machine-readable form.

Raw tables behind the figures (outputs/01-sim_val_n_problem_setting/raw/):
  problem_setting_segments.csv   one row per damaged segment: endpoints, class, severity,
                                 pre-disaster flow, expected + nominal duration, demand drop
  problem_setting_durations.csv  realized durations, scenario x segment
  problem_setting_meta.json      the law, its parameters, T, crews, damage physics, access rule
  problem_setting_duration_cells.csv  the exact per-cell PMF behind the ridge plot
"""
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
from util.scenarios import (_cell_pmf, expected_durations, nominal_durations,
                            sample_scenarios)
from viz.style import C, save_pub, severity_color, use_pub

ROOT = Path(__file__).resolve().parent.parent
# Scale-agnostic BASE; each render writes into its own n{N} subfolder (N read off the instance
# actually drawn, never off config) so future scales -- n13, n16, ... -- sit side by side and a
# rerun of one scale cannot overwrite another's figures.
OUT_FIG = ROOT / "outputs" / "01-sim_val_n_problem_setting" / "03-problem_setting"
OUT_RAW = ROOT / "outputs" / "01-sim_val_n_problem_setting" / "raw"

# Color semantics shared by 02_scenario_setting and 03_duration_cells: HUE names the road
# class, SHADE the severity (deeper = heavier damage). The map panel (01) keeps the project's
# severity ramp -- there the class is legible from the network itself.
_CLASS_HUE = {"local": "teal", "major": "purple", "highway": "accent"}


def _cell_color(road_class, severity):
    """The (class, severity) cell's color: the class hue blended toward white, less white the
    heavier the damage."""
    import matplotlib.colors as mc
    base = np.array(mc.to_rgb(C[_CLASS_HUE[road_class]]))
    f = {1: 0.40, 2: 0.68, 3: 1.0}[int(severity)]
    return tuple(1.0 - f * (1.0 - base))


# ----------------------------------------------------------------------------------------- #
# 01  the nominal scenario: network, damage physics, nominal repair time, demand drop
# ----------------------------------------------------------------------------------------- #
#: Line width per road class on the network map. Width carries the CLASS (a fixed property of
#: the road: capacity, free-flow speed, BPR parameters) rather than pre-disaster flow, which is
#: an equilibrium OUTCOME and only loosely tracks class here -- measured over the 38 segments the
#: class rank correlates 0.508 with flow and the ranges overlap heavily (highway 9.0k-46.3k,
#: major 12.5k-36.8k, local 10.5k-25.0k; the quietest segment in the network is a highway).
_CLASS_LW = {"local": 1.6, "major": 3.6, "highway": 6.4}


def _map_panel(a, dis, ctx, phi):
    """The damaged network. Line width carries the ROAD CLASS, color the severity; the two
    severed (severity-3) edges are the two highest-flow links, so the damage hits both busy
    corridors and lighter roads whose repair can afford to wait. The depot triangle marks where
    crews enter the network (the accessibility constraint's reference point). `phi` is still
    taken so the panel keeps one signature with its callers, and remains available if a future
    version wants to encode flow again."""
    nodes = pd.read_csv(Path(TOY) / "network" / "nodes.csv")
    pos = {int(r.node_id): (r.x, r.y) for r in nodes.itertuples(index=False)}
    sev_of = {int(r.edge_id): int(r.severity) for r in dis.itertuples(index=False)}
    for r in ctx["edges"].itertuples(index=False):
        eid, u, v = int(r.edge_id), int(r.u), int(r.v)
        w = _CLASS_LW[str(r.road_class)]
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
    # The crew depot: repairs can only start on segments reachable from here through passable
    # (undamaged or already-repaired) roads -- the accessibility constraint of the 2026-08-24
    # problem redesign, part of the setting and so part of this map.
    dx, dy = pos[P.ACCESS_DEPOT]
    a.scatter([dx], [dy], s=210, marker="^", color=C["accent"], edgecolor="white", lw=1.4,
              zorder=6)
    a.set_aspect("equal")
    a.axis("off")
    a.legend(handles=[Line2D([], [], color=severity_color(s), lw=3.5,
                             label=f"severity {s}" + (" (severed)" if s >= P.SEVER_SEVERITY
                                                      else ""))
                      for s in (1, 2, 3)] +
                     [Line2D([], [], color=C["neutral_light"], lw=3.5, label="intact road"),
                      Line2D([], [], color=C["accent"], marker="^", ls="", ms=12,
                             markeredgecolor="white",
                             label=f"crew depot (node {P.ACCESS_DEPOT})")],
             loc="lower left", frameon=False, fontsize=18)
    # BOTH legends sit INSIDE the axes. Anchoring one below the axes (the previous
    # bbox_to_anchor=(0.5, -0.01)) does not survive fig.tight_layout(): the axes is expanded to
    # fill the figure and the legend lands off-canvas at negative y, where the tight bounding box
    # does not recover it. The map's left half is empty, so both fit without covering the network.
    # A second legend also needs the first re-attached as a plain artist, since one axes holds
    # only one `legend_`.
    a.add_artist(a.get_legend())
    a.legend(handles=[Line2D([], [], color=C["neutral_mid"], lw=_CLASS_LW[rc], label=rc)
                      for rc in ("highway", "major", "local")],
             loc="upper left", frameon=False, fontsize=16,
             title="road class", title_fontsize=16)


def _nominal_map(dis, ctx, phi, path):
    """The damaged network on its own (was the left column of the old 01_nominal_setting)."""
    fig, a = plt.subplots(figsize=(9.6, 8.4))
    _map_panel(a, dis, ctx, phi)
    fig.tight_layout()
    save_pub(fig, path)
    plt.close(fig)


def _nominal_damage_physics(dis, path):
    """The damage physics on its own: exactly the rule build_damaged_edges applies -- at or
    above SEVER_SEVERITY the road leaves the network entirely (nothing retained), below it
    capacity is scaled by CAP_RETAIN and free-flow speed by SPEED_RETAIN."""
    fig, b = plt.subplots(figsize=(9.6, 5.4))
    sevs = [1, 2, 3]
    x = np.arange(len(sevs))
    capr = [0.0 if s >= P.SEVER_SEVERITY else P.CAP_RETAIN[s] for s in sevs]
    spdr = [0.0 if s >= P.SEVER_SEVERITY else P.SPEED_RETAIN[s] for s in sevs]
    b.bar(x - 0.19, capr, width=0.34, color=C["accent"], label="capacity retained")
    b.bar(x + 0.19, spdr, width=0.34, color=C["teal"], label="free-flow speed retained")
    b.set_xticks(x)
    b.set_xticklabels([f"severity {s}" + ("\n(severed: road removed)"
                                          if s >= P.SEVER_SEVERITY else "") for s in sevs],
                      fontsize=19)
    b.tick_params(axis="y", labelsize=19)
    b.set_ylabel("fraction retained", fontsize=21)
    b.set_ylim(0, 1.0)
    b.legend(fontsize=18)
    fig.tight_layout()
    save_pub(fig, path)
    plt.close(fig)


def _nominal_durations(dis, nom, path):
    """The nominal repair time of each damaged segment on its own: the single duration vector
    the nominal world uses (expected duration rounded to whole slots), segments in ascending
    nominal duration (shared with the demand-drop figure so a segment tracks across them)."""
    sev_of = {int(r.edge_id): int(r.severity) for r in dis.itertuples(index=False)}
    order = sorted(sev_of, key=lambda e: (nom[e], e))
    fig, c = plt.subplots(figsize=(9.6, 5.4))
    c.bar(range(len(order)), [nom[e] for e in order],
          color=[severity_color(sev_of[e]) for e in order], edgecolor="white", lw=1.0)
    c.set_xticks(range(len(order)))
    c.set_xticklabels([str(e) for e in order], fontsize=18)
    c.tick_params(axis="y", labelsize=19)
    c.set_xlabel("damaged segment", fontsize=21)
    c.set_ylabel("nominal repair\nduration (slots)", fontsize=21)
    c.legend(handles=[Line2D([], [], color=severity_color(s), lw=8,
                             label=f"severity {s}" + (" (severed)" if s >= P.SEVER_SEVERITY
                                                      else "")) for s in (1, 2, 3)],
             fontsize=18, loc="upper left")
    fig.tight_layout()
    save_pub(fig, path)
    plt.close(fig)


def _nominal_demand_drops(dis, drops, nom, path):
    """The demand drop each segment triggers while unrepaired, on its own: the column sum of
    the shortfall matrix B -- the per-slot demand suppressed across every OD pair whose
    free-flow shortest path uses the segment. Same segment order as the nominal-duration
    figure."""
    sev_of = {int(r.edge_id): int(r.severity) for r in dis.itertuples(index=False)}
    order = sorted(sev_of, key=lambda e: (nom[e], e))
    fig, d = plt.subplots(figsize=(9.6, 5.4))
    d.bar(range(len(order)), [drops[e] for e in order],
          color=[severity_color(sev_of[e]) for e in order], edgecolor="white", lw=1.0)
    d.set_xticks(range(len(order)))
    d.set_xticklabels([str(e) for e in order], fontsize=18)
    d.tick_params(axis="y", labelsize=19)
    d.set_xlabel("damaged segment", fontsize=21)
    d.set_ylabel("demand drop\n(trips per slot)", fontsize=21)
    d.legend(handles=[Line2D([], [], color=severity_color(s), lw=8,
                             label=f"severity {s}" + (" (severed)" if s >= P.SEVER_SEVERITY
                                                      else "")) for s in (1, 2, 3)],
             fontsize=18, loc="upper left")
    fig.tight_layout()
    save_pub(fig, path)
    plt.close(fig)


# ----------------------------------------------------------------------------------------- #
# 02  the same setting across ALL evaluation scenarios (durations are all that varies)
# ----------------------------------------------------------------------------------------- #
def _scenario_setting(dis, segs, scen, nom, T, path):
    """ONE panel: per segment, every scenario's realized duration as one dot, the nominal
    duration as a dark tick. The vertical spread IS the uncertainty the solvers are scored
    under; the segment order matches 01_nominal_setting. Dot color follows the shared cell
    semantics: hue = road class, shade = severity (deeper = heavier). T is unused here but kept
    in the signature so render_problem_setting passes one set of arguments either way."""
    cell_of = {int(r.edge_id): (str(r.road_class), int(r.severity))
               for r in dis.itertuples(index=False)}
    order = sorted(segs, key=lambda e: (nom[e], e))
    dur = np.array([[s[e] for e in order] for s in scen])          # scenario x segment
    rng = np.random.RandomState(0)                                 # jitter only, reproducible

    fig, a = plt.subplots(figsize=(13.0, 5.8))
    for i, e in enumerate(order):
        xj = i + rng.uniform(-0.18, 0.18, dur.shape[0])
        a.scatter(xj, dur[:, i], s=30, color=_cell_color(*cell_of[e]), alpha=0.75, lw=0,
                  zorder=2)
        a.hlines(nom[e], i - 0.30, i + 0.30, color=C["neutral_dark"], lw=2.6, zorder=3)
    a.set_xticks(range(len(order)))
    a.set_xticklabels([str(e) for e in order], fontsize=15)
    a.set_xlabel("damaged segment")
    a.set_ylabel("repair duration (slots)")
    combos = sorted({cell_of[e] for e in segs},
                    key=lambda cs: (("local", "major", "highway").index(cs[0]), cs[1]))
    a.legend(handles=[Line2D([], [], color=_cell_color(c, sv), marker="o", ls="", ms=8,
                             label=f"{c}, severity {sv}") for c, sv in combos] +
                     [Line2D([], [], color=C["neutral_dark"], lw=2.6, label="nominal duration")],
             fontsize=13, loc="upper left", ncol=2, framealpha=0.9)
    fig.tight_layout()
    save_pub(fig, path)
    plt.close(fig)


# ----------------------------------------------------------------------------------------- #
# 03  the duration law itself: the 9 cells' exact PMFs on one shared axis
# ----------------------------------------------------------------------------------------- #
def _duration_cells(dis, path, raw_csv=None):
    """The generative law as a RIDGE PLOT: all 9 (road_class, severity) cells on ONE shared
    duration axis, ordered by cell mean top-to-bottom, each row the cell's exact PMF
    (util.scenarios._cell_pmf, the same function every sampler inverts) drawn as a filled
    profile. Overlap IS the point of the figure -- neighbouring rows share the axis, so the
    designed cross-cell overlap and the equal-mean pairs ((local,2)~(major,1),
    (local,3)~(highway,1), (major,3)~(highway,2), adjacent rows with the same mean line) are
    read directly. Hue = road class, shade = severity, matching 02_scenario_setting. Row
    heights are peak-normalized (shape and support carry the comparison); the exact
    probabilities behind every profile go to `raw_csv`. The dark line marks each cell's exact
    mean; row labels count this instance's damaged segments in the cell."""
    classes = ("local", "major", "highway")
    n_of = {}
    for r in dis.itertuples(index=False):
        n_of[(r.road_class, int(r.severity))] = n_of.get((r.road_class, int(r.severity)), 0) + 1
    cells = [(c, sv) for c in classes for sv in (1, 2, 3)]
    mean_of = {cs: sum(k * pr for k, pr in _cell_pmf(cs)) for cs in cells}
    order = sorted(cells, key=lambda cs: (mean_of[cs], classes.index(cs[0])))
    kmax = max(max(k for k, _ in _cell_pmf(cs)) for cs in cells)

    if raw_csv is not None:
        pd.DataFrame([dict(road_class=c, severity=sv, duration_slots=k, probability=pr)
                      for (c, sv) in order for k, pr in _cell_pmf((c, sv))]).to_csv(
            raw_csv, index=False)

    fig, a = plt.subplots(figsize=(14.2, 8.8))
    h = 1.65                                       # peak height in row units: rows interleave
    for i, (c, sv) in enumerate(order):
        y0 = -i
        pmf = dict(_cell_pmf((c, sv)))
        xs = np.arange(0, kmax + 2)
        ys = np.array([pmf.get(int(k), 0.0) for k in xs])
        ys = ys / ys.max() * h
        col = _cell_color(c, sv)
        a.fill_between(xs, y0, y0 + ys, step="mid", color=col, alpha=0.88, lw=0,
                       zorder=2 * i + 2)
        a.step(xs, y0 + ys, where="mid", color="white", lw=1.6, zorder=2 * i + 3)
        m = mean_of[(c, sv)]
        a.plot([m, m], [y0, y0 + 0.55], color=C["neutral_dark"], lw=2.0, zorder=2 * i + 4)
    a.set_yticks([-i for i in range(len(order))])
    a.set_yticklabels([f"{c} road, severity {sv}  ({n_of.get((c, sv), 0)} damaged)"
                       for c, sv in order], fontsize=20)
    a.set_ylim(-len(order) + 0.6, h + 0.35)
    a.set_xlim(0, kmax + 1)
    a.tick_params(axis="x", labelsize=20)
    a.set_xlabel("repair duration (slots)", fontsize=22)
    a.legend(handles=[Line2D([], [], color=C["neutral_dark"], lw=2.0,
                             label="cell mean (exact)")],
             loc="upper right", fontsize=19, frameon=False)
    fig.tight_layout()
    save_pub(fig, path)
    plt.close(fig)


# ----------------------------------------------------------------------------------------- #
# 04  the severity law: estimate -> true severity, and what a true severity does
# ----------------------------------------------------------------------------------------- #
def _severity_law(dis, path):
    """ONE panel: each reported severity ESTIMATE as a group of probabilities over the TRUE
    severity (config.SEVERITY_CONFUSION), the tick labels carrying how many of this instance's
    segments are exposed to that estimate. (The former second panel -- capacity/speed retained
    per true severity -- was removed on the project owner's instruction, 2026-08-25; those
    constants remain on record in config.py's damage block and in the run_meta.)"""
    n_est = {}
    for r in dis.itertuples(index=False):
        n_est[int(r.severity)] = n_est.get(int(r.severity), 0) + 1
    fig, a = plt.subplots(figsize=(9.6, 4.8))
    x = np.arange(3)
    w = 0.26
    for j, s_true in enumerate((1, 2, 3)):
        a.bar(x + (j - 1) * w, [P.SEVERITY_CONFUSION[s_hat][j] for s_hat in (1, 2, 3)],
              width=w, color=severity_color(s_true), edgecolor="white", lw=1.0,
              label=f"true severity {s_true}")
    a.set_xticks(x)
    a.set_xticklabels([f"estimate {s}\n({n_est.get(s, 0)} segments)" for s in (1, 2, 3)],
                      fontsize=16)
    a.set_ylabel("P(true severity | estimate)")
    a.set_ylim(0, 0.88)
    a.legend(fontsize=14, ncol=3, loc="upper center")
    fig.tight_layout()
    save_pub(fig, path)
    plt.close(fig)


# ----------------------------------------------------------------------------------------- #
def render_problem_setting(n=None):
    """Render the problem-setting figures plus the raw tables behind them, for instance size
    `n` (default: config's N_DISRUPTED_ORACLE, overridden at CALL time so the config value is
    never edited). All sizes share ONE deterministic generator and ONE law
    (util.oracle.select_oracle_instance; same duration cells, same severity confusion, same
    demand model), but since 2026-08-26 the generator has two recipes: n < 16 keeps the
    original rule (two highest-flow segments severed, the rest spread down the flow ranking at
    alternating severity 2/1), while n >= 16 amplifies what the scenario draw can change on the
    project owner's instruction -- non-critical picks stay in the upper half of the flow
    ranking and estimates skew 2:1 toward severity 2, the confusion matrix's
    maximum-uncertainty row. Output lands in 03-problem_setting/n{n}/ beside the other
    sizes."""
    use_pub(slide=True)
    # Everything EXCEPT the panel titles goes bigger still; the titles keep the slide scale.
    plt.rcParams.update({"font.size": 17, "axes.labelsize": 20,
                         "xtick.labelsize": 17, "ytick.labelsize": 17,
                         "legend.fontsize": 16})
    n = P.N_DISRUPTED_ORACLE if n is None else int(n)
    dis = select_oracle_instance(TOY, n)
    segs = sorted(int(e) for e in dis["edge_id"])
    ctx = build_context(TOY, dis, ue_cores=1)
    flow = _baseline_twoway_flow(TOY, cores=1)
    end_of = {int(r.edge_id): (int(r.u), int(r.v)) for r in ctx["edges"].itertuples(index=False)}
    phi = {e: flow.get(tuple(sorted(end_of[e])), 0.0) for e in end_of}
    exp = expected_durations(dis)
    nom = nominal_durations(dis, segs)
    scen = sample_scenarios(dis, P.M_SCENARIOS, P.SEED)     # THE frozen ruler, no viz-local law
    T = compute_horizon(segs, scen)
    bcol = {int(eid): j for j, (eid, *_) in enumerate(ctx["disrupted"])}
    # The ACTUAL per-slot demand drop a still-damaged segment causes: the evaluator's damage
    # vector carries the SEVERITY value (util.evaluate.evaluate_schedule builds v_vec from s,
    # not from a 0/1 indicator), so the standing shortfall target is severity * column-sum(B).
    sev_of = {int(r.edge_id): int(r.severity) for r in dis.itertuples(index=False)}
    drops = {e: sev_of[e] * float(ctx["B"][:, bcol[e]].sum()) for e in segs}

    # n is read OFF THE INSTANCE (len of the drawn damaged set), the provenance convention:
    # a config override can never mislabel the folder.
    fig_dir = OUT_FIG / f"n{len(segs)}"
    fig_dir.mkdir(parents=True, exist_ok=True)
    OUT_RAW.mkdir(parents=True, exist_ok=True)
    _nominal_map(dis, ctx, phi, fig_dir / "01a_damaged_network")
    _nominal_damage_physics(dis, fig_dir / "01b_damage_physics")
    _nominal_durations(dis, nom, fig_dir / "01c_nominal_durations")
    _nominal_demand_drops(dis, drops, nom, fig_dir / "01d_demand_drops")
    _scenario_setting(dis, segs, scen, nom, T, fig_dir / "02_scenario_setting")
    _severity_law(dis, fig_dir / "04_severity_law")
    _duration_cells(dis, fig_dir / "03_duration_cells",
                    raw_csv=OUT_RAW / f"problem_setting_duration_cells_n{len(segs)}.csv")

    # The raw tables behind the figures, so every plotted number has a data row.
    seg_rows = []
    for r in dis.itertuples(index=False):
        e = int(r.edge_id)
        seg_rows.append(dict(edge_id=e, u=int(r.u), v=int(r.v), road_class=str(r.road_class),
                             severity=int(r.severity), pre_disaster_flow=phi[e],
                             expected_duration=exp[e], nominal_duration=int(nom[e]),
                             demand_drop_per_unit_severity=drops[e] / int(r.severity),
                             demand_drop_per_slot=drops[e]))
    pd.DataFrame(seg_rows).to_csv(OUT_RAW / f"problem_setting_segments_n{len(segs)}.csv", index=False)
    pd.DataFrame([{**{"scenario": m}, **{f"seg_{e}": s[e] for e in segs}}
                  for m, s in enumerate(scen)]).to_csv(
        OUT_RAW / f"problem_setting_durations_n{len(segs)}.csv", index=False)
    meta = dict(
        law="per-segment truncated lognormal over (road_class, severity) cells",
        dur_mean={f"{c}-{sv}": P.DUR_MEAN[(c, sv)] for (c, sv) in P.DUR_MEAN},
        dur_sd={f"{c}-{sv}": P.DUR_SD[(c, sv)] for (c, sv) in P.DUR_SD},
        dur_trunc_mult=P.DUR_TRUNC_MULT,
        severity=dict(reported="the instance's severity column is a rapid-assessment ESTIMATE",
                      confusion={str(k): v for k, v in P.SEVERITY_CONFUSION.items()},
                      # LABEL revelation: still never happens -- no solver is handed a true
                      # severity or an unstarted segment's duration.
                      revealed_during_execution=False,
                      # What IS observable since 2026-08-25 (the observation ban was lifted):
                      # the realized network's field state during execution -- live traffic,
                      # OD-level disconnection, realized demand shortfall. Currently exploited
                      # by the rl_s2v_saa _adaptive variants only (util/rl_s2v.py deviation 24;
                      # plain rl_s2v measured unable to learn the channels, and plain
                      # rl_s2v_saa keeps them off so the pair isolates the channel).
                      field_state_observable="traffic state at slots <= now (2026-08-25)",
                      note="true severity is drawn per scenario; it sets capacity/speed "
                           "retention, severing, the demand shortfall and the duration cell"),
        instance=dict(selection="flow-ranked: two highest-baseline-flow segments severed at "
                                "severity 3, the rest spread over lower-flow segments at "
                                "alternating severity 2/1", segments=sorted(segs)),
        access=dict(depot=P.ACCESS_DEPOT,
                    rule="a segment is repairable only while an endpoint is reachable from the "
                         "depot through passable (undamaged or completed) edges; damaged and "
                         "under-repair segments block crew passage"),
        n_disrupted=n, M=P.M_SCENARIOS, seed=P.SEED, eval_sampling=P.EVAL_SAMPLING,
        horizon_T=int(T), crews=P.C_MAX, rho=P.RHO, kappa=P.KAPPA,
        cap_retain=P.CAP_RETAIN, speed_retain=P.SPEED_RETAIN,
        sever_severity=P.SEVER_SEVERITY)
    (OUT_RAW / f"problem_setting_meta_n{len(segs)}.json").write_text(json.dumps(meta, indent=1),
                                                       encoding="utf-8")
    print(f"problem-setting figures written to {fig_dir}  (law: {meta['law']}, T={T})")
    return fig_dir


if __name__ == "__main__":
    render_problem_setting()
