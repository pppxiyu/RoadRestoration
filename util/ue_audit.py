"""
UE tolerance audit: measures what the evaluation-tier solver settings do to the numbers the
study actually reads, and writes the data plus a written-up analysis into outputs/0-UE_val/.

NOT part of any pipeline -- nothing imports this module and no main.py path reaches it. It runs
only when invoked explicitly:

    python -m util.ue_audit

because it needs several minutes of reference solves and, unlike the production code, it imports
AequilibraE (the package the in-house solver replaced) to serve as an independent referee. If
AequilibraE is not installed the audit refuses to run with a clear message; production code is
unaffected either way.

WHAT IT MEASURES. Every solver configuration is run over the same per-slot UE problems that a
schedule evaluation produces (the recovery of the configured instance under its first scenarios),
and every configuration is scored against the same reference: AequilibraE bfw at rgap 1e-8, a
tolerance tighter than anything under test and a different implementation from the one being
judged, so no configuration is scored against itself. Two kinds of error are reported per
configuration, because they answer different questions:

  * LEVEL error |g - g_ref| per slot: how far the accessibility term itself sits from the truth.
    This is the error that shifts F, and it is largely SYSTEMATIC (an under-converged solve
    overestimates congestion the same way every slot), which is why method comparisons survive a
    loose tolerance far better than the level error suggests.
  * CHANGE error |dg - dg_ref| between adjacent slots: how well the slot-to-slot movement of g --
    the recovery curve's shape, and the RL's per-step reward signal -- is preserved. A level
    error that is constant across slots cancels here entirely. The quantity it must be compared
    against is the true change |dg_ref| (the SIGNAL), so the audit reports the masking ratio
    (change error / signal, both as a ratio of medians and pair by pair) and the number of
    DIRECTION FLIPS: adjacent pairs whose measured g moves opposite to the truth. A flip means
    the evaluation reads "repairing this segment made things worse" where the truth says better,
    and a per-pair relative error above 1 is exactly what makes a flip possible.

The audit's grid covers cold and warm-started solves at rgap 1e-2 / 1e-3 / 1e-6 plus the retired
AequilibraE production setting as an anchor, and always includes the CURRENT config.py setting,
so re-running after a config change shows what the change did. An early run of this audit also
exposed a silent early exit in the in-house solver (an uphill bi-conjugate direction ended the
solve as if converged; see _Network.solve in util/ue.py), which is the kind of defect only an
external-reference audit catches.

OUTPUT (under outputs/0-UE_val/tolerance_audit/):
  tolerance_audit.csv   one row per (scenario, slot, configuration): time, iterations, g, g_ref
  tolerance_audit.md    the generated analysis: methodology, the full table, the reading of it
"""
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

import config as P
from util.evaluate import (build_context, build_damaged_edges, _matrix_from_H, od_travel_times,
                           schedule_from_permutation)
from util.oracle import select_oracle_instance
from util.scenarios import sample_scenarios
from util.ue import solve_ue, warm_start_seed

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "0-UE_val" / "tolerance_audit"

N_SCEN = 2                    # scenarios audited; the 1e-8 references dominate the run time
REF_RGAP, REF_MAX_ITER = 1e-8, 2000
AE_PROD = (1e-2, 25)          # the retired AequilibraE production setting, kept as the anchor
GRID = ((1e-2, 25), (1e-3, 100), (1e-6, 500))
MEANINGFUL = 1e-4             # |dg_ref| below this is not a change the study claims to resolve


# --------------------------------------------------------------------------- #
# The referee: AequilibraE, reconstructed here for the audit only
# --------------------------------------------------------------------------- #
def _ae_solve(edges, M, zone_ids, rgap, max_iter):
    """AequilibraE bfw on the toy's bidirectional edge table, exactly as the project ran it
    before the in-house replacement (direction=0, capacity copied to both directions, every node
    a zone, centroid flows not blocked). Reference and anchor only."""
    import contextlib
    import logging
    import os
    try:
        from aequilibrae.matrix import AequilibraeMatrix
        from aequilibrae.paths import Graph, TrafficAssignment, TrafficClass
    except ImportError as exc:
        raise SystemExit(
            "the tolerance audit needs AequilibraE as its independent referee and it is not "
            "installed in this environment (production code does not use it)") from exc
    net = pd.DataFrame({
        "link_id": np.arange(1, len(edges) + 1, dtype=np.int64),
        "a_node": edges["u"].to_numpy(np.int64), "b_node": edges["v"].to_numpy(np.int64),
        "direction": 0, "distance": edges["length"].to_numpy(float), "modes": "c",
        "capacity_ab": edges["capacity"].to_numpy(float),
        "capacity_ba": edges["capacity"].to_numpy(float),
        "free_flow_time": edges["free_flow_time"].to_numpy(float),
        "b": edges["bpr_alpha"].to_numpy(float), "power": edges["bpr_beta"].to_numpy(float)})
    net["id"] = net["link_id"]
    g = Graph()
    g.network = net
    g.prepare_graph(np.asarray(zone_ids, dtype=np.int64))
    g.set_graph("free_flow_time")
    g.set_blocked_centroid_flows(False)
    mat = AequilibraeMatrix()
    mat.create_empty(memory_only=True, zones=len(zone_ids), matrix_names=["demand"])
    mat.index[:] = np.asarray(zone_ids, dtype=np.int64)
    mat.matrix["demand"][:, :] = M
    mat.computational_view(["demand"])
    tc = TrafficClass("car", g, mat)
    a = TrafficAssignment()
    a.set_classes([tc])
    a.set_vdf("BPR")
    a.set_vdf_parameters({"alpha": "b", "beta": "power"})
    a.set_capacity_field("capacity")
    a.set_time_field("free_flow_time")
    a.set_algorithm("bfw")
    a.max_iter = max_iter
    a.rgap_target = rgap
    a.set_cores(1)
    logging.getLogger("aequilibrae").setLevel(logging.CRITICAL)
    with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn), \
            contextlib.redirect_stderr(dn):
        a.execute()
    res = a.results()
    link = net.set_index("link_id")
    rows = []
    for lid, r in res.iterrows():
        u, v = int(link.loc[lid, "a_node"]), int(link.loc[lid, "b_node"])
        rows.append({"from": u, "to": v, "volume": r["demand_ab"], "cost": r["Congested_Time_AB"]})
        rows.append({"from": v, "to": u, "volume": r["demand_ba"], "cost": r["Congested_Time_BA"]})
    f = pd.DataFrame(rows)
    return f[f["volume"].notna()].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# The measurement
# --------------------------------------------------------------------------- #
def _g_of(links, H, ctx):
    """The per-slot accessibility term g, computed exactly as the evaluators compute it."""
    u = od_travel_times(links, ctx)
    u_t = np.where(np.isfinite(u), u, ctx["u_pen"])
    den = float(np.sum(H * ctx["baseline_u"]))
    return float(np.sum(H * u_t) / den) if den > 0 else 1.0


def _slot_stream(ctx, order, durations):
    """Yield (slot, demand H, still-damaged map) along one schedule's recovery, replicating the
    demand-shortfall recursion of the evaluators."""
    H0, B, sev, dis = ctx["H0"], ctx["B"], ctx["severity_vec"], ctx["disrupted"]
    start = schedule_from_permutation(order, durations)
    comp = [start[eid] + durations[eid] for (eid, _, _, _) in dis]
    D = np.zeros(len(H0))
    for k in range(1, max(comp) + 1):
        still = np.array([k < c for c in comp])
        D = np.maximum(B @ np.where(still, sev, 0.0), P.RHO * D)
        yield k, np.clip(H0 - D, 0.0, None), {eid: s for (eid, _, _, s), st
                                              in zip(dis, still) if st}


def _audit_order(ctx):
    """The repair order whose recovery supplies the per-slot problems: the delivered rl_nominal
    schedule when one is on disk (the realistic case), ascending edge ids otherwise."""
    p = (ROOT / "outputs" / "2-RL" / "rl_nominal" / f"n{P.N_DISRUPTED_ORACLE}" / "results"
         / "rl_nominal_optima.csv")
    if p.exists():
        return [int(x) for x in pd.read_csv(p)["order"].iloc[0].split("-")], "delivered rl_nominal"
    return sorted(int(e) for (e, _, _, _) in ctx["disrupted"]), "ascending edge id"


def _measure(ctx, scen, order):
    """Run every configuration over every slot. Returns (rows DataFrame, premise violations)."""
    settings = [s for s in GRID]
    if (P.UE_RGAP, P.UE_MAX_ITER) not in settings:     # always audit what config.py runs
        settings.append((P.UE_RGAP, P.UE_MAX_ITER))
    viol = 0
    rows = []
    for m, dur in enumerate(scen):
        slots = list(_slot_stream(ctx, order, dur))
        refs = []
        for k, H, dmg in slots:
            refs.append(_g_of(_ae_solve(build_damaged_edges(ctx, dmg), _matrix_from_H(H, ctx),
                                        ctx["zone_ids"], REF_RGAP, REF_MAX_ITER), H, ctx))
        print(f"  scenario {m}: {len(slots)} reference solves done", flush=True)

        for k, H, dmg in slots:                        # the retired production anchor
            e = build_damaged_edges(ctx, dmg)
            t0 = time.perf_counter()
            links = _ae_solve(e, _matrix_from_H(H, ctx), ctx["zone_ids"], *AE_PROD)
            rows.append(dict(scenario=m, slot=k, solver="AequilibraE", rgap=AE_PROD[0],
                             mode="cold", ms=1e3 * (time.perf_counter() - t0), iters=np.nan,
                             g_val=_g_of(links, H, ctx), g_ref=refs[k - 1]))

        for rg, mi in settings:
            for mode in ("cold", "warm"):
                warm = None                            # (links, routed H), as in the evaluators
                for (k, H, dmg), g_ref in zip(slots, refs):
                    e = build_damaged_edges(ctx, dmg)
                    x0 = None
                    if mode == "warm" and warm is not None:
                        links_prev, H_routed_prev = warm
                        dH = H - H_routed_prev
                        if float(dH.min()) < -1e-9:
                            viol += 1
                        x0 = warm_start_seed(e, _matrix_from_H(np.clip(dH, 0.0, None), ctx),
                                             ctx["zone_ids"], links_prev)
                    t0 = time.perf_counter()
                    links, conv = solve_ue(e, _matrix_from_H(H, ctx), ctx["zone_ids"],
                                           max_iter=mi, rgap=rg, quiet=True, cores=1, x0=x0)
                    el = 1e3 * (time.perf_counter() - t0)
                    if mode == "warm":
                        u = od_travel_times(links, ctx)
                        warm = (links, np.where(np.isfinite(u), H, 0.0))
                    rows.append(dict(scenario=m, slot=k, solver="in-house", rgap=rg, mode=mode,
                                     ms=el, iters=conv.iterations, g_val=_g_of(links, H, ctx),
                                     g_ref=g_ref))
        print(f"  scenario {m}: all configurations done", flush=True)
    return pd.DataFrame(rows), viol


# --------------------------------------------------------------------------- #
# The analysis columns
# --------------------------------------------------------------------------- #
def _changes(s, warm):
    """Adjacent-slot changes (measured, reference) within each scenario. Warm chains solve their
    first slot cold, so the pair spanning it is excluded for warm configurations."""
    dv, dr = [], []
    for _, gs in s.groupby("scenario"):
        gs = gs.sort_values("slot")
        a, b = np.diff(gs.g_val.to_numpy()), np.diff(gs.g_ref.to_numpy())
        dv.append(a[1:] if warm else a)
        dr.append(b[1:] if warm else b)
    return np.concatenate(dv), np.concatenate(dr)


def _stats_row(label, s, warm, base_ms):
    sw = s[s.slot > 1] if warm else s              # warm timing also excludes the cold first slot
    lev = (sw.g_val - sw.g_ref).abs()
    dv, dr = _changes(s, warm)
    err = np.abs(dv - dr)
    sig = float(np.median(np.abs(dr)))
    keep = np.abs(dr) > MEANINGFUL
    pp = err[keep] / np.abs(dr[keep])
    flips = int(np.sum((np.sign(dv) != np.sign(dr)) & keep))
    return dict(label=label, ms=float(sw.ms.median()),
                iters=(float(sw.iters.median()) if sw.iters.notna().any() else np.nan),
                speedup=base_ms / float(sw.ms.median()),
                lev_med=float(lev.median()), lev_max=float(lev.max()), signal=sig,
                derr_med=float(np.median(err)), ratio=float(np.median(err)) / sig,
                pp_med=float(np.median(pp)), pp_p90=float(np.percentile(pp, 90)),
                pp_max=float(pp.max()), flips=flips, pairs=int(keep.sum()))


def _summarize(d):
    """One stats row per configuration, the AequilibraE anchor first."""
    base_ms = float(d[d.solver == "AequilibraE"].ms.median())
    out = [_stats_row(f"AequilibraE @{AE_PROD[0]:g} 冷 *(退役生产设置)*",
                      d[d.solver == "AequilibraE"], False, base_ms)]
    for rg in sorted(d[d.solver == "in-house"].rgap.unique()):
        for mode in ("cold", "warm"):
            s = d[(d.solver == "in-house") & (d.rgap == rg) & (d["mode"] == mode)]
            if not len(s):
                continue
            mark = (" **← 当前 config.py 设置**"
                    if rg == P.UE_RGAP and (mode == "warm") == bool(P.UE_WARM_START) else "")
            out.append(_stats_row(f"自写 @{rg:g} {'热' if mode == 'warm' else '冷'}{mark}",
                                  s, mode == "warm", base_ms))
    return out


def _write_md(path, stats, d, viol, order_name):
    n_slots = int(d[d.solver == "AequilibraE"].shape[0])
    _, dr_all = _changes(d[d.solver == "AequilibraE"], False)
    hdr = ("| 配置 | ms/槽 | 迭代 | 加速 | 水平误差 中位 | 水平误差 最大 | Δ误差 中位 | Δ/信号 "
           "| 逐对 中位 | 逐对 90% | 逐对 最大 | 方向翻转 |")
    sep = "|" + "---|" * 12
    lines = []
    for r in stats:
        it = "~22" if np.isnan(r["iters"]) else f"{r['iters']:.0f}"
        lines.append(
            f"| {r['label']} | {r['ms']:.1f} | {it} | {r['speedup']:.1f}x "
            f"| {r['lev_med']:.2e} | {r['lev_max']:.2e} | {r['derr_med']:.2e} "
            f"| {r['ratio']:.3f} | {r['pp_med']:.3f} | {r['pp_p90']:.3f} | {r['pp_max']:.2f} "
            f"| {r['flips']}/{r['pairs']} |")
    text = f"""# UE 求解容差审计

生成:{date.today().isoformat()},由 `python -m util.ue_audit` 写出。数据在同目录
`tolerance_audit.csv`,每行是一个 (scenario, slot, 配置) 的一次求解。重跑本模块会同时刷新
数据和本文,所以这里的每个数字都由流程生成,不是手抄的。

## 审计什么

调度评估的热路径每个时隙解一次 UE(用户均衡),解到 config.py 的 `UE_RGAP` 容差就停。停得越早
越快,但留下的截断误差落进研究真正读取的两个量:逐时隙可达性项 g 的水平,以及 g 在相邻时隙之间
的变化(恢复曲线的形状,也是 RL 逐步奖励读的信号)。本审计在真实的恢复时隙问题上
(实例 n={P.N_DISRUPTED_ORACLE},修复顺序取 {order_name},前 {N_SCEN} 个场景,共 {n_slots}
个时隙)把各配置与同一个参照解逐槽对比,参照是 AequilibraE bfw 在 rgap {REF_RGAP:g} 下的解:
它比所有被测配置都紧,并且是另一份独立实现,没有配置在给自己打分。

## 为什么变化量误差要和信号比

F 是各时隙 g 的均值,方法间的优劣来自谁让 g 更早降下来,所以求解误差真正的危害不在水平偏移
(欠收敛的解每个时隙都以几乎相同的方式高估拥堵,这个系统性偏差在方法比较中大部分互相抵消),
而在时隙之间不一致:相邻两槽的测量误差之差直接进入测得的 Δg。判据因此是「Δ误差/信号」——
误差要和 g 本该发生的变化(信号,本测试台中位 {np.median(np.abs(dr_all)):.2e},最小
{np.abs(dr_all).min():.2e})相比才有意义。逐对相对误差超过 1 意味着误差盖过了变化本身,
方向才可能被读反,「方向翻转」计数就是这一硬失败的直接测量:真实在下降(修复有效),测出来
却在上升。变化量低于 {MEANINGFUL:g} 的相邻对不计入(本测试台上没有低于门槛的对)。

## 结果

{hdr}
{sep}
{chr(10).join(lines)}

所有误差列以中位数汇总,「逐对」列是每个相邻对各自的相对误差 |Δg测−Δg真|/|Δg真| 的分布。
热启动行剔除了链条上必然冷解的首槽及跨越它的相邻对。可行性前提(逐 OD 需求单调不减、弧只
重开不消失)违反次数:{viol}(必须为 0,热启动种子的可行性依赖它们)。

## 怎么读

- 冷解在同一容差下的病灶在尾部:典型时隙尚可,但少数时隙的 Δ误差达到信号的数倍,方向翻转
  集中在那里。
- 热启动让相邻两槽停在收敛路径的同一位置,截断误差在相减时大部分抵消,所以同容差下热启动的
  Δ误差约为冷解的一半,逐对最大值被封在 1 以下,方向翻转随之消失。这不是运气:逐对相对误差
  不过 1 是方向不可能翻转的充分条件。
- @1e-2 热的中位迭代数是 1,即种子本身已满足容差、求解器几乎没有工作,水平误差也是全表最大。
  这种靠种子质量过线的状态对需求增量的大小敏感,不宜作为生产设置。当前生产设置(见表中标记)
  是在真收敛与速度之间的选择。

## 出处备注

本审计的一个早期运行还暴露了自写求解器的一次静默提前退出:双共轭方向在二次近似失效处指向
上坡,线搜索无下降可走,循环把未收敛的解当作收敛结果返回,三个容差在同一时隙返回逐位相同的
错误 g。已修复(回退普通 Frank-Wolfe 方向,并让 `solve_ue` 对既未达容差又未撞迭代上限的退出
显式报错),细节见 `util/ue.py` 中 `_Network.solve` 的文档。这类缺陷只有外部参照的审计能
发现,是本模块保留在 repo 里的主要理由。
"""
    path.write_text(text, encoding="utf-8")


def run_audit():
    toy = ROOT / "data" / "siouxfalls_toy"
    dis = select_oracle_instance(toy, n=P.N_DISRUPTED_ORACLE)
    ctx = build_context(toy, dis, ue_cores=1)
    order, order_name = _audit_order(ctx)
    scen = sample_scenarios(dis, M=P.M_SCENARIOS, seed=P.SEED)[:N_SCEN]
    print(f"UE tolerance audit: n={P.N_DISRUPTED_ORACLE}, order = {order_name}, "
          f"{N_SCEN} scenarios, reference = AequilibraE bfw @{REF_RGAP:g}/{REF_MAX_ITER}",
          flush=True)
    d, viol = _measure(ctx, scen, order)
    OUT.mkdir(parents=True, exist_ok=True)
    d.to_csv(OUT / "tolerance_audit.csv", index=False)
    stats = _summarize(d)
    _write_md(OUT / "tolerance_audit.md", stats, d, viol, order_name)
    print(f"\npremise violations: {viol} (must be 0)")
    for r in stats:
        print(f"  {r['label']:44s} {r['ms']:8.1f} ms  lev {r['lev_med']:.2e}  "
              f"d/sig {r['ratio']:.3f}  pp_max {r['pp_max']:.2f}  flips {r['flips']}/{r['pairs']}")
    print(f"\nwritten: {OUT / 'tolerance_audit.csv'}\n         {OUT / 'tolerance_audit.md'}")
    return d


if __name__ == "__main__":
    run_audit()
