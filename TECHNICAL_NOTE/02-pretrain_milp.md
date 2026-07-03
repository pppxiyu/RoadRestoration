# 02 — `run_pretrain_milp()` 完整执行逻辑（中文版）

这份文档追踪 `run_pretrain_milp()`（`util/pretrain_milp.py`）从入口到收尾的**完整解题逻辑**：
它要解决什么问题、按什么顺序做了哪几步、每一步为什么这么做、每一步的关键产物又被后面哪一步用到。
文中会提到变量，但变量只是为了说清楚"这一步在做什么"，本文**只讲逻辑步骤，不做纯 dataflow 的变量对照表**。
所有源码行号都保留下来方便查阅；technical term（MILP、UE、Frank-Wolfe、surrogate、horizon 等）保留英文，
变量名 / 函数名 / 文件路径保留原样。

---

## 0. 这个模型到底在解什么问题（先讲整体思路）

背景：一批道路 segment 在洪水后受损，需要安排 crew 去修复。每个 segment 有一个开工时刻（start slot），
修复要占用 crew 若干个 slot（duration）。我们想找一个**开工时间表 schedule**，让整段 horizon 内路网的
可达性损失（objective F1）最小。

难点在于：F1 依赖于每个时刻路网的 UE（user equilibrium）travel time，而 travel time 又依赖于哪些 segment 此刻
还坏着——也就是依赖于 schedule 本身。这是一个 schedule ↔ travel time 互相耦合的问题，直接把它塞进一个
optimization solver 是非线性、且要在每次评估内部反复跑 UE，代价极高。

§2.1.1 pretraining solver 的核心思路是 **traffic-fixation 交替优化**：

1. 先**冻结**整段 horizon 的 travel time（用上一次 UE 跑出来的结果，记为 `u_tilde`，形状 `(T, |R|)`）。
2. 在 travel time 被冻结的前提下，用一个**解析的 sensitivity 系数** `c_e^k` 把 F1 近似成一个**线性 surrogate**
   ——`c_e^k` 表示"让 segment e 在 slot k 开工，能带来多少 F1 改善"。这一步**完全不跑 UE**，纯 numpy。
3. 解一个小的 **start-time MILP**（决策变量是"segment e 是否在 slot k 开工"），最大化这个线性 surrogate，
   得到一个新的 schedule。
4. 用这个新 schedule **重新跑一遍完整 UE pipeline**，刷新 travel time。
5. 回到第 1 步，反复迭代，直到 schedule 不再变化（fixed point）或触发保护条件。

因为每一步 2/3 的 surrogate 是线性的、MILP 很小，代价主要落在第 4 步的 UE 上；而 UE 每个 outer iteration
只跑一轮（`T` 次），远比"把 UE 塞进 optimization 内层"便宜。

**入口。** `util/pretrain_milp.py :: run_pretrain_milp(toy_dir=TOY, out_dir=OUT, M=P.M_SCENARIOS, seed=P.SEED)`，
定义在 `util/pretrain_milp.py:184`。

模块级常量（`util/pretrain_milp.py:44-46`）：
- `ROOT` = 项目根目录（`…/Research_RoadRestoration`）。
- `TOY` = `ROOT/data/siouxfalls_toy`。
- `OUT` = `ROOT/outputs/pretrain_milp`。

项目参数统一放在根目录的 `config.py`，代码里 `import config as P` 引用（不再是旧的 `util/params.py`）。

> **无数据泄露说明（详见 §12 / 附录 B）。** oracle 的"解"只在**最后**画对比图 / 算 gap 时才被 post-hoc 读取，
> 那时每个 scenario 的 MILP schedule 早已算完并 checkpoint。oracle 的解**从不进入 MILP 求解过程**。
> MILP 与 oracle 唯一共享的是**问题定义**（同一个 instance、同样的 scenarios、同一个 horizon、同一个 objective），
> 这是故意为之，好让两者的 F 直接可比。

---

## 1. Call-tree 总览（一眼看全流程）

```
run_pretrain_milp()                                   util/pretrain_milp.py:184
├─ scale_dir(out_dir)                                 util/oracle.py:59      → outputs/pretrain_milp/n{N}/
├─ (out_dir/"figures").mkdir(...)
├─ select_oracle_instance(toy_dir, N)                 util/oracle.py:85      ← 注意：现在没有 seed 参数
│   ├─ pd.read_csv(edges.csv)
│   └─ _baseline_twoway_flow(toy)                     util/oracle.py:65      ← 自算 baseline UE（solve_ue，无损网络）
├─ segments = sorted(int(e) for e in disrupted.edge_id)
├─ build_context(toy_dir, disrupted)                  util/evaluate.py:106
│   ├─ load_toy_network(toy_dir)                      util/io.py:10          ← edges/od_pairs/nodes.csv
│   ├─ solve_ue(undamaged, H0, …)  [baseline UE]      util/ue.py:81
│   │   ├─ _build_graph(edges, zone_ids)              util/ue.py:46          (AequilibraE Graph)
│   │   └─ _build_matrix(M, zone_ids)                 util/ue.py:71          (AequilibraE Matrix)
│   ├─ _matrix_from_H(H0, ctx)                        util/evaluate.py:48
│   ├─ od_travel_times(base_links, ctx)               util/evaluate.py:27    (networkx Dijkstra)
│   └─ B(Φ) 通过 networkx free-flow shortest_path 构造
├─ sample_scenarios(disrupted, M, seed)               util/scenarios.py:14   (DURATION_SUPPORT × ETA)
├─ compute_horizon(segments, scenarios)               util/oracle.py:112
│   └─ 对每个 perm × scenario:
│       ├─ schedule_from_permutation(perm, dur)       util/evaluate.py:82
│       └─ makespan_slot(start, dur)                  util/evaluate.py:94
├─ _param_fingerprint()                               util/oracle.py:46      + hashlib SHA1 扩展 (damp/maxit/cyc)
├─ RESUME: 读 milp_optima.csv + milp_trace.csv + milp_progress.json → 建 `done` 集合
│
├─ for m, dur in enumerate(scenarios):  (m in done 则跳过)
│   └─ alternating_optimize(ctx, dur, segments, T)    util/pretrain_milp.py:132
│       ├─ schedule_from_permutation(segments, dur)   util/evaluate.py:82    (greedy-first 初始化)
│       ├─ evaluate_schedule(start, …, return_u=True) util/evaluate.py:156   [每 slot 一次 UE → u_tilde (T×|R|)]
│       │   ├─ f2_value / makespan_slot               util/evaluate.py:98/94
│       │   ├─ build_damaged_edges(ctx, damaged)      util/evaluate.py:54
│       │   ├─ _matrix_from_H(H, ctx)                 util/evaluate.py:48
│       │   ├─ solve_ue(dmg_edges, H, …)              util/ue.py:81
│       │   └─ od_travel_times(links, ctx)            util/evaluate.py:27
│       └─ loop 至多 MILP_MAX_ITER 次:
│           ├─ precompute_c(ctx, u_tilde, dur, …)     util/pretrain_milp.py:52   (解析 c_e^k, 不跑 UE)
│           ├─ build_and_solve_milp(c, dur, …)        util/pretrain_milp.py:83   (scipy.optimize.milp / HiGHS)
│           ├─ evaluate_schedule(new_start, …)        util/evaluate.py:156       [刷新 UE]
│           ├─ _surrogate_value(new_start, c, …)      util/pretrain_milp.py:124
│           ├─ STOP 判定: fixed-point / cycle-guard / max-iter
│           └─ damped 更新: u_tilde = damp·u_new + (1-damp)·u_tilde
│       └─ best = argmin_history 真实 F
│   └─ checkpoint: milp_optima.csv, milp_trace.csv, milp_progress.json
│
├─ run_meta.json  (timing / total_ue_solves / mean_iters / mean_F_milp)
├─ make_process_figures(...)                          viz/pretrain_viz.py:111  (fig 03/04)
└─ POST-HOC ORACLE 对比 (仅当 oracle_optima.csv 存在):
    ├─ 读 outputs/oracle/n{N}/oracle_optima.csv + oracle_landscape.csv
    ├─ merge on scenario → gap = F_milp − F_oracle → milp_vs_oracle.csv
    ├─ make_comparison(...)                           viz/pretrain_viz.py:21   (fig 01)
    └─ make_landscape(...)                            viz/pretrain_viz.py:62   (fig 02)
```

下面按逻辑步骤逐段展开。

---

## 2. Shared setup 第 1 步 — 确定输出目录（`scale_dir`）

**这一步要解决什么问题。** 不同问题规模（disrupted segment 数 `N`）的结果不能互相覆盖，得各自有独立的输出目录。

**做了什么（`util/pretrain_milp.py:185-186`）。**
```python
out_dir = scale_dir(out_dir)                     # outputs/pretrain_milp/n{N}/
(out_dir / "figures").mkdir(parents=True, exist_ok=True)
```

`scale_dir(base=OUT, n=None)`（`util/oracle.py:59-62`）在 `n is None` 时取 `n = P.N_DISRUPTED_ORACLE`，返回
`Path(base)/f"n{n}"`。默认 `N_DISRUPTED_ORACLE = 4`，所以 `out_dir → outputs/pretrain_milp/n4/`，并创建
`…/n4/figures/`。

**为什么这么做。** 这套 `n{N}` 分目录方案和 oracle 端完全一致，保证 MILP run 永远不会覆盖另一个 problem size 的结果，
也保证后面 post-hoc 对比时能按同样的 `n{N}` 路径去找对应规模的 oracle 结果。

**关键产物。** `out_dir`（本次 run 所有 CSV / JSON / 图的落盘目录），后面每一步的写文件都基于它。

---

## 3. Shared setup 第 2 步 — 选定受损实例（`select_oracle_instance` + `_baseline_twoway_flow`）

**这一步要解决什么问题。** 需要确定"到底哪几条 segment 受损、各自 severity 多少"。为了让"修复顺序"真正影响 F1，
故意混入既有关键干道、又有次要小路的组合。

**做了什么（`util/pretrain_milp.py:188-189`）。**
```python
disrupted = select_oracle_instance(toy_dir, P.N_DISRUPTED_ORACLE)
segments = sorted(int(e) for e in disrupted["edge_id"])
```

> **与旧文档不一致（已更正）。** 当前 `select_oracle_instance` 的签名是
> `def select_oracle_instance(toy_dir, n=P.N_DISRUPTED_ORACLE):`，**已经没有 `seed` 参数**。
> 这里也是按 `select_oracle_instance(toy_dir, P.N_DISRUPTED_ORACLE)` 调用的，不再传 seed。
> 旧文档 3.1 节写的 `select_oracle_instance(toy_dir, n, seed)` 是过时的。选实例的逻辑本来就完全确定性、不用 RNG。

### 3.1 `select_oracle_instance(toy_dir, n=P.N_DISRUPTED_ORACLE)` — `util/oracle.py:85-108`

按**重要性**（baseline 双向 UE flow）挑 `n` 条 segment，是**确定性**的（不用任何 RNG）。内部逻辑：

1. `edges = pd.read_csv(toy/"network"/"edges.csv")`（`oracle.py:92`）。
2. `flow = _baseline_twoway_flow(toy)`（`oracle.py:93`）——见 §3.2。
3. 给每条 edge 挂上一列 `flow`：按无向 key `(min(u,v), max(u,v))` 查表，查不到默认 `0.0`（`oracle.py:94-95`）。
4. `ranked = edges.sort_values("flow", ascending=False)`（`oracle.py:96`）——按 flow 从高到低排序。
5. `n_crit = min(2, n)`；`picks = list(range(n_crit))`——取 flow 最高的 `n_crit`（≤2）条作为"critical"干道（`oracle.py:98-99`）。
6. `rest = n - n_crit`；若 `rest > 0`，用 `np.linspace(len(ranked)//5, len(ranked)-1, rest)` 取整，把剩下的名额
   均匀铺到 flow 较低的 edge 上（`oracle.py:100-102`）。
7. `sub = ranked.iloc[picks]`（`oracle.py:103`）。
8. 分配 severity（`oracle.py:104`）：`n_crit` 条 critical edge 给 severity **3**（severed，UE 网络里被彻底移除，见 §7.2）；
   其余按奇偶交替给 **2 / 1**（`2 if i%2==0 else 1`）。
9. 拼 `level_id = road_class + "-S" + severity`（`oracle.py:105`）。
10. `out` = 子表 `[edge_id, u, v, road_class, severity, level_id]`，按 `edge_id` 排序（`oracle.py:106-107`）。
11. **写** `toy/"disruption"/f"disrupted_segments_oracle{n}.csv"`（`oracle.py:108`），返回 `out`。

**为什么这么做。** 这是 oracle 端调用的**同一个**选实例函数（`util/oracle.py:126`），保证 MILP 与 oracle 面对**完全相同的
instance**——这是刻意共享的问题定义，不是泄露。severity 3 会导致真正的网络断开，让"先修哪条"对可达性影响巨大，
从而让 schedule 的优化有意义。

**关键产物。** `disrupted`（DataFrame），以及紧接着由它得到的 `segments = sorted(int(e) …)`（`pretrain_milp.py:189`）
——这个排好序的 edge id 列表是下游到处用的**规范列顺序**，必须和 `ctx["B"]` 的列、`durations` 的 key 对齐。

### 3.2 `_baseline_twoway_flow(toy_dir)` — `util/oracle.py:65-82`

用**项目自己的 UE 引擎**在**无损**网络上算 baseline 双向 flow，用来给 link 按重要性排序。**不再读任何外部参考解文件。**
- `edges, od, zone_ids = load_toy_network(toy_dir)`（`oracle.py:74`）——载入无损路网 + OD 对（见 §4.2）。
- `flows, _ = solve_ue(edges, od_to_matrix(od, zone_ids), zone_ids, rgap=P.UE_RGAP, max_iter=P.UE_MAX_ITER, quiet=True)`
  （`oracle.py:75-76`）——先 `od_to_matrix(od, zone_ids)` 把 OD 对展成稠密矩阵，再在**无损网络 + baseline 需求 H0** 上解一次 UE
  （见 §4.3）。
- 把 `flows` 里每条**有向** link 的 `volume` 按**无向** key `(min(u,v), max(u,v))` 两向累加：
  `f[key] = f.get(key, 0.0) + volume`（`oracle.py:77-81`）。
- 返回 `{(min,max): 两向合计流量}`（`oracle.py:82`）。

**为什么这么做。** 旧实现逐行读开源参考解文件 `raw/SiouxFalls_flow.tntp`（published equilibrium flow）来排序；现在改成自己跑
一次 baseline UE 算出同样的双向 flow，**去掉了对那个 shipped 参考文件的依赖**——换一套 network / OD 数据集时不必再另配一个外部
参考-flow 文件。**经验证：换成自算 UE flow 后，选出的受损实例完全不变**（仍是 edge_ids `[1,12,15,17]`、severity `[1,2,3,3]`，
两向流量与旧参考解逐边误差仅约 0.14%）。注意 `raw/SiouxFalls_flow.tntp` 现在**只被** `util/ue.py:_validate` 的 UE 自校验读取，
选实例流程不再碰它。

---

## 4. Shared setup 第 3 步 — 构建静态 context（`build_context`）

**这一步要解决什么问题。** 后面每个 scenario、每个 slot 都要反复用到一堆**一次算好就不再变**的东西：路网、OD 对、
baseline travel time、断连惩罚时间、以及把"损坏"翻译成"需求缺口"的矩阵 `B(Φ)`。这一步一次性把它们全算出来，装进 `ctx`。

**做了什么（`util/pretrain_milp.py:190`）。** `ctx = build_context(toy_dir, disrupted)`。

### 4.1 `build_context(toy_dir, disrupted)` — `util/evaluate.py:106-150`

逐步：

1. `edges, od, zone_ids = load_toy_network(toy_dir)`（`evaluate.py:108`）——见 §4.2。
2. `zone_pos = {node_id: index}`（`evaluate.py:109`）——node id → 在 zone 列表里的位置。
3. `od_pairs = [(origin, destination) …]`（`evaluate.py:110`）——所有 OD 对。
4. `H0 = od["h0"].to_numpy()`——baseline（正常时期）OD 需求向量（`evaluate.py:111`）。
5. `oi`、`di`——每个 OD 的 origin/destination 在 zone 列表里的整数位置（`evaluate.py:112-113`）；
   给 `_matrix_from_H` 把需求向量散射回矩阵用。
6. `edge_row = {edge_id: 行号}`、`eid_of = {(min,max): edge_id}`（`evaluate.py:114-116`）。
7. 组装 `ctx` 字典：`edges, zone_ids, od_pairs, H0, oi, di, nz=len(zone_ids), edge_row, origins_unique`
   （`evaluate.py:118-120`）。

8. **baseline UE** `u_r^{t0}`：在**未受损**网络上、用**正常**需求 `H0` 跑一次 UE（`evaluate.py:123-124`）：
   ```python
   base_links, _ = solve_ue(edges, _matrix_from_H(H0, ctx), zone_ids,
                            rgap=P.UE_RGAP, max_iter=P.UE_MAX_ITER, quiet=True)
   ctx["baseline_u"] = od_travel_times(base_links, ctx)
   ```
   `_matrix_from_H` 见 §4.5；`solve_ue` 见 §4.3；`od_travel_times` 见 §4.4。
   **为什么。** `baseline_u` 是 F1 里的分母基准（"当前 travel time 比正常时期差多少"），也是 `c_e^k` 里 `α_r^{k'}` 的基准。
9. **惩罚时间** `u_pen = UPEN_FACTOR × max(有限的 baseline_u)`（`evaluate.py:126`）——当某 OD 对被彻底断开、travel time 为
   `inf` 时，用这个大数替代，让 F1 有限但被重罚。

10. **受损元组** `dis = [(edge_id, u, v, severity), …]` 存进 `ctx["disrupted"]`（`evaluate.py:129-131`）。
    这是 `evaluate_schedule` 每个 slot 判定"哪些还坏着"的数据源。

11. **B(Φ) 矩阵构建**（`evaluate.py:132-148`）：
    - 先建一个 **free-flow、双向**的 networkx `DiGraph` `Gff`，权重是 `free_flow_time`（`evaluate.py:132-135`）。
    - `col = {edge_id: 列号 j}` 把每条受损 edge 映到 B 的一列（`evaluate.py:137`）。
    - 对每个 OD 对 `(o,d)`：算 free-flow 最短路（`nx.shortest_path`，`evaluate.py:139-142`，`NetworkXNoPath` 则跳过），
      收集这条路上的无向 edge 集合（`evaluate.py:143`）。对其中每条**属于受损 segment** 的 edge，令
      `B[i, col[eid]] = KAPPA · (H0[i] / 3.0)`（`evaluate.py:144-147`）。
    - 即 **B[r,e]** = `κ·(h_r0/3)`（若受损 e 落在 OD r 的 free-flow 最短路上），否则 0。
    - `ctx["B"] = B`（`evaluate.py:148`）。
    **为什么。** `B` 是"损坏 → demand shortfall"的线性映射：某条受损路只压制那些原本要经过它的 OD 需求。
12. `ctx["severity_vec"] = np.array([sev for (_,_,_,sev) in dis])`（`evaluate.py:149`）——按 `dis` 顺序（与 B 列对齐）
    排的 severity 向量。**这个向量在 §8 的 `precompute_c` 里被当作 `v_e*`（segment 完全恢复时的需求尺度）用**。
13. 返回 `ctx`。

**关键产物。** `ctx` 字典，含
`edges, zone_ids, od_pairs, H0, oi, di, nz, edge_row, origins_unique, baseline_u, u_pen, disrupted, B, severity_vec`。
其中 `B`、`severity_vec`、`baseline_u` 是 §8 算 `c_e^k` 的三块核心输入；`disrupted`、`u_pen`、`edges` 是 §7 每 slot 评估的核心输入。

**用到的参数。** `UE_RGAP`、`UE_MAX_ITER`、`UPEN_FACTOR`、`KAPPA`。

### 4.2 `load_toy_network(toy_dir)` — `util/io.py:10-23`

读 `network/edges.csv`、`network/od_pairs.csv`、`network/nodes.csv`（`io.py:19-21`）；
`zone_ids = nodes["node_id"]`（Sioux Falls 里每个 node 都是 OD zone，`io.py:22`）。返回 `(edges_df, od_df, zone_ids)`。
`edges_df` 列：`edge_id,u,v,capacity,length,free_flow_time,bpr_alpha,bpr_beta,road_class`；`od_df` 列：`od_id,origin,destination,h0`。

### 4.3 `solve_ue(edges, od_matrix, zone_ids, algorithm="bfw", max_iter=1000, rgap=1e-10, quiet=False)` — `util/ue.py:81-121`

静态 user-equilibrium（Wardrop / Beckmann）分配，走 AequilibraE 的 bi-conjugate Frank-Wolfe。
这是**整条 pipeline 里最贵的数值步骤**：`evaluate_schedule` 里每个 slot 跑一次，`build_context` 里 baseline 跑一次。

子逻辑：
1. `graph, net = _build_graph(edges, zone_ids)`——见 §4.3.1。
2. `mat = _build_matrix(od_matrix, zone_ids)`——见 §4.3.2。
3. 包成 `TrafficClass("car", graph, mat)`，建 `TrafficAssignment`（`ue.py:92-93`）。
4. `set_vdf("BPR")` + `set_vdf_parameters({"alpha":"b","beta":"power"})`——BPR 拥堵成本
   `t_a(x) = t0·(1 + α·(x/c)^β)`（`ue.py:95-96`）。
5. `set_capacity_field("capacity")`、`set_time_field("free_flow_time")`、`set_algorithm("bfw")`（`ue.py:97-99`）。
6. `assig.max_iter = max_iter`；`assig.rgap_target = rgap`（`ue.py:100-101`）。
7. 若 `quiet`：把 AequilibraE 日志级别抬到 CRITICAL 并把 stdout/stderr 重定向到 `os.devnull`，再 `assig.execute()`；
   否则直接跑（`ue.py:102-108`）。`execute()` 跑 FW 循环：all-or-nothing → line search → 更新 cost，直到 relative gap < `rgap`。
8. `res = assig.results()`——每条 link 的 AB/BA volume + congested time（`ue.py:110`）。
9. 重整成 tidy 有向行：每条 link 出两行，一行 `{from:u,to:v,volume:demand_ab,cost:Congested_Time_AB}`，
   一行反向 `{from:v,to:u,volume:demand_ba,cost:Congested_Time_BA}`（`ue.py:113-118`）。
10. 丢掉 volume 为 `NaN` 的行，返回 `(flows_df, assig)`，`flows_df` 列 `from,to,volume,cost`（`ue.py:119-121`）。

**调用方约定。** 本 pipeline 里总是用 `rgap=UE_RGAP (1e-6)`、`max_iter=UE_MAX_ITER (100)`、`quiet=True`。

#### 4.3.1 `_build_graph(edges, zone_ids)` — `util/ue.py:46-68`
从无向 edge 表建可路由的 AequilibraE `Graph`：每条 edge 一条双向 link（`direction=0`），AB/BA capacity 对称，
带 `free_flow_time`、BPR `b`(=α) / `power`(=β)。`prepare_graph(zone_ids)` 把每个 node 标成 zone；
`set_graph("free_flow_time")`；`set_blocked_centroid_flows(False)` 允许过境流量。返回 `(graph, net)`。

#### 4.3.2 `_build_matrix(M, zone_ids)` — `util/ue.py:71-78`
把稠密 OD 矩阵 `M` 包成一个内存里的 `AequilibraeMatrix`，单 core `"demand"`，用 `zone_ids` 索引，设好计算视图。返回矩阵。

> `beckmann_objective(...)`（`ue.py:124-134`）和 `_validate()`（`ue.py:141-195`）在模块里但**不在 pretraining 路径上**
> （仅用于验证 / 自检）。

### 4.4 `od_travel_times(link_df, ctx)` — `util/evaluate.py:27-45`

把 congested 有向 link time 通过最短路转成每个 OD 对的 travel time。
- 从 `link_df[from,to,cost]` 建 networkx `DiGraph`，每条 edge 权重 = congested `cost`（`evaluate.py:31-35`）。
- 对每个 unique origin 跑 `single_source_dijkstra_path_length`（origin 不在图里——它所有 incident edge 都被 severed 时
  ——则空 dict）（`evaluate.py:37-42`）。
- 按 `ctx["od_pairs"]` 顺序拼 `u`；O、D 断连的项默认 `np.inf`（`evaluate.py:36, 43-44`）。
- 返回 `u`（长度 `|R|`，断连处为 `np.inf`）。

### 4.5 `_matrix_from_H(H, ctx)` — `util/evaluate.py:48-51`

把一个需求**向量** `H`（按 `od_pairs` 排）散射回稠密 `nz×nz` 矩阵：`M[oi, di] = H`。返回 `M`。
这是"OD 对展平"的逆操作，也是 `solve_ue` 要吃的输入格式。

---

## 5. Shared setup 第 4 步 — 采样 duration scenarios（`sample_scenarios`）

**这一步要解决什么问题。** 修复时长是不确定的，我们要在 `M` 个 duration scenario 上评估 schedule 的稳健性。

**做了什么（`util/pretrain_milp.py:191`）。** `scenarios = sample_scenarios(disrupted, M, seed)`。

### 5.1 `sample_scenarios(disrupted, M=P.M_SCENARIOS, seed=P.SEED)` — `util/scenarios.py:14-28`

抽 `M` 个修复时长 scenario；时长是**按 level `(road_class, severity)` 抽**的，同一 level 的所有受损 segment 在一个 scenario 里
共用一个时长。
- `rng = np.random.default_rng(seed)`——固定 `seed=42` 则完全确定（`scenarios.py:17`）。
- `levels = sorted({(road_class, severity)})`（`scenarios.py:19`）。
- 对每个 scenario、每个 level：`base = rng.choice(DURATION_SUPPORT[level])`、`eta = rng.choice(ETA)`、
  `lvl_dur[level] = max(1, round(base·eta))`（`scenarios.py:21-26`）。
- 输出 `scenario[m] = {edge_id: lvl_dur[(road_class,severity)]}`（每条受损 edge 一项）（`scenarios.py:27`）。
- 返回 `M` 个 dict `{edge_id: duration_slots}` 的列表。

**为什么这么做。** 这里的 `seed` 与 oracle 端用同一个，保证同 seed 下两边采到**完全一样的 scenarios**，F 才可比。

**用到的参数。** `M_SCENARIOS`、`SEED`、`DURATION_SUPPORT`（Table 1 的 base support 集合）、
`ETA`（crew-efficiency 乘子 `[0.8, 1.0, 1.2]`）。

---

## 6. Shared setup 第 5 步 — 计算全局 horizon `T`（`compute_horizon`）

**这一步要解决什么问题。** F1 是在 `[1, T]` 上做平均的，所有 schedule / 所有 scenario 必须共用**同一个** `T`，才能公平比较；
且 `T` 要足够大，让**任何** work-conserving schedule 都能在 `T` 内完工。

**做了什么（`util/pretrain_milp.py:192-193`）。**
```python
T = compute_horizon(segments, scenarios)
print(f"instance: {len(segments)} segments {segments}; M={M}; horizon T={T}")
```

### 6.1 `compute_horizon(segments, scenarios)` — `util/oracle.py:112-119`

全局 horizon `T` = **所有 permutation × 所有 scenario** 上的最大完工 slot。
- 双层遍历 `itertools.permutations(segments)` 和 `scenarios`（`oracle.py:116-117`）。
- 每组：`T = max(T, makespan_slot(schedule_from_permutation(list(perm), dur), dur))`（`oracle.py:118`）。
- 返回最大的 `T`（int）。

#### 6.1.1 `schedule_from_permutation(perm, durations, c_max=P.C_MAX)` — `util/evaluate.py:82-91`
work-conserving 的列表排程，`c_max` 个相同 crew，最早开工 slot = 1。
- `crew_free = [1]*c_max`（每个 crew 下次空闲的 slot）。
- 按优先级顺序对每条 edge `e`：挑最早空闲的 crew `c = argmin(crew_free)`；`start[e] = crew_free[c]`；
  `crew_free[c] = start[e] + durations[e]`（该 crew 忙到完工）。
- 返回 `{edge_id: start_slot}`。不留空档 → schedule 完全由 permutation 决定。

#### 6.1.2 `makespan_slot(start, durations)` — `util/evaluate.py:94-95`
`max_e (start[e] + durations[e])`——最后一条 segment 的完工 slot。

**为什么这么做。** `T` 是后面 MILP 决策变量的时间维度、F1 平均的窗口、`c_e^k` 里 `1/T` 归一化的分母，必须先定死。

> 默认实例 `N=4`（`4! = 24` 个 permutation）× `M=10` 个 scenario，就是 240 次很便宜的组合评估（不跑 UE）。

---

## 7. 真实 objective `F(x|ω)` 的完整追踪（`evaluate_schedule`）

`evaluate_schedule` 是精确的 Figure-1 evaluator，和 oracle 端**一字不差地复用**。在 pretraining 循环里它总是带
`return_u=True` 调用，所以除了返回 F/F1/F2，还额外返回 `(T×|R|)` 的 travel-time 矩阵 `u_tilde`——这个 `u_tilde`
正是 §8 `precompute_c` 要冻结、要拿来算 `c_e^k` 的东西。

**它在整体逻辑里的角色。** 这是**真实**目标函数（唯一会跑 UE 的评估器）。交替优化每一轮都靠它来：
(a) 拿到当前 schedule 的真实 F（用来事后挑最优 iterate）、(b) 拿到刷新后的 `u_tilde`（喂给下一轮的 surrogate）。

### 7.1 `evaluate_schedule(start, durations, T, ctx, collect_traces=False, return_u=False)` — `util/evaluate.py:156-207`

先拆出 `dis = ctx["disrupted"]`、`H0`、`B`、`base_u = ctx["baseline_u"]`（`evaluate.py:159-162`）。

**Step 2 — F2（不跑 UE）：** `F2 = f2_value(start, durations)`（`evaluate.py:164`）。
- `f2_value`（`util/evaluate.py:98-101`）：`makespan_slot / Σ_e durations[e]`（`Δt` 约掉）。F2 只用于 logging；`MU=1` 时不进 `F`。

**Step 3 — per-slot F1 循环**（`evaluate.py:169-192`），`k = 1 … T`：
1. 循环前 `D = np.zeros(len(H0))`（`evaluate.py:167`）——上一 slot 的 shortfall，供 recovery 递推。
2. **损坏状态（Eq. 2）：** `damaged = {eid: s for (eid,_,_,s) in dis if k < start[eid] + durations[eid]}`
   ——只要 `k` 还没到完工 slot，该 segment 就算损坏（`evaluate.py:171`）。`v_vec = [s if damaged else 0]`（`evaluate.py:172`）。
3. **demand shortfall → `H`：** `target = B @ v_vec`；`D = max(target, RHO·D)`（逐元素）；`H = clip(H0 - D, 0, None)`
   （`evaluate.py:174-176`）。这是 shortfall 模型：onset 时急剧掉到当前损坏驱动的缺口 `B·v`，随后随路修好、`B·v` 缩小，
   需求以速率 `RHO` 逐步恢复。
4. **受损网络：** `dmg_edges = build_damaged_edges(ctx, damaged)`（`evaluate.py:178`）——见 §7.2。
5. **UE → OD travel time：** `links,_ = solve_ue(dmg_edges, _matrix_from_H(H,ctx), zone_ids, rgap=UE_RGAP,
   max_iter=UE_MAX_ITER, quiet=True)`（`evaluate.py:180-181`）；再 `u = od_travel_times(links, ctx)`（`evaluate.py:182`）。
   **这就是"每 slot 一次 UE"。**
6. **惩罚替换：** `u_tilde = where(isfinite(u), u, u_pen)`——断连 OD 对取 `u_pen`（`evaluate.py:183`）。这一行 append 进
   `u_rows`（`evaluate.py:184`），最后堆成 `(T×|R|)` 矩阵。
7. **F1 项：** demand-weighted 的"实现/baseline travel time 比值"：`den = Σ(H·base_u)`；
   `term = Σ(H·u_tilde)/den`（`den>0` 时，否则 `1.0`）（`evaluate.py:186-188`）。append 进 `terms`；`active` 记录本 slot 是否还有损坏（`evaluate.py:189-190`）。
8. `traces` 视 `collect_traces` 收集（MILP 路径上不用）（`evaluate.py:190-191`）。

**聚合（`evaluate.py:194-206`）：**
- `terms = np.asarray(terms)`（`evaluate.py:194`）。
- 若 `F1_ACTIVE_ONLY`（默认 **False**）：`F1 = terms.mean()`，对整段 `[1,T]` 平均（`evaluate.py:198-199`）；否则只对 active slot 平均。
  **整段平均和 `precompute_c` 里的 `1/T` 归一化保持一致**（这是 `F1_ACTIVE_ONLY=False` 的原因）。
- `F = MU·F1 + (1-MU)·F2`（`evaluate.py:200`）。`MU=1` 时 `F ≡ F1`。
- `out = {F, F1, F2}`；若 `return_u`：`out["u_tilde"] = np.asarray(u_rows)`——那块 **(T×|R|)** 的冻结 travel-time 矩阵（`evaluate.py:201-203`）。
- 返回 `out`。

**关键产物。** `{F, F1, F2, u_tilde}`。**每次调用恰好跑 `T` 次 UE。**

**用到的参数。** `MU`、`F1_ACTIVE_ONLY`、`RHO`、`UE_RGAP`、`UE_MAX_ITER`（以及 `ctx` 里的 `u_pen`/`B`）。

### 7.2 `build_damaged_edges(ctx, damaged)` — `util/evaluate.py:54-76`

产出当前 slot 的受损 edge 表。
- 复制 `ctx["edges"]`（`evaluate.py:59`）。
- 若有 `damaged`：复制 `capacity`/`free_flow_time` 数组；对每个 `(eid, sev)`：
  - `sev >= SEVER_SEVERITY`（默认 3）→ 标记该行**移除**（真正断连）（`evaluate.py:67-68`）；
  - 否则 `cap[j] *= CAP_RETAIN[sev]`、`fft[j] /= SPEED_RETAIN[sev]`（降级但仍在网中）（`evaluate.py:70-72`）。
  - 写回数组；丢掉 severed 行并重置索引（`evaluate.py:73-76`）。
- 未损坏 / 已完工的 link 保持不变。返回 edges DataFrame。

**用到的参数。** `SEVER_SEVERITY`、`CAP_RETAIN`、`SPEED_RETAIN`。

---

## 8. 交替优化第 1-2 步 — 解析 F1 敏感度系数 `c_e^k`（`precompute_c`，不跑 UE）

**这一步要解决什么问题。** MILP 需要一个**线性**的目标来最大化，但真实 F1 依赖 UE、非线性。这一步在 travel time 被冻结
（`u_tilde` 固定）的前提下，用**纯解析、纯 numpy** 的方式算出每个 `(segment e, slot k)` 的 F1 敏感度 `c_e^k`——
即"让 e 在 k 开工能带来多少 F1 改善"。**整个过程不再跑任何 UE**，这是交替优化便宜的关键。

**在哪里调用。** `util/pretrain_milp.py:157`（`alternating_optimize` 内部，每个 outer iteration 一次）。

### 8.1 `precompute_c(ctx, u_by_slot, durations, segments, T)` — `util/pretrain_milp.py:52-77`

闭式（Problem-2 shortfall 修正，`pretrain_milp.py:56-58`）：
```
c_e^k = (1/T) · Σ_{k'=k+d_e}^{T} (1 − ρ^{k'−k−d_e+1}) · Σ_r B[r,e]·v_e*·α_r^{k'}
α_r^{k'} = 1 − baseline_u[r] / u_by_slot[k'−1, r]
```

直觉：`α_r^{k'}` 是 OD r 在 slot k' 相对 baseline 的 travel-time 超额（越拥堵越大）。把 e 修好后，它原本压制的需求
`B[r,e]·v_e*` 会回到网上，回到得越多、这些 OD 越拥堵，修它的收益越大。收益从完工 slot `k+d_e` 起才开始产生，
并随时间以 `(1 − ρ^{n+1})` **增长**（路越修越多、需求越恢复）。

实现：
1. 拆 `B = ctx["B"]`、`sev = ctx["severity_vec"]`、`base_u = ctx["baseline_u"]`、`rho = P.RHO`（`pretrain_milp.py:60-63`）。
2. `assert B.shape[1] == len(segments)`——B 的列必须和 segment 顺序对齐（`pretrain_milp.py:64`）。
3. **α 矩阵：** `alpha = 1.0 - base_u[None,:] / u_by_slot` → 形状 `(T, |R|)`（`pretrain_milp.py:66`）。
   当某 slot 的 congested time **低于** baseline 时 `α_r^{k'}` 可以 **< 0**（合法，例如恢复期低需求下）。
4. `c = zeros((|segments|, T))`（`pretrain_milp.py:67`）。
5. 对每个 segment `j`（edge `e`）、duration `d = durations[e]`（`pretrain_milp.py:68-69`）：
   - `Bv = B[:, j] * sev[j]`——这条 segment 完全恢复时能放回的需求（`|R|` 向量）（`pretrain_milp.py:70`）。
     **这里 `sev[j]` 就是 `v_e*`**（该 segment 的 severity，被用作"恢复的需求尺度"）——这正是 `ctx["severity_vec"]`
     在整个系数里的作用。
   - `w = alpha @ Bv`——`(T,)` 向量，`w[k'-1] = Σ_r α_r^{k'}·B[r,e]·v*`（`pretrain_milp.py:71`）。
   - 对每个**可行**开工 `k in 1 … T-d`（完工 `k+d ≤ T`）（`pretrain_milp.py:72`）：
     - `kp = arange(k+d, T+1)`——从完工 slot 到 horizon 末尾（`pretrain_milp.py:73`）。
     - `decay = 1.0 - rho**(kp - k - d + 1)`——增长的恢复因子 `(1 − ρ^{n+1})`，`n = k' − k − d`（`pretrain_milp.py:75`）。
       *这是 Problem-2 修正：恢复的需求随时间**增长**，而不是旧版的衰减。*
     - `c[j, k-1] = (decay @ w[kp-1]) / T`（`pretrain_milp.py:76`）。
6. 不可行开工（`k > T-d`）保持为 `0`——MILP 的 bounds（§9）会强制那些 `y` 为 0。
7. 返回 `(c, alpha)`；`c` 形状 `(|E|, T)`，`alpha` 形状 `(T, |R|)`。

**关键产物。** 系数矩阵 `c (|E|×T)`。它就是 §9 MILP 的目标系数：MILP 只需最大化 `Σ c·y`，不再碰 UE。

**用到的参数。** `RHO`。**UE 调用次数：0。**

---

## 9. 交替优化第 3 步 — start-time MILP（`build_and_solve_milp`，scipy / HiGHS）

**这一步要解决什么问题。** 给定线性系数 `c`，在"每条 segment 恰好开工一次、crew 数不超上限、都在 horizon 内完工"的
约束下，选出让 surrogate `Σ c·y` 最大的开工时间表。

**在哪里调用。** `util/pretrain_milp.py:158`。

### 9.1 `build_and_solve_milp(c, durations, segments, T, c_max=P.C_MAX)` — `util/pretrain_milp.py:83-121`

**决策变量。** `y[j,k] ∈ {0,1}`：segment `segments[j]` 是否在 slot `k` 开工。`E = len(segments)`，共 `n = E·T` 个变量；
索引 `idx(j,k) = j·T + (k-1)`（`k` 从 1 起）（`pretrain_milp.py:85-90`）。

1. **目标函数：** `obj = -c.reshape(-1)`——展平并取负，因为 HiGHS 是**最小化**，取负即等价于最大化 surrogate（`pretrain_milp.py:92`）。
2. **horizon 边界约束（用 Bounds 实现）：** `ub = ones(n)`，再对每条 segment 把 `k in T-d+1 … T` 的
   `ub[idx(j,k)] = 0`——禁止任何会越过 horizon 的开工。**这是唯一真正的可行性守卫**（`pretrain_milp.py:94-99`）。
   `bounds = Bounds(zeros(n), ub)`（`pretrain_milp.py:99`）。这样一来，`k > T-d` 的那些"不可行开工"（其 `c` 本就是 0）被强制取 0。
3. **start-once 等式：** `A_eq (E×n)`，`A_eq[j, idx(j,k)] = 1`（所有 `k`）；`con_eq = LinearConstraint(A_eq, 1, 1)` →
   `Σ_k y[j,k] = 1`（每条 segment 恰好开工一次）（`pretrain_milp.py:101-105`）。
4. **crew 上限：** `A_ub (T×n)`；对每个 slot `k`、每条 duration 为 `d` 的 segment `j`，把所有"此刻仍在修"的开工
   `kp in max(1,k-d+1) … k` 标 1（`pretrain_milp.py:107-112`）；`con_ub = LinearConstraint(A_ub, -inf, c_max)` →
   每个 slot 同时在修的 segment ≤ `c_max`（`pretrain_milp.py:113`）。
5. **求解：** `res = milp(c=obj, constraints=[con_eq, con_ub], integrality=ones(n), bounds=bounds)`——全整数（配合
   bounds 即 binary）（`pretrain_milp.py:115-116`）。失败则 `raise RuntimeError(res.message)`（`pretrain_milp.py:117-118`）。
6. **解码：** 把 `res.x` reshape 成 `(E, T)`，返回 `{edge_id: argmax_k y[j] + 1}`——每条 segment 选中的 1-based 开工 slot（`pretrain_milp.py:120-121`）。

**为什么用 MILP 而不是 permutation 枚举。** MILP 的可行域是**所有满足约束的开工时间表**，比 oracle 那种
work-conserving（不留空档）的 schedule 更大——它允许 crew 故意空等。所以 MILP 可能找到 work-conserving 集合外的更优解
（这正是 post-hoc gap 可能为负的原因，见 §12 / 附录 B）。

**关键产物。** `{edge_id: start_slot}` 新 schedule。**Solver：** `scipy.optimize.milp`（HiGHS branch-and-bound，随 scipy 附带，无需 license）。

### 9.2 `_surrogate_value(start, c, segments, T)` — `util/pretrain_milp.py:124-126`

给定一个 schedule，算它的（符号翻回来的）MILP 目标值：`Σ_j c[j, start[e]-1]`（对开工在 `[1,T]` 内的 segment）。
用于把每轮的 surrogate 记进 trace（`pretrain_milp.py:161`），以及 `level_a` 里当 brute-force 目标。返回 float。

---

## 10. 交替优化主循环 — fix → MILP → refresh（`alternating_optimize`）

**这一步要解决什么问题。** 把 §7（真实评估 + 冻结 travel time）、§8（算 `c`）、§9（解 MILP）串成迭代，
直到 schedule 收敛到 fixed point，并**始终返回历史上真实 F 最优的那个 iterate**（所以即使不收敛也无害）。

**在哪里调用。** `util/pretrain_milp.py:214`（每个 scenario 一次）。

### 10.1 `alternating_optimize(ctx, durations, segments, T, damping=None)` — `util/pretrain_milp.py:132-178`

返回 `(best_start, best_result, n_iter, converged, trace)`。

**初始化（`pretrain_milp.py:138-154`）：**
- `damping = P.MILP_DAMPING if damping is None else damping`（默认 0.5）（`pretrain_milp.py:138`）。
- `_row(...)` 是个局部小工具，把一轮的 `{iter, F, F1, F2, surrogate, elapsed_s, start_<e>…}` 打成 trace dict（`pretrain_milp.py:141-145`）。
- **greedy-first 初始化：** `start = schedule_from_permutation(list(segments), durations)`——按 segment id 原顺序的
  work-conserving schedule（§6.1.1）（`pretrain_milp.py:147`）。
- `res = evaluate_schedule(start, …, return_u=True)`——拿真实 F + `u_tilde`（§7）（`pretrain_milp.py:148`）。
- `history = [(dict(start), res)]`；`trace = [_row(0, res, nan, start)]`（iter 0 还没有 surrogate）（`pretrain_milp.py:149-150`）。
- `seen = {frozenset(start.items()): 1}`——**出现次数计数器**，给放宽的 cycle guard 用（`pretrain_milp.py:151`）。
- `u_tilde = res["u_tilde"]`；`converged=False`；`n_iter=0`（`pretrain_milp.py:152-154`）。

**主循环 `for _ in range(P.MILP_MAX_ITER)`（`pretrain_milp.py:155-172`）：**
1. `n_iter += 1`（`pretrain_milp.py:156`）。
2. `c, _ = precompute_c(ctx, u_tilde, durations, segments, T)`——用**当前冻结的** travel time 算 `c`（§8）（`pretrain_milp.py:157`）。
3. `new_start = build_and_solve_milp(c, durations, segments, T)`——解 MILP（§9）（`pretrain_milp.py:158`）。
4. `new_res = evaluate_schedule(new_start, …, return_u=True)`——给新 schedule **刷新 UE**（§7）（`pretrain_milp.py:159`）。
5. append `(dict(new_start), new_res)` 到 `history`；append 一条带 `_surrogate_value(new_start, c, …)` 的 trace 行
   （`pretrain_milp.py:160-161`）。
6. **停止条件 ① — fixed point：** 若 `new_start == start` → `converged=True; break`（`pretrain_milp.py:162-164`）。
   这是真正的收敛：新解就是上一轮的起点，迭代到了不动点。
7. **停止条件 ② — 放宽的 cycle guard：** `h = frozenset(new_start.items())`；`seen[h] += 1`；
   若 `seen[h] >= P.MILP_CYCLE_TOL`（默认 **3**）→ `break`（`pretrain_milp.py:165-168`）。
   即：某个 schedule **重复出现的次数**达到 3 次才停（基于**计数**，不是一见到重复就停）。
   这样允许被 damping 平滑后的循环继续探索一阵，跳出小的 2-cycle / 极限环。
8. **damped travel-time 更新（under-relaxation / MSA 风格）：**
   `u_tilde = damping·new_res["u_tilde"] + (1-damping)·u_tilde`（`MILP_DAMPING=0.5`）（`pretrain_milp.py:171`）。
   即把冻结的 travel time 往新 UE 结果**混合一半**，让 `c_e^k`（进而 MILP 解）在迭代间**渐变**而不是跳变，抑制振荡。
   **关键点：damping 只平滑迭代轨迹，不改变 fixed point**——若某个 `u_tilde` 是不动点（`new_res["u_tilde"] == u_tilde`），
   凸组合仍等于它本身，所以稳态解不受 damping 影响；damping 只影响"怎么走到那里"。
9. `start = new_start`（`pretrain_milp.py:172`）。
10. **停止条件 ③ — max-iter 上限：** `for` 的 range 本身封顶 `P.MILP_MAX_ITER`（默认 **20**）。

**选择（`pretrain_milp.py:174-178`）：**
- `best_idx = argmin([h[1]["F"] for h in history])`——在**全历史**里按**真实 `F`** 取 argmin（`pretrain_milp.py:174`）。
- 给 `trace[i]["is_best"] = (i == best_idx)` 打标（`pretrain_milp.py:175-176`）。
- `best_start, best_res = history[best_idx]`（`pretrain_milp.py:177`）。
- 返回 `(best_start, best_res, n_iter, converged, trace)`。

**为什么"best-by-true-F"。** surrogate 只是近似，迭代不一定单调下降，甚至可能在极限环里摆动；但我们**在每个 iterate 上都算了真实 F**，
所以直接对历史取真实 F 最小者即可，收敛与否都无所谓——这让整个 heuristic 的输出有下界保证（至少不比 greedy 初始化差）。

**关键产物。** 该 scenario 的最优 schedule + 其结果（F/F1/F2/u_tilde）+ 迭代次数 + 收敛标志 + 逐轮 trace。

**UE 调用次数。** `1`（初始化）+ 每轮 `1` ⇒ `(n_iter + 1)` 次 `evaluate_schedule`，每次 `T` 次 UE ⇒
每个 scenario `(n_iter + 1)·T` 次 UE（与 §11 的 `ue_solves` 一致）。

**用到的参数。** `MILP_DAMPING`、`MILP_MAX_ITER`、`MILP_CYCLE_TOL`、`C_MAX`（经 `schedule_from_permutation` 与 MILP）、
`RHO`、`MU`、`F1_ACTIVE_ONLY`、`UE_RGAP`、`UE_MAX_ITER`（经 `evaluate_schedule` / `precompute_c`）。

---

## 11. resume 支持、per-scenario 循环、checkpoint 写盘

### 11.1 fingerprint（`_param_fingerprint` + MILP 扩展）

**这一步要解决什么问题。** 断点续跑必须确保"续的是同一套参数的 run"。任何影响 F 的参数或 MILP loop 参数变了，缓存就该失效、重跑。

**做了什么（`util/pretrain_milp.py:196-198`）。**
```python
from util.oracle import _param_fingerprint
_, base_fp = _param_fingerprint()
fp = hashlib.sha1(f"{base_fp}|damp={P.MILP_DAMPING}|maxit={P.MILP_MAX_ITER}|cyc={P.MILP_CYCLE_TOL}".encode()).hexdigest()
```
- `_param_fingerprint()`（`util/oracle.py:46-56`）：读 `FINGERPRINT_PARAMS` 列表（`oracle.py:39-43`——
  `N_DISRUPTED_ORACLE, MU, CAP_RETAIN, SPEED_RETAIN, SEVER_SEVERITY, F1_ACTIVE_ONLY, RHO, KAPPA, UPEN_FACTOR,
  DELTA_T_H, C_MAX, M_SCENARIOS, SEED, UE_RGAP, UE_MAX_ITER, DURATION_SUPPORT, ETA`），把 dict 的 key 字符串化
  （`DURATION_SUPPORT` 是 tuple key）以求稳定 JSON，返回 `(values, sha1(json.dumps(..., sort_keys=True)))`。
- MILP 层再在 base fingerprint 后**追加**三个 MILP 专属参数（`MILP_DAMPING`、`MILP_MAX_ITER`、`MILP_CYCLE_TOL`），
  重新 SHA1 得 `fp`（`pretrain_milp.py:198`）。所以任何 F-affecting 参数或 MILP loop 参数一变，缓存 run 就失效。

### 11.2 resume 逻辑（建 done 集合）

**做了什么（`util/pretrain_milp.py:199-207`）。**
- 路径：`opt_path = milp_optima.csv`、`trace_path = milp_trace.csv`、`prog_path = milp_progress.json`（`pretrain_milp.py:199-200`）。
- `rows, trace_rows, done = [], [], set()`（`pretrain_milp.py:201`）。
- **仅当**三个文件都存在**且** `json.load(milp_progress.json)["hash"] == fp`（`pretrain_milp.py:202-203`）：
  - `rows = pd.read_csv(milp_optima.csv).to_dict("records")`（`pretrain_milp.py:204`）。
  - `trace_rows = pd.read_csv(milp_trace.csv).to_dict("records")`（`pretrain_milp.py:205`）。
  - `done = {int(r["scenario"]) for r in rows}`——已完成的 scenario 集合（`pretrain_milp.py:206`）。
  - 打印 `[resume] k/M …`（`pretrain_milp.py:207`）。
- hash 不匹配（参数改了）则忽略 checkpoint → 全新 run。

### 11.3 per-scenario 循环 + checkpoint

**做了什么（`util/pretrain_milp.py:209-229`）。** `for m, dur in enumerate(scenarios)`：
1. `if m in done: continue`——跳过已完成的 scenario（`pretrain_milp.py:211-212`）。
2. `best_start, best_res, n_iter, converged, trace = alternating_optimize(ctx, dur, segments, T)`（`pretrain_milp.py:214`）——§10。
3. `scen_s = perf_counter() - t_s`（`pretrain_milp.py:215`）。
4. 组 optima `row`（`pretrain_milp.py:216-220`）：`scenario=m, F_milp=best_res["F"], F1, F2, n_iter, converged, time_s=scen_s,
   ue_solves=(n_iter+1)*T, durations="-".join(按 segment 顺序的 durations)`，再加每条 segment 的 `start_<e>`。
5. append 到 `rows`；把每条 `trace` dict（前置 `scenario=m`）append 到 `trace_rows`；`done.add(m)`（`pretrain_milp.py:221-224`）。
6. **每个 scenario 结束都 checkpoint**（`pretrain_milp.py:225-227`）：写 `milp_optima.csv`、`milp_trace.csv`、
   `milp_progress.json`（`{"hash": fp, "done": sorted(done)}`）。
7. 打印 per-scenario 摘要行（`pretrain_milp.py:228-229`）。

**为什么每个 scenario 都写盘。** 全量枚举 UE 很慢，中途被打断也能从上次 checkpoint 接着跑，不用重算已完成的 scenario。

**写的文件（checkpoint，每个 scenario）：** `outputs/pretrain_milp/n{N}/milp_optima.csv`、`…/milp_trace.csv`、`…/milp_progress.json`。

### 11.4 最终聚合 + `run_meta.json`

**做了什么（`util/pretrain_milp.py:230-243`）。**
- `total_s = perf_counter() - t_all`（`pretrain_milp.py:230`）。
- 用完整 `rows`/`trace_rows` 重写 `milp_optima.csv` 和 `milp_trace.csv`（`pretrain_milp.py:232-234`）。
- `total_ue = int(milp_opt["ue_solves"].sum())`（`pretrain_milp.py:235`）。
- `meta = {N, M, T, segments, seed, total_time_s, mean_scenario_time_s, total_ue_solves,
  s_per_ue = total_s/max(1,total_ue), mean_iters, mean_F_milp}`（`pretrain_milp.py:236-240`）。
- **写** `outputs/pretrain_milp/n{N}/run_meta.json`（`pretrain_milp.py:241`）。

---

## 12. process figures + POST-HOC oracle 对比

### 12.1 process figures（fig 03/04）— `make_process_figures`

**做了什么（`util/pretrain_milp.py:245-246`）。**
```python
from viz.pretrain_viz import make_process_figures
make_process_figures(out_dir, pd.DataFrame(trace_rows), milp_opt, segments, T)
```
- `make_process_figures(...)`（`viz/pretrain_viz.py:111-172`），从逐轮 trace 出诊断图：
  - **03_optimization_process.png**——panel a：真实 `F` 的 best-so-far（`F.cummin()`）随 iteration 下降，每 scenario 一条线，
    并标出实际返回的最优 iterate（真实 F 最小点）；panel b：MILP surrogate（`Σ c y`）随 iteration（`pretrain_viz.py:124-149`）。
  - **04_runtime.png**——panel a：每 scenario 的 wall-clock（`time_s`）；panel b：迭代次数（`n_iter`）（`pretrain_viz.py:152-171`）。
  - 用 `viz.style` 的 `use_pub / save_pub / panel_label / C`（PNG @ 600 dpi）。
- **写的文件：** `…/figures/03_optimization_process.png`、`…/figures/04_runtime.png`。

### 12.2 POST-HOC oracle 对比（fig 01/02 + `milp_vs_oracle.csv`）

**这一块严格在 §11 之后运行——每个 scenario 的 MILP schedule 早已算完并 checkpoint。oracle 结果在这里只用于对比、
永远不影响 MILP 算法，没有泄露。**

**做了什么（`util/pretrain_milp.py:248-267`）。**
1. `oracle_dir = scale_dir(ROOT/"outputs"/"oracle")` → `outputs/oracle/n{N}/`（`pretrain_milp.py:248`）。
2. `oracle_opt_path = oracle_dir/"oracle_optima.csv"`（`pretrain_milp.py:249`）。
3. **若** `oracle_optima.csv` 存在（`pretrain_milp.py:250`）：
   - import `make_comparison, make_landscape`（`pretrain_milp.py:251`）。
   - `oracle_opt = pd.read_csv(oracle_optima.csv)`（`pretrain_milp.py:252`）。
   - 把 MILP optima 与 oracle 的 per-scenario `F`（改名 `F_oracle`）按 `scenario` left join（`pretrain_milp.py:253-254`）。
   - `merged["gap"] = merged["F_milp"] - merged["F_oracle"]`——**gap 为负 ⇒ MILP 打败了 work-conserving oracle**（`pretrain_milp.py:255`）。
   - **写** `…/milp_vs_oracle.csv`（`pretrain_milp.py:256`）。
   - `make_comparison(out_dir, merged, segments, T)`——**fig 01_milp_vs_oracle.png**（`viz/pretrain_viz.py:21-59`；
     panel a：per-scenario `F_oracle` vs `F_milp`；panel b：gap bar）（`pretrain_milp.py:257`）。
   - `make_landscape(out_dir, pd.read_csv(oracle_dir/"oracle_landscape.csv"), milp_opt, oracle_opt, segments, T)`——
     **fig 02_cross_scenario_landscape.png**（`viz/pretrain_viz.py:62-108`；固定单策略 schedule 按 mean `F` 排名，
     叠上 MILP adaptive mean 与 hindsight mean）（`pretrain_milp.py:258-259`）。
   - 打印 mean `F_milp` / mean `F_oracle` / mean gap（`pretrain_milp.py:260-263`）。
4. **否则**（该规模的 oracle 还没算好）：打印一句延迟提示，指向 `python -m util.pretrain_milp --landscape`（`pretrain_milp.py:264-266`）。
5. 打印 `Wrote {out_dir}`（`pretrain_milp.py:267`）。

- **post-hoc 读的文件：** `outputs/oracle/n{N}/oracle_optima.csv`、`outputs/oracle/n{N}/oracle_landscape.csv`。
- **写的文件：** `…/milp_vs_oracle.csv`、`…/figures/01_milp_vs_oracle.png`、`…/figures/02_cross_scenario_landscape.png`。

> `make_comparison`/`make_landscape`/`make_process_figures` 都调 `viz.style.use_pub()` 和 `save_pub()`
> （PNG @ 600 dpi；SVG/PDF 默认关，见 `viz/style.py:78-87`）。它们纯粹是出图，不回灌任何计算。

---

## 13. 完整 file-I/O map

| 方向 | 路径 | 在哪一步 | 章节 |
|---|---|---|---|
| read | `data/siouxfalls_toy/network/edges.csv` | `select_oracle_instance`、`load_toy_network` | §3.1, §4.2 |
| read | `data/siouxfalls_toy/network/od_pairs.csv` | `load_toy_network` | §4.2 |
| read | `data/siouxfalls_toy/network/nodes.csv` | `load_toy_network` | §4.2 |
| read | `data/siouxfalls_toy/raw/SiouxFalls_flow.tntp` | 仅 `util/ue.py:_validate` UE 自检读取；选实例流程改为内部 `solve_ue`（不读此文件） | §3.2 |
| write | `data/siouxfalls_toy/disruption/disrupted_segments_oracle{N}.csv` | `select_oracle_instance` | §3.1 |
| read (resume) | `outputs/pretrain_milp/n{N}/milp_optima.csv` | resume | §11.2 |
| read (resume) | `outputs/pretrain_milp/n{N}/milp_trace.csv` | resume | §11.2 |
| read (resume) | `outputs/pretrain_milp/n{N}/milp_progress.json` | resume | §11.2 |
| write (checkpoint) | `outputs/pretrain_milp/n{N}/milp_optima.csv` | 每 scenario | §11.3 |
| write (checkpoint) | `outputs/pretrain_milp/n{N}/milp_trace.csv` | 每 scenario | §11.3 |
| write (checkpoint) | `outputs/pretrain_milp/n{N}/milp_progress.json` | 每 scenario | §11.3 |
| write | `outputs/pretrain_milp/n{N}/run_meta.json` | 最终 | §11.4 |
| write | `outputs/pretrain_milp/n{N}/figures/03_optimization_process.png` | figs | §12.1 |
| write | `outputs/pretrain_milp/n{N}/figures/04_runtime.png` | figs | §12.1 |
| read (post-hoc) | `outputs/oracle/n{N}/oracle_optima.csv` | oracle 对比 | §12.2 |
| read (post-hoc) | `outputs/oracle/n{N}/oracle_landscape.csv` | oracle 对比 | §12.2 |
| write | `outputs/pretrain_milp/n{N}/milp_vs_oracle.csv` | oracle 对比 | §12.2 |
| write | `outputs/pretrain_milp/n{N}/figures/01_milp_vs_oracle.png` | oracle 对比 | §12.2 |
| write | `outputs/pretrain_milp/n{N}/figures/02_cross_scenario_landscape.png` | oracle 对比 | §12.2 |

---

## 附录 A — 用到的参数默认值（出自 `config.py`）

| 参数 | 默认值 | 在本 pipeline 里的作用 |
|---|---|---|
| `DELTA_T_H` | 3.0 | 小时/slot；在 F2 里约掉，只作为 fingerprint 的一部分 |
| `C_MAX` | 2 | `schedule_from_permutation` 与 MILP crew-cap 约束里的 crew 上限 |
| `MU` | 1.0 | `F = MU·F1 + (1-MU)·F2` ⇒ 目标只看 F1（F2 仅 logging） |
| `CAP_RETAIN` | {1:0.3, 2:0.1, 3:0.02} | `build_damaged_edges` 里的受损 capacity 乘子 |
| `SPEED_RETAIN` | {1:0.5, 2:0.3, 3:0.2} | `build_damaged_edges` 里的受损 free-flow-time 除子 |
| `SEVER_SEVERITY` | 3 | severity ≥ 此值 ⇒ edge 被移除（断连 → `u_pen`） |
| `F1_ACTIVE_ONLY` | False | F1 对整段 `[1,T]` 平均（与 `c_e^k` 的 `1/T` 一致） |
| `RHO` | 0.7 | recovery inertia；真实 demand 模型和 `c_e^k` 的 `(1−ρ^{n+1})` 都用它 |
| `KAPPA` | 1.0 | `B(Φ)` 里"损坏 → demand shortfall"的尺度 |
| `UPEN_FACTOR` | 10.0 | `u_pen = 10 × max 有限 baseline OD time` |
| `M_SCENARIOS` | 10 | duration scenario 个数（循环长度） |
| `SEED` | 42 | `sample_scenarios` 的 RNG seed |
| `N_DISRUPTED_ORACLE` | 4 | 实例规模；决定 `n{N}` 输出目录 |
| `UE_RGAP` | 1e-6 | 单步 UE 的 relative-gap 目标 |
| `UE_MAX_ITER` | 100 | 单步 UE 的 max Frank-Wolfe 迭代数 |
| `MILP_MAX_ITER` | 20 | 交替循环的硬上限（停止条件 ③） |
| `MILP_CYCLE_TOL` | 3 | 放宽的 cycle guard（停止条件 ②，基于重复计数） |
| `MILP_DAMPING` | 0.5 | MSA travel-time under-relaxation `u=damp·u_new+(1-damp)·u_prev` |
| `DURATION_SUPPORT` | Table 1（见 `config.py:51-55`） | 按 (class, severity) 的 base duration support 集合 |
| `ETA` | [0.8, 1.0, 1.2] | `sample_scenarios` 的 crew-efficiency 乘子 |

## 附录 B — 共享问题定义 vs post-hoc 读 oracle（刻意为之，无泄露）

与 oracle 的"共享"有两类，只有一类在算法上游：

1. **共享问题定义（刻意、上游）。** MILP run 和 oracle run 用**同一批**函数来定义 instance 和 objective：
   `select_oracle_instance`（相同受损 segment + severity）、`sample_scenarios`（同 seed 下相同 durations）、
   `compute_horizon`（相同 `T`）、`evaluate_schedule`（相同真实 `F(x|ω)`）。这是故意的，好让 `F_milp` 与 `F_oracle`
   直接可比。**这不是信息泄露——MILP 搜索过程从不查阅任何 oracle 的"解"。**

2. **post-hoc 读 oracle（仅对比、下游）。** oracle 的最优 `F` 和 landscape 只在 §12.2 被读，
   那时每个 scenario 的 MILP schedule 早已算完并 checkpoint 进 `milp_optima.csv`。它们只喂给对比 CSV
   （`milp_vs_oracle.csv`）和图 01/02。MILP 的目标（`−Σ c_e^k y_e^k`）与 `c_e^k` 系数**只依赖于该 schedule 自身迭代
   产生的冻结 UE travel time**，从不依赖 oracle。因此无论 oracle 有没有算过，MILP 结果都一模一样
   （若没算，run 只是把对比图延迟）。

**为什么 gap 可能为负。** oracle 的"最优"只是**最优的 work-conserving schedule**（列表排程、不留空档）；而 MILP 的可行域
是**所有**满足约束的开工时间表（允许 crew 空等），是 work-conserving 集合的**超集**。所以 MILP 完全可能找到一个更好的、
非 work-conserving 的解，使 `gap = F_milp − F_oracle < 0`——这不是 bug，而是 MILP 搜索空间更大的自然结果。
`level_a`（`util/pretrain_milp.py:274-311`）正是用来做这个 encoding sanity check：在固定 `c` 下，MILP surrogate 最优值
必须 ≥ 最优 work-conserving schedule 的 surrogate 值（因为前者的可行域包含后者），否则说明 MILP 编码有 bug。
