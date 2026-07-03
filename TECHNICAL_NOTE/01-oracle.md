# 01 — `run_oracle()`：完整执行逻辑

**这份文档在做什么。** 它逐步追踪 `run_oracle()`（在 `util/oracle.py`）的完整执行逻辑：为了得到一个"上帝视角"（oracle）的 ground-truth，需要按什么顺序做哪几步、每一步为什么这么做、关键产物是什么、又被后面哪一步用到。**本文只讲逻辑步骤**（"要解决什么问题 → 做了什么 → 为什么 → 产物给谁用"），不是变量之间的 dataflow 对照表，也不是 API 手册。文中会引用源码行号（如 `util/oracle.py:119`）方便查阅，technical term 与变量名/函数名/路径保留英文原样。

**这个 oracle 到底解什么问题。** 对一批被摧毁的路段（disrupted segments），枚举**所有** work-conserving 的修复 schedule（也就是这些路段的所有 permutation），在 `M` 个随机抽出的 duration scenario 下，对每个 schedule 计算**精确的** Figure-1 目标 `F(x|ω)`。最终产出两样东西：(a) 每个 scenario 下的事后最优（hindsight optimum）`F*`，以及 (b) 完整的 landscape（每一个被测过的 `x` 及其 `F/F1/F2`）。这就是日后用来验证 pretraining MILP 的 ground-truth（Level B — 用真实目标验证，见附录 B）。

**入口。** `util/oracle.py :: run_oracle()`（定义在 `util/oracle.py:119`）。

**运行方式。**
```
python -m util.oracle --probe     # 只测 s/UE + 估算总运行时间，然后停
python -m util.oracle             # 完整枚举 + 出图
python -m util.oracle --force     # 忽略 cache / 部分 checkpoint，强制重算
python -m util.oracle --figs      # render_figs()：从已存 CSV 重画图（独立路径，不重算）
```
`__main__` 块（`util/oracle.py:242-246`）把 `--figs` 路由到 `render_figs()`，否则调用 `run_oracle(probe=…, force=…)`。

---

## 目录
1. [Call-tree 总览](#1-call-tree)
2. [模块常量与 fingerprint 机制](#2-fingerprint)
3. [Step 0 — 输出目录 + figures 目录](#step0)
4. [Step 1 — `select_oracle_instance`：选定 disruption set](#step1)
5. [Step 2 — `build_context`：网络 / index maps / baseline / B(Φ) / u_pen / severity](#step2)
6. [Step 3 — `sample_scenarios`：抽 duration scenario](#step3)
7. [Step 4 — permutations](#step4)
8. [Step 5 — `compute_horizon`：全局 horizon T](#step5)
9. [Step 6 — cache 检查（meta.json）](#step6)
10. [Step 7 — probe timing](#step7)
11. [Step 8 — 从 checkpoint resume](#step8)
12. [Step 9 — 主枚举循环 + `evaluate_schedule`](#step9)
13. [Step 10 — landscape 排序 + per-scenario 最优](#step10)
14. [Step 11 — summary.txt](#step11)
15. [Step 12 — figures（`make_figures`）](#step12)
16. [Step 13 — meta.json 写入 + checkpoint 清理](#step13)
17. [附录 A — 参数默认值表](#appA)
18. [附录 B — validation levels 与 comparison metrics](#appB)
19. [附录 C — Figure-1 fidelity 与建模 caveats](#appC)

---

## 1. Call-tree 总览 <a id="1-call-tree"></a>

下面这棵树是整篇文档的骨架——它按执行顺序把 `run_oracle` 会走到的每一层调用都摊开，后文各 Step 逐个展开讲"为什么"。

```
run_oracle()                                    util/oracle.py:119
├─ scale_dir(out_dir)                            util/oracle.py:57      → outputs/oracle/n{N}/
├─ (out_dir/"figures").mkdir(...)
├─ select_oracle_instance(toy, N)                util/oracle.py:83      （注意：已无 seed 参数）
│   └─ _reference_twoway_flow(toy)               util/oracle.py:63      读 raw/SiouxFalls_flow.tntp
│      · 读 network/edges.csv                     (pd.read_csv)
│      · 写 disruption/disrupted_segments_oracle{n}.csv
├─ segments = sorted(edge_id)
├─ build_context(toy, disrupted)                 util/evaluate.py:106
│   ├─ load_toy_network(toy)                      util/io.py:10          读 edges.csv, od_pairs.csv, nodes.csv
│   ├─ solve_ue(edges, M(H0), zone_ids)          util/ue.py:81          在“未受损”网络上做 UE → baseline
│   │   ├─ _build_graph(edges, zone_ids)          util/ue.py:46          AequilibraE Graph
│   │   ├─ _build_matrix(M, zone_ids)             util/ue.py:71          AequilibraeMatrix
│   │   └─ TrafficAssignment.execute()            (AequilibraE bi-conjugate Frank-Wolfe)
│   ├─ _matrix_from_H(H0, ctx)                    util/evaluate.py:48
│   ├─ od_travel_times(base_links, ctx)           util/evaluate.py:27    networkx single-source dijkstra
│   └─ 构建 B(Φ), u_pen, severity_vec             (在 free-flow 图上跑 networkx shortest_path)
├─ sample_scenarios(disrupted, M, seed)          util/scenarios.py:14   在 DURATION_SUPPORT × ETA 上抽签
├─ perms = itertools.permutations(segments)
├─ compute_horizon(segments, scenarios)          util/oracle.py:109
│   └─ 对每个 perm × scenario:
│       schedule_from_permutation(perm, dur)      util/evaluate.py:82
│       makespan_slot(start, dur)                 util/evaluate.py:94
├─ _param_fingerprint()                          util/oracle.py:44      对 FINGERPRINT_PARAMS 做 sha1
├─ [cache] meta.json 的 hash 匹配 → 读 CSV, make_figures, RETURN         util/oracle.py:134-145
├─ [probe] 评一个 schedule，估算总时间；若 --probe 则 RETURN               util/oracle.py:148-156
├─ [resume] 读 landscape_progress.json + 部分 oracle_landscape.csv        util/oracle.py:159-169
├─ 主循环  for m,dur in scenarios: for perm in perms:                     util/oracle.py:174-192
│   ├─ schedule_from_permutation(perm, dur)       util/evaluate.py:82
│   ├─ evaluate_schedule(start, dur, T, ctx)      util/evaluate.py:156   ← 精确目标 F(x|ω)
│   │   ├─ f2_value(start, dur)                    util/evaluate.py:98    (makespan_slot util/evaluate.py:94)
│   │   └─ 逐 slot 循环 k=1..T:
│   │       ├─ demand shortfall  D=max(B·v, ρD); H=clip(H0−D)
│   │       ├─ build_damaged_edges(ctx, damaged)  util/evaluate.py:54
│   │       ├─ _matrix_from_H(H, ctx)             util/evaluate.py:48
│   │       ├─ solve_ue(dmg_edges, M(H), ...)      util/ue.py:81          每个 slot 一次 UE
│   │       └─ od_travel_times(links, ctx)         util/evaluate.py:27
│   ├─ append row(scenario,perm,F,F1,F2,eval_s,start_e…)
│   └─ per-scenario checkpoint: 写 oracle_landscape.csv + landscape_progress.json
├─ land.sort_values(["scenario","perm"]); 写 oracle_landscape.csv
├─ opt = land.groupby("scenario")["F"].idxmin(); 写 oracle_optima.csv    util/oracle.py:197-198
├─ 写 summary.txt                                 util/oracle.py:200-211
├─ make_figures(...)                              viz/oracle_viz.py:23   figs 01/02/03
├─ 写 meta.json (hash+params+timing)              util/oracle.py:216-222
└─ prog_path.unlink()                             util/oracle.py:223    删掉 resume 标记
```

**关键观察：整份计算的成本几乎全在 UE 上。** 只有两处调用真正跑 AequilibraE UE：(1) `build_context` 里一次 baseline UE；(2) `evaluate_schedule` 里**每个 slot 一次** UE。其余全是纯算术 + networkx 最短路。一次完整运行的 UE 求解次数 ≈ `1 + (perms × M × T)`（再加 probe 的 1 次和 baseline 的 1 次）。这也是为什么整套设计要围着"能不能省掉一次 UE 枚举"来做 cache / resume。

---

## 2. 模块常量与 fingerprint 机制 <a id="2-fingerprint"></a>

在讲 Step 0 之前，先说清楚两件贯穿全程的基础设施：**路径常量**和 **fingerprint**。它们决定了"结果存到哪、什么时候可以复用、什么时候必须重算"。

**路径常量**（`util/oracle.py:31-33`）：
- `ROOT` = 仓库根目录（`util/` 的上一级）。
- `TOY = ROOT/data/siouxfalls_toy` — 默认的 `toy_dir`。
- `OUT = ROOT/outputs/oracle` — 默认的 `out_dir`（加 scale 后缀之前）。

**为什么需要 fingerprint。** 一次完整枚举很贵（几十上百次 UE），所以希望"参数没变就直接复用上次结果"。问题是"参数没变"要能被精确判定。做法是：把所有会影响 `F` 的 config 参数打包成一个 JSON，做 sha1，得到一个短哈希（fingerprint）。这个哈希写进 `meta.json`；下次运行先算当前 fingerprint，和 `meta.json` 里存的比——一样就复用，不一样就重算。

**`FINGERPRINT_PARAMS`**（`util/oracle.py:37-41`）——被纳入 fingerprint 的 config 属性清单：
```
N_DISRUPTED_ORACLE, MU, CAP_RETAIN, SPEED_RETAIN, SEVER_SEVERITY,
F1_ACTIVE_ONLY, RHO, KAPPA, UPEN_FACTOR, DELTA_T_H, C_MAX,
M_SCENARIOS, SEED, UE_RGAP, UE_MAX_ITER, DURATION_SUPPORT, ETA
```
注意这里有个两层设计：**`N_DISRUPTED_ORACLE` 单独充当 cache 文件夹的 key**（`n{N}/`，经由 `scale_dir`），而**整个清单（含 `N_DISRUPTED_ORACLE`）** 才是那个文件夹内部的"新鲜度哈希"。这样不同 `N` 的结果永远存在不同文件夹、互不覆盖；同一个 `N` 下再看别的参数有没有变。

### `_param_fingerprint()` — `util/oracle.py:44-54`
这一步要解决的问题：把 config 里那堆参数稳定地序列化成一个哈希。做法：
- 遍历 `FINGERPRINT_PARAMS`，用 `getattr(P, name)` 逐个取值（`import config as P`，参数现在都在根目录的 `config.py`）。
- 对任何 `dict` 型参数（`CAP_RETAIN`、`SPEED_RETAIN`、`DURATION_SUPPORT`），把 key 转成字符串：`{str(k): v[k] …}`。**为什么必须这么做**：`DURATION_SUPPORT` 的 key 是 tuple（如 `("local", 1)`），JSON 无法把 tuple 当作 key 序列化，不转就会报错。
- `blob = json.dumps(values, sort_keys=True)`（`sort_keys` 保证同样的参数总产生同样的字符串）。
- 返回 `(values, hashlib.sha1(blob.encode("utf-8")).hexdigest())`。
- **产物给谁用**：`values` 会原样写进 `meta.json`（便于人读），`fp`（哈希）同时用于 cache 检查（`meta.json["hash"]`）和 resume 标记（`landscape_progress.json["hash"]`）。

### `scale_dir(base=OUT, n=None)` — `util/oracle.py:57-60`
- `n` 缺省取 `P.N_DISRUPTED_ORACLE`。
- 返回 `Path(base)/f"n{n}"`，例如 `outputs/oracle/n4/`。作用是把不同规模（不同 `N`）的结果隔离到不同子目录，谁也不会覆盖谁。这是 `run_oracle` 进来做的**第一件事**（`util/oracle.py:120`）。

---

## Step 0 — 输出目录 + figures 目录 <a id="step0"></a>
`util/oracle.py:120-121`
```python
out_dir = scale_dir(out_dir)                 # outputs/oracle/n{N}/
(out_dir / "figures").mkdir(parents=True, exist_ok=True)
```
**要解决的问题**：先把本次规模对应的落地目录准备好。**做了什么**：把 `out_dir` 换成带 scale 后缀的 `outputs/oracle/n{N}/`，并建好其下的 `figures/`（含父目录）。**为什么**：后面所有 CSV、meta.json、图都往这里写；先隔离到 `n{N}/` 保证不同规模不打架。**用到的 config**：`N_DISRUPTED_ORACLE`（经 `scale_dir`）。

---

## Step 1 — `select_oracle_instance`：选定 disruption set <a id="step1"></a>
`util/oracle.py:123` → `select_oracle_instance(toy_dir, P.N_DISRUPTED_ORACLE)`，定义在 `util/oracle.py:83-106`。

> **与旧文档的差异（已更新）**：这个函数**现在没有 `seed` 参数**了。当前签名是 `def select_oracle_instance(toy_dir, n=P.N_DISRUPTED_ORACLE):`，`run_oracle` 也是按 `select_oracle_instance(toy_dir, P.N_DISRUPTED_ORACLE)` 调用（`util/oracle.py:123`）。选择过程本来就是完全确定性的（由 flow 决定），旧的 `seed` 参数是多余的，已删掉。

**这一步要解决什么问题。** 得先决定"哪些路段被摧毁、各自多严重"。这批 disruption 不能随便选——如果所有路段重要性差不多，那修复顺序（permutation）对 `F1` 的影响就很弱，枚举出来的 landscape 会平得看不出优劣，起不到验证 ground-truth 的作用。所以要**故意混搭关键路与次要路**，让 severity 和位置差异足够大，从而"先修哪条"这件事对目标值有强影响。

**为什么按 flow 排重要性。** 用 baseline two-way UE flow 当"重要性"代理：流量越大的边，一旦断掉对 accessibility 的冲击越大。

**逐步逻辑：**
1. `edges = pd.read_csv(toy/"network"/"edges.csv")`（`util/oracle.py:89`）。列：`edge_id,u,v,capacity,length,free_flow_time,bpr_alpha,bpr_beta,road_class`。
2. `flow = _reference_twoway_flow(toy)` — 见下面小节。
3. 给每条边贴一个 `flow` 列：按 `(min(u,v), max(u,v))` 无向 key 去查流量，查不到默认 `0.0`（`util/oracle.py:91-92`）。
4. `ranked = edges.sort_values("flow", ascending=False)` — 流量最大的边排最前（`util/oracle.py:93`）。
5. **挑索引**（`util/oracle.py:95-100`）：
   - `n_crit = min(2, n)` → 流量最高的前 2 条定为 "critical"。
   - `picks = [0, 1, …, n_crit-1]`（最高流量那几名）。
   - `rest = n - n_crit`。若 `rest>0`，把剩下的名额用 `np.linspace(len(ranked)//5, len(ranked)-1, rest)` 取整索引，铺在**较低流量**的边上（从 20% 分位一路到最不常用的边）。这样额外挑出来的都是刻意选的低流量"次要"边。
   - `sub = ranked.iloc[picks]`。
6. **分配 severity**（`util/oracle.py:101`）：
   - `sub["severity"] = [3]*n_crit + [2 if i%2==0 else 1 for i in range(rest)]`。
   - 也就是：**前 2 条高流量边给 severity 3**（因为 `SEVER_SEVERITY=3`，它们在受损网络里会被**整条移除**，制造真正的断连）；剩下的边 severity 在 **2, 1, 2, 1, …** 之间交替。
7. `sub["level_id"] = road_class + "-S" + severity`（如 `highway-S3`）（`util/oracle.py:102`）。
8. 选取并按 `edge_id` 排序输出列 `[edge_id,u,v,road_class,severity,level_id]`（`util/oracle.py:103-104`）。
9. **写** `toy/"disruption"/f"disrupted_segments_oracle{n}.csv"`（`util/oracle.py:105`），返回该 DataFrame。

**关键产物给谁用**：返回的 `disrupted` DataFrame 会喂给 Step 2（`build_context`）、Step 3（`sample_scenarios`）和 Step 12（`make_figures`）；同时侧写一份 CSV 到 `disruption/`。**用到的 config**：`N_DISRUPTED_ORACLE`（作为 `n`）；`SEVER_SEVERITY` 这里不直接读，但它决定了"severity=3 意味着整条移除"这个下游含义。

### `_reference_twoway_flow(toy)` — `util/oracle.py:63-80`
**要解决的问题**：给上面的"重要性排序"提供每条无向边的 baseline 流量。数据源是开源 Sioux Falls 的参考 UE 解 `raw/SiouxFalls_flow.tntp`。逐步：
- 逐行读该文件。
- 跳过空行和表头（以 `from` 开头的行，大小写不敏感）（`util/oracle.py:69`）。
- 去掉行尾 `;`、按空白切分；至少要 3 个 token（`util/oracle.py:71-73`）。
- 解析 `a,b = int`、`vol = float`（对应 From, To, Volume 三列）；`ValueError` 就跳过。
- `key = (min(a,b), max(a,b))`；**把两个有向流量累加进同一个无向 key**：`f[key] += vol`（`util/oracle.py:78-79`）。为什么合并两向：网络是无向边，重要性看的是这条边总承载多少车。
- 返回 `{(min,max): 两向合计流量}`。只读 `raw/SiouxFalls_flow.tntp` 这一个文件。

回到 `run_oracle`：`segments = sorted(int(e) for e in disrupted["edge_id"])`（`util/oracle.py:124`）——被调度的 edge id 的规范排序列表，后面 permutation、horizon、landscape 列名都以它为准。

---

## Step 2 — `build_context`：网络 / index maps / baseline / B(Φ) / u_pen / severity <a id="step2"></a>
`util/oracle.py:125` → `ctx = build_context(toy_dir, disrupted)`，定义在 `util/evaluate.py:106-150`。

**这一步整体要解决什么问题。** 后面要对 `perms × M × T` 这么多次评估反复调用同一套"静态背景"——网络结构、正常需求、基准通行时间、需求受损的敏感度矩阵、断连惩罚等等。这些东西和具体 schedule 无关，只需**算一次**，塞进一个 `ctx` dict 里，之后所有 scenario × permutation 复用。`build_context` 就是把这套背景一次性备好。

### 2a. 载入静态网络 —— `load_toy_network`（`util/io.py:10-23`）
`edges, od, zone_ids = load_toy_network(toy_dir)`（`util/evaluate.py:108`）：
- 读三个 CSV（`util/io.py:19-22`）：
  - `network/edges.csv` → `edges`（无向边表，列同上）。
  - `network/od_pairs.csv` → `od`（`od_id,origin,destination,h0`）。
  - `network/nodes.csv` → `nodes`；`zone_ids = nodes["node_id"]` 整型数组（**Sioux Falls 里每个节点都是一个 OD zone**）。
- 返回 `(edges, od, zone_ids)`。

### 2b. Index maps（`util/evaluate.py:109-116`）
**为什么需要**：UE 求解、矩阵构造都要按"稠密下标"操作，所以先把 node id / OD pair 映射到 0-based 下标，避免后面反复查找。
- `zone_pos = {node_id: 稠密下标}`。
- `od_pairs = [(origin, destination), …]`，逐 `od` 行取出。
- `H0 = od["h0"]` —— 对齐 `od_pairs` 的**正常时期需求向量**（也是"完全恢复"的目标水平）。
- `oi = [zone_pos[o]]`、`di = [zone_pos[d]]` —— 每个 OD pair 的稠密行/列下标（给 `_matrix_from_H` 用）。
- `edge_row = {edge_id: 在 edges 里的行下标}`。
- `eid_of = {(min(u,v),max(u,v)): edge_id}` —— 无向 key → edge id。
- 组装出基础 `ctx`：`edges, zone_ids, od_pairs, H0, oi, di, nz=len(zone_ids), edge_row, origins_unique=sorted(去重后的 origin)`（`util/evaluate.py:118-120`）。

### 2c. Baseline `u_r^{t0}` —— 在“未受损”网络上做 UE（`util/evaluate.py:122-125`）
```python
base_links, _ = solve_ue(edges, _matrix_from_H(H0, ctx), zone_ids,
                         rgap=P.UE_RGAP, max_iter=P.UE_MAX_ITER, quiet=True)
ctx["baseline_u"] = od_travel_times(base_links, ctx)
```
**要解决的问题**：`F1` 是"实际通行时间 / 基准通行时间"的比值，需要一个基准分母——正常情况下（网络完好、需求正常）每个 OD pair 的拥堵通行时间。**做了什么**：在原始 edge 表 + 正常需求 `H0` 上跑一次完整 UE，再由 OD 最短路得到每个 OD pair 的基准通行时间。
- `_matrix_from_H(H0, ctx)`（`util/evaluate.py:48-51`）：建一个 `nz×nz` 全零稠密矩阵，把需求散布到 OD 单元 `M[oi, di] = H0`。
- `solve_ue(...)` 在**完好**网络上做一次 user-equilibrium 分配（内部细节见 §2e）。`quiet=True` 屏蔽 AequilibraE 的日志和 stdout/stderr。
- `od_travel_times(base_links, ctx)`（§2f）→ `baseline_u[i]` = 均衡链路成本下，每个 OD pair `i` 的最短路拥堵通行时间。**产物给谁用**：它是 `F1` 的分母参考（后文 `base_u`），也是 `u_pen` 的来源。

### 2d. `u_pen` —— 断连 OD 的惩罚值（`util/evaluate.py:126`）
```python
ctx["u_pen"] = P.UPEN_FACTOR * float(np.nanmax(baseline_u[np.isfinite(baseline_u)]))
```
**要解决的问题**：受损网络里可能有些 OD pair 被彻底断连，最短路是 `inf`，没法直接进 `F1`。**做法**：给断连的 OD 记一个大惩罚 = `UPEN_FACTOR × (最大有限 baseline OD 通行时间)`。`UPEN_FACTOR = 10.0`，即断连相当于罚 10 倍最糟正常出行。这样"把关键边拖着不修导致断连"会在目标里受到重罚，从而让修复顺序真正影响 `F`。

### 2e. `solve_ue` 内部追踪（AequilibraE bi-conjugate Frank-Wolfe）— `util/ue.py:81-121`
`solve_ue(edges, od_matrix, zone_ids, algorithm="bfw", max_iter, rgap, quiet)`：
1. **`_build_graph(edges, zone_ids)`**（`util/ue.py:46-68`）：把无向 edge 表转成 AequilibraE 能路由的图。构一个 link 表：`link_id=1..n`、`a_node=u`、`b_node=v`、`direction=0`（双向）、`distance=length`、`modes="c"`、`capacity_ab=capacity_ba=capacity`、`free_flow_time`、`b=bpr_alpha`（BPR α）、`power=bpr_beta`（BPR β）。建 `Graph()`，设 `g.network`，`g.prepare_graph(zone_ids)`（所有节点都是 zone），`g.set_graph("free_flow_time")`（成本字段），`g.set_blocked_centroid_flows(False)`（允许过境交通）。返回 `(g, net)`。
2. **`_build_matrix(M, zone_ids)`**（`util/ue.py:71-78`）：把稠密 OD 矩阵包成内存里的 `AequilibraeMatrix`，只有一个 core `"demand"`；`computational_view(["demand"])`。
3. 建 `TrafficClass("car", graph, mat)`；`TrafficAssignment()`；`set_classes`；`set_vdf("BPR")`；`set_vdf_parameters({"alpha":"b","beta":"power"})`；`set_capacity_field("capacity")`；`set_time_field("free_flow_time")`；`set_algorithm("bfw")`；`assig.max_iter=max_iter`；`assig.rgap_target=rgap`（`util/ue.py:92-101`）。
4. **`assig.execute()`**（`util/ue.py:106` 在重定向下，或 `:108`）：跑 Frank-Wolfe 循环。**UE 的核心思想**（见模块 docstring，`util/ue.py:12-28`）：求解 Beckmann 目标 `Z(x)=Σ_a ∫₀^{x_a} t_a(w)dw`，其中 BPR 成本 `t_a(x)=t0_a·(1+α·(x_a/c_a)^β)`。每次 FW 迭代：(1) **all-or-nothing** —— 按当前最短路把每个 OD 出行整块分上去 → 辅助流量 `y`；(2) **line search / move** `x ← x + step·(y−x)`，令 `Z` 下降；(3) **更新成本**，当**相对 gap** < `rgap` 时停止。`bfw` 是 bi-conjugate Frank-Wolfe 变体，比朴素 FW 收敛更快。
5. `res = assig.results()` —— 每条链路的 AB/BA 均衡流量 + 拥堵时间。每条链路发出两行（AB 和 BA），列 `from,to,volume,cost`（`cost` 取自 `Congested_Time_AB/BA`）。丢掉 `volume` 为 NaN 的行。返回 `(flows_df[from,to,volume,cost], assig)`（`util/ue.py:110-121`）。

**每次 UE 调用用到的 config**：`UE_RGAP`（1e-6）、`UE_MAX_ITER`（100）。BPR α/β 和 capacity 来自（可能已受损的）edge 表。

### 2f. `od_travel_times` 内部追踪（networkx dijkstra）— `util/evaluate.py:27-45`
`od_travel_times(link_df, ctx)`：**要解决的问题**——UE 给的是逐链路的拥堵时间，但 `F1` 要的是逐 OD pair 的通行时间；所以在这些链路成本上跑最短路。
- 用 `link_df[from,to,cost]` 建 `nx.DiGraph`，每行加一条有向边、`weight=cost`（`util/evaluate.py:31-35`）。
- `u = np.full(len(od_pairs), np.inf)`（`util/evaluate.py:36`）。
- 对图中出现的每个唯一 origin `o`，算 `nx.single_source_dijkstra_path_length(G, o, weight="weight")` → 到所有可达节点的最短成本 dict，缓存进 `by_origin[o]`（`o` 不在图里则空 dict）（`util/evaluate.py:37-42`）。**为什么按 origin 缓存**：多个 OD pair 常共享 origin，单源 Dijkstra 一次算完到所有目的地，省去重复。
- 对每个 OD pair `i=(o,d)`：`u[i] = by_origin[o].get(d, inf)`（`util/evaluate.py:43-44`）。
- 返回 `u`（O、D 断连处为 `np.inf`）。

### 2g. B(Φ) 需求 shortfall 矩阵（`util/evaluate.py:129-148`）
**要解决的问题**：灾后需求会掉，且掉多少要和"哪条被摧毁的边落在这个 OD 的路径上、有多严重"挂钩。`B` 就是"某段受损 → 某 OD 需求 shortfall"的敏感度矩阵，`build_context` 里预先算好，逐 slot 用 `B @ v_vec` 得到当前 shortfall。
- `dis = [(edge_id, u, v, severity) …]` 为 disrupted 集合，存进 `ctx["disrupted"]`。
- 建一个**无向 free-flow 图** `Gff`：每条边加**两个方向**，`weight=free_flow_time`，并带属性 `eid`（`util/evaluate.py:132-135`）。
- `B = zeros(len(od_pairs), len(dis))`；`col = {edge_id: 列号 j}`，把每条受损边映到 `B` 的一列（`util/evaluate.py:136-137`）。
- 对每个 OD pair `i=(o,d)`：算**free-flow 最短路** `nx.shortest_path(Gff, o, d, weight="weight")`；`NetworkXNoPath` 就跳过（`util/evaluate.py:138-142`）。为什么用 free-flow 路径：这是"正常时期该 OD 主要走哪条路"的稳定代理，不随拥堵变化。
- 收集该路径上的无向边 key（`on_path`）；对每条，若它映到某个在 `col` 里的受损 `eid`，则置 `B[i, col[eid]] = P.KAPPA * (H0[i]/3.0)`（`util/evaluate.py:143-147`）。
- 含义：`B[r,e] = κ·(h_r^{t0}/3)·1{边 e 在 OD r 的 free-flow 最短路上}`，即"段 e 受损对 OD r 需求 shortfall 的敏感度"。
- `ctx["B"] = B`（`util/evaluate.py:148`）。

### 2h. `severity_vec`（`util/evaluate.py:149`）
- `ctx["severity_vec"] = np.array([sev for (_,_,_,sev) in dis], float)` —— 每条受损边的 severity，按 `dis` 顺序对齐。注意：`evaluate_schedule` 虽然会从 `ctx["severity_vec"]` 读出 `sev`，但在 `F` 的计算里并不真正用它——逐 slot 的实时 severity 直接来自 `dis` 里的 tuple。

**`build_context` 的产物**：一个 `ctx` dict，含 key `edges, zone_ids, od_pairs, H0, oi, di, nz, edge_row, origins_unique, baseline_u, u_pen, disrupted, B, severity_vec`。**读的文件**：`edges.csv, od_pairs.csv, nodes.csv`。**用到的 config**：`UE_RGAP, UE_MAX_ITER, UPEN_FACTOR, KAPPA`。**给谁用**：整个 `ctx` 被 Step 9 的每一次 `evaluate_schedule` 复用，是全程唯一一份静态背景。

---

## Step 3 — `sample_scenarios`：抽 duration scenario <a id="step3"></a>
`util/oracle.py:126` → `scenarios = sample_scenarios(disrupted, M, seed)`，定义在 `util/scenarios.py:14-28`。

**这一步要解决什么问题。** 修复每条边要多少个 slot 是不确定的——同一 disruption，不同的现实（crew 效率、损伤实际严重度）会给出不同工期。oracle 要在**一批** duration scenario 上评估，才能看出某个 schedule 是否稳健、以及事后最优随 scenario 怎么变。这一步就是抽 `M` 个这样的 scenario。

**为什么按 level 抽而不是按边抽。** 工期是按 **level** `(road_class, severity)` 抽的；同一 scenario 内、同一 level 的所有受损段共享一个工期（论文里的 `d_e(ω)=d_{ℓ(e)}(ω)`）。这样工期的随机性挂在"路类×严重度"上，符合建模设定。

**逐步逻辑：**
1. `rng = np.random.default_rng(seed)`（`util/scenarios.py:17`）—— seed = `P.SEED` = 42，序列确定。
2. `levels = sorted({(road_class, severity)})`，对 disrupted 行去重（`util/scenarios.py:19`）。
3. 对 `M` 个 scenario 各做一遍（`util/scenarios.py:21-26`）：
   - 对每个 level：`base = int(rng.choice(P.DURATION_SUPPORT[lvl]))`（该 `(class,severity)` 在 Table 1 的一个 slot 数）；`eta = float(rng.choice(P.ETA))`（crew 效率乘子 0.8/1.0/1.2）；`lvl_dur[lvl] = max(1, int(round(base*eta)))`。
   - 落到边：`scenario[edge_id] = lvl_dur[(road_class, severity)]`，遍历每条 disrupted 行（`util/scenarios.py:27`）。
4. 返回长度 `M` 的 `list[dict]`：`scenario[m][edge_id] = 工期（slots）`。

**产物给谁用**：`scenarios` 喂给 Step 5（`compute_horizon`）、Step 7（probe）、Step 9（主循环）和 Step 12（figures）。**用到的 config**：`M_SCENARIOS`, `SEED`, `DURATION_SUPPORT`, `ETA`。**关于 RNG 顺序**：每个 level 两次 `rng.choice`、外层按 sorted levels × M 迭代，给定 seed 就完全确定每个 scenario——所以可复现，也因此这几个参数进了 fingerprint。

---

## Step 4 — permutations <a id="step4"></a>
`util/oracle.py:127` → `perms = list(itertools.permutations(segments))`。

**要解决的问题**：把"所有可能的修复顺序"穷举出来。`N` 条受损边的每一种排列就是一个修复**优先级顺序**，喂给 work-conserving list scheduling 生成具体 schedule。`N=4` → `4! = 24` 个 permutation。**为什么可以只枚举 permutation**：见附录 C 的 work-conserving 归约——假设 idling 永远不划算，于是最优 schedule 一定对应某个优先级排列，把搜索空间压到 `|ℰ|!`。

---

## Step 5 — `compute_horizon`：全局 horizon T <a id="step5"></a>
`util/oracle.py:128` → `T = compute_horizon(segments, scenarios)`，定义在 `util/oracle.py:109-116`。

**这一步要解决什么问题。** `F1` 是逐 slot 项在 `[1,T]` 上的平均。如果每个 schedule 各用各的 horizon，`F1` 就不可比。所以要取一个**全局 T** = 所有 permutation × scenario 里的最大完工 slot，保证每个被枚举的 schedule 都在 `T` 内完工，且大家共享同一 horizon。

**逐步逻辑**（`util/oracle.py:112-116`）：
```python
T = 0
for perm in itertools.permutations(segments):
    for dur in scenarios:
        T = max(T, makespan_slot(schedule_from_permutation(list(perm), dur), dur))
return T
```
对每个 `(perm, scenario)` 生成 schedule、取其 makespan slot，全程取 max。

### `schedule_from_permutation(perm, durations, c_max=P.C_MAX)` — `util/evaluate.py:82-91`
把一个优先级排列变成具体开工时刻，用 `c_max` 台完全相同的 crew 做 work-conserving list scheduling：
- `crew_free = [1]*c_max` —— 每台 crew 从 slot 1 起空闲（严格在 onset slot 0 之后开工）。
- 对优先级顺序里的每条边 `e`：挑最早空闲的 crew `c = argmin(crew_free)`；`start[e] = crew_free[c]`；把该 crew 占到完工：`crew_free[c] = start[e] + durations[e]`。
- 返回 `{edge_id: start_slot}`。**没有 idling**——每台 crew 一空下来立刻接下一条边。
- **用到的 config**：`C_MAX`（2）。

### `makespan_slot(start, durations)` — `util/evaluate.py:94-95`
- `max(start[e] + durations[e] for e in start)` —— 最后完工那条边的完工 slot。

**产物给谁用**：整数 `T` 给 Step 7（probe）、Step 9（`evaluate_schedule`）、Step 12（figures）和 timing 用。随后 `print(...)` 打一行实例摘要（`util/oracle.py:129`）：段数、`perms`、`M`、`T`。

---

## Step 6 — cache 检查（meta.json） <a id="step6"></a>
`util/oracle.py:132-145`。
```python
values, fp = _param_fingerprint()
meta_path = out_dir / "meta.json"
if not probe and not force and meta_path.exists():
    cached = json.loads(meta_path.read_text(...))
    if cached.get("hash") == fp:
        # cache 命中
        land = pd.read_csv(out_dir / "oracle_landscape.csv")
        opt  = pd.read_csv(out_dir / "oracle_optima.csv")
        from viz.oracle_viz import make_figures
        make_figures(out_dir, land, opt, ctx, segments, scenarios, T, disrupted)
        return
```
**要解决的问题**：完整枚举很贵，若参数没变就不该重算。**做法**：算当前 fingerprint `fp`，和 `meta.json` 里存的 `"hash"` 比。
- `--probe` 或 `--force` 时整块跳过。
- **命中条件**：`meta.json` 存在 **且** 其 `"hash"` 等于当前 `fp`。命中就：重读已存的 landscape + optima CSV，重画图，打印 `[cache] reusing…`，**直接返回，不做任何 UE 枚举**。
- **读的文件**：`meta.json`, `oracle_landscape.csv`, `oracle_optima.csv`。**用到的 config**：全部 `FINGERPRINT_PARAMS`（经 `fp`），以及日志里的 `N_DISRUPTED_ORACLE`。

---

## Step 7 — probe timing <a id="step7"></a>
`util/oracle.py:148-156`。
```python
t0 = time.perf_counter()
evaluate_schedule(schedule_from_permutation(list(perms[0]), scenarios[0]), scenarios[0], T, ctx)
dt = time.perf_counter() - t0
s_ue = dt / T                     # 每次 UE 的秒数（一个 schedule = T 次 UE）
total = len(perms) * M
print(...projected full run...)
if probe:
    return
```
**要解决的问题**：完整跑之前先知道大概要多久，好决定实例规模。**做法**：真跑一次 `evaluate_schedule`（即 `T` 次 UE），拿第一个 permutation × 第一个 scenario 测出 **`s_ue` = 每次 UE 的秒数**，再估算总时间 `total × T × s_ue`。
- `--probe`：打印估算后**返回**（不枚举）。
- 注意：probe 的结果**不**存进 `rows`；主循环会从头重评（包括被 probe 过的那一对 `(perm[0], scenario[0])`）。

---

## Step 8 — 从 checkpoint resume <a id="step8"></a>
`util/oracle.py:159-169`。
```python
land_path = out_dir / "oracle_landscape.csv"
prog_path = out_dir / "landscape_progress.json"
rows, done = [], set()
if not force and land_path.exists() and prog_path.exists():
    prog = json.loads(prog_path.read_text(...))
    if prog.get("hash") == fp:
        prev = pd.read_csv(land_path)
        rows = prev.to_dict("records")
        done = {int(s) for s in prev["scenario"].unique()}
        print(f"[resume] {len(done)}/{M} scenarios already computed …")
```
**要解决的问题**：完整枚举可能跑几十分钟，中途被打断不该白跑。**做法（可中断续跑）**：如果**部分** landscape（`oracle_landscape.csv`）和进度标记（`landscape_progress.json`）都在、且标记里的 `hash` 与当前 fingerprint 一致，就把已算的行读进 `rows`、把这些 scenario 标记 `done`。主循环随后跳过它们、只算余下的。
- `--force` 会禁用它。fingerprint 不匹配 → 从头开始（过时的部分结果被无视）。
- **读的文件**：`oracle_landscape.csv`, `landscape_progress.json`。

---

## Step 9 — 主枚举循环 + `evaluate_schedule` <a id="step9"></a>
`util/oracle.py:172-192`。
```python
t_run = time.perf_counter(); scen_times = []
for m, dur in enumerate(scenarios):
    if m in done: continue                       # resume 跳过
    t_scen = time.perf_counter()
    for perm in perms:
        start = schedule_from_permutation(list(perm), dur)      # util/evaluate.py:82
        t_ev = time.perf_counter()
        res = evaluate_schedule(start, dur, T, ctx)             # util/evaluate.py:156
        row = dict(scenario=m, perm="-".join(map(str,perm)),
                   F=res["F"], F1=res["F1"], F2=res["F2"],
                   eval_s=time.perf_counter()-t_ev)
        for e in segments: row[f"start_{e}"] = start[e]
        rows.append(row)
    scen_times.append(time.perf_counter()-t_scen)
    done.add(m)
    pd.DataFrame(rows).to_csv(land_path, index=False)           # 每个 scenario 存一次
    prog_path.write_text(json.dumps({"hash": fp, "done": sorted(done)}), ...)
    print(f"  scenario {m+1}/{M} done ...")
```
**这一步在做的事**：这是整个 oracle 的主体——对每个 scenario `m`（跳过已 `done` 的），遍历所有 `perms`，构建 schedule、评估、每个 `(scenario, permutation)` 记一行。
- **行的列**：`scenario, perm`（用 `-` 连接的 edge id）、`F, F1, F2, eval_s`（这一 schedule 的墙钟时间），外加每段一个 `start_{edge}`。
- **每个 scenario 存一次 checkpoint**（`util/oracle.py:190-191`）：一个 scenario 做完就把整个 `rows` 重写进 `oracle_landscape.csv`，并把 `{hash, done}` 更新进 `landscape_progress.json`。这正是 resume 能工作的原因。

### `evaluate_schedule(start, durations, T, ctx, collect_traces=False, return_u=False)` —— 完整追踪
`util/evaluate.py:156-207`。**它就是那个精确目标 `F(x|ω)`**：给定一个 schedule（开工时刻）、一个 scenario（工期）、horizon `T`，算出 `F`。

先解包 `ctx`（`util/evaluate.py:159-162`）：`dis=ctx["disrupted"]`、`H0, B`、`base_u=ctx["baseline_u"]`、`sev=ctx["severity_vec"]`。

**(A) F2 —— 不需要 UE**（`util/evaluate.py:165`）：`F2 = f2_value(start, durations)`。
- `f2_value`（`util/evaluate.py:98-100`）：`makespan_slot(start,durations) / sum(durations.values())` = 完工 slot ÷ 总工作 slot 数。纯 schedule 算术（Δt 约掉了）。值越大 = schedule 把 makespan 相对于纯修复工作量拖得越长。**为什么放在逐 slot 循环之外**：F2 只和 schedule 时序有关，不涉及交通，不用 UE。

**(B) F1 —— 逐 slot 循环**（`util/evaluate.py:167-193`）。初始化 `D = zeros(len(H0))`（需求 shortfall，`D_0=0`）；累加器 `terms, active, traces, u_rows`。
对每个 slot `k = 1..T`：
1. **损伤状态 `v^{t_k}`（Eq. 2）**（`util/evaluate.py:171-173`）：
   - `damaged = {eid: s for (eid,_,_,s) in dis if k < start[eid] + durations[eid]}` —— 一条边在 `[start, start+dur)` 这些 slot 上算受损；一旦 `k ≥ 完工` 就恢复（从 `damaged` 里去掉）。
   - `v_vec = [s if eid in damaged else 0.0 for (eid,_,_,s) in dis]` —— 每条受损边的实时 severity（恢复后为 0）。
2. **需求 shortfall → `H_t`**（`util/evaluate.py:175-177`）：
   - `target = B @ v_vec` —— 当前由损伤驱动的 shortfall。
   - `D = np.maximum(target, P.RHO * D)` —— 灾发瞬间**急剧掉**到 `target`；等损伤修好、`target` 缩小后，`D` 以速率 `RHO` 衰减（恢复惯性；越接近 1 恢复越慢）。
   - `H = np.clip(H0 - D, 0.0, None)` —— 实际需求 = 正常减去 shortfall，下限截 0。
3. **受损网络**（`util/evaluate.py:179`）：`dmg_edges = build_damaged_edges(ctx, damaged)` —— 见下。
4. **UE + OD 时间**（`util/evaluate.py:181-185`）：
   - `links, _ = solve_ue(dmg_edges, _matrix_from_H(H, ctx), ctx["zone_ids"], rgap=UE_RGAP, max_iter=UE_MAX_ITER, quiet=True)` —— 在受损网络、减小后的需求 `H` 上做一次 UE（内部同 §2e）。
   - `u = od_travel_times(links, ctx)` —— 拥堵 OD 通行时间（§2f）；断连处为 `np.inf`。
   - `u_tilde = np.where(np.isfinite(u), u, ctx["u_pen"])` —— 把 `inf` 换成惩罚 `u_pen`。追加进 `u_rows`。
5. **需求加权比值**（`util/evaluate.py:187-189`）：
   - `den = float(np.sum(H * base_u))`（用当前需求给有限 baseline 加权）。
   - `term = np.sum(H * u_tilde) / den` if `den>0` else `1.0` —— 该 slot 的 F1 项 = 实际 / baseline 的需求加权通行时间比。
   - `terms.append(term)`；`active.append(len(damaged) > 0)`。
   - 若 `collect_traces`：追加 `{k, n_damaged, total_demand=H.sum(), f1_term=term}`（给 figure 03 用）。

**(C) F1 聚合**（`util/evaluate.py:195-200`）：
```python
terms = np.asarray(terms)
if P.F1_ACTIVE_ONLY:                       # lever 5
    mask = np.asarray(active, bool)
    F1 = terms[mask].mean() if mask.any() else terms.mean()
else:
    F1 = terms.mean()
```
- 默认 `F1_ACTIVE_ONLY=False`：在**整段 horizon `[1,T]`** 上平均——与 §2.1.1 MILP surrogate 用的 `c_e^k (1/T)` 归一化一致。若 `True`：只在"仍有损伤"的 slot 上平均。

**(D) 组合 F**（`util/evaluate.py:201`）：`F = P.MU * F1 + (1.0 - P.MU) * F2`。
- 默认 `MU=1.0`：`F = F1`（accessibility-only）；F2 仍被算出并记录，但权重为 0。

**返回**（`util/evaluate.py:202-207`）：`{F, F1, F2}`；可选返回 `u_tilde`（`(T,|R|)` 的固定通行时间，若 `return_u`）和 `traces` DataFrame（若 `collect_traces`）。

**`evaluate_schedule` 用到的 config**：`RHO`、`KAPPA`（构建时已烘进 `B`）、`SEVER_SEVERITY`/`CAP_RETAIN`/`SPEED_RETAIN`（经 `build_damaged_edges`）、`UE_RGAP`、`UE_MAX_ITER`、`UPEN_FACTOR`（已烘进 `u_pen`）、`F1_ACTIVE_ONLY`、`MU`、`C_MAX`/`DELTA_T_H`（经 schedule）。

### `build_damaged_edges(ctx, damaged)` — `util/evaluate.py:54-76`
**要解决的问题**：每个 slot 的 UE 都要跑在"当前受损状态"的网络上，所以要根据当前 `damaged`（`{edge_id: severity}`）临时改一份 edge 表。
- 复制 `ctx["edges"]`。若 `damaged` 非空（`util/evaluate.py:60-75`）：
  - `cap`、`fft` = `capacity` / `free_flow_time` 列的副本；`idx = ctx["edge_row"]`。
  - 对每个 `(eid, sev)`，`j = idx[eid]`：
    - 若 `sev >= P.SEVER_SEVERITY` → 把 `j` 记进 `sever`（该边**整条移除**——真正断连，受影响 OD 之后会吃 `u_pen`）。
    - 否则 → `cap[j] *= P.CAP_RETAIN[sev]`（capacity 收紧）且 `fft[j] /= P.SPEED_RETAIN[sev]`（free-flow-time 变慢）。
  - 把改过的 `cap`、`fft` 写回；若有 `sever`，drop 掉那些行（`edges.drop(...).reset_index`）。
- 已修复/未受损的链路原样不动。**用到的 config**：`SEVER_SEVERITY`, `CAP_RETAIN`, `SPEED_RETAIN`。

---

## Step 10 — landscape 排序 + per-scenario 最优 <a id="step10"></a>
`util/oracle.py:193-198`。
```python
total_time = time.perf_counter() - t_run
land = pd.DataFrame(rows).sort_values(["scenario","perm"]).reset_index(drop=True)
land.to_csv(land_path, index=False)                                     # 最终 landscape
opt = land.loc[land.groupby("scenario")["F"].idxmin()].reset_index(drop=True)
opt.to_csv(out_dir / "oracle_optima.csv", index=False)
```
**要解决的问题**：把主循环攒下的所有行整理成两样成品——完整 landscape，和每个 scenario 的事后最优。
- 组装完整 landscape DataFrame，按 `(scenario, perm)` 排序，写最终 `oracle_landscape.csv`。
- **per-scenario 最优**：`groupby("scenario")["F"].idxmin()` 为每个 scenario 选 `F` 最小的那行（真实事后最优 `x*` 和 `F*`）。写 `oracle_optima.csv`。
- **写的文件**：`oracle_landscape.csv`, `oracle_optima.csv`。这两个文件既是最终产物，也正是 Step 6 cache 命中时直接重读的东西。

---

## Step 11 — summary.txt <a id="step11"></a>
`util/oracle.py:200-211`。**要解决的问题**：给人一个一眼能读的运行总结。组装并写 `summary.txt`（同时回显到 stdout）：
- `segments`、`perms`、`scenarios (M)`、horizon `T`。
- `~{s_ue*1000:.0f} ms/UE`（来自 probe）；评估过的 schedule 总数 `= len(land)`。
- 总评估计算量 = `land["eval_s"].sum()/60` min；平均 `ms/schedule = land["eval_s"].mean()*1000`；本次会话分钟数 `total_time/60`。
- `mean F* over scenarios = opt["F"].mean()`；`mean F1* = opt["F1"].mean()`、`mean F2* = opt["F2"].mean()`。
- 一句说明：oracle 最优是每个 scenario 内所有 permutation 的 min（真实事后最优）。
- **写的文件**：`summary.txt`。

---

## Step 12 — figures（`make_figures`） <a id="step12"></a>
`util/oracle.py:213-214` → `make_figures(out_dir, land, opt, ctx, segments, scenarios, T, disrupted)`，定义在 `viz/oracle_viz.py:23-110`。（这与 Step 6 cache 命中时调的是**同一个**函数。）

**准备**（`viz/oracle_viz.py:24-28`）：`use_pub()`（套用 `viz/style.py:59` 的出版级 rcParams）；确保 `figures/` 存在；`sev = {edge_id: severity}`；`drep` = 代表性 scenario `rep=0` 的 landscape 行。

**Figure 01 — `01_F_landscape`**（`viz/oracle_viz.py:31-60`）：对每个 scenario 把它的 `F` 升序排；叠成 `sorted_F`（行=scenario，列=schedule 排名）。画跨 scenario 的 min–max 带、每个 scenario 的 best→worst 曲线，标注 oracle 最优 `F* = sorted_F[:,0].mean()`，以及 best→worst 的相对差 `%`。经 `save_pub` 保存。

**Figure 02 — `02_F1_F2_tradeoff`**（`viz/oracle_viz.py:63-74`）：把代表性 scenario 里每个 schedule 画在 `(F2, F1)` 空间，颜色按 `F`（colormap `CMAP_SEQ`），带一条标 `F (μ=…)` 的 colorbar。展示双目标结构。

**Figure 03 — `03_best_schedule`**（`viz/oracle_viz.py:77-110`）：取 `rep` 的最优行；重建 `start = {e: orow[start_e]}`；**重跑** `evaluate_schedule(start, dur, T, ctx, collect_traces=True)` 拿逐 slot 的 `traces`。三个堆叠面板：(a) 最优 schedule 的 Gantt（bar 按 severity 上色，经 `severity_color`），(b) 逐 slot 的 OD 总需求（急降 → 恢复，虚线为正常 `H0.sum()`），(c) 逐 slot 的 F1 项（虚线=1 表示完全恢复）。面板字母经 `panel_label`。

### 用到的 `viz/style.py` helper
- `use_pub()`（`viz/style.py:59-61`）：套用 `PUB_RC` rcParams（Arial/Helvetica、SVG/PDF 文字可编辑、紧凑期刊尺寸、去掉上/右 spine）。
- `severity_color(s)`（`:64`）：`{1:#F6CFCB, 2:#E59A93, 3:#B64342}` 暖色渐变。
- `panel_label(ax,label)`（`:72`）：左上角加粗面板字母。
- `save_pub(fig, stem, dpi=600, svg=False, pdf=False)`（`:78`）：默认存 **PNG@600 dpi**（SVG/PDF 需显式开启；`make_figures` 用默认，所以只出 PNG）。
- `C` 调色板，`CMAP_SEQ="cividis"`。

**写的文件**：`figures/01_F_landscape.png`, `figures/02_F1_F2_tradeoff.png`, `figures/03_best_schedule.png`。**用到的 config**：`MU`（colorbar 标签）。

---

## Step 13 — meta.json 写入 + checkpoint 清理 <a id="step13"></a>
`util/oracle.py:216-224`。
```python
meta_path.write_text(json.dumps(
    {"hash": fp, "params": values,
     "timing": {"total_eval_s": land["eval_s"].sum(), "n_schedules": len(land),
                "s_per_schedule": land["eval_s"].mean(),
                "s_per_ue": land["eval_s"].sum() / (len(land)*T),
                "this_session_s": total_time, "scenario_s": scen_times}},
    sort_keys=True, indent=2), ...)
prog_path.unlink(missing_ok=True)          # 完成 -> 删掉 resume 标记
print(f"\nWrote {out_dir}  (meta.json hash={fp[:12]}…)")
```
**要解决的问题**：把"这次结果是用什么参数算的"钉死，供下次 cache 判断；并清掉不再需要的 resume 标记。
- 写 `meta.json` = 新鲜度 `hash`（fingerprint）、完整 `params` 值、以及一个 timing 块（总评估秒数、schedule 数、s/schedule、s/UE、会话秒数、逐 scenario 秒数）。
- **删掉** `landscape_progress.json`（`prog_path.unlink(missing_ok=True)`）——这次完整跑完了，resume 标记没用了。否则它下次存在会被当成"上次被中断"。
- **写的文件**：`meta.json`；**删除**：`landscape_progress.json`。

至此闭环：下次同参数再运行，Step 6 读到匹配的 `meta.json["hash"]` 就直接复用，不再枚举。

---

# 附录 A — 参数默认值表 <a id="appA"></a>

下面都是**当前** `config.py` 的默认值（唯一真实来源，`import config as P`）。除注明"给定"外，每一项都是建模假设。

| 参数 | 当前默认 | 在 `run_oracle` 里的角色 |
|---|---|---|
| `DELTA_T_H` | 3.0 h/slot（给定） | slot 长度；在 F2 里约掉，设定 horizon 的物理含义 |
| `C_MAX` | 2 | `schedule_from_permutation` 里的 crew 数 |
| `MU` | **1.0** | `F = μF1 + (1−μ)F2`；**1.0 ⇒ accessibility-only**（F2 照算，权重 0） |
| `CAP_RETAIN` | **{1:0.3, 2:0.1, 3:0.02}** | 受损 capacity = retain × capacity（`build_damaged_edges`） |
| `SPEED_RETAIN` | **{1:0.5, 2:0.3, 3:0.2}** | 受损 free_flow_time = t0 / retain（`build_damaged_edges`） |
| `SEVER_SEVERITY` | 3 | severity ≥ 此值 ⇒ 整条移除（断连 → `u_pen`） |
| `F1_ACTIVE_ONLY` | False | F1 在整段 `[1,T]` 平均（对齐 MILP 的 `c_e^k` 1/T 归一化） |
| `RHO` | 0.7 | 恢复惯性：`D_t = max(B v_t, ρ D_{t-1})` |
| `KAPPA` | 1.0 | 损伤→shortfall 尺度：`B[r,e]=κ(h_r0/3)·1{on SP}` |
| `UPEN_FACTOR` | 10.0 | `u_pen = 10 × 最大有限 baseline OD 通行时间` |
| `M_SCENARIOS` | 10 | duration scenario 数量 |
| `SEED` | 42 | `sample_scenarios` 的 RNG seed（确定性） |
| `N_DISRUPTED_ORACLE` | 4 | disruption 集合大小（4! = 24 个 schedule）；充当 `n{N}/` cache 文件夹 key |
| `UE_RGAP` | 1e-6 | per-step UE 相对 gap 目标（比 1e-12 的 validation 松） |
| `UE_MAX_ITER` | 100 | per-step UE 迭代上限 |
| `DURATION_SUPPORT` | Table 1（按 (class,severity)） | `sample_scenarios` 里的 base 工期取样集 |
| `ETA` | [0.8, 1.0, 1.2]（给定） | crew 效率乘子取样 |

> **与旧文档的差异（已在本表更新）**：早年的 `oracle_validation.md` 曾写 `CAP_RETAIN={1:0.7,2:0.4,3:0.1}`、`SPEED_RETAIN={1:0.8,2:0.6,3:0.4}`、`μ=0.5`。当前 `config.py` 已改为上表更严的 retain 和 `MU=1.0`。本表以现行代码为准。

**非 fingerprint 的 config**（`config.py` 里有、但 `run_oracle` 不读，供日后 MILP 用）：`MILP_MAX_ITER=20`, `MILP_CYCLE_TOL=3`, `MILP_DAMPING=0.5`。

---

# 附录 B — validation levels 与 comparison metrics <a id="appB"></a>

（沿用自已退役的 `oracle_validation.md`。）oracle 通过枚举所有 work-conserving schedule、用精确 Figure-1 目标 `F(x|ω)` 打分，算出每个 scenario 的**真实**事后最优 `F*`。

**Validation levels — A（surrogate） vs B（true）。** 日后建 pretraining MILP 时可在两个层次检查：
- **Level A — 表述正确性。** 暴力枚举 MILP **自己的 surrogate** 目标（固定通行时间 + 线性化 F2），确认 MILP 返回同一个最小值点。用来隔离 MILP 编码里的 bug（约束、`c_e^k`、makespan 线性化）。
- **Level B — 近似质量。** 暴力枚举**真实** `F`（完整 UE 流水线——即 `run_oracle` 做的），把 `F(x_milp*)` 与真实最优 `F*` 比。检验 traffic-fixation + 线性化是否真能给出真实问题的（近似）最优 schedule。
- **当前决定**：oracle 只做 **Level B**。Level A 等 MILP 建好再说。

**Comparison metrics（未来 MILP vs oracle）。** 全部可从 `oracle_landscape.csv`（每个测过的 `x` 及其 `F/F1/F2`）算出：
- **objective gap**：`F(x_milp*) − F*`（相对）——最核心的验收数。
- **rank**：`x_milp*` 在所有枚举 schedule 里排第几（"24 中第 1" / "前 X%"）。
- **schedule match**：`x_milp* == x*` 吗？（若不同但 `F` 相等 → 存在多重最优；比 `F`，别比 argmin。）
- **landscape shape**：尖峰型还是宽平台型（决定合理的 MILP 验收容差）。

---

# 附录 C — Figure-1 fidelity 与建模 caveats <a id="appC"></a>

（沿用自已退役的 `oracle_validation.md`。）`util/evaluate.py` 逐步照搬 Figure 1；只在 Figure 1 本身没说清/不完整处才偏离：
- **F2 在逐 slot 循环外算**（不需 UE，纯 schedule 算术）。
- **Baseline `u_r^{t0}`** = 在未受损网络、正常需求 `H^{t0}` 上做一次 UE（Fig. 1 在 F1 里用它但没说它怎么来）。
- **F1 可以低于 1。** F1 是需求加权的，而灾后需求下降，于是较少的出行者去承受"正常需求下拥堵基准 `u_r^{t0}`"；在低需求的恢复窗口里，需求加权比值可以掉到 1 以下。论文说的"完全恢复时 =1"只在**网络和需求都回到正常**后才成立。恢复越快 `F` 越低（更多时间处于恢复态），所以优化信号仍然正确。

**Demand model（drop → recover）。** 开源 OD = 正常时期需求 `H^{t0}`（也是完全恢复的水平）。字面上的 `H_t = A·H_{t-1} + B·v_t`（`A=ρI`）会衰减到 0（错的）。代码建模的是 **shortfall**：
```
D_t = max(B·v_t, ρ·D_{t-1}) ,   H_t = max(0, H^{t0} − D_t) ,   D_0 = 0
B[r,e] = κ·(h_r^{t0}/3)·1{e 在 OD r 的 free-flow 最短路上}
```
→ 灾发瞬间急降（到当前损伤驱动的 shortfall `B·v`），随后随道路修复以速率 ρ 逐步恢复到正常。κ 调降幅深浅，ρ 调恢复快慢（看 `figures/03_best_schedule.png`）。

**Caveats。**
- **Work-conserving 归约。** schedule 是 permutation → list scheduling（无 idling），假设 idling 永不改善 F1/F2。在动态需求下，需求耦合使这只是**近似**成立——结果反常时要复查。好处是把枚举保持在 `|ℰ|!`。
- **没有 UE cache。** 动态需求 ⇒ UE 依赖恢复路径 ⇒ `2^|ℰ|` 的"已完成集合" cache 失效；**每个 slot 都是全新 UE 求解**。
- **AequilibraE 单次调用成本。** 这里约 ~1+ s/UE（24 节点网上每次调用的固定 setup 开销），把暴力枚举限制在小实例（`N=4`, `M=10`）。要上更大实例就得换一个快速的自研 Frank-Wolfe。
