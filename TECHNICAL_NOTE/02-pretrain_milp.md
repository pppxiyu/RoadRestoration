# 02 — `run_pretrain_milp()` 的解题逻辑(§2.1.1 traffic-fixation MILP)

这份笔记解释 §2.1.1 预训练求解器是**如何**为每个修复时长 scenario 找出一张开工时间表、并与
brute-force oracle 对比的。入口是 `util/pretrain_milp.py` 里的 `run_pretrain_milp()`,命令行
`python -m util.pretrain_milp` 直接触发它。笔记按代码真实的解题顺序组织:每一节是一个逻辑步骤,
先说这一步要解决什么问题,再说它做了什么、为什么这样做,最后交代产物被后面哪一步消费,
并以 path:line 锚点收尾,供对照当前源码查证。第三方库(AequilibraE、networkx、scipy)只概括
其角色,不深入内部。项目参数统一由根目录 `config.py` 提供(代码内引用名为 `P`),
各默认值汇总在附录 A。
(util/pretrain_milp.py:184, util/pretrain_milp.py:335-341)

---

## 总览 · 为什么走"冻结交通 → 线性 surrogate → MILP → 刷新 UE"这条路

要解决的问题:洪水打断了若干条道路 segment,每条要占用一个 crew 连续修若干个时间 slot;
要找一张"每条 segment 何时开工"的时间表,使整段恢复期内路网可达性的退化(objective F1)最小。
难点在耦合:F1 由每个 slot 的 user equilibrium(UE)通行时间构成,而通行时间取决于该 slot
哪些路还坏着,也就是取决于时间表本身。把这个耦合原样塞进一个优化器,意味着目标非线性、
且每次评估内部都要反复解 UE,代价不可接受。

这条流水线采用交替优化(traffic fixation),一轮之内分四拍:
第一,把整段 horizon 的 OD 通行时间**冻结**成常数(取上一轮 UE 的结果);
第二,在冻结的通行时间下,把"segment e 在 slot k 开工对 F1 的边际改善"解析地算成系数 c_e^k,
于是 F1 被线性 surrogate Σ_e Σ_k c_e^k·x_e^k 替代,这一步完全不碰 UE;
第三,解一个很小的 start-time MILP,在排程约束下最大化 surrogate,得到新时间表;
第四,只为这张新时间表跑一遍真实评估(每个 slot 一次 UE),刷新通行时间,回到第一拍。
如此迭代,直到时间表不再变化或触发保护条件。

为什么这样能省掉海量 UE 求解:昂贵的 UE 只出现在"每轮评估一张时间表"这一处,每轮恰好 T 次;
一个 scenario 总共 (迭代轮数+1)·T 次。相比之下,oracle 对同一 scenario 要枚举全部 N! 张
work-conserving 时间表、每张 T 次 UE;而把 UE 埋进优化器内层则连调用次数都无法封顶。
surrogate 与 MILP 部分是纯 numpy 加一次 HiGHS 求解,耗时可忽略。
(util/pretrain_milp.py:1-25)

---

## 第 1 步 · 与 oracle 共用同一份问题定义,并划清"不是数据泄露"的边界

要解决的问题:MILP 的成绩要与 brute-force oracle 直接比较,两者必须面对**完全相同**的问题;
但又必须保证 oracle 已算出的"答案"绝不能反过来帮 MILP 作弊。

做了什么:run_pretrain_milp 用与 run_oracle 完全相同的四个构件来定义问题——同一个受损实例
(select_oracle_instance)、同一批修复时长 scenarios(sample_scenarios,同一个 seed)、
同一个全局 horizon(compute_horizon)、同一个真实目标评估器(evaluate_schedule)。
而 oracle 的**解**(oracle_optima.csv、oracle_landscape.csv)只在流程最末尾的对比阶段被读取,
那时每个 scenario 的 MILP schedule 早已算完并写入 checkpoint。

为什么这样做:共享问题定义是刻意的——实例、scenarios、horizon、目标函数任何一处不同,
F_MILP 与 F_oracle 就失去可比性。数据泄露的判据是"上游算法是否消费了下游答案":MILP 的
目标系数 c_e^k 只依赖它自己迭代产生的冻结 UE 通行时间,从不查阅 oracle 的任何解;即使 oracle
从未运行,MILP 的结果也一字不差,只是最后的对比图被推迟(见第 17 步)。

产物给谁用:这条边界贯穿全文——第 2 至 6 步是共享的问题定义,第 17 步才是 post-hoc
读取 oracle 解的位置。
(util/pretrain_milp.py:188-192, util/pretrain_milp.py:248-256)

---

## 第 2 步 · 按问题规模隔离输出目录

要解决的问题:同一台机器可能跑过不同规模(受损 segment 数 N)的实验,不同规模的结果不能
互相覆盖;对比阶段还要能按规模找到对应的 oracle 结果。

做了什么:输入数据默认取 data/siouxfalls_toy,输出根目录默认为 outputs/pretrain_milp;
真正的落盘目录由 oracle 模块的 scale_dir 拼成 outputs/pretrain_milp/n{N}/,N 取 config.py 的
N_DISRUPTED_ORACLE(默认 4),并预先创建其下的 figures 子目录。

为什么这样做:MILP 端与 oracle 端共用同一条 n{N} 命名规则,post-hoc 对比时把 base 目录换成
outputs/oracle 即可拼出同规模 oracle 结果的路径,不需要任何额外配置,也永远不会覆盖
另一个规模的结果。

产物给谁用:后续所有 CSV、JSON、图的写盘路径都以这个目录为根;第 17 步用同一规则定位
oracle 目录。
(util/pretrain_milp.py:44-46, util/pretrain_milp.py:185-186, util/oracle.py:59-62)

---

## 第 3 步 · 选定受损实例:按 baseline UE flow 排重要性

要解决的问题:确定哪 N 条 segment 受损、各自 severity 多少。这份选择必须让"先修哪条"真正
影响 F1(否则 schedule 优化无意义),且必须完全确定,保证 MILP 与 oracle 拿到同一份实例。

做了什么:复用 oracle 模块的 select_oracle_instance。它先在**无损**网络、正常 OD 需求下用项目
自己的 UE 引擎解一次均衡,把每条无向 edge 的双向流量相加作为重要性指标;再按流量从高到低排序,
取最高的两条作为 critical 干道并赋 severity 3(达到 SEVER_SEVERITY,评估时整条从网络移除、
造成真断连),剩余名额沿低流量区间均匀铺开,severity 按 2、1 交替;拼出 level_id 后把结果写到
data/siouxfalls_toy/disruption/disrupted_segments_oracle{N}.csv 并返回。回到 run_pretrain_milp,
受损 edge id 升序排成规范列表,作为下游一切"按 segment 对齐"的统一顺序。

为什么这样做:重要性排序从前靠逐行读开源参考解 raw/SiouxFalls_flow.tntp;改为自算 baseline UE
之后,选实例不再依赖那个随数据集附带的参考文件,换一套 network/OD 数据也无需另配参考解
(该 .tntp 现在只被 util/ue.py 的 _validate 自检读取)。commit 62f227b 的验证记录表明改动前后
选出的实例完全一致(edge_ids [1, 12, 15, 17]、severity [1, 2, 3, 3],逐边流量与参考解相差约
0.14%)。混入 critical 与次要路段、并让 critical 路段真断连,是为了拉开不同修复顺序之间的
F1 差距。整个选择过程不使用任何随机数。

产物给谁用:受损实例表交给第 4 步构建 context、第 5 步采样时长;规范 segment 顺序被 B 矩阵
的列、时长字典的键、MILP 变量的行共同遵守。
(util/pretrain_milp.py:188-189, util/oracle.py:85-109, util/oracle.py:65-82)

---

## 第 4 步 · 一次性构建静态 context

要解决的问题:后面每个 scenario、每轮迭代、每个 slot 都要用到一批"算一次就不再变"的量——
路网表、OD 对、灾前基准通行时间、断连惩罚时间,以及把"哪些路坏着"翻译成"哪些 OD 需求被
压制"的线性映射 B(Φ)。这些必须先算好装进一个 context 反复复用,否则每次评估都要重复昂贵的
准备工作。

做了什么:build_context 先从 data/siouxfalls_toy/network/ 下的 edges.csv、od_pairs.csv、nodes.csv
读入路网与 OD 需求(Sioux Falls 中每个 node 都是 OD zone);然后在无损网络、正常需求 H0 下解
一次 UE,把逐条有向 link 的 congested time 经最短路折算成每个 OD 对的基准通行时间 u_r^{t0};
取有限基准值的最大者乘 UPEN_FACTOR(10)作为断连惩罚 u_pen——当某 OD 对被彻底切断、通行
时间为无穷时用这个大数替换,让 F1 保持有限但被重罚。接着按 free-flow 权重建一张双向图,
对每个 OD 对求 free-flow 最短路:若受损 segment e 落在 OD r 的最短路上,则
B[r, e] = κ·(h_r^0 / 3),否则为 0——B 就是"损坏 → 需求缺口"的线性映射,只压制原本要途经
受损路段的需求。最后按受损表的行序存一个 severity 向量,与 B 的列一一对齐。

为什么这样做:基准通行时间必须出自与灾后评估同一套 UE 模型,否则灾前灾后不可比,所以它在
这里用同一个引擎算出、终生复用。B 用 free-flow 最短路作判据是刻意的简化:把"受影响"离散化为
"在不在最短路上",避免每个 slot 重算受影响集合。severity 向量之所以进 context,是因为它在两处
扮演"损坏量":真实评估里作为 B·v 中 v 的分量,surrogate 系数里作为修复后放回需求的尺度 v_e*
(见第 9 步)。

关于 UE 引擎的边界:solve_ue 把无向 edge 表和稠密 OD 矩阵(由按 OD 对排列的需求向量散射而成)
交给 AequilibraE,以 BPR 拥堵函数 t_a(x) = t0·(1 + α·(x/c)^β)、bi-conjugate Frank-Wolfe 算法迭代到
relative gap 低于 UE_RGAP(1e-6,上限 UE_MAX_ITER = 100 轮),返回每条有向 link 的均衡流量与
congested time;这是整条流水线中唯一昂贵的数值内核,本文不深入其内部(同模块的 Beckmann
目标计算与 _validate 自检不在本流水线路径上)。link time 折算成 OD time 用的是 networkx 的
Dijkstra 单源最短路;某个 origin 的全部邻边都被移除时,它相关的 OD 记为无穷。

产物给谁用:context 里的基准通行时间、u_pen、B、severity 向量、受损元组与路网表,供第 7 步的
真实评估器与第 9 步的 c_e^k 计算共同使用。
(util/pretrain_milp.py:190, util/evaluate.py:106-150, util/ue.py:81-121, util/evaluate.py:27-51, util/io.py:10-23)

---

## 第 5 步 · 采样修复时长 scenarios

要解决的问题:修复时长天然不确定,单一时长下的最优 schedule 说明不了什么;需要 M 份时长实现,
逐份求解、逐份与 oracle 对比。

做了什么:sample_scenarios 以固定 seed(42)驱动随机数发生器,按 level——即 (road_class,
severity) 组合——抽时长:每个 scenario 里,每个 level 先从 DURATION_SUPPORT 的支撑集抽一个
base 时长,再乘上从 ETA(0.8 / 1.0 / 1.2)抽出的 crew-efficiency 因子,四舍五入且不低于 1;
同 level 的所有受损 segment 在该 scenario 内共用这一个时长。输出 M(默认 10)份
"edge → 时长(slot 数)"的映射。

为什么这样做:按 level 抽而不是按 edge 抽,对应论文设定 d_e(ω) = d_{ℓ(e)}(ω);seed 与 oracle 端
相同,保证两边拿到逐份一致的 scenarios——这是 F 可比的前提之一。

产物给谁用:scenarios 列表决定第 6 步的 horizon,也是第 15 步逐 scenario 循环的迭代对象。
(util/pretrain_milp.py:191, util/scenarios.py:14-28)

---

## 第 6 步 · 统一评估窗口:全局 horizon T

要解决的问题:F1 是对固定窗口 [1, T] 取平均的;不同 schedule、不同 scenario 若各用各的窗口,
数值就不可比。T 还必须足够大,让任何要被评估的 schedule 都能在窗口内完工。

做了什么:compute_horizon 遍历"受损 segment 的全部排列 × 全部 scenarios",对每个组合用
work-conserving 列表排程摆出 schedule、算出完工 slot,取所有组合的最大值作为全局 T。
列表排程的规则是:C_MAX(2)个相同 crew,按给定优先级顺序把每条 segment 派给最早空闲的
crew,最早开工 slot 为 1、不留空档;完工 slot 即所有 segment"开工 + 时长"的最大值。

为什么这样做:用"全排列 × 全 scenario 的最坏完工时刻"定 T,保证 oracle 要枚举的每张 schedule
都在窗口内,而两端共用同一个 T。这些组合评估只做排程算术、不跑 UE,N=4、M=10 时仅
24×10=240 次,代价可忽略。T 同时是 MILP 决策变量的时间维度、F1 取平均的分母、c_e^k 里 1/T
归一化的分母,必须先于一切定死。

产物给谁用:T 交给真实评估器(逐 slot 循环上限)、c_e^k 计算(求和上限与归一化)、MILP
(变量网格)共同使用。
(util/pretrain_milp.py:192, util/oracle.py:112-119, util/evaluate.py:82-95)

---

## 第 7 步 · 真实目标 F 的评估器:每个 slot 一次 UE

要解决的问题:整条流水线需要一个唯一权威的"真实目标"评估器——给定一张 schedule 和一份时长
实现,精确算出 F。交替优化靠它刷新通行时间、挑选最优 iterate;oracle 靠同一个函数做枚举,
两边共用才保证可比。

做了什么:evaluate_schedule 先算 F2 = 完工 slot / Σ_e d_e(修复效率,纯排程算术、不跑 UE);
然后对 k = 1 … T 逐 slot 循环:
(a) 判定损坏状态——只要 k 还没到某 segment 的完工 slot(开工 + 时长),它就仍按其 severity
计为损坏;开工本身不改变损坏状态,只有完工才把它清零(Eq. 2);
(b) 需求缺口递推 D_k = max(B·v_k, ρ·D_{k−1}),当期需求 H_k = max(0, H0 − D_k),其中 v_k 是
当期损坏向量(损坏为 severity、修好为 0),ρ = RHO = 0.7——onset 时需求猛跌到当前损坏驱动的
缺口,此后随路修好、B·v_k 收缩,缺口按每 slot 乘 ρ 的速率惯性消退,需求逐步回到 H0;
(c) 按当期损坏改造路网:severity 达到 SEVER_SEVERITY(3)的 segment 整条移除(真断连),
较轻的按 CAP_RETAIN 打折 capacity、按 SPEED_RETAIN 抬高 free-flow time,其余 link 原样保留;
(d) 在改造后的网络、当期需求 H_k 上解一次 UE,把 link time 折算成各 OD 通行时间,断连的 OD
以 u_pen 替换,得到该 slot 的通行时间行 ũ^k;
(e) 记录该 slot 的 F1 项:当期需求加权的"实际 / 基准"通行时间比
Σ_r H_r^k·ũ_r^k / Σ_r H_r^k·u_r^{t0}。
循环结束后 F1 取 T 个项的平均(F1_ACTIVE_ONLY 为 False,对整段 [1, T] 平均),再合成真实目标

F = μ·F1 + (1 − μ)·F2

μ = MU = 1.0,所以当前 F ≡ F1;F2 仍被计算并随结果记录,但权重为 0、不进目标。带 return_u
调用时,额外把 T 行通行时间堆成一个 (T × |R|) 矩阵一并返回。

为什么这样做:每 slot 一次 UE 是"真实"二字的来源,也是全流水线的代价瓶颈——一次调用恰好
T 次 UE。F1 对整段 [1, T] 平均而非只对"仍有损坏"的 slot 平均,是为了与 c_e^k 的 1/T 归一化
严格对齐,否则 surrogate 与真实目标会在归一化上错位。u_pen 把"断连"这种无穷大事件折算成
有限但很痛的代价,使 F1 始终有限、可比较。

产物给谁用:F / F1 / F2 供交替循环挑最优、供 optima 表落盘;(T × |R|) 通行时间矩阵正是交替
优化要"冻结"的对象,直接喂给第 9 步。
(util/evaluate.py:156-206, util/evaluate.py:54-76)

---

## 第 8 步 · 交替优化的起点:greedy 初始 schedule 与第一份冻结通行时间

要解决的问题:第一轮迭代还没有任何 UE 结果可供冻结,必须先有一张初始 schedule 和它对应的
整段通行时间。

做了什么:alternating_optimize 把受损 segment 按 edge id 原序交给 work-conserving 列表排程,
得到一张 greedy 初始 schedule;立刻用真实评估器(带 return_u)评估它,拿到初始 F 和第一份
(T × |R|) 冻结通行时间;schedule 与结果一并存入历史列表,并在"出现次数计数器"里给这张
schedule 记一次(供第 11 步的 cycle guard 使用)。

为什么这样做:初始点只需合理、不必聪明——最终输出反正是事后按真实 F 从全历史里挑的
(第 13 步),初始 schedule 也在历史里,所以输出保证不劣于这个 greedy 起点。初始评估一举
两得:给"最优"设了保底,也生产了第一份可冻结的通行时间。

产物给谁用:冻结通行时间进入第 9 步;历史列表与计数器由第 11 步维护、第 13 步消费。
(util/pretrain_milp.py:147-154)

---

## 第 9 步 · 解析敏感度系数 c_e^k:不跑 UE 的线性 surrogate

要解决的问题:MILP 需要线性目标,但真实 F1 经由 UE 依赖于 schedule,非线性。在通行时间被
冻结的前提下,要解析地回答:"若 segment e 在 slot k 开工,它对 F1 各步项的边际改善是多少?"
——并且这个回答一次 UE 都不许跑。

做了什么:precompute_c 先由冻结通行时间与基准算出每个 slot、每个 OD 的**拥堵超额**

α_r^{k'} = 1 − u_r^{t0} / ũ_r^{k'}

(ũ 取冻结矩阵的第 k' 行),再对每条 segment e(时长 d_e)和每个可行开工 slot k(须满足
k + d_e ≤ T)按闭式累加

c_e^k = (1/T) · Σ_{k'=k+d_e}^{T} (1 − ρ^{k'−k−d_e+1}) · Σ_r B[r, e] · v_e* · α_r^{k'}

其中 v_e* 是该 segment 的 severity(取自 context 的 severity 向量),B[r, e]·v_e* 即"e 完全修复时
放回网上的 OD r 需求"。不可行开工(k > T − d_e)的系数保持为 0,由 MILP 的变量上界另行禁止。
全程纯 numpy,UE 调用次数为零。

为什么这样做:直觉是"修好 e 的收益 = 它放回的需求 × 这些需求所处 OD 的拥堵超额"。收益从
完工 slot k + d_e 才开始产生;放回的需求并非一步到位,而是按恢复因子 (1 − ρ^{n+1}) 随完工后第
n 个 slot 逐步**增长**——与第 7 步真实需求模型中缺口按 ρ 消退的速率一致。这是 Problem-2 的
shortfall 修正:旧版让收益按 ρ 的幂**衰减**是错误方向,当前源码为准。对 k' 求和把这份逐 slot
收益摊到 F1 的各步项上,除以 T 与 F1 的整段平均对齐。α 可以为负——恢复期需求被压低时网络
可能比灾前基准还快——这是合法信号,表示此时修这条路的边际收益为负。severity 在系数中的
角色是把"放回需求的规模"与损坏等级挂钩:severity 越高,修复它放回的需求越多,同等拥堵
超额下收益越大。

产物给谁用:系数矩阵(|E| × T)是第 10 步 MILP 的全部目标信息;拥堵超额矩阵另返回给诊断
用途(level_a 用它统计负值占比,见第 17 步)。
(util/pretrain_milp.py:52-77)

---

## 第 10 步 · start-time MILP:在更大的可行域里最大化 surrogate

要解决的问题:在冻结通行时间的世界里,找让 surrogate 最大的开工时间表,并遵守排程规则:
每条 segment 恰好开工一次、任一时刻同时在修的数量不超过 crew 数、所有工作都在 horizon 内完工。

做了什么:决策变量 x_e^k ∈ {0, 1} 表示"segment e 是否在 slot k 开工"(源码实现中记作 y),共 |E|·T 个。目标为

max Σ_e Σ_k c_e^k · x_e^k

(实现上因 HiGHS 只做最小化,对系数取负后最小化)。约束三类:
其一,每条 segment 恰好开工一次:Σ_{k=1}^{T} x_e^k = 1;
其二,crew 上限:对每个 slot k,Σ_e Σ_{k'=max(1, k−d_e+1)}^{k} x_e^{k'} ≤ C_MAX——内层求和
枚举"在 k 时刻仍在修"的所有可能开工时刻;
其三,horizon 越界禁止:对 k > T − d_e 的变量直接把上界压为 0。这是唯一真正的可行性守卫,
与 c_e^k 在这些位置为 0 相呼应。
问题交给 scipy.optimize.milp(背后是 HiGHS branch-and-bound,随 scipy 附带、无需 license)求解,
失败即抛错;解出后每条 segment 取其行上取值为 1 的 slot,还原成"edge → 开工 slot"的映射。
模块里另有一个小工具反向计算任意给定 schedule 的 surrogate 值 Σ_e c_e^{s_e},供逐轮 trace 记录
与 level_a 检查使用。

为什么这样做:模型极小(N=4、T 为几十时不过几百个 binary 变量),HiGHS 毫秒级解出,交替
循环的成本几乎全部留给 UE。特别注意可行域:这里允许 crew 故意空等(不要求 work-conserving),
因此 MILP 的可行域是 oracle 所枚举的 work-conserving schedule 集合的**严格超集**——这是第 17 步
gap 可为负的根源。

产物给谁用:新 schedule 交回交替循环,由真实评估器打分并刷新通行时间;surrogate 值写入逐轮
trace,供过程图使用。
(util/pretrain_milp.py:83-121, util/pretrain_milp.py:124-126)

---

## 第 11 步 · 三个停止条件:fixed point、计数式 cycle guard、迭代上限

要解决的问题:surrogate 是近似,迭代既不保证单调、也不保证收敛,可能落入循环;需要明确的
停机规则,既能识别真收敛,也能兜住不收敛的情形。

做了什么:主循环每轮依次算系数、解 MILP、真实评估、记 trace,然后按三级规则判停:
其一,fixed point——新 schedule 与上一轮完全相同,置收敛标志并退出:MILP 在"由这张 schedule
自己生成的通行时间"下仍选择它自己,这是交替优化意义上的不动点;
其二,计数式 cycle guard——记录每张出现过的 schedule 的出现次数,某张累计出现
MILP_CYCLE_TOL(3)次即退出。设成 3 而不是"一见重复就停",是因为 damping(第 12 步)在
不断改变冻结通行时间,旧 schedule 重现时系数已不同,再给它一两次机会可能跳出小的 2-cycle /
极限环;
其三,硬上限——循环最多 MILP_MAX_ITER(20)轮。

为什么这样做:三级规则从"确定性收敛"到"疑似循环"再到"预算耗尽"逐级兜底;配合第 13 步的
best-by-true-F,任何一种退出方式都不损害输出质量,区别只在多跑几轮 UE。

产物给谁用:退出后的历史列表与收敛标志交给第 13 步做选择;轮数进入 optima 表与 run_meta
的统计。
(util/pretrain_milp.py:155-172, config.py:44-45)

---

## 第 12 步 · damping(under-relaxation):平滑轨迹而不移动 fixed point

要解决的问题:原始交替(每轮全盘采纳新 UE 通行时间)会让 c_e^k 逐轮大幅跳动,MILP 解随之
振荡,容易在两张 schedule 之间打摆。

做了什么:每轮末尾不整体替换冻结通行时间,而是做 MSA 风格的凸组合更新

ũ ← λ·ũ_new + (1 − λ)·ũ_old, λ = MILP_DAMPING = 0.5

即新旧各取一半,让冻结通行时间——进而 c_e^k、进而 MILP 解——在迭代间渐变而非跳变。

为什么这样做:关键性质是 damping 只平滑轨迹、不改变 fixed point:若某份通行时间已是不动点
(新 UE 结果与它相同),凸组合仍等于它自身,稳态方程不因 λ 而变;λ 只决定"以什么路径走向
那里"。所以 0.5 是纯粹的轨迹工程——更小的 λ 更平滑但更慢,取 1.0 则退化为原始交替。

产物给谁用:混合后的通行时间是下一轮第 9 步的输入。
(util/pretrain_milp.py:169-171, config.py:47-48)

---

## 第 13 步 · best-by-true-F:收敛与否都无害的选择规则

要解决的问题:即便迭代停了,最后一轮的 schedule 未必是过程中最好的——surrogate 驱动的迭代
不单调,可能后期反而变差;必须决定"最终输出哪一张"。

做了什么:每一轮(含第 0 轮的 greedy 初始)的 schedule 与真实 F 都留在历史里;退出后对全历史
按**真实 F** 取最小者,作为该 scenario 的最终输出,并在 trace 里给那一轮打上 is_best 标记。

为什么这样做:每个 iterate 的真实 F 本来就为刷新通行时间而算过,挑最优不花任何额外代价;
这让整个 heuristic 的输出有下界保证——至少不劣于 greedy 初始——而"收敛与否"从正确性问题
降级为效率问题。这也是第 11 步三个停止条件可以从容设计的原因。

产物给谁用:最优 schedule 与其 F / F1 / F2 写入 milp_optima.csv;is_best 标记供过程图标注实际
返回的 iterate。
(util/pretrain_milp.py:174-178)

---

## 第 14 步 · 断点续跑:参数 fingerprint 决定 checkpoint 是否可信

要解决的问题:整个 run 以 UE 为主要成本,可能中途被打断;续跑前必须确认磁盘上的 checkpoint
出自"同一套参数"的 run,否则参数改动后旧结果会被误当作已完成。

做了什么:先取 oracle 模块的 _param_fingerprint——把 17 个影响 F 的参数(实例规模、目标权重、
损坏物理、需求模型、scenario 抽样、UE 收敛设置等,见 FINGERPRINT_PARAMS 清单;含 tuple 键的
字典先把键字符串化以求 JSON 稳定)序列化成排序 JSON 后做 SHA1;再把三个 MILP 专属回路参数
(MILP_DAMPING、MILP_MAX_ITER、MILP_CYCLE_TOL)拼接进去做第二次 SHA1,得到本 run 的
指纹。仅当三个 checkpoint 文件(milp_optima.csv、milp_trace.csv、milp_progress.json)都存在、
且 progress 里存的指纹与当前一致时,才把已有行读回内存、把已完成的 scenario 编号收进 done
集合;任何相关参数变动都会使指纹失配,checkpoint 被整体忽略、从头重跑。

为什么这样做:指纹分两层,是因为基础参数决定"问题本身",MILP 回路参数决定"求解轨迹",
两者任一变化,已存的 optima / trace 都不再代表当前配置的结果。用内容指纹而非时间戳,
续跑判定与文件系统状态无关,更可靠。

产物给谁用:done 集合控制第 15 步逐 scenario 循环的跳过逻辑;指纹随每次 checkpoint 写回
progress 文件。
(util/pretrain_milp.py:196-207, util/oracle.py:39-56)

---

## 第 15 步 · 逐 scenario 求解、checkpoint 与全量聚合

要解决的问题:M 份 scenario 各自独立求解,要边跑边落盘,任何时刻被打断都只损失当前
scenario;全部跑完后还要给出整个 run 的汇总统计。

做了什么:对每个未完成的 scenario 调一次交替优化(第 8-13 步),把最优结果整理成一行——
scenario 编号、F_milp、F1、F2、迭代轮数、是否收敛、耗时、UE 次数((轮数+1)·T)、时长组合、
每条 segment 的开工 slot——追加进 optima 行集;逐轮 trace(前置 scenario 编号)追加进 trace
行集;随即把两张表和 progress JSON 全量重写一遍(checkpoint),并打印该 scenario 的摘要。
所有 scenario 完成后再整体重写两张 CSV,并把总耗时、平均每 scenario 耗时、总 UE 次数、
每次 UE 平均秒数、平均迭代轮数、平均 F_milp 等汇总写入 run_meta.json。

为什么这样做:checkpoint 粒度选在 scenario 级,因为单个 scenario(几轮迭代 × T 次 UE)是
分钟级的——既不至于写盘过频,又把最坏损失控制在一个 scenario 之内。trace 全量落盘,
是为了让过程图与事后分析不必重跑任何 UE。

产物给谁用:milp_optima.csv 是第 16、17 步(过程图与 oracle 对比)的主输入;milp_trace.csv
喂过程图;run_meta.json 供人查账。
(util/pretrain_milp.py:210-243)

---

## 第 16 步 · 过程诊断图:迭代轨迹与耗时

要解决的问题:交替优化是 heuristic,必须能直观检查它的行为——每个 scenario 迭代了几轮、
真实目标怎样下降、surrogate 是否稳定、时间花在哪里。

做了什么:make_process_figures 从逐轮 trace 出两张诊断图。03_optimization_process.png 的
panel a 画每个 scenario 真实 F 的 best-so-far 曲线(单调下降),并以圆点标出实际返回的最优
iterate;panel b 画 MILP surrogate Σ c_e^k·x_e^k 随迭代的轨迹。04_runtime.png 分别画每 scenario
的 wall-clock 与迭代轮数。两张图直接使用内存中的 trace / optima 数据(与已落盘的 checkpoint
文件内容一致),不回灌任何计算。

为什么这样做:panel a 用 best-so-far 而不是逐轮原始 F,因为迭代本身不单调,best-so-far 才对应
第 13 步真正返回的东西;surrogate 面板用于判断冻结-刷新回路是否在振荡。

产物给谁用:outputs/pretrain_milp/n{N}/figures/ 下的 03、04 两张 PNG,供人工检查。
(util/pretrain_milp.py:245-246, viz/pretrain_viz.py:111-172)

---

## 第 17 步 · post-hoc oracle 对比:gap 的定义与它为什么可以为负

要解决的问题:回答"traffic-fixation MILP 离 ground truth 有多远"。oracle 已对每个 scenario 枚举了
全部 work-conserving schedule,留下逐 scenario 最优 F 与完整 landscape;直到这一步才允许读它们。

做了什么:按第 2 步的同一规则拼出 outputs/oracle/n{N}/ 路径;仅当该规模的 oracle_optima.csv
存在时,把 oracle 的逐 scenario 最优 F 按 scenario 号并进 MILP optima 表,逐 scenario 定义

gap = F_MILP − F_oracle

(绝对差;gap ≤ 0 表示 MILP 追平或打败了 oracle 最优的 work-conserving schedule),写出
milp_vs_oracle.csv,并出两张图:01_milp_vs_oracle.png 逐 scenario 并排画两个 F 及 gap 柱状;
02_cross_scenario_landscape.png 把每张固定 schedule 按跨 scenario 平均 F 排名画成曲线加范围带,
叠上 MILP(逐 scenario 自适应)的平均 F 与 hindsight(逐 scenario 最优)的平均 F,回答"若必须
承诺单一固定策略,各策略站在哪里,而自适应 MILP 又落在哪里"。若该规模的 oracle 结果尚未
算好,则打印提示并跳过对比——等 oracle 完成后运行 python -m util.pretrain_milp --landscape,
只从已存 CSV 重出对比图,不重跑任何 UE。

为什么 gap 可以为负:oracle 的"最优"仅是最优的 **work-conserving** schedule(按排列的列表排程、
不留空档);而第 10 步的 MILP 在"开工一次 + crew 上限 + horizon 内完工"约束下允许 crew 故意
空等,可行域是前者的严格超集,完全可能找到集合之外更好的 schedule。这不是 bug,而是搜索
空间更大的自然结果。针对"会不会是编码错误"的疑虑,模块另备 level_a 检查(命令行 --level-a
手动运行,不在主流程内):在同一份固定系数下,MILP 的 surrogate 最优值必须不小于全部
work-conserving schedule 的最好 surrogate 值(超集关系的直接推论),并逐 slot 复核 crew 上限与
horizon 约束。

产物给谁用:milp_vs_oracle.csv 与图 01、02 是本流水线的最终交付物,支撑"MILP 相对 ground
truth 表现如何"的结论。
(util/pretrain_milp.py:248-266, util/pretrain_milp.py:274-311, util/pretrain_milp.py:314-332, viz/pretrain_viz.py:21-108)

---

## 附录 A · 参数默认值(出自 config.py)

| 参数 | 默认值 | 在本流水线里的作用 |
|---|---|---|
| `DELTA_T_H` | 3.0 | 小时/slot;在 F2 里约掉,只作为 fingerprint 的一部分 |
| `C_MAX` | 2 | 列表排程与 MILP crew 上限约束共用的 crew 数 |
| `MU` | 1.0 | F = μ·F1 + (1−μ)·F2 的权重 ⇒ 目标只看 F1,F2 仅记录 |
| `CAP_RETAIN` | {1: 0.3, 2: 0.1, 3: 0.02} | 受损 link 的 capacity 保留系数 |
| `SPEED_RETAIN` | {1: 0.5, 2: 0.3, 3: 0.2} | 受损 link 的 free-flow time 除数(越小越慢) |
| `SEVER_SEVERITY` | 3 | severity 达到此值 ⇒ link 整条移除(真断连 → u_pen) |
| `F1_ACTIVE_ONLY` | False | F1 对整段 [1, T] 平均,与 c_e^k 的 1/T 归一化对齐 |
| `RHO` | 0.7 | recovery inertia;真实需求模型与 c_e^k 的 (1−ρ^{n+1}) 共用 |
| `KAPPA` | 1.0 | B(Φ) 里"损坏 → 需求缺口"的尺度 |
| `UPEN_FACTOR` | 10.0 | u_pen = 10 × 最大有限 baseline OD 通行时间 |
| `M_SCENARIOS` | 10 | duration scenario 份数 |
| `SEED` | 42 | scenario 抽样的 RNG seed(与 oracle 端一致) |
| `N_DISRUPTED_ORACLE` | 4 | 实例规模;决定 n{N} 输出目录 |
| `UE_RGAP` | 1e-6 | 单次 UE 的 relative-gap 目标 |
| `UE_MAX_ITER` | 100 | 单次 UE 的最大 Frank-Wolfe 迭代数 |
| `MILP_MAX_ITER` | 20 | 交替循环硬上限(停止条件三) |
| `MILP_CYCLE_TOL` | 3 | 计数式 cycle guard 的重复阈值(停止条件二) |
| `MILP_DAMPING` | 0.5 | 通行时间 under-relaxation 的 λ |
| `DURATION_SUPPORT` | 按 (road_class, severity) 的支撑集 | scenario 抽样的 base 时长来源(Table 1) |
| `ETA` | [0.8, 1.0, 1.2] | crew-efficiency 乘子 |

(config.py:8-56)
