# 01 — `run_oracle()`:brute-force oracle 的解题逻辑

**oracle 要解决的总问题。** 对一批被摧毁的路段,枚举**所有** work-conserving 的修复 schedule(即这些路段的全部 permutation),在 M 个随机抽出的 duration scenario 下,对每个 schedule 计算**精确的** Figure-1 目标。总目标是加权组合

> F = μ·F1 + (1−μ)·F2

其中 F1 衡量恢复期内 accessibility 的退化程度(需要逐时段做交通均衡),F2 衡量修复工期的效率(纯排程算术)。最终产出两样东西:每个 scenario 下的事后最优(hindsight optimum)F\*,以及完整的 landscape(每一个被测过的 schedule 及其 F/F1/F2)。这就是日后验证 §2.1.1 pretraining MILP 的 ground truth(附录 B 的 Level B 验证)。入口是 `run_oracle()`,定义在 util/oracle.py:122。

下文按执行顺序写出十三个逻辑步骤,附三个附录(参数默认值表、验证层次、建模 caveats)。

---

## 第 1 步 · 确定输出目录与缓存布局

**要解决的问题:** 同一套代码会在不同问题规模(受损段数 N)下反复运行,不同规模的结果不能互相覆盖;同一规模下,参数变了要重算,参数没变要能复用。所以在做任何计算之前,先把"结果放在哪、以什么为 key"定下来。

**做了什么:** 输出根目录固定为 outputs/oracle/,在它下面按当前受损段数开一个规模专属子文件夹(形如 n4/),并预建 figures/ 子目录。缓存因此是两级 key:**规模 N 决定文件夹**,而**参数指纹(第 7 步的 fingerprint)决定该文件夹内结果的新鲜度**。

**为什么这样做:** 规模不同,枚举出的 schedule 集合完全不同,天然应该各占一份缓存;而同规模下改任何一个影响目标值的参数,旧结果就作废——把这两类失效分开,既避免误覆盖,也让"换个 N 跑一跑再换回来"不丢缓存。

**产物给谁用:** 一个规模专属的输出目录,后续所有 CSV、图、meta 都写进这里;第 7 步的缓存判断和第 9 步的断点续跑都在这个目录里找文件。
(util/oracle.py:59-62,util/oracle.py:123-124)

---

## 第 2 步 · 选定受损实例:哪些边被摧毁、各自多严重

**要解决的问题:** 得先决定 disruption set。这批受损边不能随便选——如果所有边重要性差不多,修复顺序对 F1 的影响就很弱,枚举出来的 landscape 会平得看不出优劣,oracle 就起不到 ground-truth 的作用。所以要**故意混搭关键路与次要路**,让"先修哪条"对目标值有强影响。

**做了什么:** 先给每条无向边算一个重要性分数——**在完好网络、正常需求下自算一次 baseline user equilibrium,把每条边两个方向的均衡流量加总**,作为这条边的重要性。然后按流量从高到低排序:流量最高的前两条定为"关键边",赋 severity 3(达到整条移除的阈值,会制造真正的断连);余下名额用等间隔的方式铺在**较低流量**的边上(从大约 20% 分位一路取到最不常用的边),severity 在 2 和 1 之间交替。每条边再拼出形如 highway-S3 的 level 标签,按 edge_id 排序后写一份 CSV 到 disruption/ 目录,同时返回这张受损段表。整个过程完全确定性,不含随机数。

**为什么这样做:** 用 baseline UE 流量当重要性代理,是因为流量越大的边一旦断掉,对全网 accessibility 的冲击越大;把最关键的两条设为"整条移除",保证拖着不修会触发断连重罚(第 3 步的惩罚值),从而让顺序真正值钱;次要边刻意取低流量段,拉开与关键边的对比。值得强调的是,**baseline 流量由项目自己的 UE 引擎现场算出,不再读开源参考解文件** raw/SiouxFalls_flow.tntp——这去掉了对随附外部文件的依赖,日后换别的网络/OD 数据集不需要再配一份 reference-flow 文件。经验证,换成自算流量后选出的实例与旧参考解完全一致(edge_id 为 1、12、15、17,severity 为 1、2、3、3;逐边流量差约 0.14%)。那份 .tntp 参考解现在只被 util/ue.py 末尾的 UE 自校验用到,与 oracle 无关。另注:旧文档曾提到此函数带一个 seed 参数,现已删除——选择过程本来就由流量唯一决定,seed 是多余的。

**产物给谁用:** 受损段表(边、端点、road_class、severity、level 标签)交给第 3 步构建 context、第 4 步抽 scenario、第 12 步画图;从中取出的 edge_id 规范排序列表则是后面 permutation、horizon、landscape 列名共同的基准。
(util/oracle.py:85-109 选边与赋 severity;util/oracle.py:65-82 自算两向 baseline 流量;util/oracle.py:126-127 调用与取 segments)

---

## 第 3 步 · 构建静态 context:一次算好、全程复用的背景

**要解决的问题:** 后面要做"permutation 数 × M 个 scenario × T 个时段"这么多次评估,每次都依赖同一套与 schedule 无关的背景——网络结构、正常需求、灾前通行时间基准、需求受损的敏感度矩阵、断连惩罚。这些只该算一次。

**做了什么:** `build_context` 把这套背景一次性备好,装进一个 dict,供之后每一次 `evaluate_schedule` 复用。它依次完成以下六件事。
(util/evaluate.py:106-150;util/oracle.py:128 调用)

### 3a · 载入网络与建立下标映射

从 toy 数据目录读三张 CSV:无向边表(含 capacity、length、free_flow_time、BPR 参数、road_class)、OD 对表(含正常时期需求)、节点表;Sioux Falls 里每个节点都是一个 OD zone。随后把节点 id、OD 对、边 id 都映射成稠密的 0 起始下标,并把 OD 对表里的正常需求取出为向量 H^{t0}(这同时是"完全恢复"的目标水平)。**为什么先做映射:** UE 求解和矩阵构造都按稠密下标操作,预先建好映射避免热循环里反复查表。产物是 context 的骨架:边表、zone 列表、OD 对、正常需求、以及各种下标映射。
(util/io.py:10-23 载入;util/evaluate.py:108-120 映射与骨架)

### 3b · 灾前通行时间基准:在完好网络上做一次 UE

**要解决的问题:** F1 是"实际通行时间 / 灾前基准通行时间"的比值,需要一个分母——正常情况(网络完好、需求正常)下每个 OD 对的**拥堵**通行时间 u_r^{t0}。**做了什么:** 在原始边表和正常需求上跑一次完整 user equilibrium,再在均衡链路成本上求每个 OD 对的最短路时间。**为什么:** 基准必须出自与灾后同一套 UE 模型,否则灾前灾后不可比;它与受损实例、schedule 都无关,所以只算一次。**产物给谁用:** 基准通行时间向量是每个时段 F1 项的分母参考,也是断连惩罚(3d)的量级来源。
(util/evaluate.py:122-125)

### 3c · UE 求解本身扮演的角色(solve_ue 与 Frank-Wolfe 思想)

整个 oracle 里所有"交通怎么走"的问题都交给同一个入口 `solve_ue`:给它一张(可能已受损的)边表和一个 OD 需求矩阵,它返回每条有向链路的均衡流量与拥堵通行时间。其内部把无向边表转成 AequilibraE 可路由的双向图、把稠密需求矩阵包成 AequilibraE 矩阵,然后交给 AequilibraE 的 bi-conjugate Frank-Wolfe 求解器。**方法思想**(细节属第三方库,这里只概括):user equilibrium 是"没有任何出行者能靠单方面换路线缩短自己时间"的流量形态,等价于最小化 Beckmann 目标

> Z(x) = Σ_a ∫₀^{x_a} t_a(w) dw ,其中 BPR 拥堵成本 t_a(x) = t0_a · (1 + α·(x_a/c_a)^β)

Frank-Wolfe 反复做三件事:按当前成本做 all-or-nothing 分配得到辅助流量、朝它做线搜索移动使 Z 下降、更新成本,直到相对 gap 小于收敛阈值;bfw 是收敛更快的双共轭变体。项目层面只依赖它的输入输出契约:边表 + 需求进,逐链路流量 + 拥堵时间出;收敛参数用 config 里为逐时段求解调松的阈值(相对 gap 1e-6、上限 100 次迭代)。
(util/ue.py:81-121 入口;util/ue.py:12-28 方法思想;util/ue.py:46-68 建图;config.py:40-41 收敛参数)

### 3d · 从链路时间到 OD 通行时间

**要解决的问题:** UE 给的是逐链路的拥堵时间,而 F1 要的是逐 OD 对的通行时间。**做了什么:** 用均衡链路成本建一张有向图,对每个唯一的 origin 跑一次单源 Dijkstra,得到它到所有可达节点的最短时间,再逐 OD 对取值;起讫点不连通时记为无穷大。**为什么按 origin 缓存:** 多个 OD 对常共享同一 origin,单源最短路一次算完所有目的地,省去重复。**产物:** 与 OD 对对齐的通行时间向量,3b 的基准和第 10 步每个时段的实际值都由它产生。
(util/evaluate.py:27-45)

### 3e · 断连惩罚 u_pen

**要解决的问题:** 受损网络里可能有 OD 对被彻底断连,最短路为无穷大,无法直接进 F1。**做了什么:** 预先取一个大惩罚值,等于"最大有限灾前 OD 通行时间"的 10 倍(倍数由 config 的 UPEN_FACTOR 给出);评估时把无穷大替换成它。**为什么:** 断连必须比任何"很堵但还通"的状态都糟,否则"把关键边拖着不修"不会在目标里受到应有的重罚;取灾前最糟出行的固定倍数,量纲与 F1 的分母一致,惩罚强度也可解释。**产物给谁用:** 第 10 步每个时段替换无穷大通行时间时使用。
(util/evaluate.py:126;config.py:31)

### 3f · 需求缺口敏感度矩阵 B(Φ)

**要解决的问题:** 灾后出行需求会下降,且降多少应当与"哪条受损边落在这个 OD 的常走路径上、有多严重"挂钩。需要一个静态的敏感度矩阵,评估时用它把当下的损伤状态线性映射成各 OD 的需求缺口。**做了什么:** 在**free-flow**(不拥堵)的无向图上,为每个 OD 对求一条最短路;若某条受损边恰好在这条路径上,就在矩阵对应位置写入

> B[r,e] = κ · (h_r^{t0}/3) · 1{受损边 e 在 OD r 的 free-flow 最短路上}

即该 OD 正常需求的三分之一乘以尺度系数 κ(默认 1)。**为什么用 free-flow 最短路:** 它是"正常时期这个 OD 主要走哪条路"的稳定代理,不随拥堵状态漂移,保证 B 是纯静态的、可预计算的。**产物给谁用:** 第 10 步每个时段用"B 乘以当前损伤向量"得到损伤驱动的需求缺口。
(util/evaluate.py:128-148;config.py:28)

### 3g · severity 向量(为 MILP 预备,oracle 本身不用)

context 里还按受损段顺序存了一个 severity 向量。需要澄清:**旧文档称 `evaluate_schedule` 会读它——与现行代码不符**。当前的 `evaluate_schedule` 完全不取用这个向量,逐时段的实时 severity 直接来自受损段元组;这个向量的真正消费者是 §2.1.1 的 pretraining MILP(util/pretrain_milp.py:61)。它放在 context 里只是因为 context 是两条流水线共享的静态背景。
(util/evaluate.py:149;util/evaluate.py:159-161 可见 evaluate_schedule 只解包受损段、需求、B 和基准)

---

## 第 4 步 · 抽样 duration scenarios

**要解决的问题:** 修好一条边要几个 slot 是不确定的:同一 disruption,不同的现实(施工队效率、损伤实际程度)给出不同工期。oracle 必须在一批工期情景上评估,才能看出 schedule 是否稳健、事后最优随情景怎么变。

**做了什么:** 用固定 seed 的随机数发生器抽 M 个 scenario。工期不是按边抽,而是按 **level**——即 (road_class, severity) 组合——抽:每个 level 先从 Table 1 的支撑集里抽一个基础工期,再乘上从 {0.8, 1.0, 1.2} 里抽的 crew 效率乘子 η,四舍五入且至少为 1;同一 scenario 内、同一 level 的所有受损段共享这一个工期。每个 scenario 落地为"每条受损边多少个 slot"的映射。

**为什么按 level 抽:** 这是论文的建模设定 d_e(ω) = d_{ℓ(e)}(ω)——工期的随机性挂在"路类 × 严重度"这一层,同类同严重度的段没有理由抽出不同工期。**为什么固定 seed:** 抽取顺序(按排序后的 level 集合、逐 scenario)加固定 seed 使整批 scenario 完全可复现,这也是 seed 和支撑集参数都进 fingerprint 的原因。

**产物给谁用:** M 个"边到工期"的映射,交给第 6 步定 horizon、第 8 步 probe、第 10 步主循环和第 12 步画图。
(util/scenarios.py:14-28;util/oracle.py:129 调用;config.py:51-56 支撑集与 η)

---

## 第 5 步 · 枚举全部 permutation

**要解决的问题:** 把"所有可能的修复顺序"穷举出来,作为待评估的 schedule 全集。

**做了什么:** 直接取受损边 id 排序列表的全部排列。N=4 时共 4! = 24 个。每个排列代表一个修复**优先级顺序**,交给 work-conserving list scheduling(第 6 步描述)变成具体开工时刻。

**为什么可以只枚举 permutation:** 依据附录 C 的 work-conserving 归约——假设让施工队闲置永远不划算,则最优 schedule 必对应某个优先级排列,搜索空间从连续的开工时刻压缩到 N! 个离散点。

**产物给谁用:** permutation 列表交给第 6 步(算 horizon)和第 10 步(主循环逐一评估)。
(util/oracle.py:130)

---

## 第 6 步 · 统一 horizon T

**要解决的问题:** F1 是逐时段项在 [1, T] 上的平均。若每个 schedule 各用各的时间窗,平均出来的 F1 互相不可比。必须让所有被枚举的 schedule 共享同一个 horizon,且每个都能在其内完工。

**做了什么:** 对每一个 permutation × scenario 组合,先把排列变成具体 schedule,再取其最后完工 slot;全体的最大值即全局 T。把排列变成 schedule 的规则是 work-conserving list scheduling:C_max(默认 2)台完全相同的施工队都从 slot 1 起空闲(严格晚于灾发 slot 0),按优先级顺序把每条边派给最早空闲的那台,队伍被占用到该边完工——**任何队伍一空下来立刻接下一条边,绝无闲置**。最后完工 slot 就是所有边"开工加工期"的最大值。

**为什么取全体最大而不是逐 schedule 各取各的:** 统一 T 保证"平均"这个操作对每个 schedule 都在同一分母上;取最大值保证没有任何 schedule 被截断。

**产物给谁用:** 整数 T 交给第 8 步(probe 的换算)、第 10 步(逐时段循环上限)和第 12 步(图的横轴范围);随后打印一行实例摘要(段数、排列数、M、T)。
(util/oracle.py:112-119 取最大完工 slot;util/evaluate.py:82-91 list scheduling;util/evaluate.py:94-95 完工 slot;util/oracle.py:131-132 调用与打印)

---

## 第 7 步 · 缓存判断:fingerprint 与 meta.json

**要解决的问题:** 完整枚举很贵(几十分钟量级)。如果参数没变,不该重算;但"参数没变"必须判断得精确——任何影响目标值的参数变了都要作废旧结果。

**做了什么:** 把所有会影响 F 的参数(受损段数、μ、损伤物理参数、需求动态参数、惩罚倍数、slot 长度、施工队数、scenario 数与 seed、UE 收敛参数、工期支撑集与 η——完整清单见 util/oracle.py:39-43)收进一个 dict,序列化成排序后的 JSON,取 SHA-1 作为 fingerprint。然后读输出目录里的 meta.json:若存在且其中记录的 hash 与当前 fingerprint 相等,即为命中——重读已存的 landscape 与 optima 两张 CSV、重画图、直接返回,**一次 UE 枚举都不做**。带 `--probe` 或 `--force` 时跳过此判断。

**为什么用参数指纹而不是文件时间戳:** 结果的有效性只取决于"是用什么参数算的",与何时算无关;把参数值本身连同 hash 一起存进 meta.json(第 13 步),也让人能事后核对缓存对应的确切配置。字典类参数的 key 先转成字符串,保证 JSON 序列化稳定。

**产物给谁用:** 命中则流程到此结束(图也已重画);未命中则带着 fingerprint 继续,第 9 步的断点判断和第 13 步的 meta 写入都复用同一个 fingerprint。
(util/oracle.py:39-43 参数清单;util/oracle.py:46-56 指纹计算;util/oracle.py:135-148 命中判断)

---

## 第 8 步 · 单次评估计时与总时长预估(probe)

**要解决的问题:** 完整跑之前先知道大概要多久,好决定实例规模是否合适、要不要真跑。

**做了什么:** 真评估一次——取第一个 permutation 在第一个 scenario 下的 schedule,跑一遍完整的 `evaluate_schedule`(即 T 次 UE 求解),计墙钟时间;除以 T 得到"每次 UE 的秒数",再按"排列数 × M 个 schedule、每个 T 次 UE"外推出全量运行的预计分钟数并打印。带 `--probe` 时到此返回,不做枚举。

**为什么真跑一次而不是估计:** 单次 UE 的成本由 AequilibraE 的固定 setup 开销主导,与理论复杂度关系不大,实测最可靠;而一个 schedule 恰好等于 T 次 UE,是天然的计时单元。需要注意 probe 的结果**不**计入正式结果——主循环会把这一对 (permutation, scenario) 从头重评,保证 landscape 里每一行的来源一致。

**产物给谁用:** 每次 UE 的秒数存下来供第 12 步 summary 引用;预估打印给人看。
(util/oracle.py:150-159)

---

## 第 9 步 · 断点续跑:partial checkpoint

**要解决的问题:** 全量枚举可能跑几十分钟,中途被打断(断电、Ctrl-C)不该让已算的部分白费。

**做了什么:** 主循环每做完一个 scenario 就把当前全部结果行重写进 landscape CSV,并把"fingerprint + 已完成 scenario 集合"写进一个进度标记文件(第 10 步)。重新启动时,若这两个文件都在、且进度标记里的 hash 与当前 fingerprint 一致,就把已算的行读回内存、把那些 scenario 标记为已完成,主循环只算余下的。`--force` 禁用续跑;fingerprint 不匹配则视为过时的半成品,从头开始。

**为什么以 scenario 为断点粒度:** 一个 scenario 内的所有 permutation 评估是一个自然的工作块(共享同一套工期),块间无依赖;粒度再细(逐 schedule)会让 checkpoint 写入过于频繁,再粗就失去续跑意义。**为什么进度标记也带 hash:** 防止"参数改了,却把旧半成品接着跑"这种静默错误。

**产物给谁用:** 已完成行与已完成 scenario 集合交给第 10 步的主循环,让它跳过重复劳动。
(util/oracle.py:161-172 续跑判断;util/oracle.py:193-194 checkpoint 写入)

---

## 第 10 步 · 主枚举循环与 evaluate_schedule:精确目标的完整评估

**要解决的问题:** 这是 oracle 的主体——对每个 scenario 下的每个 permutation,算出精确的 F(x|ω),并把每一次评估都记录在案。

**做了什么(外层循环):** 对每个 scenario(跳过续跑已完成的):对每个 permutation,先用第 6 步的 list scheduling 把排列变成开工时刻,再调 `evaluate_schedule` 得到 F/F1/F2,连同 scenario 编号、以短横线连接的排列串、单次评估墙钟秒数、以及每条边的开工 slot,存为一行。一个 scenario 做完即写一次 checkpoint(见第 9 步)。**为什么每行都带每边开工时刻:** 让 landscape 不仅能比较目标值,还能事后重建任何一个 schedule(第 12 步的图 03 正是这么做的)。
(util/oracle.py:174-195)

以下是 `evaluate_schedule` 的内部逻辑——给定一个 schedule(每边开工 slot s_e)、一个 scenario(每边工期 d_e)和 horizon T,它按 Figure 1 的流水线算出 F。
(util/evaluate.py:156-206)

### 10a · F2:纯排程算术,先于一切、不需要 UE

**问题与做法:** F2 衡量"修这么点活拖了多久",定义为

> F2 = makespan / Σ_e d_e

即最后完工 slot 除以总工作量 slot 数(slot 时长 Δt 在分子分母同现,约掉)。**为什么放在逐时段循环之外:** 它只依赖 schedule 的时序,与交通完全无关,一步算完。**产物:** F2 值,等最后与 F1 加权合成。
(util/evaluate.py:164;util/evaluate.py:98-100)

### 10b · 逐时段循环之一:损伤状态

对每个 slot k = 1…T,先确定"此刻哪些边仍受损"。规则是(论文 Eq. 2):**边 e 在 slot k 上仍受损,当且仅当 k < s_e + d_e**;一旦到达完工 slot 即刻恢复。由此得到实时损伤向量

> v_e^{t_k} = severity_e · 1{k < s_e + d_e}

已修复的边分量为 0。**为什么如此简单:** 修复过程被建模为"开工到完工期间维持原 severity、完工瞬间彻底恢复"的阶跃,这正是 schedule 影响一切后续量的唯一通道。**产物:** 当前受损集合与损伤向量,分别驱动 10d 的网络构造和 10c 的需求动态。
(util/evaluate.py:171-172)

### 10c · 逐时段循环之二:需求骤降与惯性恢复

**要解决的问题:** 灾后出行需求先骤降、后随道路修复渐渐回升;需要一个由损伤状态驱动、且带恢复惯性的需求轨迹。**做了什么:** 维护一个需求缺口向量 D,初值为零,逐时段更新:

> D_t = max(B·v_t, ρ·D_{t−1}) , H_t = max(0, H^{t0} − D_t) , D_0 = 0

即缺口至少是"当前损伤经 B 映射出的水平",且上一时段的缺口只按系数 ρ(默认 0.7,越接近 1 恢复越慢)衰减;实际需求是正常需求减去缺口、下限截零。**为什么写成 shortfall 形式:** 论文字面的需求递推会让需求衰减到零(错误),shortfall 形式才给出"灾发瞬间急降、随修复渐回正常"的正确形状,详见附录 C。**产物:** 当前需求向量,既作为本时段 UE 的出行需求,也作为 F1 项的权重。
(util/evaluate.py:174-176;config.py:22-28)

### 10d · 逐时段循环之三:构造当前受损网络

**要解决的问题:** 本时段的 UE 必须跑在"此刻的路网"上。**做了什么:** 复制原始边表,对每条仍受损的边:severity 达到移除阈值(默认 3)的,整行从边表删掉——制造真正的断连,受影响 OD 稍后吃惩罚值;较轻的,把 capacity 乘上保留系数、free_flow_time 除以速度保留系数(两组系数按 severity 分档,见附录 A)。已完工和未受损的边原样不动。**为什么用"改边表"而不是改图:** UE 入口只认边表,逐时段临时改一份最简单,也保证每次求解都是无状态的。**产物:** 当前受损边表,交给本时段的 UE。
(util/evaluate.py:178;util/evaluate.py:54-76)

### 10e · 逐时段循环之四:UE 求解与本时段 F1 项

在受损边表和降低后的需求上做一次 UE(与 3c 同一入口、同一收敛参数),再由均衡链路成本求各 OD 的实际通行时间;不通的 OD 以断连惩罚值代入,得到修正后的通行时间 ũ_r。本时段的 F1 项是**需求加权的"实际 / 灾前"通行时间比**:

> term_k = Σ_r H_r^{t_k}·ũ_r^{t_k} / Σ_r H_r^{t_k}·u_r^{t0}

分母若为零(需求全灭的极端情形)则该项取 1。**为什么分母也用当前需求加权:** 分子分母对同一批出行者加权,比值才是"这批人现在比灾前多花多少倍时间"的干净度量;若分母固定用正常需求加权,需求下降本身就会扭曲比值。同时记录本时段是否仍有损伤(供聚合选项用),需要画图时还可留存逐时段的需求总量与 F1 项轨迹。
(util/evaluate.py:180-192)

### 10f · F1 聚合与 F 合成

T 个时段的项取平均得到 F1。默认在**整段 horizon [1, T] 上平均**:

> F1 = (1/T) · Σ_{k=1..T} term_k

config 里留有一个开关(默认关),打开则只在"仍有损伤"的时段上平均。**为什么默认全程平均:** 与 §2.1.1 MILP surrogate 对系数采用的 1/T 归一化保持一致,两边才可比。最后合成

> F = μ·F1 + (1−μ)·F2

默认 μ = 1.0,即 F 就是 F1(accessibility-only);F2 仍被算出并记录,只是权重为零。返回 F/F1/F2,按需附带逐时段通行时间矩阵(给 MILP 取 c_e^k 用)和轨迹表(给图 03 用)。
(util/evaluate.py:194-206;config.py:11-13,19-20)

---

## 第 11 步 · landscape 排序与 per-scenario 最优

**要解决的问题:** 把主循环攒下的所有评估行整理成两样成品:完整 landscape,和每个 scenario 的事后最优。

**做了什么:** 全部行按 (scenario, permutation) 排序后写成最终的 oracle_landscape.csv;再在每个 scenario 内取 F 最小的那一行,拼成 oracle_optima.csv——这就是每个 scenario 的真实事后最优 schedule 及其 F\*。

**为什么两样都要:** 最优值用于"MILP 差多远"的核心验收;完整 landscape 则支撑更细的比较(MILP 解排第几名、landscape 是尖峰还是平台,见附录 B)。这两个文件也正是第 7 步缓存命中时直接重读的东西。

**产物给谁用:** 两张 CSV 交给第 12 步的 summary 与图,也留给日后的 MILP 对比。
(util/oracle.py:196-201)

---

## 第 12 步 · summary 与图

**要解决的问题:** 给人一份一眼能读的运行总结,和三张说明"landscape 长什么样、最优 schedule 为什么好"的图。

**做了什么(summary):** 组装并写 summary.txt,同时回显:实例规模(段、排列数、M、T)、每次 UE 的毫秒数、评估过的 schedule 总数、总评估计算量与本次会话时长、各 scenario 最优的 F\*/F1\*/F2\* 均值,以及一句"oracle 最优是每个 scenario 内全部排列的最小值(真实事后最优)"。
(util/oracle.py:203-214)

**做了什么(图):** 调 `make_figures` 出三张图(第 7 步缓存命中时调的是同一个函数;另有一个 `--figs` 入口可只重画图不重算)。图的作用概括如下:图 01 把每个 scenario 的 F 从最好到最差排序叠画,展示跨 scenario 的范围带与 oracle 最优水平线——即"选对顺序值多少";图 02 把代表性 scenario 的所有 schedule 画在 (F2, F1) 平面上、按 F 着色,展示双目标结构;图 03 取代表性 scenario 的最优行,由 landscape 里存的开工时刻**重建 schedule 并重跑一次评估**收集轨迹,画三联面板——最优 schedule 的 Gantt(按 severity 着色)、逐时段总需求(骤降后回升,虚线为正常水平)、逐时段 F1 项(虚线 1 为完全恢复)。绘图风格统一走 viz/style.py 的出版级设定(rcParams、severity 暖色渐变、面板字母、默认只存 600 dpi PNG——**oracle_viz.py 顶部 docstring 声称同时写 SVG/PDF,与 save_pub 的默认行为不符**,以代码为准)。

**产物给谁用:** summary.txt 与 figures/ 下三张 PNG,给人;不反馈给任何后续计算。
(util/oracle.py:216-217 调用;viz/oracle_viz.py:23-110 三张图;viz/style.py:59-61 与 78-86 风格与保存;util/oracle.py:230-242 仅重画图的入口)

---

## 第 13 步 · meta.json 写入与 checkpoint 清理

**要解决的问题:** 把"这次结果是用什么参数算的"钉死,供下次缓存判断;并清掉不再需要的续跑标记。

**做了什么:** 写 meta.json,内容为:fingerprint(hash)、参与指纹的全部参数值、以及一个 timing 块(总评估秒数、schedule 数、每 schedule 与每 UE 的秒数、本次会话秒数、逐 scenario 秒数)。随后删除进度标记文件。

**为什么必须删标记:** 这次已完整跑完,标记若残留,下次会被误判为"上次被中断"。**为什么参数值全文入 meta:** hash 只能判断"变没变",参数原值让人能核对"当时到底是什么"。

**产物给谁用:** meta.json 是第 7 步缓存判断的依据——至此闭环:下次同参数再运行,读到匹配的 hash 就直接复用,不再枚举。
(util/oracle.py:219-226)

---

# 附录 A — 参数默认值表

以下均为**当前** config.py 的默认值(唯一真实来源)。除注明"给定"外,每一项都是建模假设。

| 参数 | 当前默认 | 在 run_oracle 里的角色 | config.py 行 |
|---|---|---|---|
| DELTA_T_H | 3.0 h/slot(给定) | slot 的物理时长;在 F2 里约掉 | 9 |
| C_MAX | 2 | list scheduling 的施工队数 | 10 |
| MU | 1.0 | F = μF1 + (1−μ)F2;1.0 ⇒ accessibility-only(F2 照算,权重 0) | 11 |
| CAP_RETAIN | {1:0.3, 2:0.1, 3:0.02} | 受损 capacity = 保留系数 × capacity | 15 |
| SPEED_RETAIN | {1:0.5, 2:0.3, 3:0.2} | 受损 free_flow_time = 原值 ÷ 保留系数 | 16 |
| SEVER_SEVERITY | 3 | severity 达到此值 ⇒ 整条移除(断连 → 惩罚值) | 17 |
| F1_ACTIVE_ONLY | False | F1 在整段 [1,T] 平均(对齐 MILP 的 1/T 归一化) | 19 |
| RHO | 0.7 | 恢复惯性:D_t = max(B v_t, ρ D_{t−1}) | 27 |
| KAPPA | 1.0 | 损伤→缺口尺度:B[r,e] = κ(h_r^{t0}/3)·1{在最短路上} | 28 |
| UPEN_FACTOR | 10.0 | 断连惩罚 = 10 × 最大有限灾前 OD 通行时间 | 31 |
| M_SCENARIOS | 10 | duration scenario 数量 | 34 |
| SEED | 42 | scenario 抽样的 RNG seed(可复现) | 35 |
| N_DISRUPTED_ORACLE | 4 | 受损集合大小(4! = 24 个 schedule);兼作 n{N}/ 缓存文件夹 key | 36 |
| UE_RGAP | 1e-6 | 逐时段 UE 的相对 gap 目标(比 1e-12 的自校验松) | 40 |
| UE_MAX_ITER | 100 | 逐时段 UE 迭代上限 | 41 |
| DURATION_SUPPORT | Table 1,按 (road_class, severity) | 基础工期支撑集 | 51-55 |
| ETA | [0.8, 1.0, 1.2](给定) | crew 效率乘子取样集 | 56 |

> **与更早期文档的差异:** 已退役的 oracle_validation.md 曾记 CAP_RETAIN={1:0.7, 2:0.4, 3:0.1}、SPEED_RETAIN={1:0.8, 2:0.6, 3:0.4}、μ=0.5;现行 config.py 已改为上表更严的保留系数与 μ=1.0,以现行代码为准。另:config.py 顶部 docstring 仍写着"mirrored in TECHNICAL_NOTE/oracle_validation.md",该文件已退役,此说法已过时。

**不参与 oracle 的 config 项**(供 §2.1.1 MILP 用):MILP_MAX_ITER=20、MILP_CYCLE_TOL=3、MILP_DAMPING=0.5(config.py:44-48)。

---

# 附录 B — validation levels 与 comparison metrics

(沿用自已退役的 oracle_validation.md。)oracle 通过枚举所有 work-conserving schedule、用精确的 Figure-1 目标打分,得到每个 scenario 的**真实**事后最优 F\*。日后验证 pretraining MILP 时可在两个层次检查:

- **Level A — 表述正确性。** 暴力枚举 MILP **自己的 surrogate** 目标(固定通行时间 + 线性化 F2),确认 MILP 返回同一个最小值点。用来隔离 MILP 编码里的 bug(约束、系数、makespan 线性化)。
- **Level B — 近似质量。** 暴力枚举**真实的** F(完整 UE 流水线——即 run_oracle 做的事),把 MILP 解的真实 F 与真实最优 F\* 比。检验 traffic-fixation 加线性化是否真能给出真实问题的(近似)最优 schedule。
- **当前决定:** oracle 只做 Level B;Level A 等 MILP 建好再说。

**Comparison metrics(未来 MILP vs oracle),全部可由 oracle_landscape.csv 算出:**

- **objective gap:** MILP 解的 F 与 F\* 的(相对)差——最核心的验收数。
- **rank:** MILP 解在全部枚举 schedule 里排第几("24 中第 1"或"前 X%")。
- **schedule match:** MILP 解与事后最优是否同一个 schedule。若不同但 F 相等,说明存在多重最优——比 F,别比 argmin。
- **landscape shape:** 尖峰型还是宽平台型,决定合理的 MILP 验收容差。

---

# 附录 C — Figure-1 fidelity 与建模 caveats

(沿用自已退役的 oracle_validation.md,并按现行代码核对。)评估器逐步照搬论文 Figure 1,只在 Figure 1 本身没说清或不完整之处才偏离:

- **F2 在逐时段循环外算**:它不需要 UE,是纯排程算术(第 10a 步)。
- **灾前基准 u_r^{t0}** 定义为"在未受损网络、正常需求上做一次 UE"所得的 OD 拥堵通行时间——Figure 1 在 F1 里用了它但没说它怎么来(第 3b 步)。
- **F1 可以低于 1。** F1 是需求加权的,而灾后需求下降,较少的出行者走在按"正常需求的拥堵水平"定出的基准之下;在低需求的恢复窗口里,需求加权比值可以掉到 1 以下。论文说的"完全恢复时等于 1"只在网络和需求都回到正常后成立。恢复越快 F 越低(更多时段处于已恢复态),优化信号方向仍然正确。

**Demand model(骤降 → 恢复)。** 开源 OD 给的是正常时期需求 H^{t0}(也是完全恢复的水平)。论文字面的需求递推(上一时段需求乘衰减系数再加损伤项)会让需求衰减到零,是错的。代码建模的是**缺口(shortfall)**:

> D_t = max(B·v_t, ρ·D_{t−1}) , H_t = max(0, H^{t0} − D_t) , D_0 = 0
> B[r,e] = κ · (h_r^{t0}/3) · 1{e 在 OD r 的 free-flow 最短路上}

含义:灾发瞬间需求急降到当前损伤驱动的缺口水平,随后随道路修复、缺口以速率 ρ 衰减,需求逐步回到正常。κ 调降幅深浅,ρ 调恢复快慢(直观效果见 figures/03_best_schedule.png 的中间面板)。

**Caveats。**

- **Work-conserving 归约。** schedule 空间被压缩为"permutation + 无闲置的 list scheduling",隐含假设是让施工队闲置永远不会改善 F1 或 F2。在动态需求下,需求与损伤轨迹耦合,这个假设只是**近似**成立——若结果反常,应先复查它。好处是把枚举保持在 N! 的规模。
- **没有 UE 结果缓存。** 由于需求是动态的,某一时段的 UE 不仅取决于"哪些边已修好",还取决于到达该状态的**路径**(需求缺口带惯性),所以按"已完成集合"做 2^N 缓存不成立——每个时段都是全新的 UE 求解。
- **AequilibraE 单次调用成本。** 每次 UE 约 1 秒上下,主要是每次调用的固定 setup 开销(24 节点的小网上纯计算很快)。这把暴力枚举限制在小实例(N=4、M=10);要上更大实例,需换成常驻内存的自研 Frank-Wolfe。
- **源码内两处过时注释**(不影响逻辑,以行为为准):util/oracle.py:2 的模块 docstring 仍写"5 disrupted segments",实际由 config 的 N_DISRUPTED_ORACLE(当前 4)决定;viz/oracle_viz.py:3 声称每图同时写 PNG/SVG/PDF,实际默认只写 PNG。
