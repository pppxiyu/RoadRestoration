# 02 — `run_pretrain_milp()` 的解题逻辑:traffic-fixation MILP

这份笔记解释预训练求解器是**如何**为每个修复时长 scenario 找出一张开工 schedule、并与 brute-force oracle 对比的。入口是 `util/pretrain_milp.py` 里的 `run_pretrain_milp()`,命令行 `python -m util.pretrain_milp` 直接触发它。笔记按代码真实的解题顺序组织:每一节是一个逻辑步骤,先说这一步要解决什么问题,再说它做了什么、为什么这样做,最后交代产物被后面哪一步消费,并以 path:line 锚点收尾,供对照当前源码查证。第三方库(AequilibraE、networkx、scipy)只概括其角色,不深入内部。项目参数统一由根目录 `config.py` 提供(代码内引用名为 `P`),各默认值汇总在附录 A。

需要特别说明的是,本流水线的**问题定义与真实目标评估器与 oracle 完全共用**,由同一批函数实现。为避免与笔记 01 重复,凡是共享的部分本文只标明"见笔记 01 的哪一节",而把篇幅集中在 MILP 特有的逻辑上;纯工程设施(断点续跑、逐 scenario checkpoint、fingerprint 校验、运行计时)不影响目标值的定义,与笔记 01 一致地略去;与 oracle 的 post-hoc 对比以及结果解读上的注意点,集中在文末的 Caveats 节。
(util/pretrain_milp.py:260;命令行入口 util/pretrain_milp.py:425-431)

---

## 总览 · 为什么走"冻结交通 → 线性 surrogate → MILP → 刷新 UE"这条路

要解决的问题是:洪水打断了若干条道路 segment,每条要占用一个 crew 连续修若干个时间 slot;需要找一张"每条 segment 何时开工"的 schedule,使整段恢复期内路网可达性的退化(objective $F_1$)最小。难点在于耦合:$F_1$ 由每个 slot 的 user equilibrium(UE)通行时间构成,而通行时间取决于该 slot 哪些路还坏着,也就是取决于 schedule 本身。若把这个耦合原样纳入一个优化器,就意味着目标非线性、且每次评估内部都要反复解 UE,代价不可接受。

这条流水线采用交替优化(traffic fixation),一轮之内分四个阶段:第一,把整段 horizon 的 OD 通行时间**冻结**成常数(取上一轮 UE 的结果);第二,在冻结的通行时间下,把"segment $e$ 在 slot $k$ 开工对 $F_1$ 的边际改善"解析地算成系数 $c_e^k$,于是 $F_1$ 被线性 surrogate $\sum_e \sum_k c_e^k x_e^k$ 替代,这一步完全不碰 UE;第三,解一个很小的 start-time MILP,在排程约束下最大化 surrogate,得到新 schedule;第四,只为这张新 schedule 做一遍真实评估(每个 slot 一次 UE),刷新通行时间,回到第一个阶段。如此迭代,直到 schedule 不再变化或触发保护条件。

不加约束的原始交替会**振荡**:修好某条 segment 恰好疏通了当初让它显得值得修的那批路,于是 MILP 的选择在几张"互为最优应答"的 schedule 之间反复翻转(一种拥堵反馈的 cobweb),稳不下来。为此本流水线加了两个相互配合的**递减稳定器**——一个作用在连续的交通状态上(递减 damping),一个作用在离散的 schedule 上(递减 proximal 惩罚),两者合力把本会成环的迭代逼成一个 fixed point;机制细节见第 7 步。

之所以这样能省掉海量 UE 求解,是因为昂贵的 UE 只出现在"每轮评估一张 schedule"这一处,每轮恰好 $T$ 次,一个 scenario 总共 $(\text{迭代轮数}+1)\cdot T$ 次。相比之下,oracle 对同一 scenario 要枚举全部 $N!$ 张 work-conserving schedule、每张 $T$ 次 UE;而把 UE 埋进优化器内层则连调用次数都无法封顶。surrogate 与 MILP 部分是纯 numpy 加一次 HiGHS 求解,耗时可忽略。
(util/pretrain_milp.py:1-50)

---

## 第 1 步 · 共用问题定义、"非数据泄露"边界与按规模隔离的输出目录

要解决的问题是:MILP 的成绩要与 brute-force oracle 直接比较,两者必须面对**完全相同**的问题;但又必须保证 oracle 已算出的"答案"绝不能反过来帮 MILP。

做了什么:run_pretrain_milp 用与 run_oracle 完全相同的四个构件来定义问题——同一个受损实例(select_oracle_instance)、同一批修复时长 scenario(sample_scenarios,同一个 seed)、同一个全局 horizon(compute_horizon)、同一个真实目标评估器(evaluate_schedule)。而 oracle 的**解**(oracle_optima.csv、oracle_landscape.csv)只在流程最末尾的对比阶段才被读取,那时每个 scenario 的 MILP schedule 早已算完。结果写到 outputs/pretrain_milp/n{N}/ 下,N 取 config.py 的 N_DISRUPTED_ORACLE,沿用 oracle 端同一套按规模命名的目录规则(scale_dir);正因两端共用这条 n{N} 规则,末尾对比时把 base 目录换成 outputs/oracle 即可拼出同规模 oracle 结果的路径,既不需要额外配置,也永远不会覆盖另一个规模的结果。

之所以这样做:共享问题定义是刻意的——实例、scenario、horizon、目标函数任何一处不同,$F_{\text{MILP}}$ 与 $F_{\text{oracle}}$ 就失去可比性。而判断是否数据泄露的依据是"上游算法是否消费了下游的答案":MILP 的目标系数 $c_e^k$ 只依赖它自己迭代产生的冻结 UE 通行时间,从不查阅 oracle 的任何解;即使 oracle 从未运行,MILP 的结果也完全一致,只是最后的对比图被推迟(见 Caveats 的 C2)。
(util/pretrain_milp.py:267-271, util/pretrain_milp.py:331-342, util/pretrain_milp.py:264, util/oracle.py:77-82)

---

## 第 2 步 · 共享的问题机制与真实评估器(见笔记 01),外加两处 MILP 专属钩子

要解决的问题是:在 MILP 特有的逻辑开始之前,受损实例、静态 context、scenario 采样、horizon、真实目标评估器都必须先就位——而它们正是 oracle 所用的那一套,由完全相同的函数算出。

做了什么:这一步逐字复用笔记 01 已经追踪过的机制,不再重述:

- 选定受损实例(select_oracle_instance,按 baseline UE flow 排重要性、把最关键的两条边赋 severity 3 以制造真断连):见**笔记 01 的第 1 步(§1a)**;
- 一次性构建静态 context——路网表、OD 对、灾前基准通行时间 $u_r^{t_0}$、断连惩罚 $u_{\text{pen}}$、以及把"哪些路坏着"映射成"哪些 OD 需求被压制"的敏感度矩阵 $B$:见**笔记 01 的第 1 步(§1b–1f)**;
- 采样 $M$ 份修复时长 scenario(按 level 抽、固定 seed):见**笔记 01 的第 2 步**;
- 统一全局 horizon $T$:见**笔记 01 的第 4 步**;
- 真实目标评估器 evaluate_schedule(逐 slot 一次 UE,精确算出 $F=\mu F_1+(1-\mu)F_2$):见**笔记 01 的第 5 步(5a–5f)**。

因为这里的 seed 与 oracle 端相同,两边拿到的 scenario 逐份一致,这是 $F$ 可比的前提之一。

在这些共享构件之上,本流水线额外用到两处 oracle 用不到的钩子,必须点明,否则后面的推导会断链:

其一,severity 向量的第二重身份。笔记 01 §1f 已指出 oracle 自始至终不读这个向量;而在本流水线里它是 $c_e^k$ 系数的一部分——作为"某条 segment 完全修复后放回网络的需求规模" $v_e^{\ast}$,与 $B$ 的列一一对齐(用在第 4 步)。

其二,冻结通行时间矩阵。evaluate_schedule 在带 return_u 选项调用时,会额外把逐 slot 的通行时间堆成一个 $(T \times |R|)$ 矩阵返回;这正是交替优化要"冻结"的对象,直接喂给第 3、4 步。

之所以指向笔记 01 而不在此重写,是因为这些函数及其建模理由本就属于 oracle,在这里重复既与笔记 01 冗余、又可能随时间漂移不一致;本文只保留上述两处钩子,因为它们正是 MILP 特有步骤所消费的。
(util/pretrain_milp.py:267-271 调用共享构件;util/pretrain_milp.py:92 与 util/pretrain_milp.py:101 severity 钩子;util/evaluate.py:248-251 冻结通行时间矩阵)

---

## 第 3 步 · greedy 初始 schedule 与第一份冻结通行时间

要解决的问题是:第一轮迭代还没有任何 UE 结果可供冻结,必须先有一张初始 schedule 和它对应的整段通行时间。

做了什么:alternating_optimize 把受损 segment 按 edge id 原序交给 work-conserving 列表排程,得到一张 greedy 初始 schedule;随即用真实评估器(带 return_u)评估它,拿到初始 $F$ 和第一份 $(T \times |R|)$ 冻结通行时间;schedule 与结果一并存入历史列表,并在"出现次数计数器"里给这张 schedule 记一次(供第 6 步的 cycle guard 使用)。

之所以这样做:初始点只需合理、不必聪明——最终输出终究是事后按真实 $F$ 从全部历史里挑出的(第 8 步),而初始 schedule 也在历史里,所以输出保证不劣于这个 greedy 起点。初始评估既为"最优"提供了下界,又生产了第一份可冻结的通行时间。此外,proximal 惩罚的基准强度 $\gamma_0$ 也在这一步定死——用这第一份冻结通行时间算出的系数谱一次算好、全程不再更新(见第 7 步)。
(util/pretrain_milp.py:215-223)

---

## 第 4 步 · 解析敏感度系数 c_e^k:不跑 UE 的线性 surrogate

要解决的问题是:MILP 需要线性目标,但真实 $F_1$ 经由 UE 依赖于 schedule,是非线性的。在通行时间被冻结的前提下,要解析地回答"若 segment $e$ 在 slot $k$ 开工,它对 $F_1$ 各步项的边际改善是多少",并且这个回答一次 UE 都不许跑。

做了什么:precompute_c 先由冻结通行时间与基准,算出每个 slot、每个 OD 的**拥堵超额**

$$\alpha_r^{k'} = 1 - \frac{u_r^{t_0}}{\tilde{u}_r^{k'}}$$

($\tilde{u}$ 取冻结矩阵的第 $k'$ 行),再对每条 segment $e$(时长 $d_e$)和每个可行开工 slot $k$(须满足 $k + d_e \le T$)按闭式累加

$$c_e^k = \frac{1}{T} \sum_{k'=k+d_e}^{T} \big(1 - \rho^{\,k'-k-d_e+1}\big) \sum_r B_{r,e}\, v_e^{\ast}\, \alpha_r^{k'}$$

其中 $v_e^{\ast}$ 是该 segment 的 severity(取自 context 的 severity 向量),$B_{r,e}\, v_e^{\ast}$ 即"$e$ 完全修复时放回网上的 OD $r$ 需求"。不可行开工($k > T - d_e$)的系数保持为 0,由 MILP 的变量上界另行禁止。全程纯 numpy,UE 调用次数为零。

之所以这样做:直觉是"修好 $e$ 的收益 = 它放回的需求 × 这些需求所处 OD 的拥堵超额"。收益从完工 slot $k + d_e$ 才开始产生;放回的需求并非一步到位,而是按恢复因子 $(1 - \rho^{\,n+1})$ 随完工后第 $n$ 个 slot 逐步**增长**——与笔记 01 第 5 步(§5c)真实需求模型中缺口按 $\rho$ 消退的速率一致。这是对需求 shortfall 模型的一处修正:旧版让收益按 $\rho$ 的幂**衰减**是错误方向,以当前源码为准。对 $k'$ 求和把这份逐 slot 收益分配到 $F_1$ 的各步项上,再除以 $T$ 与 $F_1$ 的整段平均对齐。$\alpha$ 可以为负——恢复期需求被压低时,网络可能比灾前基准还快——这是合法信号,表示此时修这条路的边际收益为负。severity 在系数中的角色,是把"放回需求的规模"与损坏等级挂钩:severity 越高,修复它放回的需求越多,同等拥堵超额下收益越大。
(util/pretrain_milp.py:77-108)

---

## 第 5 步 · start-time MILP:在更大的可行域里最大化 surrogate

要解决的问题是:在冻结通行时间的世界里,找出使 surrogate 最大的开工 schedule,并遵守排程规则——每条 segment 恰好开工一次、任一时刻同时在修的数量不超过 crew 数、所有工作都在 horizon 内完工。

做了什么:决策变量 $x_e^k \in \{0, 1\}$ 表示"segment $e$ 是否在 slot $k$ 开工"(源码实现中记作 $y$),共 $|E|\cdot T$ 个。目标为

$$\max \sum_e \sum_k c_e^k\, x_e^k$$

(实现上因 HiGHS 只做最小化,对系数取负后最小化)。在交替循环里,build_and_solve_milp 还额外接收上一轮 schedule $y_{\text{prev}}$ 与惩罚强度 $\gamma$,在这个线性目标上再叠一项 proximal 惩罚,把 MILP 往上一轮解拉——它同样是线性的,细节与作用见第 7 步;本步之下的纯 surrogate 讨论、以及末尾的 level_a 编码检查都取 $\gamma = 0$,不受这一项影响。约束分三类:其一,每条 segment 恰好开工一次,$\sum_{k=1}^{T} x_e^k = 1$;其二,crew 上限,对每个 slot $k$,$\sum_e \sum_{k'=\max(1,\,k-d_e+1)}^{k} x_e^{k'} \le C_{\max}$,内层求和枚举了"在 $k$ 时刻仍在修"的所有可能开工时刻;其三,horizon 越界禁止,对 $k > T - d_e$ 的变量直接把上界压为 0,这是唯一真正的可行性守卫,与 $c_e^k$ 在这些位置为 0 相呼应。问题交给 scipy.optimize.milp(背后是 HiGHS branch-and-bound,随 scipy 附带、无需 license)求解,失败即抛错;解出后每条 segment 取其行上取值为 1 的 slot,还原成"edge → 开工 slot"的映射。模块里另有一个小工具反向计算任意给定 schedule 的 surrogate 值 $\sum_e c_e^{s_e}$,供逐轮 trace 记录与编码检查使用。

之所以这样做:模型极小($N=4$、$T$ 为几十时不过几百个 binary 变量),HiGHS 的求解开销可忽略,交替循环的成本几乎全部留给 UE。特别要留意可行域:这里允许 crew 故意空等(并不要求 work-conserving),因此 MILP 的可行域是 oracle 所枚举的 work-conserving schedule 集合的**严格超集**——这正是 Caveats C2 中 gap 可以为负的根源。
(util/pretrain_milp.py:114-164, util/pretrain_milp.py:167-171)

---

## 第 6 步 · 三个停止条件:fixed point、计数式 cycle guard、迭代上限

要解决的问题是:surrogate 是近似,迭代既不保证单调、也不保证收敛,还可能落入循环;需要明确的停机规则,既能识别真收敛,也能约束不收敛的情形。

做了什么:主循环每轮依次算系数、解 MILP、真实评估、记 trace,然后按三级规则判停,并把停机方式记进一个 `outcome` 字段(取值 fixed_point / cycle / iter_cap)。其一,fixed point——新 schedule 与上一轮完全相同,置 `converged=True`、`outcome="fixed_point"` 后退出:MILP 在"由这张 schedule 自己生成的通行时间"下仍选择它自己,这是交替优化意义上的不动点。其二,计数式 cycle guard——记录每张出现过的 schedule 的出现次数,某张累计出现 MILP_CYCLE_TOL(3)次即退出、记 `outcome="cycle"`;设成 3 而不是"一见重复就停",是因为两个递减稳定器(第 7 步)每轮仍在改变冻结通行时间与 proximal 拉力,旧 schedule 重现时系数已不同,再给它一两次机会可能跳出小的 2-cycle 或极限环。其三,硬上限——循环跑满 MILP_MAX_ITER(20)轮仍未停,则保留默认的 `outcome="iter_cap"`。三种退出里只有 fixed_point 置 `converged=True`,cycle 与 iter_cap 都算未收敛。

之所以这样做:三级规则从"确定性收敛"到"疑似循环"再到"预算耗尽"逐级提供退路;配合第 8 步的 best-by-true-$F$,任何一种退出方式都不损害输出质量,区别只在于多跑几轮 UE。把 `outcome` 单独记下来,是为了让过程图(第 10 步 panel b)与事后分析能区分"真收敛"与"被 guard 拦下",而不是只看一个二值的 `converged`。
(util/pretrain_milp.py:225-242, config.py:61-62)

---

## 第 7 步 · 收敛机制:递减 damping 与递减 proximal 惩罚,把振荡逼成 fixed point

要解决的问题是:原始交替(每轮全盘采纳新 UE 通行时间、MILP 每轮完全自由地重挑 schedule)不收敛。修好某条 segment 恰好疏通了当初让它显得值得修的那批路,冻结通行时间随之翻转,$c_e^k$ 大幅跳动,MILP 解在几张"互为最优应答"的 schedule 之间来回摆(一种拥堵反馈的 cobweb)。这里有两个耦合的振荡源:连续的交通状态、离散的 schedule;要真收敛,得同时把两者摁住。仅靠一个**固定权重**的 damping 不够——它只把振荡幅度按比例缩小,留下一个不消失的极限环。

做了什么:代码用两个**递减**稳定器,一个管交通状态、一个管 schedule。

其一,递减的通行时间 relaxation(config 里 `MILP_DAMPING_MODE = "decay"`)。每轮末尾不整体替换冻结通行时间,而是做凸组合更新

$$\tilde{u} \leftarrow \lambda_n\, \tilde{u}_{\text{new}} + (1 - \lambda_n)\, \tilde{u}_{\text{old}}, \qquad \lambda_n = \frac{1}{n+1}$$

其中 $n$ 是 1-based 迭代序号,故第 1 轮 $\lambda_1 = 1/2$、第 2 轮 $1/3$……步长逐轮趋零。与固定步长(旧的 `"const"` 模式,权重为常数 `MILP_DAMPING`)只按比例缩小振幅不同,趋零的步长能把 $\tilde{u}$ 的振荡真正压到零,让连续的交通状态稳定下来。`"const"` 模式仍保留供实验,`alternating_optimize` 的 `damping` 入参只在该模式下才覆盖 `MILP_DAMPING`;主流程用 `"decay"`。

其二,递减的 proximal schedule 惩罚。在第 5 步的 MILP 目标上追加一项

$$\gamma_n \, \lVert y - y_{\text{prev}} \rVert_1, \qquad \gamma_n = \frac{\gamma_0}{n+1}, \quad \gamma_0 = \texttt{MILP\_PROX\_SCALE} \cdot \operatorname{median}_e \big(\text{c 在 } e \text{ 可行 slot 上的跨 slot 极差}\big)$$

其中 $y_{\text{prev}}$ 是上一轮 schedule。$\gamma_0$ 把惩罚放到与系数 $c$ 同一量纲上,由第一份冻结通行时间的系数谱一次算定、全程不变(第 3 步已埋);$\gamma_n$ 则与 $\lambda_n$ 同样按 $1/(n+1)$ 递减。关键是这一项**是线性的**:在"每条 segment 恰好开工一次"的约束下,$\lVert y - y_{\text{prev}} \rVert_1$ 恰等于 $2\times$(改了开工 slot 的 segment 数),于是无需引入任何新变量——实现上只把每条 segment 的上一轮开工 slot 处的目标系数加上 $-2\gamma_n$ 的奖励(HiGHS 做最小化,奖励即负系数),鼓励"不必要就别翻"。它摁住的是离散的 schedule。

之所以这样做:两个稳定器分工——递减 damping 让连续的交通状态收敛,递减 proximal 惩罚让离散的 schedule 收敛;合起来把本会成环的迭代逼成一个 fixed point。两者都**不移动 fixed point**:若某份通行时间已是不动点(新 UE 结果与它相同),凸组合仍等于它自身;而在真正的不动点上 $y = y_{\text{prev}}$,proximal 项取值为 0——稳态方程不因 $\lambda_n$、$\gamma_n$ 而变,它们只决定"以什么路径、多快走向那里",且因逐轮趋零,越接近收敛越不干扰目标。强度 `MILP_PROX_SCALE = 0.1` 是能让所有测试 scenario 都收敛的最小值:再大反而过早冻结 schedule、损失精度。

在代码中,proximal 项在每轮解 MILP 时随 $\gamma_n$ 传入(第 5 步),damping 更新则位于每轮末尾、第 6 步三个停止判定之后、仅当循环继续时才执行;混合后的冻结通行时间作为下一轮第 4 步 $c_e^k$ 计算的输入——本文的步骤顺序与代码的执行顺序一致。
(util/pretrain_milp.py:243-247 递减 damping;util/pretrain_milp.py:221-223 与 util/pretrain_milp.py:230-231 递减 proximal 强度;util/pretrain_milp.py:131-135 proximal 线性编码;config.py:64-79)

---

## 第 8 步 · best-by-true-F:收敛与否都无害的选择规则

要解决的问题是:即便迭代停了,最后一轮的 schedule 未必是过程中最好的——surrogate 驱动的迭代不单调,可能后期反而变差;必须决定"最终输出哪一张"。

做了什么:每一轮(含第 0 轮的 greedy 初始)的 schedule 与真实 $F$ 都留在历史里;退出后对全部历史按**真实 $F$** 取最小者,作为该 scenario 的最终输出,并在 trace 里给那一轮打上 is_best 标记。

之所以这样做:每个 iterate 的真实 $F$ 本来就为刷新通行时间而算过,挑最优不花任何额外代价;这让整个 heuristic 的输出有下界保证——至少不劣于 greedy 初始——而"收敛与否"从正确性问题降级为效率问题。这也是第 6 步三个停止条件得以宽松设计的原因。该规则在小规模问题上的一个解读注意点,见 Caveats 的 C1。
(util/pretrain_milp.py:250-253)

---

## 第 9 步 · 逐 scenario 求解与结果聚合

要解决的问题是:$M$ 份 scenario 各自独立求解,全部跑完后需要把整个 run 的结果落盘并给出汇总。

做了什么:对每个 scenario,run_pretrain_milp 调一次交替优化(第 3–8 步),把最优结果整理成一行——scenario 编号、$F_{\text{MILP}}$、$F_1$、$F_2$、迭代轮数、是否收敛(converged)、停机方式(outcome ∈ {fixed_point, cycle, iter_cap})、耗时与该 scenario 消耗的 UE 次数、时长组合、每条 segment 的开工 slot——并把该 scenario 的逐轮 trace 追加进 trace 行集。全部 scenario 完成后,两张表分别写成 milp_optima.csv(逐 scenario 的最终 schedule 与目标值,含 converged 与 outcome 列)与 milp_trace.csv(全部逐轮 trace)。

之所以这样做:milp_optima.csv 是本流水线逐 scenario 的成品,对应 oracle 端的 oracle_optima.csv,也是第 10 步过程图与 Caveats C2 对比的主输入;逐轮 trace 全量落盘,是为了让过程图与事后分析不必重跑任何 UE。
(util/pretrain_milp.py:292-317)

---

## 第 10 步 · 过程诊断图:迭代轨迹与耗时

要解决的问题是:交替优化是 heuristic,必须能直观检查它的行为——每个 scenario 迭代了几轮、真实目标怎样下降、surrogate 是否稳定、时间花在哪里。

做了什么:make_process_figures 从逐轮 trace 生成两张诊断图。03_optimization_process.png 含三个 panel:panel a 画每个 scenario 真实 $F$ 的 best-so-far 曲线,并以圆点标出实际返回的最优 iterate;panel b 画**逐轮的原始 $F$**(非 best-so-far),按该 scenario 的 `converged` 标志用实线(达到 fixed point)与虚线(未收敛,即以 cycle guard 或迭代上限退出)区分,直接展示 $F$ 是否稳定下来;panel c 画 MILP surrogate $\sum_e c_e^k x_e^k$ 随迭代的轨迹。04_runtime.png 分别画每 scenario 的 wall-clock 与迭代轮数。两张图直接使用内存中的 trace 与 optima 数据,不回灌任何计算。

之所以这样做:panel a 的 best-so-far 对应第 8 步真正返回的那一张,但它是累计最小值、在构造上只降不升,会掩盖迭代本身是否收敛;因此单设 panel b 画原始 $F$ 来直接回答"$F$ 是否收敛"——达到 fixed point 的 scenario 末端持平,未收敛退出的则持续起伏;panel c 的 surrogate 则用于判断冻结-刷新回路是否在振荡。
(util/pretrain_milp.py:328-329, viz/pretrain_viz.py:139-227)

---

# Caveats

## C1 · best-by-true-F 是安全网,但在很小的规模下近似一次小型枚举

第 8 步的 best-by-true-F 规则把"是否收敛"从正确性问题降级为效率问题:无论迭代落到 fixed point、被 cycle guard 拦下、还是跑满上限,最终输出都是全部历史 iterate 里真实 $F$ 最小的那张,因而始终有下界保证。第 7 步的两个递减稳定器是为让迭代真正收敛到 fixed point 而加的;但即便迭代收敛,best-by-true-F 本身仍是一层与收敛与否无关的安全网,这带来一个解读上的注意点。

在**很小**的实例上,好的最终 $F$ 可能主要来自"迭代历史顺带走过了足够多的 schedule,再从中挑最优"这一效应,而不能单独归功于 surrogate 引导。以当前实例为例,work-conserving 的排列总共只有 $4! = 24$ 张,而一个 scenario 的迭代历史最多可含 21 个 iterate(greedy 初始加上限 20 轮,且可能有重复);两者规模相当,此时 best-by-true-F 在效果上已接近一次小型暴力搜索,单凭最终 $F$ 的好坏无法干净地区分"surrogate 引导得好"与"历史恰好覆盖到了最优"。因此在本实例上,对比结果(见 C2)只宜读作"MILP 的成品不劣于 oracle 的 work-conserving 最优",而不宜据以单独论证 surrogate 引导本身有多强。问题规模增大后,$N!$ 迅速增长而迭代轮数上限不变,这层枚举效应随之消失,届时 surrogate 引导的贡献与收敛行为(outcome 的分布)才需要、也才能被单独检验。
(util/pretrain_milp.py:250-253 选择规则;逐 scenario 的 converged、outcome、n_iter 列见 outputs/pretrain_milp/n4/milp_optima.csv)

## C2 · post-hoc oracle 对比:gap 的定义与它为什么可以为负

对比要回答的是"traffic-fixation MILP 离 ground truth 有多远"。oracle 已对每个 scenario 枚举了全部 work-conserving schedule,留下逐 scenario 最优 $F$ 与完整 landscape;按第 1 步的"非数据泄露"边界,直到全部 MILP 求解结束后,流程才允许读取它们。

做法是按第 1 步的同一 n{N} 规则拼出 outputs/oracle/n{N}/ 路径;仅当该规模的 oracle_optima.csv 存在时,把 oracle 的逐 scenario 最优 $F$ 按 scenario 号并进 MILP optima 表,逐 scenario 定义

$$\text{gap} = F_{\text{MILP}} - F_{\text{oracle}}$$

(绝对差;$\text{gap} \le 0$ 表示 MILP 追平或胜过了 oracle 最优的 work-conserving schedule),写出 milp_vs_oracle.csv,并生成两张图:01_milp_vs_oracle.png 逐 scenario 并排画两个 $F$ 及 gap 柱状;02_cross_scenario_landscape.png 把每张固定 schedule 按跨 scenario 平均 $F$ 排名画成曲线加范围带,叠上 MILP(逐 scenario 自适应)的平均 $F$ 与 hindsight(逐 scenario 最优)的平均 $F$,回答"若必须承诺单一固定策略,各策略站在哪里,而自适应 MILP 又落在哪里"。若该规模的 oracle 结果尚未算好,则打印提示并跳过对比;等 oracle 完成后运行 `python -m util.pretrain_milp --landscape`,只从已存 CSV 重出对比图,不重跑任何 UE。

为什么 gap 可以为负:oracle 的"最优"仅是最优的 **work-conserving** schedule(按排列的列表排程、不留空档);而第 5 步的 MILP 在"开工一次、crew 上限、horizon 内完工"约束下允许 crew 故意空等,可行域是前者的严格超集,完全可能找到集合之外更好的 schedule。这不是 bug,而是搜索空间更大的自然结果。针对"会不会是编码错误"的疑虑,模块另备 level_a 检查(命令行 --level-a 手动运行,不在主流程内):在同一份固定系数下、且**关闭 proximal 惩罚**($\gamma = 0$,回到纯 surrogate 目标),MILP 的 surrogate 最优值必须不小于全部 work-conserving schedule 的最好 surrogate 值(超集关系的直接推论),并逐 slot 复核 crew 上限与 horizon 约束。
(util/pretrain_milp.py:331-349, util/pretrain_milp.py:359-400, util/pretrain_milp.py:403-422, viz/pretrain_viz.py:35-137)

---

# 附录 A · 参数默认值(出自 config.py)

以下均为**当前** config.py 的默认值(唯一真实来源)。除注明"给定"外,每一项都是建模假设。

| 参数 | 当前默认 | 在本流水线里的作用 | config.py 行 |
|---|---|---|---|
| DELTA_T_H | 3.0 h/slot(给定) | slot 的物理时长;在 $F_2$ 里约掉 | 10 |
| C_MAX | 2 | 列表排程与 MILP crew 上限约束共用的 crew 数 | 11 |
| MU | 1.0 | $F = \mu F_1 + (1-\mu) F_2$ 的权重;取 1.0 时目标只看 $F_1$($F_2$ 仍计算并记录,权重为 0) | 12 |
| CAP_RETAIN | {1: 0.3, 2: 0.1, 3: 0.02} | 受损 link 的 capacity 保留系数 | 20 |
| SPEED_RETAIN | {1: 0.5, 2: 0.3, 3: 0.2} | 受损 link 的 free-flow time 除数(越小越慢) | 21 |
| SEVER_SEVERITY | 3 | severity 达到此值即整条移除(真断连 → $u_{\text{pen}}$) | 22 |
| F1_ACTIVE_ONLY | False | $F_1$ 对整段 $[1, T]$ 平均,与 $c_e^k$ 的 $1/T$ 归一化对齐 | 25 |
| RHO | 0.7 | recovery inertia;真实需求模型与 $c_e^k$ 的 $(1-\rho^{\,n+1})$ 共用 | 37 |
| KAPPA | 1.0 | $B$ 里"损坏 → 需求缺口"的尺度 | 38 |
| UPEN_FACTOR | 10.0 | $u_{\text{pen}} = 10 \times$ 最大有限 baseline OD 通行时间 | 41 |
| M_SCENARIOS | 10 | duration scenario 份数 | 45 |
| SEED | 42 | scenario 抽样的 RNG seed(与 oracle 端一致) | 46 |
| N_DISRUPTED_ORACLE | 4 | 实例规模;决定 n{N} 输出目录 | 47 |
| UE_RGAP | 1e-6 | 单次 UE 的 relative-gap 目标 | 54 |
| UE_MAX_ITER | 100 | 单次 UE 的最大 Frank-Wolfe 迭代数 | 55 |
| MILP_MAX_ITER | 20 | 交替循环硬上限(停止条件三,outcome=iter_cap) | 61 |
| MILP_CYCLE_TOL | 3 | 计数式 cycle guard 的重复阈值(停止条件二,outcome=cycle) | 62 |
| MILP_DAMPING_MODE | "decay" | 通行时间 relaxation 模式:"decay" 用递减步长 $\lambda_n = 1/(n+1)$(主流程),"const" 用固定权重 $\lambda$ | 64 |
| MILP_DAMPING | 0.5 | 固定权重 $\lambda$,**仅在 MILP_DAMPING_MODE == "const" 下生效** | 70 |
| MILP_PROX_SCALE | 0.1 | 递减 proximal 惩罚的强度基准:$\gamma_0 = \text{此值} \times (c \text{ 系数跨 slot 极差的中位数})$,$\gamma_n = \gamma_0/(n+1)$;取 0 关闭该惩罚 | 72 |
| DURATION_SUPPORT | 按 (road_class, severity) 的支撑集 | scenario 抽样的 base 时长来源 | 84-88 |
| ETA | [0.8, 1.0, 1.2](给定) | crew-efficiency 乘子 | 89 |
