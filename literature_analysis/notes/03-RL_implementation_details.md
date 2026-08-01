# 03 — RL Implementation Details

本笔记对 RL 筛选集逐篇提取 RL 实现细节,共 7 篇(idx 35, 50, 60, 98, 133, 1059, 2704)。idx 710(Ulusan & Ergun 2021)属于模型已知的 ADP 而非 model-free RL,已按决定移出本分析。全部结论基于全文,并经过两遍读取,先提取、再独立复核。

分析框架为每篇回答同样的八问:① 是否遵循 MDP/Bellman/Q/π 模板,以及 state、action、reward、agent 要素是否齐备(结论见总览表);② policy function 如何设置;③ 决策变量在执行层面具体决定什么,是时间还是路段、桥梁或车站的编号,变量符号与含义是什么;④ action 如何影响环境 state,交通状态用什么方法更新;⑤ reward 如何计算,精确到计算层面;⑥ Q function 依据什么更新、具体怎么做;⑦ Q function 的表示形式是表格还是深度神经网络,若是后者属于什么类型;⑧ agent 在建模里指的是什么。

## 总览

| idx | 算法 | MDP | Bellman | Q func | Q 形式 | π func | 备注 |
|---|---|---|---|---|---|---|---|
| 35 | 表格 Q-learning(小网络)+ "DDQN"(大网络,实为 vanilla DQN) | ✓ | ✓ | ✓ | 表格 + MLP(大算例) | ✓ greedy-over-Q | 双层规划上层,下层 UE(Frank-Wolfe) |
| 50 | 表格 Q-learning × 2 agents(电网+路网)+ 嵌入优化 | ✓ | ✓ | ✓ | 表格 | ✓ greedy-over-Q | RL-OP 混合,协调器耦合 |
| 60 | DQN + GCN 前端(GNN-DRL,Top-k 动作过滤) | ✓ | ✓ | ✓ | GCN + MLP | ✓ greedy-over-Q | 联合动作空间 $`2^N`$,靠启发式 Top-k 压缩 |
| 98 | 表格 Q-learning | ✓ | ✓ | ✓ | 表格 | ✓ greedy-over-Q | 环境为双层网络 + Motter-Lai 级联失效 |
| 133 | 表格 Q-learning(明确不用深度) | ✓ | ✓ | ✓ | 表格 | ✓ greedy-over-Q | 环境为 day-to-day 流量 LP(GUROBI) |
| 1059 | Dueling Double DQN + DGCN 编码 + NoisyNet + PER/HER | ✓ | ✓ | ✓ | 动态 GCN + dueling NoisyNet MLP | ✓ greedy-over-Q | 名为 constrained RL,实为 action masking |
| 2704 | 多 agent DQN + 学习型通信(Foerster et al.) | ✓ | ✓ | ✓ | NN(结构未报告) | ✓ greedy-over-Q | 独立学习者,非平稳环境 |

共性(π 按"存在 policy 决策方式"理解,不要求显式函数):七篇都有 π,而且形态完全一致,都是对 Q 的 (ε-)greedy/argmax 决策规则;没有任何一篇使用独立参数化的 policy 网络,也就是说没有 policy-gradient 或 actor-critic 方法。

---

## idx 35 — Zhao et al. 2026, *Prioritizing post-earthquake recovery of transportation networks under uncertainty*

*作者:Zhao Taiyi, Tang Yuchun, Li Qiming, Wang Jingquan;期刊:Computers & Industrial Engineering;年份:2026。*

**Agent**:单一 agent,建模上指**维修队**(maintenance team),即那支逐座修桥的队伍;它同时承担规划者的角色,因为修复顺序就是它的行动序列。环境是受损的交通网络。

**Policy function**:π 是对 Q 的 greedy/ε-greedy 决策规则,没有独立参数化的 policy 网络。训练期用 ε-greedy 探索;部署时从灾后损伤状态出发,每步对当前 Q 取 argmax,直到全部 OD 恢复可达为止;已修复的桥通过把其 Q 值置 0 而被排除在选择之外。

**决策变量**:执行层面决定的是**桥的编号**。每步动作 $`a_t \in A`$ 表示下一座要修复的损毁桥($`A`$ 为损毁桥集合),动作序列逐步给出修复顺序。修复的**时间不是直接决策**,因为底层双层模型中的开始时刻 $`x_b`$(桥 $`b`$ 的修复开始时刻,$`x_b = 0`$ 表示不修)是由动作顺序加上随机工期 $`d_b`$ 递推诱导出来的,并受同时在修桥数上限 $`N_{\max}`$ 约束。

**Action → State**:执行"修复桥 $`b`$"这一动作之后,$`b`$ 进入已修集合;修复期内其损伤等级保持不变,经过工期 $`d_b`$ 后损伤归零。链路的容量与速度按 $`C = C_0\,(1 - v/4)`$ 和 $`S = S_0\,(1 - v/4)`$ 恢复,其中 $`v`$ 为损伤等级(取值 0 到 4),$`C_0`$ 与 $`S_0`$ 为灾前容量与速度。交通状态的更新方法是,每步动作后**重解一次 UE 均衡分配**(Frank-Wolfe 算法,10 次迭代、容差 0.005;链路行程时间由 BPR 函数随流量重算,参数 $`\alpha=0.15,\ \beta=4`$),得到新的均衡链路流量与 OD 行程时间,可达集 $`N_t`$ 随之刷新;可达的判据是 OD 均衡行程时间满足 $`u_w \le 1.5\,u_w^{0}`$(上标 0 表示灾前)。若选择了维修队无法抵达的桥,则修复时间记为 $`+\infty`$,reward 记 0。

**Reward**(权重 $`\mu = 0.5`$):

```math
R = \mu\,(R_1 + R_1') + (1-\mu)\,(R_2 + R_2')
```

计算层面的记号如下:$`N_t`$ 为当前可达 OD 集,$`B_t`$ 为已修桥集,$`u_w`$ 为 OD 对 $`w`$ 在重解 UE 后的均衡 travel time,$`d_b`$ 为桥 $`b`$ 的修复工期,$`I_w = q_w/d_w`$ 为 OD 重要度,即需求量 $`q_w`$ 除以灾前最短路距离 $`d_w`$;$`u_w`$、$`d_b`$、$`I_w`$ 均经 min-max 归一化。四个分项为:

```math
R_1 = \sum_{w \in N_{t+1}\setminus N_t} \frac{I_w}{u_w^{t+1}}, \qquad R_2 = \sum_{w \in N_{t+1}\setminus N_t} \frac{I_w}{\sum_{b \in B_{t+1}} d_b}
```

```math
R_1' = \sum_{w \in N_t} \Big(\frac{I_w}{u_w^{t+1}} - \frac{I_w}{u_w^{t}}\Big), \qquad R_2' = \sum_{w \in N_t} I_w\Big(\frac{1}{\sum_{b\in B_{t+1}} d_b} - \frac{1}{\sum_{b\in B_t} d_b}\Big)
```

含义是,本步新打通的 OD 按"重要度除以均衡行程时间"计入服务收益 $`R_1`$,按"重要度除以全部已投入工期"计入效率收益 $`R_2`$;原本已可达的 OD 则计入这两个比值的增量,即 $`R_1'`$ 与 $`R_2'`$。若本步没有任何变化,reward 为 0;当长工期只换来小收益时,reward 可为负。

**Q 形式**:双轨。小算例用 **Q 表**,行是 state,列是损毁桥;大算例用**深度神经网络**,是一个 8 隐层全连接 MLP(各层宽度依次为 256, 256, 128, 128, 128, 128, 64, 64,激活函数 ReLU),输入是 40 维桥损伤态向量(离散值 0 到 4),输出是 40 维向量,每座损毁桥对应一个 Q 值,已修桥的 Q 值置 0。网络没有图结构,也没有 dueling 或 noisy 等变体。

**Q 更新**:表格版按标准 Q-learning 更新($`\alpha=0.4,\ \gamma=0.9`$):

```math
Q(S_t,a_t) \leftarrow (1-\alpha)\,Q(S_t,a_t) + \alpha\big(R + \gamma \max_{a'} Q(S_{t+1},a')\big)
```

深度版的 target 按原文 Eq.33 写作($`S'`$ 为下一 state;该式的动作选择与评估都取自 target 网络,因此实为 vanilla DQN 而非其命名的 double DQN,并且漏印了 $`\gamma`$):

```math
Y = R + Q_{\mathrm{target}}\big(S',\ \arg\max_{a'} Q_{\mathrm{target}}(S',a')\big)
```

损失为 MSE;replay 为均匀采样(容量 800,minibatch 30);每 $`L=50`$ 步把在线网络复制到 target 网络;网络先用"只剩一座桥待修"的平凡状态预训练,作为热启动权重。

---

## idx 50 — Saha et al. 2024, *Coordinated restoration of interdependent critical infrastructures*(RL-OP)

*作者:Saha Namrata, Rezapour Shabnam, Sahin Nazli Ceren, Amini M. Hadi;期刊:Sustainable Cities and Society;年份:2024。*

**Agent**:两个 agent,**每个基础设施(CI)一个**:电网 agent 与路网 agent,各自建模为该基础设施的修复调度者,分别在自己的优化环境上学习;两者通过协调器(Feasibility Module 与 Prediction Module)交换有限信息。

**Policy function**:π 是对 Q 表的 greedy/ε-greedy 决策规则,且选择顺序是**先筛可行集、再在可行集内取 (ε-)greedy/argmax**,相当于 action masking,而不是先全局 argmax 再用约束筛掉。可行集由两道筛选给出:一个候选子集要满足嵌入优化模型 (1-4) 在当前 crew 数下有有限解,耦合版还要求子集内各链路的跨 CI 前置链路已经修复(由协调器的 Feasibility Module 检查,得到缩减后的可行动作集);不可行的子集从一开始就不进入比较。训练期在可行集内用 ε-greedy 探索,收敛后从 Q 表贪心读出最终策略 π*。

**决策变量**:执行层面有两个决策。第一个是 $`a_{s_k} = L'_k`$,即第 $`k`$ 个 stage 本 CI 并行修复的**损毁链路编号子集**(集合值动作);第二个是 $`w_l`$,即每条选中链路投入的**整数 crew 数**,它不是 RL 学出来的,而是由嵌入的 min-max 整数规划决定:

```math
\min\ \max_{l \in L'_k}\ \sigma_l / w_l \qquad \mathrm{s.t.}\quad w_l \le C_l,\quad \sum_l w_l \le \Lambda_k
```

其中 $`\sigma_l`$ 为链路 $`l`$ 的修复工作量,$`C_l`$ 为该链路可容纳的 crew 上限,$`\Lambda_k`$ 为本 stage 可用的 crew 总数。开工**时刻不是决策**,它隐式地等于 stage 时钟 $`\psi_k`$,也就是前序各 stage 实际工期的累计。

**Action → State**:状态转移是确定性的集合差,即 $`s_{k+1} = s_k \setminus a_k`$。stage 的实际时长为 $`\vartheta' = \max_l \tilde{\sigma}_l / w_l^{\ast}`$($`\tilde{\sigma}`$ 为工作量的随机实现,均匀分布,约在均值的 ±20% 内),时钟按 $`\psi_{k+1} = \psi_k + \vartheta'`$ 推进;已修链路在环境优化模型中的可用参数 $`\beta_l`$ 由 0 置 1,从而改变潮流或流量的解;耦合版还会把完工时刻写入协调器,进而约束对方 agent 的可行动作。交通状态的更新方法是,路网 CI 每次动作后用 Gurobi **重解一次容量约束的 min-cost 交通分配 LP**(电网 CI 则重解 DC 潮流模型),已修链路以 $`\beta_l = 1`$ 进入新解,得到新的流量分布与总 travel time,即 reward 中的 $`Z^{\ast}`$;链路的时间与成本系数固定,不存在拥堵函数。

**Reward**:每次动作把环境优化模型**解两遍**,分别对应"选中链路仍缺席"与"选中链路已恢复"两种网络,取两个最优目标值之差,再乘以剩余期($`T`$ 为规划 horizon,$`t_k`$ 为本 stage 的决策时刻):

```math
\theta_k = \big[Z^{\ast}(\mathrm{absent}) - Z^{\ast}(\mathrm{restored})\big]\cdot\big(T - t_k - \vartheta'\big)
```

计算层面,电网的 $`Z^{\ast}`$ 是在 DC 潮流约束(节点功率平衡、线路容量)下求"未满足的日电力需求总量"的最小值;路网的 $`Z^{\ast}`$ 是在容量约束的 min-cost 交通分配 LP 下求"总 travel time/cost"的最小值,即流量乘链路时间的总和;两者均由 Gurobi 求解。耦合版再把 reward 除以"全程无修复时的总缺口",归一化到 [0,1]。

**Q 形式**:**纯表格**,即逐 stage 的 state–decision 矩阵,行是未修链路集合,列是可行修复子集;没有任何神经网络或函数近似,深度学习被作者明确留作 future work。

**Q 更新**:表格 Q-learning($`\alpha=0.25,\ \gamma=1`$),不使用 replay 或 target 网络:

```math
Q(s_k,a_k) \leftarrow (1-\alpha)\,Q(s_k,a_k) + \alpha\big[\theta_k + \gamma \max_{a'} Q(s_{k+1},a')\big]
```

耦合版在 target 内额外加上 $`+\,\lambda\, MQ^{\mathrm{other}}(s_k^{\mathrm{other}})`$($`\lambda=0.1`$),即对方 CI 当前 stage 状态的 max-Q,由 Prediction Module 共享。每次迭代逐 stage 更新;当相邻迭代的 Q 值差低于阈值或达到最大迭代数时停止。

---

## idx 60 — Ali et al. 2026, *Graph-based deep RL for adaptive and scalable post-earthquake bridge network restoration*(GNN-DRL)

*作者:Ali Muhammad, Du Ao, Cai Jiannan;期刊:Reliability Engineering & System Safety;年份:2026。*

**Agent**:单一 agent,建模上指**网络级的桥梁修复规划者**,即掌握预算、决定每月哪些桥开工的中央决策机构;它面对的环境是整个桥梁网络的物理与预算状态。

**Policy function**:π 是对 Q 的 greedy/ε-greedy 决策规则。论文明确采用 value-based 路线,文中的 "policy network" 一词指的就是在线 Q 网络。训练期用 ε-greedy(ε 在前 40 个 episodes 内从 1.0 线性降到 0.01);部署时贪心选择,并在动作选择时施加预算硬约束。

**决策变量**:执行层面决定的是**桥编号与启动月份的组合**,这是七篇中唯一显式带时间指标的决策。变量为 $`a_{i,t} \in \{0,1\}`$,其中 $`i`$ 是桥编号,$`t`$ 是月份($`T=48`$);取 1 表示桥 $`i`$ 在第 $`t`$ 月启动修复,取 0 表示不动。网络级动作是联合向量 $`a_t = (a_{1,t}, \dots, a_{N,t})`$,共 $`2^N`$ 种组合($`N`$ 为桥数);在 Top-k 变体下,动作被限制在 15 座最高分桥上。桥一旦开工就不可中断直至完工;模型不含 crew 分配或抢占决策。

**Action → State**:转移是确定性的脚本规则。启动某桥的修复后,该桥的在修标志置 1,剩余工期置为 $`\Delta_i`$(按损伤态分别为 1、4、7、12 个月),并且**全额修复费立刻从预算中扣除**($`C_{\mathrm{repair},i} = 1725\cdot C_{r,i}\cdot d_i`$,其中 $`C_{r,i}`$ 为损伤态对应的修复成本比例,取 3%、8%、25% 或 100%,$`d_i`$ 为桥面面积,1725 为全替换单价,单位 USD/m²)。此后每步在修桥的工期减 1,功能率按 $`\alpha_i(t{+}1) = \alpha_i(t) + \big(1-\alpha_i(0)\big)/\Delta_i`$ 线性恢复;完工时损伤态归为 None,$`\alpha_i=1`$。预算每步获得固定拨款,结余滚存。需要特别指出,交通状态在本文中**不更新**:全程没有任何交通分配或仿真,ADT 与绕行长度 $`L_i`$ 固定为灾前参数;action 只改变物理侧状态(在修标志、工期、功能率、预算),交通相关量仅仅是功能率 $`\alpha_i(t)`$ 的确定性线性函数,间接成本随之线性下降。

**Reward**(权重 $`w_1 = w_2 = 2,\ \lambda = 0.5`$):

```math
r_t = r_1 + r_2 + r_3 + r_4
```

第一项 $`r_1 = -\sum_i \delta_i\, IC_i(t)`$ 是中心性加权的间接成本($`\delta_i`$ 为归一化 betweenness 与 closeness 的均值),其中 $`IC_i(t) = C_O(t) + C_T(t)`$,计算层面为:

```math
C_O(t) = \big[C_{O,\mathrm{car}}\,(1-r_{\mathrm{truck}}) + C_{O,\mathrm{truck}}\, r_{\mathrm{truck}}\big]\cdot \mathrm{ADT}_i \cdot \big(1-\alpha_i(t)\big)\cdot 30
```

```math
C_T(t) = \big[C_{W,\mathrm{car}}\, O_{\mathrm{car}}\,(1-r_{\mathrm{truck}}) + C_{W,\mathrm{truck}}\, r_{\mathrm{truck}}\big]\cdot \mathrm{ADT}_i \cdot \frac{L_i}{V} \cdot \big(1-\alpha_i(t)\big)\cdot 30
```

含义是,桥 $`i`$ 当月受影响车流等于日均交通量 ADT 乘以功能缺口 $`(1-\alpha_i(t))`$,再乘以 30 天;$`C_O`$ 按 detour 车辆的单位运营成本计价,$`C_T`$ 按绕行长度 $`L_i`$ 除以车速 $`V`$ 再乘单位时间价值计价($`r_{\mathrm{truck}}`$ 为货车比,$`O_{\mathrm{car}}`$ 为载客率);所有交通参数取**固定灾前值**,不做流量重分配。其余三项分别是:$`r_2 = w_1 \sum_i f_i\,\Delta\alpha_i(t)\, C_{\mathrm{indirect},i}`$,即在修桥当步功能增量的进度激励($`f_i`$ 为在修标志,$`C_{\mathrm{indirect},i}`$ 为该桥间接成本);$`r_3 = w_2\, C_{\mathrm{network}}`$,即全网全部修复完成那一步一次性发放的奖励,金额等于全网重置成本;$`r_4 = -\lambda\,\max(0,\, -B_{\mathrm{avail}})`$,即对预算 $`B_{\mathrm{avail}}`$ 透支量的软惩罚。

**Q 形式**:**深度神经网络(图神经网络)**。输入是桥梁图,节点特征包括当前与初始 HAZUS 损伤态、中心性 $`\delta`$、功能率 $`\alpha(t)`$、ADT、桥面面积、绕行长度、货车比、修复成本与时长、在修标志、可用预算,Top-k 变体另加一个 top-k 成员指示位。结构为 1 层 GCN(hidden 36, ReLU)做节点编码,经 global pooling 得到图级嵌入,再经 2 层 FC(36, ReLU)与线性输出层,对每个联合动作组合输出一个 Q 值;GCN 编码器与 DQN 头端到端联合训练。

**Q 更新**:标准 DQN target 加平方 TD 误差($`\gamma=0.9`$;$`\theta`$ 为在线网络参数,$`\theta'`$ 为 target 网络参数):

```math
y = r + \gamma \max_{a'} Q_{\theta'}(s', a'), \qquad L(\theta) = \big(y - Q_{\theta}(s,a)\big)^2
```

使用 PER 缓冲(容量 $`10^5`$),minibatch 为 36(另一处写 32,原文不一致);target 网络每 40 episodes 拷贝一次。训练使用随机初始损伤情景,测试使用震害情景(Mw 7.7 的 Monte Carlo 实现)。

---

## idx 98 — Hongbo et al. 2026, *An emergency recovery strategy for road network resilience based on dynamic structure-function adaptive trade-offs*

*作者:Qin Hongbo, Cao Taiyu, Li Penglong, Zhu Yongming;期刊:Transportation;年份:2026。*

**Agent**:单一 agent,建模上指**恢复决策者**,即为给定失效场景排定节点修复次序的中央规划者;它面对的环境是双层(交通层与流量层)网络及其级联失效动力学。

**Policy function**:π 是对 Q 表的 greedy 决策规则,即 $`\pi^{\ast}(s) = \arg\max_a Q(s,a)`$(π 在 MDP 元组中被形式化定义)。每步只修一个节点且不可逆,因此候选集逐步收缩;非法动作(重复修复或不可行转移)通过惩罚 reward 抑制,而不是硬 masking。

**决策变量**:执行层面决定的是**失效节点编号**。每步动作 $`a_t \in A_t`$ 表示下一个要修复的失效节点(候选集 $`A_t`$ 随修复进程收缩,一步修一个、不可逆);时间隐含在步序中,没有显式的时刻变量。最终交付一张有序的修复优先级表,例如场景 1 的最优序列为 J17, J26, J11, J28 等,对应 resilience 0.8555,比运行性能基线高 24.50%。

**Action → State**:修复节点 $`i`$ 之后,其状态位由 0 置 1(转移确定,成功率为 1)。环境是交通层与流量层构成的双层网络,恢复动作把枢纽节点及其连边同步加回两层。交通状态的更新方法是 **Motter-Lai 级联失效模型的负载再分配**,它不是交通分配,也没有 OD 或路径概念:节点容量正比于初始流量乘容忍系数;失效节点把全部流量、过载节点把超额流量,按邻接节点的剩余承载力比例转移;修复因此可以解除级联性拥堵,例如场景 2 中一步就消除了全部再分配拥堵。运行性能(延误指数、排队长度)由再分配后各节点的负载与容量之比重算,综合性能随后按下式重算:

```math
P(t) = w_1(t)\,P_{\mathrm{structural}} + w_2(t)\,P_{\mathrm{operational}}, \qquad w_2 = 1 - w_1
```

**Reward**:$`r_t = P(t{+}1) - P(t)`$,即执行动作后综合性能的增量(Eq.21);非法动作施加固定惩罚(Eq.22)。计算层面,$`P_{\mathrm{structural}}`$ 是网络归一化 closeness 与 betweenness 中心性的组合,属于纯拓扑量,恢复节点后重算;$`P_{\mathrm{operational}}`$ 由节点延误指数与排队长度构成,二者来自 Motter-Lai 再分配后各节点的负载与容量之比;权重 $`w_1`$ 是恢复度的 Sigmoid 函数,随恢复推进从"重结构"切换到"重运行",其中恢复度 $`D(t)`$ 等于已修复节点数除以失效节点总数(连通性阈值 0.8)。由于逐步增量累加是望远镜式求和,累计 reward 恰好等于总性能恢复,与 resilience 最大化目标一致。需要说明,精确公式(Eq.7 到 10、21 到 22)在 PDF 提取文本中全部是占位符,以上内容由正文文字重建。

**Q 形式**:**纯表格**,以失效节点恢复状态的二元向量为 state、候选修复节点为 action 的查找表,覆盖 $`\{0,1\}^{N_f}`$($`N_f \le 12`$);全文没有神经网络或函数近似。

**Q 更新**:标准 Q-learning,朝 $`r + \gamma \max_{a'} Q(s',a')`$ 更新($`0 < \gamma \le 1`$);不使用 replay、target 网络或 double/n-step 变体。由于转移确定,backup 是精确的。具体更新式与超参数只出现在 Table 1 的伪代码里,而该表在 PDF 中是一张图片,无法从文本核实。

---

## idx 133 — Babaee et al. 2026, *Optimizing post-disaster road restoration with RL: traveler-behavior-aware*(TARM)

*作者:Babaee Maryam, Saha Namrata, Ponce Frank Mediavilla, Rezapour Shabnam, Amini M. Hadi;期刊:Reliability Engineering & System Safety;年份:2026。*

**Agent**:单一 agent,论文明确指 **FHWA 式的修复规划机构**(restoration planner),即统筹损毁链路修复次序与 crew 调配的公路管理部门;它面对的环境是 day-to-day 演化的交通网络。

**Policy function**:π 是对 Q 表的 greedy/ε-greedy 决策规则。候选动作必须使内层 crew-MILP 有可行解才算合法;第一轮迭代纯随机选择,之后使用衰减 ε-greedy;收敛后按 Eq.(17) 从 Q 表贪心读出确定性策略。

**决策变量**:执行层面有两个决策。第一个是 $`L''_k`$,即第 $`k`$ 个 stage 并行修复的**损毁链路编号子集**(允许空集,表示本轮闲置);第二个是 $`w(n,m)`$,即链路 $`(n,m)`$ 投入的 **crew 数**,它由内层 min-max MILP 决定($`\min\ \theta = \max_l \sigma_l / w_l`$,其中 $`\sigma_l`$ 为链路修复工作量,$`\theta`$ 为本轮完工时间),不是 RL 直接选的。开工**时刻不是决策**,它等于决策 epoch $`t_k`$,也就是上一 stage 的完工时刻。

**Action → State**:集合转移为 $`s_{k+1} = s_k \setminus L''_k`$。完工时长是随机的,$`\theta' = \max_l \tilde{\sigma}_l / w_l^{\ast}`$($`\tilde{\sigma}`$ 为工作量的随机实现,均匀分布,约在均值 ±20% 内,并正比于链路长度)。修复完成后,链路重新进入网络,使目标均衡 $`Y`$ 发生变化,流量随之按 day-to-day 规则演化:

```math
X_{t+1} = X_t + \alpha\,(Y - X_t), \qquad Y = \arg\min_y\ \lambda \sum c(x)\,y + \lambda' \sum \lvert x - y \rvert
```

其中 $`\lambda`$ 与 $`\lambda'`$ 分别是成本项与惯性项的权重;$`c`$ 是 BPR 链路时间,$`c = c_0\big[1 + 0.15\,(x/B)^4\big]`$,$`x`$ 为链路流量,$`B`$ 为链路容量。交通状态的更新方法因此是 **day-to-day 流量演化**:每天走一步,流量向 GUROBI 重解出的目标均衡 $`Y`$ 移动比例 $`\alpha`$,而不是瞬时跳到均衡;链路时间由 BPR 随流量逐日重算。

**Reward**(Eq.9 与 Eq.10,取"不修这些链路"与"修了"两条轨迹的总旅行时间之差):

```math
r(L'') = \sum_{t=\theta''+1}^{T} \Big[\sum_{l} c\big(x'_{l,t}\big)\,x'_{l,t} - \sum_{l} c\big(x_{l,t}\big)\,x_{l,t}\Big]
```

计算层面,每一天的总旅行时间等于各链路"BPR 时间乘流量"之和;$`x'_{l,t}`$ 是**不修** $`L''`$ 时的 day-to-day 仿真流量轨迹,$`x_{l,t}`$ 是**修了**之后的轨迹,两条轨迹都从修复完成日 $`\theta''{+}1`$ 仿真到 horizon $`T`$;reward 就是两条轨迹逐日总旅行时间之差的累加。它在抽样的修复时长实现上计算而不取期望,Q-learning 的更新平均隐式地逼近期望。链路完全修复之前没有部分收益。

**Q 形式**:**纯表格**,Q 矩阵以未修链路集合为行、并行修复子集为列,覆盖损毁链路的幂集;论文明确不用神经网络,深度 RL 留作 future work。

**Q 更新**:标准单步 Q-learning(Eq.16,$`\xi`$ 为学习率),在线逐转移更新;不使用 replay、target 网络或函数近似;收敛判据为 $`\max \lvert \Delta Q \rvert < 10^{-2}`$:

```math
Q(s_k,d_k) \leftarrow (1-\xi)\,Q(s_k,d_k) + \xi\big[r_k + \gamma \max_{d'} Q(s_{k+1},d')\big]
```

---

## idx 1059 — Liu et al. 2026, *Dynamic recovery sequence optimization of subway-bus network with constrained RL*

*作者:Liu Shan, Meng Zhenhao, Jiang Rui, Wang Zhengli, Zhang Ya, Ge Ying-En;期刊:Transportation Research Part D: Transport and Environment;年份:2026。*

**Agent**:单一 agent,建模上对应**那支唯一的维修 crew**(或者说它的调度者):全网只有一支队伍,agent 的每个决策就是这支队伍下一个去修哪座车站;它面对的环境是地铁与公交耦合网络及其客流仿真。

**Policy function**:π 是 masked Q 上的 greedy 决策规则,动作取可行集内的 argmax。可行集由硬过滤给出(剩余工期须在 horizon 内、crew 空闲、网络可达),另有一条软过滤(移动距离不超过 3 km,若过滤后为空则回退到硬过滤集)。探索依靠 NoisyNet 参数噪声,而不是 ε-greedy。

**决策变量**:执行层面决定的是**损毁车站编号**。每个决策 epoch 的动作 $`a_t`$ 是维修 crew 下一个要修复的车站(地铁站或公交站)编号;时间隐含在序列与工期占用中,因为选站之后 crew 会被占用 $`T_{re}(a_t)`$,期间不再接受指派。动作序列最终给出整个恢复顺序 $`S^{\ast}`$(案例为北京网络,135 个地铁站加 1452 个公交站,horizon 为 12 小时,即 48 个 15 分钟步)。

**Action → State**:选站 $`a_t`$ 之后,crew 被占用 $`T_{re}(a_t)`$ 且期间禁止新指派,该站状态翻转为已修,crew 位置更新。环境仿真随后传播这一动作的后果:网络拓扑与连通性被更新(一条线路只有当其全部节点、包括换乘站都可用时才运营),站点处理能力恢复,滞留乘客按处理能力转移,15 分钟粒度的实测客流更新各损毁站的被困人数(乘客重选路的变体只作稳健性测试)。交通状态(客流)的更新方法是 **rule-based 环境仿真**而非交通分配:每 15 分钟把实测进站客流注入网络,滞留乘客按可用站点的处理能力转移消化,连通性按"整线全部节点可用才运营"的规则重算;拥堵不通过速度或时间函数表达,而体现为滞留存量 $`STU_i`$ 的增减。

**Reward**(Eq.1;权重 $`\alpha=0.001,\ \beta=100,\ \lambda=100`$,手工调参):

```math
r_t = -\alpha\, PST_t + \beta \sum_{n \in N_i} \eta_{i,n}\, Re + \lambda\, U_t
```

计算层面($`D`$ 为损毁站集合,$`N_i`$ 为被修站 $`i`$ 的一阶邻站集):$`PST_t = \sum_{i \in D} STU_i`$,其中 $`STU_i`$ 是站 $`i`$ 的滞留乘客**存量**,由 15 分钟实测进站客流(北京刷卡历史日曲线,含早晚高峰)持续流入损毁站、再按邻近可用站的处理能力转移流出后的余量逐期累计而成;$`Re`$ 表示若被修站的一阶邻站已修复,则计入该邻站的节点重要度 $`E_i`$(换乘站 1.5,普通站 1),$`\eta_{i,n}`$ 为连通边权;$`U_t = \lvert D_{re,t}\rvert / \lvert D \rvert`$ 是已修站数占损毁站总数的比例。若移动距离 $`d`$ 超过 $`d_{\max}`$(3 km),另加软惩罚 $`r_t \leftarrow r_t - \psi\,(d - d_{\max})`$;HER 会对 relabel 的转移按新 goal 重算 reward。

**Q 形式**:**深度神经网络(动态图神经网络加 dueling 头)**。输入 state 包括各站运行与损毁状态、滑动窗口内的实时进出客流($`\omega=6`$ 个 15 分钟)、各站维修时长、crew 当前位置、站间连通性。DGCN 编码先对滑窗客流张量 $`Z_{t-\omega:t}`$ 作 Conv1D,构造动态邻接 $`A_{\mathrm{Dyn}} = A_{\mathrm{Static}} \odot \sigma\big(\mathrm{ConvNet}(Z_{t-\omega:t})\big)`$,再经双路 GCN(ELU)得到节点嵌入,并与 MLP 编码的上下文融合;随后两条 NoisyNet 全连接流分别输出 $`V(s)`$ 与 $`A(s,a)`$,按 dueling 方式合成,每个候选损毁站对应一个 Q 值:

```math
Q(s,a) = V(s) + \Big(A(s,a) - \frac{1}{|A|}\sum_{a'} A(s,a')\Big)
```

**Q 更新**:标准 Double-DQN target 加 masking($`\gamma=0.95`$;$`M(s')`$ 为可行动作掩码;终止态时 $`Y = r`$):

```math
a^{\ast} = \arg\max_{a}\ \big[Q_{\theta}(s',a) \odot M(s')\big], \qquad Y = r + \gamma\, Q_{\theta^-}(s', a^{\ast})
```

TD 误差 $`\delta = Y - Q_{\theta}(s,a)`$ 进入逐样本 Huber loss(batch 128,Adam,L2 正则 1e-5)。回放采用 SumTree PER(优先级 $`p = \lvert\delta\rvert + \epsilon`$,$`\epsilon`$ 为小正数,带重要性采样加权)与 HER(单站 goal relabel 生成的虚拟成功转移)混合进同一优先缓冲;target 网络每 200 步硬拷贝一次。

---

## idx 2704 — Joo et al. 2022, *Road-reconstruction after multi-locational flooding in multi-agent deep RL*

*作者:Joo Soo-hyun, Ogawa Yoshiki, Sekimoto Yoshihide;期刊:International Journal of Disaster Risk Reduction;年份:2022。*

**Agent**:多 agent,建模上指**工程队**(operation crew):共 12 个 agent,分 3 类规模(4、8、17 人),每个 agent 是一支独立选路施工的队伍。各 agent 是独立学习者,把其他 agent 视为环境的一部分,并通过学习型通信协议交换消息。

**Policy function**:π 是对 Q 的衰减 ε-greedy 决策规则,论文的 Eq.2 就是以 $`\pi(s)`$ 的形式给出的。

**决策变量**:执行层面决定的是**道路编号**。每个工程队每步(1 步为 1 天)的动作 $`u_t^k`$ 是从本组 15 条目标损毁路中选出 1 条的编号,并投入当日工作量(人均 256 m²);时间隐含在天步中。3 个组共覆盖 45 条损毁路;12 个 agent 同时行动,同组 agent 共享同一动作空间,因此存在竞争。episode 在总体恢复率达到 75% 或走满 35 步时终止。

**Action → State**:选择道路 $`m`$ 之后,该路的累计工作量加上当日投入,进度率按 sigmoid 演化($`W_k`$ 为队伍 $`k`$ 的当日工作量,$`S^m`$ 为路 $`m`$ 的总工作量,按损伤风险加权,满负荷投入时 14 天修完):

```math
P^m_t = \frac{1}{1 + e^{-0.8\,x_t}}, \qquad x_t = -7 + 13\cdot\frac{\sum_k W_k}{S^m}
```

道路容量随进度成比例恢复(初始容量为 0);OD 需求 $`q_{(rs),t}`$ 随其常日路径上损毁路段的**最小**累计进度恢复(Assumption 2);流量按容量比例分配,$`F^k_t = q_{(rs),t}\cdot C^k_t / \sum_k C^k_t`$($`F^k_t`$ 为路径 $`k`$ 的流量,$`C^k_t`$ 为该路径的瓶颈容量,即沿线最小链路容量),再叠加得到路段流量,重算各路与总体 mobility 恢复率并回填进下一 state。交通状态的更新方法,论文自称是 stochastic traffic assignment,实际是**容量比例的启发式分流**:门控后的 OD 需求按各可行路径的瓶颈容量占比一次性分配、叠加成路段流量;其中没有 travel cost 项,没有拥堵反馈,也没有迭代均衡。

**Reward**(shaped,总 reward 等于基本项 R 加额外项 F)。

基本项的计算链是,先算全网 mobility 恢复率:

```math
TR^{O}_t = \frac{\sum_a F_{a,t}}{\sum_a F_{a,n}}
```

分子是当日全网各路段**估计交通量** $`F_{a,t}`$ 之和,分母是正常日(灾前)各路段交通量之和(下标 $`n`$ 表示正常日)。路段交通量的来源是,GPS OD 需求 $`q_{(rs),t}`$ 先被"途经损毁路段的最小修复进度"门控,再按各可行路径的容量占比分流,叠加到各路段得到 $`F_{a,t}`$。基本 reward 等于恢复率的**增量**乘以被选路的权重:

```math
R_{t+1} = \big(TR^{O}_{t+1} - TR^{O}_{t}\big)\times \alpha_{rt} \times 100
```

其中 $`\alpha_{rt}`$ 是被选路 $`r`$ 的交通量占全网总交通量的比例(traffic weight),100 是整数化权重。

额外项 F 有两部分。其一是合作加成 $`W_t = \sum_k CP^k_t`$,其中(Eq.8)$`CP^{A_o}_t`$ 等于**同时施工的两条路共同承载**的流量除以全部被选路承载流量之和,乘以权重 20;其二是通勤惩罚,按队伍驻地到工地的行程时间以 20 分钟为一档,计 0 到 −3。敏感性分析变换过各权重(恢复权重取 20 到 180,合作权重取 4 到 36,替换为全局均一 reward,以及改用 10 分钟分档)。

**Q 形式**:**神经网络,但类型与结构完全未报告**。论文只说用神经网络近似 action-value function,层数、架构、优化器、loss 均未给出,无法判断是 MLP 还是其他架构。输入是 state 向量(Eq.9),包括各路施工进度率 $`Pr`$、各路 mobility 恢复率 $`TR`$、行程时间 $`T_t`$、总体恢复率 $`TR^O`$、来自其他 agent 的通信消息 $`CP`$(消息 $`m_t^k`$ 拼进输入层,由网络学会解读)以及选中路的 traffic weight $`\alpha_{rt}`$;输出是该组 15 条候选路各自的 Q 值。

**Q 更新**:Q-learning 更新,而且 shaping 奖励 $`F`$ 直接进入 target(Eq.3;需要注意,印刷的 max 是对 $`s_t`$ 而非 $`s_{t+1}`$ 取的,疑似 typo):

```math
Q_{t+1}(s_t,a_t) = (1-\alpha)\,Q_t(s_t,a_t) + \alpha\big[R_{t+1} + F_{t+1} + \gamma \max_a Q_t(s_t,a)\big]
```

更新在 episode 结束时用累积经验批量进行;replay、target 网络、loss 与更新频率均无描述;$`\alpha`$ 与 $`\gamma`$ 的数值未报告;没有 double、dueling 或 n-step 变体。
