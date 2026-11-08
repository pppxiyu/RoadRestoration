# Road restoration research code

本仓库区分当前标准方法和已用于论文的历史实验。`main.py`运行当前标准，`src.config.results_profile`选择已登记的Results配置。历史模型按真实训练设置复现，不会自动套用当前默认值。

## 1 · 从哪里开始

| 目的 | 入口 |
| --- | --- |
| 查看论文各组图及分析逻辑 | [outputs/sections/README.md](outputs/sections/README.md) |
| 找实验、模型、数据和HPC位置 | [outputs/experiments/README.md](outputs/experiments/README.md) |
| 查看探索性研究 | [outputs/studies/README.md](outputs/studies/README.md) |
| 当前方法参数与Results配置 | `src/config.py`及`src/results_profiles.json` |
| 当前完整RL训练与阶段改进 | `src/experiment/daily_rl.py`及`src/methods/daily_improvement.py` |

`doc/`、`paper/`、`.cache/`和`.legacy/`保留既有结构。根目录不再另设analysis、reproducibility或临时编译目录。

## 2 · 当前标准问题与RL

问题版本为`finalized_poi_public50_floor_20261108`。初始公开受损道路数量为`floor(0.5*n)`，其余初始隐藏。n为6、11、17、23时，公开与隐藏数量分别为3/3、5/6、8/9、11/12。道路分层抽样、隐藏道路发现和真实维修时长继续由`src/problem`生成，不能只按数量替换已有场景。

完整环境保留七日需求响应、需求偏差记忆和累计未满足需求的POI反馈。一个非抢占式维修队按整数日维修，不允许有候选动作时主动等待。系统评价累加50%时间损失和50%逐OD流量损失。所有维修完成且交通满足连续两日恢复条件后才结束，day35不是恢复期限。

RL使用双向OD编码和需求记忆。基础训练从随机权重开始，学习率0.0005，奖励是道路开通后至灾后day60的单次维修长期改善。阶段改进使用64场景基础训练之外的扩展训练集合，共256个训练场景，11个验证场景和50个测试场景保持分离。

阶段改进覆盖全部决策位置，每次抽取8个状态。每轮从紧邻上一轮候选继续，历史最佳模型单独保留。连续5轮没有刷新历史最佳验证损失后停止，最终只按验证结果选模。改进初始学习率0.001，ReduceLROnPlateau跨轮保留，factor为0.5、patience为1、最小学习率0.00001。每轮新建Adam，轮内中断恢复优化器、调度器和随机状态。这些新标准不表示历史4.2或4.3模型已经按它重训。

## 3 · 运行当前方法

直接依赖见`requirements.txt`。当前本地环境为Python3.14系列，历史HPC运行使用各实验记录中的独立环境。代理模型研究有自己的DGL环境，不是主程序依赖。

```powershell
python -m pip install -r requirements.txt
python main.py --n 11 --describe
python main.py --n 11 --solve rl_s2v_saa64_adaptive --training-behavior response7d --workers 4
python main.py --n 11 --solve ga --workers 4
```

`--describe`不训练策略。`--n`支持6、11、17、23及mixed。`natural`是明确的训练环境对照，`both`分别训练两种环境。各模型在自身训练假设下验证，最终完整环境测试与训练选择分开。新实验写入`outputs/experiments`，既有源码和输入身份不同则不混用检查点。

## 4 · 重现已确定的Results

```powershell
python -m src.analysis.results list
python -m src.analysis.results verify all
python -m src.analysis.results render all
python -m src.analysis.results prepare behavior_transfer --output C:/work/replay_transfer
```

`verify`检查原文件身份并独立重算关键统计。`render`只读已保存科学结果，重画图和动画。`prepare`建立新的独立计算目录，复制对应冻结源码和输入，写明执行命令及HPC脚本，不自动训练或提交。排名与动画的大数组留在HPC，逐文件位置和SHA有记录，具体步骤见实验README。

当前归位的7张PNG和2个完整HTML均保留原交付形式。4.2的简化训练模型、受控需求实验与4.3完整RL是不同的研究配置，不能合并成一个默认模型。实验代码可用于独立重算，改变科学设置后应另存新实验。

这些图已从保留数据重画并逐字节核对通过。完整仿真重算尚未全部验证，双环境评价的小样本检查中有两条完整环境轨迹出现微小数值差异，原结果保留，具体检查范围及后续复核见实验README。

## 5 · 模块与文件责任

| 模块 | 责任 |
| --- | --- |
| `src/problem` | 路网、场景、信息披露及维修条件 |
| `src/environment` | 道路和需求演化、交通分配、观测及系统评价 |
| `src/methods` | 决策网络、规则、GA及阶段改进算法 |
| `src/experiment` | 训练、评价、检查点和独立实验重放 |
| `src/analysis` | Results统计核对和可视化 |

`outputs/sections`面向论文，`outputs/studies`保存研究分析，`outputs/experiments`保存实验和来源。原模型、缓存及失败证据不因整理而删除。图形修改不改变原始数据，重算不覆盖历史实验。

## 6 · 版本控制范围

Git仅发布源码、输入数据、源码中的Results配置及选定的根目录文件。`doc/`和`outputs/`全部保留在本地，不纳入Git，也不上传GitHub。上文指向这些目录的链接用于本地仓库。大权重、原始数组和本地历史材料保存在本地或HPC，并通过本地`outputs/experiments/index.json`和各`artifact_manifest.json`匹配。只有源码的干净克隆不包含这些文档、实验索引和结果数据。
