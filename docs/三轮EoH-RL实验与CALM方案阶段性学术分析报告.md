# 三轮 EoH-RL 实验与 CALM 方案阶段性学术分析报告

> 报告性质：阶段性学术科研分析报告  
> 分析日期：2026 年 7 月 14 日  
> 理论与方案依据：《AAAI2027 - 分支 · CALM论文与代码分析》  
> 实验数据范围：logs/TSP/eoh_local_rl  
> 主统计单位：独立 run/seed  
> 任务范围：TSP50  

## 摘要

本报告依据《AAAI2027 - 分支 · CALM论文与代码分析》最终收敛的研究路线，对本项目已经完成的三轮 EoH-RL 实验进行可复核的阶段性分析。PDF 将核心科学问题界定为：**在线强化学习究竟提高了本地模型生成高质量 heuristic 的条件概率，还是主要提高了合法候选供给率，并由外部进化搜索放大这一变化？** 因此，本报告不将“训练执行成功”或“合法率提高”直接解释为 heuristic-design knowledge internalization，而是分别分析工程真实性、端到端搜索性能、合法性漏斗、相对 parent 的条件质量、分布退化、资源成本及实验设计完备性。

三轮实验依次为：

1. 第一轮：6 seeds、500 代、每代 4 responses 的 B0 固定模型与 aggregate GRPO 对照；
2. 第二轮：3 seeds、100 代的 LR=0、validity-only、performance-shuffled、novelty、reset/persistent 判别性 pilot；
3. 第三轮：6 seeds、500 代的 contemporaneous B0、aggregate reset、aggregate persistent+低学习率、novelty reset 扩展验证。

第三轮主体中三个未完成 run 由 recovery 批次补齐，最终形成 4 个变体 × 6 seeds × 500 updates × 4 responses 的完整 canonical 数据集。所有纳入第三轮统计的 run 均具有 completed manifest、500 次 online update 和完整 finish checkpoint；中断目录没有被混入统计。

核心结论如下。

1. **工程链路已经基本可信。** 第二、三轮日志能够验证 GRPO 开关、optimizer reset/persistent 语义、adapter 参数变化及显式 GRPO 数学配置。LR=0 control 虽完成 Trainer/optimizer 调用，但 adapter norm 完全不变；persistent 每个 seed 仅首次初始化 optimizer，后续状态持续复用。
2. **第一轮没有观察到 aggregate GRPO 的端到端优势。** GRPO 相对 B0 的 paired mean difference 为 -0.001568，95% CI 为 [-0.062364, 0.059228]，3 胜 3 负，paired t-test p=0.9497。
3. **第二轮只能作为候选筛选。** aggregate/persistent 和 novelty/reset 相对 LR=0 的均值分别为 +0.077170 和 +0.050808，且均为 3 胜 0 负，但 n=3，置信区间均跨 0。
4. **第三轮仍未产生显著收益。** aggregate/reset 相对 B0 平均下降 -0.023929，胜 1/6；persistent/LR=1e-5 平均提高 +0.026153，胜 4/6；novelty/reset 平均提高 +0.011351，胜 2/6。三者的 paired t-test 和 exact Wilcoxon 均未显著。
5. **aggregate GRPO 的主要稳定效应是提高合法候选供给。** 第三轮 B0、aggregate/reset、persistent、novelty 的 population-eligible rate 分别为 77.68%、91.59%、92.08%、92.11%。但 aggregate/reset 与 persistent 的 archive duplicate rate 分别达到 33.80% 和 43.49%，parent tie rate分别达到 39.62% 和 32.89%。
6. **合法率提高没有转化为条件质量提高。** 第三轮在 eligible 条件下的 parent-win rate，B0 为约 0.966%，aggregate/reset 为 0.628%，persistent 为 0.624%，novelty 为 0.805%。aggregate 系列提高了进入评估漏斗的数量，却降低了单个 eligible candidate 击败 direct parent 的条件概率。
7. **当前仍停留在 PDF 所定义的第二阶段。** A1 repair/template、A5 VC-PAIR、fixed prompt/parent bank、same-valid curve、search-free comparison、constrained-valid control、no-SFT fair core、OBP 主任务和 parameter-only transfer 均未完成。
8. **当前最符合数据的解释是 validity-mediated search amplification 伴随分布集中。** 尚无证据足以支持本地模型已学习可跨搜索轨迹复用的 problem-family proposal prior。

**关键词：** EoH；GRPO；CALM；自动启发式设计；validity-mediated search amplification；online post-training；paired experiment；reward collapse；VC-PAIR

---

## 1. 研究问题与 PDF 判定框架

### 1.1 PDF 最终收敛的研究主线

PDF 经文献、机制、因果、方法、系统和审稿红队讨论后，将宽泛方案：

> decomposition + contrastive reward + operator scheduling + curriculum + SFT + reuse

收敛为：

> validity-mediated diagnosis + VC-PAIR + controlled internalization evaluation

最终科学问题不是“GRPO 能否作为组件嵌入 EoH”，因为现有代码已经实现了这一工程事实；真正需要回答的是：

> 在控制协议遵循、代码合法性、parent、prompt、实例批次和搜索预算后，在线 RL 是否提高了合法输出条件下的 heuristic quality、右尾质量以及跨 family 的 proposal prior？

PDF 明确反对从“GRPO 后 end-to-end best score 改善”直接推出“模型内化了 heuristic-design knowledge”。替代解释包括：

- 模型只学会输出可解析函数；
- 模型只学会保持函数签名、避免异常或规避 randomness；
- 预训练代码模式被激活，但没有形成任务族层面的 proposal prior；
- 搜索器获得更多可评估候选，真正的质量筛选由 population selection 完成；
- 不同方法在第一代后进入不同 parent/prompt 分布，造成表观收益；
- on-policy 训练使模型重复 current parent/frontier 附近的程序；
- SFT adapter 已经承担了协议学习，GRPO 的独立贡献被混淆。

### 1.2 描述性漏斗分解

令：

- \(C\)：response 满足代码解析和目标函数 contract；
- \(V\)：程序可执行并通过 population eligibility gate；
- \(\Delta\)：candidate 相对 direct parent 的 oriented improvement，统一为越大越好；
- \(Y=1[C=1]1[V=1]\max(\Delta,0)\)：有用候选产出。

则可进行以下 operational decomposition：

\[
E[Y]=P(C)P(V\mid C)P(\Delta>0\mid C,V)
E[\Delta\mid C,V,\Delta>0].
\]

该公式仅是**描述性漏斗分解**，不是 causal mediation decomposition。在线训练会同时改变 valid-output distribution、parent population、prompt/operator distribution 和 archive 状态；动态搜索日志中的候选并非来自共同输入分布，不能仅凭上述乘积分解识别模型能力的直接因果效应。

### 1.3 PDF 推荐的最终实验矩阵

| 变体 | 训练 | 合法性处理 | 性能信号 | 科学用途 |
|---|---|---|---|---|
| A0 Base + EoH | 否 | 原始生成 | 无 | 基础搜索基线 |
| A1 Base + repair/template + EoH | 否 | 强制/修复 | 无 | 判断合法率收益是否无需训练 |
| A2 CALM-style GRPO | 是 | 分层 penalty | aggregate parent-relative | 强 RL baseline |
| A3 Validity-only GRPO | 是 | 分层 penalty | 无真实性能信号 | 估计 validity contribution |
| A4 Performance-shuffled GRPO | 是 | 同 A2 | group 内打乱 valid candidate 的性能对应 | 检验 performance credit |
| A5 VC-PAIR GRPO | 是 | validity gate | paired uncertainty-aware 三态信号 | PDF 推荐主方法 |
| A6 VC-PAIR + scheduler | 是 | 同 A5 | 同 A5 | 仅在 operator pilot 足够强时进入 |

VC-PAIR 的关键不是增加 reward 项，而是要求 candidate 与 direct parent 在**相同实例批次**上成对比较，并依据测量不确定性将结果分为 positive、neutral、negative，防止微小噪声被误当成稳定性能差异。

### 1.4 本报告的研究问题

- **RQ1 工程真实性：** GRPO 开关是否真实控制训练？reset/persistent 是否真实改变 optimizer 状态语义？
- **RQ2 端到端效果：** 相同初始 population、seed 和搜索预算下，GRPO 是否提高 final best？
- **RQ3 合法性放大：** GRPO 是否主要提高 contract/execution/eligible rate？
- **RQ4 条件质量：** 在合法条件下，GRPO 是否提高 parent-win、frontier-win 或正向改进幅度？
- **RQ5 性能 credit：** 正确性能 reward 是否优于 LR=0、validity-only 和 shuffled control？
- **RQ6 分布退化：** zero-variance、tie、duplicate、completion length 是否随训练累积？
- **RQ7 内部化：** 数据能否排除动态搜索、trajectory memory、SFT 协议学习和合法率供给解释？
- **RQ8 阶段判断：** 当前实现覆盖了 PDF 路线的哪一阶段，下一阶段的 Go/No-Go 条件是什么？

### 1.5 PDF 关键论点页码索引

为保证本报告与方案文档的对应关系，以下页码按 PDF 文件的物理页序号计数：

| PDF 论点 | 主要页码 | 本报告对应 |
|---|---|---|
| Validity-mediated diagnosis | 1、10、17、19、22、25–26 | 第 7、11、14 节 |
| Repair/template control | 4–5、8、10、17、19、21–25 | 第 9、10、12 节 |
| Performance-shuffled control | 4、9、25 | 第 4、9、12 节 |
| Same-valid evaluation | 7、10、17、19–25 | 第 7、9、12 节 |
| VC-PAIR | 11、13、19–21、23、25–26 | 第 1、9、12 节 |
| No-SFT fair core | 5、17 | 第 2、9、12 节 |
| Parameter-only transfer | 15 | 第 9、12 节 |
| TSP 与 OBP 主任务 | 21、24 | 第 9、10、12 节 |

报告中的“Go/No-Go”是对 PDF 多轮讨论中方法路线/机制路线判据的归纳标签，不是 PDF 中必须逐字出现的标题。

---

## 2. 数据、实现与统计方法

### 2.1 三轮 canonical 数据

| 轮次 | canonical 日志目录 | 日期 | 纳入内容 |
|---|---|---|---|
| R1 | tsp50_grpo_500_6seed/20260712_run01 | 2026-07-12 | init、B0、aggregate/reset，seed 42–47 |
| R2 | tsp50_stage2_pilot/20260713_stage2_pilot01 | 2026-07-13 | 6 variants，seed 42–44 |
| R3 主体 | tsp50_stage3_500_6seed/20260713_stage3_run01 | 2026-07-13 | 三个 RL 变体的大部分完整 run |
| R3 recovery | tsp50_stage3_recovery_500_6seed/20260714_recovery01 | 2026-07-14 | 补齐 aggregate seed47、persistent seed47、novelty seed46/47 |
| R3 B0 | tsp50_stage3_b0_500_6seed/20260714_b0_run01 | 2026-07-14 | contemporaneous B0，seed 42–47 |

第三轮主体中的 aggregate/reset seed47、persistent seed47 和 novelty seed46 没有 completed finish manifest，本报告排除这些中断目录，使用 recovery 中同 variant/seed 的完整运行；novelty seed47 也取自 recovery。由此避免了“把中途 best 当作 final best”以及同一 seed 被重复计数。

### 2.2 共同实验设置

| 项目 | 实际设置 |
|---|---|
| Task | TSP constructive heuristic |
| Problem size | 50 |
| Evaluation instances | 64 |
| Score | 64 个实例平均路径长度的负值，越大越好 |
| Base model | deepseek-coder-7b-instruct-v1.5 |
| 初始 adapter | TSP SFT LoRA |
| LoRA rank | 32 |
| Population size | 10 |
| Operator cycle | e1 → e2 → m1 → m2 |
| Prompt/update | 1 |
| Responses/prompt | 4 |
| Temperature/top-p | 1.0/1.0 |
| GRPO loss | DAPO |
| Reward normalization | group |
| KL beta | 0.04 |
| Clip epsilon | 0.15 |
| Clip epsilon high | 0.28 |
| Importance sampling | token level，启用 vLLM correction |
| Quantization | 4-bit |
| Optimizer | AdamW 8-bit |
| Main seeds | 42–47 |

三轮均从 SFT LoRA 起步，因而当前数据不是 PDF 推荐的 no-SFT fair core。它可以回答“在当前工程配置中追加 online GRPO 是否有用”，但不能独立估计 base model、SFT 协议学习和 GRPO 的各自贡献。

### 2.3 当前 reward 的实际数学语义

当前 aggregate reward 不是 VC-PAIR，也不应称为严格 CALM 复现。合法 candidate 的 reward 为：

- 超过 current frontier：\(r=1+\operatorname{clip}(\Delta_f/|f|,0,1)\)；
- 仅超过 direct parent：\(r=0.3+\operatorname{clip}(\Delta_p/|p|,0,0.7)\)；
- 未超过 parent：\(r=0.5\operatorname{clip}(\Delta_p/|p|,-1,0)\)。

非法输出采用分层 penalty：

| 失败类型 | reward |
|---|---:|
| parse/contract failure | -1.0 |
| execution exception | -0.8 |
| None 或 non-finite | -0.6 |
| randomness 或 metadata leak | -0.7 |

其余诊断 reward：

- validity-only：所有 eligible candidate 为 +1，不含真实性能 credit；
- performance-shuffled：仅在同一 prompt/group 的 eligible candidate 中循环置换 performance reward；
- aggregate+novelty：archive duplicate 为 -0.3，parent tie 为 -0.1。

因此，aggregate reward 同时混合协议、可执行性、相对 parent 性能、frontier 性能和分段尺度；现有实验不能把其中任一项的效应自动解释为最终性能学习。

### 2.4 指标定义

- **best score：** 每个 update 后 active population 的最优分数；TSP 中 -6.20 优于 -6.30。
- **contract rate：** parse_success/completion_count，验证代码解析和函数 contract。
- **execution rate：** exec_success/completion_count。
- **population-eligible rate：** execution success 且非 randomness、metadata leak、exact parent copy。
- **archive duplicate rate：** eligible candidate 的代码已存在于 archive。
- **archive novel rate：** eligible 且不属于 archive duplicate。
- **parent/frontier win：** candidate score 超过 direct parent/current frontier。
- **zero-reward-std update：** 同一 4-response GRPO group 的 reward 标准差为 0。
- **registered：** candidate 通过 gate 并进入 archive/population 管理过程。
- **survived：** candidate 注册后进入当前 active population。
- **generation：** 日志中 population 演化推进次数，并不等于 completion_count。

contract rate 不要求 response 含完整 idea 文本，所以高 contract 与低 idea rate 并不逻辑矛盾，但会暴露协议指标口径比完整输出规范更宽。

### 2.5 统计原则

1. 第一、三轮主统计单位为 6 个 seed；第二轮为 3 个 seed。
2. 同 seed 使用 paired difference：\(d_s=score_{method,s}-score_{control,s}\)。
3. 报告 paired mean、sample SD、95% t confidence interval、Cohen's \(d_z\)、paired t-test 和 exact Wilcoxon signed-rank test。
4. 12,000 个 candidate 不是 12,000 个独立实验样本；candidate 比例仅作机制描述。
5. 动态 EoH 在第一代后产生不同 parent/prompt trajectory，故 candidate quality 不是 search-free 模型能力的无偏估计。
6. 第二轮以 LR=0 为同期内部 control；第一轮 B0@100 只能作跨轮参考。
7. 第一与第三轮虽复用同 initial population 和 seed，但存在代码版本、日志实现、采样及运行时间差异，不将跨轮差异解释为方法因果效应。
8. 当前 n=6 仍不足以对小效应进行高 power 检验；“不显著”不等于严格证明无效，但置信区间限定了当前可支持的效应范围。

---

## 3. 第一轮：GRPO 开关与冻结初始种群基线

### 3.1 目的与设计

第一轮检验：在相同冻结 initial population、相同 seed 和相同 500×4 预算下，开启 aggregate GRPO 是否优于固定 SFT policy 的纯 rollout/search。

| 变体 | 在线训练 | LR | optimizer | 作用 |
|---|---|---:|---|---|
| init | 否 | — | — | 构造并冻结 10 个初始 population member |
| b0_fixed | 否 | — | 无 | 固定 policy，仅执行 EoH 搜索 |
| b1_grpo | 是 | 5e-5 | 每轮 reset | aggregate GRPO |

### 3.2 冻结初始种群中间值

| Seed | 初始化 completions | 初始化 evaluations | initial best | population |
|---:|---:|---:|---:|---:|
| 42 | 12 | 10 | -6.424640 | 10 |
| 43 | 16 | 14 | -6.986714 | 10 |
| 44 | 12 | 11 | -6.424640 | 10 |
| 45 | 12 | 12 | -6.387759 | 10 |
| 46 | 12 | 10 | -6.986714 | 10 |
| 47 | 20 | 13 | -6.603477 | 10 |
| 均值 ± SD | — | — | -6.635657 ± 0.282144 | 10 |

initial best 的 seed 间极差约为 0.599，远大于第三轮候选方法的平均差异 0.01–0.03。冻结并配对 initial population 是必要条件，否则方法效应会被初始化方差淹没。

### 3.3 逐 seed 终点结果

| Seed | B0 final | Aggregate GRPO final | GRPO-B0 | 判定 |
|---:|---:|---:|---:|---|
| 42 | -6.230170 | -6.223923 | +0.006247 | GRPO 胜 |
| 43 | -6.240269 | -6.248961 | -0.008692 | B0 胜 |
| 44 | -6.294642 | -6.237697 | +0.056945 | GRPO 胜 |
| 45 | -6.266577 | -6.318471 | -0.051894 | B0 胜 |
| 46 | -6.283799 | -6.216755 | +0.067044 | GRPO 胜 |
| 47 | -6.211750 | -6.290808 | -0.079058 | B0 胜 |
| mean | -6.254535 | -6.256103 | -0.001568 | 3 胜/3 负 |
| SD | 0.032366 | 0.040169 | 0.057932（paired） | — |

Paired inference：

| paired mean | 95% CI | Cohen's dz | paired t p | exact Wilcoxon p |
|---:|---:|---:|---:|---:|
| -0.001568 | [-0.062364, 0.059228] | -0.0271 | 0.9497 | 1.0000 |

平均值接近 0 并不意味着每个 seed 都不受影响。seed 46 的正效应为 +0.0670，seed 47 的负效应为 -0.0791，正负幅度均远大于总体均值，反映的是高方差效应互相抵消。

### 3.4 逐代轨迹中间值

| 变体 | 代数 | best mean | SD | 较 initial 提升 | 相对同代 B0 | contract | execution | eligible |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 | 50 | -6.359386 | 0.049716 | +0.276271 | — | 95.83% | 88.25% | 81.33% |
| B0 | 100 | -6.315480 | 0.051446 | +0.320177 | — | 96.33% | 87.92% | 80.63% |
| B0 | 200 | -6.285024 | 0.041811 | +0.350634 | — | 95.67% | 87.23% | 78.42% |
| B0 | 300 | -6.282416 | 0.042314 | +0.353241 | — | 95.28% | 86.78% | 77.86% |
| B0 | 400 | -6.269087 | 0.042828 | +0.366570 | — | 95.24% | 86.51% | 77.67% |
| B0 | 500 | -6.254535 | 0.032366 | +0.381123 | — | 95.17% | 86.28% | 77.18% |
| Aggregate/reset | 50 | -6.369073 | 0.064219 | +0.266585 | -0.009686 | 98.25% | 94.33% | 88.25% |
| Aggregate/reset | 100 | -6.344154 | 0.067464 | +0.291504 | -0.028674 | 98.79% | 95.17% | 90.33% |
| Aggregate/reset | 200 | -6.307151 | 0.067701 | +0.328507 | -0.022127 | 99.02% | 95.73% | 91.44% |
| Aggregate/reset | 300 | -6.273303 | 0.044467 | +0.362354 | +0.009113 | 99.10% | 96.03% | 91.47% |
| Aggregate/reset | 400 | -6.266716 | 0.047369 | +0.368941 | +0.002371 | 99.19% | 96.41% | 91.77% |
| Aggregate/reset | 500 | -6.256103 | 0.040169 | +0.379555 | -0.001568 | 99.17% | 96.30% | 91.70% |

GRPO 从前 50 代起就稳定提高 contract/execution/eligible，但 best 在 50–200 代落后，300–400 代短暂领先，500 代再次回到近似持平。这是第一条直接证据：**合法候选供给提高与最终高质量尾部提高不是同一个命题。**

### 3.5 全程机制中间量

| 指标 | B0 | Aggregate/reset |
|---|---:|---:|
| updates | 3000 | 3000 |
| responses | 12000 | 12000 |
| population-eligible | 77.18% | 91.70% |
| parent win/all responses | 0.642% | 0.725% |
| frontier win/all responses | 0.442% | 0.608% |
| parent tie | 5.15% | 41.83% |
| registered | 75.98% | 60.28% |
| survived active population | 4.11% | 5.41% |
| zero-reward-std updates | 0.30% | 12.07% |
| response 含显式 idea | 2.95% | 0.20% |

Aggregate 将 eligible 提高 14.52 个百分点，但 parent tie 增加 36.68 个百分点，registered 下降 15.70 个百分点。parent win/all responses 只增加 0.083 个百分点，远小于 eligible 的绝对增量。因此第一轮更符合“validity improvement + distribution concentration”，而不是 heuristic quality 已被学习。

### 3.6 第一轮工程局限

第一轮能确认 B0 optimizer update 为 0、GRPO update 为 500，但早期日志尚未完整固化 optimizer_steps_total、optimizer_state_reused、adapter_norm_before/after、逐步 GRPO 指标和 archive duplicate/novel funnel。第一轮不能独立排除“训练调用执行但参数没有变化”，也无法精确证明每轮重建 optimizer 的语义；第二轮针对这些缺口增加了可审计日志。

---
## 4. 第二轮：reward 与 optimizer 判别性 pilot

### 4.1 目的与变体

第二轮为 3 seeds、100 updates 的 pilot。它不承担论文主结果，而用于区分第一轮中的工程和奖励归因：

1. 只有 Trainer/rollout 路径、没有参数更新时会发生什么；
2. validity-only 能否复现 aggregate 的候选表现；
3. 打乱 candidate-performance 对应后是否合法率仍然提高；
4. novelty penalty 能否缓解 duplicate/tie；
5. persistent optimizer 是否形成可检测的状态延续。

| 变体 | reward | LR | optimizer | 诊断含义 |
|---|---|---:|---|---|
| c1_lr0 | aggregate | 0 | reset | 完整训练调用但参数不更新 |
| validity_reset | validity-only | 5e-5 | reset | 仅学习合法性层级 |
| shuffled_reset | performance-shuffled | 5e-5 | reset | 保留 reward 边际，破坏真实性能对应 |
| novelty_reset | aggregate+novelty | 5e-5 | reset | 抑制 duplicate/tie |
| aggregate_persistent | aggregate | 5e-5 | persistent | optimizer 状态持续 |
| novelty_persistent | aggregate+novelty | 5e-5 | persistent | novelty 与 persistent 联合 |

每个 run 为 100 prompts × 4 responses，共 400 responses；每个变体 3 seeds，共 1200 responses。所有变体继续使用第一轮对应 seed 的冻结 initial population。

### 4.2 最终结果

| 变体 | final best mean ± SD | contract | execution | eligible | archive novel | generation | wall/run |
|---|---:|---:|---:|---:|---:|---:|---:|
| LR=0 control | -6.367554 ± 0.029277 | 97.67% | 90.00% | 83.58% | 62.67% | 251.3 | 0.415 h |
| Validity-only/reset | -6.365515 ± 0.079963 | 98.25% | 91.08% | 87.08% | 82.17% | 330.7 | 0.419 h |
| Shuffled/reset | -6.393262 ± 0.043054 | 98.58% | 91.25% | 89.17% | 79.92% | 322.7 | 0.410 h |
| Novelty/reset | -6.316746 ± 0.052301 | 98.00% | 92.42% | 89.92% | 82.00% | 329.7 | 0.430 h |
| Aggregate/persistent | -6.290384 ± 0.026677 | 98.58% | 93.17% | 86.00% | 76.58% | 307.3 | 0.399 h |
| Novelty/persistent | -6.349659 ± 0.058622 | 98.17% | 92.58% | 90.33% | 81.58% | 327.3 | 0.366 h |

在 100 代终点，aggregate/persistent 最好，novelty/reset 次之，shuffled/reset 最差；validity-only 与 LR=0 接近。该排序是第三轮选择 aggregate reset、persistent 低学习率和 novelty reset 的依据，而不是显著性结论。

### 4.3 逐 25 代中间值

| 变体 | 代数 | best mean | SD | 较 initial 提升 | 相对 LR=0 | contract | execution | eligible | archive novel |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LR=0 | 25 | -6.401037 | 0.001571 | +0.210962 | — | 97.33% | 88.00% | 84.67% | 73.33% |
| LR=0 | 50 | -6.384500 | 0.017913 | +0.227498 | — | 97.67% | 90.17% | 84.33% | 71.33% |
| LR=0 | 75 | -6.381068 | 0.023505 | +0.230930 | — | 97.67% | 89.56% | 83.67% | 65.00% |
| LR=0 | 100 | -6.367554 | 0.029277 | +0.244444 | — | 97.67% | 90.00% | 83.58% | 62.67% |
| Validity/reset | 25 | -6.396759 | 0.076743 | +0.215239 | +0.004278 | 98.67% | 90.33% | 85.33% | 83.33% |
| Validity/reset | 50 | -6.394483 | 0.080607 | +0.217515 | -0.009982 | 98.00% | 90.83% | 86.67% | 84.17% |
| Validity/reset | 75 | -6.366336 | 0.079547 | +0.245662 | +0.014732 | 98.22% | 91.44% | 87.56% | 84.11% |
| Validity/reset | 100 | -6.365515 | 0.079963 | +0.246483 | +0.002039 | 98.25% | 91.08% | 87.08% | 82.17% |
| Shuffled/reset | 25 | -6.436666 | 0.059586 | +0.175332 | -0.035630 | 99.33% | 86.33% | 84.33% | 80.00% |
| Shuffled/reset | 50 | -6.407896 | 0.029018 | +0.204102 | -0.023396 | 98.83% | 89.83% | 86.83% | 79.33% |
| Shuffled/reset | 75 | -6.407896 | 0.029018 | +0.204102 | -0.026829 | 98.67% | 90.11% | 87.78% | 79.89% |
| Shuffled/reset | 100 | -6.393262 | 0.043054 | +0.218736 | -0.025708 | 98.58% | 91.25% | 89.17% | 79.92% |
| Novelty/reset | 25 | -6.360887 | 0.019466 | +0.251111 | +0.040149 | 98.00% | 92.33% | 89.33% | 85.00% |
| Novelty/reset | 50 | -6.323399 | 0.060089 | +0.288599 | +0.061102 | 98.00% | 93.00% | 89.33% | 82.00% |
| Novelty/reset | 75 | -6.323125 | 0.059744 | +0.288873 | +0.057943 | 97.78% | 92.89% | 90.11% | 82.78% |
| Novelty/reset | 100 | -6.316746 | 0.052301 | +0.295252 | +0.050808 | 98.00% | 92.42% | 89.92% | 82.00% |
| Aggregate/persistent | 25 | -6.384156 | 0.012634 | +0.227842 | +0.016880 | 98.33% | 92.33% | 86.00% | 79.00% |
| Aggregate/persistent | 50 | -6.347446 | 0.040397 | +0.264552 | +0.037055 | 98.33% | 92.83% | 86.50% | 78.67% |
| Aggregate/persistent | 75 | -6.315485 | 0.035131 | +0.296513 | +0.065583 | 98.44% | 93.33% | 85.11% | 76.11% |
| Aggregate/persistent | 100 | -6.290384 | 0.026677 | +0.321614 | +0.077170 | 98.58% | 93.17% | 86.00% | 76.58% |
| Novelty/persistent | 25 | -6.441945 | 0.167357 | +0.170053 | -0.040908 | 95.67% | 88.33% | 84.67% | 81.00% |
| Novelty/persistent | 50 | -6.386052 | 0.088226 | +0.225946 | -0.001552 | 97.67% | 91.33% | 88.67% | 81.00% |
| Novelty/persistent | 75 | -6.349659 | 0.058622 | +0.262339 | +0.031409 | 98.00% | 91.56% | 89.11% | 80.44% |
| Novelty/persistent | 100 | -6.349659 | 0.058622 | +0.262339 | +0.017895 | 98.17% | 92.58% | 90.33% | 81.58% |

Aggregate/persistent 的相对差从 25 代 +0.0169 增至 100 代 +0.0772；novelty/reset 在四个观测点均优于 LR=0；validity-only 合法率提高但 best 没有稳定优势；shuffled 合法率较高而 best 更差。由于 n=3，这些只能生成第三轮假设。

### 4.4 相对 LR=0 的 paired statistics

| 变体 | paired mean | paired SD | 95% CI | dz | t p | Wilcoxon p | 胜/负 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Validity-only/reset | +0.002039 | 0.099355 | [-0.244773, 0.248851] | 0.021 | 0.9749 | 1.000 | 2/1 |
| Shuffled/reset | -0.025708 | 0.053012 | [-0.157396, 0.105981] | -0.485 | 0.4893 | 0.750 | 1/2 |
| Novelty/reset | +0.050808 | 0.080802 | [-0.149916, 0.251532] | 0.629 | 0.3899 | 0.250 | 3/0 |
| Aggregate/persistent | +0.077170 | 0.043597 | [-0.031132, 0.185472] | 1.770 | 0.0920 | 0.250 | 3/0 |
| Novelty/persistent | +0.017895 | 0.046210 | [-0.096897, 0.132687] | 0.387 | 0.5715 | 0.750 | 2/1 |

Aggregate/persistent 的 standardized paired effect 较大，但 95% CI 仍跨 0。n=3 下显著性检验的 power 极低，正确表述是“优先进入扩展验证”，不是“persistent 已有效”。

### 4.5 GRPO、reward 与候选漏斗诊断

| 变体 | loss | KL | grad norm | reward std | zero-std | length | ≥2 eligible/group | idea | duplicate | tie | parent win | frontier win | registered | survived |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LR=0 | 0.0032 | 0.1363 | 0.3185 | 0.2632 | 0.00% | 323.1 | 99.67% | 1.67% | 20.92% | 4.08% | 1.00% | 0.75% | 62.83% | 9.58% |
| Validity/reset | 0.0118 | 0.0436 | 0.1180 | 0.3511 | 59.67% | 355.2 | 98.67% | 0.58% | 4.92% | 8.00% | 1.25% | 0.75% | 82.67% | 9.75% |
| Shuffled/reset | 0.0107 | 0.1102 | 0.2510 | 0.2207 | 12.33% | 341.9 | 98.67% | 2.17% | 9.25% | 4.42% | 1.08% | 0.67% | 80.67% | 8.83% |
| Novelty/reset | 0.0093 | 0.1217 | 0.2755 | 0.2270 | 0.00% | 359.4 | 99.33% | 0.25% | 7.92% | 5.25% | 2.00% | 1.42% | 82.42% | 12.25% |
| Aggregate/persistent | 0.0115 | 0.1530 | 0.2876 | 0.2280 | 1.67% | 337.5 | 98.00% | 10.92% | 9.42% | 15.92% | 2.50% | 2.00% | 76.83% | 18.00% |
| Novelty/persistent | 0.0149 | 0.1267 | 0.2982 | 0.2095 | 1.67% | 302.7 | 99.00% | 0.67% | 8.75% | 6.83% | 1.00% | 0.67% | 81.83% | 10.42% |

Validity-only 的 59.67% zero-std group 是其数学结构的直接后果：当 4 个 response 全部 eligible 时均得到 +1，经 group normalization 后不存在相对梯度。它的 adapter 仍发生变化，是因为包含合法/非法混合 response 的 group 提供协议梯度。这恰好说明 validity-only 可以训练协议，却不能证明 heuristic-quality learning。

### 4.6 optimizer 与 adapter 真实性

| 变体 | optimizer steps | reinitialized | state reused | adapter start | adapter end | net norm delta | norm changed updates |
|---|---:|---:|---:|---:|---:|---:|---:|
| LR=0/reset | 300 | 300 | 0 | 52.999493 | 52.999493 | 0 | 0 |
| Validity/reset | 300 | 300 | 0 | 52.999493 | 52.932563 | -0.066930 | 300 |
| Shuffled/reset | 300 | 300 | 0 | 52.999493 | 53.062110 | +0.062617 | 300 |
| Novelty/reset | 300 | 300 | 0 | 52.999493 | 53.088492 | +0.088999 | 299 |
| Aggregate/persistent | 300 | 3 | 297 | 52.999493 | 53.109085 | +0.109592 | 300 |
| Novelty/persistent | 300 | 3 | 297 | 52.999493 | 53.085553 | +0.086061 | 300 |

该表确认：

1. LR=0 经过完整 Trainer 路径和 300 次 optimizer step，adapter norm 仍严格不变；
2. reset 的 300 个 update 均重建 optimizer；
3. persistent 的每个 seed 只初始化一次，后续 99 次复用状态；
4. 非零 LR 变体几乎每个 update 都有可检测参数变化。

因此，“GRPO 开关只是日志开关”“persistent 配置没有落到实际 optimizer”两类虚假实现风险已被排除。

### 4.7 第二轮可支持与不可支持的归因

可支持：

- 参数实际变化是当前训练效应的必要条件；LR=0 不产生 adapter drift。
- shuffled 在高合法率下仍表现较差，说明合法率并非充分条件。
- novelty penalty 明显降低 duplicate，并提高 archive novel/registered。
- persistent 是值得进入大样本扩展的候选。

不可支持：

- 不能宣称 performance credit 已被证实，因为没有同轮 aggregate/reset 的严格对照；
- 不能把 persistent 优势单独归因于 optimizer state，因为第三轮又同时改变了 LR；
- 不能进行 internalization claim，因为没有 fixed prompt、same-valid、repair、search-free 或 transfer。

---

## 5. 第三轮：6-seed、500 代扩展验证

### 5.1 设计与 canonical run 选择

第三轮将第二轮候选扩展到 6 seeds、500 updates，并补跑 contemporaneous B0。

| 变体 | GRPO | reward | LR | optimizer | 目的 |
|---|---|---|---:|---|---|
| b0_fixed | 关闭 | 仅记录评价 | — | 无 | 同期纯 rollout/search 基线 |
| aggregate_reset | 开启 | aggregate | 5e-5 | reset | 复验 vanilla A2-like |
| aggregate_persistent_lr1e5 | 开启 | aggregate | 1e-5 | persistent | 低 LR 持续优化候选 |
| novelty_reset | 开启 | aggregate+novelty | 5e-5 | reset | 多样性退化诊断 |

每个 run 严格包含 500 updates × 4 responses=2000 responses。4 个变体 × 6 seeds 共 12,000 updates 和 48,000 responses。

需要明确一个设计混淆：persistent 变体同时改变了 optimizer persistence 和 LR，故第三轮不能将其结果单独归因于 persistent state。第二轮 persistent 使用 5e-5，第三轮使用 1e-5，是一个新的联合配置。

第三轮 canonical 来源如下：

| 变体 | 主体批次使用 seed | recovery 使用 seed | B0 批次 |
|---|---|---|---|
| Aggregate/reset | 42–46 | 47 | — |
| Persistent/LR1e-5 | 42–46 | 47 | — |
| Novelty/reset | 42–45 | 46–47 | — |
| B0 | — | — | 42–47 |

主批次中未完成的目录不纳入分析；统计中每个 variant/seed 只有一个 completed run。

### 5.2 逐 seed 最终分数

| Seed | B0 | Aggregate/reset | Persistent/LR1e-5 | Novelty/reset |
|---:|---:|---:|---:|---:|
| 42 | -6.240935 | -6.263852 | -6.248457 | -6.246769 |
| 43 | -6.230648 | -6.317348 | -6.223389 | -6.251225 |
| 44 | -6.276638 | -6.335614 | -6.253002 | -6.247296 |
| 45 | -6.258949 | -6.259178 | -6.223162 | -6.261933 |
| 46 | -6.321269 | -6.273290 | -6.220255 | -6.232881 |
| 47 | -6.249179 | -6.271911 | -6.252432 | -6.269408 |
| mean | -6.262936 | -6.286865 | -6.236783 | -6.251585 |
| SD | 0.032626 | 0.031654 | 0.016015 | 0.012777 |

### 5.3 相对 B0 的配对推断

| 变体 | seed42–47 paired differences | mean | paired SD | 95% CI | dz | t p | Wilcoxon p | 胜/负 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Aggregate/reset | -0.022917, -0.086700, -0.058977, -0.000229, +0.047979, -0.022732 | -0.023929 | 0.046698 | [-0.072936, 0.025077] | -0.512 | 0.2649 | 0.2188 | 1/5 |
| Persistent/LR1e-5 | -0.007521, +0.007259, +0.023635, +0.035787, +0.101014, -0.003253 | +0.026153 | 0.040147 | [-0.015978, 0.068285] | 0.651 | 0.1714 | 0.2188 | 4/2 |
| Novelty/reset | -0.005834, -0.020577, +0.029341, -0.002984, +0.088388, -0.020228 | +0.011351 | 0.041895 | [-0.032615, 0.055317] | 0.271 | 0.5363 | 1.0000 | 2/4 |

没有任何 GRPO 变体达到传统统计显著标准。Persistent/LR1e-5 的均值、方差和胜率在三个训练变体中最好，但其 95% CI 同时包含轻微负效应和中等正效应，正确结论是“保留为机制研究候选”，而不是“显著优于 B0”。Aggregate/reset 在 5/6 seeds 上不优于 B0，是对 vanilla aggregate GRPO 的重复负面证据。

### 5.4 逐 50/100 代 best 与漏斗中间值

| 变体 | 代数 | best mean | SD | 较 initial 提升 | 相对 B0 | contract | execution | eligible | archive novel |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 | 50 | -6.332287 | 0.055939 | +0.303370 | — | 94.58% | 87.42% | 80.33% | 79.67% |
| B0 | 100 | -6.298574 | 0.037051 | +0.337083 | — | 95.21% | 87.42% | 79.96% | 79.29% |
| B0 | 200 | -6.296411 | 0.036832 | +0.339247 | — | 94.81% | 85.92% | 78.81% | 77.48% |
| B0 | 300 | -6.289869 | 0.030806 | +0.345789 | — | 94.47% | 85.43% | 78.29% | 76.78% |
| B0 | 400 | -6.270689 | 0.027128 | +0.364969 | — | 94.32% | 85.16% | 78.02% | 76.48% |
| B0 | 500 | -6.262936 | 0.032626 | +0.372721 | — | 94.22% | 84.99% | 77.68% | 75.87% |
| Aggregate/reset | 50 | -6.361979 | 0.053682 | +0.273678 | -0.029691 | 98.42% | 93.50% | 88.92% | 76.92% |
| Aggregate/reset | 100 | -6.338857 | 0.053807 | +0.296800 | -0.040283 | 98.88% | 94.79% | 90.33% | 71.46% |
| Aggregate/reset | 200 | -6.317658 | 0.046763 | +0.318000 | -0.021247 | 98.88% | 94.85% | 90.46% | 67.69% |
| Aggregate/reset | 300 | -6.313547 | 0.044885 | +0.322110 | -0.023679 | 98.88% | 95.22% | 91.04% | 63.51% |
| Aggregate/reset | 400 | -6.305995 | 0.035325 | +0.329662 | -0.035307 | 98.93% | 95.54% | 91.46% | 59.61% |
| Aggregate/reset | 500 | -6.286865 | 0.031654 | +0.348792 | -0.023929 | 98.98% | 95.75% | 91.59% | 57.79% |
| Persistent/LR1e-5 | 50 | -6.320164 | 0.074450 | +0.315493 | +0.012124 | 98.08% | 93.17% | 88.75% | 77.25% |
| Persistent/LR1e-5 | 100 | -6.290242 | 0.046412 | +0.345416 | +0.008333 | 98.58% | 94.25% | 90.12% | 68.46% |
| Persistent/LR1e-5 | 200 | -6.255581 | 0.020320 | +0.380077 | +0.040830 | 98.96% | 95.71% | 90.46% | 59.62% |
| Persistent/LR1e-5 | 300 | -6.244954 | 0.015637 | +0.390704 | +0.044915 | 99.18% | 96.50% | 91.06% | 54.37% |
| Persistent/LR1e-5 | 400 | -6.239272 | 0.016224 | +0.396385 | +0.031417 | 99.30% | 96.89% | 91.77% | 51.20% |
| Persistent/LR1e-5 | 500 | -6.236783 | 0.016015 | +0.398874 | +0.026153 | 99.33% | 97.08% | 92.08% | 48.59% |
| Novelty/reset | 50 | -6.357490 | 0.052596 | +0.278167 | -0.025203 | 97.25% | 91.58% | 88.92% | 84.25% |
| Novelty/reset | 100 | -6.318004 | 0.045267 | +0.317653 | -0.019429 | 97.62% | 91.21% | 89.21% | 84.13% |
| Novelty/reset | 200 | -6.262593 | 0.012044 | +0.373065 | +0.033818 | 98.23% | 92.08% | 89.98% | 83.56% |
| Novelty/reset | 300 | -6.260101 | 0.010627 | +0.375557 | +0.029768 | 98.50% | 93.13% | 91.14% | 83.04% |
| Novelty/reset | 400 | -6.251853 | 0.012572 | +0.383805 | +0.018836 | 98.50% | 93.30% | 91.54% | 83.42% |
| Novelty/reset | 500 | -6.251585 | 0.012777 | +0.384072 | +0.011351 | 98.53% | 93.69% | 92.11% | 83.73% |

Persistent/LR1e-5 在 200 代达到 +0.0408、300 代达到 +0.0449，之后回落至 +0.0262。Novelty/reset 在 200 代转正，300 代达到 +0.0298，500 代缩小至 +0.0114。Aggregate/reset 在所有观测点均落后。终点分数会掩盖阶段依赖性，后续主实验必须同时报告 best-curve AUC、queries-to-target 和 valid-evaluations-to-target。

### 5.5 第三轮全程训练与候选诊断

| 指标 | B0 | Aggregate/reset | Persistent/LR1e-5 | Novelty/reset |
|---|---:|---:|---:|---:|
| contract | 94.22% | 98.98% | 99.33% | 98.53% |
| execution | 84.99% | 95.75% | 97.08% | 93.69% |
| population eligible | 77.68% | 91.59% | 92.08% | 92.11% |
| archive novel | 75.87% | 57.79% | 48.59% | 83.73% |
| archive duplicate | 1.81% | 33.80% | 43.49% | 8.38% |
| parent tie | 4.20% | 39.62% | 32.89% | 4.15% |
| parent win/all | 0.750% | 0.575% | 0.575% | 0.742% |
| frontier win/all | 0.500% | 0.433% | 0.450% | 0.617% |
| registered | 76.43% | 60.25% | 48.95% | 84.33% |
| survived | 4.09% | 4.61% | 4.65% | 5.15% |
| zero-reward-std | 0.47% | 13.13% | 6.27% | 0.23% |
| idea rate | 3.51% | 0.31% | 0.16% | 0.38% |
| generation mean/run | 1528.7 | 1205.0 | 979.0 | 1686.5 |
| loss | — | 0.0060 | 0.0028 | 0.0031 |
| KL | — | 0.0823 | 0.0955 | 0.0927 |
| grad norm | — | 0.2040 | 0.2503 | 0.2182 |
| reward std | — | 0.1438 | 0.1455 | 0.1756 |
| completion length | — | 435.5 | 377.3 | 472.4 |

三个训练变体都将 eligible 推至约 92%，但 archive 行为完全不同。Aggregate/reset 每 100 个 response 中约 34 个是 archive duplicate，persistent 约 43 个；novelty 仅约 8 个。Persistent 的 generation 只有 979，显著低于 B0 的 1529 和 novelty 的 1687，说明大量 completion 没有转化为新的 population 演化事件。

idea rate 从 B0 的 3.51% 降至训练变体的 0.16%–0.38%，而 contract 反而升至约 99%。这不是直接运行错误，因为当前 contract 只要求代码解析和函数签名；但它说明日志中的 contract success 不能等价于完整遵循“idea+code”输出协议。

### 5.6 每 100 代训练动力学

| 变体 | 窗口 | loss | KL | grad | reward std | zero-std | length | duplicate | tie | parent win | frontier win | registered | survived |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Aggregate/reset | 1–100 | 0.0196 | 0.1249 | 0.3000 | 0.1905 | 2.50% | 302.8 | 18.88% | 13.50% | 1.46% | 1.04% | 71.88% | 12.25% |
| Aggregate/reset | 101–200 | 0.0093 | 0.1010 | 0.2229 | 0.1467 | 10.33% | 420.5 | 26.67% | 33.79% | 0.63% | 0.42% | 67.58% | 5.29% |
| Aggregate/reset | 201–300 | -0.0009 | 0.0688 | 0.1793 | 0.1281 | 15.33% | 476.0 | 37.04% | 38.92% | 0.17% | 0.17% | 58.00% | 1.96% |
| Aggregate/reset | 301–400 | 0.0005 | 0.0601 | 0.1618 | 0.1265 | 18.17% | 477.9 | 44.79% | 53.29% | 0.21% | 0.21% | 50.96% | 1.25% |
| Aggregate/reset | 401–500 | 0.0014 | 0.0568 | 0.1562 | 0.1273 | 19.33% | 500.1 | 41.63% | 58.63% | 0.42% | 0.33% | 52.83% | 2.29% |
| Persistent/LR1e-5 | 1–100 | 0.0045 | 0.1120 | 0.2965 | 0.2123 | 0.00% | 322.3 | 21.67% | 9.88% | 1.50% | 1.17% | 68.58% | 12.08% |
| Persistent/LR1e-5 | 101–200 | -0.0023 | 0.0980 | 0.2598 | 0.1629 | 1.50% | 361.3 | 40.00% | 23.25% | 0.83% | 0.63% | 50.96% | 4.88% |
| Persistent/LR1e-5 | 201–300 | -0.0044 | 0.0915 | 0.2412 | 0.1328 | 6.00% | 389.6 | 48.38% | 35.38% | 0.29% | 0.21% | 44.29% | 2.92% |
| Persistent/LR1e-5 | 301–400 | 0.0079 | 0.0942 | 0.2347 | 0.1131 | 9.33% | 393.8 | 52.25% | 45.54% | 0.17% | 0.17% | 42.13% | 2.13% |
| Persistent/LR1e-5 | 401–500 | 0.0080 | 0.0816 | 0.2193 | 0.1063 | 14.50% | 419.7 | 55.17% | 50.42% | 0.08% | 0.08% | 38.79% | 1.25% |
| Novelty/reset | 1–100 | 0.0075 | 0.1194 | 0.2825 | 0.2283 | 0.67% | 356.3 | 5.08% | 4.33% | 1.88% | 1.54% | 84.25% | 11.92% |
| Novelty/reset | 101–200 | 0.0061 | 0.0894 | 0.2328 | 0.1937 | 0.17% | 457.3 | 7.75% | 4.88% | 1.12% | 0.96% | 83.37% | 7.00% |
| Novelty/reset | 201–300 | -0.0039 | 0.0898 | 0.2043 | 0.1577 | 0.00% | 503.8 | 11.46% | 3.58% | 0.21% | 0.13% | 82.67% | 2.29% |
| Novelty/reset | 301–400 | 0.0058 | 0.0847 | 0.1923 | 0.1592 | 0.00% | 513.3 | 8.21% | 3.87% | 0.46% | 0.42% | 85.29% | 3.04% |
| Novelty/reset | 401–500 | -0.0002 | 0.0804 | 0.1788 | 0.1390 | 0.33% | 531.1 | 9.42% | 4.08% | 0.04% | 0.04% | 86.04% | 1.50% |

动力学表显示了 aggregate 系列的明确退化轨迹：

1. Aggregate/reset 的 duplicate 从 18.88% 增至 41.63%，tie 从 13.50% 增至 58.63%，zero-std 从 2.50% 增至 19.33%。
2. Persistent 的 duplicate 从 21.67% 增至 55.17%，registered 从 68.58% 降至 38.79%，parent win 从 1.50% 降至 0.08%。
3. Novelty 的 duplicate 始终约 5%–11%，tie 始终约 4%，registered 稳定在 82%–86%，证明 novelty penalty 确实改变了 candidate distribution。
4. 三个变体的 parent/frontier win 均在最后窗口接近 0，说明搜索后期 frontier 变难是共同因素；但 aggregate 的重复和 tie 恶化远强于 novelty，不能完全归因于搜索自然饱和。
5. completion length 持续增长，Aggregate 从 302.8 增至 500.1，Novelty 从 356.3 增至 531.1。更长输出没有转化为更高 parent-win，提示 token 成本与质量脱钩。

---

### 5.7 第三轮 optimizer 与 adapter 审计

| 变体 | optimizer steps | reinitialized | state reused | adapter start | adapter end | net delta | norm changed |
|---|---:|---:|---:|---:|---:|---:|---:|
| B0 | 0 | 0 | 0 | 52.999493 | 52.999493 | 0 | 0 |
| Aggregate/reset | 3000 | 3000 | 0 | 52.999493 | 53.456481 | +0.456989 | 3000 |
| Persistent/LR1e-5 | 3000 | 6 | 2994 | 52.999493 | 53.030598 | +0.031106 | 2985 |
| Novelty/reset | 3000 | 3000 | 0 | 52.999493 | 53.452822 | +0.453329 | 2995 |

这些计数与 6 seeds × 500 updates 完全一致。Reset 的每一步均重新初始化 optimizer；persistent 每个 seed 只初始化 1 次，之后共复用 2994 次。Persistent 的 adapter norm 净变化更小，既可能来自较低 LR，也可能来自持续动量和不同 trajectory，当前设计不能拆分二者。

Adapter norm 是“参数确实变化”的审计量，不是模型质量量。Aggregate/reset 和 novelty/reset 具有接近的 norm 增量，但最终行为和 archive duplicate 完全不同，说明 norm 大小不能替代 reward/behavior 分析。

### 5.8 按 EoH operator 分解

| 变体 | operator | contract | execution | eligible | archive novel | parent win | frontier win | registered | survived |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 | e1 | 95.53% | 85.83% | 80.80% | 79.57% | 0.33% | 0.33% | 79.90% | 2.87% |
| B0 | e2 | 95.10% | 86.33% | 82.80% | 81.27% | 0.37% | 0.37% | 82.03% | 2.03% |
| B0 | m1 | 95.07% | 84.27% | 75.03% | 73.13% | 1.40% | 0.80% | 73.70% | 6.97% |
| B0 | m2 | 91.17% | 83.53% | 72.07% | 69.50% | 0.90% | 0.50% | 70.10% | 4.50% |
| Aggregate/reset | e1 | 99.50% | 95.87% | 93.20% | 60.67% | 0.63% | 0.63% | 63.00% | 4.77% |
| Aggregate/reset | e2 | 99.33% | 96.80% | 93.70% | 56.77% | 0.47% | 0.47% | 58.77% | 3.60% |
| Aggregate/reset | m1 | 98.93% | 94.90% | 89.40% | 59.33% | 0.73% | 0.43% | 61.47% | 6.93% |
| Aggregate/reset | m2 | 98.13% | 95.43% | 90.07% | 54.40% | 0.47% | 0.20% | 57.77% | 3.13% |
| Persistent/LR1e-5 | e1 | 99.63% | 96.77% | 92.47% | 49.70% | 0.53% | 0.53% | 50.00% | 4.47% |
| Persistent/LR1e-5 | e2 | 99.50% | 97.47% | 93.97% | 48.80% | 0.30% | 0.30% | 49.20% | 3.23% |
| Persistent/LR1e-5 | m1 | 99.33% | 96.77% | 92.13% | 49.43% | 0.83% | 0.57% | 49.73% | 7.10% |
| Persistent/LR1e-5 | m2 | 98.87% | 97.30% | 89.77% | 46.43% | 0.63% | 0.40% | 46.87% | 3.80% |
| Novelty/reset | e1 | 99.13% | 94.53% | 93.20% | 85.60% | 0.87% | 0.87% | 86.13% | 5.23% |
| Novelty/reset | e2 | 99.07% | 94.67% | 93.33% | 84.23% | 0.57% | 0.57% | 84.80% | 4.53% |
| Novelty/reset | m1 | 98.47% | 93.37% | 91.37% | 81.97% | 0.93% | 0.70% | 82.67% | 6.87% |
| Novelty/reset | m2 | 97.47% | 92.20% | 90.53% | 83.10% | 0.60% | 0.33% | 83.70% | 3.97% |

operator 分解的主要结论：

- B0 中 m1 的 parent-win 和 survived 最高，说明 mutation operator 的条件潜力较强；
- 所有训练变体对四种 operator 的 contract/eligible 均有提升，不是单一 operator 驱动；
- novelty 在四种 operator 上均维持高 archive novel，说明 penalty 的行为效应具有 operator 一致性；
- 现有 operator 差异不足以支持 PDF 中更复杂的 scheduler。先实现 VC-PAIR 和受控评测，比增加调度器更符合证据优先级。

### 5.9 第三轮结论

第三轮同时给出一个弱正面结果和一个强机制警告。

弱正面结果是 Persistent/LR1e-5 在 4/6 seeds 上优于 B0，paired mean 为 +0.0262，且 200–500 代保持均值领先。它值得保留为工程候选。

强机制警告是：该变体的 archive novel 从前 100 代的约 68.46% 降到全程 48.59%，最后窗口 duplicate 达 55.17%、parent win 仅 0.08%。这意味着平均 best 优势并不能证明它持续生成更优候选；优势可能来自早期少数成功、trajectory path dependence 或更高合法候选供给，后期训练则明显退化。

---

## 6. 三轮实验的纵向对比

### 6.1 端到端结果总览

| 轮次 | 对照 | 主要候选 | 预算 | seeds | paired mean | 95% CI | 胜/负 | 结论 |
|---|---|---|---:|---:|---:|---:|---:|---|
| R1 | B0 | Aggregate/reset 5e-5 | 500×4 | 6 | -0.001568 | [-0.062364, 0.059228] | 3/3 | 无总体优势 |
| R2 | LR=0 | Validity/reset | 100×4 | 3 | +0.002039 | [-0.244773, 0.248851] | 2/1 | 与 control 近似 |
| R2 | LR=0 | Shuffled/reset | 100×4 | 3 | -0.025708 | [-0.157396, 0.105981] | 1/2 | 方向为负 |
| R2 | LR=0 | Novelty/reset | 100×4 | 3 | +0.050808 | [-0.149916, 0.251532] | 3/0 | 候选信号 |
| R2 | LR=0 | Aggregate/persistent 5e-5 | 100×4 | 3 | +0.077170 | [-0.031132, 0.185472] | 3/0 | 最强 pilot 信号 |
| R3 | B0 | Aggregate/reset 5e-5 | 500×4 | 6 | -0.023929 | [-0.072936, 0.025077] | 1/5 | 重复负面证据 |
| R3 | B0 | Persistent 1e-5 | 500×4 | 6 | +0.026153 | [-0.015978, 0.068285] | 4/2 | 弱正面、未显著 |
| R3 | B0 | Novelty/reset 5e-5 | 500×4 | 6 | +0.011351 | [-0.032615, 0.055317] | 2/4 | 均值小、未显著 |

### 6.2 第一轮与第三轮复验差异

相同 seed 的 B0 在两轮间并非 bitwise deterministic：

| Seed | R1 B0 | R3 B0 | R3-R1 |
|---:|---:|---:|---:|
| 42 | -6.230170 | -6.240935 | -0.010766 |
| 43 | -6.240269 | -6.230648 | +0.009621 |
| 44 | -6.294642 | -6.276638 | +0.018004 |
| 45 | -6.266577 | -6.258949 | +0.007628 |
| 46 | -6.283799 | -6.321269 | -0.037470 |
| 47 | -6.211750 | -6.249179 | -0.037429 |
| mean | -6.254535 | -6.262936 | -0.008401 |

Aggregate/reset 从 R1 的 -6.256103 变为 R3 的 -6.286865，跨轮下降 -0.030762；同期 B0 跨轮下降 -0.008401，因此相对效应又下降约 -0.02236。该变化说明固定 seed 和 initial population 尚不能保证完全复现采样轨迹。可能来源包括 vLLM sampling 非确定性、并发执行、代码版本或日志修复后的运行路径差异。

这不是直接证明代码有 bug，但要求主论文实验：

- 固定 commit hash、环境版本和 model/adapter hash；
- 保存完整 resolved config；
- 记录 CUDA/vLLM/transformers/TRL 版本；
- 将同一比较组安排在同一代码版本和近似时间窗；
- 使用 paired seeds 并报告跨重复批次的方差。

### 6.3 三轮工程成熟度变化

| 能力 | R1 | R2 | R3 |
|---|---|---|---|
| GRPO on/off 可辨识 | 基本具备 | 已验证 | 已验证 |
| 冻结 initial population | 是 | 是 | 是 |
| optimizer reset/persistent 日志 | 不完整 | 完整 | 完整 |
| adapter norm 审计 | 不完整 | 完整 | 完整 |
| LR=0 negative control | 无 | 有 | 无 |
| validity/shuffled control | 无 | pilot | 未扩展 |
| archive duplicate/novel | 不完整 | 完整 | 完整 |
| 6-seed 500 代 | 是 | 否 | 是 |
| recovery 与 canonical 选择 | 不需要 | 不需要 | 已执行 |
| search-free/fixed bank | 无 | 无 | 无 |
| VC-PAIR | 无 | 无 | 无 |

因此“第二阶段”不是指工程仍不可靠，而是指**科学识别仍停留在 reward/optimizer pilot 和 end-to-end 扩展**；关键受控因果比较尚未开始。

---

## 7. Validity-mediated 漏斗分析

### 7.1 第三轮条件概率分解

根据第三轮全程计数，可计算：

| 变体 | \(P(C)\) | \(P(V\mid C)\) | \(P(eligible)\) | \(P(parent\ win\mid eligible)\) | 正向 parent improvement 均值 | \(P(frontier\ win\mid eligible)\) |
|---|---:|---:|---:|---:|---:|---:|
| B0 | 94.22% | 90.20% | 77.68% | 0.966% | 0.00756 | 0.644% |
| Aggregate/reset | 98.98% | 96.74% | 91.59% | 0.628% | 0.00672 | 0.473% |
| Persistent/LR1e-5 | 99.33% | 97.72% | 92.08% | 0.624% | 0.00925 | 0.489% |
| Novelty/reset | 98.53% | 95.09% | 92.11% | 0.805% | 0.00686 | 0.669% |

注：\(P(V\mid C)\) 以 execution/completion 除以 contract/completion 得到；parent/frontier conditional rate 以 all-response win rate 除以 eligible rate得到。正向 improvement 均值只对 parent-win candidate 计算。

### 7.2 解释

Aggregate/reset 相比 B0：

- \(P(C)\) 增加约 4.76 个百分点；
- \(P(V\mid C)\) 增加约 6.54 个百分点；
- \(P(eligible)\) 增加约 13.91 个百分点；
- 但 \(P(parent\ win\mid eligible)\) 从 0.966% 降至 0.628%；
- \(P(frontier\ win\mid eligible)\) 从 0.644% 降至 0.473%。

Persistent 的合法性漏斗更强，但 conditional parent-win 同样只有 0.624%。它的正向 improvement 均值 0.00925 高于 B0，说明“成功次数更少、成功时幅度略大”，但样本事件极稀疏且来自不同动态 trajectory，不能作为 search-free 质量证据。

Novelty 的 \(P(frontier\ win\mid eligible)\) 为 0.669%，接近并略高于 B0，同时保持最高 archive novel。它说明多样性 penalty 能改变候选供给结构，但终点均值仍不足以证明稳定搜索收益。

### 7.3 与 PDF 核心假设的关系

现有结果强力支持 PDF 要求首先检验的 validity-mediated hypothesis：

1. 在线 GRPO 稳定提高 \(P(C)\) 和 \(P(V\mid C)\)；
2. 这种提高产生更多 eligible candidate；
3. 但 aggregate 没有提高 \(P(\Delta>0\mid C,V)\)，反而下降；
4. 动态 search 仍可能因更多合法抽样而偶尔获得更好的 best；
5. 因而 end-to-end best 的弱正效应不足以证明 conditional quality internalization。

更严格的结论仍需 fixed prompt/parent bank，因为上述条件概率来自不同 on-policy trajectories。

---

## 8. 资源成本与实验效率

### 8.1 三轮成本汇总

以下 GPU-hours 以每个 run 独占一张 GPU 的 wall time 求和；不是硬件功耗测量。日志目录体积为分析时观测值，早期主批次包含更完整的逐步 artifact，后续 recovery/B0 使用压缩策略，故目录体积不能跨轮直接视为方法固有成本。

| 轮次/变体 | responses | execution valid | evaluations | GPU-hours | train hours | eval-program hours | output tokens | 日志体积 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| R1 B0 | 12000 | 10354 | 11491 | 10.79 | 0 | 2.91 | 14.92M | 411.5 MiB |
| R1 Aggregate/reset | 12000 | 11556 | 11970 | 16.53 | 16.09 | 4.22 | 16.40M | 2250.3 MiB |
| R2 LR=0 | 1200 | 1080 | 1207 | 1.24 | 1.22 | 0.34 | 1.36M | 883.0 MiB |
| R2 Validity/reset | 1200 | 1093 | 1214 | 1.26 | 1.23 | 0.16 | 1.37M | 885.3 MiB |
| R2 Shuffled/reset | 1200 | 1095 | 1218 | 1.23 | 1.20 | 0.19 | 1.39M | 884.8 MiB |
| R2 Novelty/reset | 1200 | 1109 | 1211 | 1.29 | 1.26 | 0.24 | 1.41M | 891.6 MiB |
| R2 Aggregate/persistent | 1200 | 1118 | 1218 | 1.20 | 1.17 | 0.15 | 1.41M | 910.5 MiB |
| R2 Novelty/persistent | 1200 | 1111 | 1213 | 1.10 | 1.07 | 0.12 | 1.33M | 885.8 MiB |
| R3 B0 | 12000 | 10199 | 11376 | 9.98 | 0 | 1.95 | 14.34M | 107.0 MiB |
| R3 Aggregate/reset | 12000 | 11490 | 11947 | 15.27 | 14.75 | 3.25 | 15.53M | 1788.6 MiB |
| R3 Persistent/LR1e-5 | 12000 | 11649 | 11990 | 14.18 | 13.71 | 4.53 | 15.13M | 1760.7 MiB |
| R3 Novelty/reset | 12000 | 11243 | 11894 | 16.65 | 16.01 | 2.55 | 16.15M | 1574.3 MiB |

### 8.2 成本解释

1. R3 训练变体相对 B0 增加约 4.2–6.7 GPU-hours/6 seeds，即每个 500 代 run 额外约 0.7–1.1 GPU-hours。
2. Aggregate/reset 相对 B0 多获得 1291 个 execution-valid response，却最终平均低 0.0239，说明“更多有效评估”并未自动产生更好终点。
3. Persistent 的执行合法数量最高，且平均 best 最好，但 eval-program 时间也最高；其优势必须用 queries/valid-evals-to-target 衡量，不能只按总 wall time。
4. GRPO 输出 token 比 B0 多约 0.8M–1.8M，和 completion length drift 一致。
5. 压缩后 recovery/B0 的单 run 日志可控制在约 20–25 MiB 量级，而早期保存逐步 adapter/checkpoint 使主批次达到 GB 量级。论文复现实验应保留标量、manifest、关键 checkpoint 和最终 adapter，不应保存每代完整 LoRA。

当前缺失的成本指标包括 peak GPU memory、energy、queries-to-target、valid-evaluations-to-target 和 best-curve AUC。这些应在第三阶段纳入统一日志，而不是事后从 console 猜测。

---

## 9. 与 PDF 方案的逐项对齐审计

### 9.1 最终实验矩阵覆盖率

| PDF 项目 | 当前状态 | 已有证据 | 缺口 |
|---|---|---|---|
| A0 Base+EoH | 部分完成 | B0 fixed，6 seeds×500 | 实际从 SFT LoRA 起步，不是 base/no-SFT |
| A1 repair/template | 未完成 | 无 | 无训练合法性增强 control |
| A2 aggregate GRPO | 已完成 | R1、R3 两轮 6-seed×500 | 不是 VC-PAIR；跨轮复现性有限 |
| A3 validity-only | pilot | R2 3-seed×100 | 未扩展至 6 seeds/500；无 no-SFT |
| A4 performance-shuffled | pilot | R2 group-local shuffle | 未与同期 aggregate/reset 大样本对照 |
| A5 VC-PAIR | 未完成 | 无 | 核心三态 paired reward 尚未实现 |
| A6 VC-PAIR+scheduler | 未完成且不优先 | operator 描述统计 | 尚无证据值得增加 scheduler |

### 9.2 Controlled internalization protocol 覆盖率

| PDF 要求 | 状态 | 对当前结论的影响 |
|---|---|---|
| 固定 prompt bank | 未实现 | 不同方法输入分布不一致 |
| 固定 parent bank | 仅 initial population 固定 | 第一代后 parent trajectory 分叉 |
| candidate-parent same-query | 动态搜索中近似有 | 不能跨方法对齐同一 query |
| same instance batch | 有 | parent-relative 分数可比较，但无 uncertainty threshold |
| same-valid evaluation | 未实现 | 合法率贡献与条件质量不能受控分离 |
| constrained-valid generation | 未实现 | 不知道无需 RL 的合法约束能否复现收益 |
| repair/template control | 未实现 | 无法排除协议修复替代训练 |
| search-free best-of-K | 未实现 | 无法识别模型独立 proposal quality |
| prompt/parent bank held-out | 未实现 | 无法排除 trajectory memory |
| parameter-only transfer | 未实现 | 无法证明更新进入参数而非搜索状态 |
| no-SFT fair core | 未实现 | SFT 和 GRPO 贡献混淆 |
| protocol-SFT secondary arm | 当前只有 SFT 起点 | 缺少 no-SFT 对照 |
| TSP+OBP 两主任务 | 仅 TSP50 | 不能证明跨任务族有效 |
| pilot 3 seeds | 已完成 | R2 符合候选筛选用途 |
| main ≥5 seeds | TSP R3 已完成 | 仍缺主方法与第二任务 |
| cluster-aware inference | 未完成 | candidate/parent/prompt 层依赖未建模 |
| cost accounting | 部分完成 | 缺 peak memory 与 target-based efficiency |

### 9.3 当前阶段判定

PDF 路线可按以下阶段理解：

1. **阶段一：工程真实性与日志口径。** GRPO 开关、baseline、数学配置、optimizer 语义、adapter 更新可审计；
2. **阶段二：reward/optimizer 诊断和端到端 pilot。** validity、shuffle、novelty、reset/persistent、6-seed 扩展；
3. **阶段三：受控机制识别。** repair、fixed bank、same-valid、constrained-valid、search-free；
4. **阶段四：VC-PAIR 方法验证。** paired uncertainty-aware reward、多任务、多 seed；
5. **阶段五：internalization 与 transfer。** held-out family、parameter-only transfer、消融和论文主结果。

阶段一已经完成；阶段二已经得到较充分的 TSP50 负面/混合结果；阶段三尚未开始。因此“目前仍停留在第二阶段”的判断正确。继续只跑更多 aggregate/novelty seeds 不会自动进入第三阶段。

---

## 10. 当前项目与实验存在的问题

### 10.1 科学识别问题

**问题 1：动态 trajectory 混淆。**  
不同方法在第一个 update 后选择不同 candidate，导致后续 parent、prompt、archive 和 operator context 分叉。当前 end-to-end 比较有效，但不能回答模型在共同输入下的条件质量差异。

**问题 2：缺失 A1 repair/template。**  
训练变体最稳定的收益是 contract/execution。若 deterministic repair 或 constrained decoding 在无训练条件下也能获得同等合法率和搜索性能，GRPO 的主要贡献将不再是方法创新。

**问题 3：没有 VC-PAIR。**  
当前 aggregate reward 使用固定阈值和分段尺度，没有利用成对实例差、标准误或置信阈值，无法区分真实改善与测量噪声，也没有 neutral 区域的严格统计语义。

**问题 4：所有实验从 SFT LoRA 起步。**  
PDF 推荐 no-SFT 为公平核心、protocol-SFT 为次级 arm。当前设计无法判断协议能力来自 SFT 还是 GRPO。

**问题 5：只有 TSP50。**  
单一 problem size 和单一任务无法证明 task-family proposal prior。OBP、跨 TSP size 或 held-out distribution 尚未运行。

**问题 6：第二轮 controls 样本过小且第三轮未同步扩展。**  
Validity 和 shuffled 只有 3 seeds×100，第三轮没有 contemporaneous validity/shuffled；因此 performance credit 的必要性仍未获得大样本识别。

### 10.2 方法与配置混淆

**问题 7：persistent 与 LR 同时变化。**  
R3 persistent 使用 1e-5，reset 使用 5e-5。它们的差异不能归因于 optimizer persistence。需要 2×2：reset/persistent × 1e-5/5e-5，或在进入 VC-PAIR 前固定唯一 optimizer 语义。

**问题 8：group size=4 限制 reward resolution。**  
每个 prompt 只有 4 responses。对于 parent-win 约 0.6%–1.0% 的稀疏事件，大多数 group 没有正向质量样本，训练主要由合法性差异、tie 和负向分数主导。

**问题 9：validity-only 天然产生 zero variance。**  
当四个 response 全部 eligible 时 reward 都为 +1；R2 59.67% 的 zero-std 不是实现 bug，而是目标函数不可辨识。若继续使用，需要引入 paired quality 或不同层级信号，而不是增加训练代数。

**问题 10：aggregate reward 诱发重复集中。**  
R3 aggregate/persistent 最后窗口 duplicate 55.17%、tie 50.42%、parent-win 0.08%。这说明 on-policy update 在搜索后期接近 reward collapse/behavior concentration。

### 10.3 指标和日志语义问题

**问题 11：contract 不等于完整 response 协议。**  
训练变体 contract 约 99%，idea rate 却低至 0.16%–0.38%。当前字段准确描述 parser/函数 contract，但若论文写成“格式完全遵循”会形成口径夸大。应明确拆分 idea_present、code_present、signature_valid、parse_success。

**问题 12：终点 best 不足。**  
Persistent 与 novelty 在 200–300 代领先更大，随后回落；只有 final best 会遗漏学习动态和资源效率。需增加 AUC、time-to-target、valid-evals-to-target。

**问题 13：缺少 reference policy 身份审计。**  
日志可审计 KL 数值，但尚不能仅凭结果文件证明 reference policy 究竟是 base、SFT initial adapter 还是每轮其他 snapshot。应记录 reference model/adapter hash 和 frozen 参数校验。

**问题 14：缺 peak memory 和完整环境指纹。**  
当前有 wall/train/eval/token，但缺 peak VRAM、依赖版本、commit hash、base model hash 和 CUDA/vLLM determinism 设置。

### 10.4 复现与运行风险

**问题 15：相同 seed 跨轮不完全复现。**  
R1 与 R3 B0 均值相差 -0.0084，个别 seed 相差约 -0.0375。对于当前 0.01–0.03 的候选效应，这种跨批次漂移不可忽略。

**问题 16：自动 recovery 的状态语义仍需专门测试。**  
本轮通过重新启动相同 variant/seed 补齐，而不是证明任意中断点能够完整恢复 model、optimizer、RNG、population 和 archive。尤其 persistent optimizer 的 mid-run resume 尚需 failure-injection test。

**问题 17：主批次 artifact 仍偏大。**  
压缩策略已经显著改善新批次，但早期逐 update 保存 adapter/checkpoint 会产生约 1.5–2.3 GiB/variant-batch。应只保留 final、milestone 和 failure checkpoint。

### 10.5 不是 bug 的现象

为避免错误修复，以下现象目前不应直接判为代码 bug：

- validity-only 的高 zero-std：由同组合法 reward 恒等导致；
- contract 高而 idea rate 低：由两者字段定义不同导致，但需要改进报告口径；
- persistent adapter norm 增量较小：同时受低 LR 影响；
- 500 代后 parent-win 极低：部分来自 frontier 饱和，但 aggregate 的重复恶化仍需治理；
- generation 小于 responses：candidate 未注册/重复/被 gate 拒绝不会推进有效演化。

---

## 11. 综合结论与证据等级

### 11.1 对研究问题的回答

| RQ | 当前回答 | 证据等级 |
|---|---|---|
| RQ1 工程真实性 | GRPO on/off、reset/persistent、adapter update 均已真实落地 | 强 |
| RQ2 端到端效果 | vanilla aggregate 无稳定收益；persistent/低 LR 有弱正信号 | 中 |
| RQ3 合法性放大 | GRPO 稳定提高 contract/execution/eligible | 强 |
| RQ4 条件质量 | aggregate 未提高 eligible 条件下 parent-win | 中，受 trajectory 混淆 |
| RQ5 性能 credit | shuffled pilot 方向较差，但尚未大样本严格识别 | 弱至中 |
| RQ6 分布退化 | aggregate/persistent duplicate、tie、zero-std 随代数显著恶化 | 强 |
| RQ7 内部化 | 不能支持，关键 controlled evaluation 全部缺失 | 强否定当前 claim，不是否定未来方法 |
| RQ8 阶段 | 阶段一完成，阶段二完成度较高，阶段三未开始 | 强 |

### 11.2 可写入阶段性研究记录的结论

1. 将 GRPO 作为训练组件嵌入 EoH 已经工程可行，并且训练参数确实发生更新。
2. 在当前 TSP50、SFT 起点、4 responses/update 配置下，aggregate GRPO 没有可重复的 final-best 增益。
3. GRPO 最稳定的效果是提高输出 contract、execution 和 population eligibility。
4. Aggregate reward 在训练后期产生明显的 archive duplicate、parent tie 和 zero-reward-variance 集中。
5. Novelty penalty 能有效缓解重复并提高注册率，但没有形成统计显著的 final-best 优势。
6. Persistent/LR=1e-5 是当前最好的工程候选，但其优势与 optimizer/LR 混淆，且后期候选分布明显退化。
7. 当前证据支持 validity-mediated search amplification 作为主要工作假设。

### 11.3 当前不能写入论文主张的内容

- “CALM/GRPO 显著优于 EoH baseline”；
- “模型已经内化 heuristic-design knowledge”；
- “persistent optimizer 是性能提升原因”；
- “novelty reward 提高最终优化质量”；
- “性能 credit assignment 已被 shuffled control 证实”；
- “方法可迁移到其他组合优化任务”；
- “当前实现等价于 PDF 推荐的 VC-PAIR”。

### 11.4 当前最合理的路线判断

若继续把主要预算投入 aggregate/reset、novelty 或更多 500 代重复，预期只能更精确地估计一个接近零的小效应，无法解决 internalization 识别问题。下一步必须从“更多动态搜索”转向“共同输入上的受控模型比较”。

---

## 12. 第三阶段最小实验方案

### 12.1 优先级一：冻结 query bank 的 search-free 诊断

建立固定 parent/prompt/operator bank，所有 policy 接收完全相同的输入。

建议最小设计：

| 项目 | 建议 |
|---|---|
| Parent bank | 从冻结 initial population 和独立 B0 trajectory 分层抽样 |
| Operator | e1/e2/m1/m2 平衡 |
| Parent quality | 按 score 分位数分层 |
| Query 数 | pilot 200–400；正式实验再做 power analysis |
| 每 query responses | 保持 K=4，并可增加 K=8 作为 tail sensitivity |
| Instance batch | candidate 与 direct parent 使用完全相同的 64 instances |
| Policies | B0、repair/template、aggregate、validity、shuffled、novelty、persistent |
| 指标 | contract、execution、same-valid parent-win、\(\Delta\) 分布、best-of-K、right-tail AUC |

该实验不执行 population selection，也不把新 candidate 反馈为下一轮 parent。它直接回答共同 query 下的 proposal quality。

### 12.2 优先级二：A1 repair/template 与 constrained-valid control

至少实现一个无训练合法性 control：

1. deterministic wrapper/template 保证函数签名；
2. parse failure 的最小 repair；
3. 禁止 metadata leak/randomness 的静态 gate；
4. 可行时使用 constrained generation。

比较 A0 与 A1 后，若 A1 已复现 GRPO 的 contract/execution 和 end-to-end 收益，则主机制更可能是合法候选供给，而不是 RL 方法创新。

### 12.3 优先级三：same-valid curve

对每个 policy：

1. 从固定 query bank 生成相同数量 raw responses；
2. 仅保留 eligible outputs；
3. 通过分层下采样或 bootstrap，使不同 policy 的 valid-output 数一致；
4. 比较同等 valid budget 下的 parent-win、frontier-win、best-of-K 和 \(\Delta\) tail。

若 GRPO 优势在 same-valid 后消失，则收益主要由 validity-mediated amplification 解释；若仍保留稳定质量优势，才有进入 VC-PAIR 主方法的依据。

### 12.4 优先级四：实现 VC-PAIR

VC-PAIR 的最小 reward 应满足：

\[
d_i=s_p(x_i)-s_c(x_i)
\]

其中 score 定义需按最小化/最大化统一方向，使 candidate 更好对应正 improvement。对相同实例批次的 paired differences 计算：

\[
\hat{\Delta}=\frac{1}{n}\sum_i d_i,\quad
SE(\hat{\Delta})=\frac{sd(d_i)}{\sqrt n}.
\]

用置信阈值 \(\tau=\lambda SE(\hat{\Delta})+\epsilon\) 定义：

- \(\hat{\Delta}>\tau\)：positive；
- \(|\hat{\Delta}|\le\tau\)：neutral；
- \(\hat{\Delta}<-\tau\)：negative；
- invalid：独立的分层 penalty。

正式实现时需统一当前代码的 score orientation，记录 paired instance vector，并避免先聚合成单一 score 后丢失不确定性。

### 12.5 优先级五：公平训练矩阵

在进入正式主结果前至少采用：

| Arm | 初始 policy | 在线训练 |
|---|---|---|
| No-SFT B0 | base/instruct | 无 |
| No-SFT A2 | base/instruct | aggregate |
| No-SFT A3 | base/instruct | validity-only |
| No-SFT A4 | base/instruct | shuffled |
| No-SFT A5 | base/instruct | VC-PAIR |
| Protocol-SFT B0 | SFT LoRA | 无 |
| Protocol-SFT A5 | SFT LoRA | VC-PAIR |

SFT 两个现有代码文件无需为当前诊断改动，但实验矩阵必须把 SFT 明确作为独立因素，而不是所有 arm 的隐含前置条件。

### 12.6 Go/No-Go 判据

**进入方法论文路线的 Go 条件：**

1. VC-PAIR 在 fixed bank 上提高 same-valid parent-win 和 \(\Delta\) 右尾；
2. VC-PAIR 优于 validity-only 和 group-local shuffled；
3. 优势不能被 repair/template 或 constrained-valid 复现；
4. search-free best-of-K 有一致提升；
5. 至少在 TSP 与 OBP 两个主任务、每任务 ≥5 seeds 上方向一致；
6. parameter-only transfer 在新 instance distribution 或 held-out problem size 上保留收益；
7. 成本增量可被 valid-evals-to-target 或 wall-time-to-target 的改善抵消。

**转向机制论文路线的 No-Go 条件：**

1. Aggregate/VC-PAIR 与 validity-only、shuffled、repair 在 same-valid 后相当；
2. fixed bank/search-free 下无条件质量收益；
3. only end-to-end dynamic search 有弱改善；
4. parameter-only transfer 消失；
5. duplicate/tie collapse 随训练重复出现。

No-Go 并不意味着实验失败。它可以形成一个清晰机制结论：在线 RL 的主要作用是提高合法 proposal supply，收益由外部搜索放大，而非学习通用 heuristic quality prior。

### 12.7 不建议立即执行的工作

- 不优先加入 operator scheduler；
- 不继续堆叠 rank/novelty/entropy 多项 reward；
- 不先扩展到更多 500 代 aggregate 重复；
- 不将 optimizer reuse、SFT、curriculum 同时加入主方法；
- 不在 fixed-bank 结果出来前进行大规模跨任务训练。

这些改动会增加因素数量，却不能解决当前最关键的因果识别缺口。

---

## 13. 统计与复现附录

### 13.1 聚合规则

- final best：读取 completed run 的最终 active-population best；
- trajectory best：对每个 seed 取指定 update 的 cumulative best，再跨 seed 计算均值与 sample SD；
- 比例指标：先跨 canonical run 汇总 numerator/denominator，再求比例；
- 训练指标：按有效 GRPO update 聚合；
- window 指标：按 1–100、101–200、201–300、301–400、401–500 独立汇总，不使用累计值；
- SD：sample standard deviation，分母 \(n-1\)；
- paired CI：\(\bar d\pm t_{0.975,n-1}s_d/\sqrt n\)；
- Cohen's \(d_z=\bar d/s_d\)；
- Wilcoxon：双侧 exact signed-rank，零差按实现规则处理。

### 13.2 日志完整性判据

第三轮 run 只有同时满足以下条件才纳入：

1. run_manifest status 为 completed；
2. online_update_count=500；
3. checkpoints/latest_finish.json 存在；
4. 每代 completion_count=4；
5. update 序号连续且无重复；
6. final best、reward、loss、KL 等数值为 finite；
7. variant、seed、reward mode、LR 和 optimizer mode 与 resolved config 一致。

### 13.3 主要数据文件

| 文件 | 用途 |
|---|---|
| run_manifest.json | run 状态、时间和完整性 |
| config_resolved.yaml / config.json | 实际配置而非 shell 默认值 |
| run_summary.json | 聚合成本和终点摘要 |
| checkpoints/latest_finish.json | completed 终点确认 |
| samples/samples_*.json | completion、reward、candidate 漏斗和逐步信息 |
| population/pop_*.json 或压缩 history | active population 与 best trajectory |
| run.log / run_log.txt | 运行事件和异常复核 |

### 13.4 已排除数据

- 2026-07-09 至 10 的早期 TSP50/TSP100/TSP200 exploratory runs：实验口径不同，不属于本报告三轮；
- optimizer/persistent smoke runs：只验证工程路径，不进入性能统计；
- 第三轮主体中没有 completed finish 的 aggregate seed47、persistent seed47、novelty seed46；
- 同一 variant/seed 的中断目录与 recovery 结果不重复计数；
- 初始化阶段只用于冻结 initial population，不与 500 代 final score 混为一个 treatment。

### 13.5 研究限制

1. 本报告只分析 TSP50，外部有效性有限；
2. n=6 只能排除较大效应，不能精确判断很小的真实收益；
3. 动态 trajectory 使 conditional quality 仍有选择偏差；
4. 未进行 hierarchical/cluster bootstrap；
5. 未获得 peak VRAM 和 energy；
6. 尚无 fixed-bank raw response 数据，无法绘制真正同分布的质量曲线；
7. PDF 是方案讨论文档，不是已经被当前代码完整复现的方法规范。

---

## 14. 最终阶段判断

三轮实验已经完成了从“GRPO 是否真实运行”到“哪些 reward/optimizer 候选值得继续”的工程与 pilot 工作，并通过第三轮 6-seed、500 代实验发现：

- vanilla aggregate GRPO 没有稳定收益；
- persistent/低 LR 只有弱正信号；
- novelty 解决重复但没有解决最终显著性；
- 最稳定的学习结果是合法候选率提高；
- aggregate 系列后期出现显著分布集中；
- conditional parent-win 没有随合法率同步提高。

因此，本项目当前应正式结束第二阶段，不再把“继续扩大相同动态搜索实验”视为进入下一阶段。下一项决定性工作是 fixed parent/prompt bank + repair/constrained-valid + same-valid + search-free 对照；其结果决定是实现 VC-PAIR 并进入方法路线，还是将 validity-mediated search amplification 发展为机制路线。
