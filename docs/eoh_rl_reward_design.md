# EoH-RL 奖励机制设计方案

## 1. 文档目标

本文档用于系统整理 `eoh_rl` 在 **本地模型 SFT + GRPO 后训练 + EoH 遗传进化搜索** 闭环中的奖励机制设计。重点是解决当前实验日志中已经真实发生的问题：

1. reward 大量零方差，导致 GRPO 后期缺少有效学习信号；
2. parent improvement 被压成固定 `0.0`，无法区分不同程度的局部改进；
3. valid non-improving 被压成固定 `-0.25`，导致有效但未突破的探索样本全部同质化；
4. CALM 的奖励机制依赖强 seed 和细粒度算子，不能直接迁移到 EoH-RL；
5. 当前 EoH-RL 的 reward 需要同时服务 **搜索进化** 和 **GRPO 后训练**，不能只满足其中一个。

本文档给出三版奖励方案：

| 方案 | 中文名称 | 核心思想 | 改动量 | 推荐阶段 |
|---|---|---|---:|---|
| 方案一 | 连续相对性能奖励方案 | 将离散状态奖励改为连续相对改进奖励 | 小 | 第一版可实现 |
| 方案二 | 连续性能与组内区分混合奖励方案 | 性能奖励 + 组内排序 + 小幅结构新颖性 | 中 | 最推荐 |
| 方案三 | 注册后搜索证据奖励方案 | 将是否进入种群/归档作为真实搜索证据 | 大 | 长期版 |

最终建议：**先修种群维护，再实现方案二的轻量版奖励。**

---

## 2. 当前 EoH-RL 奖励机制的问题

### 2.1 当前 four_state_v1 的基本逻辑

当前 `llm4ad/method/eoh_rl/rl/grpo_trainer.py` 中，`EoHGRPOReward._performance_reward()` 的实际逻辑可概括为：

```text
如果 candidate 超过当前 population frontier:
    reward = 1.0 + clip(relative_frontier_improvement)

否则如果 candidate 超过 parent:
    reward = 0.0

否则如果 candidate 有效但未改进:
    reward = -0.25

否则无代码、解析失败、执行失败、随机、泄漏、复制父代等:
    reward = 固定负值
```

这套机制的优点是简单，并且强调 frontier improvement。但它在 GRPO 场景下出现了严重问题。

### 2.2 问题一：parent improvement 被压成固定 0

如果某个 prompt 下 4 个 completion 都超过了各自 parent，但都没有超过全局 frontier，那么这组 reward 是：

```text
0.0, 0.0, 0.0, 0.0
```

这意味着：

1. 从 EoH 搜索角度看，这些 completion 都是有局部搜索价值的；
2. 从 GRPO 训练角度看，它们没有任何组内差异；
3. 模型无法学习哪个 parent improvement 更好；
4. `frac_reward_zero_std=1.0`，这一组几乎没有有效 advantage 信号。

### 2.3 问题二：valid non-improving 被压成固定负值

如果某个 prompt 下 4 个 completion 都能解析、能执行、非随机、非泄漏、非父代复制，但都没有超过 parent，那么这组 reward 是：

```text
-0.25, -0.25, -0.25, -0.25
```

这同样会导致 GRPO 无法区分不同 completion。

更重要的是，有效但没突破的样本在程序搜索中并不完全没有价值：

1. 它可能结构新颖，但性能暂时不足；
2. 它可能接近 parent，只差很小；
3. 它可能对后续 crossover/mutation 有价值；
4. 它可能在另一组实例或更大规模上表现更好。

但当前 reward 将所有这类样本压成同一个数，直接造成学习信号塌缩。

### 2.4 问题三：frontier improvement 太稀疏

当前实验日志显示，6 个 TSP50/TSP100 run 中，每个 run 只有极少数 frontier improvement，大多数 best improvement 都发生在早期 update。

如果 reward 主要依赖 frontier improvement，那么后期训练会变成：

```text
绝大多数 completion 都没有正向差异
  -> reward 组内零方差
  -> GRPO 无 advantage
  -> LoRA 后训练大部分轮次空转
```

### 2.5 问题四：当前 reward 没有区分“搜索有效性”和“训练有效性”

EoH-RL 奖励需要同时回答两个问题：

```text
这个候选对搜索有没有价值？
这个候选对 GRPO 学习有没有区分信号？
```

当前机制偏向第一个问题，即只有真正突破才强奖励。但它没有很好地解决第二个问题：同组 completion 之间需要有连续差异。

---

## 3. 为什么不能照搬 CALM 奖励

### 3.1 CALM 的奖励依赖强 seed

CALM 初始化时通常加载固定 seed：

```text
seed.py -> HeuristicPolicy -> evaluate -> self.algos
```

这个 seed 本身质量较好，因此 CALM 的搜索更像：

```text
在强 seed 附近做局部规则修改、组件注入和交叉组合。
```

而 EoH-RL 当前初始化是：

```text
SFT 模型根据 i1 prompt 采样若干初始算法
```

这些初始算法不是固定强 seed，质量和结构都更不稳定。

### 3.2 CALM 的算子语义更细粒度

CALM 的算子包括：

```text
simplification
injection
replacement_ins
replacement_hyp
replacement_crd
crossover
```

这些算子很多是在已有算法上做细粒度局部修改。

EoH-RL 当前算子是：

```text
m1: 单父代引入新组件
m2: 单父代替换弱片段或决策规则
```

虽然有相似之处，但 EoH-RL 的 `e1/e2/m1/m2` 更粗粒度，且父代质量不稳定，因此不能使用 CALM 那种高度依赖 base parent 的 reward。

### 3.3 CALM 的 reward 更适合局部改良

CALM reward 的核心是：

```text
best_base_perf = 当前 prompt 中父代的最好性能
如果 candidate > best_base_perf:
    reward = 1 + relative improvement
否则:
    reward = negative relative drop
```

这适合强 seed 场景，但在 EoH-RL 中容易出现两个问题：

1. 如果 parent 本身较弱，击败 parent 不一定代表全局有价值；
2. 如果 parent 本身很强，长期无法击败 parent 又会导致 reward 稀疏。

因此，EoH-RL reward 必须同时考虑：

```text
相对 parent 的局部改进
相对 frontier 的全局改进
同组 completion 的相对差异
候选结构是否有效且新颖
```

---

## 4. 本地模型后训练强化学习的奖励设计原则

### 4.1 SFT 的角色

SFT 在本项目中的作用是：

```text
让本地模型学会任务 prompt、输出格式、代码模板和基本启发式模式。
```

SFT 不负责探索最优算法。它只是提供初始策略分布。

因此，RL reward 不应只是格式奖励，而应重点驱动：

1. 更优算法性能；
2. 更稳定的有效代码生成；
3. 更有搜索价值的局部变异和交叉；
4. 更少的无效、随机、泄漏、复制父代行为。

### 4.2 GRPO 的关键要求

GRPO 使用同一 prompt 下多个 completion 的组内 reward 差异来计算 advantage。

如果一组 completion 的 reward 完全相同：

```text
reward_std = 0
advantage 无法区分 completion
本轮训练几乎没有策略改进信号
```

因此，EoH-RL reward 必须尽量避免把大量有效 completion 压成同一个奖励值。

### 4.3 程序搜索的奖励必须分层

程序搜索 reward 应该按层次判断：

```text
格式层：有没有输出代码？有没有目标函数？
安全层：是否随机？是否泄漏 score metadata？是否复制父代？
执行层：能否执行？score 是否有限？
性能层：是否超过 parent？是否超过 frontier？差距是多少？
搜索层：是否进入种群/归档？是否存活？是否带来多样性？
```

当前第一阶段最适合修的是性能层和组内差异层。

### 4.4 新颖性只能作为小幅辅助

结构新颖性有价值，但不能主导 reward。

原因：

```text
模型很容易为了新颖写出不同但性能差的代码。
```

因此，novelty bonus 必须满足：

1. 只对 valid candidate 生效；
2. 权重远小于 performance reward；
3. 对明显劣于 parent 的样本设置上限；
4. 不给 invalid/random/leak/copy 样本任何新颖性奖励。

---

# 方案一：连续相对性能奖励方案

## 5.1 设计定位

方案一是最小 reward 修复方案。它不引入结构新颖性，也不改变训练流程，只把当前离散状态奖励改成连续相对性能奖励。

适合快速验证：

```text
把 parent improvement 和 valid non-improving 从固定值改成连续值，是否能降低 zero-std。
```

## 5.2 基本定义

定义 utility：

```text
U(s) = s,      如果分数越大越好
U(s) = -s,     如果分数越小越好
```

当前 TSP/CVRP/OBP 的日志中通常使用负成本作为 score，因此越接近 0 越好，可以直接用：

```text
U(s) = s
```

对每个 candidate：

```text
u_c = U(candidate_score)
u_p = U(parent_best_score)
u_f = U(population_best_score)
```

相对改进：

```text
delta_parent = u_c - u_p
delta_frontier = u_c - u_f

rel_parent = delta_parent / (abs(u_p) + eps)
rel_frontier = delta_frontier / (abs(u_f) + eps)
```

其中：

```text
eps = 1e-6
```

## 5.3 奖励规则

### 5.3.1 无代码或解析失败

```text
r = no_code_reward
```

推荐：

```text
no_code_reward = -1.0
```

### 5.3.2 执行失败或 score 非有限

```text
r = infeasible_reward
```

推荐：

```text
infeasible_reward = -0.75
```

### 5.3.3 随机、score 泄漏、复制父代

```text
r = gate_penalty
```

推荐：

```text
gate_penalty = -0.60
```

这里 penalty 可以略轻于 `no_code`，因为这类样本至少能解析/执行，但必须明确惩罚，不能进入种群。

### 5.3.4 超过 frontier

```text
if delta_frontier > eps:
    r = frontier_base + clip(rel_frontier / tau_frontier, 0.0, frontier_bonus_cap)
```

推荐：

```text
frontier_base = 1.0
tau_frontier = 0.02
frontier_bonus_cap = 1.0
```

即：

```text
r ∈ [1.0, 2.0]
```

### 5.3.5 超过 parent 但未超过 frontier

```text
elif delta_parent > eps:
    r = parent_base + clip(rel_parent / tau_parent, 0.0, parent_bonus_cap)
```

推荐：

```text
parent_base = 0.15
tau_parent = 0.02
parent_bonus_cap = 0.75
```

即：

```text
r ∈ [0.15, 0.90]
```

这样 parent improvement 不再是固定 `0.0`，但仍低于 frontier improvement。

### 5.3.6 有效但未超过 parent

```text
else:
    penalty = clip((-rel_parent) / tau_worse, worse_min_penalty, worse_max_penalty)
    r = -penalty
```

推荐：

```text
tau_worse = 0.05
worse_min_penalty = 0.02
worse_max_penalty = 0.50
```

即：

```text
r ∈ [-0.50, -0.02]
```

这使得有效但较差的样本仍然可区分：

```text
接近 parent 的样本只小罚；
远差于 parent 的样本重罚。
```

## 5.4 方案一的完整公式

```text
if no_code_or_parse_fail:
    r = -1.0

elif exec_failed_or_non_finite:
    r = -0.75

elif random_or_leak_or_exact_parent_copy:
    r = -0.60

elif delta_frontier > eps:
    r = 1.0 + clip(rel_frontier / 0.02, 0.0, 1.0)

elif delta_parent > eps:
    r = 0.15 + clip(rel_parent / 0.02, 0.0, 0.75)

else:
    r = -clip((-rel_parent) / 0.05, 0.02, 0.50)
```

## 5.5 优点

1. 改动小；
2. 能直接降低 parent improvement 全部为 0 的问题；
3. 能直接降低 valid non-improving 全部为 -0.25 的问题；
4. 不改变 GRPO 训练流程；
5. 不引入结构距离等新模块。

## 5.6 缺点

1. 如果同组候选 score 完全相同，仍可能 zero-std；
2. 不利用同分但不同代码的结构信息；
3. 不显式利用组内排序；
4. 对搜索多样性的帮助有限。

## 5.7 适用阶段

方案一适合快速验证 reward 离散化是否为主因。但从当前日志看，单靠方案一可能不够，因为同分/近似同分仍会导致 reward 相同。

---

# 方案二：连续性能与组内区分混合奖励方案

## 6.1 设计定位

方案二是本文最推荐的奖励机制。

它在方案一的连续性能奖励基础上，加入两个轻量辅助项：

```text
reward = performance_reward
       + group_rank_bonus
       + novelty_bonus
```

三者分别服务不同目标：

```text
performance_reward: 保证奖励方向仍以真实算法性能为主。
group_rank_bonus: 给 GRPO 提供组内相对差异。
novelty_bonus: 在同分/近似同分时给结构探索极小正信号。
```

## 6.2 第一部分：连续性能奖励

沿用方案一。

```text
r_perf = continuous_relative_performance_reward
```

它负责回答：

```text
candidate 相对 parent/frontier 到底好多少？
```

## 6.3 第二部分：组内排序奖励

GRPO 的核心是同一 prompt 下多个 completion 的组内比较。因此，除了绝对 reward，还应给同组 completion 一个小幅排序信号。

### 6.3.1 分组方式

按 `prompt_id` 分组：

```text
group = all completions with same prompt_id
```

只对 valid candidate 计算组内排序。

无效样本不参与 rank bonus。

### 6.3.2 排名定义

对同组 valid candidates，按 utility 排序。

设：

```text
m = 组内 valid candidate 数量
rank_i = candidate i 在组内的排名，0 表示最好
```

定义百分位：

```text
rank_percentile_i = 1 - rank_i / max(m - 1, 1)
```

如果 `m=1`，则：

```text
rank_percentile_i = 0.5
```

### 6.3.3 组内排序奖励

```text
rank_bonus_i = lambda_rank * (rank_percentile_i - 0.5)
```

推荐：

```text
lambda_rank = 0.10
```

因此：

```text
组内最好 candidate:  +0.05
组内最差 candidate:  -0.05
```

如果想更强一点，可以用：

```text
lambda_rank = 0.15
```

但第一版不建议超过 0.15，避免 rank bonus 覆盖真实 performance。

### 6.3.4 同分处理

如果组内所有 valid candidate 的 score 完全相同，不建议随机打破 tie。

推荐：

```text
如果 score 完全相同：
    rank_bonus = 0
```

或者采用极弱 novelty tie-break：

```text
如果 score 完全相同，但代码结构不同：
    rank_bonus 最大幅度降为 0.03
    按 novelty 排序
```

第一版建议保守：

```text
score 完全相同 -> rank_bonus = 0
```

主要依赖 novelty bonus 给微弱差异。

## 6.4 第三部分：结构新颖性奖励

结构新颖性不是主目标，只是辅助探索。

### 6.4.1 距离定义

候选代码与父代代码的距离：

```text
d_code(candidate, parents) = min_j distance(candidate, parent_j)
```

第一版推荐 token Jaccard distance：

```text
tokens(code) = 代码中的标识符、关键字、运算符、常量类别

Jaccard(a,b) = |tokens(a) ∩ tokens(b)| / |tokens(a) ∪ tokens(b)|

distance(a,b) = 1 - Jaccard(a,b)
```

如果暂时不想实现 tokenizer，可以先用简单 split/token regex。

### 6.4.2 新颖性奖励公式

```text
novelty_bonus = lambda_novelty * clip(d_code, 0.0, 1.0)
```

推荐：

```text
lambda_novelty = 0.03
```

### 6.4.3 新颖性上限

如果 candidate 明显差于 parent：

```text
delta_parent <= 0
```

则限制：

```text
novelty_bonus = min(novelty_bonus, 0.01)
```

如果 candidate 无效、随机、泄漏、复制父代：

```text
novelty_bonus = 0
```

这样能避免模型为了新颖而写低质量代码。

## 6.5 方案二完整公式

对无效样本：

```text
reward = invalid_penalty
```

对有效样本：

```text
reward = r_perf + rank_bonus + novelty_bonus
```

其中：

```text
r_perf:
  来自方案一连续性能奖励

rank_bonus:
  lambda_rank * (rank_percentile - 0.5)

novelty_bonus:
  lambda_novelty * code_distance_to_parents
```

推荐初始参数：

```text
eps = 1e-6

frontier_base = 1.0
tau_frontier = 0.02
frontier_bonus_cap = 1.0

parent_base = 0.15
tau_parent = 0.02
parent_bonus_cap = 0.75

tau_worse = 0.05
worse_min_penalty = 0.02
worse_max_penalty = 0.50

lambda_rank = 0.10
lambda_novelty = 0.03

no_code_reward = -1.0
infeasible_reward = -0.75
gate_penalty = -0.60
```

## 6.6 方案二的实现位置

当前代码中，reward 在 `EoHGRPOReward.__call__()` 内返回。

现有流程是：

```text
parse completion
evaluate pending programs
_finish(row, result) 计算 reward
return final reward list
```

方案二需要稍微调整：

```text
1. _finish() 不直接返回最终 reward，而是返回 r_perf 和基础字段；
2. 所有 completion 评估完成后，按 prompt_id 分组；
3. 对每组 valid candidates 计算 rank_bonus；
4. 对每个 valid candidate 计算 novelty_bonus；
5. 写入 row:
      reward_perf_component
      reward_rank_component
      reward_novelty_component
      reward
6. 返回最终 reward list。
```

这样不会改变 TRL 的接口，只改变 reward callback 内部计算。

## 6.7 需要新增的日志字段

每个 reward event 建议新增：

```text
reward_perf_component
reward_rank_component
reward_novelty_component
reward_total
group_reward_std
group_score_std
group_utility_std
code_distance_to_parent
rank_percentile
zero_std_reason
```

summary 中新增：

```text
reward_perf_stats
reward_rank_stats
reward_novelty_stats
group_reward_std_mean
group_score_std_mean
zero_std_after_components_count
```

## 6.8 优点

1. 能显著降低 reward zero-std；
2. 保留 frontier improvement 的最高优先级；
3. 给 parent improvement 连续正信号；
4. 给 valid non-improving 连续负信号；
5. 给同组 completion 提供轻量区分；
6. 给结构探索很小但明确的奖励；
7. 不需要重构 GRPOTrainer；
8. 适合当前 SFT -> GRPO -> EoH 的闭环。

## 6.9 缺点

1. 比方案一复杂；
2. 需要实现 code distance；
3. 需要小心 novelty 权重，避免鼓励无效多样性；
4. 如果 score 完全相同且 novelty 也相似，仍可能 zero-std。

## 6.10 推荐使用阶段

方案二是当前最推荐的 reward 方案。

建议在完成种群 admission 修复后实施。

---

# 方案三：注册后搜索证据奖励方案

## 7.1 设计定位

方案三是长期版，目标是让 reward 不仅基于 evaluation score，还基于候选是否真正进入搜索状态。

当前 reward callback 中，注册种群发生在 GRPO 训练之后：

```text
GRPO reward callback 计算 reward
  -> trainer.train()
  -> EoH._register_candidate()
  -> 回填 registered_to_population / blocked_from_population
```

因此，当前注册结果不能参与本轮 reward。

方案三则希望变成：

```text
completion
  -> parse/evaluate
  -> register into population/archive
  -> compute final reward with registration evidence
  -> train
```

## 7.2 搜索证据字段

可用于 reward 的搜索证据包括：

```text
beats_frontier
beats_parent
accepted_to_archive
entered_active_population
survived_active_population
blocked_duplicate_code
blocked_gate
exact_parent_copy
```

## 7.3 奖励结构

```text
reward = performance_component
       + search_acceptance_component
       + survival_component
       - blocking_penalty
```

示例：

```text
frontier improvement:
    +1.5 到 +2.0

parent improvement:
    +0.2 到 +0.8

accepted_to_archive:
    +0.05

entered_active_population:
    +0.10

valid but not accepted:
    0 到小负值

duplicate_code:
    -0.20

gate blocked:
    -0.60 到 -1.00
```

## 7.4 优点

1. reward 与真实搜索状态完全一致；
2. 能避免奖励那些虽然有效但完全没有进入搜索系统的候选；
3. 适合 archive + active population 结构；
4. 适合后续自适应算子 credit；
5. 可以更好地服务 LoRA gate / rollback。

## 7.5 缺点

1. 改动大；
2. 当前 TRL GRPOTrainer 的在线 reward callback 不天然支持注册后再训练；
3. 可能需要自定义 rollout -> reward -> train 流程；
4. 不适合作为当前第一阶段修复。

## 7.6 折中方案

短期内不要把注册后证据纳入 GRPO reward，而是用于：

```text
operator credit
population metrics
LoRA gate 诊断
后续 DPO preference 数据构造
```

等方案一/方案二稳定后，再考虑 delayed reward。

---

## 8. 可选方向：离线偏好学习辅助

### 8.1 为什么不是当前主方案

DPO/RLAIF 类方法适合从已有候选中构造 preference pair：

```text
winner: 更优算法
loser: 较差算法
prompt: 同一父代上下文或同一任务上下文
```

但当前系统是在线 GRPO + 搜索闭环，主要问题是在线 reward zero-std。因此，DPO 不应替代当前 reward 修复。

### 8.2 未来用途

当 archive 足够大后，可以周期性构造：

```text
(prompt, winner_code, loser_code)
```

用于：

1. 离线增强 SFT；
2. 周期性 DPO 微调；
3. 稳定模型输出分布；
4. 提高下一轮 EoH-RL 初始化质量。

---

## 9. 推荐执行顺序

### 9.1 第一阶段：先修种群 admission

奖励修复前，应先修种群维护：

```text
去掉 score duplicate
只按 normalized code 判重
补 blocked_reason 日志
```

原因：

```text
否则 reward 修好了，候选仍可能因为同分被挡在种群外。
```

### 9.2 第二阶段：实现奖励方案二轻量版

建议实现：

```text
连续性能奖励
组内 rank bonus
小幅 novelty bonus
```

先不要实现：

```text
注册后 reward
DPO
LoRA rollback
复杂 collapse
```

### 9.3 第三阶段：跑同配置对照实验

对 TSP50/TSP100 各跑 3 次，对比：

```text
zero_std_updates
reward_std_mean
group_score_std_mean
registered_total
population_generation
frontier_improve_count
parent_improve_count
best_improvement_last_update
best_score
```

### 9.4 第四阶段：根据日志决定下一步

如果：

```text
zero_std 显著下降
registered_total 上升
best 后期仍无提升
```

则考虑 archive active population 或 diversity parent selection。

如果：

```text
zero_std 仍高
```

则继续调整 reward 连续性和 rank/novelty 组件。

如果：

```text
reward 有差异但模型输出退化
```

则再考虑 LoRA trust-region 或 rollback。

---

## 10. 最终建议

当前最适合本项目的奖励机制不是 CALM reward，也不是纯 frontier reward，而是：

```text
以真实性能改进为主
以组内排序降低 GRPO 零方差
以小幅结构新颖性鼓励探索
严格惩罚无效、随机、泄漏和复制父代
```

推荐第一版公式：

```text
reward = r_perf + 0.10 * (rank_percentile - 0.5) + 0.03 * code_distance_to_parents
```

其中：

```text
r_perf =
  1.0 + clip(rel_frontier / 0.02, 0, 1.0),       if beats_frontier
  0.15 + clip(rel_parent / 0.02, 0, 0.75),       if beats_parent only
 -clip((-rel_parent) / 0.05, 0.02, 0.50),        if valid_non_improving
```

无效样本：

```text
no_code = -1.0
infeasible = -0.75
gate_blocked = -0.60
```

最关键原则：

```text
不要把所有 parent improvement 压成 0。
不要把所有 valid non-improving 压成 -0.25。
不要让 novelty 主导 reward。
不要在 reward 还离散时急着堆 collapse、operator scheduler、LoRA rollback。
```
