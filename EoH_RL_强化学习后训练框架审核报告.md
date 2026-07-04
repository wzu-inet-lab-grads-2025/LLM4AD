# EoH-RL 强化学习后训练框架深度审核报告

> 审核日期：2026-07-04
> 审核范围：`llm4ad/method/eoh_rl/` 全部代码（6095行，13个Python文件）+ 最近10次运行日志
> 对比基准：CALM 源码（1158行，5个Python文件）

---

## 一、核心结论

**当前 EoH-RL 框架的强化学习闭环存在根本性设计缺陷，导致 RL 训练不仅无法提升性能，反而严重退化。**

关键数据对比：

| 指标 | EoH-RL（带LoRA门控） | EoH-RL（无门控回滚） | 目标值 |
|------|---------------------|---------------------|--------|
| 初始最优分数 | -9.566 | -9.566 | -8.50 |
| 最终最优分数 | **-9.167** | **-9.159** | **-8.50** |
| LoRA接受率 | **13.0%**（16/123） | 100%（114/114） | - |
| 坍缩触发次数 | 1次（第120步） | 0次 | - |
| 每步采样数 | 4 | 4 | - |
| 代码总行数 | 6095 | 6095 | ~1158（CALM） |

**结论：带门控的运行（-9.167）和无门控的运行（-9.159）效果几乎一样差，都远未达到 -8.50 的目标。这表明问题不仅仅是 LoRA 门控，而是整个 RL 闭环的设计缺陷。**

---

## 二、五大致命问题详解

### 2.1 致命问题一：LoRA 门控摧毁了 RL 学习闭环

#### 问题描述

`SFTAnchorTrustRegionGate`（`rl/lora_gate.py`）在每次 GRPO 训练后检查是否接受更新后的 LoRA 权重。接受条件是：

```python
# lora_gate.py:89
accepted = bool((not no_valid_completion) and has_search_evidence)
# has_search_evidence = frontier > 0 or parent > 0 or registered > 0 or positive > 0
```

当门控拒绝时（`accepted=False`），调用 `_restore_policy(previous)` **将所有可训练权重恢复到训练前的状态**，意味着这一步的 GRPO 训练完全被丢弃。

#### 实际运行数据

```
总更新次数: 123
接受次数: 16 (13.0%)
拒绝次数: 107 (87.0%)
拒绝原因: 全部是 "gate_no_search_evidence"
```

**87% 的训练计算被浪费**。更严重的是，这 creates 了一个恶性循环：

```
模型陷入局部最优
→ 采样结果无法突破 frontier
→ gate 判定无搜索证据
→ 权重回滚到旧状态
→ 模型仍然是旧的（无法学习的）状态
→ 继续产生类似结果
→ 再次被拒绝...
```

#### 根本原因

RL 的核心原理是**从失败中学习**。当模型生成的解不好时，GRPO 的梯度信号恰恰告诉模型"不要这样做了"。但 LoRA 门控把这种负面信号的训练效果完全丢弃了，只保留了"碰巧产生好结果"的训练步骤。

这等价于：**一个学生只在做对题时才允许记住知识，做错题时强制遗忘。** 这种学习策略永远无法让模型从错误中改进。

#### 对比 CALM

CALM **完全没有 LoRA 门控**。每次 GRPO 训练后无条件接受权重更新。CALM 的 `reward_func`（`calm_trainer.py:349-481`）直接返回奖励值，TRL 的 GRPOTrainer 正常更新权重，不做任何回滚。

### 2.2 致命问题二：每步仅 4 个样本，GRPO 优势估计噪声极大

#### 问题描述

配置参数：
```yaml
prompts_per_update: 1    # 每次更新仅1个prompt
num_generations: 4        # 每个prompt生成4个completion
```

这意味着每次 GRPO 训练步只有 **4 个 completion** 用于计算组内优势（group advantage）。

#### 为什么这不够

GRPO 的核心公式是：
```
advantage_i = (reward_i - mean(rewards)) / std(rewards)
```

当组大小 N=4 时：
- 平均值和标准差都高度不稳定
- 4 个样本中只要有 1 个异常值，整个组的优势估计就会严重偏斜
- 前沿突破（frontier_improve）的概率极低：如果突破概率为 5%，4 个样本中至少 1 个突破的概率仅 18.5%

#### 对比 CALM

CALM 默认配置（`args.py`）：
```python
n_prompts: 1              # 但实际每epoch有8个prompt
ub_simplification: 1       # 1个简化
ub_injection: 1            # 1个注入
ub_replacement: 2          # 2个替换
ub_crossover: 4            # 4个交叉
# 合计: 8个prompt × 4个generation = 32个completion/epoch
```

CALM 每步有 **32 个 completion**，是 EoH-RL 的 **8 倍**。GRPO 的优势估计远比 EoH-RL 稳定。

### 2.3 致命问题三：BQR 奖励函数在"卡住"时几乎无梯度信号

#### 问题描述

BQR（Bounded Quadrant Reward）将奖励分为四个象限（`rl/reward.py:34-137`）：

| 象限 | 条件 | 奖励范围 | 发生频率（卡住时） |
|------|------|----------|-------------------|
| Q1 突破 | beats_parent | [1.0, 2.0] | 极低（<5%） |
| Q2 同分新颖 | ties_parent + 结构新颖≥0.5 | [0.0, 0.4] | 低（需要精确同分+高新颖度） |
| Q3 同质化 | 不突破不同分 | **[-0.05, 0.0]** | **极高（>80%）** |
| Q4 无效 | 执行失败/随机/泄露 | -1.0 | 中（~15%） |

**核心问题**：当模型陷入局部最优时，80%+ 的 completion 落入 Q3，其奖励范围仅为 **[-0.05, 0.0]**——几乎为零。

GRPO 的优势计算：当组内 4 个样本的奖励分别为 [-0.03, -0.02, -0.04, -0.01] 时：
```
mean = -0.025, std = 0.011
advantages = [-0.005, 0.005, -0.015, 0.015]  # 极小的梯度信号
```

这种信号几乎无法推动模型改变行为。

#### 对比 CALM 的奖励设计

CALM 的奖励（`calm_trainer.py:443-453`）：

```python
delta_perf = np.clip(abs(perf - best_base_perf) / min(abs(perf), abs(best_base_perf)), 1e-10, 1.0)
if is_better:
    reward = 1.0 + delta_perf          # [1.0, 2.0] 突破奖励
else:
    if perf >= best_base_perf:
        reward = 0.0                    # 平局
    else:
        reward = reward_random_algorithm / 2 * (delta_perf if ...)  
        # = -0.375 * delta_perf  ← 比例惩罚！
```

CALM 的关键设计：**失败时给予与性能差距成正比的惩罚**（`-0.375 * delta_perf`）。如果生成了一个极差的解（delta_perf 接近 1.0），惩罚达到 -0.375——是 EoH-RL Q3 最大惩罚（-0.05）的 **7.5 倍**。

CALM 还对"与父代相同的解"给予更大惩罚（`2*0.8` 系数），有效阻止模型偷懒复制父代。

### 2.4 致命问题四：坍缩机制阈值过高，形同虚设

#### 问题描述

自适应坍缩（`rl/collapse.py:249-255`）的阈值计算：

```python
def _adaptive_threshold(self, *, target_size: int, operator_count: int) -> float:
    coverage = max(1, int(target_size)) * max(1, int(operator_count))
    return max(float(coverage), float(fill_horizon), float(interval_ref))
```

当 `target_size=10`，`operator_count=4` 时：
```
coverage = 10 * 4 = 40
# 加上 fill_horizon 和 interval_ref
# 实际阈值 = 100.0（从日志确认）
```

**需要连续 100 次更新无突破才能触发坍缩**。但总运行仅 123 步，坍缩只在第 120 步触发了一次，此时运行几乎已经结束。

#### 对比 CALM

CALM 的坍缩（`calm_trainer.py:171-181`）：

```python
max_stuck_threshold = self.calm_args.max_steps // 20  # 默认 max_steps=500, 即25步
if np.random.random() < self.calm_args.speed_collapse * self.age_stuck or self.age_stuck >= max_stuck_threshold:
    # 触发坍缩
```

CALM 的坍缩触发条件：
1. **概率性触发**：`speed_collapse(0.0005) × age_stuck`——随着停滞步数增加，触发概率线性增长
2. **确定性触发**：`age_stuck >= max_steps/20`——即 25 步无突破就必定坍缩

CALM 的坍缩在 25 步就会触发，而 EoH-RL 需要 100 步。**CALM 的坍缩响应速度是 EoH-RL 的 4 倍**。

### 2.5 致命问题五：种群初始化质量极差

#### 问题描述

EoH-RL 的初始化（`eoh_rl.py:1124-1156`）：
```python
configured_init_budget = 2 * int(self._pop_size)  # = 2 * 10 = 20
```

仅生成 **20 个候选**，从中选 10 个作为初始种群。

实际运行日志：
```
初始化进度: evaluated=20/20, candidates=13, target_pop=10
初始化选优完成: candidates=13, population=10
初始最优: -9.566
```

**初始最优 -9.566 与目标 -8.50 相差 1.07**，这意味着初始化阶段就没有找到好的解。

#### 对比 CALM

CALM 从**种子算法**（`configs/{problem}/seed.py`）启动，种子算法是已知的优秀启发式。CALM 的初始种群第一步就有一个好的基线，RL 只需在此基线上改进。

EoH-RL 完全从随机采样开始，没有利用任何先验知识。

---

## 三、三大机制逐一审核

### 3.1 LoRA 稳定机制（SFT锚定信任域门控）

#### 设计意图
> "lora机制：用来稳定性能和模型输出格式框架，因为强化学习与SFT的优化目标不一致"

#### 审核结论：**设计方向正确，但实现方式根本性错误**

**正确的部分**：
- 使用 SFT LoRA 作为锚点（anchor）确实能防止 RL 偏离 SFT 的输出格式
- 信任域（trust region）概念在 RL 中有理论基础（TRPO/PPO）

**致命错误**：
1. **门控的接受条件要求"搜索证据"**，但搜索证据（frontier_improve/parent_improve）正是模型需要通过 RL 学习才能产生的。这是先有鸡还是先有蛋的问题。

2. **权重回滚（_restore_policy）等同于丢弃训练梯度**。在标准 RL 中，即使当前 rollout 不好，梯度仍然包含有价值的信号（"不要这样做"）。回滚权重等于丢弃了这部分学习信号。

3. **信任域投影（project_trainable_state）在权重空间操作**，但 LoRA 的权重空间高度非线性。对 LoRA 权重做 L2 球投影并不能保证输出行为的稳定性。

4. **anchor_radius=0.5, step_radius=0.25 的物理意义不明确**。这两个超参数的合理值依赖于 LoRA rank、target_modules 和具体任务，但没有自适应调整机制。

#### 修复建议

如果要用 LoRA 稳定机制，应该：

1. **不回滚权重，而是用 KL 惩罚**：在 GRPO 的 `beta`（KL 系数）上做文章，当模型偏离 SFT 过远时增大 beta。这是标准 PPO/TRPO 的做法。

2. **或者用 LoRA rank 冻结**：只训练部分 LoRA 层（如只训练 B 矩阵），保持 A 矩阵固定为 SFT 初始化。这在物理上保证了权重不会偏离太远。

3. **或者用 EMA（指数移动平均）**：不回滚，而是用训练权重和 SFT 权重的 EMA 来做推理。这既保留了学习信号，又平滑了权重变化。

### 3.2 自适应算子权重机制

#### 设计意图
> "自适应算子权重通过动态调整算子比例可以加快生成更好的启发式算法"

#### 审核结论：**在当前配置下完全无效**

**无效原因**：

1. **每步仅 1 个算子**：`prompts_per_update=1` 意味着每次 GRPO 更新只使用 1 个算子。自适应调度器的 `update_from_summary` 每次只收到 1 个算子的反馈，无法做有意义的比较。

2. **债务选择（debt_selection）强制轮转**：`operator_scheduler.py:184-195` 的债务机制确保每个算子被均匀选择，实质上退化为固定轮转，自适应完全失效。

3. **信号计算过于复杂**：`_operator_signal`（`operator_scheduler.py:204-232`）的计算公式：
   ```python
   (2.0 * frontier + parent + 0.25 * positive - failed) / max(1, total)
   ```
   但每次只有 1 个算子的 4 个样本，frontier 和 parent 几乎总是 0，信号接近 `-failed/4`，无法区分算子好坏。

4. **CALM 的做法更简单有效**：CALM 直接用固定比例采样算子（`ub_simplification=1, ub_injection=1, ub_replacement=2, ub_crossover=4`），不做自适应。CALM 发现交叉（crossover）是最有效的算子，所以分配最高配额。

#### 修复建议

如果要保留自适应算子：
1. 增加每步 prompt 数到至少 8（每个算子 2 个 prompt）
2. 简化信号：直接用 `frontier_improve_rate + parent_improve_rate`
3. 或者直接放弃自适应，用 CALM 的固定比例

### 3.3 自适应坍缩机制

#### 设计意图
> "自适应坍缩机制是防止长期陷入局部最优，通过清空种群带来扰动性和多样性从而跳出局部最优"

#### 审核结论：**方向正确，但阈值严重过高**

**正确部分**：
- 保留最优个体 + 多样性恢复的概念是合理的
- 基于 `age_stuck`（停滞步数）触发是标准做法

**问题**：
1. **阈值 100 步过高**：`_adaptive_threshold` 返回 `max(coverage=40, fill_horizon, interval_ref)`，实际计算结果为 100。在 500 步的运行中只触发 1 次。

2. **坍缩后的恢复机制过于保守**：`recovery_min_updates = operator_count = 4`，需要至少 4 步才能完成恢复。在种群从 2 恢复到 10 的过程中，采样效率很低（每步仅 4 个样本）。

3. **CALM 的坍缩更激进且有效**：
   ```python
   # CALM: 25步无突破就坍缩，保留最优+所有种子算法
   self.algos = [best_algo] + self.seed_algos
   # 种子算法提供多样性，不需要复杂的恢复过程
   ```

#### 修复建议

1. 降低阈值：`threshold = max(target_size * 2, 20)`——20 步无突破就触发
2. 加入概率性触发：`P(collapse) = 0.01 * age_stuck`
3. 坍缩后注入种子算法或随机新解来快速恢复多样性

---

## 四、与 CALM 的架构对比

### 4.1 代码复杂度对比

| 组件 | EoH-RL 行数 | CALM 行数 | 倍数 |
|------|------------|----------|------|
| 主训练器 | 1305 (eoh_rl.py) | 572 (calm_trainer.py) | 2.3x |
| GRPO 后端 | 911 (grpo_trainer.py) | 0（用TRL原生） | ∞ |
| 奖励函数 | 369 (reward.py) | ~130 (内联在trainer中) | 2.8x |
| LoRA 门控 | 264 (lora_gate.py) | 0 | ∞ |
| 坍缩控制 | 325 (collapse.py) | ~15 (内联在trainer中) | 21.7x |
| 算子调度 | 388 (operator_scheduler.py) | ~20 (内联在trainer中) | 19.4x |
| SFT 训练 | 670 (sft_train.py) | 0（离线SFT） | ∞ |
| SFT 数据 | 782 (sft_data.py) | 0 | ∞ |
| **总计** | **6095** | **1158** | **5.3x** |

### 4.2 架构哲学对比

**EoH-RL 的设计哲学**：把进化搜索的每个环节都做成独立模块，用复杂的门控/调度/投影机制来控制 RL 训练的每一步。

**CALM 的设计哲学**：把 GRPOTrainer 直接子类化，在 `reward_func` 中完成"解析→评估→入种群→返回奖励"的闭环，让 TRL 的标准 GRPO 流程自然地驱动整个搜索-训练循环。

```
CALM 的流程：
  prepare_dataset() → trainer.train() → reward_func() → prepare_dataset() → ...
  （简单循环，TRL 原生驱动）

EoH-RL 的流程：
  init_population → build_prompt_records → train_once → DirectRewardCallback 
  → commit_candidates → _deploy_lora → lora_gate.decide → 
  [accept/restore] → _observe_collapse → _execute_collapse → 
  _finish_recovery → _save_checkpoint → ...
  （10+ 个环节，每步都可能出错）
```

### 4.3 CALM 奖励设计的简洁性

CALM 的奖励逻辑（`calm_trainer.py:443-464`）：

```python
if prompt.op == 'initialization':
    reward = 0.0
else:
    delta_perf = np.clip(abs(perf - best_base_perf) / min(abs(perf), abs(best_base_perf)), 1e-10, 1.0)
    if is_better:
        reward = 1.0 + delta_perf                          # 突破：1.0~2.0
    else:
        if perf >= best_base_perf:
            reward = 0.0                                    # 平局：0.0
        else:
            reward = -0.375 * delta_perf                    # 退化：-0.375~0.0（比例惩罚）
            if algo in base_algos:
                reward = -0.375 * 1.6                       # 复制父代：-0.6（额外惩罚）
```

**CALM 奖励的 3 个关键设计**：
1. **比例惩罚**：退化解的惩罚与性能差距成正比，梯度信号强
2. **复制惩罚**：对复制父代的行为给予额外 1.6 倍惩罚
3. **无结构新颖度计算**：不做 token 级别的结构分析，省去大量计算

---

## 五、奖励机制跨问题跨规模通用性审核

### 5.1 当前 BQR 奖励的问题

1. **`structure_novelty` 依赖 Python AST/tokenize**：对不同编程语言（如果换到非 Python 问题）不通用
2. **`structure_threshold = 0.50` 是硬编码常量**：不同问题/规模的最优结构新颖度不同
3. **Q3 惩罚（`q3_max_penalty = 0.05`）太小**：对任何问题都提供不了足够的负梯度
4. **`epsilon = 1e-6` 对分数尺度敏感**：TSP 的分数是 -8~-10，而 OBP 的分数可能是 0~100，同样的 epsilon 在不同问题上行为完全不同

### 5.2 CALM 奖励的通用性

CALM 的 `delta_perf` 使用**相对性能差距**：
```python
delta_perf = abs(perf - best_base_perf) / min(abs(perf), abs(best_base_perf))
```
这天然归一化到 [0, 1] 范围，对任何分数尺度都适用。这是跨问题通用的关键。

---

## 六、修复方案与重构建议

### 6.1 短期修复（不改架构，调整参数）

```yaml
# 1. 关闭 LoRA 门控的权重回滚
grpo:
  lora_gate:
    enabled: false              # 直接关闭门控
  
# 2. 增加每步采样数
  prompts_per_update: 8          # 从1改为8（每步32个completion）
  num_generations: 4

# 3. 降低坍缩阈值
collapse:
  enabled: true
  # 需要修改代码: threshold = max(target_size * 2, 20)

# 4. 调整奖励参数
task_rl_common:
  reward_type: bqr_v1
  minimize: false
  # 需要修改代码: q3_max_penalty = 0.375（与CALM对齐）
  # 需要修改代码: 去掉结构新颖度门槛，Q2直接给0
```

### 6.2 中期重构（简化架构）

参照 CALM 的设计，将 EoH-RL 简化为：

```
eoh_rl/
├── trainer.py          # 继承 GRPOTrainer，内联奖励函数和种群管理（~400行）
├── prompt.py           # 进化算子 prompt 模板（~200行）
├── population.py       # 种群管理（~100行）
├── config.py           # 配置解析（~100行）
└── run.py              # 入口脚本（~100行）
总计: ~900行（比当前减少 85%）
```

### 6.3 长期建议（设计原则）

1. **让 TRL 原生 GRPO 流程驱动一切**：不要在外部包装 train_once/deploy_lora/restore_policy 等环节。直接子类化 GRPOTrainer，在 `reward_func` 中完成闭环。

2. **奖励设计遵循三个原则**：
   - **比例性**：奖励/惩罚与性能差距成正比
   - **归一化**：使用相对差距而非绝对差距
   - **简洁性**：不做复杂的结构分析

3. **坍缩机制参照 CALM**：
   - 25 步无突破触发坍缩
   - 概率性提前触发
   - 保留最优 + 种子算法

4. **LoRA 稳定用 KL 惩罚**：
   - 设置较大的 `beta`（如 0.1~0.2）
   - 让 TRL 的 GRPO 自然地通过 KL 约束模型不偏离 SFT 太远
   - 不需要权重回滚或信任域投影

---

## 七、具体代码问题清单

### 7.1 严重Bug

| # | 文件:行号 | 问题 | 影响 |
|---|----------|------|------|
| 1 | `lora_gate.py:89` | 接受条件要求 search_evidence，导致87%训练被丢弃 | 致命 |
| 2 | `grpo_trainer.py:666` | `_restore_policy` 回滚权重，丢弃梯度信号 | 致命 |
| 3 | `reward.py:136-137` | Q3 惩罚最大仅0.05，梯度信号几乎为零 | 严重 |
| 4 | `collapse.py:254` | 阈值=coverage(40)+，实际100步，太保守 | 严重 |
| 5 | `eoh_rl.py:149` | 初始化仅20个样本，种群质量差 | 中等 |

### 7.2 性能问题

| # | 文件:行号 | 问题 | 影响 |
|---|----------|------|------|
| 6 | `lora_gate.py:697-700` | 每次gate决策都拷贝全部LoRA权重到CPU | 每步耗时1-4秒 |
| 7 | `reward.py:233` | `@lru_cache(maxsize=20000)` 对代码token分析做缓存，但不同代码不命中率高 | 内存浪费 |
| 8 | `grpo_trainer.py:616-621` | 每次 GRPO 更新都重新创建 GRPOTrainer 实例 | 初始化开销 |

### 7.3 设计问题

| # | 文件:行号 | 问题 | 建议 |
|---|----------|------|------|
| 9 | `eoh_rl.py:138` | 硬编码 `pop_size == 10` 检查 | 应参数化 |
| 10 | `operator_scheduler.py:184-195` | debt_selection 退化为轮转 | 需要更多样本才能自适应 |
| 11 | `reward.py:39-41` | 硬编码 `structure_threshold=0.50` | 应可配置 |
| 12 | `collapse.py:251` | `fill_horizon` 计算公式不合理 | 简化为固定值 |

---

## 八、总结

### 核心诊断

EoH-RL 的 RL 训练效果远不如纯本地推理进化，根本原因是：

1. **LoRA 门控的权重回滚机制（87%拒绝率）摧毁了 RL 的学习闭环**——这是最致命的问题
2. **每步仅4个样本导致 GRPO 优势估计噪声极大**——是 CALM 采样量的 1/8
3. **BQR 奖励函数在卡住时几乎无梯度信号（Q3惩罚≤0.05）**——无法推动模型改进
4. **坍缩机制阈值过高（100步）**——在500步运行中几乎不触发
5. **种群初始化仅20个样本**——初始最优就远低于SFT水平

### 与 CALM 的差距

CALM 用 1158 行代码实现了 EoH-RL 用 6095 行代码试图实现的功能，且效果更好。核心差距在于：

- CALM **信任 TRL 的标准 GRPO 流程**，不做外部干预
- CALM 的**奖励函数简单且提供强梯度信号**（比例惩罚）
- CALM 的**坍缩机制简单且响应迅速**（25步触发）
- CALM **不回滚权重**，让模型从所有经验中学习

### 最优先修复顺序

1. **关闭 LoRA 门控**（`lora_gate.enabled: false`）——立即生效
2. **增加 prompts_per_update 到 8**——提升采样效率8倍
3. **将 Q3 惩罚从 0.05 提升到 0.375**——提供有效梯度信号
4. **降低坍缩阈值到 20 步**——及时跳出局部最优
5. **增加初始化样本到 40-80 个**——提升初始种群质量

完成以上5项修复后，预计可达到接近 CALM 的性能水平。
