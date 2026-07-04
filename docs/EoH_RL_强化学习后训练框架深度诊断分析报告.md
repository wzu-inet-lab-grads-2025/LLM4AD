# EoH-RL 强化学习后训练框架深度诊断分析报告

> 分析日期：2026-07-04
> 分析对象：`llm4ad/method/eoh_rl/`
> 对比基准：CALM (`/home/yuanyilun/projects/source/CALM/`)
> 数据来源：`logs/TSP/eoh_local_rl/` 最近10次实验日志

---

## 一、问题现象总览

### 1.1 核心表现

| 指标 | 实际表现 | 预期表现 |
|------|---------|---------|
| 最佳分数 | -8.71（RL后） | -8.50 左右（需持续提升） |
| LoRA 稳定性 | 三次运行差异大，-8.65/-8.55/-8.57 | 三次均在 -8.50 附近 |
| RL 阶段突破 | 500步仅前100步有5次提升 | 应持续产生搜索证据 |
| 有效代码生成率 | 后期 valid_rate ≈ 0 | 应 > 50% |
| 前沿突破事件 | 前100步偶尔发生，之后完全为0 | 应持续发生 |

### 1.2 最新一次实验的直接失败

```
[EoHRL] 初始化进度: evaluated=20/20, candidates=0, target_pop=10
[EoHRL] 初始化选优完成: candidates=0, population=0
```

20次采样全部 `missing_idea=4`，种群为空，实验直接终止。SFT训练好的模型在RL入口就已经无法生成有效启发式代码。

---

## 二、根本原因分析

### 2.1 致命缺陷：BQR 奖励函数设计方向性错误

**这是所有问题的根源。**

#### BQR v1 奖励分布

| 象限 | 条件 | 奖励范围 | 含义 | 出现频率 |
|------|------|---------|------|---------|
| q1 | 超越父代/前沿 | +1.0 ~ +2.0 | 正面突破 | **极罕见**（500步约5次） |
| q2 | 打平父代 + 结构新颖 | 0 ~ +0.4 | 新颖探索 | 罕见 |
| q3 | 性能更差 | -0.05 ~ 0 | 同质化惩罚 | **极其常见**（500步中 >95%） |
| q4 | 无效代码 | -1.0 | 硬惩罚 | 常见（约25-50%） |

#### 问题分析

```python
# reward.py:136 - BQR q3 奖励公式
diversity = max(0.0, min(1.0, structure_novelty or 0.0))
reward = -self.q3_max_penalty * (1.0 - min(diversity / self.structure_threshold, 1.0))
# q3_max_penalty = 0.05, structure_threshold = 0.50
# 结果：reward ∈ [-0.05, 0]，几乎总是负数
```

**q3 奖励的本质是"惩罚探索"**：模型生成的代码如果和父代不同但性能更差，就会得到负奖励。这意味着：

1. 模型被训练成**不要探索**——因为探索大概率导致性能下降
2. 模型被训练成**精确复制父代**——这样 diversity=0，reward=0（最好的q3结果）
3. 随着训练进行，模型逐渐丧失多样性，输出坍缩到固定模式
4. 最终坍缩的模式也退化——因为 GRPO 在全是负奖励时仍有梯度，模型会学习输出"最不差"的内容

#### CALM 的对比

```python
# CALM calm_trainer.py:446-453
if is_better:
    reward = 1.0 + delta_perf          # 正面奖励 1.0-2.0
else:
    if perf >= best_base_perf:
        reward = 0.0                    # 打平 = 中性
    else:
        reward = self.calm_args.reward_random_algorithm / 2 * (delta_perf if ...)
        # 即使更差，也是温和的惩罚，且基于 delta_perf 缩放
```

CALM 的关键差异：
- **非改进代码不是负奖励**：打平得0分（中性），更差也是温和惩罚
- **改进奖励显著**：1.0 + delta_perf 给模型明确的正面信号
- **探索不被惩罚**：只要代码有效且不随机，就不会被严重惩罚

### 2.2 GRPO + DAPO Loss 的死亡螺旋放大效应

#### 配置分析

```yaml
# config_resolved.yaml
grpo:
  prompts_per_update: 1      # 每次仅1个prompt
  num_generations: 4         # 每个prompt生成4个completion
  loss_type: dapo            # DAPO loss
  beta: 0.04                 # KL散度系数
  scale_rewards: group       # 组内归一化
```

#### 问题机制

GRPO 在组内（4个completion）计算 advantage：

```
advantage_i = (reward_i - mean(rewards)) / std(rewards)
```

当4个completion全是q3（奖励约-0.03）时：
- mean ≈ -0.03, std ≈ 0.01
- advantage 分布在 [-1, +1] 之间
- **即使所有结果都不好，仍然有"相对较好"的那个获得正 advantage**

DAPO loss 使用这个 advantage 来更新策略：
- 模型被训练成产生"最不差"的q3输出
- 但这个方向并不指向性能改进
- 每步微小漂移累积 → 模型逐渐偏离SFT锚点

#### 实验证据

从日志数据可见的退化轨迹：

| 更新步 | 有效事件 | q1次数 | q3次数 | q4次数 | 典型分数 | 状态 |
|--------|---------|--------|--------|--------|---------|------|
| 1 | 0 | 0 | 0 | 4 | - (全部无效) | 无法解析 |
| 25 | 3 | 0 | 3 | 1 | -10.75 ~ -9.56 | 可解析但无改进 |
| 50 | 4 | 0 | 4 | 0 | -74.9 ~ -9.56 | 部分严重退化 |
| 100 | 4 | 1 | 3 | 0 | -74.7 ~ -8.71 | 最后一次突破 |
| 200 | 4 | 0 | 4 | 0 | -73.98 ~ -9.56 | 全部q3 |
| 300 | 4 | 0 | 4 | 0 | -73.98 ~ -9.56 | 全部q3 |
| 400 | 4 | 0 | 4 | 0 | -36.8 ~ -9.56 | 分数严重退化 |
| 491-500 | 3-4 | 0 | 3-4 | 0-1 | q3/q4 | **无任何搜索证据** |

关键观察：**更新100之后，连续400步零突破**。模型已经完全丧失了生成改进启发式的能力。

### 2.3 LoRA Gate 的失效

#### Gate 的接受条件

```python
# lora_gate.py:79-89
no_valid_completion = bool(valid_count <= 0 or candidate_count <= 0)
has_search_evidence = bool(frontier > 0 or parent > 0 or registered > 0 or positive > 0)
accepted = bool((not no_valid_completion) and has_search_evidence)
```

#### 问题1：`registered > 0` 条件过于宽松

在早期RL更新中，种群可能尚未饱和，候选只要有效就能入池（`registered_to_population=True`）。这被Gate误认为是"搜索证据"，导致早期更新被接受。

```
更新2-6: 所有都是 accepted_lora=True
  但 q1=0, q3=全部, q4=部分
  这些更新不应该被接受！
```

#### 问题2：单一q1事件触发接受后模型退化

更新100时出现一个q1事件（frontier_improve=1, parent_improve=1），Gate接受。但从更新101开始，模型再也无法产生q1，连续400步全q3。

#### 问题3：Gate无法检测"全局退化"

Gate只看单次更新的统计，无法判断模型是否在长期退化。即使所有completion都是q3（模型已经坏掉），只要有一个valid completion，Gate就可能接受。

### 2.4 自适应坍缩机制的无效性

#### 坍缩触发条件

```python
# collapse.py:125-134
should_collapse = bool(
    population_full and not in_recovery and self._best_score is not None
    and mature and cooldown_ready and long_stagnation
    and local_progress_stalled and diversity_low
)
```

#### 为什么坍缩无效

1. **坍缩只清空种群，不恢复模型权重**。模型已经在500步RL训练中严重退化，清空种群后重新生成的仍然是垃圾代码。

2. **坍缩保留了"最佳个体"**，但这个最佳个体（-8.71）本身就是RL退化前的产物。坍缩后，从退化模型生成的新个体质量远不如它。

3. **坍缩无法区分"种群卡住"和"模型退化"**。当前设计假设种群多样性低是因为进化停滞，但实际上是因为模型已经无法生成多样化的代码。

### 2.5 自适应算子权重的失效

算子调度器根据每个算子的历史表现动态调整概率，但：
- 所有算子的q1产出在后期都是0
- 所有算子都只有q3产出
- 动态调整无法区分不同算子的质量（都是差）
- 调度器退化为均匀分布或基于噪声的选择

### 2.6 缺少 System Prompt

```python
# prompt.py:21-22
@classmethod
def get_system_prompt(cls) -> str:
    return ""  # 空字符串！
```

对比 CALM：

```python
# CALM calm_trainer.py:484-486
@property
def system_prompt(self):
    return f"""Searching superior heuristics on the {self.problem.name} problem...
    ## Your Task
    You should first present a concise conceptual description...
    * The description must:
        * Be enclosed with a double brace and starts with "The idea of..."
    * The code must:
        * Strictly follow the input-output variable names...
        * Be a single Python function formatted within Python code blocks...
    """
```

EoH-RL 完全没有告诉模型：
- 输出格式要求（双花括号包裹想法 + Python代码块）
- 确定性要求
- 代码质量要求
- 任务上下文

这导致模型输出格式不一致，解析失败率高。

### 2.7 单 Prompt 采样不足

```yaml
prompts_per_update: 1      # 每次GRPO仅1个prompt
num_generations: 4         # 每个prompt仅4个completion
```

对比 CALM：`n_prompts` 通常为 10-20，每个 prompt 也生成多个 completion。

4个completion的组太小：
- 统计意义上不足以估计 advantage
- 如果4个都是q3，没有对比基准
- GRPO 的 group-relative advantage 在小样本下噪声极大

---

## 三、架构对比：EoH-RL vs CALM

| 维度 | EoH-RL | CALM | 影响 |
|------|--------|------|------|
| 奖励设计 | BQR q1-q4，q3惩罚探索 | 性能delta奖励，非改进中性 | **EoH-RL惩罚探索 → 模型坍缩** |
| 奖励范围 | q3: [-0.05, 0] 全负 | 非改进: 0 或温和正数 | **CALM保持探索动力** |
| GRPO设置 | 1 prompt × 4 completions | n_prompts × n_generations | **EoH-RL样本太少** |
| Loss类型 | DAPO (beta=0.04) | 标准GRPO | DAPO放大负信号 |
| 系统提示 | 空字符串 | 详细的任务+格式说明 | **EoH-RL解析失败率高** |
| LoRA管理 | Gate-based信任域 | 直接保存最佳模型 | Gate逻辑有缺陷 |
| 种群管理 | 自适应坍缩 | 简单stuck计数+坍缩 | 坍缩无法修复退化权重 |
| 算子调度 | 自适应SGCA | 固定上限+随机采样 | 退化时无差别 |
| 进化算子 | i1/m1/m2/e1/e2 | create/simplify/inject/replace/crossover | CALM更多样 |
| 代码量 | ~450 KB（核心+RL） | ~60 KB（全部） | **EoH-RL过度工程化** |

---

## 四、死亡螺旋的完整机制

```
步骤1: SFT模型正常初始化种群 → best ≈ -9.56
步骤2: RL开始，模型生成代码 → 大多q3（负奖励）
步骤3: GRPO+DAPO用负奖励更新 → 模型微小漂移
步骤4: 偶尔出现q1 → LoRA Gate接受 → 模型权重更新
步骤5: 更新后模型探索更激进 → 更多q3+q4 → 更负的奖励
步骤6: GRPO继续用负奖励更新 → 模型进一步退化
步骤7: 模型退化到只能生成q3/q4 → 分数坍缩到-74/-55/-36
步骤8: Gate拒绝所有更新（无搜索证据） → 但模型已经坏了
步骤9: 坍缩触发 → 清空种群 → 但退化模型生成的全是垃圾
步骤10: 实验在-8.71终止 → 远不如纯进化（-8.50）
```

**关键转折点：更新100的q1事件**
- 这是最后一次突破，score=-8.71 → -8.712
- Gate接受此更新
- 之后400步零突破 → 模型已经不可逆退化

---

## 五、具体代码问题清单

### 5.1 奖励函数（reward.py）

| 行号 | 问题 | 严重性 |
|------|------|--------|
| 135-137 | q3奖励公式恒为负，惩罚所有探索 | **致命** |
| 126-128 | q1需要同时击败父代和前沿，条件过严 | 高 |
| 131-133 | q2需要打平父代+高新颖度，几乎不可能触发 | 中 |
| 340 | `invalid_reward=-1.0`，与q3差距过大，q3→q4梯度太陡 | 中 |

### 5.2 LoRA Gate（lora_gate.py）

| 行号 | 问题 | 严重性 |
|------|------|--------|
| 87 | `registered > 0` 被算作搜索证据 | **高** |
| 89 | `accepted` 只需要一个有效completion + 任一搜索证据 | **高** |
| 62-65 | 不考虑历史趋势，只看单步统计 | 中 |
| 88 | `positive > 0` 判断不准确，q3 reward=0 不是正数 | 低 |

### 5.3 GRPO Trainer（grpo_trainer.py）

| 行号 | 问题 | 严重性 |
|------|------|--------|
| 671-672 | `prompts_per_update=1` 太小 | **高** |
| 487-488 | `DirectRewardCallback` 每次新建，不保留历史 | 中 |
| 639-667 | `_deploy_lora` 恢复机制只在Gate拒绝时触发 | 中 |
| 679 | `beta=0.04` 太小，KL约束弱 | 中 |

### 5.4 坍缩管理（collapse.py）

| 行号 | 问题 | 严重性 |
|------|------|--------|
| 125-134 | 坍缩条件只看种群状态，不看模型退化 | **高** |
| 617-622 | `_select_collapse_retained_core` 保留2个个体太少 | 中 |
| 249-255 | `_adaptive_threshold` 计算复杂但效果存疑 | 低 |

### 5.5 Prompt设计（prompt.py）

| 行号 | 问题 | 严重性 |
|------|------|--------|
| 21-22 | `get_system_prompt()` 返回空字符串 | **高** |
| 129-131 | `with_recovery_context()` 直接返回原prompt，无实际效果 | 低 |

### 5.6 主循环（eoh_rl.py）

| 行号 | 问题 | 严重性 |
|------|------|--------|
| 697-698 | `_run_grpo_update` 跳过条件检查只验证种群大小 | 中 |
| 1182 | `_iteratively_init_population` 初始化样本数太少（2×pop_size=20） | 中 |
| 1158-1177 | `_iteratively_use_online_rl` 无安全回退机制 | **高** |

---

## 六、修复方案（按优先级排序）

### 6.1 紧急修复：重新设计奖励函数

```python
# 新的奖励设计原则：
# 1. 有效代码 = 正奖励（鼓励模型保持格式正确）
# 2. 改进性能 = 显著正奖励
# 3. 非改进 = 中性或微小正奖励（不惩罚探索）
# 4. 无效代码 = 负奖励（仅此一项为负）

def compute_reward_v2(score, parent_score, frontier_score, is_valid, is_novel):
    if not is_valid:
        return -1.0          # 仅无效代码为负
    
    base_reward = 0.1        # 有效代码的基础正奖励
    
    if score > frontier_score:  # 突破前沿
        delta = (score - frontier_score) / abs(frontier_score)
        return 1.0 + min(delta, 1.0)  # 1.0 ~ 2.0
    
    if score > parent_score:     # 改进父代
        delta = (score - parent_score) / abs(parent_score)
        return 0.5 + 0.5 * min(delta, 1.0)  # 0.5 ~ 1.0
    
    # 非改进：给予基础正奖励 + 新颖性奖励
    novelty_bonus = 0.1 if is_novel else 0.0
    return base_reward + novelty_bonus  # 0.1 ~ 0.2
```

**核心改变**：将"有效但非改进"从负奖励（q3: -0.05~0）改为正奖励（0.1~0.2），让模型保持生成有效代码的动力。

### 6.2 高优先级修复

#### 6.2.1 添加 System Prompt

```python
@classmethod
def get_system_prompt(cls) -> str:
    return """You are an expert in designing heuristic algorithms for combinatorial optimization.

## Output Format
Your response must follow this exact structure:
1. First, describe your algorithm idea in one sentence, enclosed in double braces:
   {{The idea of the algorithm is to ...}}
2. Then, provide the complete Python function in a code block:
   ```python
   def function_name(...):
       ...
   ```

## Requirements
- The algorithm must be deterministic (no random, no time-based operations)
- Follow the exact function signature provided
- Keep the code concise and efficient
- Do not include any additional explanations beyond the idea and code
"""
```

#### 6.2.2 修复 LoRA Gate 接受条件

```python
# 修改后的接受条件
def decide(self, ...):
    # 必须有 frontier_improve，而不仅仅是 registered
    has_real_evidence = bool(frontier > 0 or (parent > 1 and registered > 1))
    # 或者：要求连续N步有改进才接受
    accepted = has_real_evidence and quality_ema > 0
```

#### 6.2.3 增加 prompts_per_update

```yaml
grpo:
  prompts_per_update: 4      # 从1增加到4
  num_generations: 4         # 保持4
  # 总计16个completion，组内advantage更稳定
```

#### 6.2.4 添加安全回退机制

```python
def _iteratively_use_online_rl(self):
    no_improve_streak = 0
    max_no_improve = 50  # 连续50步无改进则回退
    
    while ...:
        executed = self._run_grpo_update()
        if best_improved:
            no_improve_streak = 0
        else:
            no_improve_streak += 1
        
        if no_improve_streak >= max_no_improve:
            # 回退到SFT锚点
            self._restore_to_anchor_lora()
            no_improve_streak = 0
```

### 6.3 中优先级修复

#### 6.3.1 坍缩时恢复模型权重

```python
def _execute_population_collapse(self, ...):
    # 坍缩时同时恢复到上一个已知良好的LoRA检查点
    self._restore_to_best_lora()
    # 然后清空种群
    self._population.replace_population(retained)
```

#### 6.3.2 增加 GRPO beta 参数

```yaml
grpo:
  beta: 0.1    # 从0.04增加到0.1，增强KL约束
```

更大的beta意味着对偏离参考策略的惩罚更强，可以减缓模型退化速度。

#### 6.3.3 考虑切换 loss_type

```yaml
grpo:
  loss_type: grpo    # 从 dapo 改为标准 GRPO
```

DAPO loss 的 clipping 机制可能在全负奖励时产生意外行为。

### 6.4 低优先级优化

#### 6.4.1 代码精简

当前代码量约450KB，CALM仅60KB。可精简：
- `rl/grpo_trainer.py` (55KB): 合并 DirectRewardCallback 和 ResidentGRPOPolicy 的重复逻辑
- `rl/collapse.py` (15KB): 坍缩逻辑可简化到类似CALM的简单stuck计数
- `rl/operator_scheduler.py` (18KB): debt-based选择机制过于复杂

#### 6.4.2 算子权重简化

当前SGCA调度器过于复杂，可参考CALM的简单方式：固定上限 + 随机采样。

---

## 七、建议的实验验证顺序

1. **仅修复奖励函数**：将q3从负奖励改为正奖励 → 重新运行实验
   - 预期：模型能持续生成有效代码，valid_rate > 50%
   
2. **奖励函数 + System Prompt**：添加格式指导
   - 预期：解析失败率大幅下降
   
3. **奖励函数 + System Prompt + 4x prompts**：增加采样量
   - 预期：GRPO advantage 更稳定，搜索证据出现频率增加

4. **全部修复**：加上Gate修复 + 安全回退 + 坍缩恢复权重
   - 预期：性能应接近或超过纯进化（-8.50），且LoRA稳定

---

## 八、总结

EoH-RL 后训练框架效果差的**根本原因**是奖励函数设计错误：

> **BQR q3 奖励对"有效但非改进"的代码给予负奖励，惩罚了探索行为。GRPO + DAPO loss 在全负奖励环境下产生错误的梯度信号，导致模型逐步退化。LoRA Gate、自适应坍缩和算子调度三个机制都是在"模型正常"的假设下设计的，无法阻止和修复模型退化。**

三个机制的失效原因：
- **LoRA Gate**：接受条件过于宽松，单一q1事件就触发接受，之后模型退化时无法回退
- **自适应算子权重**：所有算子退化后无差别，调度失去意义
- **自适应坍缩**：只重置种群不恢复权重，无法修复已退化的模型

对比CALM的成功，其核心优势在于：
1. 奖励函数鼓励探索（非改进给予中性/温和奖励）
2. 系统提示清晰（模型知道输出格式）
3. 代码简洁（60KB vs 450KB）
4. 没有过度设计的Gate/自适应机制

**建议优先修复奖励函数，这是所有问题的根源。其余机制在奖励函数正确的前提下才能发挥作用。**

---

## 附录：实验数据速查

### A.1 最佳分数演化（run1_20260704_092442）

```
初始化: -9.566
更新 14: -8.903  ← 第1次提升
更新 19: -8.897  ← 第2次
更新 28: -8.876  ← 第3次
更新 91: -8.732  ← 第4次
更新100: -8.712  ← 第5次（最后一次）
更新101-500: 无变化（停滞在-8.712）
```

### A.2 LoRA 接受率

```
总更新: 500
接受次数: 67 (13.4%)
其中前100步: 约20次接受（包含错误接受）
后400步: 约47次接受（大部分无实际改进）
```

### A.3 算子表现（全500步汇总）

所有算子(e1/e2/m1/m2)的 q1 产出在更新100后均为0，全部退化为 q3/q4 产出。

### A.4 坍缩事件

最终更新500触发了一次坍缩，但此时模型已完全退化，坍缩无效。
