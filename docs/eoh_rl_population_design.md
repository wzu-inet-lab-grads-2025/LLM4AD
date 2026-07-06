# EoH-RL 种群维护与父代选择三版设计方案

## 1. 背景与目标

本文件用于整理 `eoh_rl` 在本地模型 SFT + GRPO 后训练闭环中的种群维护与父代选择方案。目标不是直接复刻 CALM，而是在已有 EoH-RL 实验日志、当前代码实现、CALM 源码设计，以及 AEL/EoH、FunSearch、AlphaEvolve 等程序搜索/算法发现范式的基础上，给出三种可逐步落地的设计方案。

当前 EoH-RL 的真实链路是：

```text
SFT LoRA 初始化本地模型
  -> resident Unsloth + TRL GRPO 在线后训练
  -> EoH prompt 构造 e1/e2/m1/m2
  -> reward callback 解析/执行/打分 completion
  -> 候选注册进种群
  -> 父代池继续驱动下一轮 prompt 和 GRPO
```

因此，种群维护不只是传统遗传算法里的搜索状态，它同时影响两个系统：

1. **EoH 搜索系统**：决定哪些算法能作为父代，影响搜索多样性、精英保留和局部改进。
2. **GRPO 后训练系统**：决定 prompt 的父代上下文，影响模型 completion 的多样性、reward 方差和策略梯度信号。

当前日志暴露出的首要问题是：**有效候选大量因为“同分数”而无法进入种群**。这会导致 `next_generation` 难以填满，`active_population` 长期不更新，父代池僵死，进一步造成 prompt 同质化和 reward zero-std。

本文件给出三版方案：

| 方案 | 中文名称 | 核心结构 | 改动量 | 推荐阶段 |
|---|---|---|---:|---|
| 方案一 | 精确代码去重的双区种群最小修复方案 | `active_population + next_generation` | 小 | 立即验证 |
| 方案二 | 归档库与活跃父代池分离的 CALM 启发方案 | `archive + active_parent_pool` | 中 | 方案一后仍停滞时 |
| 方案三 | 多样性归档与停滞恢复的完整演化方案 | `archive + elite/diversity/recovery` | 大 | 论文完整机制/长期版 |

---

## 2. 从 CALM 源码得到的客观结论

### 2.1 CALM 的主流程

CALM 的核心入口在 `main.py` 和 `calm_trainer.py`。本地模型模式下，它使用 Unsloth + LoRA + TRL GRPO，训练流程大致为：

```text
main.py
  -> 加载本地模型与 LoRA
  -> PatchFastRL("GRPO")
  -> 构造 Trainer
  -> 循环执行：
       trainer.train()
       trainer.prepare_dataset()
       trainer.save_trace()
       trainer.save_model()
```

CALM 的 `Trainer` 继承自 TRL `GRPOTrainer`，并用自定义 `reward_func()` 同时完成：

1. 解析 LLM completion；
2. 抽取 idea 和 code；
3. 构造 `HeuristicPolicy`；
4. 执行任务评估；
5. 根据相对父代改进计算 reward；
6. 将新算法写入 `self.algos`。

### 2.2 CALM 的种群不是固定大小种群

CALM 的 `self.algos` 实际上更像一个全局 archive，而不是固定大小 population。

核心逻辑是：

```python
is_new = algo not in self.algos
if is_new:
    self.algos.append(algo)

sorted_indices = np.argsort([a.perf for a in self.algos])[::-1]
self.algos = [self.algos[i] for i in sorted_indices]
algos_head = self.algos[:self.population_size]
```

也就是说：

```text
self.algos                = 全局算法归档库
self.algos[:population_size] = 当前父代候选池
```

CALM 的 `population_size` 不是 archive 的最大容量，而是每轮构造 prompt 时使用的 top-K 父代池大小。

### 2.3 CALM 的父代选择

CALM 从 `algos_head` 中选父代，使用 rank-biased probability：

```python
rank = 1 + np.arange(len(algos_head))
p = 1 / rank
p /= np.sum(p)
```

含义是：

```text
更好的算法有更高概率被选中，但较差算法仍有机会被选为父代。
```

在 crossover 中，CALM 还会以一定概率使用 diversity-driven parent selection：

```python
distances = [- idea_distance(base_idea=algo_0.idea, new_idea=algo_1.idea) for algo_1 in self.algos]
distance_rank = np.argsort(np.argsort(distances)) + 1
distance_based_p = 1 / distance_rank
distance_based_p /= np.sum(distance_based_p)
```

它的思想是：第二父代不总是按性能选，也可以按 idea 差异选，避免交叉退化成相似算法之间的小修小补。

### 2.4 CALM 的同分判重不能照搬

CALM 的 `HeuristicPolicy.__eq__` 是：

```python
def __eq__(self, other):
    return isinstance(other, HeuristicPolicy) and abs(self.perf - other.perf) < 1e-7
```

这意味着 CALM 的 `algo not in self.algos` 主要按 performance 近似相等判断重复，而不是按代码结构判断重复。

这一点不能照搬到 EoH-RL，原因如下：

1. EoH-RL 日志已经证明，不同代码同分的候选大量存在。
2. TSP/CVRP/OBP/JSSP 的有限实例评估很容易产生相同或近似相同分数。
3. 同分不代表算法结构相同。
4. 当前 EoH-RL 的同分拦截已经真实影响了 `next_generation` 填充、generation 推进和父代池更新。

因此，CALM 的思想可以借鉴为：

```text
归档库 archive + 活跃父代池 active parent pool
```

但不能借鉴为：

```text
performance duplicate == algorithm duplicate
```

### 2.5 CALM 的 reward 也不能照搬

CALM 的 reward 强依赖以下前提：

1. 存在固定高质量 seed；
2. 算子更偏局部细粒度修改，例如 simplification、injection、replacement；
3. reward 是相对 prompt 中 base algorithm 的 improvement；
4. CALM 的搜索常常是在强 seed 附近局部改进。

而 EoH-RL 当前是：

1. 初始化不是固定强 seed，而是 SFT 模型从 `i1` prompt 采样形成初始种群；
2. 算子是 EoH 风格的 `e1/e2/m1/m2`；
3. 父代质量更不稳定；
4. GRPO reward 还要服务组内 advantage 信号。

所以 CALM 的 reward 不能直接迁移，种群方案也应避免绑定 CALM 的 reward 语义。

---

## 3. 其他相关方法的共性经验

### 3.1 AEL / EoH 类方法

AEL/EoH 类方法通常具有以下共性：

1. 保存一批已评估算法。
2. 根据性能排序或 rank 选择父代。
3. 使用 LLM 作为 mutation/crossover/operator。
4. 通过 evaluator 给候选打分。
5. 用 survival/selection 保留较好算法。

这类方法的重点是：

```text
父代应偏向高分，但不能完全失去多样性。
```

### 3.2 FunSearch / AlphaEvolve 类方法

FunSearch 和 AlphaEvolve 更强调 program database/archive：

1. 长期保存大量程序候选。
2. 持续用 evaluator 反馈驱动程序改写。
3. 从高质量或多样化的候选中采样上下文。
4. 不把短期 active population 限死为唯一搜索状态。

这类方法对 EoH-RL 的启发是：

```text
archive 和 active parent pool 应分离。
```

也就是说，历史上生成过的有效算法不一定都进入当前父代池，但也不应因为暂时没进入 top-K 就彻底丢失。

### 3.3 GRPO 后训练对种群的特殊要求

在普通遗传搜索中，种群维护主要影响搜索效率；但在 EoH-RL 中，它还影响训练信号。

如果父代池长期不更新，会出现：

```text
父代上下文重复
  -> prompt 分布重复
  -> completion 结构重复
  -> reward 全部相同
  -> frac_reward_zero_std 高
  -> GRPO advantage 无差异
  -> LoRA 后训练基本空转
```

因此，EoH-RL 的种群维护应满足两个目标：

1. **搜索目标**：保留高分算法并维持结构多样性。
2. **训练目标**：提供足够多样的 prompt/reward 对，使 GRPO 有非零组内差异。

---

## 4. 设计原则

### 4.1 去重原则

候选是否重复，应按结构判断，而不是按分数判断。

推荐：

```text
duplicate = same normalized code
```

暂不推荐：

```text
duplicate = same score
duplicate = near same score
```

原因：score 是评价结果，不是算法身份。它应该用于排序、survival、父代选择，而不应作为 admission 阶段的拒绝条件。

### 4.2 父代选择原则

父代选择应具备三点：

1. 偏向高分；
2. 保留非最优算法被选中的概率；
3. 对 crossover 类算子引入一定结构多样性。

第一版推荐继续使用 rank-based selection，因为它简单、稳定、可解释。

### 4.3 先修根因，后加机制

当前日志中已经确认两个根因：

1. 同分判重阻断不同代码候选；
2. reward zero-std 让 GRPO 后期缺少有效学习信号。

因此，种群机制应先修 admission rule，不应一开始就叠加 collapse、自适应算子、LoRA gate 等复杂机制。

---

# 方案一：精确代码去重的双区种群最小修复方案

## 5.1 设计定位

方案一是最小改动方案，目标是直接修复当前实验中已经实锤的问题：**同分不同代码被错误拒绝**。

它保留当前 EoH-RL 的 two-zone population：

```text
active_population: 当前父代池
next_generation: 新候选缓冲区
```

只修改 admission duplicate rule。

## 5.2 当前问题

当前逻辑等价于：

```text
如果代码相同，拒绝。
如果分数相同，也拒绝。
否则进入 next_generation。
```

这会造成：

1. 不同代码但同分数的算法被拒绝；
2. `next_generation` 填不满；
3. `survival()` 不触发；
4. `population_generation` 推进慢；
5. 父代池长期不更新；
6. prompt 和 reward 进一步同质化。

## 5.3 新 admission rule

改成：

```text
如果代码结构重复，拒绝。
如果代码不同，即使分数相同，也允许进入 next_generation。
```

形式化：

```text
Accept(x) = finite_score(x)
            ∧ pass_safety_gate(x)
            ∧ code_key(x) not in code_keys(population ∪ next_generation)
```

其中：

```text
code_key(x) = normalized source code string
```

第一版 normalization 推荐保持简单：

```text
1. 去掉首尾空白；
2. 统一换行；
3. 去掉每行尾部空白；
4. 保留代码主体语义。
```

不建议第一版上复杂 AST normalize，因为当前 sampler 已经做了比较严格的函数名、签名、单函数结构检查。

## 5.4 Survival 逻辑

方案一不改 survival。

仍然使用：

```text
当 len(next_generation) >= pop_size:
  combined = active_population + next_generation
  combined = deduplicate_by_code(combined)
  combined = sort_by_utility(combined)
  active_population = combined[:pop_size]
  next_generation = []
  generation += 1
```

这里 score 只用于：

```text
sort_by_utility
selection probability
best tracking
```

不再用于：

```text
duplicate 判断
admission 拒绝
```

## 5.5 父代选择

保持当前 rank-based selection。

推荐继续使用：

```text
P(parent_i) ∝ 1 / (rank_i + N)
```

或保持当前代码实现，不引入新变量。

原因：

1. 当前父代选择不是首要问题；
2. 先修 admission，可以干净归因；
3. 过早修改 selection 会让实验难以判断改善来自哪里。

## 5.6 日志与指标设计

必须新增或细化以下字段：

```text
registered_to_next_generation_count
blocked_duplicate_code_count
blocked_non_finite_score_count
blocked_random_algo_count
blocked_score_metadata_leak_count
blocked_exact_parent_copy_count
next_generation_size
population_generation
```

每个 candidate row 建议记录：

```text
blocked_reason: null | duplicate_code | non_finite_score | random_algo | score_metadata_leak | exact_parent_copy
registered_to_population: bool
survived_main_population: bool
```

注意：不要再设计 `duplicate_score` 作为拒绝理由。

## 5.7 预期效果

方案一预期改善：

1. `registered_total` 上升；
2. `population_generation` 上升；
3. `same_score_post_not_code` 现象消失；
4. TSP50 run2/run3 这类 generation 卡在 2-3 的情况缓解；
5. 父代池更新频率提升。

方案一不保证立刻解决：

1. reward zero-std；
2. 后期没有 frontier improvement；
3. LoRA 有效更新不足。

因为这些还需要 reward 设计配合。

## 5.8 优点

1. 改动最小。
2. 风险最低。
3. 直接对应日志中实锤问题。
4. 不改变主循环结构。
5. 不引入复杂超参数。
6. 便于和旧实验做公平对比。

## 5.9 缺点

1. 仍然依赖 `next_generation` 填满才更新 active population。
2. 如果后期候选质量低，active pool 仍可能变化慢。
3. 不具备显式 archive，历史候选可能在 survival 后丢失。
4. 不主动维护结构多样性。

## 5.10 推荐使用阶段

方案一是当前最推荐立即验证的版本。

如果方案一后：

```text
registered_total 明显上升
population_generation 明显上升
best 后期仍不提升
zero_std 仍很高
```

则说明种群 admission 已修复，下一步应修 reward。

如果方案一后：

```text
registered_total 上升
但 active population 仍长期变化慢
```

则进入方案二。

---

# 方案二：归档库与活跃父代池分离的 CALM 启发方案

## 6.1 设计定位

方案二借鉴 CALM、FunSearch、AlphaEvolve 的 archive 思想，将“保存所有有效算法”和“当前用于父代选择的算法池”分离。

核心结构：

```text
archive: 所有通过 gate 且代码不重复的有效候选
active_population: 从 archive 中选出的当前父代池，大小 pop_size
```

## 6.2 为什么需要 archive

当前 two-zone 结构的问题是：

```text
candidate -> next_generation
只有 next_generation 满 pop_size 后，active_population 才更新
```

如果 `next_generation` 长期不满，父代池就会僵死。

archive 结构则是：

```text
candidate -> archive
只要足够好，下一轮就可能进入 active_population
```

这能减少“缓冲区未满导致搜索状态不更新”的问题。

## 6.3 数据结构

建议：

```python
class Population:
    _archive: list[Function]
    _active_population: list[Function]
    _code_keys: set[str]
    _generation: int
    _pop_size: int
    _minimize: bool
```

字段含义：

```text
_archive:
  保存所有 unique-code 有效算法。

_active_population:
  当前父代池，只从 archive 中选出 top-K。

_code_keys:
  用于快速判断代码重复。

_generation:
  active_population 更新次数。
```

## 6.4 注册流程

```text
register(candidate):
  if score is not finite:
      reject(non_finite_score)

  if blocked_by_random_or_leak_or_parent_copy:
      reject(gate_blocked)

  key = code_key(candidate)
  if key in _code_keys:
      reject(duplicate_code)

  archive.append(candidate)
  code_keys.add(key)

  old_active = active_population
  rebuild_active_population()

  return {
      accepted_to_archive: true,
      entered_active_population: candidate in active_population,
      evicted_from_active: old_active - active_population
  }
```

## 6.5 Active population 构建

第一版建议简单：

```text
active_population = top pop_size archive members by utility(score)
```

排序规则：

```text
1. utility(score) descending
2. birth_sample_order descending
3. code_key stable order
```

这里同分不删除，而是通过 tie-break 决定排序。

同分排序不影响算法是否保留在 archive，只影响是否进入 active parent pool。

## 6.6 父代选择

父代只从 `active_population` 选择。

推荐继续使用 rank-based：

```text
P(parent_i) ∝ 1 / rank_i
```

或者为了不要过度偏 best，可以使用更平滑版本：

```text
P(parent_i) ∝ 1 / sqrt(rank_i)
```

但第一版建议保留当前 rank 逻辑，不额外引入 selection 超参数。

## 6.7 与 CALM 的相同点和不同点

相同点：

```text
CALM: self.algos = archive
CALM: algos_head = top population_size
EoH-RL 方案二: archive + active_population
```

不同点：

```text
CALM: performance near-equality 判断重复
EoH-RL 方案二: code_key 判断重复

CALM: reward 相对强 seed / base parent
EoH-RL 方案二: 不绑定 reward，只维护搜索状态
```

## 6.8 是否还需要 next_generation

方案二中，`next_generation` 可以删除，也可以降级成诊断用 buffer。

推荐：

```text
删除 next_generation 的 survival 语义。
保留 recent_candidates 作为日志窗口。
```

例如：

```text
recent_candidates = 最近 N 个通过 gate 的候选
```

它只用于：

1. 分析多样性；
2. 统计 operator 近期表现；
3. 辅助后续 collapse/recovery。

不要让它阻塞 active population 更新。

## 6.9 Checkpoint 设计

需要保存：

```json
{
  "archive": [...],
  "active_population": [...],
  "generation": 123,
  "total_sample_nums": 2000,
  "rl_update_count": 500,
  "best_curve": [...]
}
```

`code_keys` 可以恢复时从 archive 重算，不需要保存。

## 6.10 日志与指标设计

新增：

```text
archive_size
active_population_size
accepted_to_archive_count
entered_active_population_count
evicted_from_active_count
blocked_duplicate_code_count
active_rebuild_count
```

每轮 GRPO metrics 里记录：

```text
registered_count              -> 可改名 accepted_to_archive_count
entered_active_count
archive_size_after
active_best_before
active_best_after
active_population_changed
```

## 6.11 预期效果

方案二预期改善：

1. 有效候选不会因为缓冲区没满而无法影响搜索状态。
2. active parent pool 更新更及时。
3. 历史候选不会在一次 survival 中被彻底丢弃。
4. 后续实现 diversity selection 和 operator credit 更自然。

## 6.12 优点

1. 更接近 CALM/FunSearch/AlphaEvolve 的 archive 思想。
2. 避免 `next_generation` 阻塞父代池更新。
3. 能保留更多历史候选。
4. 更适合长期后训练搜索。
5. metrics 更清晰地区分 archive acceptance 和 active survival。

## 6.13 缺点

1. 改动比方案一大。
2. checkpoint 结构要改。
3. archive 可能膨胀，需要后续压缩策略。
4. 如果只按 top-K active，active pool 仍可能同质化。

## 6.14 推荐使用阶段

方案二适合在方案一后仍出现以下现象时启用：

```text
accepted candidates 增加
但 active parent pool 变化慢
best 仍只在早期提升
父代 prompt 仍高度同质
```

---

# 方案三：多样性归档与停滞恢复的完整演化方案

## 7.1 设计定位

方案三是论文/长期完整机制版，目标是同时处理：

1. 精英保留；
2. 结构多样性；
3. 长期停滞；
4. active parent pool 同质化；
5. 后训练 reward zero-std 的间接诱因。

它不适合第一步实现，因为复杂度高、超参数多、归因难。

## 7.2 核心结构

```text
archive: 所有 unique-code 有效候选
elite_pool: 高分候选集合
diversity_pool: 结构差异较大的候选集合
active_population: elite + diverse 混合父代池
recent_window: 最近若干 update 的 reward/event/candidate 窗口
stagnation_state: 停滞监控状态
```

## 7.3 Active population 构建

主动维护 active parent pool：

```text
active_population = {global_best}
                    ∪ utility_elites
                    ∪ diversity_elites
```

建议第一版比例：

```text
global_best: 1 个，必保留
utility_elites: ceil((pop_size - 1) * 0.6)
diversity_elites: 剩余名额
```

例如 `pop_size=10`：

```text
1 个 global best
5 个 utility elites
4 个 diversity elites
```

## 7.4 Diversity 计算

不要第一版使用 embedding，建议从简单到复杂：

### 7.4.1 代码 token Jaccard 距离

```text
tokens(code) = 标识符、关键字、操作符的集合
distance(a,b) = 1 - |tokens(a) ∩ tokens(b)| / |tokens(a) ∪ tokens(b)|
```

优点：

1. 简单；
2. 快；
3. 易维护；
4. 不需要额外模型。

缺点：

1. 不能完全理解语义；
2. 变量名变化可能影响距离。

### 7.4.2 AST 结构距离

```text
AST histogram = 各类 AST node type 计数
distance = cosine distance 或 L1 distance
```

优点：

1. 比 token 更结构化；
2. 对格式变化不敏感。

缺点：

1. 实现稍复杂；
2. 对细微启发式差异不一定敏感。

### 7.4.3 strategy text 距离

可以参考 CALM 的 `idea_distance`，但不建议作为主距离。

原因：

1. LLM strategy 文本可能不稳定；
2. 文字不同不代表代码行为不同；
3. 代码相似但描述不同会误导 diversity。

推荐只作为辅助诊断。

## 7.5 Diversity active selection

选择 diversity elites 时，从 archive 的候选池中挑选与当前 active 差异最大的算法。

伪代码：

```text
active = [global_best]

add top utility elites into active

while len(active) < pop_size:
    candidates = top M archive candidates not in active
    choose x maximizing:
        min_distance(x, active) + alpha * normalized_utility(x)
    active.append(x)
```

其中：

```text
M = 3 * pop_size 或 5 * pop_size
alpha = 0.2 ~ 0.5
```

为了减少超参数，第一版可以固定：

```text
M = 3 * pop_size
alpha = 0.3
```

## 7.6 父代选择

父代选择根据 operator 区分。

### 7.6.1 m1/m2 单父代变异

```text
parent ~ rank-based(active_population)
```

但可以加入轻量 anti-repeat：

```text
如果同一父代连续被选太多次，临时降低其概率。
```

不建议第一版实现 anti-repeat，先记录即可。

### 7.6.2 e1/e2 双父代交叉

```text
parent1: rank-based from active_population
parent2:
  50% rank-based
  50% diversity-biased relative to parent1
```

diversity-biased：

```text
P(parent2 = y) ∝ distance(parent1, y) * rank_weight(y)
```

这样既避免低质量远距离父代被过度选择，也避免两个父代高度相似。

## 7.7 停滞恢复机制

方案三可以引入 modified collapse，但不能照搬 CALM。

CALM collapse 是：

```text
keep best
keep seed_algos
reset used_prompts
```

EoH-RL 不应保留固定 seed，除非 seed 被验证为强 anchor。

推荐恢复策略：

```text
if stagnation_triggered:
    retain global_best
    retain top diverse elites
    clear recent_window
    rebuild active_population with higher diversity weight
    temporarily favor m1/m2 for recovery_period
```

## 7.8 停滞触发条件

不要用固定 25 代阈值。

推荐使用最近突破间隔窗口：

```text
I_k = t_k - t_{k-1}
```

其中 `t_k` 是第 k 次 global best improvement 的 update id。

窗口：

```text
W = 5
```

当前停滞长度：

```text
A_t = t - t_last_best
```

触发：

```text
if len(I_window) >= W
   and active_population is full
   and A_t > gamma * max(I_window)
   and not in_recovery:
       trigger recovery
```

推荐：

```text
gamma = 2.0
```

## 7.9 与 reward zero-std 的关系

方案三可以把 reward zero-std 作为诊断信号，但不建议单独作为 collapse 触发条件。

原因：

```text
zero-std 可能来自 reward 设计，也可能来自父代池同质化。
```

更合理的使用方式：

```text
如果 zero_std_rate 高且长期无 best improvement，增加 diversity active selection 权重。
```

## 7.10 日志与指标设计

新增：

```text
archive_size
active_population_size
active_diversity_mean
active_diversity_min
operator_parent_diversity
stagnation_age
best_improvement_intervals
recovery_active
recovery_count
active_rebuild_reason
```

每次 recovery 记录：

```json
{
  "trigger_update": 320,
  "age_since_best": 180,
  "interval_window": [20, 35, 42, 60, 75],
  "retained_best": "...",
  "retained_diverse_elites": [...],
  "active_before": [...],
  "active_after": [...]
}
```

## 7.11 优点

1. 搜索机制完整。
2. 能主动对抗父代池同质化。
3. 支持长期运行。
4. 与后续自适应算子、LoRA trust-region、collapse 论文叙事兼容。
5. 能解释为什么不是简单堆模块，而是围绕 archive/active/recovery 的闭环。

## 7.12 缺点

1. 实现复杂。
2. 超参数更多。
3. 维护成本高。
4. 如果 reward 仍然 zero-std，复杂种群也无法提供有效 GRPO 信号。
5. 不适合当前第一修复。

## 7.13 推荐使用阶段

方案三应在以下条件满足后再考虑：

1. score duplicate 已移除；
2. reward zero-std 已明显下降；
3. 仍存在长期局部最优；
4. 需要完整论文机制和消融实验。

---

## 8. 三种方案的推荐执行顺序

### 8.1 第一阶段：先做方案一

原因：

```text
当前日志已经实锤 admission rule 错误。
方案一直接修根因，改动最小，最容易验证。
```

验证指标：

```text
registered_total
population_generation
blocked_duplicate_code_count
same_score_post_not_code
frontier_improve_count
best_improvement_last_update
zero_std_updates
```

### 8.2 第二阶段：根据方案一结果选择是否做方案二

如果方案一后：

```text
registered_total 上升
population_generation 上升
active pool 能正常更新
```

则暂不做方案二。

如果方案一后：

```text
候选能进入 next_generation
但 active pool 仍变化慢
或 next_generation 结构仍阻塞搜索状态更新
```

则做方案二。

### 8.3 第三阶段：方案三只作为长期机制

方案三不应和方案一同时实现。

原因：

```text
如果一次改太多，无法判断效果来自哪里。
```

方案三适合在论文机制稳定后作为完整版本。

---

## 9. 最终建议

当前最推荐的路线是：

```text
方案一：立即做
  -> 去掉 score duplicate
  -> 只按 normalized code 判重
  -> 补 blocked_reason 日志
  -> 跑 TSP50/TSP100 对照

方案二：条件触发
  -> 如果 two-zone 仍阻塞父代池更新
  -> 改成 archive + active parent pool

方案三：长期论文版
  -> reward 修好后
  -> 再加 diversity active pool 和 stagnation recovery
```

最关键原则：

```text
不要把 score 当作算法身份。
不要在 admission 阶段因为同分拒绝不同代码。
先修最小真实问题，再逐步引入复杂机制。
```
