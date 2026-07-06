# EoH-RL 三机制历版设计与当前代码审计总结

> 目的：给后续新对话提供完整上下文。本文总结此前围绕 `llm4ad/method/eoh_rl` 后训练流水线讨论过、实现过、删除过的自适应算子权重、自适应坍缩机制、LoRA 稳定/回滚机制，并记录当前代码审计结论。

## 0. 当前代码状态

### 0.1 当前 `llm4ad/method/eoh_rl` 代码量

截至本次审计，`llm4ad/method/eoh_rl` 下 Python 文件共 **4430 行**：

| 文件 | 行数 | 作用 |
|---|---:|---|
| `__init__.py` | 7 | 包导出入口 |
| `args.py` | 208 | 运行参数默认值，把精简 YAML 补齐为 runtime 配置 |
| `config_runner.py` | 278 | 本地 EoH-RL 配置入口，负责 YAML、SFT LoRA、任务、GRPO policy、EoH 主循环接线 |
| `eoh_rl.py` | 711 | EoH-RL 主循环：初始化种群、构造 prompt、调用 GRPO、注册候选、保存 checkpoint |
| `population.py` | 145 | 种群维护与父代选择 |
| `profiler.py` | 187 | 日志、样本、种群、manifest 记录 |
| `prompt.py` | 192 | i1/e1/e2/m1/m2 提示词 |
| `rl/grpo_trainer.py` | 857 | Unsloth + TRL GRPO resident policy、奖励回调、在线训练 |
| `rl/sft_data.py` | 782 | SFT 数据清洗和代码校验 |
| `rl/sft_train.py` | 670 | SFT/LoRA 训练或加载 |
| `sampler.py` | 393 | 模型输出解析、代码抽取、strict/lenient parser |

### 0.2 当前真实实现的后训练链路

当前代码的实际链路是：

```text
config_runner.run_from_config
  -> apply_runtime_defaults
  -> run_sft_pipeline 加载或训练 SFT LoRA
  -> ResidentGRPOPolicy 加载 base/SFT adapter
  -> EoH 初始化种群
  -> 固定 EoH 算子 cycle 构造 prompt
  -> Unsloth + TRL GRPO 每轮训练一次
  -> reward callback 解析/评估 completion
  -> EoH 注册有效候选到单一种群
  -> checkpoint / metrics / token usage 记录
```

当前代码中 **没有真正启用** 以下旧机制：

| 机制 | 当前状态 |
|---|---|
| 自适应算子权重 | 独立 `operator_scheduler.py` 已删除；当前只使用固定 cycle：`e1 -> e2 -> m1 -> m2` |
| 自适应坍缩机制 | 独立 `collapse.py` 已删除；当前无 collapse/recovery 状态机 |
| LoRA gate / rollback | 独立 gate/rollback 逻辑已删除；当前 TRL 训练直接更新 resident adapter，未做接受/拒绝/回滚 |
| BQR reward.py | 独立 `reward.py` 已删除；当前奖励在 `rl/grpo_trainer.py` 内为 `four_state_v1` |

这意味着：如果当前实验名仍叫 `adaptive_op_plus_collapse` 或论文叙事仍声称启用了“三机制”，就是不成立的；当前版本只能称为 **EoH + SFT LoRA + resident GRPO 的极简后训练闭环**。

### 0.3 本次代码审计校验

已执行：

```bash
python -m py_compile $(find llm4ad/method/eoh_rl -type f -name '*.py' | sort) \
  example/run_eoh/tsp/run_eoh_local_rl.py \
  example/run_eoh/cvrp/run_eoh_local_rl.py \
  example/run_eoh/obp/run_eoh_local_rl.py \
  example/run_eoh/jssp/run_eoh_local_rl.py
```

结果：语法编译通过。

已执行：

```bash
DRY_RUN=1 RUNS_PER_VARIANT=1 bash scripts/eoh_local_rl/run_tsp100_eoh_rl.sh
```

结果：dry-run 能正确解析 TSP100、GPU、端口、SFT LoRA、pop_size、num_generations。

### 0.4 当前发现的真实问题和风险

#### 问题 1：当前种群判重把“同分数”当成重复个体，会削弱多样性

当前 `llm4ad/method/eoh_rl/population.py`：

```python
def _same_member(left, right):
    if str(left).strip() == str(right).strip():
        return True
    left_score = getattr(left, "score", None)
    right_score = getattr(right, "score", None)
    return left_score is not None and right_score is not None and float(left_score) == float(right_score)
```

影响：

1. 两个代码完全不同但分数相同的算法会被判为重复。
2. 组合优化任务的平均分数/离散评估分数很容易相同。
3. 初始化和后续进化会错误丢弃候选，种群多样性下降。
4. 这会加剧“长期卡在局部最优”的问题。

注意：原始 `llm4ad/method/eoh/population.py` 也有“同代码或同分数判重”的逻辑；但此前我们曾修复过 EoH-RL 版本，使其只按 exact-code 判重。若目标是性能和多样性，建议 EoH-RL 使用 exact-code 判重，而不要继承原始 EoH 的同分判重。

#### 问题 2：`args.py` 仍保留若干当前主链路未使用的字段

`args.py` 中存在：

```python
"trigger_every_n_generations": 1
"operator_cycle": ["e1", "e2", "m1", "m2"]
"anchor_baseline": "best"
```

当前实际：

1. `trigger_every_n_generations` 没有被 `EoH.run()` 使用。
2. `grpo.operator_cycle` 没有驱动主循环，主循环使用 `EoH._build_operator_cycle()`。
3. `anchor_baseline` 当前没有 LoRA gate/rollback，所以无实际作用。

这不是语法 bug，但属于“配置表面存在、机制实际不存在”的虚假语义风险。

#### 问题 3：`max_sample_nums/max_samples` 当前主要约束初始化，不约束 GRPO 总 rollouts

当前 `EoH._sample_initialize_population()` 使用：

```python
self._initial_sample_nums_max = self._max_sample_nums
```

但 `run()` 的 GRPO 阶段只检查：

```python
self._grpo_update_count < self._max_grpo_updates
```

影响：

1. 如果 `max_samples` 被理解为总采样预算，则当前实现不符合语义。
2. 如果模型初始化阶段输出质量差，可能消耗大量初始化样本才终止。
3. 如果只想训练 500 个 GRPO update，则当前行为可接受，但脚本里的 `init_sample_budget` 文案需要避免误导。

#### 问题 4：当前脚本名与旧脚本名不兼容

当前存在：

```text
scripts/eoh_local_rl/run_tsp50_eoh_rl.sh
scripts/eoh_local_rl/run_tsp100_eoh_rl.sh
scripts/eoh_local_rl/run_tsp200_eoh_rl.sh
```

旧的：

```text
run_tsp100_adaptive_op_plus_collapse.sh
run_tsp100_adaptive_op_plus_collapse_no_lora_rollback.sh
```

已经删除。若实验记录、命令文档或自动化脚本仍调用旧名字，会直接失败。

#### 问题 5：当前终端输出仍有脚本层 banner

当前 `_run_tsp_eoh_rl.sh` 真实运行也会打印：

```text
EoH-RL TSP run ...
GPU=...
task=...
config=...
```

这比原版 Python logger 多一层脚本输出。若要求“终端输出完全与原版一致”，应进一步只在 `DRY_RUN=1` 打印展开配置，真实运行直接进入 Python logger。

## 1. 机制总览

此前我们围绕三个机制反复迭代：

1. **自适应算子权重**：让更可能产出高质量算法的 EoH 算子被更频繁选择。
2. **自适应坍缩机制**：当种群长期局部最优、缺乏突破时，清空/重置部分种群，制造扰动和多样性。
3. **LoRA 稳定/回滚机制**：解决 SFT 和 GRPO 优化目标不一致导致的模型漂移，使输出稳定且性能不退化。

三者在论文叙事中应当是一个闭环：

```text
SFT LoRA 约束输出格式和初始质量
  -> GRPO 根据在线搜索奖励更新模型
  -> 自适应算子权重调整搜索方向
  -> LoRA 稳定机制防止模型过度偏离 SFT/已验证策略
  -> 自适应坍缩机制在长期局部最优时作为保底扰动
```

关键原则：

1. LoRA 机制不应只是“保守回滚”，而应保证结果稳定且优秀。
2. 自适应算子权重是主驱动力之一，不应因坍缩频繁触发而被重置/污染。
3. 坍缩机制是保底机制，不应前中期频繁触发。
4. 三个机制不能堆模块，应能用少量公式描述。

## 2. 自适应算子权重历版设计

### 2.1 CALM 参考版：Fixed / State / Adaptive 三策略

这部分来自原版项目文档 `/home/yuanyilun/projects/LLM4AD/docs/adaptive_weight_scheme_and_results.md`，是我们讨论自适应算子的参考背景。

CALM 算子集合：

```text
O = {simplify, injection, replacement, crossover}
```

基础权重：

```math
u = [1, 1, 2, 4]
```

归一化：

```math
p_i = \frac{u_i}{\sum_j u_j}
```

当种群不足 2 个个体时，`crossover` 不可行，概率置 0 后重新归一化。

#### 2.1.1 Fixed 策略

固定概率：

```math
p = \operatorname{Normalize}([1, 1, 2, 4])
```

种群未满时 injection 提权：

```math
u_{\text{inj}} \leftarrow \max_j u_j
```

示例：

```text
种群未满：simplify=0.111, injection=0.444, replacement=0.222, crossover=0.222
种群已满：simplify=0.125, injection=0.125, replacement=0.250, crossover=0.500
```

直觉：

1. 前期需要 injection 引入新组件。
2. 种群满后 crossover 更重要。
3. 简单、稳定，但不能适应不同任务/规模/训练阶段。

#### 2.1.2 State 策略：种群填充率驱动

定义填充率：

```math
r_t = \min(|P_t|, N) / N
```

其中：

```text
P_t = 当前种群
N = population_size
```

动态权重：

```math
u_{\text{inj},t} = u_{\text{inj}} \cdot (1 + \alpha(1-r_t))
```

```math
u_{\text{cross},t} = u_{\text{cross}} \cdot r_t^2
```

其余算子保持基础权重。

归一化：

```math
p_t = \operatorname{Normalize}(u_t)
```

默认：

```text
alpha = 3.0
```

直觉：

1. 种群空：更需要 injection，少用 crossover。
2. 种群满：crossover 恢复高概率。
3. 与坍缩机制自然耦合：坍缩后 `r_t` 下降，系统自动回到探索。

问题：

1. 如果坍缩触发太频繁，State 策略会长期偏向 injection。
2. 如果种群长期满但不突破，State 可能长期偏 crossover。
3. 它只看宏观状态，不看每个算子的真实收益。

#### 2.1.3 Adaptive 策略：历史奖励 EMA 驱动

每个算子维护 credit：

```math
c_{i,t} = (1-\lambda)c_{i,t-1} + \lambda \hat r_{i,t}
```

其中：

```text
c_{i,t} = 算子 i 的 credit
\hat r_{i,t} = 本步算子 i 产生候选的代表奖励，文档中取 max reward
\lambda = ema_decay = 0.1
```

中心化：

```math
\bar c_t = \frac{1}{|O|}\sum_i c_{i,t}
```

指数映射：

```math
\tilde p_{i,t} \propto p^{prior}_{i,t} \cdot \exp((c_{i,t}-\bar c_t)/\tau)
```

其中：

```text
tau = ema_tau = 0.5
```

PPO 风格 clip：

```math
\rho_{i,t} = \frac{\tilde p_{i,t}}{p_{i,t-1}}
```

```math
\rho'_{i,t} = \operatorname{clip}(\rho_{i,t}, 1-\epsilon, 1+\epsilon)
```

```math
p'_{i,t} \propto p_{i,t-1}\rho'_{i,t}
```

最低概率：

```math
p''_{i,t} = \max(p'_{i,t}, p_{\min})
```

再归一化。

默认：

```text
clip_eps = 0.3
min_op_prob = 0.05
```

优点：

1. 能发现当前任务/规模下更有用的算子。
2. 能自动偏向高收益算子。
3. 适合作为自适应算子权重的论文基线。

问题：

1. 正反馈循环：某算子被多选，偶尔得到正奖励，credit 上升，之后更常被选。
2. 小样本时容易被噪声误导。
3. 若奖励函数设计不好，credit 只是在学习噪声或退化模式。

### 2.2 EoH-RL 原版/重构早期版：EoH 算子固定 cycle

EoH 算子集合：

```text
O = {i1, e1, e2, m1, m2}
```

语义：

| 算子 | 语义 |
|---|---|
| i1 | 初始化，从任务描述和模板生成新算法 |
| e1 | 两父代交叉/启发，生成可能更好的算法 |
| e2 | 两父代交叉后更偏“抽取强主干并替换关键片段” |
| m1 | 单父代变异，引入新组件 |
| m2 | 单父代变异，替换弱片段或决策规则 |

早期固定 cycle：

```math
o_t = O[(t-1) \bmod |O|]
```

即：

```text
e1 -> e2 -> m1 -> m2 -> e1 -> ...
```

优点：

1. 简洁。
2. 避免自适应调度在奖励噪声下偏向错误算子。
3. 对齐 EoH 原始算子语义。

问题：

1. 无法利用算子历史效果。
2. 如果某个问题/规模上 m1 或 m2 更有效，固定 cycle 不能提高其概率。
3. 不能支持“自适应算子权重强于无自适应”的消融叙事。

当前代码就是这一类固定 cycle。

### 2.3 SGCA v1：信号融合自适应算子

这是后来从 CALM EMA-exp 思路转向 EoH-RL 算子语义后的一版设计，原版文件为：

```text
/home/yuanyilun/projects/LLM4AD/llm4ad/tools/rl/operator_scheduler.py
```

在线算子：

```text
O_online = {m1, m2, e1, e2}
```

恢复期算子：

```text
O_recovery = {i1, m1, m2, e1, e2}
```

正常 prior：

```math
p^{prior} =
\{m1:0.30,\ m2:0.45,\ e1:0.15,\ e2:0.10\}
```

直觉：

1. EoH-RL 的 m1/m2 更像局部规则替换/增强，早期更稳定。
2. e1/e2 双父代交叉更高风险，不能像 CALM crossover 那样默认很高。
3. recovery 阶段优先使用 m1/m2，从保留核心附近重新扩展。

#### 2.3.1 算子信号

对某个算子 `o` 在某步产生的事件集合 `E_o`：

```text
N_o = |E_o|
F_o = frontier improvement count
P_o = parent improvement count
B_o = failure / blocked count
V_o = structure violation count
```

性能项：

```math
\operatorname{perf}_o =
\frac{2F_o + P_o - B_o}{N_o}
```

结构保真项：

```math
\operatorname{fid}_o =
1 - \frac{V_o}{N_o}
```

clip 到 `[0,1]`：

```math
\operatorname{fid}_o = \operatorname{clip}(\operatorname{fid}_o,0,1)
```

最终信号：

```math
s_o = \operatorname{perf}_o \cdot \operatorname{fid}_o
```

含义：

1. `frontier improvement` 权重最高，系数 2。
2. `parent improvement` 次之，系数 1。
3. 失败/被拦截给负贡献。
4. 随机、泄漏、复制父代、无效代码等结构违规会乘法压低信号。

#### 2.3.2 running average credit

这一版不是 EMA，而是样本均值式 running average：

```math
c_{o,n+1}=c_{o,n}+\frac{s_o-c_{o,n}}{n+1}
```

其中 `n` 是该算子已有有效信号次数。

优点：

1. 比 EMA 稳定，不会被最近一次异常奖励大幅带偏。
2. 适合事件少、噪声大的在线后训练。

问题：

1. 适应速度慢。
2. 早期如果信号错误，后续修正也慢。

#### 2.3.3 learned distribution

对可行算子集合 `A_t`：

```math
q_{o,t} =
\frac{\exp(c_{o,t}-\max_{j\in A_t}c_{j,t})}
{\sum_{j\in A_t}\exp(c_{j,t}-\max_{k\in A_t}c_{k,t})}
```

#### 2.3.4 prior 与 learned 融合

定义观测强度：

```math
\alpha_t =
\frac{\sum_{o\in A_t} n_o}{\sum_{o\in A_t} n_o + |A_t|}
```

最终概率：

```math
p_{o,t} =
(1-\alpha_t)p^{prior}_{o,t}+\alpha_t q_{o,t}
```

恢复期使用更保守的：

```math
\alpha^{rec}_t =
\frac{\sum_{o\in A_t} n_o}{\sum_{o\in A_t} n_o + 2|A_t|}
```

#### 2.3.5 最大概率上限

为避免单一算子垄断：

```math
p_{o,t} \le p_{\max}
```

默认：

```text
p_max = 0.60
```

如果某算子超过上限，固定到上限，剩余概率按其他算子的相对比例重新分配。

#### 2.3.6 这一版的问题

1. 逻辑比 CALM 复杂很多。
2. 依赖 reward events 的真实性；如果 `registered_to_population`、`blocked_from_population` 或 `validity` 更新时机不对，credit 会被污染。
3. 如果整体模型退化，所有算子都产出低质量事件，自适应调度无法拯救模型。
4. 当前极简代码已删除这一机制，因此不能再声称使用 SGCA。

### 2.4 “修复版”算子信用更新

这一版主要修复前一版的统计时机问题。

问题背景：

1. GRPO reward callback 先产生 reward events。
2. EoH 主循环再把候选注册到种群。
3. 如果在注册前就更新算子 credit，则不知道候选是否真的进入种群、是否 survived。

修复思想：

```text
先完成候选注册 -> 回填 event 中的 registered_to_population / survived_main_population / blocked_from_population -> 再更新算子 credit
```

修复后的事件字段包括：

```text
exec_success
validity
beats_parent
beats_frontier
registered_to_population
survived_main_population
blocked_from_population
random_algo
score_metadata_leak
exact_parent_copy
```

修复后的算子信号应以“最终进入搜索状态的证据”为主：

```math
s_o =
\frac{
2F_o + P_o + \eta R_o + \xi S_o - B_o
}{N_o}
\cdot \operatorname{fid}_o
```

其中：

```text
F_o = beats_frontier count
P_o = beats_parent count
R_o = registered_to_population count
S_o = survived_main_population count
B_o = blocked / invalid / random / leak / copy count
```

我们后来倾向于谨慎使用 `R_o`，因为早期种群未满时“注册成功”太容易，不能等同于真正搜索证据。

建议保守取：

```math
s_o =
\frac{2F_o + P_o - B_o}{N_o}\cdot \operatorname{fid}_o
```

只把 `registered_to_population/survived_main_population` 用作诊断指标，不作为强正信用。

## 3. 自适应坍缩机制历版设计

### 3.1 CALM 参考版：固定 stuck 阈值 + 概率触发

CALM 坍缩机制大意：

```text
age_stuck = 连续无改进步数
```

触发条件：

```math
age\_stuck > \frac{max\_steps}{20}
```

或以概率触发：

```math
P(\text{collapse}) = speed\_collapse \cdot age\_stuck
```

其中文档记录：

```text
max_steps=500 -> threshold=25
speed_collapse=0.0005
```

执行：

```text
清空种群，仅保留当前最佳算法 + seed 算法
```

优点：

1. 极简。
2. 很容易论文描述。
3. 在 CALM 中有效，因为它的 seed 质量很高，且每次突破幅度本来小。

为什么不能直接照搬到本项目：

1. 本项目初始化不是固定高质量 seed，而是 SFT 模型从 0 生成。
2. 前期很多潜力算法可能还没完全发展，25 步固定阈值容易过早清掉。
3. 本项目还有自适应算子和 LoRA 更新，坍缩过频会扰乱这两个机制。

### 3.2 早期 EoH-RL 坍缩版：复杂条件 + recovery

早期设计试图避免 CALM 式频繁坍缩，条件包含：

```text
population_full
not in_recovery
mature
cooldown_ready
long_stagnation
local_progress_stalled
diversity_low
```

形式上：

```math
C_t =
\mathbf{1}[
full_t
\land \neg recovery_t
\land mature_t
\land cooldown_t
\land stagnation_t
\land local\_stall_t
\land low\_diversity_t
]
```

问题：

1. 超参数多，论文难讲。
2. 不同日志里触发频率不稳定。
3. 前期可能过早触发，清掉有潜力算法。
4. 后期如果条件过严，又长期不触发。
5. 坍缩只动种群，不修复模型权重退化。

### 3.3 最近突破间隔窗口版：自适应阈值

这是后来讨论中更认可的一版，原版文件中也能看到类似实现：

```text
/home/yuanyilun/projects/LLM4AD/llm4ad/tools/rl/collapse.py
```

核心思想：

> 不用固定 25 代，而是用当前运行自身的历史突破节奏定义“长期停滞”。

记录最近全局 best 改进的间隔：

```math
I_k = t_k - t_{k-1}
```

窗口大小：

```text
W = 5
```

最近窗口：

```math
\mathcal I_t = \{I_{k-W+1},...,I_k\}
```

只有当窗口满时才允许坍缩：

```math
|\mathcal I_t| \ge W
```

参考停滞长度：

```math
G_t = \max(\mathcal I_t)
```

自适应阈值：

```math
T_t = \gamma G_t
```

默认：

```text
gamma = 2.0
```

当前无突破长度：

```math
A_t = t - t_{last\_best}
```

触发：

```math
C_t =
\mathbf{1}[
|P_t|=N
\land \neg recovery_t
\land |\mathcal I_t|\ge W
\land A_t > T_t
\land \neg collapsed\_in\_current\_plateau
]
```

性质：

1. 前期没有足够突破历史，不会触发。
2. 如果前期突破很快，则阈值小，但仍需要窗口满。
3. 如果历史突破间隔大，后期阈值自动变大，避免误触发。
4. 每个平台期最多触发一次，避免连续坍缩。

执行后：

```text
collapse_count += 1
recovery_active = true
recovery_bootstrap_pending = true
collapsed_in_current_plateau = true
```

优点：

1. 超参数少：窗口 `W` 和倍率 `gamma`。
2. 公式清晰，适合论文描述。
3. 更符合“坍缩是保底机制”的定位。

问题：

1. 如果前期一直没有突破，窗口永远不满，坍缩不会触发。
2. 如果 LoRA/GRPO 已经把模型训练坏了，坍缩种群也没用。
3. recovery 如何恢复种群很关键，不能长期停留在 2-3 个个体。

### 3.4 我们后续讨论的论文友好版：前中期保护 + 后期保底扰动

用户明确要求：

1. 前中期不要轻易触发。
2. 中后期长期局部最优时要触发。
3. 触发太频繁会清掉潜力算法，扰乱自适应算子和 LoRA。
4. 不要固定 25 代，否则和 CALM 区别不明显。
5. 公式简单，少超参数，论文好写。

因此推荐形式：

```math
C_t =
\mathbf{1}[
H_t
\land Full_t
\land A_t > \gamma \cdot Q_{q}(\mathcal I_t)
\land D_t < \delta_t
]
```

其中：

```text
H_t = 历史突破信息是否足够，例如 |I_t| >= W
Full_t = 种群已满
A_t = 当前距离上次全局 best 的无突破长度
Q_q(I_t) = 最近突破间隔的高分位数，例如 max 或 0.8 quantile
D_t = 最近候选结构/行为多样性
delta_t = 当前搜索阶段的多样性下界
```

为了简洁，最终可去掉显式多样性项，仅用历史突破间隔：

```math
C_t =
\mathbf{1}[
|P_t|=N
\land \neg recovery_t
\land |\mathcal I_t|\ge W
\land A_t > \gamma \max(\mathcal I_t)
]
```

触发后建议：

1. 单一种群，不维护三套种群。
2. 保留当前 best 和少量结构差异最大的 elite。
3. 清空 next_generation。
4. recovery 阶段优先 m1/m2 或 i1+m1/m2，快速补满到 `pop_size`。
5. recovery 完成前不再次触发坍缩。

保留核心可定义为：

```math
R = \{x^*\} \cup \operatorname{TopK}_{x\in P_t\setminus x^*} d(x,x^*)
```

其中：

```text
x^* = 当前 best
d(x,x*) = 结构距离，可用 AST token Jaccard distance 或代码规范化字符串距离
```

如果为了代码简洁，也可只保留：

```text
best + second-best-different-code
```

但这会牺牲“多样性扰动”的理论表达。

## 4. LoRA 稳定/回滚机制历版设计

### 4.1 版本 0：无 LoRA gate，GRPO 直接更新

流程：

```text
SFT LoRA 或 base model
  -> 每轮 GRPO trainer.train()
  -> adapter in memory 更新
  -> 下一轮直接用更新后的 adapter 推理和训练
```

公式：

```math
\theta_{t+1} = \theta_t + \Delta\theta^{GRPO}_t
```

优点：

1. 简单。
2. 快。
3. 与 Unsloth + TRL 默认使用方式一致。

问题：

1. SFT 和 GRPO 优化目标不一致时，模型容易偏离 SFT。
2. 如果奖励有噪声或负向设计，模型会持续退化。
3. 无法解释“稳定性创新点”。

当前极简代码接近这一版。

### 4.2 版本 1：单步搜索证据 Gate

早期 gate 的接受逻辑大意：

```text
no_valid_completion = valid_count <= 0 or candidate_count <= 0
has_search_evidence = frontier > 0 or parent > 0 or registered > 0 or positive > 0
accepted = (not no_valid_completion) and has_search_evidence
```

数学形式：

```math
A_t =
\mathbf{1}[
V_t>0
\land (F_t>0 \lor P_t>0 \lor R_t>0 \lor Pos_t>0)
]
```

其中：

```text
V_t = valid completion count
F_t = frontier improvement count
P_t = parent improvement count
R_t = registered_to_population count
Pos_t = positive reward count
```

接受则：

```math
\theta_t \leftarrow \theta'_t
```

拒绝则：

```math
\theta_t \leftarrow \theta_{t-1}
```

问题：

1. `registered_to_population > 0` 太宽松，早期种群未满时有效候选很容易入池。
2. positive reward 受奖励函数影响，如果 q3/q2 设计不合理，会误判。
3. 单步证据不能发现长期退化。
4. 一次偶然 q1 就可能接受有害 LoRA。

### 4.3 版本 2：全局改进 + 保真度 Gate

后来原版 backend 中出现更严格的接受逻辑：

```python
improved = best_completion_score better than best_before_rl
fidelity = 1 - violation / total
accept = improved and fidelity >= 0.75
```

数学形式：

定义 utility：

```math
U(s)=
\begin{cases}
-s, & minimize\\
s, & maximize
\end{cases}
```

全局改进：

```math
\Delta_t = U(s^{best}_{completion,t}) - U(s^{best}_{before,t})
```

结构违规数：

```math
Vio_t =
\#\{x: blocked \lor invalid \lor random \lor leak \lor exact\_copy\}
```

保真度：

```math
\phi_t = 1 - \frac{Vio_t}{N_t}
```

接受：

```math
A_t = \mathbf{1}[\Delta_t > 0 \land \phi_t \ge 0.75]
```

拒绝时：

```text
discard population updates
restore reward state
restore search state
restore trainable LoRA tensors
or reload deployed policy
```

优点：

1. `registered_to_population` 不再作为强证据。
2. 只有真正突破全局 best 才接受。
3. 保真度防止格式/随机/泄漏/复制污染策略。

问题：

1. 过于严格时，模型长期不更新，RL 作用弱。
2. 如果每次更新都要保存/回滚 LoRA，时间开销明显。
3. 只看单步全局 best，无法表达“稳定但暂时未突破”的有效学习。
4. 如果 reward 和种群注册有 bug，接受/拒绝仍会被误导。

### 4.4 版本 3：SFT Anchor Trust Region 理论版

这一版是为了论文创新点讨论提出的：LoRA 机制不是简单回滚，而是把 SFT adapter 作为 anchor，构造后训练的信任域。

动机：

1. SFT 目标：模仿高质量启发式代码和格式。
2. GRPO 目标：最大化在线 reward。
3. 两者目标不完全一致，GRPO 会推动模型偏离 SFT。
4. LoRA 参数提供了低维可控子空间，可用 adapter 距离近似策略漂移。

定义：

```text
\theta_0 = SFT LoRA 参数
\theta_t = 当前 LoRA 参数
\theta'_t = 本轮 GRPO 更新后的候选 LoRA
```

LoRA drift：

```math
D_t =
\frac{\|\theta'_t-\theta_0\|_F}{\|\theta_0\|_F+\epsilon}
```

或相对上一 accepted policy：

```math
D^{step}_t =
\frac{\|\theta'_t-\theta^{acc}_{t-1}\|_F}{\|\theta^{acc}_{t-1}\|_F+\epsilon}
```

性能证据：

```math
E_t =
\alpha \cdot \mathbf{1}[\Delta^{frontier}_t>0]
+ \beta \cdot \mathbf{1}[\Delta^{parent}_t>0]
+ \gamma \cdot valid\_rate_t
- \lambda \cdot violation\_rate_t
```

接受：

```math
A_t =
\mathbf{1}[
E_t \ge \tau_E
\land D_t \le \rho_t
]
```

其中信任域半径可自适应：

```math
\rho_{t+1} =
\begin{cases}
\min(\rho_{max}, \rho_t(1+\eta)), & A_t=1 \land \Delta_t>0\\
\max(\rho_{min}, \rho_t(1-\eta)), & A_t=0 \text{ or violation high}
\end{cases}
```

论文叙事：

```text
LoRA gate 是 SFT-anchored trust-region policy update。
它不是额外堆模块，而是约束 GRPO 在 SFT 学到的启发式代码分布附近搜索。
```

优点：

1. 有数学支撑。
2. 可解释 SFT/RL 目标不一致。
3. 稳定性可通过 drift、accept rate、跨 run 方差做消融。

问题：

1. 实现若过重会影响训练时间。
2. 距离阈值需要调参；超参数太多会影响论文消融。
3. 如果性能证据设计过严，会困住模型；过松又会漂移。

### 4.5 版本 4：当前极简版

当前代码：

```text
无 gate
无 rollback
无 accepted/rejected LoRA checkpoint
无 SFT anchor drift 统计
```

GRPO 更新后直接：

```python
self.prepare_for_inference(force_reload=True)
```

本质：

```math
\theta_{t+1}=\theta'_t
```

这适合先验证后训练主链路是否正常，但不能支撑 LoRA 稳定机制的论文创新点。

## 5. 奖励机制相关版本

虽然用户要求重点总结三机制，但奖励机制直接决定三机制是否有效，因此简要记录。

### 5.1 BQR q1-q4 版本

象限：

| 状态 | 条件 | 奖励 |
|---|---|---|
| q1 | 超越父代/前沿 | 大正奖励 |
| q2 | 接近父代且结构新颖 | 中性/小正 |
| q3 | 有效但性能更差 | 小负或接近 0 |
| q4 | 无效代码 | 大负 |

问题：

1. q3 常见且为负，会惩罚探索。
2. GRPO 组内归一化会在全负样本中仍给“相对不差”的 completion 正 advantage。
3. 模型可能学习“最不差但不突破”的模式，最终退化。

### 5.2 当前 four_state_v1

当前 `rl/grpo_trainer.py` 中：

```math
\Delta_p = U(s)-U(s_{parent})
```

```math
\Delta_f = U(s)-U(s_{frontier})
```

若突破前沿：

```math
r = 2 + \operatorname{clip}_{[0,1]}\left(\frac{\Delta_f}{|U(s_{frontier})|+\epsilon}\right)
```

若突破父代：

```math
r = 1 + \operatorname{clip}_{[0,1]}\left(\frac{\Delta_p}{|U(s_{parent})|+\epsilon}\right)
```

否则：

```math
r = infeasible\_reward
```

默认：

```text
no_code_reward = -2.0
infeasible_reward = -1.5
```

优点：

1. 逻辑简洁。
2. 强调真正突破。
3. 不再维护复杂 q2/q3/q4。

风险：

1. 有效但不突破也给负奖励，仍可能惩罚探索。
2. 如果每组 completion 都没有突破，GRPO 仍可能学习相对噪声。
3. 对组合优化“长时间局部精修”的阶段不友好。

## 6. 父代选择与种群维护

### 6.1 父代选择

当前选择：

```python
return [self._population.selection() for _ in range(count)]
```

`Population.selection()` 使用 rank-based probability：

```math
p_r \propto \frac{1}{r+N}
```

其中 `r=0` 是最优个体。

优点：

1. 保留最优偏好。
2. 仍有概率选到较差个体，维持探索。

问题：

1. e1/e2 可能两次抽到同一个父代。
2. 如果种群判重过严或坍缩后种群太小，父代多样性不足。
3. 如果提示词偏“局部精修”，父代选择再偏 best，会加剧精修化。

### 6.2 种群维护

当前是两区结构：

```text
active population: 父代池
next_generation: 新候选缓冲
```

当 `next_generation` 满 `pop_size`：

```math
P_{t+1} = \operatorname{TopN}(P_t \cup Q_t)
```

其中：

```text
Q_t = next_generation
N = pop_size
```

然后清空 `Q_t`，generation 加 1。

优点：

1. 父代选择不会被本轮新候选即时污染。
2. 简洁。

问题：

1. 如果每轮只有少量有效候选，survival 很久不触发。
2. 如果判重把同分算法丢弃，`next_generation` 更难满。
3. 如果没有坍缩机制，中后期可能长期围绕少数模式精修。

## 7. 适合下一轮设计的建议基线

如果后续新对话继续设计，建议先不要恢复复杂旧代码，而是从当前 4430 行极简版本上增加少量必要机制。

### 7.1 自适应算子建议

建议使用简化 SGCA：

```math
s_o = \frac{2F_o + P_o - B_o}{N_o}\cdot (1-\frac{V_o}{N_o})
```

```math
c_{o,t}=(1-\lambda)c_{o,t-1}+\lambda s_o
```

```math
p_{o,t}\propto p^{prior}_o \exp(c_{o,t}/\tau)
```

并加入：

```math
p_{o,t}\le 0.6,\quad p_{o,t}\ge p_{min}
```

但要注意：

1. 只用注册后的真实事件更新。
2. 不把单纯 `registered_to_population` 当强正证据。
3. 先修复种群判重，避免 credit 受错误 blocked 污染。

### 7.2 坍缩建议

建议采用最近突破间隔窗口：

```math
C_t =
\mathbf{1}[
|P_t|=N
\land \neg recovery_t
\land |\mathcal I_t|\ge 5
\land A_t > 2\max(\mathcal I_t)
]
```

触发后：

```text
保留 best + 结构差异最大 elite
清空 next_generation
进入 recovery
recovery 补满 pop_size 后退出
```

不要使用固定 25 代阈值。

### 7.3 LoRA 稳定建议

建议从“全局改进 + drift”开始，而不是恢复复杂旧 gate：

```math
D_t =
\frac{\|\theta'_t-\theta^{acc}_{t-1}\|_F}{\|\theta^{acc}_{t-1}\|_F+\epsilon}
```

```math
\Delta_t = U(s^{best}_{after})-U(s^{best}_{before})
```

```math
\phi_t = 1-\frac{Vio_t}{N_t}
```

接受：

```math
A_t=\mathbf{1}[
\Delta_t>0
\land \phi_t\ge \phi_{min}
\land D_t\le\rho
]
```

如果要避免过严，可以允许“小步稳定更新”：

```math
A_t=\mathbf{1}[
(\Delta_t>0)
\lor
(\phi_t\ge\phi_{min}\land valid\_rate_t\ge v_{min}\land D_t\le\rho_{small})
]
```

但论文叙事必须明确：

```text
LoRA 机制目标不是保守不训练，而是在 SFT 锚点附近允许有证据的小步策略改进。
```

### 7.4 必须先修复/确认的实现点

1. 种群判重应改成 exact-code duplicate，不应按同分判重。
2. 当前配置里的无效字段要删掉或真正接入，避免虚假机制。
3. 如果恢复三机制，脚本名、日志名、run_id 必须真实反映开关。
4. 任何 gate/collapse/operator 事件都必须写入 metrics，便于消融。
5. 三机制必须能关闭，做消融：
   - no adaptive operator
   - no collapse
   - no lora stability
   - all enabled
6. 完整 500 代时间目标应保持约 5 小时，不能用大量保存/回滚牺牲速度。

## 8. 当前版本不能声称的内容

当前代码不能声称：

1. 使用了自适应算子权重。
2. 使用了自适应坍缩机制。
3. 使用了 LoRA 回滚或 SFT-anchored trust region。
4. 支持 `adaptive_op_plus_collapse_no_lora_rollback` 这类消融脚本。
5. `grpo.operator_cycle` 配置实际控制了算子调度。

当前可以声称：

1. 使用 SFT LoRA 初始化本地模型。
2. 使用 Unsloth + TRL GRPO resident 训练。
3. 把 GRPO 作为组件嵌入 EoH 主循环。
4. 使用固定 EoH 算子语义 `e1/e2/m1/m2`。
5. 使用单一种群 + next_generation survival。
6. 记录 token usage、timing、reward events、candidate registration。

## 9. 给新对话的最短上下文

可以把下面这段直接交给新对话：

```text
本项目要做 EoH-RL 后训练：加载 SFT LoRA，用 Unsloth+TRL GRPO 更新本地模型，把 GRPO 作为组件嵌入 EoH 遗传进化搜索。目标是 TSP/CVRP/OBP/JSSP 跨问题跨规模稳定提升，500 代约 5 小时。

核心机制想设计三项：
1. 自适应算子权重：根据 e1/e2/m1/m2 的真实搜索证据调整概率，避免固定 cycle。
2. 自适应坍缩：只在长期局部最优时触发，前中期不轻易触发；清空/扰动种群以恢复多样性。
3. LoRA 稳定机制：因为 SFT 与 GRPO 优化目标不一致，用 SFT/accepted LoRA 作为 trust-region anchor，防止模型输出偏移和性能退化。

当前代码已经极简化到 4430 行，但三机制实际上都未启用：没有 operator scheduler，没有 collapse，没有 LoRA rollback。当前只有固定 e1/e2/m1/m2 cycle + four_state_v1 reward + resident GRPO + 单一种群。

当前发现的真实实现风险：Population 判重把“同分数”也当重复，会丢掉不同代码但同分算法，削弱多样性；args.py 仍有 operator_cycle/anchor_baseline 等未使用字段；max_samples 主要限制初始化，不限制总 GRPO rollouts。

下一步应该先修复种群判重和配置虚假字段，再用极简公式重新实现三机制：自适应算子用注册后事件的信用；坍缩用最近突破间隔窗口；LoRA 用全局改进+保真度+adapter drift 的 trust-region gate。
```
