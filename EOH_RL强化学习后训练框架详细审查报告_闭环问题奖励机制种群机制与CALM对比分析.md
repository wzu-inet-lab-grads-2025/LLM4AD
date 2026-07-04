# EOH_RL强化学习后训练框架详细审查报告：闭环问题、奖励机制、种群机制与CALM对比分析

## 一、报告目的

本文档用于系统审查 `llm4ad/method/eoh_rl` 当前这套基于遗传进化搜索算法的强化学习后训练框架，重点回答以下问题：

1. 为什么当前强化学习训练效果显著差于纯本地推理进化。
2. 当前问题是否本质上来自强化学习闭环设计失真。
3. 你设计的三个机制，即 `LoRA稳定机制`、`自适应算子权重机制`、`自适应坍缩机制`，目前分别卡在什么地方。
4. 当前框架与 `CALM` 相比，核心差距在哪里。
5. 如果目标是做出一套像 `CALM` 一样干净、整洁、高效、可跨问题跨规模泛化的后训练框架，下一步应该怎么改。

---

## 二、审查范围

本次审查覆盖以下内容：

### 1. 被审查框架代码

- `llm4ad/method/eoh_rl/eoh_rl.py`
- `llm4ad/method/eoh_rl/population.py`
- `llm4ad/method/eoh_rl/sampler.py`
- `llm4ad/method/eoh_rl/prompt.py`
- `llm4ad/method/eoh_rl/config_runner.py`
- `llm4ad/method/eoh_rl/rl/grpo_trainer.py`
- `llm4ad/method/eoh_rl/rl/reward.py`
- `llm4ad/method/eoh_rl/rl/lora_gate.py`
- `llm4ad/method/eoh_rl/rl/operator_scheduler.py`
- `llm4ad/method/eoh_rl/rl/collapse.py`

### 2. 对照框架代码

- 原始 `EoH`：`llm4ad/method/eoh/eoh.py`
- 原始 `EoH` 种群：`llm4ad/method/eoh/population.py`
- `CALM`：`/home/yuanyilun/projects/source/CALM/main.py`
- `CALM`：`/home/yuanyilun/projects/source/CALM/calm_trainer.py`

### 3. 配置与日志

- 配置：`configs/run_eoh_local_rl/eoh_local_rl_tsp.yaml`
- 日志目录：`logs/TSP/eoh_local_rl`

---

## 三、最终结论摘要

### 结论一：当前问题本质上是强化学习闭环失真，不是单纯的超参数问题

当前 `eoh_rl` 不是一个“策略更新后，下一轮搜索能力随之提升”的标准在线后训练闭环，而是一个：

- 一边搜索
- 一边训练
- 一边把当前 batch 候选直接写入种群
- 再根据这同一批样本是否“看起来有证据”来决定 LoRA 要不要部署

这导致：

1. `搜索状态` 和 `策略状态` 没有被干净地区分。
2. `奖励优化目标`、`入池选择目标`、`LoRA接受目标` 三者不一致。
3. 强化学习不是在学“如何长期提升未来搜索表现”，而是在学“如何在当前 rollout 上拿一点局部正反馈”。

因此，即使训练 loss 正常下降，也完全可能对真实搜索没有帮助，甚至比纯本地推理进化更差。

### 结论二：你设计的三个机制里，思路都不算错，但当前实现方式都没有落在正确位置

- `LoRA稳定机制`：目前不是“全局稳定器”，只是“当前 rollout 批次的局部门控器”。
- `自适应算子权重机制`：当前主要在吸收解析失败噪声和稀疏奖励噪声，不是在学算子真实贡献。
- `自适应坍缩机制`：当前更容易把系统推入“恢复期长期补不满种群”的半死锁状态，而不是高效跳出局部最优。

### 结论三：目前框架差于 CALM 的关键，不在于你没有足够复杂，而在于你比 CALM 更复杂，但闭环更不干净

`CALM` 的训练循环虽然简单，但它的核心闭环非常直接：

- 构造 prompt
- 采样候选
- 评估候选
- 更新搜索状态
- 再进入下一轮 prompt

而你当前的 `eoh_rl` 在一个主循环里同时塞进了：

- prompt 生成
- GRPO rollout
- reward 计算
- 入池更新
- LoRA gate
- 策略恢复
- 算子调度
- 坍缩恢复
- checkpoint 恢复

这使得系统表面上更强，实际上更难确认每个环节到底有没有在为搜索服务。

---

## 四、当前框架运行流程还原

根据 `llm4ad/method/eoh_rl/config_runner.py`、`llm4ad/method/eoh_rl/eoh_rl.py`、`llm4ad/method/eoh_rl/rl/grpo_trainer.py`，当前主流程大致如下。

### 第一阶段：加载或训练初始 SFT LoRA

入口在：`llm4ad/method/eoh_rl/config_runner.py:95-118`

流程为：

1. 从 YAML 读取本地基础模型路径。
2. 加载已有 SFT LoRA，或者重新跑 SFT。
3. 把该 LoRA 作为 resident policy 的初始化权重。

### 第二阶段：初始化种群

初始化逻辑在：`llm4ad/method/eoh_rl/eoh_rl.py:1124-1156`

流程为：

1. 用 `i1` prompt 连续采样。
2. 走 `strict_contract=False` 的宽松解析路径。
3. 所有通过评估且有算法描述的候选进入 `_initial_candidates`。
4. 最后用 `_select_score_diverse` 做一次性选优，填满初始 population。

这里的关键点是：

- 初始化阶段的解析契约是宽松的。
- 在线 RL 阶段的解析契约是严格的。

这两个阶段并不一致。

### 第三阶段：在线 RL 主循环

入口在：`llm4ad/method/eoh_rl/eoh_rl.py:1158-1177`

每一轮调用 `_run_grpo_update()`，其中主要流程在：`llm4ad/method/eoh_rl/eoh_rl.py:693-802`

每次 update 做的事情是：

1. 从当前 population 里采样父代，构造在线 prompt。
2. 调用 resident policy 运行一次 `train_once()`。
3. 在 reward callback 中解析 completion、评估代码、计算 reward。
4. 直接把“有效候选”提交给 EoH 主循环入池。
5. 再根据这一批 rollout summary 做 LoRA gate 决策。
6. 更新算子调度器、坍缩管理器、日志和 checkpoint。

这个设计最大的问题是：

### 当前 batch 的训练、评估、入池、LoRA部署判定，是耦合在同一批 rollout 上完成的

这会直接破坏“训练后的策略是否真的改进了下一轮搜索”的可验证性。

---

## 五、与原始 EoH 的核心差异

### 原始 EoH 的结构很简单

参考：

- `llm4ad/method/eoh/eoh.py:169-202`
- `llm4ad/method/eoh/eoh.py:214-254`
- `llm4ad/method/eoh/population.py:42-86`

原始 `EoH` 的基本逻辑是：

1. 用某个算子生成 1 个候选。
2. 评估它。
3. 暂存到 `_next_gen_pop`。
4. 当 `_next_gen_pop` 积累到一定数量后做 survival。

它的好处是：

- 搜索状态更新规则很单纯。
- 算子作用可以被比较清楚地观察。
- 没有策略权重更新带来的额外噪声。

### 当前 EOH_RL 的复杂度显著提高，但代价是闭环不再透明

你在 `eoh_rl` 中引入了：

- batch rollout
- strict parser
- BQR 奖励
- LoRA gate
- adaptive operator scheduler
- collapse manager
- checkpoint restore
- resident model lifecycle

这些模块每一个单看都“有道理”，但组合后形成了一个高耦合系统，一旦效果差，很难判断究竟是：

- prompt 出问题
- parser 出问题
- reward 出问题
- 入池规则出问题
- LoRA gate 出问题
- 算子调度出问题
- collapse 出问题

这是当前框架的根本工程风险。

---

## 六、最关键的问题一：强化学习闭环没有真正闭合

### 1. 证据位置

- 训练主逻辑：`llm4ad/method/eoh_rl/rl/grpo_trainer.py:475-518`
- reward callback 提交候选：`llm4ad/method/eoh_rl/rl/grpo_trainer.py:276-296`
- EoH 主循环接收候选：`llm4ad/method/eoh_rl/eoh_rl.py:934-1013`

### 2. 当前闭环实际发生了什么

每次 update：

1. resident policy 基于当前 prompt 生成 completion。
2. completion 被 reward callback 评估。
3. 有些 completion 在 `commit_candidates()` 中直接提交到 population。
4. 然后 LoRA gate 再决定本轮训练后的 adapter 是否接受。

也就是说：

### 候选入池发生在 LoRA 是否部署之前

这会出现一种逻辑不一致：

- 如果候选样本推动了 population 改变，但 LoRA 最终被回滚，那么搜索状态已经变了，策略状态却没有真正变成对应的那一版。
- 如果 LoRA 被接受，它的接受依据也是“训练时这批样本”的统计，而不是“训练后新策略在 fresh prompt 上的真实表现”。

这使得你无法回答一个最关键的问题：

### 这次 RL 更新，究竟是因为策略真的更好了，还是只是同一批 rollout 里偶然出了几个好样本？

严格来说，这不是一个合格的后训练闭环。

### 3. 为什么这会导致 RL 比纯推理进化更差

纯推理进化的逻辑是：

- 当前模型直接生成候选
- 候选靠搜索规则生死存亡

而你这里额外加入了：

- 训练噪声
- LoRA 部署噪声
- gate 决策噪声
- 策略回滚噪声

如果这些额外模块不能稳定地产生“未来更好的候选分布”，那它们只会消耗搜索预算和破坏搜索稳定性。

---

## 七、最关键的问题二：奖励目标、入池目标、LoRA门控目标三者不一致

### 1. 奖励函数的结构

参考：`llm4ad/method/eoh_rl/rl/reward.py:34-137`

当前 `BQR` 的四类行为是：

1. `q1`：优于 parent，或者在某些情况下优于 frontier，给强正奖励。
2. `q2`：和 parent 打平，但结构新颖，给小正奖励。
3. `q3`：没有提升，但只给很轻的负奖励。
4. `q4`：无效输出，给 `-1.0`。

### 2. 入池规则的结构

参考：`llm4ad/method/eoh_rl/population.py:61-75`

population 的选择依据不是 reward，而是：

- 分数是否有限
- 代码是否重复
- 排序后能不能进入 top-k

也就是说：

### RL 在优化 reward，搜索在优化 raw score 排名

这两者不是一回事。

### 3. LoRA gate 的结构

参考：`llm4ad/method/eoh_rl/rl/lora_gate.py:42-127`

LoRA gate 并不只认 frontier 提升，它会把以下信号都视作“search evidence”：

- `frontier_improve_count`
- `parent_improve_count`
- `registered_count`
- `positive_reward_count`

这意味着：

### 只要 reward 正，哪怕只是 q2 平局创新，LoRA gate 也可能被误导

### 4. 日志证据：Q2 正奖励并没有带来搜索进步

典型证据文件：

- `logs/TSP/eoh_local_rl/adaptive_op_plus_collapse_no_lora_rollback_np1_TSP100_run2_20260704_020111/rl_training/rl_update_129/metrics.json`

该轮的关键信息：

- `valid_count = 4`
- `positive_reward_count = 4`
- 全部是 `bqr_q2_parent_tie_novelty`
- `registered_count = 0`
- `best_after = best_before`

也就是说：

### 模型在这一轮被明确鼓励去生成“结构上新颖但分数完全不涨”的代码

从搜索角度看，这些样本完全没有贡献；从 RL 角度看，它们却是正例。这是严重目标错位。

### 5. 日志证据：极差候选只受到很弱惩罚

典型证据文件：

- `logs/TSP/eoh_local_rl/adaptive_op_plus_collapse_no_lora_rollback_np1_TSP100_run1_20260702_123929/rl_training/rl_update_020/metrics.json`

其中一个候选：

- 分数退化到 `-73.96759388334885`
- reward 仅为 `-0.0462962962962963`

这说明当前 `q3` 的惩罚幅度太弱，不能把“灾难性错误”与“轻微退化”有效区分开。

### 6. 直接结论

当前 reward 设计没有围绕“真正推动搜索长期进步”来组织主信号，而是在鼓励：

- 一部分真正进步的候选
- 一大批结构新颖但搜索无效的候选
- 对严重退化候选的打击又太轻

这会让 RL 学到一个错误倾向：

### 只要格式合法、结构看起来有变化，就可以活得还不错

而不是：

### 必须产生长期提升搜索质量的候选

---

## 八、最关键的问题三：初始化阶段与在线RL阶段的输出契约不一致

### 1. 宽松初始化，严格在线

初始化阶段：

- `llm4ad/method/eoh_rl/eoh_rl.py:1133-1137`
- 使用 `strict_contract=False`

在线 RL 阶段：

- `llm4ad/method/eoh_rl/rl/grpo_trainer.py:181-200`
- 使用 `strict_contract=True`

这意味着：

- 初始化能接收的一些代码风格
- 到了 RL 阶段会被视为 invalid

这本身就是闭环断裂。

### 2. 提示词并没有把严格契约讲清楚

参考：`llm4ad/method/eoh_rl/prompt.py:120-127`

当前输出要求大致是：

- 先一句话描述算法
- 再实现函数
- 不要额外解释

但严格解析器真正要求的是：

- 顶层只能有 import 和唯一一个 function
- function 里不能再嵌套 function/class/global/nonlocal
- 不能 decorator
- 必须完全匹配模板签名

参考：`llm4ad/method/eoh_rl/sampler.py:147-191`

### 3. 大量 `parse_error` 不是“模型没写代码”，而是“模型写了你没在 prompt 中禁止的结构”

典型证据：

- `logs/TSP/eoh_local_rl/adaptive_op_plus_collapse_np1_TSP100_run1_20260704_092442/rl_training/rl_update_001/metrics.json`

里面多条 completion 都生成了看起来正常的 Python 函数，但函数体内带了内部 helper function，于是被严格解析器拒绝。

例如：

- 内部定义 `def potential_function(...)`
- 内部定义 `def calculate_remaining_total_spread(...)`
- 内部定义 `def calculate_centroid(...)`

这类写法在当前 strict parser 下会被直接打成 `parse_error`。

### 4. AST gate 还没有打开

配置中：

- `configs/run_eoh_local_rl/eoh_local_rl_tsp.yaml:131-145`
- `enable_ast_gate: false`

这意味着当前系统允许大量不符合严格契约的输出一路进入 rollout，最后集中在 reward callback 阶段爆炸成 invalid。

### 5. 直接后果

这会导致 RL 学到的第一件事，不是“如何写更优启发式”，而是：

### 如何尽量不要被解析器打死

这当然会极大稀释真正的搜索信号。

---

## 九、最关键的问题四：LoRA机制当前不是全局稳定机制，而是局部批次门控机制

### 1. 你理想中的 LoRA 机制目标

从你的描述来看，你想要的是：

- 即使 RL 优化目标和 SFT 目标不一致
- 仍然通过 LoRA 约束保持输出格式稳定、全局性能稳定
- 三次运行都应稳定接近某个性能带，例如 `-8.50` 左右

这意味着 LoRA 机制应该像：

### 一个跨轮次、跨 prompt、跨规模的全局稳定器

### 2. 当前 LoRA gate 实际在做什么

参考：`llm4ad/method/eoh_rl/rl/lora_gate.py:42-127`

当前 gate 输入几乎完全来自本轮 rollout summary：

- valid_count
- parent_improve_count
- frontier_improve_count
- registered_count
- positive_reward_count
- 与 anchor/previous 的参数距离

它不做的事情包括：

- 不看 fresh prompt bank
- 不看 held-out prompt bank
- 不看 held-out instance set
- 不看 mixed-scale 验证
- 不看训练后新策略在下一轮 prompt 上的真实采样分布

### 3. 当前 LoRA gate 判断基础过于局部

当前等于是在问：

### 这一轮训练所对应的这批 rollout，看起来有没有一点证据？

而不是在问：

### 训练后的策略，是否在新的搜索上下文下稳定地产生更优、且格式更稳的候选？

这两者不是一回事。

### 4. 注释与实现不完全一致

代码注释：

- `llm4ad/method/eoh_rl/rl/lora_gate.py:19-23`

注释强调：

- 只看“进入主种群”或“突破 frontier”这类闭环证据

但实际实现中：

- `positive_reward_count > 0`
- `parent_improve_count > 0`

也会被视作证据，见：`llm4ad/method/eoh_rl/rl/lora_gate.py:70-89`

如果 reward 本身已经目标错位，那么 LoRA gate 也会一起被带偏。

### 5. 日志证据：关闭 gate 并没有从根本上解决问题

你有两类日志：

- 有 LoRA gate 的版本
- `no_lora_rollback` 版本

从日志看：

- 有 gate 的版本，很多更新被拒绝，RL 常常表现得像没真正推进。
- 没有 gate 的版本，LoRA 基本全接收，但大量更新仍然不改善 best score。

这说明：

### gate 不是唯一问题，但当前 gate 也不是正确的稳定机制

它只能解释一部分“为什么 RL 看起来没推进”，不能解释“为什么把 gate 去掉后仍然明显差于纯进化”。

---

## 十、最关键的问题五：自适应算子权重当前主要在学习噪声

### 1. 当前 update 粒度太小

配置中：

- `prompts_per_update: 1`
- `num_generations: 4`
- `num_iterations: 1`

见：`configs/run_eoh_local_rl/eoh_local_rl_tsp.yaml:71-72,92,104`

也就是说，每次 RL update 只有：

- 1 个 prompt
- 4 个 completion

### 2. 算子调度器却试图从这种高方差信号中估算信用

参考：`llm4ad/method/eoh_rl/rl/operator_scheduler.py:197-233`

它会根据：

- frontier 数量
- parent improve 数量
- positive reward 数量
- failure 数量

更新每个算子的 running average。

但当每轮只有 4 个样本时：

- 一个 parse failure 批次
- 一个 q2 正奖励批次
- 一个偶然 parent improve 批次

就足以严重扭曲某个算子的历史信用。

### 3. 日志证据：算子 credit 会被锁死

典型证据文件：

- `logs/TSP/eoh_local_rl/adaptive_op_plus_collapse_np1_TSP100_run1_20260703_123341/rl_training/rl_update_370/metrics.json`

其中：

- `credit_updates.m1 = 369`
- `credit_updates.m2 = 1`
- `credit_updates.e1 = 0`
- `credit_updates.e2 = 0`

这说明后期几乎整个系统都在重复同一个 operator。

这不是合理的算子自适应，而是：

### 在高噪声小样本更新下，算子分布被早期偶然性锁死

### 4. 与 CALM 的对比

`CALM` 虽然也有 operator 分配，但它更简单、更可解释：

- 先给出每轮 operator 上限
- 根据 population 状态做简单可控采样
- 不把 operator 学习逻辑做成一个强耦合的隐式控制器

参考：`CALM/calm_trainer.py:193-279`

你的当前实现更“高级”，但在观测不足时，反而更不可靠。

---

## 十一、最关键的问题六：自适应坍缩机制存在恢复期半死锁风险

### 1. 当前坍缩机制设计逻辑

参考：

- 坍缩判定：`llm4ad/method/eoh_rl/rl/collapse.py:73-178`
- 执行坍缩：`llm4ad/method/eoh_rl/eoh_rl.py:624-647`
- 恢复结束：`llm4ad/method/eoh_rl/eoh_rl.py:648-669`

当前逻辑是：

1. 发现长期无突破且多样性低。
2. 保留少数 retained core。
3. 把 active population 缩小。
4. 进入 recovery。
5. 等 population 补回满额且多样性恢复，再退出 recovery。

### 2. 这个设计在高 invalid 率下会卡死

因为 recovery 结束依赖于：

- population 先补回 `pop_size`

但如果 recovery 期间：

- parse failure 高
- strict contract 失败高
- 入池率低

那么种群可能永远补不满。

### 3. 日志证据：恢复期长期补不满种群

典型证据文件：

- `logs/TSP/eoh_local_rl/adaptive_op_plus_collapse_np1_TSP100_run1_20260703_123341/rl_training/rl_update_370/metrics.json`

其中：

- `recovery_active = true`
- `current_population_size = 7`
- `target_population_size = 10`
- `current_no_improve_streak = 369`

这说明系统在恢复期已经停留了极长时间，而且一直没有真正恢复。

### 4. 这不是“跳出局部最优”，而是“进入恢复期后系统半瘫痪”

从搜索视角看，这种坍缩并没有带来有效扰动，反而：

- 降低了 active parent 池
- 削弱了 crossover 可行性
- 放大了 parse/invalid 噪声
- 导致后续 operator scheduler 观测更加失真

### 5. 与 CALM 的对比

`CALM` 的 collapse 更轻：

- 保留 best
- 重新加入 seed
- 清 prompt 历史
- 继续搜索

参考：`CALM/calm_trainer.py:170-182`

它不会把系统拖进一个依赖“必须先补满 active population 才能恢复”的复杂状态机里。

---

## 十二、日志层面的关键证据总结

下面列出最有代表性的日志证据。

### 证据一：有些 update 从一开始就完全死在解析阶段

文件：

- `logs/TSP/eoh_local_rl/adaptive_op_plus_collapse_np1_TSP100_run1_20260704_092442/rl_training/rl_update_001/metrics.json`

特征：

- `parse_failures = 4`
- `valid_count = 0`
- `accepted_lora = false`
- `reject_reason = gate_no_valid_completion`

结论：

- rollout 根本没有形成有效 reward 信号。
- 这种 update 对策略学习几乎没有正向价值。

### 证据二：去掉 LoRA rollback 后，很多 update 仍然完全不改善 best score

文件：

- `logs/TSP/eoh_local_rl/adaptive_op_plus_collapse_no_lora_rollback_np1_TSP100_run1_20260702_123929/rl_training/rl_update_020/metrics.json`

特征：

- `accepted_lora = true`
- `committed_candidate_count = 1`
- `parent_improve_count = 1`
- `best_after_rl = best_before_rl`

结论：

- 就算 LoRA 不回滚，RL 仍然主要在做“局部替换”，没有推动全球最优。

### 证据三：Q2 正奖励会大面积出现，但不带来搜索收益

文件：

- `logs/TSP/eoh_local_rl/adaptive_op_plus_collapse_no_lora_rollback_np1_TSP100_run2_20260704_020111/rl_training/rl_update_129/metrics.json`

特征：

- `positive_reward_count = 4`
- 全是 `bqr_q2_parent_tie_novelty`
- `registered_count = 0`
- `best_after = best_before`

结论：

- 奖励机制正在鼓励“结构新颖但无用”的候选。

### 证据四：长期平 plateau 的 run 确实存在

文件：

- `logs/TSP/eoh_local_rl/adaptive_op_plus_collapse_np1_TSP100_run1_20260703_123341/run_summary.json`

特征：

- 从 `rl_update_id = 1` 到 `370`，`best_at_trigger` 一直固定在 `-9.131325491982896`

结论：

- 这是你所说“强化学习阶段几乎不突破”的直接证据。

### 证据五：即使相对好一些的 run，也会在某个局部最优附近长期横盘

文件：

- `logs/TSP/eoh_local_rl/adaptive_op_plus_collapse_no_lora_rollback_np1_TSP100_run1_20260702_123929/run_summary.json`

特征：

- 前几十轮有进步
- 但很快长期横盘在 `-8.714776398092443`

结论：

- 当前框架不是完全不能改进，而是：
- 能靠搜索偶发性推进一点
- 但 RL 没有把这种进步持续放大

这正好说明问题在闭环，而不是在“模型完全不会优化”。

---

## 十三、你提出的三个机制逐项评估

## 13.1 LoRA稳定机制评估

### 你的目标

- 稳定输出格式
- 稳定全局性能
- 降低 RL 与 SFT 目标不一致带来的漂移

### 当前卡点

1. gate 依据来自同一批 rollout，不是 fresh validation。
2. gate 不做跨规模验证。
3. gate 目标被 q2 和 parent_improve 这类局部信号污染。
4. 当前 anchor 更接近“静态 SFT 锚点”，不是真正意义上的在线全局性能锚点。

### 结论

当前实现不能证明自己是“全局稳定器”。

更准确地说，它只是：

### 一个基于本轮 rollout summary 的局部 LoRA 部署门

这和你想要的机制级别不一致。

---

## 13.2 自适应算子权重机制评估

### 你的目标

- 动态提升好算子占比
- 加速生成更优启发式

### 当前卡点

1. 每次 update 的观测样本太少。
2. reward 信号本身混入了 q2/q3 噪声。
3. parse/invalid 会过早打击某个算子信用。
4. collapse/recovery 会改变算子可行性和 parent pool，从而让 credit 非平稳。

### 结论

当前算子调度器更像：

### 基于高噪声局部 rollout 的隐式偏置器

而不是一个可靠的 operator policy learner。

---

## 13.3 自适应坍缩机制评估

### 你的目标

- 防止长期陷入局部最优
- 用清空种群制造扰动和多样性

### 当前卡点

1. 坍缩后 active parent pool 过小。
2. recovery 结束依赖补满 population，容易卡住。
3. 在 strict contract 和高 invalid 率下，扰动变成纯噪声而不是有效探索。

### 结论

当前坍缩机制更容易：

### 放大恢复成本，而不是降低局部最优成本

---

## 十四、与 CALM 的关键差异分析

## 14.1 CALM 的搜索-训练闭环更干净

参考：`CALM/calm_trainer.py:161-287,349-481`

CALM 的核心循环是：

1. 基于当前 algo 集合构造 prompt 数据集。
2. 模型生成候选。
3. reward 函数直接评估候选好坏。
4. 合格算法进入 `algos`。
5. 下一轮 prompt 基于新的 `algos` 继续构造。

这里没有“入池与策略部署交错”的复杂结构。

## 14.2 CALM 的奖励更直白

参考：`CALM/calm_trainer.py:439-464`

CALM 的主奖励逻辑很单纯：

- 比 parent 好，强正奖励。
- 不好，负奖励或零奖励。
- bug、随机、格式错，固定惩罚。

它的 reward 没有引入一个会大面积奖励“平局创新”的主分支。

## 14.3 CALM 的 collapse 更轻，更容易 debug

参考：`CALM/calm_trainer.py:170-182`

CALM 的 collapse 做法是：

- 保留 best
- 重新引入 seed
- 清掉用过的 prompt 历史

它更像“搜索重启”，不是“状态机恢复”。

## 14.4 CALM 的 prompt 更强约束、更明确

参考：`CALM/calm_trainer.py:483-546`

CALM 有比较明确的输出协议：

- `{{idea}}`
- 单个 python function
- deterministic
- 不要额外解释

而你当前：

- `EoHPrompt.get_system_prompt()` 是空的，见 `llm4ad/method/eoh_rl/prompt.py:21-22`
- 在线 strict parser 的限制并没有被 prompt 完整表达

这会直接影响格式稳定性。

---

## 十五、当前框架为什么会“看起来像在训练，但实质上很少突破”

可以把这个问题分成四层。

### 第一层：策略还没学到搜索能力，就先被格式约束击穿

表现为：

- parse_error 多
- wrong_signature 多
- invalid reward 多

### 第二层：好不容易格式合法了，reward 也不一定对准真实搜索目标

表现为：

- q2 正奖励很多
- 但 registered_count 和 best_after 没变化

### 第三层：即使某轮出现局部好样本，也未必能转化为未来更好的采样分布

因为：

- LoRA 部署判定和样本入池不是 clean two-stage 流程
- 训练后没有 fresh 验证

### 第四层：系统被额外机制锁进恢复期、算子偏置、局部 plateau

表现为：

- 长期 recovery_active
- operator credit 锁死
- best curve 长时间水平

所以当前系统不是“RL 很弱”，而是：

### RL 根本没有获得一个能稳定积累有效搜索信用的运行环境

---

## 十六、跨问题跨规模通用性的审查结论

你要求三个机制必须跨问题、跨规模通用。当前实现距离这个目标还有明显差距。

### 1. 框架层面虽然预留了 mixed-scale evaluator 入口，但当前主链没真正用起来

参考：`llm4ad/method/eoh_rl/config_runner.py:162-173`

代码支持 mixed-scale，但当前 TSP 配置仍然是：

- `default_scale = small`
- `scales = {small, medium-100, medium-200}`
- 没有启用 `mixed_scales.enabled`

参考：`configs/run_eoh_local_rl/eoh_local_rl_tsp.yaml:149-170`

### 2. 你的 reward 仍然主要基于单一当前任务单一当前规模的评估分数

参考：`llm4ad/task/optimization/tsp_construct/evaluation.py:154-209`

当前 reward 是当前 evaluator 上平均分数的函数，不是多规模归一化后的综合指标。

### 3. 某些代码里还残留任务特征痕迹

例如：

- `llm4ad/method/eoh_rl/sampler.py:233-256`

`_derive_strategy_from_python()` 中显式使用了 `destination_node`、`distance_matrix` 这类 TSP 风格信号。这种做法对通用框架是不利的。

### 结论

当前实现更接近：

### 单任务单规模在线后训练实验框架

而不是：

### 已经满足跨问题跨规模通用要求的主框架

---

## 十七、工程结构上的冗余与设计异味

### 1. `eoh_rl` 不是薄扩展，而是整套 fork

你现在基本上把：

- `population`
- `sampler`
- `prompt`
- `config_runner`

都重新复制出了一份 RL 版本。

这导致：

- 原始 EoH 行为和 EOH_RL 行为逐渐漂移。
- 很多 bug 修起来要双份维护。

### 2. 两个核心文件过大

- `llm4ad/method/eoh_rl/eoh_rl.py`
- `llm4ad/method/eoh_rl/rl/grpo_trainer.py`

这两个文件承担的职责过多：

- 搜索调度
- 候选注册
- 评估桥接
- LoRA 生命周期
- reward logging
- adaptive operator
- collapse recovery
- checkpoint

这会让调试和重构成本急剧上升。

### 3. 存在一些硬编码运行约束

例如：

- `pop_size == 10`
- `trigger_every_n_generations == 1`
- operator_cycle 必须正好是 `e1/e2/m1/m2`

参考：`llm4ad/method/eoh_rl/eoh_rl.py:136-139,214-221`

这些都说明当前框架还没有真正抽象成通用主线。

---

## 十八、修复优先级建议：先改什么，后改什么

下面给出我认为最重要的修复顺序。

## 第一优先级：统一输出契约

必须先解决：

1. 初始化阶段和 RL 阶段解析规则不一致。
2. prompt 没有表达 strict parser 的全部要求。
3. AST gate 没前置启用。

### 建议

- 要么强化 prompt，明确要求只能输出单函数、不能嵌套 helper、不能额外顶层语句。
- 要么放宽 strict parser，让它接受一部分结构上安全的 helper function。
- 更推荐先把 AST gate 提前到采样端，而不是等 reward callback 再集中判 invalid。

如果这一步不做，后面所有 RL 信号都不干净。

## 第二优先级：收窄主奖励，暂停把 q2 当作主训练正例

### 建议

第一版奖励主线应该简单到足够可控：

- `frontier improve`：强正奖励
- `parent improve but not frontier improve`：中等正奖励
- `tie`：零奖励
- `regression`：按退化幅度给负奖励
- `invalid`：强负奖励

`q2` 可以保留，但不应该继续作为当前主 RL 正反馈来源，更不应该直接进入 LoRA gate 证据。

## 第三优先级：把“候选入池”和“LoRA部署”拆成两阶段

### 推荐结构

1. 用 batch A 训练候选 LoRA。
2. 用 fresh batch B 或固定 prompt bank 测候选 LoRA。
3. 如果 B 通过：
   - 部署 LoRA
   - 提交本轮 search-state update
4. 如果 B 不通过：
   - 回滚 LoRA
   - 丢弃本轮 search-state update

这样才能保证：

### 搜索状态和策略状态是一致推进的

## 第四优先级：把坍缩机制改轻，而不是改重

建议先改成更像 CALM 的轻量版本：

- 保留 best
- 重新引入 seed 或 archive exemplar
- 清 prompt history
- 重置 operator debt
- 继续搜索

先不要依赖“必须补满种群才能退出 recovery”的重状态机。

## 第五优先级：扩大每次 RL update 的观测量

当前 `1 prompt x 4 completions` 太小。

建议至少提高：

- `prompts_per_update`

否则 adaptive operator 和 LoRA gate 永远在高方差状态下工作。

---

## 十九、面向“像CALM一样干净整洁高效”的重构建议

如果你的目标不是“小修补”，而是做出一套真正可持续迭代的后训练框架，我建议把结构改成下面这样。

## 19.1 把主系统拆成四层

### 第一层：搜索状态层

建议单独抽象一个 `搜索状态管理器`，只负责：

- active population
- archive
- seed pool
- prompt history
- collapse/reset state

它不关心 LoRA，也不关心 trainer。

### 第二层：prompt 与候选生成层

只负责：

- 从当前搜索状态构造 prompt
- 调用当前 policy 生成 completion
- 做前置格式过滤

### 第三层：评估与奖励层

只负责：

- 解析 completion
- 执行 evaluator
- 产出 raw score
- 产出 reward
- 产出结构化事件

### 第四层：策略更新与部署层

只负责：

- 根据事件训练候选 LoRA
- 做 fresh validation
- 决定是否部署

这四层必须解耦，否则框架永远不干净。

## 19.2 把 archive 做成一等公民

当前你几乎只依赖 active population。

建议像 CALM 一样保留：

- 活跃 top-k
- 历史 archive
- seed/archive exemplar

这样：

- active population 负责 exploitation
- archive 负责 diversity retrieval

而不是每次坍缩都只靠活跃种群自己长回来。

## 19.3 把 LoRA gate 从“当前批次门控器”改成“全局验证器”

建议 gate 输入改成：

- fresh prompt bank 上的 parse rate
- fresh prompt bank 上的 valid rate
- fresh prompt bank 上的 parent/frontier improve 率
- mixed-scale evaluator 上的泛化分数
- 与 anchor/previous 的权重距离

只有这样，它才配得上“稳定机制”这个名字。

---

## 二十、建议的消融实验矩阵

为了验证到底是哪一层出问题，我建议你不要再继续混合修改，而是做强制分离的消融。

### 实验一：只修契约，不改奖励

目标：

- 看 parse/invalid 是否显著下降
- 看 RL 是否从“死在格式”变成“至少能稳定产生活候选”

### 实验二：保留 strict parser，但去掉 q2 正奖励

目标：

- 看 best score 改进率是否上升
- 看正奖励和真实 best 改进是否更一致

### 实验三：关闭 adaptive operator，只保留固定 operator cycle

目标：

- 看 plateau 是否减轻
- 看是不是调度器本身在放大噪声

### 实验四：关闭 collapse，只保留简单重启

目标：

- 看 recovery 死锁是否消失

### 实验五：训练后不立即入池，改成 fresh rollout 验证后再入池

目标：

- 直接检验当前最大闭环问题是否属实

### 实验六：单规模 reward 与 mixed-scale reward 对比

目标：

- 检查你想要的跨规模通用性，是否在 reward 层就已经丢失

---

## 二十一、我对当前版本的总体判断

如果只问一句话：

### 目前效果为什么远不如没有强化学习？

我的回答是：

### 因为当前这套 RL 不是在稳定放大搜索信号，而是在用一个高耦合、高噪声、目标错位的在线训练闭环去稀释原本纯净的进化搜索信号。

再展开一点就是：

1. 输出契约不一致，很多样本死于格式和解析。
2. 奖励鼓励了大量“结构新颖但搜索无效”的候选。
3. 候选入池和 LoRA 部署不一致推进，闭环不干净。
4. 算子调度和坍缩机制建立在高噪声观测上，容易锁死或半死锁。
5. LoRA gate 不是全局稳定器，只是本轮 rollout 的局部门。

这几个问题叠加起来，结果自然会比“直接拿当前模型做纯进化搜索”更差。

---

## 二十二、最核心的落地建议

如果你近期只想做最少修改、最快看到方向性改善，我建议优先只做三件事：

### 建议一：统一契约

- 让初始化和在线 RL 使用同一套合法输出标准。
- prompt 中明确写死单函数、不可嵌套 helper、不可额外顶层结构。
- 提前开启 AST gate，减少 reward 端无效样本爆炸。

### 建议二：收紧主奖励

- 去掉 q2 作为主正奖励来源。
- tie 给 0。
- 退化按幅度给更强负奖励。
- invalid 继续强负。

### 建议三：把策略部署改成 two-stage validation

- 训练和候选入池先不要绑在同一批样本上。
- LoRA 是否部署，必须看 fresh validation。

如果这三步做完后效果仍然不好，再去查：

- adaptive operator 是否真的有价值
- collapse 是否真的有价值

而不是反过来继续先调这些高级机制。

---

## 二十三、附：本次审查中最关键的代码与日志引用索引

### 代码引用

- `llm4ad/method/eoh_rl/eoh_rl.py:693-802`
- `llm4ad/method/eoh_rl/eoh_rl.py:934-1013`
- `llm4ad/method/eoh_rl/eoh_rl.py:1124-1156`
- `llm4ad/method/eoh_rl/population.py:61-75`
- `llm4ad/method/eoh_rl/sampler.py:113-191`
- `llm4ad/method/eoh_rl/prompt.py:120-127`
- `llm4ad/method/eoh_rl/rl/grpo_trainer.py:181-264`
- `llm4ad/method/eoh_rl/rl/grpo_trainer.py:276-296`
- `llm4ad/method/eoh_rl/rl/grpo_trainer.py:475-518`
- `llm4ad/method/eoh_rl/rl/reward.py:56-137`
- `llm4ad/method/eoh_rl/rl/lora_gate.py:42-127`
- `llm4ad/method/eoh_rl/rl/operator_scheduler.py:197-233`
- `llm4ad/method/eoh_rl/rl/collapse.py:73-178`
- `llm4ad/method/eoh_rl/config_runner.py:162-173`

### 日志引用

- `logs/TSP/eoh_local_rl/adaptive_op_plus_collapse_np1_TSP100_run1_20260704_092442/rl_training/rl_update_001/metrics.json`
- `logs/TSP/eoh_local_rl/adaptive_op_plus_collapse_no_lora_rollback_np1_TSP100_run1_20260702_123929/rl_training/rl_update_020/metrics.json`
- `logs/TSP/eoh_local_rl/adaptive_op_plus_collapse_np1_TSP100_run1_20260703_123341/rl_training/rl_update_370/metrics.json`
- `logs/TSP/eoh_local_rl/adaptive_op_plus_collapse_no_lora_rollback_np1_TSP100_run2_20260704_020111/rl_training/rl_update_129/metrics.json`
- `logs/TSP/eoh_local_rl/adaptive_op_plus_collapse_np1_TSP100_run1_20260703_123341/run_summary.json`
- `logs/TSP/eoh_local_rl/adaptive_op_plus_collapse_no_lora_rollback_np1_TSP100_run1_20260702_123929/run_summary.json`

---

## 二十四、最后结论

你现在遇到的不是“RL 没调好”，而是：

### 这套后训练框架当前还没有形成一个目标一致、状态一致、验证一致的强化学习搜索闭环。

在这个前提下：

- LoRA 机制无法真正承担全局稳定器职责。
- 自适应算子无法可靠学习算子价值。
- 自适应坍缩无法稳定充当跳出局部最优的手段。

如果你要和 `CALM` 做正面对比，最需要补的不是更多机制，而是：

### 更干净的主闭环、更统一的契约、更对齐的主奖励、更可验证的策略部署流程。
