# 对照main审查实现

> 你是LLM，自动启发式算法设计，LLM后训练，自动权重设计的专家。不要相信这个项目的任何代码。系统的审查项目的实现问题。可以先对照main分支，理清整个的设计思路，任何再系统的分析实现出现的问题

<details><summary>63 previous messages</summary>

> 我会先把当前工作树、分支关系、项目入口和与 `main` 的差异摸清楚，再按“设计意图 -> 实现偏差 -> 风险级别”的方式审查。不会先假设现有实现是可信的。
>
> <details><summary>Explored a list, ran 3 commands</summary>
>
> - Ran `git status --short --branch`
>
> - Ran `git branch --all --no-color`
>
> - Ran `git rev-parse --show-toplevel`
>
> <details><summary>Explored a list</summary>
>
> - Listed files
>
> </details>
>
> </details>
>
> 当前在 detached HEAD，`origin/main` 存在；我会用 `origin/main...HEAD` 作为主比较基准。文件范围里明显新增了本地后训练/RL、collapse、自适应算子调度、多任务 SFT 相关代码，这是重点审查面。
>
> <details><summary>Ran 5 commands</summary>
>
> - Ran `git merge-base HEAD origin/main`
> - Ran `git diff --stat origin/main...HEAD`
> - Ran `git rev-parse HEAD`
> - Ran `git diff --name-status origin/main...HEAD`
> - Ran `git log --oneline --decorate --graph --max-count=30 --all`
>
> </details>
>
> 这里有一个重要背景：当前提交和 `origin/main` 没有 merge base，说明这个分支历史可能是轻量上传/重建出来的，而不是从 main 正常演进。接下来我会改用树级别的 `git diff origin/main HEAD` 对照内容差异，同时单独读 main 的入口和当前入口。
>
> <details><summary>Ran 4 commands</summary>
>
> - Ran `git diff --stat origin/main HEAD`
> - Ran `git diff --name-status origin/main HEAD`
> - Ran `git log --oneline --decorate --max-count=20 origin/main`
> - Ran `git ls-tree -r --name-only origin/main`
>
> </details>
>
> 树级 diff 很大：约 3.5 万行新增，同时删除了 main 上的 MLES、PartEvo、若干示例和环境文件。审查上我会把“破坏 main 既有能力”和“新增后训练链路自身不可运行/指标不可信”分开看。
>
> <details><summary>Explored 4 files, ran a command</summary>
>
> <details><summary>Explored a file</summary>
>
> - Read `./README.md`
>
> </details>
>
> - Ran `git show origin/main:README.md`
>
> <details><summary>Explored 3 files</summary>
>
> - Read `./tests/test_adaptive_collapse_and_scheduler.py`
> - Read `./requirements.txt`
> - Read `./setup.py`
>
> </details>
>
> </details>
>
> 已经确认几个项目级风险苗头：README 仍引用 `assets/figs/*`，但这些资源在当前树被删；README 的许可证徽章从 BSD 改成 MIT，而 main 文档和许可证描述仍偏 BSD。下面我继续看新增运行链路本身，尤其是 `EoHRL`、RL trainer、model manager、checkpoint/collapse/scheduler。
>
> <details><summary>Explored 4 files, a search</summary>
>
> - Searched for `^(class|def) ` in `eoh_rl`
> - Read `./llm4ad/method/eoh_rl/__init__.py`
> - Read `./llm4ad/method/__init__.py`
> - Read `./llm4ad/method/eoh_rl/eoh_rl.py`
> - Read `./llm4ad/tools/rl/trainer_base.py`
>
> </details>
>
> 新增代码的组织意图大致是：`EoHRL` 保留 EoH 的采样-评测-种群循环，在采样结果上构造 reward/event，再用 online GRPO 或 TRL backend 更新 LoRA，并重启/切换推理模型；`collapse` 监测停滞后收缩种群，`operator_scheduler` 动态调节 m/e/i 类算子。现在我开始找会导致实验结论不可信的实现细节。
>
> <details><summary>Explored a file, a search, ran a command</summary>
>
> <details><summary>Explored a search</summary>
>
> - Searched for `def run|def _sample|def _evaluate|def _train|def _perform|def _bootstrap|def _evolve|def _record|def _update|def close|def __del__` in `eoh_rl.py`
>
> </details>
>
> - Ran `rg "except Exception|except BaseException|pass$|TODO|FIXME|assert |return None|fallback|mock|dummy|fake|random" llm4ad/method/eoh_rl llm4ad/tools/rl llm4ad/tools/llm llm4ad/tools/config -n`
>
> <details><summary>Explored 3 files</summary>
>
> - Read `./llm4ad/method/eoh_rl/eoh_rl.py`
> - Read `./llm4ad/method/eoh_rl/eoh_rl.py`
> - Read `./llm4ad/method/eoh_rl/eoh_rl.py`
>
> </details>
>
> </details>
>
> 一个核心实现点值得后续重点判断：RL 路径里 `_rl_update_count` 先递增，但样本预算 `_tot_sample_nums` 是 backend 完成后按返回 `total` 一次性增加；这会让注册候选时的 `_continue_loop()` 看不到本轮已消耗的 completion 数，存在超预算和 generation/sample 语义混淆风险。
>
> 我继续沿着 integrated GRPO 的候选注册、prompt 构造、collapse/recovery 逻辑往下读；这里最容易出现“指标看起来在动，但训练信号和进化种群不是同一个东西”的问题。
>
> <details><summary>Explored 15 files</summary>
>
> - Read `./llm4ad/method/eoh_rl/eoh_rl.py`
> - Read `./llm4ad/method/eoh_rl/eoh_rl.py`
> - Read `./llm4ad/tools/rl/backends/online_grpo.py`
> - Read `./llm4ad/method/eoh_rl/eoh_rl.py`
> - Read `./llm4ad/tools/rl/backends/online_grpo.py`
> - Read `./llm4ad/tools/rl/rewards.py`
> - Read `./llm4ad/tools/rl/backends/online_grpo.py`
> - Read `./llm4ad/tools/rl/backends/trl_grpo_backend.py`
> - Read `./llm4ad/tools/rl/backends/__init__.py`
> - Read `./llm4ad/tools/rl/backends/trl_grpo_backend.py`
> - Read `./llm4ad/tools/rl/rewards.py`
> - Read `./llm4ad/method/eoh_rl/sampler.py`
> - Read `./llm4ad/tools/rl/operator_scheduler.py`
> - Read `./llm4ad/method/eoh_rl/population.py`
> - Read `./llm4ad/tools/rl/operator_scheduler.py`
>
> </details>
>
> 到这里已看到一个更实质的信号设计问题：候选先用 `compute_bqr_result` 产生训练 reward，再才注册进种群；但“是否真正进入/存活于 population”只作为统计或 gating，默认不是 reward 的一等输入。也就是说 RL 优化的是 parent/frontier-relative completion reward，不一定优化真实 survivor 选择后的长期种群质量。
>
> <details><summary>Explored 3 files, ran 2 commands</summary>
>
> - Ran `git diff -- llm4ad/base/evaluate.py`
>
> <details><summary>Explored 3 files</summary>
>
> - Read `./llm4ad/base/sample.py`
> - Read `./llm4ad/base/evaluate.py`
> - Read `./llm4ad/base/code.py`
>
> </details>
>
> - Ran `git diff -- llm4ad/tools/llm/llm_api_https.py llm4ad/tools/llm/local_vllm/vllm_local_http_api.py llm4ad/tools/llm/local_vllm/vllm_llama.py`
>
> </details>
>
> `SecureEvaluator` 的诊断增强目前仍只在外层把 `None` 归为 `eval_or_timeout`，它没有把超时、执行异常、任务约束失败、返回非法值分开；这会直接影响 RL 的负样本分型和 collapse/scheduler 的信用分配。
>
> <details><summary>Explored 2 files, ran 2 commands</summary>
>
> - Ran `git diff origin/main HEAD -- llm4ad/base/sample.py llm4ad/base/code.py llm4ad/method/eoh/eoh.py llm4ad/method/eoh/profiler.py llm4ad/method/reevo/reevo.py`
>
> <details><summary>Explored a file</summary>
>
> - Read `./llm4ad/base/evaluate.py`
>
> </details>
>
> - Ran `git diff origin/main HEAD -- llm4ad/base/evaluate.py`
>
> <details><summary>Explored 2 files</summary>
>
> - Read `./llm4ad/tools/config/loader.py`
> - Read `./llm4ad/tools/config/loader.py`
>
> </details>
>
> </details>
>
> 接下来我会做一轮低成本验证：先编译新增 Python 文件找语法/导入层面的硬错误，再跑新增测试文件。测试失败本身也会作为实现审查证据，不会因为测试“看起来是作者写的”就默认可信。
>
> <details><summary>Ran 2 commands</summary>
>
> - Ran `python -m compileall -q llm4ad tests scripts example/distill_and_evolve`
> - Ran `python -m pytest -q tests/test_adaptive_collapse_and_scheduler.py`
>
> </details>
>
> 编译全仓库通过；新增测试文件 35 个用例通过。但这些测试集中覆盖 scheduler/collapse 的局部数学行为，没有覆盖真实 GRPO 训练、vLLM/Unsloth 生命周期、数据采集、配置到对象构建、resume 或 main 既有方法兼容性。
>
> <details><summary>Explored 10 files</summary>
>
> - Read `./llm4ad/__init__.py`
> - Read `./llm4ad/tools/rl/llm_factory.py`
> - Read `./llm4ad/tools/llm/resident_unsloth_llm.py`
> - Read `./llm4ad/tools/llm/resident_unsloth_llm.py`
> - Read `./llm4ad/tools/rl/llm_factory.py`
> - Read `./llm4ad/tools/llm/resident_unsloth_llm.py`
> - Read `./llm4ad/tools/llm/local_vllm/vllm_local_http_api.py`
> - Read `./llm4ad/tools/llm/local_vllm/vllm_local_http_api.py`
> - Read `./llm4ad/tools/llm/rl_augmented_llm.py`
> - Read `./llm4ad/tools/llm/llm_api_https.py`
>
> </details>
>
> 本地/在线 LLM 接口还有一个运行风险：在线 `HttpsApi.draw_sample()` 没有显式最大重试次数，非 debug 模式下会永久重试；这会让采集或初始化阶段在 API 配置错误、额度耗尽、模型返回格式变更时无限挂起，而不是作为失败样本进入统计。
>
> <details><summary>Explored 5 files</summary>
>
> - Read `./example/distill_and_evolve/run_sft_and_evolve.py`
> - Read `./example/distill_and_evolve/run_sft_and_evolve.py`
> - Read `./example/distill_and_evolve/run_collect_data_multi.py`
> - Read `./example/distill_and_evolve/run_collect_data.py`
> - Read `./example/distill_and_evolve/run_collect_data_multi.py`
>
> </details>
>
> 发现一个确定的 P0：`run_sft_and_evolve.py` 创建 `EoHRL` 时没有传 `collapse_config`，但 `EoHRL.__init__` 在启用 RL 时强制要求显式传入 collapse 配置。因此官方两阶段入口按当前代码会在 Phase 2 构造阶段直接失败。
>
> <details><summary>Explored 3 files, 2 searches</summary>
>
> - Searched for `EoHRL\(` in `example`
> - Read `./configs/eoh_rl.example.yaml`
> - Read `./configs/eoh_rl.example.yaml`
> - Searched for `collapse_config|CollapseConfig|require_collapse` in `example`
> - Read `./llm4ad/tools/rl/__init__.py`
>
> </details>
>
> 还有第二个确定入口问题：默认 `configs/eoh_rl.example.yaml` 同时设置 `sft.skip_sft: true` 与 `cold_start_from_base: true`，注释说可从基座冷启动；但 `run_sft_and_evolve.py` 在 `skip_sft` 时仍强制查找 `SFT_OUTPUT_DIR/lora_sft`。这会让默认配置无法按注释语义冷启动。
>
> <details><summary>Explored 5 files</summary>
>
> - Read `./llm4ad/tools/rl/task_registry.py`
> - Read `./llm4ad/task/optimization/tsp_construct/evaluation.py`
> - Read `./llm4ad/task/optimization/cvrp_construct/evaluation.py`
> - Read `./llm4ad/task/optimization/jssp_construct/evaluation.py`
> - Read `./llm4ad/task/optimization/online_bin_packing/evaluation.py`
>
> </details>
>
> 任务 reward 方向本身看起来是“最大化负目标值”：TSP/CVRP/JSSP/OBP 都返回负成本或负用箱数，因此 `minimize=false` 在这些任务上是合理的。这个点暂时不作为问题。
>
> <details><summary>Explored 5 files</summary>
>
> - Read `./llm4ad/tools/rl/offline_data_collector.py`
> - Read `./llm4ad/tools/rl/offline_data_collector.py`
> - Read `./llm4ad/tools/rl/data_cleaner.py`
> - Read `./llm4ad/tools/rl/data_cleaner.py`
> - Read `./llm4ad/tools/rl/data_cleaner.py`
>
> </details>
>
> Context compaction
>
> Source: automatic
> Status: completed
>
> 我已经把当前工作树和 `origin/main` 做了第一轮结构对比：这个分支和 `main` 没有共同 merge-base，所以我按完整树差异审查，而不是按普通 PR diff 审查。现在我在补齐关键问题的精确行号，避免只给泛泛判断。
>
> <details><summary>Explored 6 files</summary>
>
> - Read `./llm4ad/method/eoh_rl/eoh_rl.py`
> - Read `./example/distill_and_evolve/run_sft_and_evolve.py`
> - Read `./example/distill_and_evolve/run_sft_and_evolve.py`
> - Read `./configs/eoh_rl.example.yaml`
> - Read `./setup.py`
> - Read `./llm4ad/tools/llm/llm_api_https.py`
>
> </details>
>
> 入口脚本这块已经确认出一个硬失败：`EoHRL` 在启用 RL 时强制要求 `collapse_config`，但 `run_sft_and_evolve.py` 构造时没有传。接下来我再看 `skip_sft/cold_start` 的实际控制流和包依赖边界。
>
> <details><summary>Explored 4 files, 2 searches</summary>
>
> - Searched for `SKIP_SFT|cold_start|sft_lora|skip_sft|lora_sft|SFT_OUTPUT` in `run_sft_and_evolve.py`
> - Read `./example/distill_and_evolve/run_sft_and_evolve.py`
> - Read `./example/distill_and_evolve/run_sft_and_evolve.py`
> - Read `./example/distill_and_evolve/run_sft_and_evolve.py`
> - Read `./example/distill_and_evolve/run_sft_and_evolve.py`
> - Searched for `collapse_config|CollapseConfig|adaptive_operator_config|AdaptiveOperatorConfig` in `run_sft_and_evolve.py`
>
> </details>
>
> 另一个入口问题也基本坐实：示例 YAML 把 `skip_sft` 和 `cold_start_from_base` 放在 `eoh_rl.sft`，但入口脚本实际从 `distill` 派生 `SKIP_SFT`，并且跳过 SFT 时只支持已有 LoRA。我要确认 loader 有没有做字段桥接，否则这属于默认配置和入口逻辑直接不一致。
>
> <details><summary>Explored 3 files, 3 searches</summary>
>
> - Searched for `require_distill_plain|skip_sft|cold_start|distill|sft_output_dir|sft_source_dirs` in `config`
> - Read `./llm4ad/tools/config/loader.py`
> - Searched for `^\s*distill:|sft_output_dir|sft_source_dirs|skip_sft|cold_start_from_base` in `eoh_rl.example.yaml`
> - Searched for `class EoHRL|def __init__|collapse_config|adaptive_operator_config|debug_mode` in `eoh_rl.py`
> - Read `./llm4ad/method/eoh_rl/eoh_rl.py`
> - Read `./llm4ad/tools/config/loader.py`
>
> </details>
>
> 我又确认了 loader 的桥接：`require_distill_plain` 确实从 `eoh_rl.sft` 读字段，所以不是字段路径读错；问题在于 `cold_start_from_base` 被读取后没有在入口控制流使用。现在继续看奖励、评估隔离和安装边界。
>
> <details><summary>Explored 5 files, a search</summary>
>
> - Searched for `collapse:|adaptive_operator:|enabled: false|recovery_target_size|recovery_until_full` in `eoh_rl.example.yaml`
> - Read `./configs/eoh_rl.example.yaml`
> - Read `./llm4ad/base/evaluate.py`
> - Read `./llm4ad/tools/rl/online_grpo.py`
> - Read `./llm4ad/tools/rl/online_grpo.py`
> - Read `./llm4ad/__init__.py`
>
> </details>
>
> 评估层这里也有一个设计风险：虽然加了诊断接口，但失败仍大量塌缩成 `eval_or_timeout`，这会让 RL 奖励和自适应算子拿不到真实失败来源。我还在追 reward/backend 文件名，刚才路径名猜错了。
>
> <details><summary>Explored 2 files, 2 searches, a list, ran a command</summary>
>
> <details><summary>Explored 2 files, 2 searches, a list</summary>
>
> - Listed files in `rl`
> - Searched for `class Online|CompletionReward|_apply_bqr|failure_source|registered_to_population|survived_main_population|candidate_meta|reward_event` in `rl`
> - Searched for `from datasets|from omegaconf|from unsloth|from trl|import vllm|from peft|from transformers|ortools|networkx|codebleu|tree_sitter|tree-sitter|gym` in `llm4ad`
> - Read `./requirements.txt`
> - Read `./llm4ad/method/__init__.py`
>
> </details>
>
> - Ran `git diff --name-status origin/main HEAD | rg "(^D\s+(assets|llm4ad/method/(mles|partevo)|environment\.yaml|example)|README|LICENSE|setup\.py)"`
>
> </details>
>
> 依赖边界比代码本身更脆：`requirements.txt` 已经承认 Unsloth/TRL/vLLM/Transformers/OmegaConf 等是运行必需，但 `setup.py` 没同步；而包初始化还会吞掉所有导入异常。这会让“安装成功但核心路径不可用”变得很难定位。
>
> <details><summary>Explored 6 files</summary>
>
> - Read `./configs/eoh_rl.example.yaml`
> - Read `./README.md`
> - Read `./llm4ad/tools/rl/backends/online_grpo.py`
> - Read `./llm4ad/tools/rl/backends/online_grpo.py`
> - Read `./llm4ad/tools/rl/backends/online_grpo.py`
> - Read `./llm4ad/tools/rl/backends/online_grpo.py`
>
> </details>
>
> 主 README 和 main 的关系也有明显断裂：这个分支删除了 assets、MLES、PartEvo 等 main 上的公开内容，但 README 仍在引用这些资源或承诺这些能力。这个问题不是算法细节，而是会直接误导安装和复现实验。
>
> <details><summary>Explored a file, a search, ran 3 commands</summary>
>
> <details><summary>Explored 2 files, a search</summary>
>
> - Searched for `PartEvo|MLES|pip install|requirements|method|EoH|ReEvo|LLaMEA|assets/figs` in `README.md`
> - Read `./README.md`
> - Read `./README.md`
>
> </details>
>
> - Ran `git diff --stat origin/main HEAD`
>
> - Ran `git merge-base HEAD origin/main`
>
> - Ran `git status --short --branch`
>
> </details>
>
> 我已经有足够证据区分“会立刻跑不起来”的问题和“实验结论会偏”的问题。最后我再检查 EoH-RL 主循环的样本预算、LoRA 接受门和 checkpoint/resume 相关逻辑，避免漏掉系统性风险。
>
> <details><summary>Explored 7 files, 2 searches, ran 3 commands</summary>
>
> <details><summary>Explored 13 files, 2 searches</summary>
>
> - Read `./llm4ad/tools/rl/backends/online_grpo.py`
> - Read `./llm4ad/tools/rl/backends/trl_grpo_backend.py`
> - Read `./llm4ad/method/eoh_rl/eoh_rl.py`
> - Read `./llm4ad/method/eoh_rl/eoh_rl.py`
> - Read `./llm4ad/method/eoh_rl/eoh_rl.py`
> - Read `./llm4ad/tools/rl/backends/trl_grpo_backend.py`
> - Searched for `remaining_budget|group_size|_continue_loop|_rl_update_count|max_generations|max_sample_nums|records_to_dataset|train_update` in `eoh_rl.py`
> - Read `./llm4ad/method/eoh_rl/eoh_rl.py`
> - Read `./llm4ad/method/eoh_rl/eoh_rl.py`
> - Read `./llm4ad/tools/rl/backends/trl_grpo_backend.py`
> - Read `./llm4ad/tools/rl/backends/__init__.py`
> - Read `./llm4ad/tools/rl/__init__.py`
> - Searched for `EoHRL|run_sft_and_evolve|ResidentUnsloth|OnlineCompletionReward|compileall|pytest|collapse|AdaptiveOperator` in `tests`
> - Read `./tests/test_adaptive_collapse_and_scheduler.py`
> - Read `./llm4ad/method/eoh/profiler.py`
>
> </details>
>
> - Ran `git diff origin/main HEAD -- README.md setup.py llm4ad/method/__init__.py | sed -n '1,240p'`
>
> - Ran `git diff origin/main HEAD -- llm4ad/method/eoh/profiler.py | sed -n '1,160p'`
>
> - Ran `git log --oneline --decorate --max-count=8 --all`
>
> </details>
>
> 我还会扫一遍新增任务注册和几个本地 RL 示例脚本，因为这类项目最容易出现“核心类看起来成立，但脚本和配置把任务规模、minimize/maximize、日志归属接错”的问题。
>
> <details><summary>Explored 6 files</summary>
>
> - Read `./llm4ad/tools/rl/task_registry.py`
> - Read `./example/run_eoh/tsp/run_eoh_local_rl.py`
> - Read `./example/run_eoh/tsp/run_eoh_local_rl.py`
> - Read `./llm4ad/task/optimization/tsp_construct/evaluation.py`
> - Read `./llm4ad/tools/rl/task_registry.py`
> - Read `./llm4ad/task/optimization/cvrp_construct/evaluation.py`
>
> </details>
>
> 任务评估层也有问题：CVRP evaluator 对候选返回值没有 fail-fast，模型只要反复返回 depot 就能让评估循环不前进，最后靠外层 timeout。对 RL 来说，这会把“明显非法动作”变成昂贵的超时样本。
>
> <details><summary>Explored 3 files, a search, ran a command</summary>
>
> <details><summary>Explored 2 files, a search</summary>
>
> - Read `./llm4ad/task/optimization/jssp_construct/evaluation.py`
> - Read `./llm4ad/task/optimization/online_bin_packing/evaluation.py`
> - Searched for `def _evaluate|_capture_eval_diag|eval_exception|timeout|process.terminate|_pack_eval_result` in `evaluate.py`
>
> </details>
>
> - Ran `git diff origin/main HEAD -- llm4ad/task/optimization/cvrp_construct/evaluation.py llm4ad/task/optimization/tsp_construct/evaluation.py llm4ad/task/optimization/jssp_construct/evaluation.py llm4ad/task/optimization/online_bin_packing/evaluation.py | sed -n '1,220p'`
>
> <details><summary>Explored 2 files</summary>
>
> - Read `./llm4ad/base/evaluate.py`
> - Read `./llm4ad/base/evaluate.py`
>
> </details>
>
> </details>
>
> 任务的打分方向看起来是“返回负目标值、奖励按 maximize 处理”，这和 YAML 的 `minimize: false` 是一致的；这里不是主要问题。真正的问题是非法候选没有被便宜地拒绝，而是经常靠异常或 timeout 间接淘汰。
>
> <details><summary>Explored 6 files</summary>
>
> - Read `./llm4ad/tools/rl/rewards.py`
> - Read `./llm4ad/tools/rl/rewards.py`
> - Read `./llm4ad/tools/rl/collapse.py`
> - Read `./llm4ad/tools/rl/operator_scheduler.py`
> - Read `./llm4ad/tools/rl/collapse.py`
> - Read `./llm4ad/base/evaluate.py`
>
> </details>
>
> 坍缩配置还有一个可复现性问题：`CollapseConfig` 把 `recovery_target_size` 写死为启用时必须等于 10，而 `EoHRL` 又要求 `pop_size` 等于这个值。这会让“是否启用坍缩”改变可用种群规模，而不是一个独立机制。
>
> <details><summary>Explored 4 files, 2 searches</summary>
>
> - Searched for `collapse:|recovery_target_size|pop_size:|adaptive_operator:|enabled: true` in `configs`
> - Searched for `recovery_target_size 必须为 10|collapse.enabled=true 时 evolution/scale pop_size` in `collapse.py`
> - Read `./llm4ad/tools/rl/collapse.py`
> - Read `./llm4ad/method/eoh_rl/eoh_rl.py`
> - Read `./configs/experiments/eoh_rl_tsp_muti_np1_adaptive_collapse_gpu1.yaml`
> - Read `./configs/experiments/eoh_rl_tsp_muti_np1_adaptive_collapse_gpu1.yaml`
>
> </details>
>
> 本地单任务 RL 脚本比两阶段脚本更完整：它读取 `collapse_config`，也明显在配置里把启用坍缩的实验 pop_size 固定成 10。这个进一步说明两阶段入口是落后/不一致的，而不是核心类不需要该参数。
>
> <details><summary>Explored 2 files, 2 searches, ran a command</summary>
>
> <details><summary>Explored 2 files, a search</summary>
>
> - Searched for `cold_start_from_base|skip_sft|sft_lora_path|lora_sft|initial_lora_path` in `run_eoh_local_rl.py`
> - Read `./example/run_eoh/tsp/run_eoh_local_rl.py`
> - Read `./llm4ad/tools/rl/backends/__init__.py`
>
> </details>
>
> - Ran `git diff --name-only origin/main HEAD | wc -l`
>
> <details><summary>Explored a file, a search</summary>
>
> - Searched for `def create_rl_backend|validate_rl_backend_config` in `__init__.py`
> - Read `./llm4ad/tools/rl/backends/__init__.py`
>
> </details>
>
> </details>
>
> 单任务入口还证明了应该如何修两阶段脚本：它把 `cold_start_from_base=true` 映射成 `initial_lora_path=None`，并把 collapse 配置传给 `EoHRL`。我准备汇总了。
>
> <details><summary>Explored 2 files, 3 searches, ran a command</summary>
>
> <details><summary>Explored 2 files, 3 searches</summary>
>
> - Searched for `validate_inference_training_lifecycle` in `.`
> - Searched for `EoHRL\(|collapse_config=|AdaptiveOperatorConfig\.from_dict|CollapseConfig\.from_dict` in `distill_and_evolve`
> - Read `./llm4ad/tools/rl/llm_factory.py`
> - Searched for `inference_backend|unsloth_resident|resident_unsloth_trl|create_eohrl_llm|RLAugmentedLLM|local_model_path` in `llm_factory.py`
> - Read `./llm4ad/tools/llm/inference_backends.py`
>
> </details>
>
> - Ran `git status --short --branch`
>
> </details>
>
> 我看到生命周期校验函数目前没有调用点；如果外部直接用 `create_eohrl_llm`，仍可能走到旧的 `RLAugmentedLLM` 路径。这不如入口硬失败严重，但说明配置校验分散，主路径约束没有统一入口。
>
> <details><summary>Explored 3 files, 2 searches, ran a command</summary>
>
> <details><summary>Explored 3 files, 2 searches</summary>
>
> - Read `./llm4ad/tools/llm/rl_augmented_llm.py`
> - Read `./llm4ad/tools/rl/llm_factory.py`
> - Read `./llm4ad/tools/rl/model_manager.py`
> - Searched for `__pycache__` in `.`
> - Searched for `def validate|require_collapse_plain|require_adaptive_operator_plain|validate_inference_training_lifecycle|create_rl_backend` in `config`
>
> </details>
>
> - Ran `git diff --check`
>
> </details>
>
> <details><summary>Plan</summary>
>
> - [x] 对照 origin/main 梳理分支结构和新增 EoH-RL 设计
> - [x] 审查入口脚本、配置、依赖安装与主 README 兼容性
> - [x] 审查 RL 奖励/主循环、评估隔离、任务 evaluator 的系统性风险
> - [x] 运行静态/单元验证并汇总 findings
>
> </details>
</details>

**审查结果**
- **P0 两阶段主入口会直接失败**：`EoHRL` 启用 RL 时强制要求 `collapse_config`，见 [eoh_rl.py](./llm4ad/method/eoh_rl/eoh_rl.py:192)。但 [run_sft_and_evolve.py](./example/distill_and_evolve/run_sft_and_evolve.py:401) 构造 `EoHRL` 时只传了 `adaptive_operator_config`，没有传 `CollapseConfig.from_dict(...)`。同类单任务脚本是传了的，说明这是入口脚本漏改。
- **P0 默认冷启动配置不可用**：示例配置写着 `skip_sft: true` 且 `cold_start_from_base: true`，见 [eoh_rl.example.yaml](./configs/eoh_rl.example.yaml:63)。但两阶段入口在 `SKIP_SFT` 时仍强制查找 `lora_sft` 目录并报错，见 [run_sft_and_evolve.py](./example/distill_and_evolve/run_sft_and_evolve.py:474)。单任务脚本正确行为是 `cold_start_from_base` 时传 `initial_lora_path=None`，见 [run_eoh_local_rl.py](./example/run_eoh/tsp/run_eoh_local_rl.py:325)。
- **P0 安装元数据不支持新增主链路**：`setup.py` 的依赖只到基础包，缺 `omegaconf/datasets/unsloth/trl/vllm/peft/transformers/bitsandbytes/accelerate/ortools/networkx/codebleu` 等，见 [setup.py](./setup.py:30)；而这些在代码和 `requirements.txt` 里已经是运行路径依赖，见 [requirements.txt](./requirements.txt:9)。`pip install .` 后核心 EoH-RL 路径大概率不可用。
- **P1 在线 API 会无限重试**：`HttpsApi.draw_sample()` 是 `while True`，非 debug 模式永不抛出累计失败，见 [llm_api_https.py](./llm4ad/tools/llm/llm_api_https.py:209)。坏 key、坏 host、限流、响应 schema 变化都会把采集/初始化挂死。
- **P1 评估失败诊断仍然不可用于 RL 控制**：timeout 路径只返回 `None`，见 [evaluate.py](./llm4ad/base/evaluate.py:194)；诊断接口又把 `None` 统一压成 `eval_or_timeout`，见 [evaluate.py](./llm4ad/base/evaluate.py:272)。这会污染 reward、operator scheduler、collapse 的失败归因。
- **P1 CVRP evaluator 对非法动作不 fail-fast**：候选若反复返回 depot，`while unvisited_nodes` 不推进，只能靠 timeout 结束，见 [cvrp_construct/evaluation.py](./llm4ad/task/optimization/cvrp_construct/evaluation.py:153)。对后训练来说，这会把明显非法 completion 变成昂贵负样本。
- **P1 RL 优化目标和实际生存/部署目标不完全一致**：BQR reward 只按 score、父代、frontier、结构新颖性计算，见 [online_grpo.py](./llm4ad/tools/rl/backends/online_grpo.py:714)；候选注册和 `survived_main_population` 是后续记录，见 [online_grpo.py](./llm4ad/tools/rl/backends/online_grpo.py:495) 和 [eoh_rl.py](./llm4ad/method/eoh_rl/eoh_rl.py:1200)。这会让训练奖励和“进入/留在主种群”的搜索目标偏离。
- **P1 当前分支不是 main 的兼容增量**：和 `origin/main` 没有 merge-base，树差异是 236 个文件；还删除了 assets、MLES、PartEvo、environment.yaml 等，但 README 仍引用已删除图片和方法表，见 [README.md](./README.md:3) 和 [README.md](./README.md:183)。
- **P2 导入错误被吞掉**：`llm4ad.__init__` 和 `llm4ad.method.__init__` 捕获所有 `Exception` 并置为 `None`，见 [llm4ad/__init__.py](./llm4ad/__init__.py:10) 和 [method/__init__.py](./llm4ad/method/__init__.py:32)。这会隐藏真实实现错误，不只是缺可选依赖。

**设计理解**
当前分支的核心意图是：先用 EoH 初始化种群，再把 EoH operator 生成的 grouped prompts 交给 resident Unsloth/TRL GRPO；reward 回调即时评估 completion、写入 population，再按全局改进和 fidelity 决定是否接受 LoRA。自适应算子和坍缩机制再基于 reward events 调整搜索分布。

**验证**
我做了只读审查，没有改代码。`python -m compileall -q llm4ad tests scripts example/distill_and_evolve` 通过；`python -m pytest -q tests/test_adaptive_collapse_and_scheduler.py` 通过 35 项；`git diff --check` 通过。测试覆盖主要集中在 scheduler/collapse 的局部逻辑，未覆盖两阶段入口、真实 Unsloth/TRL/vLLM 生命周期、默认 cold-start 配置和 API 失败路径。

> 我们实际主要应该是进行的本地EOH类型的方法搜索和训练的循环？可能核心创新点还有坍缩机制，自适应算子权重等等？

是的。按现在代码的真实方向，主线应该表述为：

**本地 EoH-style 搜索和本地后训练的在线闭环**，而不是传统“先采样一批数据，再离线训练，再重新搜索”。

核心循环是：

1. 用本地模型生成 EoH 候选，初始化 population。
2. 从当前 population 构造 `m1/m2/e1/e2` 这类 EoH operator prompt。
3. 本地 resident Unsloth/vLLM 生成 grouped completions。
4. 评估器即时打分，BQR reward 做父代/frontier 相对奖励塑形。
5. 合格候选直接注册进 population。
6. 同一批 completion 同时用于 GRPO 更新 LoRA。
7. LoRA 通过 global improvement + fidelity gate 决定接受或回滚。
8. 自适应算子权重和坍缩机制根据 reward events 继续影响后续搜索。

代码上这个主线主要落在 [EoHRL](./llm4ad/method/eoh_rl/eoh_rl.py:1040)、[online_grpo.py](./llm4ad/tools/rl/backends/online_grpo.py:120)、[trl_grpo_backend.py](./llm4ad/tools/rl/backends/trl_grpo_backend.py:255)。

我会把核心创新点归纳成四个：

1. **EoH 搜索和 GRPO 后训练在线耦合**
   不是把 EoH 当数据生成器，而是每轮搜索样本立即影响 policy，policy 又影响下一轮 EoH operator 搜索。

2. **BQR 奖励**
   奖励不是直接用 raw score，而是相对父代和 frontier 的 bounded reward，并加入结构新颖性、随机性检测、metadata leak、parent copy gate。核心在 [rewards.py](./llm4ad/tools/rl/rewards.py:157)。

3. **自适应算子权重**
   `m1/m2/e1/e2` 不应固定轮换，而是按最近 reward events、frontier improvement、parent improvement、失败率、结构违规率动态调权。核心在 [operator_scheduler.py](./llm4ad/tools/rl/operator_scheduler.py:280)。

4. **坍缩与恢复机制**
   长期停滞时压缩 population，保留核心个体，再用 recovery 策略重新填充，目的是逃离同质化和局部搜索停滞。核心在 [collapse.py](./llm4ad/tools/rl/collapse.py:121)。

但要注意：这些是“设计上看起来的创新点”。实现上我前面指出的问题会影响它们是否真的成立，尤其是入口脚本不一致、reward 与 population survival 未完全对齐、失败诊断过粗、非法候选靠 timeout 淘汰。下一步如果要继续审查，我建议按这四个创新点逐一验证“机制是否真的按论文/设想生效”。

> 那么这些的实现是否实际存在问题？

<details><summary>6 previous messages</summary>

> 我会把问题限定在你刚刚说的三块：本地 EoH 搜索-训练闭环、坍缩机制、自适应算子权重。这里不是“风格问题”，而是这些机制现在能否稳定、可解释地支撑实验结论。
>
> <details><summary>Explored 5 files, a search</summary>
>
> - Read `./llm4ad/tools/rl/backends/online_grpo.py`
> - Read `./llm4ad/tools/rl/backends/trl_grpo_backend.py`
> - Searched for `num_train_epochs|per_device_train_batch_size|num_generations|prompts_per_update|gradient_accumulation_steps|save_strategy|remove_unused_columns` in `eoh_rl.example.yaml`
> - Read `./llm4ad/tools/rl/operator_scheduler.py`
> - Read `./llm4ad/tools/rl/operator_scheduler.py`
> - Read `./llm4ad/method/eoh_rl/eoh_rl.py`
>
> </details>
>
> 我看到一个关键风险：reward function 本身会评估并写 population，这是把训练回调变成有副作用的搜索步骤。当前配置 `num_train_epochs=1` 降低了风险，但只要 TRL 调用模式、epoch 数或重试策略变化，这个闭环就可能重复注册或改变搜索状态。
>
> <details><summary>Explored 5 files, a search</summary>
>
> - Searched for `def _train_with_trl|GRPOConfig|GRPOTrainer|reward_funcs|num_train_epochs|num_generations|dataset` in `trl_grpo_backend.py`
> - Read `./llm4ad/tools/rl/backends/trl_grpo_backend.py`
> - Read `./llm4ad/tools/rl/backends/online_grpo.py`
> - Read `./llm4ad/method/eoh_rl/eoh_rl.py`
> - Read `./llm4ad/tools/rl/backends/online_grpo.py`
> - Read `./llm4ad/method/eoh_rl/eoh_rl.py`
>
> </details>
>
> 另一个具体点：`prompts_per_update=1` 时，自适应算子每轮最多只观测一个 operator 的一组 completion。权重更新的统计粒度很小，早期信号会非常噪，而且 fallback 还会强制把没有 mutation 的批次改成 `m2`，这会给算子信用分配带来偏差。
>
> <details><summary>Explored 6 files</summary>
>
> - Read `./llm4ad/tools/rl/backends/trl_grpo_backend.py`
> - Read `./llm4ad/tools/rl/backends/trl_grpo_backend.py`
> - Read `./llm4ad/tools/rl/operator_scheduler.py`
> - Read `./llm4ad/tools/rl/operator_scheduler.py`
> - Read `./llm4ad/method/eoh_rl/eoh_rl.py`
> - Read `./llm4ad/tools/rl/operator_scheduler.py`
>
> </details>
</details>

是，实际存在问题。不是“这些机制没写”，而是**机制写了，但闭环、信用分配、实验可比性和失败处理都有硬伤**。

**主要问题**
1. **搜索-训练闭环有副作用耦合**
   `OnlineCompletionReward.__call__()` 在 TRL 训练 reward 回调里直接做解析、评估、注册 population，见 [online_grpo.py](./llm4ad/tools/rl/backends/online_grpo.py:109)。这个 reward function 被 `GRPOTrainer` 作为 `reward_funcs` 使用，见 [trl_grpo_backend.py](./llm4ad/tools/rl/backends/trl_grpo_backend.py:97)，随后 `trainer.train()` 触发，见 [trl_grpo_backend.py](./llm4ad/tools/rl/backends/trl_grpo_backend.py:124)。
   这意味着训练框架的调用次数、batch 行为、epoch 行为会直接改变搜索状态。当前配置 `num_train_epochs=1` 降低风险，但设计上不稳。更干净的结构应该是：先 rollout/evaluate/register，冻结一份 rollout dataset，再做 GRPO 更新。

2. **RL 奖励没有真正对齐 population survival**
   BQR reward 主要看“是否优于父代/是否优于 frontier/结构新颖性”，见 [rewards.py](./llm4ad/tools/rl/rewards.py:250)。但候选是否注册、是否留在主 population，是后处理事件，见 [online_grpo.py](./llm4ad/tools/rl/backends/online_grpo.py:165) 和 [eoh_rl.py](./llm4ad/method/eoh_rl/eoh_rl.py:1200)。
   结果是 policy 可能学习“相对父代有提升但不真正推动主种群前沿”的样本。

3. **BQR 的结构新颖奖励几乎可能是死的**
   `ties_parent` 用的是 `abs(delta) <= epsilon`，默认 `epsilon=1e-6`，见 [rewards.py](./llm4ad/tools/rl/rewards.py:199)。只有几乎完全同分时才进入 q2 结构新颖奖励，见 [rewards.py](./llm4ad/tools/rl/rewards.py:260)。对 TSP/CVRP/JSSP 这种浮点评估，精确 tie 很少，结构新颖性多数时候只会变成 q3 同质性小惩罚，而不是正向探索奖励。

4. **自适应算子权重的统计太薄**
   默认 `prompts_per_update=1`，每次只观测一个 operator 的一组 completion，见 [eoh_rl.example.yaml](./configs/eoh_rl.example.yaml:84)。算子信用分配基于 `frontier_count/parent_count/failure_count`，见 [operator_scheduler.py](./llm4ad/tools/rl/operator_scheduler.py:280)。
   这会导致早期少量 noisy 样本迅速影响概率。机制存在，但统计基础偏弱，实验结论需要更强 ablation 和置信区间。

5. **坍缩机制触发条件不适合“从一开始就停滞”的场景**
   collapse 需要 `history_ready = best_interval_count >= 1`，见 [collapse.py](./llm4ad/tools/rl/collapse.py:177)。如果初始化后一直没有产生过新的 global improvement，`history_ready` 永远不成立，坍缩不会触发。
   这和“用于摆脱停滞/同质化”的目标冲突。

6. **坍缩机制强行绑定 population size=10**
   `CollapseConfig` 启用时要求 `recovery_target_size == 10`，见 [collapse.py](./llm4ad/tools/rl/collapse.py:27)；`EoHRL` 又要求 `pop_size == recovery_target_size`，见 [eoh_rl.py](./llm4ad/method/eoh_rl/eoh_rl.py:197)。
   这会让“启用坍缩”同时改变种群规模，破坏和 fixed/no-collapse baseline 的公平比较。

7. **失败诊断过粗，会污染自适应权重和坍缩信号**
   timeout、非法返回、任务约束失败大量被压成 `None` 或 `eval_or_timeout`，见 [evaluate.py](./llm4ad/base/evaluate.py:272)。算子调权又把这些粗失败用于 `failure_count`，见 [operator_scheduler.py](./llm4ad/tools/rl/operator_scheduler.py:292)。
   所以权重更新不一定是在学习“哪个算子好”，也可能是在学习 evaluator 的失败噪声。

**判断**
核心创新点确实在代码里：本地 EoH-RL 闭环、BQR、自适应算子、坍缩恢复都存在。但目前实现还不能算严谨成立。最大问题是：**训练回调带搜索副作用、reward 与主种群生存不对齐、坍缩触发和 pop size 设计不合理、自适应算子统计信号偏弱**。这些会直接影响实验结论可信度。

> 实际上还可以看一些实验代码来理解设计

<details><summary>19 previous messages</summary>

> 对，这里应该把“实验脚本实际怎么跑”纳入设计理解，否则只看核心类容易误判。接下来我会看配置矩阵、ablation 脚本和本地 `run_eoh_local_rl.py`，重点找它们到底在比较 fixed / adaptive / collapse / combined 哪几组，以及有没有把变量控制干净。
>
> <details><summary>Explored 2 searches, 3 lists</summary>
>
> - Listed files in `configs`
> - Searched for `(自适应|坍缩|adaptive|collapse|local_rl|eoh_rl_.*gpu|对比实验|混合规模)`
> - Listed files in `configs`
> - Searched for `VARIANT|variant|adaptive|collapse|fixed|pop_size|recovery_target|seed|runs|matrix|PLAN|任务|规模|config` in `scripts`
> - Listed files in `scripts`
>
> </details>
>
> 目前看到实验目录分成两层：`configs/对比实验` 是小规模单任务 ablation，`configs/混合规模训练` 是 multi-scale / np1 版本；scripts 再动态改 YAML 生成 variant。下面我会读生成配置的 Python 片段，这是判断变量是否控制干净的关键。
>
> <details><summary>Explored 6 files</summary>
>
> - Read `./scripts/运行B组tsp自适应算子坍缩完整矩阵_gpu1.sh`
> - Read `./scripts/运行B组tsp无自适应无坍缩完整矩阵_gpu0.sh`
> - Read `./scripts/运行B组单问题多规模自适应算子坍缩_np1_gpu1.sh`
> - Read `./scripts/run_cvrp50_adaptive_op_only_gpu1.sh`
> - Read `./scripts/run_tsp50_fixed_no_collapse_gpu3.sh`
> - Read `./scripts/run_tsp50_collapse_only_gpu2.sh`
>
> </details>
>
> 这些脚本显示出真实实验意图：TSP 有完整 2x2 ablation（fixed / adaptive / collapse / both）加 np=1/4；CVRP 单独只跑 adaptive-only；multi-scale 只跑 adaptive+collapse。下一步我看 YAML 细节，确认 baseline 是否只是开关不同，还是同时换了预算、SFT、pop_size、尺度。
>
> <details><summary>Explored 6 files</summary>
>
> - Read `./configs/对比实验/B组自适应算子坍缩_gpu1/eoh_rl_tsp_small_v3_full_gpu1.yaml`
> - Read `./configs/对比实验/B组自适应算子坍缩_gpu1/eoh_rl_cvrp_small_v3_full_gpu1.yaml`
> - Read `./configs/混合规模训练/B组自适应算子坍缩_np1_gpu1/eoh_rl_tsp_muti_np1_adaptive_collapse_gpu1.yaml`
> - Read `./configs/混合规模训练/B组自适应算子坍缩_np1_gpu1/eoh_rl_cvrp_muti_np1_adaptive_collapse_gpu1.yaml`
> - Read `./configs/eoh_rl_tsp_small_4bit_cold.yaml`
> - Read `./configs/eoh_rl_cvrp_small_4bit_cold.yaml`
>
> </details>
>
> Context compaction
>
> Source: automatic
> Status: completed
>
> 我已经把主实现、RL 后端、调度器、坍缩管理和一批实验脚本串起来看过了。现在补看混合规模实验和入口脚本，确认“设计声称”和“实际跑到的路径”是否一致，然后给你按严重程度列实现问题。
>
> <details><summary>Explored 3 files, a search</summary>
>
> - Read `./llm4ad/tools/rl/mixed_scale_evaluation.py`
> - Read `./example/run_eoh/tsp/run_eoh_local_rl.py`
> - Read `./example/run_eoh/cvrp/run_eoh_local_rl.py`
> - Searched for `mixed_scales|MixedScale|score_normalizer|state_scope|composite_score|scale` in `llm4ad`
>
> </details>
>
> 实验代码已经给出一个很关键的信号：所谓“collapse-only”消融在脚本里确实被配置出来了，但核心构造器又禁止 collapse 在 adaptive operator 关闭时启用。这不是解释层面的争议，是会直接让一组实验启动失败的实现矛盾。
>
> <details><summary>Explored 16 files</summary>
>
> - Read `./llm4ad/method/eoh_rl/eoh_rl.py`
> - Read `./scripts/run_tsp50_collapse_only_gpu2.sh`
> - Read `./scripts/运行B组tsp自适应算子坍缩完整矩阵_gpu1.sh`
> - Read `./scripts/run_tsp50_fixed_no_collapse_gpu3.sh`
> - Read `./example/distill_and_evolve/run_sft_and_evolve.py`
> - Read `./example/run_eoh/tsp/run_eoh_local_rl.py`
> - Read `./llm4ad/method/eoh_rl/eoh_rl.py`
> - Read `./llm4ad/tools/rl/backends/online_grpo.py`
> - Read `./example/run_eoh/tsp/run_eoh_local_rl.py`
> - Read `./llm4ad/tools/rl/operator_scheduler.py`
> - Read `./example/run_eoh/cvrp/run_eoh_local_rl.py`
> - Read `./llm4ad/tools/rl/collapse.py`
> - Read `./llm4ad/tools/rl/backends/trl_grpo_backend.py`
> - Read `./example/run_eoh/cvrp/run_eoh_local_rl.py`
> - Read `./llm4ad/tools/rl/rewards.py`
> - Read `./llm4ad/base/evaluate.py`
>
> </details>
>
> 目前主链路可以确认：本地 EoH 负责 prompt/种群，TRL/Unsloth 的 GRPO reward callback 里直接评估并把候选写回种群；调度器和坍缩都读这一批 reward event 再更新状态。这个设计是存在的，但几个实现点会让实验结论不可靠，尤其是消融和坍缩触发。
>
> <details><summary>Explored 9 files, a search, ran 3 commands</summary>
>
> <details><summary>Explored 9 files</summary>
>
> - Read `./llm4ad/base/evaluate.py`
> - Read `./llm4ad/task/optimization/cvrp_construct/evaluation.py`
> - Read `./configs/eoh_rl.example.yaml`
> - Read `./setup.py`
> - Read `./requirements.txt`
> - Read `./configs/混合规模训练/B组自适应算子坍缩_np1_gpu1/eoh_rl_tsp_muti_np1_adaptive_collapse_gpu1.yaml`
> - Read `./configs/混合规模训练/B组自适应算子坍缩_np1_gpu1/eoh_rl_cvrp_muti_np1_adaptive_collapse_gpu1.yaml`
> - Read `./scripts/运行B组单问题多规模自适应算子坍缩_np1_gpu1.sh`
> - Read `./llm4ad/tools/rl/mixed_scale_evaluation.py`
>
> </details>
>
> - Ran `git merge-base HEAD origin/main; git rev-parse --short HEAD; git rev-parse --short origin/main`
>
> <details><summary>Explored a search</summary>
>
> - Searched for `collapse_only|adaptive_op_plus_collapse|adaptive_op_only|fixed_no_collapse|VARIANT_PLAN_OVERRIDE|MAX_SAMPLE_NUMS|TARGET_GPU|TARGET_PORT|SFT_OUTPUT_DIR|seed:` in `scripts`
>
> </details>
>
> - Ran `git diff --stat origin/main HEAD`
>
> - Ran `git status --short --branch`
>
> </details>
>
> 我还检查了测试和基础配置。已有测试能覆盖一些配置校验和调度器行为，但没覆盖“实验脚本生成的消融配置是否能跑通”、reward callback 对种群的副作用、以及混合规模 reward 的公平性。
>
> <details><summary>Explored a file, a search, ran 2 commands</summary>
>
> <details><summary>Explored a file, a search</summary>
>
> - Read `./tests/test_adaptive_collapse_and_scheduler.py`
> - Searched for `collapse_only|run_tsp|script|from_dict|CollapseConfig|AdaptiveOperatorScheduler|observe_update|prompts_per_update|MixedScale` in `tests`
>
> </details>
>
> - Ran `python -m compileall -q llm4ad tests scripts example/distill_and_evolve`
>
> - Ran `python -m pytest -q tests/test_adaptive_collapse_and_scheduler.py`
>
> </details>
>
> 另外一个实现层面的模式也很明确：LoRA 的“部署验收/回滚”和 EoH 种群更新不是一个事务。即使本轮 LoRA 被判定不部署，reward callback 已经可能把候选写入了种群，所以“坏策略不影响搜索状态”这个结论不能从代码推出。
>
> <details><summary>Explored 3 files, a search</summary>
>
> - Searched for `latest_lora_path|set_latest_lora_path|accept_trained_lora|restore_trainable_state|reload_deployed_policy|snapshot_trainable_state` in `model_manager.py`
> - Read `./llm4ad/tools/rl/model_manager.py`
> - Read `./llm4ad/tools/rl/checkpoint.py`
> - Read `./llm4ad/method/eoh_rl/resume.py`
>
> </details>
>
> 还看到两套模型管理代码并存：旧的 `tools/rl/model_manager.py` 和新的 resident Unsloth 管理器。当前 GRPO 后端走的是 resident manager；因此我把 LoRA 路径和 checkpoint 恢复也按 resident 路径追了一遍，避免把旧代码误判成主路径。
>
> <details><summary>Explored 7 files, 2 searches</summary>
>
> - Read `./llm4ad/tools/llm/resident_unsloth_llm.py`
> - Read `./llm4ad/tools/rl/checkpoint.py`
> - Searched for `save_checkpoint|build_state|latest_lora_path|get_latest_lora_path|set_latest_lora_path|current_lora_path|deployed_lora_path|anchor_lora_path` in `eoh_rl.py`
> - Read `./llm4ad/method/eoh_rl/eoh_rl.py`
> - Searched for `mixed_scales|create_mixed_scale_evaluation|scale_name = \"mixed\"|mixed_scales_enabled` in `run_eoh_local_rl.py`
> - Read `./scripts/运行B组tsp自适应算子坍缩完整矩阵_gpu2.sh`
> - Read `./scripts/运行B组tsp无自适应无坍缩完整矩阵_gpu0.sh`
> - Read `./scripts/run_tsp50_adaptive_op_only_gpu1.sh`
> - Read `./scripts/run_tsp50_adaptive_op_plus_collapse_gpu0.sh`
>
> </details>
</details>

结论：核心设计确实是“本地 EoH 搜索 + 在线 GRPO 后训练”的闭环，且自适应算子权重、坍缩恢复机制都在代码里存在。但实现和实验脚本之间有严重不一致，足以影响实验是否能跑通以及结论是否可信。

对照 `origin/main` 时要注意：当前 `HEAD=109a743` 与 `origin/main=0dd9b3d` 没有 merge-base，我只能做 tree diff，不是严格 PR diff。主要变化是新增 EoH-RL/TRL/Unsloth/实验脚本，同时删除了不少旧方法和资源。

**主要问题**

- **P0：`collapse_only` 消融实验会直接失败。** 实验脚本把 `collapse_only` 配成 `adaptive_operator=false, collapse=true`，见 [运行B组tsp自适应算子坍缩完整矩阵_gpu1.sh](./scripts/运行B组tsp自适应算子坍缩完整矩阵_gpu1.sh:126)。但核心构造器明确禁止这种组合，见 [eoh_rl.py](./llm4ad/method/eoh_rl/eoh_rl.py:253)。所以所谓 2x2 消融矩阵并不成立。

- **P0：旧入口 `run_sft_and_evolve.py` 与当前 EoHRL 构造约束不兼容。** EoHRL 在 RL 启用时要求显式传入 `collapse_config`，见 [eoh_rl.py](./llm4ad/method/eoh_rl/eoh_rl.py:193)，但该入口只传了 `adaptive_operator_config`，没有传 `collapse_config`，见 [run_sft_and_evolve.py](./example/distill_and_evolve/run_sft_and_evolve.py:447)。这个入口如果继续作为实验入口会启动失败。

- **P1：坍缩机制最需要工作的时候可能永远不触发。** `CollapseManager` 要求已经观察到至少一个“全局最优改进间隔”才认为 `history_ready`，否则一直返回 `history_insufficient`，见 [collapse.py](./llm4ad/tools/rl/collapse.py:178)。如果初始化后一直停滞，没有后续全局改进，坍缩机制反而不会触发。

- **P1：LoRA 回滚不是搜索状态回滚。** GRPO reward callback 会在训练过程中直接注册候选进 EoH 种群，见 [online_grpo.py](./llm4ad/tools/rl/backends/online_grpo.py:165)。之后 LoRA 才做部署验收和回滚，见 [trl_grpo_backend.py](./llm4ad/tools/rl/backends/trl_grpo_backend.py:302)。所以“拒绝坏 LoRA”并不等于“拒绝本轮坏探索对种群的影响”。

- **P1：自适应算子权重不是一个充分可解释的 learned weighting。** 当前主要是固定先验 + 运行平均信号，先验写死为 `m1/m2/e1/e2 = 0.30/0.45/0.15/0.10`，见 [operator_scheduler.py](./llm4ad/tools/rl/operator_scheduler.py:11)。信号又把“击败父代”和“击败全局 frontier”混在一起，见 [operator_scheduler.py](./llm4ad/tools/rl/operator_scheduler.py:331)，而 BQR 的 q1 只要击败父代就给突破奖励，见 [rewards.py](./llm4ad/tools/rl/rewards.py:250)。这会偏向局部父代改进，不一定优化全局搜索效率。

- **P1：实验脚本实际跑的是 SFT 初始化，不是冷启动。** 单规模脚本强制 `skip_sft=true` 且 `cold_start_from_base=false`，见 [运行B组tsp自适应算子坍缩完整矩阵_gpu1.sh](./scripts/运行B组tsp自适应算子坍缩完整矩阵_gpu1.sh:144)。多规模脚本也一样，见 [运行B组单问题多规模自适应算子坍缩_np1_gpu1.sh](./scripts/运行B组单问题多规模自适应算子坍缩_np1_gpu1.sh:132)。如果论文/报告声称是基座模型直接在线训练，需要改说法或改脚本。

- **P2：多规模 reward 的归一化是手工常数，且一处失败整条候选失败。** Mixed-scale evaluation 对每个 scale 的 raw score 除以 `score_normalizer` 后平均，见 [mixed_scale_evaluation.py](./llm4ad/tools/rl/mixed_scale_evaluation.py:144)；任一 scale 返回 `None` 就整体失败，见 [mixed_scale_evaluation.py](./llm4ad/tools/rl/mixed_scale_evaluation.py:121)。TSP/CVRP 多规模配置还使用相同的 `7.071/10/14.142` 正则化常数，见 [eoh_rl_cvrp_muti_np1_adaptive_collapse_gpu1.yaml](./configs/混合规模训练/B组自适应算子坍缩_np1_gpu1/eoh_rl_cvrp_muti_np1_adaptive_collapse_gpu1.yaml:182)，这需要实验依据支撑。

- **P2：评估失败诊断太粗，调度器/坍缩统计会被污染。** timeout 最后只变成 `None`，见 [evaluate.py](./llm4ad/base/evaluate.py:204)；诊断接口也承认无法区分 timeout/exec/合法 None，见 [evaluate.py](./llm4ad/base/evaluate.py:272)。例如 CVRP heuristic 返回 depot 时可能不推进，最后靠超时或异常兜底，见 [evaluation.py](./llm4ad/task/optimization/cvrp_construct/evaluation.py:153)。

验证情况：`python -m compileall -q llm4ad tests scripts example/distill_and_evolve` 通过；`python -m pytest -q tests/test_adaptive_collapse_and_scheduler.py` 通过，35 个测试。但这些测试没有覆盖实验脚本生成配置的可运行性，也没有覆盖 LoRA 回滚与种群副作用的一致性。

> 目前实际上我遇到的最大的问题是进行消融实验，比起传统eoh（使用本地模型的情况下），加上自适应算子权重和坍缩机制效果比基础的好，但是去掉任意其中一个，反而会比现在两个机制都加的效果更好？

<details><summary>7 previous messages</summary>

> 这个现象本身不奇怪，但在你这个实现里我会先把它当成“机制耦合/消融定义有问题”，而不是先解释成统计波动。我再把坍缩执行和 recovery 期间算子调度的代码追一下，确认两个机制叠加时到底额外改变了哪些状态。
>
> <details><summary>Explored 3 files, a search</summary>
>
> - Searched for `execute_population_collapse|finish_recovery|recovery_active|shrink_memory|retained_core|simple_pool|recovery_bootstrap|observe_collapse|adaptive_operator_context` in `eoh_rl.py`
> - Read `./llm4ad/method/eoh_rl/eoh_rl.py`
> - Read `./llm4ad/tools/rl/operator_scheduler.py`
> - Read `./llm4ad/method/eoh_rl/eoh_rl.py`
>
> </details>
>
> 两个机制叠加时，坍缩不是单纯“清种群再恢复”：它保留 top core、维护一个偏简单程序的 `simple_pool`，并且恢复期第一轮强制 `i1`，后续 recovery 概率又由自适应调度器决定。这会改变父代分布、算子分布和 GRPO 训练样本分布，确实可能让“两个都开”比“只开一个”差。
>
> <details><summary>Explored 4 files, 2 searches</summary>
>
> - Read `./llm4ad/method/eoh_rl/eoh_rl.py`
> - Read `./llm4ad/tools/rl/collapse.py`
> - Read `./llm4ad/method/eoh_rl/eoh_rl.py`
> - Searched for `def _execute_population_collapse|def _finish_recovery_if_full|def _recovery_prior_distribution|def _build_online_prompt_records|bootstrap_pending|target_n = 1|sampled_ops = \[\"i1\"\]` in `eoh_rl.py`
> - Read `./llm4ad/tools/rl/operator_scheduler.py`
> - Searched for `compute_recovery_probabilities|_sample_recovery_ops|update_recovery_from_summary` in `llm4ad`
>
> </details>
>
> 还有一个很具体的实现发现：代码里写了 recovery 专用算子概率 `compute_recovery_probabilities()`，也写了 `_sample_recovery_ops()`，但主 prompt 构造路径没有调用它；恢复期除了第一轮强制 `i1` 外，后面还是走普通 adaptive `select_ops()`。这会让“坍缩恢复 + 自适应权重”的实际行为和设计意图不一致。
>
> <details><summary>Explored 2 files, 2 searches</summary>
>
> - Searched for `activate_anchor_policy|deactivate_anchor_policy|anchor_recovery|anchor_policy` in `resident_unsloth_llm.py`
> - Read `./llm4ad/tools/llm/resident_unsloth_llm.py`
> - Searched for `random.seed|set_seed|seed` in `eoh_rl`
> - Read `./llm4ad/tools/rl/backends/trl_grpo_backend.py`
>
> </details>
>
> 我也确认了坍缩会触发模型侧的 anchor policy：坍缩时切回 anchor，恢复满种群后再退出。这样“坍缩机制”不是单纯种群操作，而是同时改变 rollout policy，这会让和自适应算子的交互更复杂。
</details>

是的，这个现象更像是**组合机制存在负交互**，不是单纯“实验波动”。而且你当前实现里还有几个会直接放大这个现象的问题。

最关键的判断：

1. **“坍缩 + 自适应”不是两个独立增益相加。**
   坍缩会把种群压到 1-2 个核心个体，见 [eoh_rl.py](./llm4ad/method/eoh_rl/eoh_rl.py:2276)。自适应算子又根据近期 reward event 调整算子分布。坍缩后父代数少、结构多样性低，自适应调度很容易被短期信号带偏，特别是 `prompts_per_update=1` 时信用分配方差很大。

2. **当前 recovery 专用算子概率几乎没真正用到。**
   代码有 `compute_recovery_probabilities()` 和 `_sample_recovery_ops()`，但主 prompt 构造路径在 recovery bootstrap 后仍然调用普通 `select_ops()`，见 [eoh_rl.py](./llm4ad/method/eoh_rl/eoh_rl.py:1563) 和 [operator_scheduler.py](./llm4ad/tools/rl/operator_scheduler.py:67)。所以“坍缩后的自适应恢复策略”实际没有按设计运行。

3. **坍缩不只是改种群，还会切换模型策略。**
   坍缩时会调用 `activate_anchor_policy()`，恢复结束才 `deactivate_anchor_policy()`，见 [eoh_rl.py](./llm4ad/method/eoh_rl/eoh_rl.py:2297)。这意味着 combined 版本同时改变了父代分布、算子分布、rollout policy。单独看自适应或坍缩可能更好，但叠加后会过度回退/过度探索。

4. **你现在的“collapse-only”消融在主代码里不是合法实验。**
   核心代码禁止 `collapse.enabled=true` 且 `adaptive_operator.enabled=false`，见 [eoh_rl.py](./llm4ad/method/eoh_rl/eoh_rl.py:253)。而脚本确实生成了 `collapse_only = (False, True)`，见 [运行B组tsp自适应算子坍缩完整矩阵_gpu1.sh](./scripts/运行B组tsp自适应算子坍缩完整矩阵_gpu1.sh:126)。所以如果你有 collapse-only 结果，要先确认它是不是来自旧代码、改过的代码，或者实际标签混了。

我的优先诊断是：

- adaptive-only 好：说明当前算子权重能利用局部搜索信号。
- collapse-only 好：说明种群重启/anchor policy 对逃离局部停滞有效。
- both 反而差：说明两者叠加后**过度收缩种群 + 普通 adaptive 在低多样性状态下继续学习 + anchor policy 切换**，导致有效 frontier improvement 下降。

建议下一步先做三个修正/验证：

1. 先定义真正的四组消融：`fixed_no_collapse`、`adaptive_only`、`collapse_fixed_recovery`、`adaptive_plus_collapse`。当前 `collapse_only` 不成立。

2. 修主路径：recovery active 时应真的调用 `compute_recovery_probabilities()`，而不是普通 `select_ops()`；同时 metrics 也要记录 recovery 概率。

3. 做一组 diagnostic 表，不只看最终 best：比较每组的 `collapse_count`、recovery 持续轮数、`i1/m1/m2/e1/e2` 实际比例、`frontier_improve_count`、`parent_improve_count`、`survived_main_population_count`、`accepted_lora` 和 `reject_reason`。
如果 both 组 parent improve 高但 frontier improve 低，就是典型“局部改进被奖励，自适应调度过拟合”的证据。
