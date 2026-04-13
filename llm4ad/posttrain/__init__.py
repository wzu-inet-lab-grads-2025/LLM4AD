from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "BuilderConfig": "llm4ad.posttrain.config",
    "CollectorConfig": "llm4ad.posttrain.config",
    "CompositeReward": "llm4ad.posttrain.rewards",
    "EvalTraceRecorder": "llm4ad.posttrain.eval_trace",
    "EventStore": "llm4ad.posttrain.event_store",
    "EventStoreConfig": "llm4ad.posttrain.config",
    "GateConfig": "llm4ad.posttrain.config",
    "ModelRegistry": "llm4ad.posttrain.registry",
    "PostTrainConfig": "llm4ad.posttrain.config",
    "PostTrainLLMProxy": "llm4ad.posttrain.llm_proxy",
    "PostTrainOrchestrator": "llm4ad.posttrain.orchestrator",
    "PostTrainRuntime": "llm4ad.posttrain.runtime",
    "PromotionGate": "llm4ad.posttrain.gates",
    "RegistryModelRouter": "llm4ad.posttrain.serve",
    "RegistryConfig": "llm4ad.posttrain.config",
    "ReplayBuffer": "llm4ad.posttrain.replay_buffer",
    "ReplayBufferConfig": "llm4ad.posttrain.config",
    "RewardConfig": "llm4ad.posttrain.config",
    "RewardProviderBase": "llm4ad.posttrain.rewards",
    "RewardShaperBase": "llm4ad.posttrain.rewards",
    "RoundConfig": "llm4ad.posttrain.config",
    "ServeConfig": "llm4ad.posttrain.config",
    "ServeManagerBase": "llm4ad.posttrain.serve",
    "SmokeSearchGate": "llm4ad.posttrain.gates",
    "StaticValidationGate": "llm4ad.posttrain.gates",
    "SynthesizeConfig": "llm4ad.posttrain.config",
    "SynthesizeStrategyBase": "llm4ad.posttrain.synthesize",
    "TrainerConfig": "llm4ad.posttrain.config",
    "UnifiedCollector": "llm4ad.posttrain.collector",
    "VLLMServeManager": "llm4ad.posttrain.serve",
    "WorkflowConfig": "llm4ad.posttrain.config",
    "WorkflowSmokeRunner": "llm4ad.posttrain.smoke",
    "BestScoreExtractor": "llm4ad.posttrain.smoke",
    "load_posttrain_config": "llm4ad.posttrain.config_loader",
    "load_posttrain_config_from_hydra": "llm4ad.posttrain.hydra_adapter",
}


def __getattr__(name):
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(list(globals().keys()) + list(_EXPORTS.keys()))


__all__ = sorted(_EXPORTS.keys())
