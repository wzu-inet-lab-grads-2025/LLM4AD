from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "EoHOnlineWorkflow": "llm4ad.posttrain.workflows.online_eoh",
    "MEoHOnlineWorkflow": "llm4ad.posttrain.workflows.online_meoh",
    "MLESPrepWorkflow": "llm4ad.posttrain.workflows.prep_mles",
    "MOEADOnlineWorkflow": "llm4ad.posttrain.workflows.online_moead",
    "NSGA2OnlineWorkflow": "llm4ad.posttrain.workflows.online_nsga2",
    "PartEvoPrepWorkflow": "llm4ad.posttrain.workflows.prep_partevo",
    "ReEvoOnlineWorkflow": "llm4ad.posttrain.workflows.online_reevo",
    "build_online_eoh_workflow": "llm4ad.posttrain.workflows.online_eoh",
    "build_online_meoh_workflow": "llm4ad.posttrain.workflows.online_meoh",
    "build_online_moead_workflow": "llm4ad.posttrain.workflows.online_moead",
    "build_online_nsga2_workflow": "llm4ad.posttrain.workflows.online_nsga2",
    "build_online_reevo_workflow": "llm4ad.posttrain.workflows.online_reevo",
    "build_prep_mles_workflow": "llm4ad.posttrain.workflows.prep_mles",
    "build_prep_partevo_workflow": "llm4ad.posttrain.workflows.prep_partevo",
    "run_online_eoh": "llm4ad.posttrain.workflows.online_eoh",
    "run_online_meoh": "llm4ad.posttrain.workflows.online_meoh",
    "run_online_moead": "llm4ad.posttrain.workflows.online_moead",
    "run_online_nsga2": "llm4ad.posttrain.workflows.online_nsga2",
    "run_online_reevo": "llm4ad.posttrain.workflows.online_reevo",
    "run_prep_mles_and_build": "llm4ad.posttrain.workflows.prep_mles",
    "run_prep_mles": "llm4ad.posttrain.workflows.prep_mles",
    "run_prep_partevo_and_build": "llm4ad.posttrain.workflows.prep_partevo",
    "run_prep_partevo": "llm4ad.posttrain.workflows.prep_partevo",
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
