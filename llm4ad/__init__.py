from importlib import import_module

_LAZY_MODULES = {
    "base": "llm4ad.base",
    "method": "llm4ad.method",
    "task": "llm4ad.task",
    "tools": "llm4ad.tools",
    "profiler": "llm4ad.tools.profiler",
    "llm": "llm4ad.tools.llm",
}

__version__ = "1.0.0"


def __getattr__(name):
    if name in _LAZY_MODULES:
        module = import_module(_LAZY_MODULES[name])
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + list(_LAZY_MODULES.keys()))


__all__ = ["base", "llm", "method", "profiler", "task", "tools"]
