from importlib import import_module

_LAZY_MODULES = {
    "llm": "llm4ad.tools.llm",
    "profiler": "llm4ad.tools.profiler",
}


def __getattr__(name):
    if name in _LAZY_MODULES:
        module = import_module(_LAZY_MODULES[name])
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + list(_LAZY_MODULES.keys()))


__all__ = ["llm", "profiler"]
