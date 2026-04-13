import os
import inspect
import importlib

_LAZY_MODULES = {
    "funsearch": "llm4ad.method.funsearch",
    "hillclimb": "llm4ad.method.hillclimb",
    "randsample": "llm4ad.method.randsample",
    "eoh": "llm4ad.method.eoh",
    "meoh": "llm4ad.method.meoh",
    "mles": "llm4ad.method.mles",
    "moead": "llm4ad.method.moead",
    "nsga2": "llm4ad.method.nsga2",
    "partevo": "llm4ad.method.partevo",
    "llamea": "llm4ad.method.llamea",
    "reevo": "llm4ad.method.reevo",
}


def __getattr__(name):
    if name in _LAZY_MODULES:
        module = importlib.import_module(_LAZY_MODULES[name])
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + list(_LAZY_MODULES.keys()))


__all__ = list(_LAZY_MODULES.keys())


def import_all_method_classes_from_subfolders(root_directory: str):
    """Dynamically imports all classes from Python files that share the same name as their parent folder.
    Args:
        root_directory: The root directory (e.g., 'method') to start the search.
    """
    # Iterate through the subdirectories
    for subdir in os.listdir(root_directory):
        subdir_path = os.path.join(root_directory, subdir)
        profiler_name = "profiler"

        # Check if it's a directory and contains a .py file with the same name
        if os.path.isdir(subdir_path):
            module_file = f"{subdir}.py"
            profiler_file = f"{profiler_name}.py"
            module_path = os.path.join(subdir_path, module_file)
            profiler_path = os.path.join(subdir_path, profiler_file)

            # import the method
            if os.path.exists(module_path):
                # Build the module name for importing (e.g., method.eoh.eoh)
                module_name = f"{__name__}.{subdir}.{subdir}"

                # Dynamically import the module
                module = importlib.import_module(module_name)

                # Import all classes from the module
                for attribute_name in dir(module):
                    attribute = getattr(module, attribute_name)
                    if isinstance(attribute, type):  # Only import class objects
                        # Use inspect to check if the class is defined in the current module
                        if inspect.getmodule(attribute).__file__ == module.__file__:
                            globals()[attribute_name] = (
                                attribute  # Add the class to the global namespace
                            )
                            # print(f'Imported class {attribute_name} from {module_name}')

            # import the profiler
            if os.path.exists(profiler_path):
                # Build the module name for importing (e.g., method.eoh.eoh)
                module_name = f"{__name__}.{subdir}.{profiler_name}"

                # Dynamically import the module
                module = importlib.import_module(module_name)

                # Import all classes from the module
                for attribute_name in dir(module):
                    attribute = getattr(module, attribute_name)
                    if isinstance(attribute, type):  # Only import class objects
                        # Use inspect to check if the class is defined in the current module
                        if inspect.getmodule(attribute).__file__ == module.__file__:
                            globals()[attribute_name] = (
                                attribute  # Add the class to the global namespace
                            )
                            # print(f'Imported class {attribute_name} from {module_name}')
