import importlib.util
import inspect
import logging
import warnings

import gymnasium
from stable_baselines3.common.base_class import BaseAlgorithm

logger = logging.getLogger(__name__)

# Keyed by class_name → algo class
CUSTOM_ALGOS: dict[str, type] = {}


def _load_module(filepath: str):
    spec = importlib.util.spec_from_file_location("_custom_user_module", filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_custom_env(filepath: str) -> tuple[str, str]:
    """
    Load a .py file, find the first gymnasium.Env subclass, register it,
    and return (class_name, registered_env_id).
    """
    module = _load_module(filepath)

    env_classes = [
        cls for _, cls in inspect.getmembers(module, inspect.isclass)
        if issubclass(cls, gymnasium.Env) and cls is not gymnasium.Env
        and cls.__module__ == module.__name__
    ]

    if not env_classes:
        raise ValueError("No gymnasium.Env subclass found in file")

    if len(env_classes) > 1:
        warnings.warn(
            f"Multiple gymnasium.Env subclasses found; using {env_classes[0].__name__}",
            stacklevel=2,
        )

    env_class = env_classes[0]
    class_name = env_class.__name__
    env_id = f"Custom/{class_name}-v0"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            gymnasium.register(id=env_id, entry_point=env_class)
        except gymnasium.error.Error:
            pass

    return class_name, env_id


def load_custom_algo(filepath: str) -> tuple[str, type]:
    """
    Load a .py file, find the first BaseAlgorithm subclass, store it in
    CUSTOM_ALGOS, and return (class_name, algo_class).
    """
    module = _load_module(filepath)

    algo_classes = [
        cls for _, cls in inspect.getmembers(module, inspect.isclass)
        if issubclass(cls, BaseAlgorithm) and cls is not BaseAlgorithm
        and cls.__module__ == module.__name__
    ]

    if not algo_classes:
        raise ValueError("No stable_baselines3 BaseAlgorithm subclass found in file")

    if len(algo_classes) > 1:
        warnings.warn(
            f"Multiple BaseAlgorithm subclasses found; using {algo_classes[0].__name__}",
            stacklevel=2,
        )

    algo_class = algo_classes[0]
    class_name = algo_class.__name__
    CUSTOM_ALGOS[class_name] = algo_class
    return class_name, algo_class
