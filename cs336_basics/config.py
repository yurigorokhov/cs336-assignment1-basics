from typing import Any
from collections.abc import Callable


import importlib


def import_from_string(import_path: str):
    """
    Import a class or function from a string path.

    Args:
        import_path: String like "module.submodule.ClassName.method"

    Returns:
        The imported class, method, or function
    """
    # Split the path into parts
    parts = import_path.split(".")

    # Try to find the module by working backwards through the parts
    module = None
    for i in range(len(parts), 0, -1):
        module_path = ".".join(parts[:i])
        try:
            module = importlib.import_module(module_path)
            remaining_parts = parts[i:]
            break
        except ImportError:
            continue

    if module is None:
        raise ImportError(f"Could not import {import_path}")

    # Navigate through the remaining parts using getattr
    obj = module
    for part in remaining_parts:
        obj = getattr(obj, part)

    return obj


def get_from_string(obj, path: str):
    """
    Get value from object using dot-separated path.
    Supports both attributes and dict keys.
    """
    parts = path.split(".")
    result = obj
    for part in parts:
        if isinstance(result, dict):
            result = result[part]
        else:
            v = getattr(result, part)
            if isinstance(v, Callable):
                v = v()
            result = v
    return result


def instantiate(data: Any) -> None:
    def dfs(obj: Any) -> None:
        if isinstance(obj, dict):
            for key, value in obj.items():
                obj[key] = dfs(value)
            if "__target__" in obj:
                target = import_from_string(obj.pop("__target__"))
                return target(**obj)
        elif isinstance(obj, str) and obj.startswith("$$"):
            return get_from_string(data, obj[2:])
        return obj

    return dfs(data)


def set_nested(obj, key, value):
    parts = key.split(".")
    d = obj
    for p in parts[:-1]:
        d = d[p]
    d[parts[-1]] = value
