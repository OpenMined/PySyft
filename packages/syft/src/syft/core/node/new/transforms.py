# stdlib
from typing import Any
from typing import Callable


def make_set_default(key: str, value: Any) -> Callable:
    def set_default(output: dict) -> dict:
        if not output.get(key, None):
            output[key] = value
        return output

    return set_default


def drop(list_keys) -> Callable:
    def drop_keys(output: dict) -> dict:
        for key in list_keys:
            del output[key]
        return output

    return drop_keys


def keep(list_keys) -> Callable:
    def drop_keys(output: dict) -> dict:
        keys = list(output.keys())
        for key in keys:
            if key not in list_keys:
                del output[key]
        return output

    return drop_keys
