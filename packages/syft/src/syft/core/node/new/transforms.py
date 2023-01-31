# stdlib
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Union


class NotNone:
    pass


def geteitherattr(
    _self: Any, output: Dict, key: str, default: Any = NotNone
) -> Optional[Any]:
    if key in output:
        return output[key]
    if default == NotNone:
        return getattr(_self, key)
    return getattr(_self, key, default)


def make_set_default(key: str, value: Any) -> Callable:
    def set_default(_self: Any, output: Dict) -> Dict:
        if not geteitherattr(_self, output, key, None):
            output[key] = value
        return output

    return set_default


def drop(list_keys: List[str]) -> Callable:
    def drop_keys(_self: Any, output: Dict) -> Dict:
        for key in list_keys:
            if key in output:
                del output[key]
        return output

    return drop_keys


def rename(old_key: str, new_key: str) -> Callable:
    def drop_keys(_self: Any, output: Dict) -> Dict:
        output[new_key] = geteitherattr(_self, output, old_key)
        if old_key in output:
            del output[old_key]
        return output

    return drop_keys


def keep(list_keys: List[str]) -> Callable:
    def drop_keys(_self: Any, output: Dict) -> Dict:
        for key in list_keys:
            if key not in output:
                output[key] = getattr(_self, key)

        keys = list(output.keys())

        for key in keys:
            if key not in list_keys and key in output:
                del output[key]

        return output

    return drop_keys


def convert_types(list_keys: List[str], types: Union[type, List[type]]) -> Callable:
    if not isinstance(types, list):
        types = [types] * len(list_keys)

    if isinstance(types, list) and len(types) != len(list_keys):
        raise Exception("convert types lists must be the same length")

    def run_convert_types(_self: Any, output: dict) -> dict:
        for key, _type in zip(list_keys, types):
            output[key] = _type(geteitherattr(_self, output, key))
        return output

    return run_convert_types
