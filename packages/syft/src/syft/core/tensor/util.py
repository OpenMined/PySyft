# stdlib
from typing import Any
from typing import Dict

HANDLED_FUNCTIONS: Dict[Any, Any] = {}


def query_implementation(tensor_type: Any, func: Any) -> Any:
    name = func.__name__
    cache = HANDLED_FUNCTIONS.get(tensor_type, None)
    if cache and name in cache:
        return HANDLED_FUNCTIONS[tensor_type][func.__name__]
    return None


def implements(tensor_type: Any, np_function: Any) -> Any:
    def decorator(func: Any) -> Any:
        if tensor_type not in HANDLED_FUNCTIONS:
            HANDLED_FUNCTIONS[tensor_type] = {}

        HANDLED_FUNCTIONS[tensor_type][np_function.__name__] = func
        return func

    return decorator
