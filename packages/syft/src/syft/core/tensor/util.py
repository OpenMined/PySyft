# stdlib
from typing import Callable
from typing import Dict
from typing import Type
from typing import Union

HANDLED_FUNCTIONS: Dict[Type, Dict[str, Callable]] = {}


def query_implementation(tensor_type: Type, func: Callable) -> Union[None, Callable]:
    name = func.__name__
    cache = HANDLED_FUNCTIONS[tensor_type]
    if name in cache:
        return HANDLED_FUNCTIONS[tensor_type][func.__name__]
    return None


def implements(tensor_type: Type, np_function: Callable) -> Callable:
    def decorator(func: Callable) -> Callable:
        if tensor_type not in HANDLED_FUNCTIONS:
            HANDLED_FUNCTIONS[tensor_type] = {}

        HANDLED_FUNCTIONS[tensor_type][np_function.__name__] = func
        return func

    return decorator
