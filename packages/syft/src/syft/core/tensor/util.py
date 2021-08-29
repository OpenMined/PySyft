# stdlib
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple as TypeTuple

HANDLED_FUNCTIONS: Dict[Any, Any] = {}


def inputs2child(
    *args: List[Any], **kwargs: Dict[Any, Any]
) -> TypeTuple[List[Any], Dict[Any, Any]]:

    # relative
    from .passthrough import PassthroughTensor  # type: ignore

    args = tuple([x.child if isinstance(x, PassthroughTensor) else x for x in args])
    kwargs = {
        x[0]: x[1].child if isinstance(x[1], PassthroughTensor) else x[1]
        for x in kwargs.items()
    }
    return args, kwargs  # type: ignore


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
