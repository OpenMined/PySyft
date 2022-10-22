# stdlib
from typing import Any
from typing import Dict
from typing import Tuple
from typing import Tuple as TypeTuple

HANDLED_FUNCTIONS: Dict[Any, Any] = {}


def inputs2child(*args: Any, **kwargs: Any) -> TypeTuple[Tuple[Any], Any]:

    # relative
    from .passthrough import PassthroughTensor  # type: ignore

    out_args_list = list()

    for out in tuple(
        [x.child if isinstance(x, PassthroughTensor) else x for x in args]
    ):
        out_args_list.append(out)

    out_kwargs = {}

    for x in kwargs.items():
        if isinstance(x[1], PassthroughTensor):
            out_kwargs[x[0]] = x[1].child
        else:
            out_kwargs[x[0]] = x[1]

    out_args = tuple(out_args_list)

    return out_args, out_kwargs  # type: ignore


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
