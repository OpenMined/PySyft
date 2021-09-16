# stdlib
from typing import Any
from typing import Callable


def call_func_and_wrap_result(
    func: Callable, wrap_type: Any = None, *args: Any, **kwargs: Any
) -> Any:
    func_res = func(*args, **kwargs)
    return wrap_type(child=func_res)
