# type: ignore
from typing import Any, Callable


def isfunc(x: Any) -> bool:
    return hasattr(x, "__call__")


def call_func_and_wrap_result(
    func: Callable, wrap_type: Any = None, *args, **kwargs
) -> Any:
    func_res = func(*args, **kwargs)
    return wrap_type(child=func_res)
