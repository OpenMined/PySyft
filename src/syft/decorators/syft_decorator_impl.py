# stdlib
import inspect
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

# syft relative
from .typecheck import type_hints

# this flag is set in syft.__init__.py
LONG_TYPECHECK_STACK_TRACES = None


def syft_decorator(
    typechecking: bool = False,
    prohibit_args: bool = True,
    enforce_policies: bool = False,
    syft_logger: bool = False,
    other_decorators: Optional[List] = None,
) -> Callable:
    def decorator(function: Callable) -> Callable:

        if typechecking:
            function = type_hints(function, prohibit_args=prohibit_args)

        def wrapper(*args: Tuple[Any], **kwargs: Dict[Any, Any]) -> Callable:

            return function(*args, **kwargs)
            # try:
            #     return function(*args, **kwargs)
            # except Exception as e:
            #     if LONG_TYPECHECK_STACK_TRACES:
            #         raise e
            #     # Truncate stacktrace concerned with the type checking decorator
            #     # so that the true problem is easier to see
            #     raise Exception(str(e))

        if other_decorators:
            for other_decorator in other_decorators:
                wrapper = other_decorator(wrapper)

        wrapper.__annotations__ = function.__annotations__
        wrapper.__qualname__ = function.__qualname__
        wrapper.__name__ = function.__name__
        wrapper.__doc__ = function.__doc__
        wrapper.__module__ = function.__module__

        old_signature = inspect.signature(function)
        wrapper.__signature__ = old_signature  # type: ignore

        return wrapper

    return decorator
