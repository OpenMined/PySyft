from __future__ import annotations

import inspect
from .typecheck import type_hints


def syft_decorator(
    typechecking=False,
    enforce_policies=False,
    syft_logger=False,
    other_decorators: list = None,
):
    def decorator(function):
        if typechecking:
            function = type_hints(function)

        def wrapper(*args, **kwargs):
            return function(*args, **kwargs)

        if other_decorators:
            for other_decorator in other_decorators:
                wrapper = other_decorator(wrapper)

        wrapper.__annotations__ = function.__annotations__
        wrapper.__qualname__ = function.__qualname__
        wrapper.__name__ = function.__name__
        wrapper.__doc__ = function.__doc__
        wrapper.__module__ = function.__module__

        old_signature = inspect.signature(function)
        wrapper.__signature__ = old_signature

        return wrapper

    return decorator
