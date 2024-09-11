# stdlib
from collections.abc import Callable
import functools
from typing import Any
import warnings

# relative
from ..types.errors import SyftException


def singleton(cls: Any) -> Callable:
    """
    Handy decorator for creating a singleton class
    Description:
        - Decorate your class with this decorator
        - If you happen to create another instance of the same class, it will return the previously created one
        - Supports creation of multiple instances of same class with different args/kwargs
        - Works for multiple classes
    Use:
        >>> from decorators import singleton
        >>>
        >>> @singleton
        ... class A:
        ...     def __init__(self, *args, **kwargs):
        ...         pass
        ...
        >>>
        >>> a = A(name='First')
        >>> b = A(name='First', lname='Last')
        >>> c = A(name='First', lname='Last')
        >>> a is b  # has to be different
        False
        >>> b is c  # has to be same
        True
        >>>
    """
    previous_instances: dict[Any, Any] = {}

    @functools.wraps(cls)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if cls in previous_instances and previous_instances.get(cls, None).get(
            "args"
        ) == (args, kwargs):
            return previous_instances[cls].get("instance")
        else:
            previous_instances[cls] = {
                "args": (args, kwargs),
                "instance": cls(*args, **kwargs),
            }
            return previous_instances[cls].get("instance")

    return wrapper


def deprecated(
    reason: str = "This function is deprecated and may be removed in the future.",
    return_syfterror: bool = False,
) -> Callable:
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: list, **kwargs: dict) -> Any:
            message = f"{func.__qualname__} is deprecated: {reason}"
            if return_syfterror:
                raise SyftException(public_message=message)
            warnings.warn(
                message,
                category=DeprecationWarning,
                stacklevel=2,
            )
            return func(*args, **kwargs)

        return wrapper

    return decorator
