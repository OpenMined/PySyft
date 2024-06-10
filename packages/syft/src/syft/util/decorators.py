# stdlib
from collections.abc import Callable
import functools
from typing import Any
import warnings

# relative
from ..service.response import SyftError


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
        instance_info = previous_instances.get(cls)
        if instance_info is not None and instance_info.get("args") == (args, kwargs):
            return instance_info.get("instance")
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
                return SyftError(message=message)
            warnings.warn(
                message,
                category=DeprecationWarning,
                stacklevel=2,
            )
            return func(*args, **kwargs)

        return wrapper

    return decorator
