# stdlib
import functools
from typing import Any
from typing import Callable


def recursive_serialize_kwargs(func: Callable) -> Callable:
    """Helps detect recursive calls and pass-through kwargs used in the first invocation"""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Callable:
        if not kwargs.get("for_hashing", None):
            return func(*args, **kwargs)

        if getattr(wrapper, "depth", 0) > 0:
            # in recursion, carry forward the cached args
            kwargs["for_hashing"] = wrapper.for_hashing
            wrapper.depth += 1
        else:
            # cache kwargs
            wrapper.for_hashing = kwargs["for_hashing"]
            wrapper.debug = kwargs.get("debug", False)
            wrapper.depth = 1

        if wrapper.debug:
            print(f'{"-" * wrapper.depth}>', args, kwargs)

        result = func(*args, **kwargs)

        wrapper.depth -= 1

        if wrapper.depth == 0:
            delattr(wrapper, "depth")
            delattr(wrapper, "for_hashing")
            delattr(wrapper, "debug")

        return result

    return wrapper


@recursive_serialize_kwargs
def _serialize(
    obj: object,
    to_proto: bool = True,
    to_bytes: bool = False,
    for_hashing: bool = False,
    debug: bool = False,
) -> Any:
    # relative
    from .recursive import rs_object2proto

    proto = rs_object2proto(obj, for_hashing=for_hashing)

    if to_bytes:
        return proto.to_bytes()

    if to_proto:
        return proto
