# stdlib
import functools
from typing import Any
from typing import Callable


def recursive_serialize_kwargs(func: Callable) -> Callable:
    """Helps detect recursive calls and pass-through kwargs used in the first invocation"""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Callable:
        recurse_step = getattr(wrapper, "recurse_step", 0)

        # We want to preserve the `for_hashing` flag during recursive _serialize() calls..
        # ..skip if not being called for first call
        if recurse_step == 0 and not kwargs.get("for_hashing", None):
            return func(*args, **kwargs)
        elif recurse_step == 0:
            # cache kwargs
            wrapper.for_hashing = kwargs["for_hashing"]
            wrapper.debug = kwargs.get("debug", False)
            wrapper.recurse_step = 1
        else:
            # carry forward the cached args during recursive _serialize() call
            kwargs["for_hashing"] = wrapper.for_hashing
            wrapper.recurse_step += 1

        if wrapper.debug:
            print(f'{"-" * wrapper.recurse_step}>', args, kwargs)

        result = func(*args, **kwargs)

        wrapper.recurse_step -= 1

        if wrapper.recurse_step == 0:
            delattr(wrapper, "recurse_step")
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
