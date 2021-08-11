# stdlib
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union

# relative
from .primitive_factory import PrimitiveFactory
from .primitive_factory import isprimitive


def downcast(value: Any, recurse: bool = True) -> Any:
    if isprimitive(value=value):
        # Wrap in a SyPrimitive
        return PrimitiveFactory.generate_primitive(value=value, recurse=recurse)
    else:
        return value


def downcast_args_and_kwargs(
    args: Union[List[Any], Tuple[Any, ...]], kwargs: Dict[Any, Any]
) -> Tuple[List[Any], Dict[Any, Any]]:
    # when we try to handle primitives they often need to be converted to PyPrimitive
    # first so they can be serialized and sent around
    downcasted_args = []
    downcasted_kwargs = {}
    for arg in args:
        # check if its primitive
        downcasted_args.append(downcast(value=arg))

    for k, arg in kwargs.items():
        # check if its primitive
        downcasted_kwargs[k] = downcast(value=arg)

    return (downcasted_args, downcasted_kwargs)


def upcast(value: Any) -> Any:
    upcast_method = getattr(value, "upcast", None)
    if upcast_method is not None:
        return upcast_method()
    return value


def upcast_args_and_kwargs(
    args: Union[List[Any], Tuple[Any, ...]], kwargs: Dict[Any, Any]
) -> Tuple[List[Any], Dict[Any, Any]]:
    # When we invoke remote methods that use C code and cannot utilise our wrapped
    # types through duck typing, we must convert them to the their original form.
    upcasted_args = []
    upcasted_kwargs = {}
    for arg in args:
        # try to upcast if possible
        upcasted_args.append(upcast(value=arg))

    for k, arg in kwargs.items():
        # try to upcast if possible
        upcasted_kwargs[k] = upcast(value=arg)

    return (upcasted_args, upcasted_kwargs)
