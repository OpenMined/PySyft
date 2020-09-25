# stdlib
from typing import Any
from typing import Dict
from typing import List
from typing import NewType
from typing import Tuple
from typing import Union

# syft relative
from .primitive_factory import PrimitiveFactory
from .primitive_factory import isprimitive
from .primitive_interface import PyPrimitive

NotImplementedType = NewType("NotImplementedType", type(NotImplemented))  # type: ignore
SyPrimitiveRet = NewType("SyPrimitiveRet", Union[PyPrimitive, NotImplementedType])  # type: ignore


def downcast_args_and_kwargs(
    args: Union[List[Any], Tuple[Any, ...]], kwargs: Dict[Any, Any]
) -> Tuple[List[Any], Dict[Any, Any]]:
    # when we try to handle primitives they often need to be converted to PyPrimitive
    # first so they can be serialized and sent around
    downcasted_args = []
    downcasted_kwargs = {}
    for arg in args:
        # check if its primitive
        if isprimitive(value=arg):
            # Wrap in a SyPrimitive
            wrapped_arg = PrimitiveFactory.generate_primitive(value=arg, recurse=True)
            downcasted_args.append(wrapped_arg)
        else:
            downcasted_args.append(arg)

    for k, arg in kwargs.items():
        # check if its primitive
        if isprimitive(value=arg):
            # Wrap in a SyPrimitive
            wrapped_arg = PrimitiveFactory.generate_primitive(value=arg, recurse=True)
            downcasted_kwargs[k] = wrapped_arg
        else:
            downcasted_kwargs[k] = arg

    return (downcasted_args, downcasted_kwargs)


def upcast_args_and_kwargs(
    args: Union[List[Any], Tuple[Any, ...]], kwargs: Dict[Any, Any]
) -> Tuple[List[Any], Dict[Any, Any]]:
    # When we invoke remote methods that use C code and cannot utilise our wrapped
    # types through duck typing, we must convert them to the their original form.
    upcasted_args = []
    upcasted_kwargs = {}
    for arg in args:
        # try to upcast if possible
        upcast_method = getattr(arg, "upcast", None)
        # if we decide to ShadowWrap NoneType we would need to check here
        if upcast_method is not None:
            upcasted_args.append(upcast_method())
        else:
            upcasted_args.append(arg)

    for k, arg in kwargs.items():
        # try to upcast if possible
        upcast_method = getattr(arg, "upcast", None)
        # if we decide to ShadowWrap NoneType we would need to check here
        if upcast_method is not None:
            upcasted_kwargs[k] = upcast_method()
        else:
            upcasted_kwargs[k] = arg

    return (upcasted_args, upcasted_kwargs)
