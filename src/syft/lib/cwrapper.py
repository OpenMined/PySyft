# stdlib
from typing import Any
from typing import Callable
from typing import Dict
from typing import Tuple
from typing import Type

# syft relative
from ..core.common.uid import UID

# Where ever possible we try to subclass our target classes so that they inherit all of
# the existing functionality and .mro() (Method Resolution Order) through the class
# heirachy. The aim is to then augment these subclasses with additional functionality,
# such as a UID property, serde capabilities, an automatically generated Pointer and
# methods to send and receive Pointers and their underlying data. However there are
# some types that are exposed from CPython which Python will not allow us to subclass.
# These are often marked as Final in CPython to prevent weird unexpected behavior.
#
# An example of this is torch.device:
#
# >>> import torch
# >>> type(torch.device)
# <class 'type'>
# >>> torch.device
# <class 'torch.device'>
# >>> device = torch.device("cuda:0")
# >>> type(device)
# <class 'torch.device'>
#
# However attempts to subclass result in this:
#
# <class 'torch.device'>
# >>> class A(torch.device):
# ...     pass
# ...
# Traceback (most recent call last):
#   File "<stdin>", line 1, in <module>
# TypeError: type 'torch.device' is not an acceptable base type
#
# The solution below is a lightweight proxy class which does nothing but wrap the
# original C constructor and instance, proxying the *args and **kwargs in __init__ and
# any call to getattr which one would expect to execute on the underlying wrapped
# instance. The exceptions to this are listed below in the __getattribute__ method.
# To see how this is used see generic.py for more context.


class CWrapperMeta(type):
    @staticmethod
    def proxy_dunder(op: str) -> Callable:
        def dunder(_self: Any, *args: Tuple[Any, ...], **kwargs: Dict[str, Any]) -> Any:
            _wrapped = getattr(_self, "_wrapped", None)
            if _wrapped is None:
                raise Exception("CWrapper is unusable when _wrapped is None.")
            attr = getattr(_wrapped, op)
            return attr(*args, **kwargs)

        return dunder

    def __new__(
        cls: Type,
        name: str,
        bases: Tuple[Type, ...],
        dct: Dict[str, Any],
        shadow_type: type,
    ) -> "CWrapperMeta":
        # attrs are the attributes / properties of our new class
        attrs: Dict[str, Any] = {}

        attrs["__name__"] = "device"

        new_class_name = "syft.proxy.torch.device"
        parts = new_class_name.split(".")
        name = parts.pop(-1)
        attrs["_wrapped"] = None
        attrs["__module__"] = ".".join(parts)

        # create our init function
        def __init__(
            self: Any, *args: Tuple[Any, ...], **kwargs: Dict[Any, Any]
        ) -> None:
            self._wrapped = shadow_type(*args, **kwargs)

            def id_get(__self: Any) -> UID:
                return __self._id

            def id_set(__self: Any, new_id: UID) -> None:
                __self._id = new_id

            self.id = property(fget=id_get, fset=id_set)

        attrs["__init__"] = __init__

        # create our getattribute override
        def __getattribute__(self: Any, name: str) -> Any:
            # intercept any getter and proxy everything except the following through
            # to the self._wrapper reference
            if name in [
                "id",
                "_original_constructor",
                "_wrapped",
                "__mro__",
                "upcast",
            ]:
                try:
                    sub = object.__getattribute__(self, name)
                    if type(sub).__name__ == "property":
                        return sub.__get__(self)
                    return sub
                except Exception:
                    return None
            main = self._wrapped.__getattribute__(name)
            return main

        attrs["__getattribute__"] = __getattribute__
        attrs["_original_constructor"] = shadow_type

        def upcast(self: Any) -> Type[type]:
            return self._wrapped

        attrs["upcast"] = upcast

        new_type = type(
            name,
            (object,),
            attrs,
        )

        for k in dir(new_type):
            if k in [
                "__class__",
                "__dict__",
                "__init__",
                "__init_subclass__",
                "__name__",
                "__getattribute__",
                "__new__",
                "__setattr__",
                "__dir__",
            ]:
                # skip these dunder methods
                continue
            if k.startswith("__") and k.endswith("__"):
                super_method = getattr(new_type, k)
                if type(super_method).__name__ in ["property"]:
                    continue
                setattr(new_type, k, CWrapperMeta.proxy_dunder(op=k))

            new_type.mro = getattr(shadow_type, "mro")  # type: ignore

        return new_type  # type: ignore


def CWrapperFactory(shadow_type: type) -> type:
    class CWrapper(metaclass=CWrapperMeta, shadow_type=shadow_type):
        pass

    return CWrapper
