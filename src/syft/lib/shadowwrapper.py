# stdlib
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Tuple
from typing import Type
from typing import Union

# syft relative
from ..core.common.uid import UID
from ..lib.util import full_name_with_qualname

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
# original constructor and instance, proxying the *args and **kwargs in __init__ and
# any call to getattr which one would expect to execute on the underlying wrapped
# instance. The exceptions to this are listed below in the __getattribute__ method.
# To see how this is used see generic.py for more context.


class ShadowWrapperMeta(type):
    @classmethod
    def proxy_dunder(cls, op: str) -> Callable:
        def dunder(_self: Any, *args: Tuple[Any, ...], **kwargs: Dict[str, Any]) -> Any:
            _wrapped = getattr(_self, "_wrapped", None)
            attr = getattr(_wrapped, op)
            return attr(*args, **kwargs)

        return dunder

    def __new__(
        cls: Type,
        name: str,
        bases: Tuple[Type, ...],
        dct: Dict[str, Any],
        shadow_type: Union[type, None],
    ) -> "ShadowWrapperMeta":
        # attrs are the attributes / properties of our new class
        attrs: Dict[str, Any] = {}

        if shadow_type is not None:
            name = shadow_type.__name__
            fqn = full_name_with_qualname(klass=shadow_type)
        else:
            # for creating proxy classes of None if so desired
            name = type(shadow_type).__name__
            fqn = full_name_with_qualname(klass=type(shadow_type))

        attrs["__name__"] = name
        new_class_name = f"syft.proxy.{fqn}"
        parts = new_class_name.split(".")
        name = parts.pop(-1)
        attrs["__module__"] = ".".join(parts)

        attrs["_wrapped"] = None

        def id_get(_self: Any) -> UID:
            return _self._id

        def id_set(_self: Any, new_id: UID) -> None:
            _self._id = new_id

        # add our custom id property
        attrs["id"] = property(fget=id_get, fset=id_set)

        # create our init function
        def init(self: Any, *args: Tuple[Any, ...], **kwargs: Dict[Any, Any]) -> None:
            if callable(shadow_type):
                self._wrapped = shadow_type(*args, **kwargs)
            else:
                self._wrapped = shadow_type

        attrs["__init__"] = init

        # create our getattribute override
        def getattribute(self: Any, name: str) -> Any:
            # intercept any getter and proxy everything except the following through
            # to the self._wrapper reference
            if name in [
                "id",
                "_id",
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

        attrs["__getattribute__"] = getattribute

        # we need this to return the underlying _wrapped value for instances where
        # callables cant use a Python ShadowWrapped duck type and require the real
        # underlying un-subclassable CPython type
        def upcast(self: Any) -> Type[type]:
            return self._wrapped

        attrs["upcast"] = upcast
        attrs["__subclass__"] = {shadow_type}

        new_type = type(
            name,
            (object,),
            attrs,
        )

        for k in dir(new_type):
            if k in [
                "__class__",
                "__dict__",
                "__dir__",
                "__getattribute__",
                "__init__",
                "__init_subclass__",
                "__module__",
                "__name__",
                "__new__",
                "__setattr__",
            ]:
                # skip these dunder methods
                continue
            if k.startswith("__") and k.endswith("__"):
                super_method = getattr(new_type, k)
                if type(super_method).__name__ in ["property"]:
                    continue
                setattr(new_type, k, ShadowWrapperMeta.proxy_dunder(op=k))

        mro_attribute = getattr(shadow_type, "mro", None)
        shadow_mro = []
        if mro_attribute is not None:
            shadow_mro = mro_attribute()
        else:
            mro_type_attribute = getattr(type(shadow_type), "mro", None)
            if mro_type_attribute is not None:
                shadow_mro = mro_type_attribute()

        # we are able to get isinstance checks to pass however the ability to achieve
        # issubclass has still not been solved, mro is part of these checks
        # if we were normally able to subclass this is what the mro would look like
        def mro() -> List[type]:
            # return this new type in the chain of the result of shadow_type.mro()
            return [new_type] + shadow_mro

        new_type.mro = mro  # type: ignore

        return new_type  # type: ignore


def ShadowWrapperFactory(shadow_type: type) -> type:
    class ShadowWrapper(metaclass=ShadowWrapperMeta, shadow_type=shadow_type):
        pass

    return ShadowWrapper
