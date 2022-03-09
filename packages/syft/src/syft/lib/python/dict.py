# stdlib
from collections import UserDict
from collections.abc import ItemsView
from collections.abc import KeysView
from collections.abc import ValuesView
from typing import Any
from typing import Dict as TypeDict
from typing import Iterable
from typing import Optional
from typing import Union

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType

# syft absolute
import syft as sy

# relative
from ...core.common import UID
from ...core.common.serde.serializable import serializable
from ...logger import traceback_and_raise
from ...logger import warning
from ...proto.lib.python.dict_pb2 import Dict as Dict_PB
from .iterator import Iterator
from .primitive_factory import PrimitiveFactory
from .primitive_factory import isprimitive
from .primitive_interface import PyPrimitive
from .types import SyPrimitiveRet
from .util import downcast
from .util import upcast


@serializable()
class Dict(UserDict, PyPrimitive):
    # the incoming types to UserDict __init__ are overloaded and weird
    # see https://github.com/python/cpython/blob/master/Lib/collections/__init__.py
    # this is the version from python 3.7 because we need to support 3.7
    # python 3.8 signature includes a new PEP 570 (args, /, kwargs) syntax:
    # https://www.python.org/dev/peps/pep-0570/

    def __init__(*args: Any, **kwargs: Any) -> None:
        if not args:
            traceback_and_raise(
                TypeError("descriptor '__init__' of 'Dict' object " "needs an argument")
            )
        self, *args = args  # type: ignore
        if len(args) > 1:
            traceback_and_raise(
                TypeError(f"expected at most 1 arguments, got {len(args)}")
            )
        if args:
            args_dict = args[0]
        elif "dict" in kwargs:
            args_dict = kwargs.pop("dict")

            warning(
                "Passing 'dict' as keyword argument is deprecated",
                DeprecationWarning,
                stacklevel=2,
            )
        else:
            args_dict = None

        self.data = {}
        if args_dict is not None:
            self.update(args_dict)
        if kwargs:
            self.update(kwargs)

        # We cant add UID from kwargs or it could easily be overwritten by the dict
        # that is being passed in for __init__
        # If you want to update it use the _id setter after creation.
        self._id = UID()

        temporary_box = kwargs["temporary_box"] if "temporary_box" in kwargs else False
        if temporary_box:
            PyPrimitive.__init__(
                self,
                temporary_box=temporary_box,
            )

    @property
    def id(self) -> UID:
        """We reveal PyPrimitive.id as a property to discourage users and
        developers of Syft from modifying .id attributes after an object
        has been initialized.

        :return: returns the unique id of the object
        :rtype: UID
        """
        return self._id

    def upcast(self) -> TypeDict:
        # recursively upcast
        result = {k: upcast(v) for k, v in self.items()}
        if "temporary_box" in result:
            del result["temporary_box"]
        return result

    def __contains__(self, other: Any) -> SyPrimitiveRet:
        res = super().__contains__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    def __eq__(self, other: Any) -> SyPrimitiveRet:
        res = super().__eq__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    def __format__(self, format_spec: str) -> str:
        # python complains if the return value is not str
        res = super().__format__(format_spec)
        return str(res)

    def __ge__(self, other: Any) -> SyPrimitiveRet:
        res = super().__ge__(other)  # type: ignore
        return PrimitiveFactory.generate_primitive(value=res)

    def __getitem__(self, key: Any) -> Union[SyPrimitiveRet, Any]:
        res = super().__getitem__(key)
        if isprimitive(value=res):
            return PrimitiveFactory.generate_primitive(value=res)
        else:
            # we can have torch.Tensor and other types
            return res

    def __gt__(self, other: Any) -> SyPrimitiveRet:
        res = super().__gt__(other)  # type: ignore
        return PrimitiveFactory.generate_primitive(value=res)

    def __hash__(self) -> SyPrimitiveRet:
        res = super().__hash__()
        return PrimitiveFactory.generate_primitive(value=res)

    def __iter__(self, max_len: Optional[int] = None) -> Iterator:
        return Iterator(super().__iter__(), max_len=max_len)

    def __le__(self, other: Any) -> SyPrimitiveRet:
        res = super().__le__(other)  # type: ignore
        return PrimitiveFactory.generate_primitive(value=res)

    def __len__(self) -> SyPrimitiveRet:
        res = super().__len__()
        return PrimitiveFactory.generate_primitive(value=res)

    def __lt__(self, other: Any) -> SyPrimitiveRet:
        res = super().__lt__(other)  # type: ignore
        return PrimitiveFactory.generate_primitive(value=res)

    def __ne__(self, other: Any) -> SyPrimitiveRet:
        res = super().__ne__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    def __sizeof__(self) -> SyPrimitiveRet:
        res = super().__sizeof__()
        return PrimitiveFactory.generate_primitive(value=res)

    def copy(self) -> SyPrimitiveRet:
        res = super().copy()
        return PrimitiveFactory.generate_primitive(value=res)

    @classmethod
    def fromkeys(
        cls, iterable: Iterable, value: Optional[Any] = None
    ) -> SyPrimitiveRet:
        res = super().fromkeys(iterable, value)
        return PrimitiveFactory.generate_primitive(value=res)

    def dict_get(self, key: Any, default: Any = None) -> Any:
        res = super().get(key, default)
        if isprimitive(value=res):
            return PrimitiveFactory.generate_primitive(value=res)
        else:
            # we can have torch.Tensor and other types
            return res

    def items(self, max_len: Optional[int] = None) -> Iterator:  # type: ignore
        return Iterator(ItemsView(self), max_len=max_len)

    def keys(self, max_len: Optional[int] = None) -> Iterator:  # type: ignore
        return Iterator(KeysView(self), max_len=max_len)

    def values(self, *args: Any, max_len: Optional[int] = None) -> Iterator:  # type: ignore
        # this is what the super type does and there is a test in dict_test.py
        # test_values which checks for this so we could disable the test or
        # keep this workaround
        if len(args) > 0:
            traceback_and_raise(
                TypeError("values() takes 1 positional argument but 2 were given")
            )
        return Iterator(ValuesView(self), max_len=max_len)

    def pop(self, key: Any, *args: Any) -> SyPrimitiveRet:
        res = super().pop(key, *args)
        return PrimitiveFactory.generate_primitive(value=res)

    def popitem(self) -> SyPrimitiveRet:
        res = self.data.popitem()
        return PrimitiveFactory.generate_primitive(value=res)

    def setdefault(self, key: Any, default: Any = None) -> SyPrimitiveRet:
        res = PrimitiveFactory.generate_primitive(value=default)
        res = super().setdefault(key, res)
        return res

    def clear(self) -> None:
        # we get the None return and create a SyNone
        # this is to make sure someone doesn't rewrite the method to return nothing
        return PrimitiveFactory.generate_primitive(value=super().clear())

    def _object2proto(self) -> Dict_PB:
        id_ = sy.serialize(obj=self.id)

        keys = [
            sy.serialize(obj=downcast(value=element), to_bytes=True)
            for element in self.data.keys()
        ]

        values = [
            sy.serialize(obj=downcast(value=element), to_bytes=True)
            for element in self.data.values()
        ]

        if hasattr(self, "temporary_box"):
            temporary_box = self.temporary_box
        else:
            temporary_box = False

        return Dict_PB(
            id=id_,
            keys=keys,
            values=values,
            temporary_box=temporary_box,
        )

    @staticmethod
    def _proto2object(proto: Dict_PB) -> "Dict":
        id_: UID = sy.deserialize(blob=proto.id)

        values = [
            upcast(value=sy.deserialize(blob=element, from_bytes=True))
            for element in proto.values
        ]

        keys = [
            upcast(value=sy.deserialize(blob=element, from_bytes=True))
            for element in proto.keys
        ]
        new_dict = Dict(dict(zip(keys, values)))
        new_dict.temporary_box = proto.temporary_box
        new_dict._id = id_
        return new_dict

    @staticmethod
    def get_protobuf_schema() -> GeneratedProtocolMessageType:
        return Dict_PB
