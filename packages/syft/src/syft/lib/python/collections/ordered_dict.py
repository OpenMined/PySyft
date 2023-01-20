# stdlib
from collections import OrderedDict as PyOrderedDict
from collections.abc import ItemsView
from collections.abc import KeysView
from collections.abc import ValuesView
from typing import Any
from typing import Optional

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType

# relative
from ....core.common.serde.deserialize import _deserialize as deserialize
from ....core.common.serde.serializable import serializable
from ....core.common.serde.serialize import _serialize as serialize
from ....logger import traceback_and_raise
from ....proto.lib.python.collections.ordered_dict_pb2 import (
    OrderedDict as OrderedDict_PB,
)
from ..iterator import Iterator
from ..primitive_factory import PrimitiveFactory
from ..primitive_factory import isprimitive
from ..primitive_interface import PyPrimitive
from ..types import SyPrimitiveRet
from ..util import downcast
from ..util import upcast


@serializable()
class OrderedDict(PyOrderedDict, PyPrimitive):
    def __init__(self, *args: Any, **kwds: Any):
        super().__init__(*args, **kwds)

    def __contains__(self, other: Any) -> SyPrimitiveRet:
        res = super().__contains__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    def __delitem__(self, other: Any) -> None:
        res = super().__delitem__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    def __eq__(self, other: Any) -> SyPrimitiveRet:
        res = super().__eq__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    def __getitem__(self, other: Any) -> SyPrimitiveRet:
        res = super().__getitem__(other)
        if isprimitive(value=res):
            return PrimitiveFactory.generate_primitive(value=res)
        else:
            # we can have torch.Tensor and other types
            return res

    def __iter__(self, max_len: Optional[int] = None) -> Iterator:
        return Iterator(super().__iter__(), max_len=max_len)

    def __len__(self) -> SyPrimitiveRet:
        res = super().__len__()
        return PrimitiveFactory.generate_primitive(value=res)

    def __ne__(self, other: Any) -> SyPrimitiveRet:
        res = super().__ne__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    def __reversed__(self) -> Any:
        # returns <class 'odict_iterator'>
        return super().__reversed__()

    def __setitem__(self, key: Any, value: Any) -> None:
        res = super().__setitem__(key, value)
        return PrimitiveFactory.generate_primitive(value=res)

    def clear(self) -> None:
        res = super().clear()
        return PrimitiveFactory.generate_primitive(value=res)

    def copy(self) -> SyPrimitiveRet:
        res = super().copy()
        return PrimitiveFactory.generate_primitive(value=res)

    @classmethod
    def FromKeys(cls, iterable: Any, value: Any = None) -> SyPrimitiveRet:
        res = cls(PyOrderedDict.fromkeys(iterable, value))
        return PrimitiveFactory.generate_primitive(value=res)

    def fromkeys(  # type: ignore
        self, iterable: Any, value: Optional[object] = None
    ) -> SyPrimitiveRet:
        res = super().fromkeys(iterable, value)
        return PrimitiveFactory.generate_primitive(value=res)

    def dict_get(self, other: Any) -> Any:
        res = super().get(other)
        if isprimitive(value=res):
            return PrimitiveFactory.generate_primitive(value=res)
        else:
            # we can have torch.Tensor and other types
            return res

    def items(self, max_len: Optional[int] = None) -> Iterator:  # type: ignore
        return Iterator(ItemsView(self), max_len=max_len)

    def keys(self, max_len: Optional[int] = None) -> Iterator:  # type: ignore
        return Iterator(KeysView(self), max_len=max_len)

    def move_to_end(self, other: Any, last: Any = True) -> Any:
        res = super().move_to_end(other, last)
        return PrimitiveFactory.generate_primitive(value=res)

    def pop(self, *args: Any, **kwargs: Any) -> SyPrimitiveRet:
        res = super().pop(*args, **kwargs)
        return PrimitiveFactory.generate_primitive(value=res)

    def popitem(self, last: Any = True) -> SyPrimitiveRet:
        res = super().popitem(last)
        return PrimitiveFactory.generate_primitive(value=res)

    def setdefault(self, key: Any, default: Optional[object] = None) -> SyPrimitiveRet:
        res = super().setdefault(key, default)
        return PrimitiveFactory.generate_primitive(value=res)

    def update(self, *args, **kwds: Any) -> SyPrimitiveRet:  # type: ignore
        res = super().update(*args, **kwds)
        return PrimitiveFactory.generate_primitive(value=res)

    def values(self, *args: Any, max_len: Optional[int] = None) -> Iterator:  # type: ignore
        # this is what the super type does and there is a test in dict_test.py
        # test_values which checks for this so we could disable the test or
        # keep this workaround
        if len(args) > 0:
            traceback_and_raise(
                TypeError("values() takes 1 positional argument but 2 were given")
            )
        return Iterator(ValuesView(self), max_len=max_len)

    def _object2proto(self) -> OrderedDict_PB:
        # serialize to bytes so that we can avoid using StorableObject
        # otherwise we get recursion where the permissions of StorableObject
        # themselves utilise Dict
        keys = [
            serialize(obj=downcast(value=element), to_bytes=True)
            for element in self.keys()
        ]
        # serialize to bytes so that we can avoid using StorableObject
        # otherwise we get recursion where the permissions of StorableObject
        # themselves utilise Dict
        values = [
            serialize(obj=downcast(value=element), to_bytes=True)
            for element in self.values()
        ]
        return OrderedDict_PB(keys=keys, values=values)

    @staticmethod
    def _proto2object(proto: OrderedDict_PB) -> "OrderedDict":
        # deserialize from bytes so that we can avoid using StorableObject
        # otherwise we get recursion where the permissions of StorableObject
        # themselves utilise OrderedDict
        values = [
            deserialize(blob=upcast(value=element), from_bytes=True)
            for element in proto.values
        ]
        # deserialize from bytes so that we can avoid using StorableObject
        # otherwise we get recursion where the permissions of StorableObject
        # themselves utilise OrderedDict
        keys = [
            deserialize(blob=upcast(value=element), from_bytes=True)
            for element in proto.keys
        ]
        new_dict = OrderedDict(dict(zip(keys, values)))
        return new_dict

    @staticmethod
    def get_protobuf_schema() -> GeneratedProtocolMessageType:
        return OrderedDict_PB

    def upcast(self) -> PyOrderedDict:
        # recursively upcast
        return OrderedDict((k, upcast(v)) for k, v in self.items())
