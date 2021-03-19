# stdlib
from collections import OrderedDict as PyOrderedDict
from typing import Any
from typing import Optional

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType

# syft relative
from .... import deserialize
from .... import serialize
from ....core.common.serde.serializable import bind_protobuf
from ....core.common.uid import UID
from ....proto.lib.python.collections.ordered_dict_pb2 import (
    OrderedDict as OrderedDict_PB,
)
from ..primitive_factory import PrimitiveFactory
from ..primitive_factory import isprimitive
from ..primitive_interface import PyPrimitive
from ..types import SyPrimitiveRet
from ..util import downcast
from ..util import upcast


@bind_protobuf
class OrderedDict(PyOrderedDict, PyPrimitive):
    def __init__(self, *args: Any, _id: UID = UID(), **kwds: Any):
        super().__init__(*args, **kwds)
        self._id = _id

    @property
    def id(self) -> UID:
        """We reveal PyPrimitive.id as a property to discourage users and
        developers of Syft from modifying .id attributes after an object
        has been initialized.

        :return: returns the unique id of the object
        :rtype: UID
        """
        return self._id

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

    def dict_get(self, other: Any) -> SyPrimitiveRet:
        res = super().get(other)
        if isprimitive(value=res):
            return PrimitiveFactory.generate_primitive(value=res)
        else:
            # we can have torch.Tensor and other types
            return res

    def items(self) -> SyPrimitiveRet:
        res = list(super().items())
        return PrimitiveFactory.generate_primitive(value=res)

    def keys(self) -> SyPrimitiveRet:
        res = list(super().keys())
        return PrimitiveFactory.generate_primitive(value=res)

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

    def values(self) -> SyPrimitiveRet:
        res = list(super().values())
        return PrimitiveFactory.generate_primitive(value=res)

    def _object2proto(self) -> OrderedDict_PB:
        id_ = serialize(obj=self.id)
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
        return OrderedDict_PB(id=id_, keys=keys, values=values)

    @staticmethod
    def _proto2object(proto: OrderedDict_PB) -> "OrderedDict":
        id_: UID = deserialize(blob=proto.id)
        # deserialize from bytes so that we can avoid using StorableObject
        # otherwise we get recursion where the permissions of StorableObject
        # themselves utilise OrederedDict
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
        new_dict._id = id_
        return new_dict

    @staticmethod
    def get_protobuf_schema() -> GeneratedProtocolMessageType:
        return OrderedDict_PB
