# stdlib
from collections import OrderedDict as PyOrderedDict
from typing import Any
from typing import List
from typing import Optional

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType

# syft relative
from .... import deserialize
from .... import serialize
from ....core.common.uid import UID
from ....core.store.storeable_object import StorableObject
from ....decorators import syft_decorator
from ....proto.lib.python.collections.ordered_dict_pb2 import (
    OrderedDict as OrderedDict_PB,
)
from ....util import aggressive_set_attr
from ..primitive_factory import PrimitiveFactory
from ..primitive_interface import PyPrimitive
from ..util import SyPrimitiveRet
from ..util import downcast
from ..util import upcast


class OrderedDict(PyOrderedDict, PyPrimitive):
    def __init__(self, other: Any = None, _id: Optional[UID] = None):
        if other is None:
            other = {}

        super().__init__(other)

        self._id = UID() if _id is None else _id

    @property
    def id(self) -> UID:
        """We reveal PyPrimitive.id as a property to discourage users and
        developers of Syft from modifying .id attributes after an object
        has been initialized.

        :return: returns the unique id of the object
        :rtype: UID
        """
        return self._id

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __contains__(self, other: Any) -> SyPrimitiveRet:
        res = super().__contains__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __delitem__(self, other: Any) -> SyPrimitiveRet:
        res = super().__delitem__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __eq__(self, other: Any) -> SyPrimitiveRet:
        res = super().__eq__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __getitem__(self, other: Any) -> SyPrimitiveRet:
        res = super().__getitem__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __len__(self) -> SyPrimitiveRet:
        res = super().__len__()
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __ne__(self, other: Any) -> SyPrimitiveRet:
        res = super().__ne__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __reversed__(self) -> Any:
        # returns <class 'odict_iterator'>
        return super().__reversed__()

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __setitem__(self, key: Any, value: Any) -> SyPrimitiveRet:
        res = super().__setitem__(key, value)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def clear(self) -> SyPrimitiveRet:
        res = super().clear()
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def copy(self) -> SyPrimitiveRet:
        res = super().copy()
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def fromkeys(self, other: Any) -> SyPrimitiveRet:
        res = super().fromkeys(other)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def dict_get(self, other: Any) -> SyPrimitiveRet:
        res = super().get(other)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def items(self) -> SyPrimitiveRet:
        res = list(super().items())
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def keys(self) -> SyPrimitiveRet:
        res = list(super().keys())
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def move_to_end(self, other: Any, last: Any = True) -> SyPrimitiveRet:
        res = super().move_to_end(other, last)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def pop(self, other: Any) -> SyPrimitiveRet:
        res = super().pop(other)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def popitem(self, last: Any = True) -> SyPrimitiveRet:
        res = super().popitem(last)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def setdefault(self, key: Any, default: Any) -> SyPrimitiveRet:
        res = super().setdefault(key, default)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def update(self, other: Any) -> SyPrimitiveRet:
        res = super().update(other)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def values(self) -> SyPrimitiveRet:
        res = list(super().values())
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True)
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
    @syft_decorator(typechecking=True)
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


class DictWrapper(StorableObject):
    def __init__(self, value: object):
        super().__init__(
            data=value,
            id=getattr(value, "id", UID()),
            tags=getattr(value, "tags", []),
            description=getattr(value, "description", ""),
        )
        self.value = value

    def _data_object2proto(self) -> OrderedDict_PB:
        _object2proto = getattr(self.data, "_object2proto", None)
        if _object2proto:
            return _object2proto()

    @staticmethod
    def _data_proto2object(proto: OrderedDict_PB) -> "OrderedDict":  # type: ignore
        return OrderedDict._proto2object(proto=proto)

    @staticmethod
    def get_data_protobuf_schema() -> GeneratedProtocolMessageType:
        return OrderedDict_PB

    @staticmethod
    def get_wrapped_type() -> type:
        return OrderedDict

    @staticmethod
    def construct_new_object(
        id: UID,
        data: StorableObject,
        description: Optional[str],
        tags: Optional[List[str]],
    ) -> StorableObject:
        setattr(data, "_id", id)
        data.tags = tags
        data.description = description
        return data


aggressive_set_attr(obj=OrderedDict, name="serializable_wrapper_type", attr=DictWrapper)
