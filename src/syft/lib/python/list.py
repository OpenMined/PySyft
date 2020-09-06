# stdlib
from collections import UserList
from typing import Any
from typing import List as TypeList
from typing import Optional

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType

# syft relative
from ... import deserialize
from ... import serialize
from ...core.common import UID
from ...core.store.storeable_object import StorableObject
from ...decorators import syft_decorator
from ...proto.lib.python.list_pb2 import List as List_PB
from ...util import aggressive_set_attr
from .primitive_factory import PrimitiveFactory
from .primitive_interface import PyPrimitive


class List(UserList, PyPrimitive):
    @syft_decorator(typechecking=True, prohibit_args=False)
    def __init__(self, value: Optional[Any] = None, id: Optional[UID] = None):
        if value is None:
            value = []

        UserList.__init__(self, value)

        self._id: UID = id if id else UID()

    @property
    def id(self) -> UID:
        """We reveal PyPrimitive.id as a property to discourage users and
        developers of Syft from modifying .id attributes after an object
        has been initialized.

        :return: returns the unique id of the object
        :rtype: UID
        """
        return self._id

    @syft_decorator(typechecking=True)
    def _object2proto(self) -> List_PB:
        id_ = serialize(obj=self.id)
        data = [serialize(obj=element) for element in self.data]
        return List_PB(id=id_, data=data)

    @staticmethod
    @syft_decorator(typechecking=True)
    def _proto2object(proto: List_PB) -> "List":
        id_: UID = deserialize(blob=proto.id)
        value = [deserialize(blob=element) for element in proto.data]
        new_list = List(value=value)
        new_list._id = id_
        return new_list

    @staticmethod
    def get_protobuf_schema() -> GeneratedProtocolMessageType:
        return List_PB


class ListWrapper(StorableObject):
    def __init__(self, value: object):
        super().__init__(
            data=value,
            id=getattr(value, "id", UID()),
            tags=getattr(value, "tags", []),
            description=getattr(value, "description", ""),
        )
        self.value = value

    def _data_object2proto(self) -> List_PB:
        _object2proto = getattr(self.data, "_object2proto", None)
        if _object2proto:
            return _object2proto()

    @staticmethod
    def _data_proto2object(proto: List_PB) -> "ListWrapper":
        return List._proto2object(proto=proto)

    @staticmethod
    def get_data_protobuf_schema() -> GeneratedProtocolMessageType:
        return List_PB

    @staticmethod
    def get_wrapped_type() -> type:
        return List

    @staticmethod
    def construct_new_object(
        id: UID,
        data: StorableObject,
        description: Optional[str],
        tags: Optional[TypeList[str]],
    ) -> StorableObject:
        setattr(data, "_id", id)
        data.tags = tags
        data.description = description
        return data


aggressive_set_attr(obj=List, name="serializable_wrapper_type", attr=ListWrapper)
PrimitiveFactory.register_primitive(python_primitive=list, syft_primitive=List)
