# stdlib
from collections import UserString
import string
from typing import Any
from typing import List
from typing import Optional
from typing import Union

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType

# syft relative
from ... import deserialize
from ... import serialize
from ...core.common import UID
from ...core.store.storeable_object import StorableObject
from ...decorators import syft_decorator
from ...proto.lib.python.string_pb2 import String as String_PB
from ...util import aggressive_set_attr
from .primitive_factory import PrimitiveFactory
from .primitive_interface import PyPrimitive


class String(UserString, PyPrimitive):
    @syft_decorator(typechecking=True, prohibit_args=False)
    def __init__(self, value: Any = None, id: Optional[UID] = None):
        if value is None:
            value = ""

        UserString.__init__(self, value)

        self._id: UID = id if id else UID()

    @syft_decorator(typechecking=True, prohibit_args=True)
    def split(
        self, separator: Union[str, UserString] = string.whitespace, maxsplit: int = -1
    ) -> PyPrimitive:
        res = self.split(separator, maxsplit)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=True)
    def rsplit(
        self, separator: Union[str, UserString] = string.whitespace, maxsplit: int = -1
    ) -> PyPrimitive:
        res = self.rsplit(separator, maxsplit)
        return PrimitiveFactory.generate_primitive(value=res)

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
    def _object2proto(self) -> String_PB:
        return String_PB(data=self.data, id=serialize(obj=self.id))

    @staticmethod
    @syft_decorator(typechecking=True)
    def _proto2object(proto: String_PB) -> "String":
        str_id: UID = deserialize(blob=proto.id)
        return String(value=proto.data, id=str_id)

    @staticmethod
    def get_protobuf_schema() -> GeneratedProtocolMessageType:
        return String_PB


class StringWrapper(StorableObject):
    def __init__(self, value: object):
        super().__init__(
            data=value,
            id=getattr(value, "id", UID()),
            tags=getattr(value, "tags", []),
            description=getattr(value, "description", ""),
        )
        self.value = value

    def _data_object2proto(self) -> String_PB:
        _object2proto = getattr(self.data, "_object2proto", None)
        if _object2proto:
            return _object2proto()

    @staticmethod
    def _data_proto2object(proto: String_PB) -> "StringWrapper":
        return String._proto2object(proto=proto)

    @staticmethod
    def get_data_protobuf_schema() -> GeneratedProtocolMessageType:
        return String_PB

    @staticmethod
    def get_wrapped_type() -> type:
        return String

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


aggressive_set_attr(obj=String, name="serializable_wrapper_type", attr=StringWrapper)
