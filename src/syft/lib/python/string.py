# stdlib
from collections import UserString
import sys
from typing import Any
from typing import List
from typing import Optional
from typing import Tuple
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
    def upcast(self) -> str:
        return str(self)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def count(
        self,
        substring: Union[str, UserString],
        start: Optional[Union[int, None]] = None,
        end: Optional[Union[int, None]] = None,
    ) -> PyPrimitive:
        start_pos = 0 if start is None else start
        end_pos = len(self.data) if end is None else end
        res = super().count(substring, start_pos, end_pos)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __contains__(self, val: Union[str, PyPrimitive]) -> PyPrimitive:
        res = super().__contains__(val)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def endswith(
        self,
        val: Union[str, Tuple[str, ...]],
        start: Optional[Union[int, None]] = None,
        end: Optional[Union[int, None]] = None,
    ) -> PyPrimitive:
        start_pos = 0 if start is None else start
        end_pos = len(self.data) if end is None else end
        res = super().endswith(val, start_pos, end_pos)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def startswith(
        self,
        val: Union[str, Tuple[str, ...]],
        start: Optional[Union[int, None]] = None,
        end: Optional[Union[int, None]] = None,
    ) -> PyPrimitive:
        start_pos = 0 if start is None else start
        end_pos = len(self.data) if end is None else end
        res = super().startswith(val, start_pos, end_pos)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def partition(self, val: str) -> PyPrimitive:
        res = super().partition(val)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def rpartition(self, val: str) -> PyPrimitive:
        res = super().rpartition(val)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def splitlines(
        self,
        keepends: bool = False,
    ) -> PyPrimitive:
        res = super().splitlines(keepends)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def find(
        self,
        val: Union[str, UserString],
        start: Optional[Union[int, None]] = None,
        end: Optional[Union[int, None]] = None,
    ) -> PyPrimitive:
        start_pos = 0 if start is None else start
        end_pos = len(self.data) if end is None else end
        res = super().find(val, start_pos, end_pos)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def rfind(
        self,
        val: Union[str, UserString],
        start: Optional[Union[int, None]] = None,
        end: Optional[Union[int, None]] = None,
    ) -> PyPrimitive:
        start_pos = 0 if start is None else start
        end_pos = len(self.data) if end is None else end
        res = super().rfind(val, start_pos, end_pos)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def index(
        self,
        val: str,
        start: Optional[Union[int, None]] = None,
        end: Optional[Union[int, None]] = None,
    ) -> PyPrimitive:
        start_pos = 0 if start is None else start
        end_pos = len(self.data) if end is None else end
        res = super().index(val, start_pos, end_pos)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def rindex(
        self,
        val: str,
        start: Optional[Union[int, None]] = None,
        end: Optional[Union[int, None]] = None,
    ) -> PyPrimitive:
        start_pos = 0 if start is None else start
        end_pos = len(self.data) if end is None else end
        res = super().rindex(val, start_pos, end_pos)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def join(self, val: str) -> PyPrimitive:
        res = super().join(val)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=True)
    def isupper(self) -> PyPrimitive:
        res = super().isupper()
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=True)
    def islower(self) -> PyPrimitive:
        res = super().islower()
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=True)
    def isascii(self) -> PyPrimitive:
        if sys.version_info >= (3, 7):
            res = self.data.isascii()
            return PrimitiveFactory.generate_primitive(value=res)
        else:
            raise AttributeError("'String' object has no attribute 'isascii'")

    @syft_decorator(typechecking=True, prohibit_args=True)
    def isalpha(self) -> PyPrimitive:
        res = super().isalpha()
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=True)
    def isspace(self) -> PyPrimitive:
        res = super().isspace()
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=True)
    def istitle(self) -> PyPrimitive:
        res = super().istitle()
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=True)
    def isalnum(self) -> PyPrimitive:
        res = super().isalnum()
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=True)
    def isdigit(self) -> PyPrimitive:
        res = super().isdigit()
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def split(self, sep: Optional[str] = None, maxsplit: int = -1) -> PyPrimitive:
        res = super().split(sep, maxsplit)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def rsplit(self, sep: Optional[str] = None, maxsplit: int = -1) -> PyPrimitive:
        res = super().rsplit(sep, maxsplit)
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

    # fixes __rmod__ in python <= 3.7
    # https://github.com/python/cpython/commit/7abf8c60819d5749e6225b371df51a9c5f1ea8e9
    def __rmod__(self, template: Union[PyPrimitive, str]) -> PyPrimitive:
        return self.__class__(str(template) % self)


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
