# stdlib
from collections import UserString
import sys
from typing import Any
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union
from typing import Mapping
from typing import Iterable

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
    def __add__(self, other: Any) -> PyPrimitive:
        res = super().__add__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __contains__(self, other: Any) -> PyPrimitive:
        res = super().__contains__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __eq__(self, other: Any) -> PyPrimitive:
        res = super().__eq__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __float__(self) -> PyPrimitive:
        res = super().__float__()
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __ge__(self, other: Any) -> PyPrimitive:
        res = super().__ge__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __getitem__(self, other: Any) -> PyPrimitive:
        res = super().__getitem__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __gt__(self, other: Any) -> PyPrimitive:
        res = super().__gt__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __hash__(self) -> PyPrimitive:
        res = super().__hash__()
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __int__(self) -> PyPrimitive:
        res = super().__int__()
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __iter__(self) -> PyPrimitive:
        #TODO fix this
        res = super().__iter__()
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __le__(self, other: Any) -> PyPrimitive:
        res = super().__le__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __len__(self) -> PyPrimitive:
        res = super().__len__()
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __lt__(self, other: Any) -> PyPrimitive:
        res = super().__lt__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __mod__(self, other: Any) -> PyPrimitive:
        res = super().__mod__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __mul__(self, other: Any) -> PyPrimitive:
        res = super().__mul__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __ne__(self) -> PyPrimitive:
        res = super().__ne__()
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __reversed__(self) -> PyPrimitive:
        res = super().__reversed__()
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __sizeof__(self) -> PyPrimitive:
        res = super().__sizeof__()
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __str__(self) -> PyPrimitive:
        res = super().__str__()
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def capitalize(self) -> PyPrimitive:
        res = super().capitalize()
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def casefold(self) -> PyPrimitive:
        res = super().casefold()
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def center(self, width: int) -> PyPrimitive:
        res = super().center(width)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def count(self, sub: Any, start: int, end: int) -> PyPrimitive:
        res = super().count(sub, start, end)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def encode(self, encoding: Any, errors: Any) -> PyPrimitive:
        res = super().encode(encoding, errors)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def endswith(self, suffix: Any, start: int = ..., end: int = ...) -> PyPrimitive:
        res = super().endswith(suffix, start, end)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def expandtabs(self, tabsize: int = ...) -> PyPrimitive:
        res = super().expandtabs(tabsize)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def find(self, sub: Any, start: int = ..., end: int = ...) -> PyPrimitive:
        res = super().find(sub, start, end)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def format(self, *args: Any, **kwargs: Any) -> PyPrimitive:
        res = super().format(*args, **kwargs)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def format_map(self, mapping: Mapping[Any, Any]) -> PyPrimitive:
        res = super().format_map(mapping)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def index(self, sub: Any, start: int = ..., end: int = ...) -> PyPrimitive:
        res = super().index(sub, start, end)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def isalnum(self) -> PyPrimitive:
        res = super().isalnum()
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def isalpha(self) -> PyPrimitive:
        res = super().isalpha()
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def isdecimal(self) -> PyPrimitive:
        res = super().isdecimal()
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def isdigit(self) -> PyPrimitive:
        res = super().isdigit()
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def isidentifier(self) -> PyPrimitive:
        res = super().isidentifier()
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def islower(self) -> PyPrimitive:
        res = super().islower()
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def isnumeric(self) -> PyPrimitive:
        res = super().isnumeric()
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def isprintable(self) -> PyPrimitive:
        res = super().isprintable()
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def isspace(self) -> PyPrimitive:
        res = super().isspace()
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def istitle(self) -> PyPrimitive:
        res = super().istitle()
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def isupper(self) -> PyPrimitive:
        res = super().isupper()
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def join(self, seq: Iterable[Any]) -> PyPrimitive:
        res = super().join(seq)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def ljust(self, width: int, *args: Any) -> PyPrimitive:
        res = super().ljust(width, *args)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def lower(self) -> PyPrimitive:
        res = super().lower()
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def lstrip(self, chars: Optional[str] = ...) -> PyPrimitive:
        res = super().lstrip(chars)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def partition(self, sep: Any) -> PyPrimitive:
        res = super().partition(sep)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def replace(self, old: Any, new: Any, maxsplit: int = ...) -> PyPrimitive:
        res = super().replace(old, new, maxsplit)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def rfind(self, sub: Any, start: int = ..., end: int = ...) -> PyPrimitive:
        res = super().rfind(sub, start, end)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def rindex(self, sub: Any, start: int = ..., end: int = ...) -> PyPrimitive:
        res = super().rindex(sub, start, end)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def rjust(self, width: int, *args: Any) -> PyPrimitive:
        res = super().rjust(width, *args)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def rpartition(self, sep: Any) -> PyPrimitive:
        res = super().rpartition(sep)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def rsplit(self, sep: Optional[Any] = ..., maxsplit: int = ...) -> PyPrimitive:
        res = super().rsplit()
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def rstrip(self, chars: Optional[Any] = ...) -> PyPrimitive:
        res = super().rstrip(chars)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def split(self, sep: Optional[Any] = ..., maxsplit: int = ...) -> PyPrimitive:
        res = super().split()
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def splitlines(self, keepends: Any = ...) -> PyPrimitive:
        res = super().splitlines()
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def startswith(self, prefix: Any, start: int = ..., end: int = ...) -> PyPrimitive:
        res = super().startswith(prefix, start, end)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def strip(self, chars: Optional[str] = ...) -> PyPrimitive:
        res = super().strip(chars)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def swapcase(self) -> PyPrimitive:
        res = super().swapcase()
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def title(self) -> PyPrimitive:
        res = super().title()
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def translate(self, *args: Any) -> PyPrimitive:
        res = super().translate(*args)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def upper(self) -> PyPrimitive:
        res = super().upper()
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def zfill(self, width: int) -> PyPrimitive:
        res = super().zfill(width)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __contains__(self, val: Any) -> PyPrimitive:
        res = super().__contains__(val)
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
