# stdlib
from collections import UserString
from typing import Any
from typing import List
from typing import Mapping
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
from .int import Int
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
        # TODO fix this
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
    def __mod__(self, *args: Any) -> PyPrimitive:
        res = super().__mod__(
            *[str(arg) if isinstance(arg, String) else arg for arg in args]
        )
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __mul__(self, other: Any) -> PyPrimitive:
        res = super().__mul__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __ne__(self, other: Any) -> PyPrimitive:
        res = super().__ne__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __reversed__(self) -> Any:
        # returns <class 'reversed'>
        return super().__reversed__()

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __sizeof__(self) -> PyPrimitive:
        res = super().__sizeof__()
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __str__(self) -> str:
        return super().__str__()

    @syft_decorator(typechecking=True, prohibit_args=False)
    def capitalize(self) -> PyPrimitive:
        res = super().capitalize()
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def casefold(self) -> PyPrimitive:
        res = super().casefold()
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def center(
        self, width: Union[int, Int], character: Union[str, "String"] = " "
    ) -> PyPrimitive:
        character = str(character) if isinstance(character, String) else character
        res = super().center(width, character)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def count(
        self, sub: Any, start: Optional[int] = None, end: Optional[int] = None
    ) -> PyPrimitive:
        res = super().count(sub, start, end)  # type: ignore
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def encode(self, encoding: Any, errors: Any) -> PyPrimitive:
        res = super().encode(encoding, errors)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def endswith(
        self,
        suffix: Union[str, "String", tuple],
        start: Optional[int] = None,
        end: Optional[int] = None,
    ) -> PyPrimitive:
        suffix = str(suffix) if isinstance(suffix, String) else suffix
        _suffix = (
            tuple(str(elem) if isinstance(elem, String) else elem for elem in suffix)
            if isinstance(suffix, tuple)
            else suffix
        )
        res = super().endswith(_suffix, start, end)  # type: ignore
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def expandtabs(self, tabsize: int = 8) -> PyPrimitive:
        res = super().expandtabs(tabsize)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def find(
        self, sub: Any, start: Optional[int] = 0, end: Optional[int] = None
    ) -> PyPrimitive:
        if end is None:
            end = super().__len__()

        res = super().find(sub, start, end)  # type: ignore
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
    def index(
        self,
        sub: Union[str, "String"],
        start: Optional[int] = 0,
        end: Optional[int] = None,
    ) -> PyPrimitive:
        if end is None:
            end = super().__len__()
        res = super().index(str(sub), start, end)  # type: ignore
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def isalnum(self) -> PyPrimitive:
        res = super().isalnum()
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def isascii(self) -> PyPrimitive:
        res = super().isascii()  # type: ignore
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
    def join(self, seq: Any) -> PyPrimitive:
        res = super().join(
            [str(elem) if isinstance(elem, String) else elem for elem in seq]
        )
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def ljust(self, width: Union[int], fill: Union[str, "String"] = " ") -> PyPrimitive:
        fill = str(fill) if isinstance(fill, String) else fill
        res = super().ljust(width, fill)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def lower(self) -> PyPrimitive:
        res = super().lower()
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def lstrip(self, chars: Optional[Union[str, "String"]] = None) -> PyPrimitive:
        chars = str(chars) if isinstance(chars, String) else chars
        res = super().lstrip(chars)  # type: ignore
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def partition(self, sep: Optional[Union[str, "String"]] = " ") -> PyPrimitive:
        sep = str(sep) if isinstance(sep, String) else sep
        res = super().partition(sep)  # type: ignore
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def replace(
        self,
        oldvalue: Union[str, "String"],
        newvalue: Union[str, "String"],
        count: Optional[int] = -1,
    ) -> PyPrimitive:
        res = super().replace(str(oldvalue), str(newvalue), count)  # type: ignore
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def rfind(
        self,
        sub: Union[str, "String"],
        start: Optional[int] = 0,
        end: Optional[int] = None,
    ) -> PyPrimitive:
        sub = str(sub) if isinstance(sub, String) else sub
        res = super().rfind(sub, start, end)  # type: ignore
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def rindex(
        self,
        sub: Union[str, "String"],
        start: Optional[int] = 0,
        end: Optional[int] = None,
    ) -> PyPrimitive:
        sub = str(sub) if isinstance(sub, String) else sub
        res = super().rindex(sub, start, end)  # type: ignore
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def rjust(self, width: int, fill: Union[str, "String"] = " ") -> PyPrimitive:
        fill = str(fill) if isinstance(fill, String) else fill
        res = super().rjust(width, fill)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def rpartition(self, sep: Optional[Union[str, "String"]] = " ") -> PyPrimitive:
        sep = str(sep) if isinstance(sep, String) else sep
        res = super().rpartition(sep)  # type: ignore
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def rsplit(
        self, sep: Optional[Union[str, "String"]] = None, maxsplit: int = -1
    ) -> PyPrimitive:
        sep = str(sep) if isinstance(sep, String) else sep
        res = super().rsplit(sep=sep, maxsplit=maxsplit)  # type: ignore
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def rstrip(self, chars: Optional[Union[str, "String"]] = None) -> PyPrimitive:
        chars = str(chars) if isinstance(chars, String) else chars
        res = super().rstrip(chars)  # type: ignore
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def split(
        self, sep: Optional[Union[str, "String"]] = None, maxsplit: int = -1
    ) -> PyPrimitive:
        sep = str(sep) if isinstance(sep, String) else sep
        res = super().split(sep=sep, maxsplit=maxsplit)  # type: ignore
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def splitlines(self, keepends: bool = False) -> PyPrimitive:
        res = super().splitlines(keepends)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def startswith(
        self,
        suffix: Union[str, "String", tuple],
        start: Optional[int] = None,
        end: Optional[int] = None,
    ) -> PyPrimitive:
        suffix = str(suffix) if isinstance(suffix, String) else suffix
        suffix = (
            tuple(str(elem) if isinstance(elem, String) else elem for elem in suffix)
            if isinstance(suffix, tuple)
            else suffix
        )
        res = super().startswith(suffix, start, end)  # type: ignore
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def strip(self, chars: Optional[Union[str, "String"]] = None) -> PyPrimitive:
        chars = str(chars) if isinstance(chars, String) else chars
        res = super().strip(chars)  # type: ignore
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
    def zfill(self, width: Union[int, Int]) -> PyPrimitive:
        res = super().zfill(width)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __contains__(self, val: Union[str, "String"]) -> PyPrimitive:
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
