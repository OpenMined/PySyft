# future
from __future__ import annotations

# stdlib
from collections import UserString
from typing import Any
from typing import Mapping
from typing import Optional
from typing import Union

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType
from typing_extensions import SupportsIndex

# relative
from ...core.common.serde.serializable import serializable
from ...proto.lib.python.string_pb2 import String as String_PB
from .int import Int
from .primitive_factory import PrimitiveFactory
from .primitive_interface import PyPrimitive
from .slice import Slice
from .types import SyPrimitiveRet


@serializable()
class String(UserString, PyPrimitive):
    def __init__(
        self,
        value: Any = None,
        temporary_box: bool = False,
    ):

        if value is None:
            value = ""

        UserString.__init__(self, value)
        PyPrimitive.__init__(self, temporary_box=temporary_box)

    def upcast(self) -> str:
        return str(self)

    def __add__(self, other: Any) -> SyPrimitiveRet:
        res = super().__add__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    def __eq__(self, other: Any) -> SyPrimitiveRet:
        res = super().__eq__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    def __float__(self) -> SyPrimitiveRet:
        res = super().__float__()
        return PrimitiveFactory.generate_primitive(value=res)

    def __ge__(self, other: Any) -> SyPrimitiveRet:
        res = super().__ge__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    def __getitem__(self, key: Union[int, slice, Slice, SupportsIndex]) -> Any:
        if isinstance(key, Slice):
            key = key.upcast()
        res = super().__getitem__(key)
        return PrimitiveFactory.generate_primitive(value=res)

    def __gt__(self, other: Any) -> SyPrimitiveRet:
        res = super().__gt__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    def __hash__(self) -> SyPrimitiveRet:
        res = super().__hash__()
        return PrimitiveFactory.generate_primitive(value=res)

    def __int__(self) -> SyPrimitiveRet:
        res = super().__int__()
        return PrimitiveFactory.generate_primitive(value=res)

    def __iter__(self) -> SyPrimitiveRet:
        # TODO fix this
        res = super().__iter__()
        return PrimitiveFactory.generate_primitive(value=res)

    def __le__(self, other: Any) -> SyPrimitiveRet:
        res = super().__le__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    def __len__(self) -> SyPrimitiveRet:
        res = super().__len__()
        return PrimitiveFactory.generate_primitive(value=res)

    def __lt__(self, other: Any) -> SyPrimitiveRet:
        res = super().__lt__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    def __mod__(self, *args: Any) -> SyPrimitiveRet:
        res = super().__mod__(
            *[str(arg) if isinstance(arg, String) else arg for arg in args]
        )
        return PrimitiveFactory.generate_primitive(value=res)

    def __mul__(self, other: Any) -> SyPrimitiveRet:
        res = super().__mul__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    def __ne__(self, other: Any) -> SyPrimitiveRet:
        res = super().__ne__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    def __reversed__(self) -> Any:
        # returns <class 'reversed'>
        return super().__reversed__()

    def __sizeof__(self) -> SyPrimitiveRet:
        res = super().__sizeof__()
        return PrimitiveFactory.generate_primitive(value=res)

    def __str__(self) -> str:
        return super().__str__()

    def capitalize(self) -> SyPrimitiveRet:
        res = super().capitalize()
        return PrimitiveFactory.generate_primitive(value=res)

    def casefold(self) -> SyPrimitiveRet:
        res = super().casefold()
        return PrimitiveFactory.generate_primitive(value=res)

    def center(self, width: Union[int, Int], *args: Any) -> SyPrimitiveRet:
        if args:
            _args_0 = str(args[0]) if isinstance(args[0], String) else args[0]
            res = super().center(width, _args_0, *args[1:])
        else:
            res = super().center(width)
        return PrimitiveFactory.generate_primitive(value=res)

    def count(
        self, sub: Any, start: Optional[int] = None, end: Optional[int] = None
    ) -> SyPrimitiveRet:
        res = super().count(sub, start, end)  # type: ignore
        return PrimitiveFactory.generate_primitive(value=res)

    def encode(
        self, encoding: Optional[str] = None, errors: Optional[str] = None
    ) -> SyPrimitiveRet:
        res = super().encode(encoding, errors)
        return PrimitiveFactory.generate_primitive(value=res)

    def endswith(
        self,
        suffix: Union[str, "String", tuple],
        start: Optional[int] = None,
        end: Optional[int] = None,
    ) -> SyPrimitiveRet:
        suffix = str(suffix) if isinstance(suffix, String) else suffix
        _suffix = (
            tuple(str(elem) if isinstance(elem, String) else elem for elem in suffix)
            if isinstance(suffix, tuple)
            else suffix
        )
        res = super().endswith(_suffix, start, end)  # type: ignore
        return PrimitiveFactory.generate_primitive(value=res)

    def expandtabs(self, tabsize: int = 8) -> SyPrimitiveRet:
        res = super().expandtabs(tabsize)
        return PrimitiveFactory.generate_primitive(value=res)

    def find(
        self, sub: Any, start: Optional[int] = 0, end: Optional[int] = None
    ) -> SyPrimitiveRet:
        if end is None:
            end = super().__len__()

        res = super().find(sub, start, end)  # type: ignore
        return PrimitiveFactory.generate_primitive(value=res)

    def format(self, *args: Any, **kwargs: Any) -> SyPrimitiveRet:
        res = super().format(*args, **kwargs)
        return PrimitiveFactory.generate_primitive(value=res)

    def format_map(self, mapping: Mapping[Any, Any]) -> SyPrimitiveRet:
        res = super().format_map(mapping)
        return PrimitiveFactory.generate_primitive(value=res)

    def index(
        self,
        sub: Union[str, "String"],
        start: Optional[int] = 0,
        end: Optional[int] = None,
    ) -> SyPrimitiveRet:
        if end is None:
            end = super().__len__()
        res = super().index(str(sub), start, end)  # type: ignore
        return PrimitiveFactory.generate_primitive(value=res)

    def isalnum(self) -> SyPrimitiveRet:
        res = super().isalnum()
        return PrimitiveFactory.generate_primitive(value=res)

    def isascii(self) -> SyPrimitiveRet:
        res = super().isascii()  # type: ignore
        return PrimitiveFactory.generate_primitive(value=res)

    def isalpha(self) -> SyPrimitiveRet:
        res = super().isalpha()
        return PrimitiveFactory.generate_primitive(value=res)

    def isdecimal(self) -> SyPrimitiveRet:
        res = super().isdecimal()
        return PrimitiveFactory.generate_primitive(value=res)

    def isdigit(self) -> SyPrimitiveRet:
        res = super().isdigit()
        return PrimitiveFactory.generate_primitive(value=res)

    def isidentifier(self) -> SyPrimitiveRet:
        res = super().isidentifier()
        return PrimitiveFactory.generate_primitive(value=res)

    def islower(self) -> SyPrimitiveRet:
        res = super().islower()
        return PrimitiveFactory.generate_primitive(value=res)

    def isnumeric(self) -> SyPrimitiveRet:
        res = super().isnumeric()
        return PrimitiveFactory.generate_primitive(value=res)

    def isprintable(self) -> SyPrimitiveRet:
        res = super().isprintable()
        return PrimitiveFactory.generate_primitive(value=res)

    def isspace(self) -> SyPrimitiveRet:
        res = super().isspace()
        return PrimitiveFactory.generate_primitive(value=res)

    def istitle(self) -> SyPrimitiveRet:
        res = super().istitle()
        return PrimitiveFactory.generate_primitive(value=res)

    def isupper(self) -> SyPrimitiveRet:
        res = super().isupper()
        return PrimitiveFactory.generate_primitive(value=res)

    def join(self, seq: Any) -> SyPrimitiveRet:
        res = super().join(
            [str(elem) if isinstance(elem, String) else elem for elem in seq]
        )
        return PrimitiveFactory.generate_primitive(value=res)

    def ljust(self, width: int, *args: Any) -> SyPrimitiveRet:
        if args:
            _args_0 = str(args[0]) if isinstance(args[0], String) else args[0]
            res = super().ljust(width, _args_0, *args[1:])
        else:
            res = super().ljust(width)
        return PrimitiveFactory.generate_primitive(value=res)

    def lower(self) -> SyPrimitiveRet:
        res = super().lower()
        return PrimitiveFactory.generate_primitive(value=res)

    def lstrip(self, chars: Optional[Union[str, "String"]] = None) -> SyPrimitiveRet:
        chars = str(chars) if isinstance(chars, String) else chars
        res = super().lstrip(chars)
        return PrimitiveFactory.generate_primitive(value=res)

    def partition(self, sep: Optional[Union[str, "String"]] = " ") -> SyPrimitiveRet:
        sep = str(sep) if isinstance(sep, String) else sep
        res = super().partition(sep)  # type: ignore
        return PrimitiveFactory.generate_primitive(value=res)

    def replace(
        self,
        oldvalue: Union[str, UserString],
        newvalue: Union[str, UserString],
        count: Optional[int] = -1,
    ) -> SyPrimitiveRet:
        res = super().replace(str(oldvalue), str(newvalue), count)  # type: ignore
        return PrimitiveFactory.generate_primitive(value=res)

    def rfind(
        self,
        sub: Union[str, UserString],
        start: Optional[int] = 0,
        end: Optional[int] = None,
    ) -> SyPrimitiveRet:
        sub = str(sub) if isinstance(sub, UserString) else sub
        res = super().rfind(sub, start, end)  # type: ignore
        return PrimitiveFactory.generate_primitive(value=res)

    def rindex(
        self,
        sub: Union[str, UserString],
        start: Optional[int] = 0,
        end: Optional[int] = None,
    ) -> SyPrimitiveRet:
        sub = str(sub) if isinstance(sub, String) else sub
        res = super().rindex(sub, start, end)  # type: ignore
        return PrimitiveFactory.generate_primitive(value=res)

    def rjust(self, width: int, *args: Any) -> SyPrimitiveRet:
        if args:
            _args_0 = str(args[0]) if isinstance(args[0], String) else args[0]
            res = super().rjust(width, _args_0, *args[1:])
        else:
            res = super().rjust(width)
        return PrimitiveFactory.generate_primitive(value=res)

    def rpartition(self, sep: Optional[Union[str, "String"]] = " ") -> SyPrimitiveRet:
        sep = str(sep) if isinstance(sep, String) else sep
        res = super().rpartition(sep)  # type: ignore
        return PrimitiveFactory.generate_primitive(value=res)

    def rsplit(
        self, sep: Optional[Union[str, "String"]] = None, maxsplit: int = -1
    ) -> SyPrimitiveRet:
        sep = str(sep) if isinstance(sep, String) else sep
        res = super().rsplit(sep=sep, maxsplit=maxsplit)
        return PrimitiveFactory.generate_primitive(value=res)

    def rstrip(self, chars: Optional[Union[str, "String"]] = None) -> SyPrimitiveRet:
        chars = str(chars) if isinstance(chars, String) else chars
        res = super().rstrip(chars)
        return PrimitiveFactory.generate_primitive(value=res)

    def split(
        self, sep: Optional[Union[str, "String"]] = None, maxsplit: int = -1
    ) -> SyPrimitiveRet:
        sep = str(sep) if isinstance(sep, String) else sep
        res = super().split(sep=sep, maxsplit=maxsplit)
        return PrimitiveFactory.generate_primitive(value=res)

    def splitlines(self, keepends: bool = False) -> SyPrimitiveRet:
        res = super().splitlines(keepends)
        return PrimitiveFactory.generate_primitive(value=res)

    def startswith(
        self,
        suffix: Union[str, UserString, tuple],
        start: Optional[int] = None,
        end: Optional[int] = None,
    ) -> SyPrimitiveRet:
        suffix = str(suffix) if isinstance(suffix, UserString) else suffix
        suffix = (
            tuple(
                str(elem) if isinstance(elem, UserString) else elem for elem in suffix
            )
            if isinstance(suffix, tuple)
            else suffix
        )
        start = start if start else 0
        end = end if end else len(self)
        res = super().startswith(suffix, start, end)
        return PrimitiveFactory.generate_primitive(value=res)

    def strip(self, chars: Optional[str] = None) -> SyPrimitiveRet:
        chars = str(chars) if isinstance(chars, String) else chars  # type: ignore
        res = super().strip(chars)
        return PrimitiveFactory.generate_primitive(value=res)

    def swapcase(self) -> SyPrimitiveRet:
        res = super().swapcase()
        return PrimitiveFactory.generate_primitive(value=res)

    def title(self) -> SyPrimitiveRet:
        res = super().title()
        return PrimitiveFactory.generate_primitive(value=res)

    def translate(self, *args: Any) -> SyPrimitiveRet:
        res = super().translate(*args)
        return PrimitiveFactory.generate_primitive(value=res)

    def upper(self) -> SyPrimitiveRet:
        res = super().upper()
        return PrimitiveFactory.generate_primitive(value=res)

    def zfill(self, width: Union[int, Int]) -> SyPrimitiveRet:
        res = super().zfill(width)
        return PrimitiveFactory.generate_primitive(value=res)

    def __contains__(self, val: object) -> SyPrimitiveRet:
        res = super().__contains__(val)
        return PrimitiveFactory.generate_primitive(value=res)

    def _object2proto(self) -> String_PB:
        return String_PB(
            data=self.data,
            temporary_box=self.temporary_box,
        )

    @staticmethod
    def _proto2object(proto: String_PB) -> "String":
        return String(
            value=proto.data,
            temporary_box=proto.temporary_box,
        )

    @staticmethod
    def get_protobuf_schema() -> GeneratedProtocolMessageType:
        return String_PB

    # fixes __rmod__ in python <= 3.7
    # https://github.com/python/cpython/commit/7abf8c60819d5749e6225b371df51a9c5f1ea8e9
    def __rmod__(
        self, template: Union[PyPrimitive, str, object]
    ) -> Union[SyPrimitiveRet, String]:
        return self.__class__(str(template) % self)
