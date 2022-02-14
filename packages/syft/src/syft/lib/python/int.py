# stdlib
import sys
from typing import Any
from typing import Optional
from typing import Union

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType
from typing_extensions import Literal
from typing_extensions import SupportsIndex

# syft absolute
import syft as sy

# relative
from ...core.common import UID
from ...core.common.serde.serializable import serializable
from ...proto.lib.python.int_pb2 import Int as Int_PB
from .primitive_factory import PrimitiveFactory
from .primitive_interface import PyPrimitive
from .types import SyPrimitiveRet


@serializable()
class Int(int, PyPrimitive):
    def __new__(
        cls, value: Any = None, base: Any = 10, id: Optional[UID] = None
    ) -> "Int":
        if value is None:
            value = 0

        if isinstance(value, str):
            return int.__new__(cls, value, base)

        return int.__new__(cls, value)

    def __init__(self, value: Any = None, base: Any = 10, id: Optional[UID] = None):
        if value is None:
            value = 0

        int.__init__(value)

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

    def upcast(self) -> int:
        return int(self)

    def __add__(self, other: Any) -> SyPrimitiveRet:
        res = super().__add__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    def __int__(self) -> int:
        res = super().__int__()
        return res

    def __invert__(self) -> SyPrimitiveRet:
        res = super().__invert__()
        return PrimitiveFactory.generate_primitive(value=res)

    def __abs__(self) -> SyPrimitiveRet:
        res = super().__abs__()
        return PrimitiveFactory.generate_primitive(value=res)

    def __bool__(self) -> bool:
        return super().__bool__()

    def __divmod__(self, other: Any) -> SyPrimitiveRet:
        res = super().__divmod__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    def __rdivmod__(self, other: Any) -> SyPrimitiveRet:
        res = super().__rdivmod__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    def __radd__(self, other: Any) -> SyPrimitiveRet:
        res = super().__radd__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    def __sub__(self, other: Any) -> SyPrimitiveRet:
        res = super().__sub__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    def __rsub__(self, other: Any) -> SyPrimitiveRet:
        res = super().__rsub__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    def __rtruediv__(self, other: Any) -> SyPrimitiveRet:
        res = super().__rtruediv__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    def __mul__(self, other: Any) -> SyPrimitiveRet:
        res = super().__mul__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    def __rmul__(self, other: Any) -> SyPrimitiveRet:
        res = super().__rmul__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    def __ceil__(self) -> SyPrimitiveRet:
        res = super().__ceil__()
        return PrimitiveFactory.generate_primitive(value=res)

    def __eq__(self, other: Any) -> SyPrimitiveRet:
        res = super().__eq__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    def __float__(self) -> SyPrimitiveRet:
        res = super().__float__()
        return PrimitiveFactory.generate_primitive(value=res)

    def __floor__(self) -> SyPrimitiveRet:
        res = super().__floor__()
        return PrimitiveFactory.generate_primitive(value=res)

    def __floordiv__(self, other: Any) -> SyPrimitiveRet:
        res = super().__floordiv__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    def __rfloordiv__(self, other: Any) -> SyPrimitiveRet:
        res = super().__rfloordiv__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    def __truediv__(self, other: Any) -> SyPrimitiveRet:
        res = super().__truediv__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    def __mod__(self, other: Any) -> SyPrimitiveRet:
        res = super().__mod__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    def __rmod__(self, other: Any) -> SyPrimitiveRet:
        res = super().__rmod__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    def __pow__(self, other: Any, modulo: Optional[Any] = None) -> SyPrimitiveRet:
        if modulo:
            return PrimitiveFactory.generate_primitive(
                value=super().__pow__(other, modulo)
            )
        return PrimitiveFactory.generate_primitive(value=super().__pow__(other))

    def __rpow__(self, other: Any, modulo: Optional[Any] = None) -> SyPrimitiveRet:
        if modulo:
            return PrimitiveFactory.generate_primitive(
                value=super().__rpow__(other, modulo)
            )
        return PrimitiveFactory.generate_primitive(value=super().__rpow__(other))

    def __lshift__(self, other: Any) -> SyPrimitiveRet:
        res = super(Int, self).__lshift__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    def __rlshift__(self, other: Any) -> SyPrimitiveRet:
        res = super().__rlshift__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    def __round__(self, ndigits: Any = 0) -> SyPrimitiveRet:
        res = super().__round__(ndigits)
        return PrimitiveFactory.generate_primitive(value=res)

    def __rshift__(self, other: Any) -> SyPrimitiveRet:
        res = super(Int, self).__rshift__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    def __rrshift__(self, other: Any) -> SyPrimitiveRet:
        res = super().__rrshift__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    def __and__(self, other: Any) -> SyPrimitiveRet:
        res = super().__and__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    def __rand__(self, other: Any) -> SyPrimitiveRet:
        res = super().__rand__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    def __xor__(self, other: Any) -> SyPrimitiveRet:
        res = super(Int, self).__xor__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    def __rxor__(self, other: Any) -> SyPrimitiveRet:
        res = super().__rxor__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    def __or__(self, other: Any) -> SyPrimitiveRet:
        res = super().__or__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    def __ror__(self, other: Any) -> SyPrimitiveRet:
        res = super().__ror__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    def __ge__(self, other: Any) -> SyPrimitiveRet:
        res = super().__ge__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    def __lt__(self, other: Any) -> SyPrimitiveRet:
        res = super().__lt__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    def __le__(self, other: Any) -> SyPrimitiveRet:
        res = super().__le__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    def __gt__(self, other: Any) -> SyPrimitiveRet:
        res = super().__gt__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    def __iadd__(self, other: Any) -> SyPrimitiveRet:
        return PrimitiveFactory.generate_primitive(
            value=super().__add__(other), id=self.id
        )

    def __isub__(self, other: Any) -> SyPrimitiveRet:
        return PrimitiveFactory.generate_primitive(
            value=super().__sub__(other), id=self.id
        )

    def __imul__(self, other: Any) -> SyPrimitiveRet:
        return PrimitiveFactory.generate_primitive(
            value=super().__mul__(other), id=self.id
        )

    def __ifloordiv__(self, other: Any) -> SyPrimitiveRet:
        return PrimitiveFactory.generate_primitive(
            value=super().__floordiv__(other), id=self.id
        )

    def __itruediv__(self, other: Any) -> SyPrimitiveRet:
        return PrimitiveFactory.generate_primitive(
            value=super().__truediv__(other), id=self.id
        )

    def __imod__(self, other: Any) -> SyPrimitiveRet:
        return PrimitiveFactory.generate_primitive(
            value=super().__mod__(other), id=self.id
        )

    def __ipow__(self, other: Any, modulo: Optional[Any] = None) -> SyPrimitiveRet:
        if modulo:
            PrimitiveFactory.generate_primitive(
                value=super().__pow__(other, modulo), id=self.id
            )
        return PrimitiveFactory.generate_primitive(
            value=super().__pow__(other), id=self.id
        )

    def __ne__(self, other: Any) -> SyPrimitiveRet:
        res = super().__ne__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    def __hash__(self) -> SyPrimitiveRet:
        return PrimitiveFactory.generate_primitive(value=super().__hash__())

    # add tests

    def __neg__(self) -> SyPrimitiveRet:
        return PrimitiveFactory.generate_primitive(value=super().__neg__())

    def __pos__(self) -> SyPrimitiveRet:
        return PrimitiveFactory.generate_primitive(value=super().__pos__())

    def _object2proto(self) -> Int_PB:
        int_pb = Int_PB()
        int_pb.data = self
        int_pb.id.CopyFrom(sy.serialize(obj=self.id))
        return int_pb

    @staticmethod
    def _proto2object(proto: Int_PB) -> "Int":
        int_id: UID = sy.deserialize(blob=proto.id)

        de_int = Int(value=proto.data)
        de_int._id = int_id  # can't use uid=int_id for some reason

        return de_int

    @staticmethod
    def get_protobuf_schema() -> GeneratedProtocolMessageType:
        return Int_PB

    def as_integer_ratio(self) -> SyPrimitiveRet:
        if sys.version_info < (3, 8):
            raise NotImplementedError
        else:
            tpl = super().as_integer_ratio()
            return PrimitiveFactory.generate_primitive(value=tpl)

    def bit_length(self) -> SyPrimitiveRet:
        res = super().bit_length()
        return PrimitiveFactory.generate_primitive(value=res)

    def denominator(self) -> SyPrimitiveRet:
        res = super().denominator
        return PrimitiveFactory.generate_primitive(value=res)

    def to_bytes(
        self,
        length: Union[int, SupportsIndex],
        byteorder: Literal["little", "big"],
        signed: bool = False,
    ) -> bytes:
        return int.to_bytes(self, length=length, byteorder=byteorder, signed=signed)

    @staticmethod
    def from_bytes(
        bytes: Any, byteorder: Literal["little", "big"], *, signed: Any = True
    ) -> SyPrimitiveRet:
        res = int.from_bytes(bytes, byteorder, signed=signed)
        return PrimitiveFactory.generate_primitive(value=res)

    def imag(self) -> SyPrimitiveRet:
        res = super().imag
        return PrimitiveFactory.generate_primitive(value=res)

    def numerator(self) -> int:
        res = super().numerator
        return PrimitiveFactory.generate_primitive(value=res)

    def real(self) -> int:
        res = super().real
        return PrimitiveFactory.generate_primitive(value=res)

    def conjugate(self) -> SyPrimitiveRet:
        res = super().conjugate()
        return PrimitiveFactory.generate_primitive(value=res)
