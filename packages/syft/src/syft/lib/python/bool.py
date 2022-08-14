# stdlib
from typing import Any
from typing import Optional

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType

# syft absolute
import syft as sy

# relative
from ...core.common import UID
from ...core.common.serde.serializable import serializable
from ...proto.lib.python.bool_pb2 import Bool as Bool_PB
from .primitive_factory import PrimitiveFactory
from .primitive_interface import PyPrimitive
from .types import SyPrimitiveRet


def dispatch_other(obj: Any) -> bool:
    if isinstance(obj, Bool):
        return obj.value
    return obj


@serializable()
class Bool(int, PyPrimitive):
    def __new__(cls, value: Any = None, id: Optional[UID] = None) -> "Bool":
        value = bool(value)
        obj = int.__new__(cls, value)
        return obj

    def __init__(self, value: Any = None, id: Optional[UID] = None):
        self.value: bool = bool(value)
        self._id: UID = id if id else UID()
        self.my_field: int = 0

    @property
    def id(self) -> UID:
        """We reveal PyPrimitive.id as a property to discourage users and
        developers of Syft from modifying .id attributes after an object
        has been initialized.
        :return: returns the unique id of the object
        :rtype: UID
        """
        return self._id

    def upcast(self) -> bool:
        return bool(self)

    def __abs__(self) -> SyPrimitiveRet:
        return PrimitiveFactory.generate_primitive(value=self.value.__abs__())

    def __add__(self, other: Any) -> SyPrimitiveRet:
        other = dispatch_other(other)
        return PrimitiveFactory.generate_primitive(value=self.value.__add__(other))

    def __and__(self, other: Any) -> SyPrimitiveRet:
        other = dispatch_other(other)
        return PrimitiveFactory.generate_primitive(value=self.value.__and__(other))

    def __bool__(self) -> bool:
        return bool(self.value)

    def __ceil__(self) -> SyPrimitiveRet:
        return PrimitiveFactory.generate_primitive(value=self.value.__ceil__())

    def __divmod__(self, other: Any) -> SyPrimitiveRet:
        other = dispatch_other(other)
        tpl = self.value.__divmod__(other)
        return PrimitiveFactory.generate_primitive(value=tpl)

    def __eq__(self, other: Any) -> SyPrimitiveRet:
        other = dispatch_other(other)
        return PrimitiveFactory.generate_primitive(value=self.value.__eq__(other))

    def __float__(self) -> SyPrimitiveRet:
        return PrimitiveFactory.generate_primitive(value=self.value.__float__())

    def __floor__(self) -> SyPrimitiveRet:
        return PrimitiveFactory.generate_primitive(value=self.value.__floor__())

    def __floordiv__(self, other: Any) -> SyPrimitiveRet:
        other = dispatch_other(other)
        return PrimitiveFactory.generate_primitive(value=self.value.__floordiv__(other))

    def __ge__(self, other: Any) -> SyPrimitiveRet:
        other = dispatch_other(other)
        return PrimitiveFactory.generate_primitive(value=self.value.__ge__(other))

    def __gt__(self, other: Any) -> SyPrimitiveRet:
        other = dispatch_other(other)
        return PrimitiveFactory.generate_primitive(value=self.value.__gt__(other))

    def __hash__(self) -> int:
        return PrimitiveFactory.generate_primitive(value=self.value.__hash__())

    def __invert__(self) -> SyPrimitiveRet:
        return PrimitiveFactory.generate_primitive(value=self.value.__invert__())

    def __int__(self) -> int:
        return int(self.value)

    def __le__(self, other: Any) -> SyPrimitiveRet:
        other = dispatch_other(other)
        return PrimitiveFactory.generate_primitive(value=self.value.__le__(other))

    def __lshift__(self, other: Any) -> SyPrimitiveRet:
        other = dispatch_other(other)
        return PrimitiveFactory.generate_primitive(value=self.value.__lshift__(other))

    def __lt__(self, other: Any) -> SyPrimitiveRet:
        other = dispatch_other(other)
        return PrimitiveFactory.generate_primitive(value=self.value.__lt__(other))

    def __mod__(self, other: Any) -> SyPrimitiveRet:
        other = dispatch_other(other)
        return PrimitiveFactory.generate_primitive(value=self.value.__mod__(other))

    def __mul__(self, other: Any) -> SyPrimitiveRet:
        other = dispatch_other(other)
        return PrimitiveFactory.generate_primitive(value=self.value.__mul__(other))

    def __ne__(self, other: Any) -> SyPrimitiveRet:
        other = dispatch_other(other)
        return PrimitiveFactory.generate_primitive(value=self.value.__ne__(other))

    def __neg__(self) -> SyPrimitiveRet:
        return PrimitiveFactory.generate_primitive(value=self.value.__neg__())

    def __or__(self, other: Any) -> SyPrimitiveRet:
        other = dispatch_other(other)
        return PrimitiveFactory.generate_primitive(value=self.value.__or__(other))

    def __pos__(self) -> SyPrimitiveRet:
        return PrimitiveFactory.generate_primitive(value=self.value.__pos__())

    def __pow__(self, other: Any, modulo: Optional[Any] = None) -> SyPrimitiveRet:
        if modulo:
            PrimitiveFactory.generate_primitive(value=super().__pow__(other, modulo))
        return PrimitiveFactory.generate_primitive(value=super().__pow__(other))

    def __radd__(self, other: Any) -> SyPrimitiveRet:
        other = dispatch_other(other)
        return PrimitiveFactory.generate_primitive(value=self.value.__radd__(other))

    def __rand__(self, other: Any) -> SyPrimitiveRet:
        other = dispatch_other(other)
        return PrimitiveFactory.generate_primitive(value=self.value.__rand__(other))

    def __repr__(self) -> str:
        return bool(self.value).__repr__()

    def __rdivmod__(self, other: Any) -> SyPrimitiveRet:
        other = dispatch_other(other)
        tpl = self.value.__rdivmod__(other)

        return PrimitiveFactory.generate_primitive(value=tpl)

    def __rfloordiv__(self, other: Any) -> SyPrimitiveRet:
        other = dispatch_other(other)
        return PrimitiveFactory.generate_primitive(
            value=self.value.__rfloordiv__(other)
        )

    def __rlshift__(self, other: Any) -> SyPrimitiveRet:
        other = dispatch_other(other)
        return PrimitiveFactory.generate_primitive(value=self.value.__rlshift__(other))

    def __rmod__(self, other: Any) -> SyPrimitiveRet:
        other = dispatch_other(other)
        return PrimitiveFactory.generate_primitive(value=self.value.__rmod__(other))

    def __rmul__(self, other: Any) -> SyPrimitiveRet:
        other = dispatch_other(other)
        return PrimitiveFactory.generate_primitive(value=self.value.__rmul__(other))

    def __ror__(self, other: Any) -> SyPrimitiveRet:
        other = dispatch_other(other)
        return PrimitiveFactory.generate_primitive(value=self.value.__ror__(other))

    def __round__(self, ndigits: Any = 0) -> SyPrimitiveRet:
        dispatch_ndigits = dispatch_other(ndigits)
        return PrimitiveFactory.generate_primitive(
            value=self.value.__round__(dispatch_ndigits)
        )

    def __rpow__(self, other: Any, modulo: Optional[Any] = None) -> SyPrimitiveRet:
        other = dispatch_other(other)

        if modulo:
            return PrimitiveFactory.generate_primitive(
                value=self.value.__rpow__(other, modulo)
            )
        return PrimitiveFactory.generate_primitive(value=self.value.__rpow__(other))

    def __rrshift__(self, other: Any) -> SyPrimitiveRet:
        other = dispatch_other(other)
        return PrimitiveFactory.generate_primitive(value=self.value.__rrshift__(other))

    def __rshift__(self, other: Any) -> SyPrimitiveRet:
        other = dispatch_other(other)
        return PrimitiveFactory.generate_primitive(value=self.value.__rshift__(other))

    def __rsub__(self, other: Any) -> SyPrimitiveRet:
        other = dispatch_other(other)
        return PrimitiveFactory.generate_primitive(value=self.value.__rsub__(other))

    def __rtruediv__(self, other: Any) -> SyPrimitiveRet:
        other = dispatch_other(other)
        return PrimitiveFactory.generate_primitive(value=self.value.__rtruediv__(other))

    def __rxor__(self, other: Any) -> SyPrimitiveRet:
        other = dispatch_other(other)
        return PrimitiveFactory.generate_primitive(value=self.value.__rxor__(other))

    def __str__(self) -> str:
        return bool(self.value).__str__()

    def __sub__(self, other: Any) -> SyPrimitiveRet:
        other = dispatch_other(other)
        return PrimitiveFactory.generate_primitive(value=self.value.__sub__(other))

    def __truediv__(self, other: Any) -> SyPrimitiveRet:
        other = dispatch_other(other)
        return PrimitiveFactory.generate_primitive(value=self.value.__truediv__(other))

    def __trunc__(self) -> SyPrimitiveRet:
        return PrimitiveFactory.generate_primitive(value=self.value.__trunc__())

    def __xor__(self, other: Any) -> SyPrimitiveRet:
        other = dispatch_other(other)
        return PrimitiveFactory.generate_primitive(value=self.value.__xor__(other))

    def as_integer_ratio(self) -> SyPrimitiveRet:
        res = self.value.as_integer_ratio()
        return PrimitiveFactory.generate_primitive(value=res)

    def bit_length(self) -> SyPrimitiveRet:
        return PrimitiveFactory.generate_primitive(value=self.value.bit_length())

    def conjugate(self) -> SyPrimitiveRet:
        return PrimitiveFactory.generate_primitive(value=self.value.conjugate())

    def denominator(self) -> SyPrimitiveRet:
        return PrimitiveFactory.generate_primitive(value=self.value.denominator)

    # TODO: add support for properties on these 4 functions

    def imag(self) -> SyPrimitiveRet:
        return PrimitiveFactory.generate_primitive(value=self.value.imag)

    def numerator(self) -> SyPrimitiveRet:
        return PrimitiveFactory.generate_primitive(value=self.value.numerator)

    def real(self) -> SyPrimitiveRet:
        return PrimitiveFactory.generate_primitive(value=self.value.real)

    def _object2proto(self) -> Bool_PB:
        return Bool_PB(id=sy.serialize(obj=self.id), data=self)

    @staticmethod
    def _proto2object(proto: Bool_PB) -> "Bool":
        return Bool(id=sy.deserialize(blob=proto.id), value=proto.data)

    @staticmethod
    def get_protobuf_schema() -> GeneratedProtocolMessageType:
        return Bool_PB
