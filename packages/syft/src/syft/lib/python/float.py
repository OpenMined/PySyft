# stdlib
from typing import Any
from typing import Optional

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType

# relative
from ...core.common.serde.serializable import serializable
from ...proto.lib.python.float_pb2 import Float as Float_PB
from .primitive_factory import PrimitiveFactory
from .primitive_interface import PyPrimitive
from .types import SyPrimitiveRet


@serializable()
class Float(float, PyPrimitive):
    def __new__(cls, value: Any = None) -> "Float":
        if value is None:
            value = 0.0
        return float.__new__(cls, value)

    def __init__(self, value: Any = None):
        if value is None:
            value = 0.0

        float.__init__(value)

    def upcast(self) -> float:
        return float(self)

    def __eq__(self, other: Any) -> SyPrimitiveRet:
        result = super().__eq__(other)
        return PrimitiveFactory.generate_primitive(value=result)

    def __ge__(self, other: Any) -> SyPrimitiveRet:
        result = super().__ge__(other)
        return PrimitiveFactory.generate_primitive(value=result)

    def __lt__(self, other: Any) -> SyPrimitiveRet:
        result = super().__lt__(other)
        return PrimitiveFactory.generate_primitive(value=result)

    def __le__(self, other: Any) -> SyPrimitiveRet:
        result = super().__le__(other)
        return PrimitiveFactory.generate_primitive(value=result)

    def __gt__(self, other: Any) -> SyPrimitiveRet:
        result = super().__gt__(other)
        return PrimitiveFactory.generate_primitive(value=result)

    def __add__(self, other: Any) -> SyPrimitiveRet:
        res = super().__add__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    def __abs__(self) -> SyPrimitiveRet:
        res = super().__abs__()
        return PrimitiveFactory.generate_primitive(value=res)

    def __bool__(self) -> SyPrimitiveRet:
        res = super().__bool__()
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

    def __mul__(self, other: Any) -> SyPrimitiveRet:
        res = super().__mul__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    def __rmul__(self, other: Any) -> SyPrimitiveRet:
        res = super().__rmul__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    def __divmod__(self, other: Any) -> SyPrimitiveRet:
        value = super().__divmod__(other)
        return PrimitiveFactory.generate_primitive(value=value)

    def __neg__(self) -> SyPrimitiveRet:
        res = super().__neg__()
        return PrimitiveFactory.generate_primitive(value=res)

    def __ne__(self, other: Any) -> SyPrimitiveRet:
        res = super().__ne__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    def __floordiv__(self, other: Any) -> SyPrimitiveRet:
        res = super().__floordiv__(other)
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

    def __rdivmod__(self, other: Any) -> SyPrimitiveRet:
        res = super().__rdivmod__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    def __rfloordiv__(self, other: Any) -> SyPrimitiveRet:
        res = super().__rfloordiv__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    def __round__(self, *args: Any, **kwargs: Any) -> SyPrimitiveRet:  # type: ignore
        res = super().__round__(*args, **kwargs)
        return PrimitiveFactory.generate_primitive(value=res)

    def __rtruediv__(self, other: Any) -> SyPrimitiveRet:
        res = super().__rtruediv__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    def __trunc__(self) -> SyPrimitiveRet:
        res = super().__trunc__()
        return PrimitiveFactory.generate_primitive(value=res)

    def as_integer_ratio(self) -> SyPrimitiveRet:
        tpl = super().as_integer_ratio()
        return PrimitiveFactory.generate_primitive(value=tpl)

    def is_integer(self) -> SyPrimitiveRet:
        res = super().is_integer()
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

    def __iadd__(self, other: Any) -> SyPrimitiveRet:
        return PrimitiveFactory.generate_primitive(value=super().__add__(other))

    def __isub__(self, other: Any) -> SyPrimitiveRet:
        return PrimitiveFactory.generate_primitive(value=super().__sub__(other))

    def __imul__(self, other: Any) -> SyPrimitiveRet:
        return PrimitiveFactory.generate_primitive(value=super().__mul__(other))

    def __ifloordiv__(self, other: Any) -> SyPrimitiveRet:
        return PrimitiveFactory.generate_primitive(value=super().__floordiv__(other))

    def __itruediv__(self, other: Any) -> SyPrimitiveRet:
        return PrimitiveFactory.generate_primitive(value=super().__truediv__(other))

    def __imod__(self, other: Any) -> SyPrimitiveRet:
        return PrimitiveFactory.generate_primitive(value=super().__mod__(other))

    def __ipow__(self, other: Any, modulo: Optional[Any] = None) -> SyPrimitiveRet:
        if modulo:
            return PrimitiveFactory.generate_primitive(
                value=super().__pow__(other, modulo)
            )
        return PrimitiveFactory.generate_primitive(value=super().__pow__(other))

    @property
    def real(self) -> SyPrimitiveRet:
        return PrimitiveFactory.generate_primitive(value=super().real)

    @property
    def imag(self) -> SyPrimitiveRet:
        return PrimitiveFactory.generate_primitive(value=super().imag)

    def conjugate(self) -> SyPrimitiveRet:
        return PrimitiveFactory.generate_primitive(value=super().conjugate())

    def hex(self) -> SyPrimitiveRet:
        return PrimitiveFactory.generate_primitive(value=self.upcast().hex())

    def __hash__(self) -> int:
        return super().__hash__()

    def _object2proto(self) -> Float_PB:
        return Float_PB(
            data=self,
        )

    @staticmethod
    def _proto2object(proto: Float_PB) -> "Float":
        return Float(value=proto.data)

    @staticmethod
    def get_protobuf_schema() -> GeneratedProtocolMessageType:
        return Float_PB
