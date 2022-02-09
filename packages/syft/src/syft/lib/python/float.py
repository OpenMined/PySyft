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
from ...proto.lib.python.float_pb2 import Float as Float_PB
from .primitive_factory import PrimitiveFactory
from .primitive_interface import PyPrimitive
from .types import SyPrimitiveRet


@serializable()
class Float(float, PyPrimitive):
    def __new__(cls, value: Any = None, id: Optional[UID] = None) -> "Float":
        if value is None:
            value = 0.0
        return float.__new__(cls, value)

    def __init__(self, value: Any = None, id: Optional[UID] = None):
        if value is None:
            value = 0.0

        float.__init__(value)

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
            return PrimitiveFactory.generate_primitive(
                value=super().__pow__(other, modulo), id=self.id
            )
        return PrimitiveFactory.generate_primitive(
            value=super().__pow__(other), id=self.id
        )

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
            id=sy.serialize(obj=self.id),
            data=self,
        )

    @staticmethod
    def _proto2object(proto: Float_PB) -> "Float":
        return Float(value=proto.data, id=sy.deserialize(blob=proto.id))

    @staticmethod
    def get_protobuf_schema() -> GeneratedProtocolMessageType:
        return Float_PB
