# stdlib
from typing import Any
from typing import List
from typing import Optional

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType

# syft relative
from ... import deserialize
from ... import serialize
from ...core.common import UID
from ...core.store.storeable_object import StorableObject
from ...decorators import syft_decorator
from ...proto.lib.python.float_pb2 import Float as Float_PB
from ...util import aggressive_set_attr
from .primitive_factory import PrimitiveFactory
from .primitive_interface import PyPrimitive
from .util import SyPrimitiveRet


class Float(float, PyPrimitive):
    @syft_decorator(typechecking=True, prohibit_args=False)
    def __new__(cls, value: Any = None, id: Optional[UID] = None) -> "Float":
        if value is None:
            value = 0.0
        return float.__new__(cls, value)  # type: ignore

    @syft_decorator(typechecking=True, prohibit_args=False)
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

    @syft_decorator(typechecking=True, prohibit_args=True)
    def upcast(self) -> float:
        return float(self)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __eq__(self, other: Any) -> SyPrimitiveRet:
        result = super().__eq__(other)
        return PrimitiveFactory.generate_primitive(value=result)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __ge__(self, other: Any) -> SyPrimitiveRet:
        result = super().__ge__(other)
        return PrimitiveFactory.generate_primitive(value=result)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __lt__(self, other: Any) -> SyPrimitiveRet:
        result = super().__lt__(other)
        return PrimitiveFactory.generate_primitive(value=result)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __le__(self, other: Any) -> SyPrimitiveRet:
        result = super().__le__(other)
        return PrimitiveFactory.generate_primitive(value=result)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __gt__(self, other: Any) -> SyPrimitiveRet:
        result = super().__gt__(other)
        return PrimitiveFactory.generate_primitive(value=result)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __add__(self, other: Any) -> SyPrimitiveRet:
        res = super().__add__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __abs__(self) -> SyPrimitiveRet:
        res = super().__abs__()
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __bool__(self) -> SyPrimitiveRet:
        res = super().__bool__()
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __radd__(self, other: Any) -> SyPrimitiveRet:
        res = super().__radd__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __sub__(self, other: Any) -> SyPrimitiveRet:
        res = super().__sub__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __rsub__(self, other: Any) -> SyPrimitiveRet:
        res = super().__rsub__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __mul__(self, other: Any) -> SyPrimitiveRet:
        res = super().__mul__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __rmul__(self, other: Any) -> SyPrimitiveRet:
        res = super().__rmul__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __divmod__(self, other: Any) -> SyPrimitiveRet:
        value = super().__divmod__(other)
        return PrimitiveFactory.generate_primitive(value=value)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __neg__(self) -> SyPrimitiveRet:
        res = super().__neg__()
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __ne__(self, other: Any) -> SyPrimitiveRet:
        res = super().__ne__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __floordiv__(self, other: Any) -> SyPrimitiveRet:
        res = super().__floordiv__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __truediv__(self, other: Any) -> SyPrimitiveRet:
        res = super().__truediv__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __mod__(self, other: Any) -> SyPrimitiveRet:
        res = super().__mod__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __rmod__(self, other: Any) -> SyPrimitiveRet:
        res = super().__rmod__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __rdivmod__(self, other: Any) -> SyPrimitiveRet:
        res = super().__rdivmod__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __rfloordiv__(self, other: Any) -> SyPrimitiveRet:
        res = super().__rfloordiv__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __round__(self, n: Optional[int] = None) -> SyPrimitiveRet:
        res = super().__round__(n)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __rtruediv__(self, other: Any) -> SyPrimitiveRet:
        res = super().__rtruediv__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __trunc__(self) -> SyPrimitiveRet:
        res = super().__trunc__()
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def as_integer_ratio(self) -> SyPrimitiveRet:
        tpl = super().as_integer_ratio()
        return PrimitiveFactory.generate_primitive(value=tpl)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def is_integer(self) -> SyPrimitiveRet:
        res = super().is_integer()
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __pow__(self, other: Any) -> SyPrimitiveRet:
        return PrimitiveFactory.generate_primitive(value=super().__pow__(other))

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __rpow__(self, other: Any) -> SyPrimitiveRet:
        return PrimitiveFactory.generate_primitive(value=super().__rpow__(other))

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __iadd__(self, other: Any) -> SyPrimitiveRet:
        return PrimitiveFactory.generate_primitive(
            value=super().__add__(other), id=self.id
        )

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __isub__(self, other: Any) -> SyPrimitiveRet:
        return PrimitiveFactory.generate_primitive(
            value=super().__sub__(other), id=self.id
        )

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __imul__(self, other: Any) -> SyPrimitiveRet:
        return PrimitiveFactory.generate_primitive(
            value=super().__mul__(other), id=self.id
        )

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __ifloordiv__(self, other: Any) -> SyPrimitiveRet:
        return PrimitiveFactory.generate_primitive(
            value=super().__floordiv__(other), id=self.id
        )

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __itruediv__(self, other: Any) -> SyPrimitiveRet:
        return PrimitiveFactory.generate_primitive(
            value=super().__truediv__(other), id=self.id
        )

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __imod__(self, other: Any) -> SyPrimitiveRet:
        return PrimitiveFactory.generate_primitive(
            value=super().__mod__(other), id=self.id
        )

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __ipow__(self, other: Any) -> SyPrimitiveRet:
        return PrimitiveFactory.generate_primitive(
            value=super().__pow__(other), id=self.id
        )

    @property
    def real(self) -> SyPrimitiveRet:
        return PrimitiveFactory.generate_primitive(value=super().real)

    @property
    def imag(self) -> SyPrimitiveRet:
        return PrimitiveFactory.generate_primitive(value=super().imag)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def conjugate(self) -> SyPrimitiveRet:
        return PrimitiveFactory.generate_primitive(value=super().conjugate())

    @syft_decorator(typechecking=True, prohibit_args=False)
    def hex(self) -> SyPrimitiveRet:
        return PrimitiveFactory.generate_primitive(value=self.upcast().hex())

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __hash__(self) -> int:
        return super().__hash__()

    @syft_decorator(typechecking=True)
    def _object2proto(self) -> Float_PB:
        return Float_PB(
            id=serialize(obj=self.id),
            data=self,
        )

    @staticmethod
    def _proto2object(proto: Float_PB) -> "Float":
        return Float(value=proto.data, id=deserialize(blob=proto.id))

    @staticmethod
    def get_protobuf_schema() -> GeneratedProtocolMessageType:
        return Float_PB


class FloatWrapper(StorableObject):
    def __init__(self, value: object):
        super().__init__(
            data=value,
            id=getattr(value, "id", UID()),
            tags=getattr(value, "tags", []),
            description=getattr(value, "description", ""),
        )
        self.value = value

    def _data_object2proto(self) -> Float_PB:
        _object2proto = getattr(self.data, "_object2proto", None)
        if _object2proto:
            return _object2proto()

    @staticmethod
    def _data_proto2object(proto: Float_PB) -> "FloatWrapper":
        return Float._proto2object(proto)

    @staticmethod
    def get_data_protobuf_schema() -> GeneratedProtocolMessageType:
        return Float_PB

    @staticmethod
    def get_wrapped_type() -> type:
        return Float

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


aggressive_set_attr(obj=Float, name="serializable_wrapper_type", attr=FloatWrapper)
