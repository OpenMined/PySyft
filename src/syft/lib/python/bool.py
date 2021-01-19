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
from ...proto.lib.python.bool_pb2 import Bool as Bool_PB
from ...util import aggressive_set_attr
from .primitive_factory import PrimitiveFactory
from .primitive_interface import PyPrimitive
from .util import SyPrimitiveRet


def dispatch_other(obj: Any) -> bool:
    if isinstance(obj, Bool):
        return obj.value
    return obj


class Bool(int, PyPrimitive):
    @syft_decorator(typechecking=True, prohibit_args=False)
    def __new__(cls, value: Any = None, id: Optional[UID] = None) -> "Bool":
        value = bool(value)
        obj = int.__new__(cls, value)  # type: ignore
        return obj

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __init__(self, value: Any = None, id: Optional[UID] = None):
        self.value: bool = bool(value)
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
    def upcast(self) -> bool:
        return bool(self)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __abs__(self) -> SyPrimitiveRet:
        return PrimitiveFactory.generate_primitive(value=self.value.__abs__())

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __add__(self, other: Any) -> SyPrimitiveRet:
        other = dispatch_other(other)
        return PrimitiveFactory.generate_primitive(value=self.value.__add__(other))

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __and__(self, other: Any) -> SyPrimitiveRet:
        other = dispatch_other(other)
        return PrimitiveFactory.generate_primitive(value=self.value.__and__(other))

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __bool__(self) -> bool:
        return bool(self.value)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __ceil__(self) -> SyPrimitiveRet:
        return PrimitiveFactory.generate_primitive(value=self.value.__ceil__())

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __divmod__(self, other: Any) -> SyPrimitiveRet:
        other = dispatch_other(other)
        tpl = self.value.__divmod__(other)
        return PrimitiveFactory.generate_primitive(value=tpl)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __eq__(self, other: Any) -> SyPrimitiveRet:
        other = dispatch_other(other)
        return PrimitiveFactory.generate_primitive(value=self.value.__eq__(other))

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __float__(self) -> SyPrimitiveRet:
        return PrimitiveFactory.generate_primitive(value=self.value.__float__())

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __floor__(self) -> SyPrimitiveRet:
        return PrimitiveFactory.generate_primitive(value=self.value.__floor__())

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __floordiv__(self, other: Any) -> SyPrimitiveRet:
        other = dispatch_other(other)
        return PrimitiveFactory.generate_primitive(value=self.value.__floordiv__(other))

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __ge__(self, other: Any) -> SyPrimitiveRet:
        other = dispatch_other(other)
        return PrimitiveFactory.generate_primitive(value=self.value.__ge__(other))

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __gt__(self, other: Any) -> SyPrimitiveRet:
        other = dispatch_other(other)
        return PrimitiveFactory.generate_primitive(value=self.value.__gt__(other))

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __hash__(self) -> int:
        return PrimitiveFactory.generate_primitive(value=self.value.__hash__())

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __invert__(self) -> SyPrimitiveRet:
        return PrimitiveFactory.generate_primitive(value=self.value.__invert__())

    def __int__(self) -> int:
        return int(self.value)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __le__(self, other: Any) -> SyPrimitiveRet:
        other = dispatch_other(other)
        return PrimitiveFactory.generate_primitive(value=self.value.__le__(other))

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __lshift__(self, other: Any) -> SyPrimitiveRet:
        other = dispatch_other(other)
        return PrimitiveFactory.generate_primitive(value=self.value.__lshift__(other))

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __lt__(self, other: Any) -> SyPrimitiveRet:
        other = dispatch_other(other)
        return PrimitiveFactory.generate_primitive(value=self.value.__lt__(other))

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __mod__(self, other: Any) -> SyPrimitiveRet:
        other = dispatch_other(other)
        return PrimitiveFactory.generate_primitive(value=self.value.__mod__(other))

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __mul__(self, other: Any) -> SyPrimitiveRet:
        other = dispatch_other(other)
        return PrimitiveFactory.generate_primitive(value=self.value.__mul__(other))

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __ne__(self, other: Any) -> SyPrimitiveRet:
        other = dispatch_other(other)
        return PrimitiveFactory.generate_primitive(value=self.value.__ne__(other))

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __neg__(self) -> SyPrimitiveRet:
        return PrimitiveFactory.generate_primitive(value=self.value.__neg__())

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __or__(self, other: Any) -> SyPrimitiveRet:
        other = dispatch_other(other)
        return PrimitiveFactory.generate_primitive(value=self.value.__or__(other))

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __pos__(self) -> SyPrimitiveRet:
        return PrimitiveFactory.generate_primitive(value=self.value.__pos__())

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __pow__(self, other: Any) -> SyPrimitiveRet:
        other = dispatch_other(other)
        return PrimitiveFactory.generate_primitive(value=self.value.__pow__(other))

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __radd__(self, other: Any) -> SyPrimitiveRet:
        other = dispatch_other(other)
        return PrimitiveFactory.generate_primitive(value=self.value.__radd__(other))

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __rand__(self, other: Any) -> SyPrimitiveRet:
        other = dispatch_other(other)
        return PrimitiveFactory.generate_primitive(value=self.value.__rand__(other))

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __repr__(self) -> str:
        return bool(self.value).__repr__()

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __rdivmod__(self, other: Any) -> SyPrimitiveRet:
        other = dispatch_other(other)
        tpl = self.value.__rdivmod__(other)

        return PrimitiveFactory.generate_primitive(value=tpl)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __rfloordiv__(self, other: Any) -> SyPrimitiveRet:
        other = dispatch_other(other)
        return PrimitiveFactory.generate_primitive(
            value=self.value.__rfloordiv__(other)
        )

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __rlshift__(self, other: Any) -> SyPrimitiveRet:
        other = dispatch_other(other)
        return PrimitiveFactory.generate_primitive(value=self.value.__rlshift__(other))

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __rmod__(self, other: Any) -> SyPrimitiveRet:
        other = dispatch_other(other)
        return PrimitiveFactory.generate_primitive(value=self.value.__rmod__(other))

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __rmul__(self, other: Any) -> SyPrimitiveRet:
        other = dispatch_other(other)
        return PrimitiveFactory.generate_primitive(value=self.value.__rmul__(other))

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __ror__(self, other: Any) -> SyPrimitiveRet:
        other = dispatch_other(other)
        return PrimitiveFactory.generate_primitive(value=self.value.__ror__(other))

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __round__(self) -> SyPrimitiveRet:
        return PrimitiveFactory.generate_primitive(value=self.value.__round__())

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __rpow__(self, other: Any) -> SyPrimitiveRet:
        other = dispatch_other(other)
        return PrimitiveFactory.generate_primitive(value=self.value.__rpow__(other))

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __rrshift__(self, other: Any) -> SyPrimitiveRet:
        other = dispatch_other(other)
        return PrimitiveFactory.generate_primitive(value=self.value.__rrshift__(other))

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __rshift__(self, other: Any) -> SyPrimitiveRet:
        other = dispatch_other(other)
        return PrimitiveFactory.generate_primitive(value=self.value.__rshift__(other))

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __rsub__(self, other: Any) -> SyPrimitiveRet:
        other = dispatch_other(other)
        return PrimitiveFactory.generate_primitive(value=self.value.__rsub__(other))

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __rtruediv__(self, other: Any) -> SyPrimitiveRet:
        other = dispatch_other(other)
        return PrimitiveFactory.generate_primitive(value=self.value.__rtruediv__(other))

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __rxor__(self, other: Any) -> SyPrimitiveRet:
        other = dispatch_other(other)
        return PrimitiveFactory.generate_primitive(value=self.value.__rxor__(other))

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __str__(self) -> str:
        return bool(self.value).__str__()

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __sub__(self, other: Any) -> SyPrimitiveRet:
        other = dispatch_other(other)
        return PrimitiveFactory.generate_primitive(value=self.value.__sub__(other))

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __truediv__(self, other: Any) -> SyPrimitiveRet:
        other = dispatch_other(other)
        return PrimitiveFactory.generate_primitive(value=self.value.__truediv__(other))

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __trunc__(self) -> SyPrimitiveRet:
        return PrimitiveFactory.generate_primitive(value=self.value.__trunc__())

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __xor__(self, other: Any) -> SyPrimitiveRet:
        other = dispatch_other(other)
        return PrimitiveFactory.generate_primitive(value=self.value.__xor__(other))

    @syft_decorator(typechecking=True, prohibit_args=False)
    def as_integer_ratio(self) -> SyPrimitiveRet:
        res = self.value.as_integer_ratio()
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def bit_length(self) -> SyPrimitiveRet:
        return PrimitiveFactory.generate_primitive(value=self.value.bit_length())

    @syft_decorator(typechecking=True, prohibit_args=False)
    def conjugate(self) -> SyPrimitiveRet:
        return PrimitiveFactory.generate_primitive(value=self.value.conjugate())

    @syft_decorator(typechecking=True, prohibit_args=False)
    def denominator(self) -> SyPrimitiveRet:
        return PrimitiveFactory.generate_primitive(value=self.value.denominator)

    # TODO: add support for properties on these 4 functions
    @syft_decorator(typechecking=True, prohibit_args=False)
    def imag(self) -> SyPrimitiveRet:
        return PrimitiveFactory.generate_primitive(value=self.value.imag)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def numerator(self) -> SyPrimitiveRet:
        return PrimitiveFactory.generate_primitive(value=self.value.numerator)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def real(self) -> SyPrimitiveRet:
        return PrimitiveFactory.generate_primitive(value=self.value.real)

    @syft_decorator(typechecking=True)
    def _object2proto(self) -> Bool_PB:
        return Bool_PB(id=serialize(obj=self.id), data=self)

    @staticmethod
    @syft_decorator(typechecking=True)
    def _proto2object(proto: Bool_PB) -> "Bool":
        return Bool(id=deserialize(blob=proto.id), value=proto.data)

    @staticmethod
    def get_protobuf_schema() -> GeneratedProtocolMessageType:
        return Bool_PB


class BoolWrapper(StorableObject):
    def __init__(self, value: object):
        super().__init__(
            data=value,
            id=getattr(value, "id", UID()),
            tags=getattr(value, "tags", []),
            description=getattr(value, "description", ""),
        )
        self.value = value

    def _data_object2proto(self) -> Bool_PB:
        _object2proto = getattr(self.data, "_object2proto", None)
        if _object2proto:
            return _object2proto()

    @staticmethod
    def _data_proto2object(proto: Bool_PB) -> "BoolWrapper":
        return Bool._proto2object(proto=proto)

    @staticmethod
    def get_data_protobuf_schema() -> GeneratedProtocolMessageType:
        return Bool_PB

    @staticmethod
    def get_wrapped_type() -> type:
        return Bool

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


aggressive_set_attr(obj=Bool, name="serializable_wrapper_type", attr=BoolWrapper)
