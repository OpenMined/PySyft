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
from ...proto.lib.python.int_pb2 import Int as Int_PB
from ...util import aggressive_set_attr
from .primitive_factory import PrimitiveFactory
from .primitive_interface import PyPrimitive
from .util import SyPrimitiveRet


class Int(int, PyPrimitive):
    @syft_decorator(typechecking=True, prohibit_args=False)
    def __new__(
        cls, value: Any = None, base: Any = 10, id: Optional[UID] = None
    ) -> "Int":
        if value is None:
            value = 0

        if isinstance(value, str):
            return int.__new__(cls, value, base)  # type: ignore

        return int.__new__(cls, value)  # type: ignore

    @syft_decorator(typechecking=True, prohibit_args=False)
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

    @syft_decorator(typechecking=True, prohibit_args=True)
    def upcast(self) -> int:
        return int(self)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __add__(self, other: Any) -> SyPrimitiveRet:
        res = super().__add__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __int__(self) -> int:
        res = super().__int__()
        return res

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __invert__(self) -> SyPrimitiveRet:
        res = super().__invert__()
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __abs__(self) -> SyPrimitiveRet:
        res = super().__abs__()
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __bool__(self) -> bool:
        return super().__bool__()

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __divmod__(self, other: Any) -> SyPrimitiveRet:
        res = super().__divmod__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __rdivmod__(self, other: Any) -> SyPrimitiveRet:
        res = super().__rdivmod__(other)
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
    def __rtruediv__(self, other: Any) -> SyPrimitiveRet:
        res = super().__rtruediv__(other)
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
    def __ceil__(self) -> SyPrimitiveRet:
        res = super().__ceil__()
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __eq__(self, other: Any) -> SyPrimitiveRet:
        res = super().__eq__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __float__(self) -> SyPrimitiveRet:
        res = super().__float__()
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __floor__(self) -> SyPrimitiveRet:
        res = super().__floor__()
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __floordiv__(self, other: Any) -> SyPrimitiveRet:
        res = super().__floordiv__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __rfloordiv__(self, other: Any) -> SyPrimitiveRet:
        res = super().__rfloordiv__(other)
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
    def __pow__(self, other: Any) -> SyPrimitiveRet:
        res = super().__pow__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __rpow__(self, other: Any) -> SyPrimitiveRet:
        res = super().__rpow__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __lshift__(self, other: Any) -> SyPrimitiveRet:
        res = super(Int, self).__lshift__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __rlshift__(self, other: Any) -> SyPrimitiveRet:
        res = super().__rlshift__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __round__(self) -> SyPrimitiveRet:
        res = super().__round__()
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __rshift__(self, other: Any) -> SyPrimitiveRet:
        res = super(Int, self).__rshift__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __rrshift__(self, other: Any) -> SyPrimitiveRet:
        res = super().__rrshift__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __and__(self, other: Any) -> SyPrimitiveRet:
        res = super().__and__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __rand__(self, other: Any) -> SyPrimitiveRet:
        res = super().__rand__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __xor__(self, other: Any) -> SyPrimitiveRet:
        res = super(Int, self).__xor__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __rxor__(self, other: Any) -> SyPrimitiveRet:
        res = super().__rxor__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __or__(self, other: Any) -> SyPrimitiveRet:
        res = super().__or__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __ror__(self, other: Any) -> SyPrimitiveRet:
        res = super().__ror__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __ge__(self, other: Any) -> SyPrimitiveRet:
        res = super().__ge__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __lt__(self, other: Any) -> SyPrimitiveRet:
        res = super().__lt__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __le__(self, other: Any) -> SyPrimitiveRet:
        res = super().__le__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __gt__(self, other: Any) -> SyPrimitiveRet:
        res = super().__gt__(other)
        return PrimitiveFactory.generate_primitive(value=res)

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

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __ne__(self, other: Any) -> SyPrimitiveRet:
        res = super().__ne__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __hash__(self) -> SyPrimitiveRet:
        return PrimitiveFactory.generate_primitive(value=super().__hash__())

    # add tests
    @syft_decorator(typechecking=True, prohibit_args=False)
    def __neg__(self) -> SyPrimitiveRet:
        return PrimitiveFactory.generate_primitive(value=super().__neg__())

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __pos__(self) -> SyPrimitiveRet:
        return PrimitiveFactory.generate_primitive(value=super().__pos__())

    @syft_decorator(typechecking=True)
    def _object2proto(self) -> Int_PB:
        int_pb = Int_PB()
        int_pb.data = self
        int_pb.id.CopyFrom(serialize(obj=self.id))
        return int_pb

    @staticmethod
    @syft_decorator(typechecking=True)
    def _proto2object(proto: Int_PB) -> "Int":
        int_id: UID = deserialize(blob=proto.id)

        de_int = Int(value=proto.data)
        de_int._id = int_id  # can't use uid=int_id for some reason

        return de_int

    @staticmethod
    def get_protobuf_schema() -> GeneratedProtocolMessageType:
        return Int_PB

    @syft_decorator(typechecking=True, prohibit_args=False)
    def as_integer_ratio(self) -> SyPrimitiveRet:
        tpl = super().as_integer_ratio()
        return PrimitiveFactory.generate_primitive(value=tpl)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def bit_length(self) -> SyPrimitiveRet:
        res = super().bit_length()
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def denominator(self) -> SyPrimitiveRet:
        res = super().denominator
        return PrimitiveFactory.generate_primitive(value=res)

    @staticmethod
    @syft_decorator(typechecking=True, prohibit_args=False)
    def from_bytes(bytes: Any, byteorder: str, *, signed: Any = True) -> SyPrimitiveRet:
        res = int.from_bytes(bytes, byteorder, signed=signed)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def imag(self) -> SyPrimitiveRet:
        res = super().imag
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def numerator(self) -> int:
        res = super().numerator
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def real(self) -> int:
        res = super().real
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def conjugate(self) -> SyPrimitiveRet:
        res = super().conjugate()
        return PrimitiveFactory.generate_primitive(value=res)

    # method signature override
    def to_bytes(
        self,
        length: Optional[int] = None,
        byteorder: Optional[str] = None,
        signed: Optional[bool] = True,
    ) -> bytes:
        if length is not None and byteorder is not None and signed is not None:
            return super().to_bytes(length=length, byteorder=byteorder, signed=signed)
        else:
            # get our serializable method
            _to_bytes = getattr(self, "_to_bytes", None)
            if _to_bytes is not None:
                return _to_bytes.__call__()
        return b""


class IntWrapper(StorableObject):
    def __init__(self, value: object):
        super().__init__(
            data=value,
            id=getattr(value, "id", UID()),
            tags=getattr(value, "tags", []),
            description=getattr(value, "description", ""),
        )
        self.value = value

    def _data_object2proto(self) -> Int_PB:
        _object2proto = getattr(self.data, "_object2proto", None)
        if _object2proto:
            return _object2proto()

    @staticmethod
    def _data_proto2object(proto: Int_PB) -> "IntWrapper":
        return Int._proto2object(proto=proto)

    @staticmethod
    def get_data_protobuf_schema() -> GeneratedProtocolMessageType:
        return Int_PB

    @staticmethod
    def get_wrapped_type() -> type:
        return Int

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


aggressive_set_attr(obj=Int, name="serializable_wrapper_type", attr=IntWrapper)
