from typing import Any, Tuple
from google.protobuf.reflection import GeneratedProtocolMessageType

from ...decorators import syft_decorator
from .primitive_interface import PyPrimitive
from .primitive_factory import PrimitiveFactory
from ...core.common import UID
from ...proto.lib.python.int_pb2 import Int as Int_PB
from ... import serialize, deserialize

from typing import Optional


class Int(int, PyPrimitive):
    @syft_decorator(typechecking=True, prohibit_args=False)
    def __new__(
        cls, value: Any = None, base: Any = 10, id: Optional[UID] = None
    ) -> int:
        if value is None:
            value = 0

        if isinstance(value, str):
            return int.__new__(cls, value, base)

        return int.__new__(cls, value)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __init__(self, value: Any = None, base: Any = 10, id: Optional[UID] = None):
        if value is None:
            value = 0

        int.__init__(value)

        self._id: UID = UID() if id is None else id

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __add__(self, other: Any) -> PyPrimitive:
        res = super().__add__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __int__(self) -> PyPrimitive:
        res = super().__int__()
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __invert__(self) -> PyPrimitive:
        res = super().__invert__()
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __abs__(self) -> PyPrimitive:
        res = super().__abs__()
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __bool__(self) -> PyPrimitive:
        res = super().__bool__()
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __divmod__(self, other: Any) -> Tuple[PyPrimitive, PyPrimitive]:
        q, r = super().__divmod__(other)
        return (
            PrimitiveFactory.generate_primitive(value=q),
            PrimitiveFactory.generate_primitive(value=r),
        )

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __radd__(self, other: Any) -> PyPrimitive:
        res = super().__radd__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __sub__(self, other: Any) -> PyPrimitive:
        res = super().__sub__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __rsub__(self, other: Any) -> PyPrimitive:
        res = super().__rsub__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __rtruediv__(self, other: Any) -> PyPrimitive:
        res = super().__rtruediv__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __mul__(self, other: Any) -> PyPrimitive:
        res = super(Int, self).__mul__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __rmul__(self, other: Any) -> PyPrimitive:
        res = super().__rmul__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __ceil__(self) -> PyPrimitive:
        res = super().__ceil__()
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

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __eq__(self, other: Any) -> PyPrimitive:
        res = super().__eq__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __float__(self) -> PyPrimitive:
        res = super().__float__()
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __floor__(self) -> PyPrimitive:
        res = super().__floor__()
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __floordiv__(self, other: Any) -> PyPrimitive:
        res = super().__floordiv__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __truediv__(self, other: Any) -> PyPrimitive:
        res = super().__truediv__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __mod__(self, other: Any) -> PyPrimitive:
        res = super().__mod__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __rmod__(self, other: Any) -> PyPrimitive:
        res = super().__rmod__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __pow__(self, other: Any) -> PyPrimitive:
        res = super().__pow__(other)
        return PrimitiveFactory.generate_primitive(res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __rpow__(self, other: Any) -> PyPrimitive:
        res = super().__rpow__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __lshift__(self, other: Any) -> PyPrimitive:
        res = super(Int, self).__lshift__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __rlshift__(self, other: Any) -> PyPrimitive:
        res = super().__rlshift__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __round__(self) -> PyPrimitive:
        res = super().__round__()
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __rshift__(self, other: Any) -> PyPrimitive:
        res = super(Int, self).__rshift__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __rrshift__(self, other: Any) -> PyPrimitive:
        res = super().__rrshift__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __and__(self, other: Any) -> PyPrimitive:
        res = super().__and__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __rand__(self, other: Any) -> PyPrimitive:
        res = super().__rand__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __xor__(self, other: Any) -> PyPrimitive:
        res = super(Int, self).__xor__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __rxor__(self, other: Any) -> PyPrimitive:
        res = super().__rxor__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __or__(self, other: Any) -> PyPrimitive:
        res = super().__or__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __ror__(self, other: Any) -> Any:
        res = super().__ror__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __ge__(self, other: Any) -> PyPrimitive:
        res = super().__ge__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __lt__(self, other: Any) -> PyPrimitive:
        res = super().__lt__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __le__(self, other: Any) -> PyPrimitive:
        res = super().__le__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __gt__(self, other: Any) -> PyPrimitive:
        res = super().__gt__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __iadd__(self, other: Any) -> PyPrimitive:
        return PrimitiveFactory.generate_primitive(
            value=super().__add__(other), id=self.id
        )

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __isub__(self, other: Any) -> PyPrimitive:
        return PrimitiveFactory.generate_primitive(
            value=super().__sub__(other), id=self.id
        )

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __imul__(self, other: Any) -> PyPrimitive:
        return PrimitiveFactory.generate_primitive(
            value=super().__mul__(other), id=self.id
        )

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __ifloordiv__(self, other: Any) -> PyPrimitive:
        return PrimitiveFactory.generate_primitive(
            value=super().__floordiv__(other), id=self.id
        )

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __itruediv__(self, other: Any) -> PyPrimitive:
        return PrimitiveFactory.generate_primitive(
            value=super().__truediv__(other), id=self.id
        )

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __imod__(self, other: Any) -> PyPrimitive:
        return PrimitiveFactory.generate_primitive(
            value=super().__mod__(other), id=self.id
        )

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __ipow__(self, other: Any) -> PyPrimitive:
        return PrimitiveFactory.generate_primitive(
            value=super().__pow__(other), id=self.id
        )

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __ne__(self, other: Any) -> PyPrimitive:
        return PrimitiveFactory.generate_primitive(value=super().__ne__(other))

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __hash__(self) -> PyPrimitive:
        return PrimitiveFactory.generate_primitive(value=super().__hash__())

    # add tests
    @syft_decorator(typechecking=True)
    def __neg__(self) -> PyPrimitive:
        return PrimitiveFactory.generate_primitive(value=super().__neg__())

    @syft_decorator(typechecking=True)
    def __pos__(self) -> PyPrimitive:
        return PrimitiveFactory.generate_primitive(value=super().__pos__())

    @syft_decorator(typechecking=True)
    def _object2proto(self) -> Int_PB:
        int_pb = Int_PB()
        int_pb.data = self
        int_pb.id.CopyFrom(serialize(self.id))
        return int_pb

    @staticmethod
    def _proto2object(proto: Int_PB) -> "Int":
        int_id: UID = deserialize(blob=proto.id)
        return Int(value=proto.data, id=int_id)

    @staticmethod
    def get_protobuf_schema() -> GeneratedProtocolMessageType:
        return Int_PB

    # def as_integer_ratio(self) -> Tuple[int, Literal[1]]:
    #     #TODO add this
    #     pass

    def bit_length(self) -> int:
        # TODO add this
        pass

    def denominator(self) -> int:
        # TODO add this
        pass

    # def from_bytes(cls, bytes: Union[Iterable[int], SupportsBytes], byteorder: str, *,
    #                signed: bool = ...) -> int:
    #     #TODO add this
    #     pass

    def imag(self) -> int:
        # TODO add this
        pass

    def numerator(self) -> int:
        # TODO add this
        pass

    def real(self):
        # TODO add this
        pass

    def to_bytes(self, length: int, byteorder: str, *, signed: bool = ...) -> bytes:
        # TODO add this
        pass
