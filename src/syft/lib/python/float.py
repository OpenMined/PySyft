from typing import Any
from google.protobuf.reflection import GeneratedProtocolMessageType

from ...decorators import syft_decorator
from .primitive_interface import PyPrimitive
from .primitive_factory import PrimitiveFactory
from ...core.common import UID
from ...proto.lib.python.float_pb2 import Float as Float_PB
from ... import serialize, deserialize

from typing import Optional

to_test = [
    "as_integer_ratio",
    "conjugate",
    "fromhex",
    "hex",
    "imag",
    "is_integer",
    "real",
]

get_direct_inheritance = [
    "__hash__",
    "__trunc__",
    "__repr__",
]

ID_PRESERVING_OPERATIONS = [
    "__iadd__",
    "__isub__",
    "__imul__",
    "__ifloordiv__",
    "__idiv__",
    "__itruediv__",
    "__imod__",
    "__ipow__",
]

ID_NONPRESERVING_OPERATIONS = [
    "__abs__",
    "__add__",
    # '__bool__',
    # '__divmod__',
    # '__eq__',
    # '__float__',
    # '__ge__',
    # '__gt__',
    # '__int__',
    # '__le__',
    # '__lt__',
    "__mod__",
    "__mul__",
    "__ne__",
    "__neg__",
    "__pow__",
    "__radd__",
    "__rdivmod__",
    "__rfloordiv__",
    "__rmod__",
    "__rmul__",
    "__round__",
    "__rpow__",
    "__rsub__",
    "__rtruediv__",
    "__sub__",
    "__truediv__",
]


class Float(float, PyPrimitive):
    @syft_decorator(typechecking=True, prohibit_args=False)
    def __new__(self, value: Any = None, id: Optional[UID] = None) -> float:
        if value is None:
            value = 0.0
        return float.__new__(self, value)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __init__(self, value: Any = None, id: Optional[UID] = None):
        if value is None:
            value = 0.0

        float.__init__(value)

        if id is None:
            self._id = UID()
        else:
            self._id = id

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
    def __eq__(self, other: Any) -> bool:
        return super().__eq__(other)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __ge__(self, other: Any) -> bool:
        return self.__ge__(other)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __lt__(self, other: Any) -> bool:
        return self.__lt__(other)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __le__(self, other: Any) -> bool:
        return self.__le__(other)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __gt__(self, other: Any) -> bool:
        return self.__gt__(other)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __repr__(self) -> str:
        return super().__repr__()

    @syft_decorator(typechecking=True)
    def _object2proto(self) -> Float_PB:
        Float_pb = Float_PB()
        Float_pb.data = self
        Float_pb.id.CopyFrom(serialize(self.id))
        return Float_pb

    def __hash__(self):
        return super().__hash__()

    @staticmethod
    def _proto2object(proto: Float_PB) -> "Float":
        Float_id: UID = deserialize(blob=proto.id)
        return Float(value=proto.data, id=Float_id)

    @staticmethod
    def get_protobuf_schema() -> GeneratedProtocolMessageType:
        return Float_PB

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __add__(self, other: Any) -> PyPrimitive:
        res = super().__add__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __radd__(self, other: Any) -> PyPrimitive:
        return Float(value=0.0)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __sub__(self, other: Any) -> PyPrimitive:
        res = super().__sub__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __rsub__(self, other: Any) -> PyPrimitive:
        res = super().__rsub__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __mul__(self, other: Any) -> PyPrimitive:
        res = super().__mul__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __rmul__(self, other: Any) -> PyPrimitive:
        res = super().__rmul__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __mul__(self, other: Any) -> PyPrimitive:
        res = super().__mul__(self, other)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __rmul__(self, other: Any) -> PyPrimitive:
        return self.__mul__(other)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __floordiv__(self, other: Any) -> PyPrimitive:
        res = super().__floordiv__(self, other)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __truediv__(self, other: Any) -> PyPrimitive:
        res = super().__truediv__(other)
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
        return self.__mod__(other)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __pow__(self, other: Any) -> PyPrimitive:
        return PrimitiveFactory.generate_primitive(value=super().__pow__(other))

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __rpow__(self, other: Any) -> PyPrimitive:
        return PrimitiveFactory.generate_primitive(value=super().__rpow__(other))

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
