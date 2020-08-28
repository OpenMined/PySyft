from typing import Any
from google.protobuf.reflection import GeneratedProtocolMessageType

from ...decorators import syft_decorator
from .primitive_interface import  PyPrimitive
from .primitive_factory import PrimitiveFactory
from ...core.common import UID
from ...proto.lib.python.float_pb2 import Float as Float_PB
from ... import serialize, deserialize

from typing import Optional


class Float(float, PyPrimitive):
    @syft_decorator(typechecking=True)
    def __new__(self, value: float, id: Optional[UID] = None) -> float:
        return float.__new__(self, value)

    @syft_decorator(typechecking=True)
    def __init__(self, value: float, id: Optional[UID] = None):
        float.__init__(value)

        if id is None:
            self._id = UID()
        else:
            self._id = id

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
        return self == other

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __div__(self, other: Any) -> PyPrimitive:
        res = super().__div__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __rdiv__(self, other: Any) -> PyPrimitive:
        res = super().__rdiv__(other)
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
        res = super().__div__(self, other)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __truediv__(self, other: Any) -> PyPrimitive:
        res = super().__truediv__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __rdiv__(self, other: Any) -> PyPrimitive:
        return self.__div__(other)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __floordiv__(self, other: Any) -> PyPrimitive:
        res = super().__floordiv__(other)
        return PrimitiveFactory.generate_primitive(res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __truediv__(self, other: Any) -> PyPrimitive:
        res = super().__truediv__(other)
        return PrimitiveFactory.generate_primitive(res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __mod__(self, other: Any) -> PyPrimitive:
        res = super().__mod__(other)
        return PrimitiveFactory.generate_primitive(res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __rmod__(self, other: Any) -> PyPrimitive:
        return self.__mod__(other)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __pow__(self, other: Any) -> PyPrimitive:
        res = super().__pow__(other)
        return PrimitiveFactory.generate_primitive(res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __rpow__(self, other: Any) -> PyPrimitive:
        return self.__pow__(other)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __lshift__(self, other: Any) -> PyPrimitive:
        res = super().__lshift__(other)
        return PrimitiveFactory.generate_primitive(res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __rlshift__(self, other: Any) -> PyPrimitive:
        return self.__lshift__(other)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __rshift__(self, other: Any) -> PyPrimitive:
        res = super().__rshift__(other)
        return PrimitiveFactory.generate_primitive(res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __rrshift__(self, other: Any) -> PyPrimitive:
        return self.__rshift__(other)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __and__(self, other: Any) -> PyPrimitive:
        res = super().__and__(other)
        return PrimitiveFactory.generate_primitive(res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __rand__(self, other: Any) -> PyPrimitive:
        return self.__and__(other)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __xor__(self, other: Any) -> PyPrimitive:
        res = super().__xor__(other)
        return PrimitiveFactory.generate_primitive(res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __rxor__(self, other: Any) -> PyPrimitive:
        return self.__xor__(other)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __or__(self, other: Any) -> PyPrimitive:
        res = super().__or__(other)
        return PrimitiveFactory.generate_primitive(res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __ror__(self, other: Any) -> Any:
        return self.__or__(other)

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

    @syft_decorator(typechecking=True)
    def __repr__(self) -> str:
        """Returns a human-readable version of the PyPrimitive

        Return a human-readable representation of the PyPrimitive with brackets
        so that it can be easily spotted when nested inside of the human-
        readable representations of other objects."""

        return f"<{type(self).__name__}:{self.id.value} {super(Float, self).__repr__()}>"

    @syft_decorator(typechecking=True)
    def _object2proto(self) -> Float_PB:
        Float_pb = Float_PB()
        Float_pb.data = self
        Float_pb.id.CopyFrom(serialize(self.id))
        return Float_pb

    @staticmethod
    def _proto2object(proto: Float_PB) -> "Float":
        Float_id: UID = deserialize(blob=proto.id)
        return Float(value=proto.data, id=Float_id)

    @staticmethod
    def get_protobuf_schema() -> GeneratedProtocolMessageType:
        return Float_PB