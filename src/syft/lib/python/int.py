from typing import Any, Union
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
        self, value: Any = None, base: Any = 10, id: Optional[UID] = None
    ) -> int:
        if value is None:
            value = 0

        if isinstance(value, str):
            return int.__new__(self, value, base)

        return int.__new__(self, value)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __init__(self, value: Any = None, base: Any = 10, id: Optional[UID] = None):
        if value is None:
            value = 0

        int.__init__(value)

        if id is None:
            self._id = UID()
        else:
            self._id = id

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __add__(self, other: Any) -> PyPrimitive:
        res = super().__add__(other)

        if res is NotImplemented:
            res = other.__radd__(self)

        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __radd__(self, other: Any) -> PyPrimitive:
        return self.__add__(other)

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
        res = super(Int, self).__mul__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __rmul__(self, other: Any) -> PyPrimitive:
        res = super(Int, self).__rmul__(other)
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
        return super().__eq__(other)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __div__(self, other: Any) -> PyPrimitive:
        res = super(Int, self).__div__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __rdiv__(self, other: Any) -> PyPrimitive:
        res = super(Int, self).__rdiv__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __mul__(self, other: Any) -> PyPrimitive:
        res = super(Int, self).__mul__(self, other)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __rmul__(self, other: Any) -> PyPrimitive:
        return self.__mul__(other)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __floordiv__(self, other: Any) -> PyPrimitive:
        res = super(int, self).__div__(self, other)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __truediv__(self, other: Any) -> PyPrimitive:
        res = super(Int, self).__truediv__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __rdiv__(self, other: Any) -> PyPrimitive:
        return self.__div__(other)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __floordiv__(self, other: Any) -> PyPrimitive:
        res = super(Int, self).__floordiv__(other)
        return PrimitiveFactory.generate_primitive(res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __truediv__(self, other: Any) -> PyPrimitive:
        res = super(Int, self).__truediv__(other)
        return PrimitiveFactory.generate_primitive(res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __mod__(self, other: Any) -> PyPrimitive:
        res = super(Int, self).__mod__(other)
        return PrimitiveFactory.generate_primitive(res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __rmod__(self, other: Any) -> PyPrimitive:
        return self.__mod__(other)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __pow__(self, other: Any) -> PyPrimitive:
        res = super(Int, self).__pow__(other)
        return PrimitiveFactory.generate_primitive(res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __rpow__(self, other: Any) -> PyPrimitive:
        return self.__pow__(other)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __lshift__(self, other: Any) -> PyPrimitive:
        res = super(Int, self).__lshift__(other)
        return PrimitiveFactory.generate_primitive(res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __rlshift__(self, other: Any) -> PyPrimitive:
        return self.__lshift__(other)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __rshift__(self, other: Any) -> PyPrimitive:
        res = super(Int, self).__rshift__(other)
        return PrimitiveFactory.generate_primitive(res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __rrshift__(self, other: Any) -> PyPrimitive:
        return self.__rshift__(other)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __and__(self, other: Any) -> PyPrimitive:
        res = super(Int, self).__and__(other)
        return PrimitiveFactory.generate_primitive(res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __rand__(self, other: Any) -> PyPrimitive:
        return self.__and__(other)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __xor__(self, other: Any) -> PyPrimitive:
        res = super(Int, self).__xor__(other)
        return PrimitiveFactory.generate_primitive(res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __rxor__(self, other: Any) -> PyPrimitive:
        return self.__xor__(other)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __or__(self, other: Any) -> PyPrimitive:
        res = super(Int, self).__or__(other)
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

        return f"<{type(self).__name__}:{self.id.value} {super(Int, self).__repr__()}>"

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

    # @id.setter
    # def id(self, value):
    #     self._id = value


#
# class PyPrimitiveWrapper(StorableObject):
#     def __init__(self, value: object):
#         super().__init__(
#             data=value,
#             id=getattr(value, "id", UID()),
#             tags=getattr(value, "tags", []),
#             description=getattr(value, "description", ""),
#         )
#         self.value = value
#
#     def _data_object2proto(self) -> PyPrimitive_PB:
#         return self.data._object2proto()
#
#     @staticmethod
#     def _data_proto2object(proto: PyPrimitive_PB) -> PyPrimitive:
#         return PyPrimitive._proto2object(proto)
#
#     @staticmethod
#     def get_data_protobuf_schema() -> GeneratedProtocolMessageType:
#         return PyPrimitive_PB
#
#     @staticmethod
#     def get_wrapped_type() -> type:
#         return PyPrimitive
#
#     @staticmethod
#     def construct_new_object(
#         id: UID, data: StorableObject, tags: List[str], description: Optional[str]
#     ) -> object:
#         data._id = id
#         data.tags = tags
#         data.description = description
#         return data
#
#
# aggressive_set_attr(
#     obj=PyPrimitive, name="serializable_wrapper_type", attr=PyPrimitiveWrapper
# )
