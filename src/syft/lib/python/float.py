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


class Float(float, PyPrimitive):
    @syft_decorator(typechecking=True, prohibit_args=False)
    def __new__(self, value: Any = None, id: Optional[UID] = None) -> "Float":
        if value is None:
            value = 0.0
        return float.__new__(self, value)  # type: ignore

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

    def __hash__(self) -> int:
        return super().__hash__()

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __eq__(self, other: Any) -> PyPrimitive:
        result = super().__eq__(other)
        return PrimitiveFactory.generate_primitive(value=result)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __ge__(self, other: Any) -> PyPrimitive:
        result = super().__ge__(other)
        return PrimitiveFactory.generate_primitive(value=result)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __lt__(self, other: Any) -> PyPrimitive:
        result = super().__lt__(other)
        return PrimitiveFactory.generate_primitive(value=result)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __le__(self, other: Any) -> PyPrimitive:
        result = super().__le__(other)
        return PrimitiveFactory.generate_primitive(value=result)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __gt__(self, other: Any) -> PyPrimitive:
        result = super().__gt__(other)
        return PrimitiveFactory.generate_primitive(value=result)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __repr__(self) -> str:
        return super().__repr__()

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __add__(self, other: Any) -> PyPrimitive:
        res = super().__add__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __radd__(self, other: Any) -> PyPrimitive:
        return Float(value=other)

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

    @syft_decorator(typechecking=True)
    def _object2proto(self) -> Float_PB:
        return Float_PB(
            id_at_location=serialize(obj=self.id),
            data=self,
        )

    @staticmethod
    def _proto2object(proto: Float_PB) -> "Float":
        return Float(value=proto.data, id=deserialize(blob=proto.id_at_location))

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
        if _object2proto is not None:
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
