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
from ...proto.lib.python.complex_pb2 import Complex as Complex_PB
from ...util import aggressive_set_attr
from .primitive_factory import PrimitiveFactory
from .primitive_interface import PyPrimitive


# TODO - actually make all of this work
class Complex(complex, PyPrimitive):
    @syft_decorator(typechecking=True, prohibit_args=False)
    def __new__(
        self, real: Any = None, imag: Any = None, id: Optional[UID] = None
    ) -> "Complex":
        if real is None:
            return complex.__new__(self)
        if imag is None:
            return complex.__new__(self, real=real)  # type: ignore
        if isinstance(real, str):
            raise TypeError("Complex() can't take second arg if first is a string")
        return complex.__new__(self, real=real, imag=imag)  # type: ignore

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __init__(self, real: Any = None, imag: Any = None, id: Optional[UID] = None):
        complex.__init__(self)
        self._id = id or UID()

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
    def upcast(self) -> complex:
        return complex(self)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __add__(self, x: complex) -> "Complex":
        result = complex.__add__(self, x)
        return PrimitiveFactory.generate_primitive(value=result)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __sub__(self, x: complex) -> "Complex":
        result = complex.__sub__(self, x)
        return PrimitiveFactory.generate_primitive(value=result)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __mul__(self, x: complex) -> "Complex":
        result = complex.__mul__(self, x)
        return PrimitiveFactory.generate_primitive(value=result)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __pow__(self, x: complex) -> "Complex":
        result = complex.__pow__(self, x)
        return PrimitiveFactory.generate_primitive(value=result)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __truediv__(self, x: complex) -> "Complex":
        result = complex.__truediv__(self, x)
        return PrimitiveFactory.generate_primitive(value=result)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __radd__(self, x: complex) -> "Complex":
        result = complex.__radd__(self, x)
        return PrimitiveFactory.generate_primitive(value=result)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __rsub__(self, x: complex) -> "Complex":
        result = complex.__rsub__(self, x)
        return PrimitiveFactory.generate_primitive(value=result)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __rmul__(self, x: complex) -> "Complex":
        result = complex.__rmul__(self, x)
        return PrimitiveFactory.generate_primitive(value=result)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __rpow__(self, x: complex) -> "Complex":
        result = complex.__rpow__(self, x)
        return PrimitiveFactory.generate_primitive(value=result)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __rtruediv__(self, x: complex) -> "Complex":
        result = complex.__rtruediv__(self, x)
        return PrimitiveFactory.generate_primitive(value=result)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __eq__(self, x: object) -> PyPrimitive:
        result = complex.__eq__(self, x)
        return PrimitiveFactory.generate_primitive(value=result)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __ne__(self, x: object) -> PyPrimitive:
        result = complex.__ne__(self, x)
        return PrimitiveFactory.generate_primitive(value=result)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __neg__(self) -> "Complex":
        result = complex.__neg__(self)
        return PrimitiveFactory.generate_primitive(value=result)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __pos__(self) -> "Complex":
        result = complex.__pos__(self)
        return PrimitiveFactory.generate_primitive(value=result)

    # @syft_decorator(typechecking=True, prohibit_args=False)
    # def __str__(self) -> PyPrimitive:
    #     ...

    # @syft_decorator(typechecking=True, prohibit_args=False)
    # def __complex__(self) -> "Complex":
    #     result = complex.__complex__()
    #     return PrimitiveFactory.generate_primitive(value=result)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __abs__(self) -> PyPrimitive:
        result = complex.__abs__(self)
        return PrimitiveFactory.generate_primitive(value=result)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __hash__(self) -> PyPrimitive:
        result = complex.__hash__(self)
        return PrimitiveFactory.generate_primitive(value=result)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __bool__(self) -> bool:
        # NOTE we return a real bool here, not a syft Bool
        return complex.__bool__(self)

    @syft_decorator(typechecking=True)
    def _object2proto(self) -> Complex_PB:
        return Complex_PB(id=serialize(obj=self.id), real=self.real, imag=self.imag)

    @staticmethod
    @syft_decorator(typechecking=True)
    def _proto2object(proto: Complex_PB) -> "Complex":
        return Complex(
            id=deserialize(blob=proto.id),
            real=proto.real,
            imag=proto.imag,
        )

    @staticmethod
    def get_protobuf_schema() -> GeneratedProtocolMessageType:
        return Complex_PB


class ComplexWrapper(StorableObject):
    def __init__(self, value: object):
        super().__init__(
            data=value,
            id=getattr(value, "id", UID()),
            tags=getattr(value, "tags", []),
            description=getattr(value, "description", ""),
        )
        self.value = value

    def _data_object2proto(self) -> Complex_PB:
        _object2proto = getattr(self.data, "_object2proto", None)
        if _object2proto is not None:
            return _object2proto()

    @staticmethod
    def _data_proto2object(proto: Complex_PB) -> "ComplexWrapper":
        return Complex._proto2object(proto=proto)

    @staticmethod
    def get_data_protobuf_schema() -> GeneratedProtocolMessageType:
        return Complex_PB

    @staticmethod
    def get_wrapped_type() -> type:
        return Complex

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


aggressive_set_attr(obj=Complex, name="serializable_wrapper_type", attr=ComplexWrapper)
