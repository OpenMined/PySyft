# stdlib
from typing import Any
from typing import Optional

# syft relative
from ...core.common import UID
from ...decorators import syft_decorator
from .primitive_factory import PrimitiveFactory
from .primitive_interface import PyPrimitive


# TODO - actually make all of this work
class Complex(complex, PyPrimitive):
    @syft_decorator(typechecking=True, prohibit_args=False)
    def __new__(self, real: Any = None, imag: Any = None, id: Optional[UID] = None) -> "Complex":
        if real is None:
            return complex.__new__(self)  # type: ignore
        if imag is None:
            return complex.__new__(self, real=real)  # type: ignore
        if isinstance(real, str):
            raise TypeError("Complex() can't take second arg if first is a string")
        return complex.__new__(self, real=real, imag=imag)  # type: ignore

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __init__(self, real: Any = None, imag: Any = None, id: Optional[UID] = None):
        complex.__init__(self)
        self._id = id or UID()

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
