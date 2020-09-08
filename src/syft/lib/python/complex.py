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
        if isinstance(real, str) or imag is None:
            return complex.__new__(self, real=real)  # type: ignore
        return complex.__new__(self, real=real, imag=imag)  # type: ignore

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __init__(self, real: Any = None, imag: Any = None, id: Optional[UID] = None):
        complex.__init__(self)
        self._id = id or UID()

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __add__(self, x: complex) -> "Complex":
        result = super().__add__(x)
        return PrimitiveFactory.generate_primitive(value=result)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __sub__(self, x: complex) -> "Complex":
        result = super().__sub__(x)
        return PrimitiveFactory.generate_primitive(value=result)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __mul__(self, x: complex) -> "Complex":
        result = super().__mul__(x)
        return PrimitiveFactory.generate_primitive(value=result)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __pow__(self, x: complex) -> "Complex":
        result = super().__pow__(x)
        return PrimitiveFactory.generate_primitive(value=result)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __truediv__(self, x: complex) -> "Complex":
        result = super().__truediv__(x)
        return PrimitiveFactory.generate_primitive(value=result)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __radd__(self, x: complex) -> "Complex":
        result = super().__radd__(x)
        return PrimitiveFactory.generate_primitive(value=result)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __rsub__(self, x: complex) -> "Complex":
        result = super().__rsub__(x)
        return PrimitiveFactory.generate_primitive(value=result)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __rmul__(self, x: complex) -> "Complex":
        result = super().__rmul__(x)
        return PrimitiveFactory.generate_primitive(value=result)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __rpow__(self, x: complex) -> "Complex":
        result = super().__rpow__(x)
        return PrimitiveFactory.generate_primitive(value=result)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __rtruediv__(self, x: complex) -> "Complex":
        result = super().__rtruediv__(x)
        return PrimitiveFactory.generate_primitive(value=result)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __eq__(self, x: object) -> PyPrimitive:
        result = super().__eq__(x)
        return PrimitiveFactory.generate_primitive(value=result)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __ne__(self, x: object) -> PyPrimitive:
        result = super().__ne__(x)
        return PrimitiveFactory.generate_primitive(value=result)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __neg__(self) -> "Complex":
        result = super().__neg__()
        return PrimitiveFactory.generate_primitive(value=result)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __pos__(self) -> "Complex":
        result = super().__pos__()
        return PrimitiveFactory.generate_primitive(value=result)

    # @syft_decorator(typechecking=True, prohibit_args=False)
    # def __str__(self) -> PyPrimitive:
    #     ...

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __complex__(self) -> "Complex":
        result = super().__complex__()
        return PrimitiveFactory.generate_primitive(value=result)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __abs__(self) -> PyPrimitive:
        result = super().__abs__()
        return PrimitiveFactory.generate_primitive(value=result)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __hash__(self) -> PyPrimitive:
        result = super().__hash__()
        return PrimitiveFactory.generate_primitive(value=result)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __bool__(self) -> PyPrimitive:
        result = super().__bool__()
        return PrimitiveFactory.generate_primitive(value=result)