# stdlib
from typing import Any
from typing import Optional

# relative
from ...logger import traceback_and_raise
from .primitive_factory import PrimitiveFactory
from .primitive_interface import PyPrimitive
from .types import SyPrimitiveRet


# TODO - actually make all of this work
class Complex(complex, PyPrimitive):
    def __new__(self, real: Any = None, imag: Any = None) -> "Complex":
        if real is None:
            return complex.__new__(self)
        if imag is None:
            return complex.__new__(self, real=real)
        if isinstance(real, str):
            traceback_and_raise(
                TypeError("Complex() can't take second arg if first is a string")
            )
        return complex.__new__(self, real=real, imag=imag)

    def __init__(self, real: Any = None, imag: Any = None):
        complex.__init__(self)

    def upcast(self) -> complex:
        return complex(self.real, self.imag)  # type: ignore

    def __add__(self, x: complex) -> SyPrimitiveRet:
        result = complex.__add__(self, x)
        return PrimitiveFactory.generate_primitive(value=result)

    def __sub__(self, x: complex) -> SyPrimitiveRet:
        result = complex.__sub__(self, x)
        return PrimitiveFactory.generate_primitive(value=result)

    def __mul__(self, x: complex) -> SyPrimitiveRet:
        result = complex.__mul__(self, x)
        return PrimitiveFactory.generate_primitive(value=result)

    def __pow__(self, x: complex, modulo: Any = 10) -> SyPrimitiveRet:
        result = complex.__pow__(self, x, modulo)
        return PrimitiveFactory.generate_primitive(value=result)

    def __truediv__(self, x: complex) -> SyPrimitiveRet:
        result = complex.__truediv__(self, x)
        return PrimitiveFactory.generate_primitive(value=result)

    def __radd__(self, x: complex) -> SyPrimitiveRet:
        result = complex.__radd__(self, x)
        return PrimitiveFactory.generate_primitive(value=result)

    def __rsub__(self, x: complex) -> SyPrimitiveRet:
        result = complex.__rsub__(self, x)
        return PrimitiveFactory.generate_primitive(value=result)

    def __rmul__(self, x: complex) -> SyPrimitiveRet:
        result = complex.__rmul__(self, x)
        return PrimitiveFactory.generate_primitive(value=result)

    def __rpow__(self, x: complex, modulo: Optional[Any] = None) -> SyPrimitiveRet:
        if modulo:
            return PrimitiveFactory.generate_primitive(
                value=complex.__rpow__(self, x, modulo)
            )
        return PrimitiveFactory.generate_primitive(value=complex.__rpow__(self, x))

    def __rtruediv__(self, x: complex) -> SyPrimitiveRet:
        result = complex.__rtruediv__(self, x)
        return PrimitiveFactory.generate_primitive(value=result)

    def __eq__(self, x: object) -> SyPrimitiveRet:
        result = complex.__eq__(self, x)
        return PrimitiveFactory.generate_primitive(value=result)

    def __ne__(self, x: object) -> SyPrimitiveRet:
        result = complex.__ne__(self, x)
        return PrimitiveFactory.generate_primitive(value=result)

    def __neg__(self) -> SyPrimitiveRet:
        result = complex.__neg__(self)
        return PrimitiveFactory.generate_primitive(value=result)

    def __pos__(self) -> SyPrimitiveRet:
        result = complex.__pos__(self)
        return PrimitiveFactory.generate_primitive(value=result)

    # def __str__(self) -> PyPrimitive:
    #     ...

    # def __complex__(self) -> "Complex":
    #     result = complex.__complex__()
    #     return PrimitiveFactory.generate_primitive(value=result)

    def __abs__(self) -> SyPrimitiveRet:
        result = complex.__abs__(self)
        return PrimitiveFactory.generate_primitive(value=result)

    def __hash__(self) -> SyPrimitiveRet:
        result = complex.__hash__(self)
        return PrimitiveFactory.generate_primitive(value=result)

    def __bool__(self) -> bool:
        # NOTE we return a real bool here, not a syft Bool
        return complex.__bool__(self)
