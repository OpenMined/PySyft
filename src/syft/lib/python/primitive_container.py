# stdlib
from typing import Any as TypeAny

# syft relative
from ...decorators import syft_decorator
from .util import SyPrimitiveRet


class Any:
    @syft_decorator(typechecking=True, prohibit_args=False)
    def __iter__(self) -> "Any":
        return self

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __next__(self) -> SyPrimitiveRet:
        return self.__next__()

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __add__(self, other: TypeAny) -> SyPrimitiveRet:
        return self + other

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __radd__(self, other: TypeAny) -> SyPrimitiveRet:
        return other + self

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __truediv__(self, other: TypeAny) -> SyPrimitiveRet:
        return self / other

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __rtruediv__(self, other: TypeAny) -> SyPrimitiveRet:
        return other / self

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __floordiv__(self, other: TypeAny) -> SyPrimitiveRet:
        return self / other

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __rfloordiv__(self, other: TypeAny) -> SyPrimitiveRet:
        return self / other

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __mul__(self, other: TypeAny) -> SyPrimitiveRet:
        return self * other

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __rmul__(self, other: TypeAny) -> SyPrimitiveRet:
        return other * self

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __sub__(self, other: TypeAny) -> SyPrimitiveRet:
        return self - other

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __rsub__(self, other: TypeAny) -> SyPrimitiveRet:
        return other - self
