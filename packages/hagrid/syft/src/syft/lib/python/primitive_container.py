# stdlib
from typing import Any as TypeAny

# relative
from .types import SyPrimitiveRet


class Any:
    def __iter__(self) -> "Any":
        return self

    def __next__(self) -> SyPrimitiveRet:
        return self.__next__()

    def __add__(self, other: TypeAny) -> SyPrimitiveRet:
        return self + other

    def __radd__(self, other: TypeAny) -> SyPrimitiveRet:
        return other + self

    def __truediv__(self, other: TypeAny) -> SyPrimitiveRet:
        return self / other

    def __rtruediv__(self, other: TypeAny) -> SyPrimitiveRet:
        return other / self

    def __floordiv__(self, other: TypeAny) -> SyPrimitiveRet:
        return self / other

    def __rfloordiv__(self, other: TypeAny) -> SyPrimitiveRet:
        return self / other

    def __mul__(self, other: TypeAny) -> SyPrimitiveRet:
        return self * other

    def __rmul__(self, other: TypeAny) -> SyPrimitiveRet:
        return other * self

    def __sub__(self, other: TypeAny) -> SyPrimitiveRet:
        return self - other

    def __rsub__(self, other: TypeAny) -> SyPrimitiveRet:
        return other - self

    def __len__(self) -> SyPrimitiveRet:
        return self.__len__()
