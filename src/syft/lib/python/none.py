# syft relative
from ...decorators.syft_decorator_impl import syft_decorator
from .primitive_interface import PyPrimitive

NoneType = type(None)


class SyNone(PyPrimitive):
    @syft_decorator(typechecking=True, prohibit_args=True)
    def upcast(self) -> NoneType:
        return None
