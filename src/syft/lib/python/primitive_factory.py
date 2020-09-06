# stdlib
from typing import Any
from typing import Dict
from typing import Optional
from typing import Type
from typing import Union

# syft relative
from ...core.common import UID
from ...decorators import syft_decorator
from .primitive_interface import PyPrimitive
from .util import NotImplementedType


def isprimitive(value: Any) -> bool:
    if not issubclass(type(value), PyPrimitive) and type(value) in [
        int,
        float,
        bool,
        complex,
        list,
        str,
        None,
    ]:
        return True
    return False


class PrimitiveFactory:
    KNOWN_TYPES: Dict[
        Union[
            Type[int],
            Type[float],
            Type[bool],
            Type[complex],
            Type[list],
            Type[str],
            None,
            NotImplementedType,
        ],
        Type[PyPrimitive],
    ] = {}

    DEFAULT_TYPE: Optional[Type[PyPrimitive]] = None

    @staticmethod
    @syft_decorator(typechecking=True)
    def generate_primitive(
        value: Union[int, float, bool, complex, list, str, None, NotImplementedType],
        id: Optional[UID] = None,
    ) -> Union[PyPrimitive, NotImplementedType]:

        if value is NotImplemented:
            return NotImplemented

        key = value if value is None else type(value)
        constructor = PrimitiveFactory.KNOWN_TYPES.get(
            key, PrimitiveFactory.DEFAULT_TYPE
        )

        kwargs: Dict[str, Any]
        if isinstance(value, complex):
            kwargs = {"real": value.real, "imag": value.imag, "id": id}
        else:
            kwargs = {"value": value, "id": id}

        if constructor is None:
            raise TypeError(
                f"Type sent {key} not found in the registered types {PrimitiveFactory.KNOWN_TYPES.keys()} "
                "and there is no default type constructor registered"
            )

        return constructor(**kwargs)

    @staticmethod
    @syft_decorator(typechecking=True)
    def register_primitive(
        python_primitive: Union[
            Type[int],
            Type[float],
            Type[bool],
            Type[complex],
            Type[list],
            Type[str],
            None,
            NotImplementedType,
        ],
        syft_primitive: Type[PyPrimitive],
    ) -> None:
        PrimitiveFactory.KNOWN_TYPES[python_primitive] = syft_primitive

    @staticmethod
    @syft_decorator(typechecking=True)
    def register_default(syft_primitive: Type[PyPrimitive]) -> None:
        if PrimitiveFactory.DEFAULT_TYPE is not None:
            raise ValueError(
                f"Default Type already initialized with {PrimitiveFactory.DEFAULT_TYPE}"
            )
        PrimitiveFactory.DEFAULT_TYPE = syft_primitive
