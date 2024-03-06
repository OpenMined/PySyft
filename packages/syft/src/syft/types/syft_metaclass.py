# stdlib
from typing import Any
from typing import TypeVar
from typing import final

# third party
from pydantic import BaseModel
from pydantic._internal._model_construction import ModelMetaclass

# relative
from ..serde.serializable import serializable

_T = TypeVar("_T", bound=BaseModel)


class EmptyType(type):
    def __repr__(self) -> str:
        return self.__name__

    def __bool__(self) -> bool:
        return False


@serializable()
@final
class Empty(metaclass=EmptyType):
    pass


class PartialModelMetaclass(ModelMetaclass):
    def __call__(cls: type[_T], *args: Any, **kwargs: Any) -> _T:
        for field_info in cls.model_fields.values():
            if field_info.annotation is not None and field_info.is_required():
                field_info.annotation = field_info.annotation | EmptyType
                field_info.default = Empty

        cls.model_rebuild(force=True)

        return super().__call__(*args, **kwargs)  # type: ignore[misc]
