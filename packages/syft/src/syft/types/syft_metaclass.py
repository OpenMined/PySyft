# stdlib
from typing import TypeVar
from typing import Union
from typing import final

# third party
from pydantic import BaseModel
from pydantic._internal._model_construction import ModelMetaclass
from pydantic.fields import Field
from typing_extensions import dataclass_transform

# relative
from ..serde.serializable import serializable

T = TypeVar("T", bound=BaseModel)


class _EmptyMetaclass(type):
    def __repr__(self) -> str:
        return self.__name__


@serializable
@final
class Empty(metaclass=_EmptyMetaclass):
    pass


@dataclass_transform(kw_only_default=True, field_specifiers=(Field,))
class PartialModelMetaclass(ModelMetaclass):
    def __call__(cls: type[T], *args, **kwargs) -> T:
        for field, field_info in cls.model_fields.items():
            if field_info.annotation is not None and not field_info.is_required():
                cls.model_fields[field].annotation = Union[
                    field_info.annotation, type[Empty]
                ]
                cls.model_fields[field].default = Empty

        cls.model_rebuild(force=True)

        return super().__call__(*args, **kwargs)
