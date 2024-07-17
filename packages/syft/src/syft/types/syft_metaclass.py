# stdlib
from typing import Any
from typing import final

# third party
from pydantic import BaseModel
from pydantic._internal._model_construction import ModelMetaclass

# relative
from ..serde.serializable import serializable


class EmptyType(type):
    def __repr__(self) -> str:
        return self.__name__

    def __bool__(self) -> bool:
        return False


@serializable(canonical_name="Empty", version=1)
@final
class Empty(metaclass=EmptyType):
    pass


class PartialModelMetaclass(ModelMetaclass):
    def __new__(
        mcs,
        cls_name: str,
        bases: tuple[type[Any], ...],
        namespace: dict[str, Any],
        *args: Any,
        **kwargs: Any,
    ) -> type:
        cls = super().__new__(mcs, cls_name, bases, namespace, *args, **kwargs)

        if issubclass(cls, BaseModel):
            for field_info in cls.model_fields.values():
                if field_info.annotation is not None and field_info.is_required():
                    field_info.annotation = field_info.annotation | EmptyType
                    field_info.default = Empty

            cls.model_rebuild(force=True)

        return cls
