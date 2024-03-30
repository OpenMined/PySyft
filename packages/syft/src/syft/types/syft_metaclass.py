# stdlib
from collections import defaultdict
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
        orig_field_map: dict = defaultdict(dict)

        def optionalize(restore: bool = False) -> None:
            if not restore:
                for field_name, field_info in cls.model_fields.items():
                    if field_info.annotation is not None and field_info.is_required():
                        orig_field_map[field_name]["annotation"] = field_info.annotation
                        orig_field_map[field_name]["default"] = field_info.default
                        field_info.annotation = field_info.annotation | EmptyType
                        field_info.default = Empty
            else:
                for field_name, field_info in cls.model_fields.items():
                    if field_info.default == Empty:
                        field_info.annotation = orig_field_map[field_name]["annotation"]
                        field_info.default = orig_field_map[field_name]["default"]

        optionalize()
        cls.model_rebuild(force=True)

        _obj = super().__call__(*args, **kwargs)  # type: ignore[misc]

        optionalize(restore=True)
        cls.model_rebuild(force=True)

        return _obj
