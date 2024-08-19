# stdlib
import base64
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from typing import Generic
from typing import TypeVar
from typing import Union
from typing import get_args

# third party
import pydantic

# syft absolute
import syft as sy

# relative
from ...server.credentials import SyftSigningKey
from ...server.credentials import SyftVerifyKey
from ...types.datetime import DateTime
from ...types.uid import UID

T = TypeVar("T")
_PYDANTICTYPE = TypeVar("T", bound=pydantic.BaseModel)
JsonValue = None | int | float | str | bool | list["JsonValue"] | dict[str, "JsonValue"]


@dataclass
class JSONSerde(Generic[T]):
    # TODO
    # - add json schema
    type_: type[T]
    serialize_fn: Callable[[T], JsonValue] | None = None
    deserialize_fn: Callable[[JsonValue], T] | None = None

    def serialize(self, obj: T) -> JsonValue:
        if self.serialize_fn is None:
            return obj
        return self.serialize_fn(obj)

    def deserialize(self, obj: JsonValue) -> T:
        if self.deserialize_fn is None:
            return obj
        return self.deserialize_fn(obj)


JSON_SERDE_REGISTRY: dict[type[T], JSONSerde[T]] = {}


def register_json_serde(
    type_: type[T],
    serialize: Callable[[T], JsonValue] | None = None,
    deserialize: Callable[[JsonValue], T] | None = None,
    overwrite: bool = False,
) -> None:
    if not overwrite and type_ in JSON_SERDE_REGISTRY:
        raise ValueError(f"Type {type_} is already registered")
    JSON_SERDE_REGISTRY[type_] = JSONSerde(
        type_=type_,
        serialize_fn=serialize,
        deserialize_fn=deserialize,
    )


register_json_serde(int)
register_json_serde(str)
register_json_serde(bool)
register_json_serde(float)
register_json_serde(None)
register_json_serde(UID, lambda uid: uid.no_dash, lambda s: UID(s))
register_json_serde(DateTime, lambda dt: dt.utc_timestamp, DateTime.from_timestamp)
register_json_serde(SyftVerifyKey, lambda key: str(key), SyftVerifyKey.from_string)
register_json_serde(SyftSigningKey, lambda key: str(key), SyftSigningKey.from_string)


def is_optional_annotation(annotation: Any) -> Any:
    return annotation | None == annotation


def get_nonoptional_type(annotation):
    """Return the annotation with None type removed, if it is present.

    examples:
    - get_nonoptional_type(Optional[int]) -> int
    - get_nonoptional_type(int) -> int
    - get_nonoptional_type(int | str | None) -> int | str

    Args:
        annotation (Any): type annotation

    Returns:
        Any: type annotation without None type
    """
    if is_optional_annotation(annotation):
        args = get_args(annotation)
        return Union[tuple(arg for arg in args if arg is not type(None))]  # noqa
    return annotation


def serialize_to_json_fallback(obj: Any) -> JsonValue:
    obj_bytes = sy.serialize(obj, to_bytes=True)
    return base64.b64encode(obj_bytes).decode("utf-8")


def deserialize_from_json_fallback(obj: JsonValue) -> Any:
    obj_bytes = base64.b64decode(obj)
    return sy.deserialize(obj_bytes, from_bytes=True)


def model_dump(obj: pydantic.BaseModel) -> JsonValue:
    result = {}

    # NOTE obj.model_dump() does not work when it contains unserializable nested objects
    for key, type_ in obj.model_fields.items():
        annotation = type_.annotation
        value = getattr(obj, key)
        # Remove None type from annotation if it is present.
        annotation = get_nonoptional_type(annotation)

        # NOTE we are permissive with None values, because validation already happened
        # on the pydantic model
        if value is None:
            result[key] = None

        if annotation in JSON_SERDE_REGISTRY:
            result[key] = JSON_SERDE_REGISTRY[annotation].serialize(value)

        elif issubclass(annotation, pydantic.BaseModel):
            result[key] = model_dump(value)

        else:
            result[key] = serialize_to_json_fallback(value)

    return result


def model_validate(obj_type: type[T], obj_dict: dict) -> T:
    for key, type_ in obj_type.model_fields.items():
        if key not in obj_dict or obj_dict[key] is None:
            continue

        annotation = type_.annotation
        annotation = get_nonoptional_type(annotation)

        if annotation in JSON_SERDE_REGISTRY:
            obj_dict[key] = JSON_SERDE_REGISTRY[annotation].deserialize(obj_dict[key])

        elif issubclass(annotation, pydantic.BaseModel):
            obj_dict[key] = model_validate(annotation, obj_dict[key])

        else:
            obj_dict[key] = deserialize_from_json_fallback(obj_dict[key])

    return obj_type.model_validate(obj_dict)


# def should_handle_as_bytes(type_) -> bool:
#     # relative
#     from ...util.misc_objs import HTMLObject
#     from ...util.misc_objs import MarkdownDescription
#     from ..action.action_object import Action
#     from ..dataset.dataset import Asset
#     from ..dataset.dataset import Contributor
#     from ..request.request import Change
#     from ..request.request import ChangeStatus
#     from ..settings.settings import PwdTokenResetConfig

#     return (
#         type_.annotation is LinkedObject
#         or type_.annotation == LinkedObject | None
#         or type_.annotation == list[UID] | dict[str, UID] | None
#         or type_.annotation == dict[str, UID] | None
#         or type_.annotation == list[Change]
#         or type_.annotation == Any | None  # type: ignore
#         or type_.annotation == Action | None  # type: ignore
#         or getattr(type_.annotation, "__origin__", None) is dict
#         or type_.annotation == HTMLObject | MarkdownDescription
#         or type_.annotation == PwdTokenResetConfig
#         or type_.annotation == list[ChangeStatus]
#         or type_.annotation == list[Asset]
#         or type_.annotation == set[Contributor]
#         or type_.annotation == MarkdownDescription
#         or type_.annotation == Contributor
#     )
