# stdlib
import base64
from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated
from typing import Any
from typing import Generic
from typing import TypeVar
from typing import Union
from typing import get_args
from typing import get_origin

# third party
import pydantic
from pydantic import TypeAdapter
from pydantic import ValidationError
from pydantic import ValidationInfo
from pydantic import ValidatorFunctionWrapHandler
from pydantic import WrapValidator
from pydantic_core import PydanticCustomError
from typing_extensions import TypeAliasType

# syft absolute
import syft as sy

# relative
from ..server.credentials import SyftSigningKey
from ..server.credentials import SyftVerifyKey
from ..types.datetime import DateTime
from ..types.syft_object import BaseDateTime
from ..types.syft_object_registry import SyftObjectRegistry
from ..types.uid import UID

T = TypeVar("T")

JSON_CANONICAL_NAME_FIELD = "__canonical_name__"
JSON_VERSION_FIELD = "__version__"
JSON_DATA_FIELD = "data"


# JSON validator from Pydantic docs
# Source: https://docs.pydantic.dev/latest/concepts/types/#named-type-aliases
def json_custom_error_validator(
    value: Any, handler: ValidatorFunctionWrapHandler, _info: ValidationInfo
) -> Any:
    """
    Simplify the error message to avoid a gross error stemming
    from exhaustive checking of all union options.
    """
    try:
        return handler(value)
    except ValidationError:
        raise PydanticCustomError(
            "invalid_json",
            "Input is not valid json",
        )


Json = TypeAliasType(  # type: ignore
    "Json",
    Annotated[
        dict[str, "Json"] | list["Json"] | str | int | float | bool | None,  # type: ignore
        WrapValidator(json_custom_error_validator),
    ],
)

# Used for validating JSON values
JSON_TYPE_ADAPTER = TypeAdapter(Json)


@dataclass
class JSONSerde(Generic[T]):
    # TODO add json schema
    klass: type[T]
    serialize_fn: Callable[[T], Json] | None = None
    deserialize_fn: Callable[[Json], T] | None = None

    def serialize(self, obj: T) -> Json:
        if self.serialize_fn is None:
            return obj  # type: ignore
        else:
            return self.serialize_fn(obj)

    def deserialize(self, obj: Json) -> T:
        if self.deserialize_fn is None:
            return obj  # type: ignore
        return self.deserialize_fn(obj)


JSON_SERDE_REGISTRY: dict[type[T], JSONSerde[T]] = {}


def register_json_serde(
    type_: type[T],
    serialize: Callable[[T], Json] | None = None,
    deserialize: Callable[[Json], T] | None = None,
) -> None:
    if type_ in JSON_SERDE_REGISTRY:
        raise ValueError(f"Type {type_} is already registered")

    JSON_SERDE_REGISTRY[(type_)] = JSONSerde(
        klass=type_,
        serialize_fn=serialize,
        deserialize_fn=deserialize,
    )


# Standard JSON primitives
register_json_serde(int)
register_json_serde(str)
register_json_serde(bool)
register_json_serde(float)
register_json_serde(type(None))

# Syft primitives
register_json_serde(UID, lambda uid: uid.no_dash, lambda s: UID(s))
register_json_serde(
    DateTime, lambda dt: dt.utc_timestamp, lambda f: DateTime(utc_timestamp=f)
)
register_json_serde(
    BaseDateTime, lambda dt: dt.utc_timestamp, lambda f: BaseDateTime(utc_timestamp=f)
)
register_json_serde(SyftVerifyKey, lambda key: str(key), SyftVerifyKey.from_string)
register_json_serde(SyftSigningKey, lambda key: str(key), SyftSigningKey.from_string)


def _is_optional_annotation(annotation: Any) -> Any:
    return annotation | None == annotation


def _get_nonoptional_annotation(annotation: Any) -> Any:
    """Return the type anntation with None type removed, if it is present.

    Args:
        annotation (Any): type annotation

    Returns:
        Any: type annotation without None type
    """
    if _is_optional_annotation(annotation):
        args = get_args(annotation)
        return Union[tuple(arg for arg in args if arg is not type(None))]  # noqa
    return annotation


def _annotation_is_subclass_of(annotation: Any, cls: type) -> bool:
    try:
        return issubclass(annotation, cls)
    except TypeError:
        return False


def _serialize_pydantic_to_json(obj: pydantic.BaseModel) -> Json:
    canonical_name, version = SyftObjectRegistry.get_canonical_name_version(obj)
    result = {
        JSON_CANONICAL_NAME_FIELD: canonical_name,
        JSON_VERSION_FIELD: version,
    }

    for key, type_ in obj.model_fields.items():
        result[key] = serialize_json(getattr(obj, key), type_.annotation)
    return result


def _deserialize_pydantic_from_json(
    obj_dict: dict[str, Json],
) -> pydantic.BaseModel:
    canonical_name = obj_dict[JSON_CANONICAL_NAME_FIELD]
    version = obj_dict[JSON_VERSION_FIELD]
    obj_type = SyftObjectRegistry.get_serde_class(canonical_name, version)

    result = {}
    for key, type_ in obj_type.model_fields.items():
        result[key] = deserialize_json(obj_dict[key], type_.annotation)

    return obj_type.model_validate(result)


def _is_serializable_iterable(annotation: Any) -> bool:
    # we can only serialize typed iterables without Union/Any
    # NOTE optional is allowed

    # 1. check if it is an iterable
    if get_origin(annotation) not in {list, tuple, set, frozenset}:
        return False

    # 2. check if iterable annotation is serializable
    args = get_args(annotation)
    if len(args) != 1:
        return False

    inner_type = _get_nonoptional_annotation(args[0])
    return inner_type in JSON_SERDE_REGISTRY or _annotation_is_subclass_of(
        inner_type, pydantic.BaseModel
    )


def _serialize_iterable_to_json(value: Any, annotation: Any) -> Json:
    return [serialize_json(v) for v in value]


def _deserialize_iterable_from_json(value: Json, annotation: Any) -> Any:
    if not isinstance(value, list):
        raise ValueError(f"Cannot deserialize {type(value)} to {annotation}")

    annotation = _get_nonoptional_annotation(annotation)

    if not _is_serializable_iterable(annotation):
        raise ValueError(f"Cannot deserialize {annotation} from JSON")

    inner_type = _get_nonoptional_annotation(get_args(annotation)[0])
    return [deserialize_json(v, inner_type) for v in value]


def _is_serializable_mapping(annotation: Any) -> bool:
    """
    Mapping is serializable if:
    - it is a dict
    - the key type is str
    - the value type is serializable and not a Union
    """
    if get_origin(annotation) != dict:
        return False

    args = get_args(annotation)
    if len(args) != 2:
        return False

    key_type, value_type = args
    # JSON only allows string keys
    if not isinstance(key_type, str):
        return False

    # check if value type is serializable
    value_type = _get_nonoptional_annotation(value_type)
    return value_type in JSON_SERDE_REGISTRY or _annotation_is_subclass_of(
        value_type, pydantic.BaseModel
    )


def _serialize_mapping_to_json(value: Any, annotation: Any) -> Json:
    _, value_type = get_args(annotation)
    return {k: serialize_json(v, value_type) for k, v in value.items()}


def _deserialize_mapping_from_json(value: Json, annotation: Any) -> Any:
    if not isinstance(value, dict):
        raise ValueError(f"Cannot deserialize {type(value)} to {annotation}")

    annotation = _get_nonoptional_annotation(annotation)

    if not _is_serializable_mapping(annotation):
        raise ValueError(f"Cannot deserialize {annotation} from JSON")

    _, value_type = get_args(annotation)
    return {k: deserialize_json(v, value_type) for k, v in value.items()}


def _serialize_to_json_bytes(obj: Any) -> str:
    obj_bytes = sy.serialize(obj, to_bytes=True)
    return base64.b64encode(obj_bytes).decode("utf-8")


def _deserialize_from_json_bytes(obj: str) -> Any:
    obj_bytes = base64.b64decode(obj)
    return sy.deserialize(obj_bytes, from_bytes=True)


def serialize_json(value: Any, annotation: Any = None) -> Json:
    """
    Serialize a value to a JSON-serializable object, using the schema defined by the
    provided annotation.

    Serialization is always done according to the annotation, as the same annotation
    is used for deserialization. If the annotation is not provided or is ambiguous,
    the JSON serialization will fall back to serializing bytes.

    'Strictly typed' means the annotation is unambiguous during deserialization:
    - `int | None` is strictly typed and serialized to int (nullable)
    - `str | int` is ambiguous and serialized to bytes
    - `list[int]` is strictly typed
    - `list`, `list[str | int]`, `list[Any]` are ambiguous and serialized to bytes
    - Optional types are treated as strictly typed if the inner type is strictly typed

    The function chooses the appropriate serialization method in the following order:
    1. Method registered in `JSON_SERDE_REGISTRY` for the annotation type.
    2. Pydantic model serialization, including all `SyftObjects`.
    3. Iterable serialization, if the annotation is a strict iterable (e.g., `list[int]`).
    4. Mapping serialization, if the annotation is a strictly typed mapping with string keys.
    5. Serialize the object to bytes and encode it as base64.

    Args:
        value (Any): Value to serialize.
        annotation (Any, optional): Type annotation for the value. Defaults to None.

    Returns:
        Json: JSON-serializable object.
    """
    if annotation is None:
        annotation = type(value)

    if value is None:
        return None

    # Remove None type from annotation if it is present.
    annotation = _get_nonoptional_annotation(annotation)

    if annotation in JSON_SERDE_REGISTRY:
        return JSON_SERDE_REGISTRY[annotation].serialize(value)
    # SyftObject, or any other Pydantic model
    elif _annotation_is_subclass_of(annotation, pydantic.BaseModel):
        return _serialize_pydantic_to_json(value)

    # Recursive types
    # NOTE only strictly annotated iterables and mappings are supported
    # example: list[int] is supported, but not list[Union[int, str]]
    elif _is_serializable_iterable(annotation):
        return _serialize_iterable_to_json(value, annotation)
    elif _is_serializable_mapping(annotation):
        return _serialize_mapping_to_json(value, annotation)
    else:
        return _serialize_to_json_bytes(value)


def deserialize_json(value: Json, annotation: Any) -> Any:
    """Deserialize a JSON-serializable object to a value, using the schema defined by the
    provided annotation. Inverse of `serialize_json`.

    Args:
        value (Json): JSON-serializable object.
        annotation (Any): Type annotation for the value.

    Returns:
        Any: Deserialized value.
    """
    if value is None:
        return None

    # Remove None type from annotation if it is present.
    annotation = _get_nonoptional_annotation(annotation)

    if annotation in JSON_SERDE_REGISTRY:
        return JSON_SERDE_REGISTRY[annotation].deserialize(value)
    elif _annotation_is_subclass_of(annotation, pydantic.BaseModel):
        return _deserialize_pydantic_from_json(value)
    elif isinstance(value, list):
        return _deserialize_iterable_from_json(value, annotation)
    elif isinstance(value, dict):
        return _deserialize_mapping_from_json(value, annotation)
    else:
        return _deserialize_from_json_bytes(value)
