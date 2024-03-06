# stdlib
from collections import OrderedDict
from collections import defaultdict
from collections.abc import Collection
from collections.abc import Iterable
from collections.abc import Mapping
from enum import Enum
from enum import EnumMeta
import functools
import pathlib
from pathlib import PurePath
import sys
from types import MappingProxyType
from types import UnionType
from typing import Any
from typing import GenericAlias
from typing import Optional
from typing import TypeVar
from typing import Union
from typing import _GenericAlias
from typing import _SpecialForm
from typing import _SpecialGenericAlias
from typing import _UnionGenericAlias
from typing import cast
import weakref

# relative
from .capnp import get_capnp_schema
from .recursive import chunk_bytes
from .recursive import combine_bytes
from .recursive import recursive_serde_register

iterable_schema = get_capnp_schema("iterable.capnp").Iterable
kv_iterable_schema = get_capnp_schema("kv_iterable.capnp").KVIterable


def serialize_iterable(iterable: Collection) -> bytes:
    # relative
    from .serialize import _serialize

    message = iterable_schema.new_message()

    message.init("values", len(iterable))

    for idx, it in enumerate(iterable):
        serialized = _serialize(it, to_bytes=True)
        chunk_bytes(serialized, idx, message.values)

    return message.to_bytes()


def deserialize_iterable(iterable_type: type, blob: bytes) -> Collection:
    # relative
    from .deserialize import _deserialize

    MAX_TRAVERSAL_LIMIT = 2**64 - 1
    values = []

    with iterable_schema.from_bytes(
        blob, traversal_limit_in_words=MAX_TRAVERSAL_LIMIT
    ) as msg:
        for element in msg.values:
            values.append(_deserialize(combine_bytes(element), from_bytes=True))

    return iterable_type(values)


_KT = TypeVar("_KT")
_VT = TypeVar("_VT")


def _serialize_kv_pairs(size: int, kv_pairs: Iterable[tuple[_KT, _VT]]) -> bytes:
    # relative
    from .serialize import _serialize

    message = kv_iterable_schema.new_message()

    message.init("keys", size)
    message.init("values", size)

    for index, (k, v) in enumerate(kv_pairs):
        message.keys[index] = _serialize(k, to_bytes=True)
        serialized = _serialize(v, to_bytes=True)
        chunk_bytes(serialized, index, message.values)

    return message.to_bytes()


def serialize_kv(map: Mapping) -> bytes:
    return _serialize_kv_pairs(len(map), map.items())


def get_deserialized_kv_pairs(blob: bytes) -> list[Any]:
    # relative
    from .deserialize import _deserialize

    MAX_TRAVERSAL_LIMIT = 2**64 - 1
    pairs = []

    with kv_iterable_schema.from_bytes(
        blob, traversal_limit_in_words=MAX_TRAVERSAL_LIMIT
    ) as msg:
        for key, value in zip(msg.keys, msg.values):
            pairs.append(
                (
                    _deserialize(key, from_bytes=True),
                    _deserialize(combine_bytes(value), from_bytes=True),
                )
            )
    return pairs


def deserialize_kv(mapping_type: type, blob: bytes) -> Mapping:
    pairs = get_deserialized_kv_pairs(blob=blob)
    return mapping_type(pairs)


def serialize_defaultdict(df_dict: defaultdict) -> bytes:
    # relative
    from .serialize import _serialize

    df_type_bytes = _serialize(df_dict.default_factory, to_bytes=True)
    df_kv_bytes = serialize_kv(df_dict)
    return _serialize((df_type_bytes, df_kv_bytes), to_bytes=True)


def deserialize_defaultdict(blob: bytes) -> Mapping:
    # relative
    from .deserialize import _deserialize

    df_tuple = _deserialize(blob, from_bytes=True)
    df_type_bytes, df_kv_bytes = df_tuple[0], df_tuple[1]
    df_type = _deserialize(df_type_bytes, from_bytes=True)
    mapping: dict = defaultdict(df_type)

    pairs = get_deserialized_kv_pairs(blob=df_kv_bytes)
    mapping.update(pairs)

    return mapping


def serialize_enum(enum: Enum) -> bytes:
    # relative
    from .serialize import _serialize

    return cast(bytes, _serialize(enum.value, to_bytes=True))


def deserialize_enum(enum_type: type, enum_buf: bytes) -> Enum:
    # relative
    from .deserialize import _deserialize

    enum_value = _deserialize(enum_buf, from_bytes=True)
    return enum_type(enum_value)


def serialize_type(serialized_type: type) -> bytes:
    # relative
    from ..util.util import full_name_with_qualname

    fqn = full_name_with_qualname(klass=serialized_type)
    module_parts = fqn.split(".")
    return ".".join(module_parts).encode()


def deserialize_type(type_blob: bytes) -> type:
    deserialized_type = type_blob.decode()
    module_parts = deserialized_type.split(".")
    klass = module_parts.pop()
    klass = "None" if klass == "NoneType" else klass
    exception_type = getattr(sys.modules[".".join(module_parts)], klass)
    return exception_type


TPath = TypeVar("TPath", bound=PurePath)


def serialize_path(path: PurePath) -> bytes:
    # relative
    from .serialize import _serialize

    return cast(bytes, _serialize(str(path), to_bytes=True))


def deserialize_path(path_type: type[TPath], buf: bytes) -> TPath:
    # relative
    from .deserialize import _deserialize

    path: str = _deserialize(buf, from_bytes=True)
    return path_type(path)


# bit_length + 1 for signed
recursive_serde_register(
    int,
    serialize=lambda x: x.to_bytes((x.bit_length() + 7) // 8 + 1, "big", signed=True),
    deserialize=lambda x_bytes: int.from_bytes(x_bytes, "big", signed=True),
)

recursive_serde_register(
    float,
    serialize=lambda x: x.hex().encode(),
    deserialize=lambda x: float.fromhex(x.decode()),
)

recursive_serde_register(bytes, serialize=lambda x: x, deserialize=lambda x: x)

recursive_serde_register(
    str, serialize=lambda x: x.encode(), deserialize=lambda x: x.decode()
)

recursive_serde_register(
    list,
    serialize=serialize_iterable,
    deserialize=functools.partial(deserialize_iterable, list),
)

recursive_serde_register(
    tuple,
    serialize=serialize_iterable,
    deserialize=functools.partial(deserialize_iterable, tuple),
)

recursive_serde_register(
    dict, serialize=serialize_kv, deserialize=functools.partial(deserialize_kv, dict)
)

recursive_serde_register(
    defaultdict,
    serialize=serialize_defaultdict,
    deserialize=deserialize_defaultdict,
)

recursive_serde_register(
    OrderedDict,
    serialize=serialize_kv,
    deserialize=functools.partial(deserialize_kv, OrderedDict),
)

recursive_serde_register(
    type(None), serialize=lambda _: b"1", deserialize=lambda _: None
)

recursive_serde_register(
    bool,
    serialize=lambda x: b"1" if x else b"0",
    deserialize=lambda x: False if x == b"0" else True,
)

recursive_serde_register(
    set,
    serialize=serialize_iterable,
    deserialize=functools.partial(deserialize_iterable, set),
)

recursive_serde_register(
    weakref.WeakSet,
    serialize=serialize_iterable,
    deserialize=functools.partial(deserialize_iterable, weakref.WeakSet),
)

recursive_serde_register(
    frozenset,
    serialize=serialize_iterable,
    deserialize=functools.partial(deserialize_iterable, frozenset),
)

recursive_serde_register(
    complex,
    serialize=lambda x: serialize_iterable((x.real, x.imag)),
    deserialize=lambda x: complex(*deserialize_iterable(tuple, x)),
)

recursive_serde_register(
    range,
    serialize=lambda x: serialize_iterable((x.start, x.stop, x.step)),
    deserialize=lambda x: range(*deserialize_iterable(tuple, x)),
)


recursive_serde_register(
    slice,
    serialize=lambda x: serialize_iterable((x.start, x.stop, x.step)),
    deserialize=lambda x: slice(*deserialize_iterable(tuple, x)),
)

recursive_serde_register(
    slice,
    serialize=lambda x: serialize_iterable((x.start, x.stop, x.step)),
    deserialize=lambda x: slice(*deserialize_iterable(tuple, x)),
)

recursive_serde_register(type, serialize=serialize_type, deserialize=deserialize_type)
recursive_serde_register(
    MappingProxyType,
    serialize=serialize_kv,
    deserialize=functools.partial(deserialize_kv, MappingProxyType),
)


for __path_type in (
    PurePath,
    pathlib.PurePosixPath,
    pathlib.PureWindowsPath,
    pathlib.Path,
    pathlib.PosixPath,
    pathlib.WindowsPath,
):
    recursive_serde_register(
        __path_type,
        serialize=serialize_path,
        deserialize=functools.partial(deserialize_path, __path_type),
    )


def serialize_generic_alias(serialized_type: _GenericAlias) -> bytes:
    # relative
    from ..util.util import full_name_with_name
    from .serialize import _serialize

    fqn = full_name_with_name(klass=serialized_type)
    module_parts = fqn.split(".")

    obj_dict = {
        "path": ".".join(module_parts),
        "__origin__": serialized_type.__origin__,
        "__args__": serialized_type.__args__,
    }
    if hasattr(serialized_type, "_paramspec_tvars"):
        obj_dict["_paramspec_tvars"] = serialized_type._paramspec_tvars
    return _serialize(obj_dict, to_bytes=True)


def deserialize_generic_alias(type_blob: bytes) -> type:
    # relative
    from .deserialize import _deserialize

    obj_dict = _deserialize(type_blob, from_bytes=True)
    deserialized_type = obj_dict.pop("path")
    module_parts = deserialized_type.split(".")
    klass = module_parts.pop()
    type_constructor = getattr(sys.modules[".".join(module_parts)], klass)
    # does this apply to all _SpecialForm?

    # Note: Some typing constructors are callable while
    # some use custom __getitem__ implementations
    # to initialize the type 😭

    try:
        return type_constructor(**obj_dict)
    except TypeError:
        _args = obj_dict["__args__"]
        # Again not very consistent 😭
        if type_constructor == Optional:
            _args = _args[0]
        return type_constructor[_args]
    except Exception as e:
        raise e


# 🟡 TODO 5: add tests and all typing options for signatures
def recursive_serde_register_type(t: type, serialize_attrs: list | None = None) -> None:
    if (isinstance(t, type) and issubclass(t, _GenericAlias)) or issubclass(
        type(t), _GenericAlias
    ):
        recursive_serde_register(
            t,
            serialize=serialize_generic_alias,
            deserialize=deserialize_generic_alias,
            serialize_attrs=serialize_attrs,
        )
    else:
        recursive_serde_register(
            t,
            serialize=serialize_type,
            deserialize=deserialize_type,
            serialize_attrs=serialize_attrs,
        )


def serialize_union_type(serialized_type: UnionType) -> bytes:
    # relative
    from .serialize import _serialize

    return _serialize(serialized_type.__args__, to_bytes=True)


def deserialize_union_type(type_blob: bytes) -> type:
    # relative
    from .deserialize import _deserialize

    args = _deserialize(type_blob, from_bytes=True)
    return functools.reduce(lambda x, y: x | y, args)


recursive_serde_register(
    UnionType,
    serialize=serialize_union_type,
    deserialize=deserialize_union_type,
)

recursive_serde_register_type(_SpecialForm)
recursive_serde_register_type(_GenericAlias)
recursive_serde_register_type(Union)
recursive_serde_register_type(TypeVar)

recursive_serde_register_type(
    _UnionGenericAlias,
    serialize_attrs=[
        "__parameters__",
        "__slots__",
        "_inst",
        "_name",
        "__args__",
        "__module__",
        "__origin__",
    ],
)
recursive_serde_register_type(_SpecialGenericAlias)
recursive_serde_register_type(GenericAlias)

recursive_serde_register_type(Any)
recursive_serde_register_type(EnumMeta)
