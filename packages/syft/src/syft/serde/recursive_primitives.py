# stdlib
from abc import ABCMeta
from collections import OrderedDict
from collections import defaultdict
from collections.abc import Collection
from collections.abc import Iterable
from collections.abc import Mapping
from enum import Enum
from enum import EnumMeta
import functools
import inspect
import pathlib
from pathlib import PurePath
import sys
import tempfile
from types import MappingProxyType
from types import UnionType
import typing
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
from ..types.syft_object_registry import SyftObjectRegistry
from .capnp import get_capnp_schema
from .recursive import chunk_bytes
from .recursive import combine_bytes
from .recursive import recursive_serde_register
from .util import compatible_with_large_file_writes_capnp

iterable_schema = get_capnp_schema("iterable.capnp").Iterable
kv_iterable_schema = get_capnp_schema("kv_iterable.capnp").KVIterable


def serialize_iterable(iterable: Collection) -> bytes:
    # relative
    from .serialize import _serialize

    message = iterable_schema.new_message()

    message.init("values", len(iterable))

    for idx, it in enumerate(iterable):
        # serialized = _serialize(it, to_bytes=True)
        chunk_bytes(it, lambda x: _serialize(x, to_bytes=True), idx, message.values)

    if compatible_with_large_file_writes_capnp(message):
        with tempfile.TemporaryFile() as tmp_file:
            # Write data to a file to save RAM
            message.write(tmp_file)
            del message
            tmp_file.seek(0)
            res = tmp_file.read()
            return res
    else:
        res = message.to_bytes()
        del message
        return res


def deserialize_iterable(iterable_type: type, blob: bytes) -> Collection:
    # relative
    from .deserialize import _deserialize

    MAX_TRAVERSAL_LIMIT = 2**64 - 1

    with iterable_schema.from_bytes(
        blob, traversal_limit_in_words=MAX_TRAVERSAL_LIMIT
    ) as msg:
        values = [
            _deserialize(combine_bytes(element), from_bytes=True)
            for element in msg.values
        ]

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
        # serialized = _serialize(v, to_bytes=True)
        chunk_bytes(v, lambda x: _serialize(x, to_bytes=True), index, message.values)

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


def serialize_type(_type_to_serialize: type) -> bytes:
    # relative
    type_to_serialize = typing.get_origin(_type_to_serialize) or _type_to_serialize
    canonical_name, version = SyftObjectRegistry.get_identifier_for_type(
        type_to_serialize
    )
    return f"{canonical_name}:{version}".encode()

    # from ..util.util import full_name_with_qualname

    # fqn = full_name_with_qualname(klass=serialized_type)
    # module_parts = fqn.split(".")
    # return ".".join(module_parts).encode()


def deserialize_type(type_blob: bytes) -> type:
    deserialized_type = type_blob.decode()
    canonical_name, version = deserialized_type.split(":", 1)
    return SyftObjectRegistry.get_serde_class(canonical_name, int(version))

    # module_parts = deserialized_type.split(".")
    # klass = module_parts.pop()
    # klass = "None" if klass == "NoneType" else klass
    # exception_type = getattr(sys.modules[".".join(module_parts)], klass)
    # return exception_type


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
    canonical_name="int",
    version=1,
)

recursive_serde_register(
    float,
    serialize=lambda x: x.hex().encode(),
    deserialize=lambda x: float.fromhex(x.decode()),
    canonical_name="float",
    version=1,
)

recursive_serde_register(
    bytes,
    serialize=lambda x: x,
    deserialize=lambda x: x,
    canonical_name="bytes",
    version=1,
)

recursive_serde_register(
    str,
    serialize=lambda x: x.encode(),
    deserialize=lambda x: x.decode(),
    canonical_name="str",
    version=1,
)

recursive_serde_register(
    list,
    serialize=serialize_iterable,
    deserialize=functools.partial(deserialize_iterable, list),
    canonical_name="list",
    version=1,
)

recursive_serde_register(
    tuple,
    serialize=serialize_iterable,
    deserialize=functools.partial(deserialize_iterable, tuple),
    canonical_name="tuple",
    version=1,
)

recursive_serde_register(
    dict,
    serialize=serialize_kv,
    deserialize=functools.partial(deserialize_kv, dict),
    canonical_name="dict",
    version=1,
)

recursive_serde_register(
    defaultdict,
    serialize=serialize_defaultdict,
    deserialize=deserialize_defaultdict,
    canonical_name="defaultdict",
    version=1,
)

recursive_serde_register(
    OrderedDict,
    serialize=serialize_kv,
    deserialize=functools.partial(deserialize_kv, OrderedDict),
    canonical_name="OrderedDict",
    version=1,
)

recursive_serde_register(
    type(None),
    serialize=lambda _: b"1",
    deserialize=lambda _: None,
    canonical_name="NoneType",
    version=1,
)

recursive_serde_register(
    bool,
    serialize=lambda x: b"1" if x else b"0",
    deserialize=lambda x: False if x == b"0" else True,
    canonical_name="bool",
    version=1,
)

recursive_serde_register(
    set,
    serialize=serialize_iterable,
    deserialize=functools.partial(deserialize_iterable, set),
    canonical_name="set",
    version=1,
)

recursive_serde_register(
    weakref.WeakSet,
    serialize=serialize_iterable,
    deserialize=functools.partial(deserialize_iterable, weakref.WeakSet),
    canonical_name="WeakSet",
    version=1,
)

recursive_serde_register(
    frozenset,
    serialize=serialize_iterable,
    deserialize=functools.partial(deserialize_iterable, frozenset),
    canonical_name="frozenset",
    version=1,
)

recursive_serde_register(
    complex,
    serialize=lambda x: serialize_iterable((x.real, x.imag)),
    deserialize=lambda x: complex(*deserialize_iterable(tuple, x)),
    canonical_name="complex",
    version=1,
)

recursive_serde_register(
    range,
    serialize=lambda x: serialize_iterable((x.start, x.stop, x.step)),
    deserialize=lambda x: range(*deserialize_iterable(tuple, x)),
    canonical_name="range",
    version=1,
)

recursive_serde_register(
    slice,
    serialize=lambda x: serialize_iterable((x.start, x.stop, x.step)),
    deserialize=lambda x: slice(*deserialize_iterable(tuple, x)),
    canonical_name="slice",
    version=1,
)

recursive_serde_register(
    type,
    serialize=serialize_type,
    deserialize=deserialize_type,
    canonical_name="type",
    version=1,
)

recursive_serde_register(
    MappingProxyType,
    serialize=serialize_kv,
    deserialize=functools.partial(deserialize_kv, MappingProxyType),
    canonical_name="MappingProxyType",
    version=1,
)

recursive_serde_register(
    PurePath,
    serialize=serialize_path,
    deserialize=functools.partial(deserialize_path, PurePath),
    canonical_name="PurePath",
    version=1,
)

for __path_type in (
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
        canonical_name=f"pathlib_{__path_type.__name__}",
        version=1,
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
def recursive_serde_register_type(
    t: type,
    serialize_attrs: list | None = None,
    canonical_name: str | None = None,
    version: int | None = None,
) -> None:
    # former case is for instance for _GerericAlias itself or UnionGenericAlias
    # Latter case is true for for instance List[str], which is currently not used
    if (isinstance(t, type) and issubclass(t, _GenericAlias)) or issubclass(
        type(t), _GenericAlias
    ):
        recursive_serde_register(
            t,
            serialize=serialize_generic_alias,
            deserialize=deserialize_generic_alias,
            serialize_attrs=serialize_attrs,
            canonical_name=canonical_name,
            version=version,
        )
    else:
        recursive_serde_register(
            t,
            serialize=serialize_type,
            deserialize=deserialize_type,
            serialize_attrs=serialize_attrs,
            canonical_name=canonical_name,
            version=version,
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


def serialize_union(serialized_type: UnionType) -> bytes:
    return b""


def deserialize_union(type_blob: bytes) -> type:  # type: ignore
    return Union  # type: ignore


def serialize_typevar(serialized_type: TypeVar) -> bytes:
    return f"{serialized_type.__name__}".encode()


def deserialize_typevar(type_blob: bytes) -> type:
    name = type_blob.decode()
    return TypeVar(name=name)  # type: ignore


def serialize_any(serialized_type: TypeVar) -> bytes:
    return b""


def deserialize_any(type_blob: bytes) -> type:  # type: ignore
    return Any  # type: ignore


recursive_serde_register(
    UnionType,
    serialize=serialize_union_type,
    deserialize=deserialize_union_type,
    canonical_name="UnionType",
    version=1,
)

recursive_serde_register_type(_SpecialForm, canonical_name="_SpecialForm", version=1)
recursive_serde_register_type(_GenericAlias, canonical_name="_GenericAlias", version=1)
recursive_serde_register(
    Union,
    canonical_name="Union",
    serialize=serialize_union,
    deserialize=deserialize_union,
    version=1,
)
recursive_serde_register(
    TypeVar,
    canonical_name="TypeVar",
    serialize=serialize_typevar,
    deserialize=deserialize_typevar,
    version=1,
)
recursive_serde_register(
    Any,
    canonical_name="Any",
    serialize=serialize_any,
    deserialize=deserialize_any,
    version=1,
)

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
    canonical_name="_UnionGenericAlias",
    version=1,
)
recursive_serde_register_type(
    _SpecialGenericAlias, canonical_name="_SpecialGenericAlias", version=1
)
recursive_serde_register_type(GenericAlias, canonical_name="GenericAlias", version=1)

# recursive_serde_register_type(Any, canonical_name="Any", version=1)
recursive_serde_register_type(EnumMeta, canonical_name="EnumMeta", version=1)

recursive_serde_register_type(ABCMeta, canonical_name="ABCMeta", version=1)

recursive_serde_register_type(inspect._empty, canonical_name="inspect_empty", version=1)
