# stdlib
from collections import OrderedDict
from enum import Enum
import functools
import sys
from typing import Iterable
from typing import Mapping
from typing import cast

# relative
from ....proto.core.common.recursive_serde_pb2 import Iterable as Iterable_PB
from ....proto.core.common.recursive_serde_pb2 import KVIterable as KVIterable_PB
from .recursive import recursive_serde_register


def serialize_iterable(iterable: Iterable) -> bytes:
    # relative
    from .capnp import create_protobuf_magic_header
    from .serialize import _serialize

    message = Iterable_PB()
    message.magic_header = create_protobuf_magic_header()

    for it in iterable:
        message.values.append(_serialize(it, to_bytes=True))

    return message.SerializeToString()


def deserialize_iterable(iterable_type: type, blob: bytes) -> Iterable:
    # relative
    from .deserialize import _deserialize

    message = Iterable_PB()
    message.ParseFromString(blob)

    values = []
    for element in message.values:
        values.append(_deserialize(element, from_bytes=True))

    return iterable_type(values)


def serialze_kv(map: Mapping) -> bytes:
    # relative
    from .capnp import create_protobuf_magic_header
    from .serialize import _serialize

    message = KVIterable_PB()
    message.magic_header = create_protobuf_magic_header()

    for k, v in map.items():
        message.keys.append(_serialize(k, to_bytes=True))
        message.values.append(_serialize(v, to_bytes=True))

    return message.SerializeToString()


def deserialize_kv(mapping_type: type, blob: bytes) -> Mapping:
    # relative
    from .deserialize import _deserialize

    message = KVIterable_PB()
    message.ParseFromString(blob)

    pairs = []
    for k, v in zip(message.keys, message.values):
        pairs.append(
            (_deserialize(k, from_bytes=True), _deserialize(v, from_bytes=True))
        )

    return mapping_type(pairs)


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
    from ....lib.util import full_name_with_qualname

    fqn = full_name_with_qualname(klass=serialized_type)
    module_parts = fqn.split(".")
    _ = module_parts.pop()  # remove incorrect .type ending
    module_parts.append(serialized_type.__name__)
    return ".".join(module_parts).encode()


def deserialize_type(type_blob: bytes) -> type:
    deserialized_type = type_blob.decode()
    module_parts = deserialized_type.split(".")
    klass = module_parts.pop()
    exception_type = getattr(sys.modules[".".join(module_parts)], klass)
    return exception_type


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
    dict, serialize=serialze_kv, deserialize=functools.partial(deserialize_kv, dict)
)

recursive_serde_register(
    OrderedDict,
    serialize=serialze_kv,
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


recursive_serde_register(type, serialize=serialize_type, deserialize=deserialize_type)
