# stdlib
import builtins
from collections import OrderedDict
import functools
from typing import Iterable
from typing import Mapping

# relative
from ....proto.core.common.recursive_serde_pb2 import Iterable as Iterable_PB
from ....proto.core.common.recursive_serde_pb2 import KVIterable as KVIterable_PB
from .recursive import recursive_serde_register
from .recursive import rs_bytes2object
from .recursive import rs_object2proto


def serialize_iterable(iterable: Iterable) -> bytes:
    message = Iterable_PB()

    for it in iterable:
        message.data.append(rs_object2proto(it).SerializeToString())

    return message.SerializeToString()


def deserialize_iterable(iterable_type: type, blob: bytes) -> Iterable:
    message = Iterable_PB()
    message.ParseFromString(blob)

    data = []
    for element in message.data:
        data.append(rs_bytes2object(element))

    return iterable_type(data)


def serialze_kv(map: Mapping) -> bytes:
    message = KVIterable_PB()

    for k, v in map.items():
        message.keys.append(rs_object2proto(k).SerializeToString())
        message.values.append(rs_object2proto(v).SerializeToString())

    return message.SerializeToString()


def deserialize_kv(mapping_type: type, blob: bytes) -> Mapping:
    message = KVIterable_PB()
    message.ParseFromString(blob)

    data = []
    for k, v in zip(message.keys, message.values):
        data.append((rs_bytes2object(k), rs_bytes2object(v)))

    return mapping_type(data)


recursive_serde_register(
    int,
    serialize=lambda x: x.to_bytes((x.bit_length() + 7) // 8, "big"),
    deserialize=lambda x_bytes: int.from_bytes(x_bytes, "big"),
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
    type(None), serialize=lambda _: b"", deserialize=lambda _: None
)

recursive_serde_register(
    bool,
    serialize=lambda x: b"1" if x else b"0",
    deserialize=lambda x: False if x == b"0" else True,
)
