"""
This file exists to provide one common place for all serialization to occur
regardless of framework. As msgpack only supports basic types and binary formats
every type must be first be converted to one of these types. Thus, we've split our
functionality into two sections.
"""
from typing import Collection
import pickle
import torch
import msgpack
import lz4

# High Level Public Functions (these are the ones you use)


def serialize(obj, compress=True):

    simple_objects = _simplify(obj)
    bin = msgpack.dumps(simple_objects)

    if compress:
        return compress(bin)
    else:
        return bin


def deserialize(bin, compressed=True):

    if compressed:
        bin = decompress(bin)

    simple_objects = msgpack.loads(bin)
    return _detail(simple_objects)


# Chosen Compression Algorithm


def compress(decompressed_input_bin):
    return lz4.frame.compress(decompressed_input_bin)


def decompress(compressed_input_bin):
    return lz4.frame.decompress(compressed_input_bin)


# Torch Tensor


def _simplify_torch_tensor(tensor):
    return pickle.dumps(tensor)


def _detail_torch_tensor(tensor):
    return pickle.loads(tensor)


# Collections (list, set, tuple, etc.)


def _simplify_collection(my_collection: Collection) -> Collection:
    # Step 0: get collection type for later use and itialize empty list
    my_type = type(my_collection)
    pieces = list()

    # Step 1: serialize each part of the collection
    for part in my_collection:
        pieces.append(_simplify(part))

    # Step 2: convert back to original type and return serialization
    return my_type(pieces)


def _detail_collection(my_collection: Collection) -> Collection:

    pieces = list()

    # Step 1: deserialize each part of the collection
    for part in my_collection:
        pieces.append(_detail(part))

    return pieces


# High Level Simplification Router


def _simplify(obj):
    t = type(obj)
    if t in simplifiers:
        return simplifiers[t](obj)
    return obj


simplifiers = {}

simplifiers[torch.Tensor] = _simplify_torch_tensor
simplifiers[tuple] = _simplify_collection
simplifiers[list] = _simplify_collection


def _detail(obj):
    t = type(obj)
    if t in detailers:
        return detailers[t](obj)
    return obj


detailers = {}
detailers[tuple] = _detail_collection
detailers[list] = _detail_collection
detailers[bytes] = _detail_torch_tensor
