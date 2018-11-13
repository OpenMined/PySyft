"""
This file exists to provide one common place for all serialization to occur
regardless of framework. As msgpack only supports basic types and binary formats
every type must be first be converted to one of these types.
"""
import pickle
import torch
import msgpack


def serialize(obj):
    simple_objects = simplify(obj)
    return base_serialize(simple_objects)


def deserialize(bin):
    simple_objects = base_deserialize(bin)
    return detail(simple_objects)


def base_serialize(obj):
    return msgpack.dumps(obj)


def base_deserialize(string):
    return msgpack.loads(string)


def simplify_torch_tensor(tensor):
    return pickle.dumps(tensor)


def detail_torch_tensor(tensor):
    return pickle.loads(tensor)


def simplify_collection(my_collection):
    # Step 0: get collection type for later use and itialize empty list
    my_type = type(my_collection)
    pieces = list()

    # Step 1: serialize each part of the collection
    for part in my_collection:
        pieces.append(simplify(part))

    # Step 2: convert back to original type and return serialization
    return my_type(pieces)


def detail_collection(my_collection):

    pieces = list()

    # Step 1: deserialize each part of the collection
    for part in my_collection:
        pieces.append(detail(part))

    return pieces


def simplify(obj):
    t = type(obj)
    if t in simplifiers:
        return simplifiers[t](obj)
    return obj


simplifiers = {}
simplifiers[torch.Tensor] = simplify_torch_tensor
simplifiers[tuple] = simplify_collection
simplifiers[list] = simplify_collection


def detail(obj):
    t = type(obj)
    if t in detailers:
        return detailers[t](obj)
    return obj


detailers = {}
detailers[tuple] = detail_collection
detailers[list] = detail_collection
detailers[bytes] = detail_torch_tensor
