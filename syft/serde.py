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

def serialize_torch_tensor(tensor):
    return pickle.dumps(tensor)

def deserialize_torch_tensor(tensor):
    return pickle.loads(tensor)

def serialize_collection(my_tuple):
    # Step 0: create initial message
    pieces = list()

    # Step 1: serialize each part of the tuple
    for part in my_tuple:
        pieces.append(simplify(part))

    # Step 2: return serialization
    return pieces

def deserialize_collection(my_tuple):

    pieces = list()

    # Step 1: deserialize each part of the tuple
    for part in my_tuple:
        pieces.append(detail(part))

    return pieces

def simplify(obj):
    t = type(obj)
    if t in simplifiers:
        return simplifiers[t](obj)
    return obj

simplifiers = {}
simplifiers[torch.Tensor] = serialize_torch_tensor
simplifiers[tuple] = serialize_collection

def detail(obj):
    t = type(obj)
    if t in detailers:
        return detailers[t](obj)
    return obj

detailers = {}
detailers[tuple] = deserialize_collection
detailers[list] = deserialize_collection
detailers[bytes] = deserialize_torch_tensor


