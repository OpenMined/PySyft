import pickle
import torch
import msgpack

# High Level Public Functions (these are the ones you use)

def serialize(obj):
    simple_objects = _simplify(obj)
    return msgpack.dumps(simple_objects)

def deserialize(bin):
    simple_objects = msgpack.loads(bin)
    return _detail(simple_objects)

# Torch Tensor

def _simplify_torch_tensor(tensor):
    return pickle.dumps(tensor)

def _detail_torch_tensor(tensor):
    return pickle.loads(tensor)

# Collections (list, set, tuple, etc.)

def _simplify_collection(my_collection):
    # Step 0: get collection type for later use and itialize empty list
    my_type = type(my_collection)
    pieces = list()

    # Step 1: serialize each part of the collection
    for part in my_collection:
        pieces.append(_simplify(part))

    # Step 2: convert back to original type and return serialization
    return my_type(pieces)

def _detail_collection(my_collection):

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
