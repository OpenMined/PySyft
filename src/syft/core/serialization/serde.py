from .serializable import Serializable
from .store import serialization_store

def serialize(obj):
    current_type = type(obj)

    syft_message = SyftMessagePB()

    if type(obj) in serialization_store.serde_types:
        schema = current_type.to_protobuf(obj)
    else:
        raise RuntimeError(f"No protobuf schema found for object of type {current_type}.")

    #TODO create
    binary = syft_message.SerializeToString()

    #TODO: compression step

    return binary

def deserialize(bin_data):
    pass