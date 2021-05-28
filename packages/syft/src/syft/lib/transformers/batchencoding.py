# syft relative
from ...lib.python.dict import Dict
from ... import serialize, deserialize
from ...proto.lib.python.dict_pb2 import Dict as Dict_PB
from ...generate_wrapper import GenerateWrapper

# Third party
from transformers.tokenization_utils_base import BatchEncoding


def object2proto(obj: object) -> Dict_PB:
    # Wrap in sy Dict to make nested serialization easy.
    obj_dict = Dict(obj.data)
    return serialize(obj_dict)


def proto2object(proto: Dict_PB) -> BatchEncoding:
    obj_dict = deserialize(proto).data
    return BatchEncoding(data=obj_dict)


GenerateWrapper(
    wrapped_type=BatchEncoding,
    import_path="transformers.tokenization_utils_base.BatchEncoding",
    protobuf_scheme=Dict_PB,
    type_object2proto=object2proto,
    type_proto2object=proto2object,
)
