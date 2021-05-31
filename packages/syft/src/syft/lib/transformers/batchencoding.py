# syft relative
from ...proto.lib.python.dict_pb2 import Dict as Dict_PB
from ...lib.python.dict import Dict
from ...lib.python.primitive_factory import PrimitiveFactory
from ...generate_wrapper import GenerateWrapper

# Third party
import transformers
from transformers.tokenization_utils_base import BatchEncoding

def object2proto(obj: BatchEncoding) -> Dict_PB:
    obj_dict = PrimitiveFactory.generate_primitive(value=obj.data)
    return obj_dict._object2proto()


def proto2object(proto: Dict_PB) -> BatchEncoding:
    obj_dict = Dict._proto2object(proto=proto)
    return BatchEncoding(data=obj_dict.data)

GenerateWrapper(
    wrapped_type=BatchEncoding,
    import_path="transformers.tokenization_utils_base.BatchEncoding",
    protobuf_scheme=Dict_PB,
    type_object2proto=object2proto,
    type_proto2object=proto2object,
)
