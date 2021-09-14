# third party
from transformers.tokenization_utils_base import BatchEncoding

# relative
from ...core.common.serde.serializable import serializable
from ...proto.lib.transformers.batch_encoding_pb2 import (
    BatchEncoding as BatchEncoding_PB,
)
from ..python.dict import Dict
from ..python.primitive_factory import PrimitiveFactory


def object2proto(obj: BatchEncoding) -> BatchEncoding_PB:
    obj_dict = PrimitiveFactory.generate_primitive(value=obj.data)
    return BatchEncoding_PB(
        id=obj_dict.id._object2proto(), data=obj_dict._object2proto()
    )


def proto2object(proto: BatchEncoding_PB) -> BatchEncoding:
    obj_dict = Dict._proto2object(proto=proto.data)
    return BatchEncoding(data=obj_dict.data)


serializable(generate_wrapper=True)(
    wrapped_type=BatchEncoding,
    import_path="transformers.tokenization_utils_base.BatchEncoding",
    protobuf_scheme=BatchEncoding_PB,
    type_object2proto=object2proto,
    type_proto2object=proto2object,
)
