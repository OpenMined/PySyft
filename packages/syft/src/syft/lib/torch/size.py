# third party
import torch

# syft relative
from ...generate_wrapper import GenerateWrapper
from ...proto.lib.python.tuple_pb2 import Tuple as Tuple_PB
from ..python import Tuple
from ..python.primitive_factory import PrimitiveFactory


def protobuf_torch_size_serializer(torch_size: torch.Size) -> Tuple_PB:
    proto_size = PrimitiveFactory.generate_primitive(value=tuple(torch_size))
    serialized_size = proto_size._object2proto()
    return serialized_size


def protobuf_torch_size_deserializer(proto_size: Tuple_PB) -> torch.Size:
    torch_size = torch.Size(Tuple._proto2object(proto=proto_size))
    return torch_size


GenerateWrapper(
    wrapped_type=torch.Size,
    import_path="torch.Size",
    protobuf_scheme=Tuple_PB,
    type_object2proto=protobuf_torch_size_serializer,
    type_proto2object=protobuf_torch_size_deserializer,
)
