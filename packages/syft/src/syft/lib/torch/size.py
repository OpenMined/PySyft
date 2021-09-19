# third party
import torch

# relative
from ...core.common.serde.serializable import serializable
from ...proto.lib.torch.size_pb2 import Size as TorchSize_PB


def protobuf_torch_size_serializer(torch_size: torch.Size) -> TorchSize_PB:
    serialized_size = TorchSize_PB(data=torch_size)
    return serialized_size


def protobuf_torch_size_deserializer(proto_size: TorchSize_PB) -> torch.Size:
    torch_size = torch.Size(proto_size.data)
    return torch_size


serializable(generate_wrapper=True)(
    wrapped_type=torch.Size,
    import_path="torch.Size",
    protobuf_scheme=TorchSize_PB,
    type_object2proto=protobuf_torch_size_serializer,
    type_proto2object=protobuf_torch_size_deserializer,
)
