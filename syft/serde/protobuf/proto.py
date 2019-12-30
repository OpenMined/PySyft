"""
This file exists to translate python classes to and from Protobuf messages.
The reason for this is to have stable serialization protocol that can be used
not only by PySyft but also in other languages.

https://github.com/OpenMined/syft-proto (`syft_proto` module) is included as
a dependency in setup.py.
"""
import torch

from google.protobuf.empty_pb2 import Empty
from syft_proto.types.syft.v1.id_pb2 import Id as IdPB
from syft_proto.types.torch.v1.device_pb2 import Device as DevicePB
from syft_proto.types.torch.v1.parameter_pb2 import Parameter as ParameterPB
from syft_proto.types.torch.v1.tensor_pb2 import TorchTensor as TorchTensorPB


MAP_PYTHON_TO_PROTOBUF_CLASSES = {
    type(None): Empty,
    torch.Tensor: TorchTensorPB,
    torch.device: DevicePB,
    torch.nn.Parameter: ParameterPB,
}

MAP_PROTOBUF_TO_PYTHON_CLASSES = {}

for key, value in MAP_PYTHON_TO_PROTOBUF_CLASSES.items():
    MAP_PROTOBUF_TO_PYTHON_CLASSES[value] = key


def create_protobuf_id(id) -> IdPB:
    protobuf_id = IdPB()
    if type(id) == type("str"):
        protobuf_id.id_str = id
    else:
        protobuf_id.id_int = id
    return protobuf_id
