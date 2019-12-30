"""
This file exists to translate python classes to and from Protobuf messages.
The reason for this is to have stable serialization protocol that can be used
not only by PySyft but also in other languages.

https://github.com/OpenMined/syft-proto (`syft_proto` module) is included as
a dependency in setup.py.
"""
import torch

from google.protobuf.empty_pb2 import Empty
from syft_proto.types.torch.v1.tensor_pb2 import TorchTensor as TorchTensorPB
from syft_proto.types.torch.v1.device_pb2 import Device as DevicePB


MAP_PYTHON_TO_PROTOBUF_CLASSES = {
    type(None): Empty,
    torch.Tensor: TorchTensorPB,
    torch.device: DevicePB,
}

MAP_PROTOBUF_TO_PYTHON_CLASSES = {}

for key, value in MAP_PYTHON_TO_PROTOBUF_CLASSES.items():
    MAP_PROTOBUF_TO_PYTHON_CLASSES[value] = key
