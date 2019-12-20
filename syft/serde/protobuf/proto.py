"""
This file exists to translate python classes to and from Protobuf messages.
The reason for this is to have stable serialization protocol that can be used
not only by PySyft but also in other languages.

https://github.com/OpenMined/syft-proto (`syft_proto` module) is included as
a dependency in setup.py.
"""
import torch

from google.protobuf.empty_pb2 import Empty

from syft.messaging.message import ObjectMessage
from syft.messaging.message import Operation
from syft.frameworks.torch.tensors.interpreters.additive_shared import AdditiveSharingTensor

from syft_proto.frameworks.torch.tensors.interpreters.v1.additive_shared_pb2 import (
    AdditiveSharingTensor as AdditiveSharingTensorPB,
)
from syft_proto.messaging.v1.message_pb2 import ObjectMessage as ObjectMessagePB
from syft_proto.messaging.v1.message_pb2 import OperationMessage as OperationMessagePB
from syft_proto.generic.v1.tensor_pb2 import Tensor as TensorPB


MAP_PYTHON_TO_PROTOBUF_CLASSES = {
    ObjectMessage: ObjectMessagePB,
    Operation: OperationMessagePB,
    torch.Tensor: TensorPB,
    AdditiveSharingTensor: AdditiveSharingTensorPB,
    type(None): Empty,
}

MAP_PROTOBUF_TO_PYTHON_CLASSES = {}

for key, value in MAP_PYTHON_TO_PROTOBUF_CLASSES.items():
    MAP_PROTOBUF_TO_PYTHON_CLASSES[value] = key
