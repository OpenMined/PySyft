"""
This file exists to translate python classes to and from Protobuf messages.
The reason for this is to have stable serialization protocol that can be used
not only by PySyft but also in other languages.

https://github.com/OpenMined/syft-proto (`syft_proto` module) is included as
a dependency in setup.py.
"""
import torch

from syft.execution.communication import CommunicationAction
from syft.execution.computation import ComputationAction
from syft.execution.placeholder import PlaceHolder
from syft.execution.placeholder_id import PlaceholderId
from syft.execution.plan import Plan
from syft.execution.protocol import Protocol
from syft.execution.role import Role
from syft.execution.state import State
from syft.frameworks.torch.tensors.interpreters.additive_shared import AdditiveSharingTensor
from syft.generic.pointers.pointer_tensor import PointerTensor
from syft.messaging.message import TensorCommandMessage
from syft.messaging.message import ObjectMessage

from google.protobuf.empty_pb2 import Empty
from syft_proto.execution.v1.communication_action_pb2 import (
    CommunicationAction as CommunicationActionPB,
)
from syft_proto.execution.v1.computation_action_pb2 import ComputationAction as ComputationActionPB
from syft_proto.execution.v1.plan_pb2 import Plan as PlanPB
from syft_proto.execution.v1.protocol_pb2 import Protocol as ProtocolPB
from syft_proto.execution.v1.role_pb2 import Role as RolePB
from syft_proto.execution.v1.state_pb2 import State as StatePB
from syft_proto.frameworks.torch.tensors.interpreters.v1.additive_shared_pb2 import (
    AdditiveSharingTensor as AdditiveSharingTensorPB,
)
from syft_proto.execution.v1.placeholder_pb2 import Placeholder as PlaceholderPB
from syft_proto.generic.pointers.v1.pointer_tensor_pb2 import PointerTensor as PointerTensorPB
from syft_proto.messaging.v1.message_pb2 import ObjectMessage as ObjectMessagePB
from syft_proto.messaging.v1.message_pb2 import TensorCommandMessage as CommandMessagePB
from syft_proto.execution.v1.placeholder_id_pb2 import PlaceholderId as PlaceholderIdPB
from syft_proto.types.torch.v1.device_pb2 import Device as DevicePB
from syft_proto.types.torch.v1.parameter_pb2 import Parameter as ParameterPB
from syft_proto.types.torch.v1.size_pb2 import Size as SizePB
from syft_proto.types.torch.v1.tensor_pb2 import TorchTensor as TorchTensorPB
from syft_proto.types.torch.v1.script_module_pb2 import ScriptModule as ScriptModulePB
from syft_proto.types.torch.v1.script_function_pb2 import ScriptFunction as ScriptFunctionPB
from syft_proto.types.torch.v1.traced_module_pb2 import TracedModule as TracedModulePB


MAP_PYTHON_TO_PROTOBUF_CLASSES = {
    type(None): Empty,
    # Torch types
    torch.Tensor: TorchTensorPB,
    torch.device: DevicePB,
    torch.nn.Parameter: ParameterPB,
    torch.jit.ScriptModule: ScriptModulePB,
    torch.jit.ScriptFunction: ScriptFunctionPB,
    torch.jit.TopLevelTracedModule: TracedModulePB,
    torch.Size: SizePB,
    # Syft types
    AdditiveSharingTensor: AdditiveSharingTensorPB,
    PlaceholderId: PlaceholderIdPB,
    ObjectMessage: ObjectMessagePB,
    TensorCommandMessage: CommandMessagePB,
    CommunicationAction: CommunicationActionPB,
    ComputationAction: ComputationActionPB,
    PlaceHolder: PlaceholderPB,
    Plan: PlanPB,
    PointerTensor: PointerTensorPB,
    Protocol: ProtocolPB,
    Role: RolePB,
    State: StatePB,
}


def set_protobuf_id(field, id):
    if type(id) == type("str"):
        field.id_str = id
    else:
        field.id_int = id


def get_protobuf_id(field):
    return getattr(field, field.WhichOneof("id"))
