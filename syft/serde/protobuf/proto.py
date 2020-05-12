"""
This file exists to translate python classes to and from Protobuf messages.
The reason for this is to have stable serialization protocol that can be used
not only by PySyft but also in other languages.

https://github.com/OpenMined/syft-proto (`syft_proto` module) is included as
a dependency in setup.py.
"""
import torch

from google.protobuf.empty_pb2 import Empty

from syft_proto.execution.v1.type_wrapper_pb2 import InputTypeDescriptor as InputTypeDescriptorPB
from syft_proto.types.torch.v1.device_pb2 import Device as DevicePB
from syft_proto.types.torch.v1.parameter_pb2 import Parameter as ParameterPB
from syft_proto.types.torch.v1.size_pb2 import Size as SizePB
from syft_proto.types.torch.v1.tensor_pb2 import TorchTensor as TorchTensorPB
from syft_proto.types.torch.v1.script_module_pb2 import ScriptModule as ScriptModulePB
from syft_proto.types.torch.v1.script_function_pb2 import ScriptFunction as ScriptFunctionPB
from syft_proto.types.torch.v1.traced_module_pb2 import TracedModule as TracedModulePB
from syft.serde.syft_serializable import SyftSerializable, get_protobuf_subclasses

MAP_PYTHON_TO_PROTOBUF_CLASSES = {
    type(None): Empty,
    type: InputTypeDescriptorPB,
    # Torch types
    torch.Tensor: TorchTensorPB,
    torch.device: DevicePB,
    torch.nn.Parameter: ParameterPB,
    torch.jit.ScriptModule: ScriptModulePB,
    torch.jit.ScriptFunction: ScriptFunctionPB,
    torch.jit.TopLevelTracedModule: TracedModulePB,
    torch.Size: SizePB,
}


def set_protobuf_id(field, id):
    if type(id) == type("str"):
        field.id_str = id
    else:
        field.id_int = id


def get_protobuf_id(field):
    return getattr(field, field.WhichOneof("id"))
