"""
This file exists to provide one common place for all serialisation and bufferize_ and _unbufferize
for all tensors (Torch and Numpy).
"""
import io

import torch
import pydoc

import syft
from syft.generic.abstract.syft_serializable import SyftSerializable
from syft.generic.pointers.pointer_tensor import PointerTensor
from syft.generic.abstract.tensor import initialize_tensor
from syft.workers.abstract import AbstractWorker
from syft.codes import TENSOR_SERIALIZATION

from syft.serde.protobuf.proto import get_protobuf_id
from syft.serde.protobuf.proto import set_protobuf_id
from syft.serde.torch.serde import TORCH_DTYPE_STR
from syft.serde.torch.serde import TORCH_STR_DTYPE
from syft.serde.torch.serde import torch_tensor_serializer
from syft.serde.torch.serde import torch_tensor_deserializer
from syft.serde.torch.serde import numpy_tensor_serializer
from syft.serde.torch.serde import numpy_tensor_deserializer

from syft_proto.types.torch.v1.script_function_pb2 import ScriptFunction as ScriptFunctionPB
from syft_proto.types.torch.v1.device_pb2 import Device as DevicePB
from syft_proto.types.torch.v1.parameter_pb2 import Parameter as ParameterPB
from syft_proto.types.torch.v1.script_module_pb2 import ScriptModule as ScriptModulePB
from syft_proto.types.torch.v1.size_pb2 import Size as SizePB
from syft_proto.types.torch.v1.tensor_data_pb2 import TensorData as TensorDataPB
from syft_proto.types.torch.v1.tensor_pb2 import TorchTensor as TorchTensorPB
from syft_proto.types.torch.v1.traced_module_pb2 import TracedModule as TracedModulePB
from syft_proto.types.torch.v1.memory_format_pb2 import MemoryFormat as MemoryFormatPB
from syft_proto.types.torch.v1.dtype_pb2 import TorchDType as TorchDTypePB

SERIALIZERS_SYFT_TO_PROTOBUF = {
    TENSOR_SERIALIZATION.TORCH: TorchTensorPB.Serializer.SERIALIZER_TORCH,
    TENSOR_SERIALIZATION.NUMPY: TorchTensorPB.Serializer.SERIALIZER_NUMPY,
    TENSOR_SERIALIZATION.ALL: TorchTensorPB.Serializer.SERIALIZER_ALL,
}
SERIALIZERS_PROTOBUF_TO_SYFT = {value: key for key, value in SERIALIZERS_SYFT_TO_PROTOBUF.items()}


def _serialize_tensor(worker: AbstractWorker, tensor) -> bin:
    """Serialize the tensor using as default Torch serialization strategy
    This function can be overridden to provide different tensor serialization strategies

    Args
        (torch.Tensor): an input tensor to be serialized

    Returns
        A serialized version of the input tensor

    """
    serializers = {
        TENSOR_SERIALIZATION.TORCH: torch_tensor_serializer,
        TENSOR_SERIALIZATION.NUMPY: numpy_tensor_serializer,
        TENSOR_SERIALIZATION.ALL: protobuf_tensor_serializer,
    }
    if worker.serializer not in serializers:
        raise NotImplementedError(
            f"Tensor serialization strategy is not supported: {worker.serializer}"
        )
    serializer = serializers[worker.serializer]
    return serializer(worker, tensor)


def _deserialize_tensor(worker: AbstractWorker, serializer: str, tensor_bin) -> torch.Tensor:
    """Deserialize the input tensor passed as parameter into a Torch tensor.
    `serializer` parameter selects different deserialization strategies

    Args
        worker: Worker
        serializer: Strategy used for tensor deserialization (e.g.: torch, numpy, all)
        tensor_bin: A simplified representation of a tensor

    Returns
        a Torch tensor
    """
    deserializers = {
        TENSOR_SERIALIZATION.TORCH: torch_tensor_deserializer,
        TENSOR_SERIALIZATION.NUMPY: numpy_tensor_deserializer,
        TENSOR_SERIALIZATION.ALL: protobuf_tensor_deserializer,
    }
    if serializer not in deserializers:
        raise NotImplementedError(
            f"Cannot deserialize tensor serialized with '{serializer}' strategy"
        )
    deserializer = deserializers[serializer]
    return deserializer(worker, tensor_bin)


def protobuf_tensor_serializer(worker: AbstractWorker, tensor: torch.Tensor) -> TensorDataPB:
    """Strategy to serialize a tensor using Protobuf"""
    dtype = TORCH_DTYPE_STR[tensor.dtype]

    protobuf_tensor = TensorDataPB()

    if tensor.is_quantized:
        protobuf_tensor.is_quantized = True
        protobuf_tensor.scale = tensor.q_scale()
        protobuf_tensor.zero_point = tensor.q_zero_point()
        data = torch.flatten(tensor).int_repr().tolist()
    else:
        data = torch.flatten(tensor).tolist()

    protobuf_tensor.dtype = dtype
    protobuf_tensor.shape.dims.extend(tensor.size())
    getattr(protobuf_tensor, "contents_" + dtype).extend(data)

    return protobuf_tensor


def protobuf_tensor_deserializer(
    worker: AbstractWorker, protobuf_tensor: TensorDataPB
) -> torch.Tensor:
    """Strategy to deserialize a binary input using Protobuf"""
    size = tuple(protobuf_tensor.shape.dims)
    data = getattr(protobuf_tensor, "contents_" + protobuf_tensor.dtype)

    if protobuf_tensor.is_quantized:
        # Drop the 'q' from the beginning of the quantized dtype to get the int type
        dtype = TORCH_STR_DTYPE[protobuf_tensor.dtype[1:]]
        int_tensor = torch.tensor(data, dtype=dtype).reshape(size)
        # Automatically converts int types to quantized types
        return torch._make_per_tensor_quantized_tensor(
            int_tensor, protobuf_tensor.scale, protobuf_tensor.zero_point
        )
    else:
        dtype = TORCH_STR_DTYPE[protobuf_tensor.dtype]
        return torch.tensor(data, dtype=dtype).reshape(size)


class TorchTensorWrapper(SyftSerializable):
    """
    Wrapper that serializes torch.Tensor using protobuffers.
    """

    @staticmethod
    def bufferize(worker: AbstractWorker, tensor: torch.Tensor) -> bin:
        """
        This method converts a Torch tensor into a serialized tensor
        using Protobuf. Depending on the worker's serializer, the tensor
        contents may be serialized to binary representations using Torch
        or Numpy, or to a generic inner Protobuf message for cross-platform
        communication.

        Args:
            tensor (torch.Tensor): an input tensor to be serialized

        Returns:
            protobuf_obj: Protobuf version of torch tensor.
        """
        serialized_tensor = _serialize_tensor(worker, tensor)

        if tensor.grad is not None:
            if hasattr(tensor, "child"):
                if isinstance(tensor.child, PointerTensor):
                    grad_chain = None
                else:
                    grad_chain = TorchTensorWrapper.bufferize(worker, tensor.grad)
            else:
                grad_chain = TorchTensorWrapper.bufferize(worker, tensor.grad)

        else:
            grad_chain = None

        chain = None
        if hasattr(tensor, "child"):
            chain = syft.serde.protobuf.serde._bufferize(worker, tensor.child)

        protobuf_tensor = TorchTensorPB()
        set_protobuf_id(protobuf_tensor.id, tensor.id)

        protobuf_tensor.serializer = SERIALIZERS_SYFT_TO_PROTOBUF[worker.serializer]
        if worker.serializer == TENSOR_SERIALIZATION.ALL:
            protobuf_tensor.contents_data.CopyFrom(serialized_tensor)
        else:
            protobuf_tensor.contents_bin = serialized_tensor

        if chain:
            protobuf_tensor.chain.CopyFrom(chain)
        if grad_chain:
            protobuf_tensor.grad_chain.CopyFrom(grad_chain)
        if tensor.description:
            protobuf_tensor.description = tensor.description

        protobuf_tensor.tags.extend(tensor.tags)

        return protobuf_tensor

    @staticmethod
    def unbufferize(worker: AbstractWorker, protobuf_tensor: "TorchTensorPB") -> torch.Tensor:
        """
        This method converts a Protobuf torch tensor back into a
        Torch tensor. The tensor contents can be deserialized from
        binary representations produced by Torch or Numpy, or from
        the generic Protobuf message format for cross-platform
        communication.

        Args:
            protobuf_tensor (bin): Protobuf message of torch tensor.

        Returns:
            tensor (torch.Tensor): a torch tensor converted from Protobuf
        """
        tensor_id = get_protobuf_id(protobuf_tensor.id)
        tags = protobuf_tensor.tags
        description = protobuf_tensor.description

        contents_type = protobuf_tensor.WhichOneof("contents")
        serialized_tensor = getattr(protobuf_tensor, contents_type)
        serializer = SERIALIZERS_PROTOBUF_TO_SYFT[protobuf_tensor.serializer]

        tensor = _deserialize_tensor(worker, (serializer), serialized_tensor)

        # note we need to do this explicitly because torch.load does not
        # include .grad information
        if protobuf_tensor.HasField("grad_chain"):
            grad_chain = protobuf_tensor.grad_chain
            tensor.grad = TorchTensorWrapper.unbufferize(worker, grad_chain)

        initialize_tensor(
            hook=syft.torch.hook,
            obj=tensor,
            owner=worker,
            id=tensor_id,
            init_args=[],
            init_kwargs={},
        )

        if protobuf_tensor.HasField("chain"):
            chain = protobuf_tensor.chain
            chain = TorchTensorWrapper.unbufferize(worker, chain)
            tensor.child = chain
            tensor.is_wrapper = True

        tensor.tags = set(tags)
        tensor.description = description

        return tensor

    @staticmethod
    def get_original_class() -> type(torch.Tensor):
        """
        This method returns the wrapped type.

        Returns:
            torch.Tensor: wrapped type.
        """
        return torch.Tensor

    @staticmethod
    def get_protobuf_schema() -> type(TorchTensorPB):
        """
        This method returns the protobuf schema used for torch.Tensor.

        Returns:
            protobuf schema for torch.tensor.
        """
        return TorchTensorPB


class TorchDeviceWrapper(SyftSerializable):
    """
    Wrapper that serializes torch.device using protobuffers.
    """

    @staticmethod
    def bufferize(worker: AbstractWorker, device: torch.device) -> DevicePB:
        """
        This method converts a Torch device into a serialized device
        using Protobuf.

        Args:
            device (torch.device): an input device to be serialized

        Returns:
            protobuf_device (DevicePB): Protobuf version of torch device.
        """
        protobuf_device = DevicePB()
        protobuf_device.type = device.type
        return protobuf_device

    @staticmethod
    def unbufferize(worker: AbstractWorker, protobuf_device: DevicePB) -> torch.device:
        """
        This method converts a serialized device into a Torch device using the protobuf schema.

        Args:
            device (DevicePB): serialized input device.

        Returns:
            torch.device: torch Device.
        """
        device_type = protobuf_device.type
        return torch.device(type=device_type)

    @staticmethod
    def get_original_class() -> type(torch.device):
        """
            This method returns the wrapped type.

        Returns:
            type: wrapped type.
        """
        return torch.device

    @staticmethod
    def get_protobuf_schema() -> type(DevicePB):
        """
        Returns the protobuf schema used for torch.device.

        Returns:
            type: protobuf schema for torch.device.
        """
        return DevicePB


class ParameterWrapper(SyftSerializable):
    """
    Wrapper that serializes torch.nn.Parameter using protobuffers.
    """

    @staticmethod
    def bufferize(worker: AbstractWorker, param: torch.nn.Parameter) -> ParameterPB:
        """
        This method converts a torch.nn.Parameter into a serialized parameter using ParameterPB.

        Args:
            param (torch.nn.Parameter): input nn.parameter to be serialized.

        Returns:
            protobuf_param: serialized torch.nn.Parameter.
        """
        protobuf_param = ParameterPB()
        set_protobuf_id(protobuf_param.id, param.id)
        protobuf_param.tensor.CopyFrom(syft.serde.protobuf.serde._bufferize(worker, param.data))
        protobuf_param.requires_grad = param.requires_grad
        if param.grad:
            protobuf_param.grad.CopyFrom(syft.serde.protobuf.serde._bufferize(worker, param.grad))
        return protobuf_param

    @staticmethod
    def unbufferize(worker: AbstractWorker, protobuf_param: ParameterPB) -> torch.nn.Parameter:
        """
        This method converts a ParameterPB into a torch.nn.Parameter.

        Args:
            protobuf_param (ParameterPB): input ParameterPB to be deserialized.

        Returns:
            param: (torch.nn.Parameter): deserialized ParameterPB.
        """
        data = syft.serde.protobuf.serde._unbufferize(worker, protobuf_param.tensor)
        param = torch.nn.Parameter(data, requires_grad=protobuf_param.requires_grad)
        param.id = get_protobuf_id(protobuf_param.id)
        if protobuf_param.HasField("grad"):
            param.grad = syft.serde.protobuf.serde._unbufferize(worker, protobuf_param.grad)
        return param

    @staticmethod
    def get_original_class() -> type(torch.nn.Parameter):
        """
        This method returns the wrapped type.

        Returns:
            Wrapped type.
        """
        return torch.nn.Parameter

    @staticmethod
    def get_protobuf_schema() -> type(ParameterPB):
        """
        This method returns the protobuf schema used for torch.nn.Parameter.

        Returns:
            Protobuf schema for torch.nn.Parameter.
        """
        return ParameterPB


class ScriptModuleWrapper(SyftSerializable):
    """
    Wrapper that serializes torch.jit.ScriptModule using protobuffers.
    """

    @staticmethod
    def bufferize(worker: AbstractWorker, script_module: torch.jit.ScriptModule) -> ScriptModulePB:
        """
        This method serializes a torch.jit.ScriptModule using ScriptModulePB.

        Args:
            script_module (torch.jit.ScriptModule): input jit.ScriptModule to be serialized.

        Returns:
            protobuf_script (ScriptModulePB): serialized jit.ScriptModule.
        """
        protobuf_script = ScriptModulePB()
        protobuf_script.obj = script_module.save_to_buffer()
        return protobuf_script

    @staticmethod
    def unbufferize(
        worker: AbstractWorker, protobuf_script: ScriptModulePB
    ) -> torch.jit.ScriptModule:
        """
        This method deserializes a serialized script module into a torch.jit.ScriptModule.

        Args:
            protobuf_script (ScriptModulePB): input ScriptModulePB to be deserialized .

        Returns:
            loaded_module (torch.jit.ScriptModule): deserialized ScriptModulePB.
        """
        script_module_stream = io.BytesIO(protobuf_script.obj)
        loaded_module = torch.jit.load(script_module_stream)
        return loaded_module

    @staticmethod
    def get_protobuf_schema() -> type(ScriptModulePB):
        """
        This methods returns the protobuf schema used for torch.nn.Parameter.

        Returns:
            Protobuf schema for torch.nn.Parameter.
        """
        return ScriptModulePB

    @staticmethod
    def get_original_class() -> type(torch.jit.ScriptModule):
        """
        This metod returns the wrapped type.

        Returns:
            Wrapped type.
        """
        return torch.jit.ScriptModule


class ScriptFunctionWrapper(SyftSerializable):
    """
    Wrapper that serializes torch.jit.ScriptFunction using protobuffers.
    """

    @staticmethod
    def bufferize(
        worker: AbstractWorker, script_module: torch.jit.ScriptFunction
    ) -> ScriptFunctionPB:
        """
        This method serializes a torch.jit.ScriptFunction into a ScriptFunctionPB.

        Args:
            script_module (torch.jit.ScriptFunction): input torch.jit.ScriptFunction
            to be serialized.

        Returns:
            protobuf_script (ScriptFunctionPB): serialized torch.jit.ScriptFunction.
        """
        protobuf_script = ScriptFunctionPB()
        protobuf_script.obj = script_module.save_to_buffer()
        return protobuf_script

    @staticmethod
    def unbufferize(
        worker: AbstractWorker, protobuf_script: ScriptFunctionPB
    ) -> torch.jit.ScriptFunction:
        """
        This method deserializes ScriptFunctionPB into a torch.jit.ScriptFunction.

        Args:
            protobuf_script (torch.jit.ScriptFunction): input serialized ScriptFunctionPB.

        Returns:
            loaded_module (torch.jit.ScriptFunction): deserialized ScriptFunctionPB.
        """
        script_module_stream = io.BytesIO(protobuf_script.obj)
        loaded_module = torch.jit.load(script_module_stream)
        return loaded_module

    @staticmethod
    def get_original_class() -> type(torch.jit.ScriptFunction):
        """
        This method returns the wrapped type.

        Returns:
            Wrapped type.
        """
        return torch.jit.ScriptFunction

    @staticmethod
    def get_protobuf_schema() -> type(ScriptFunctionPB):
        """
        This method returns the protobuf schema used for torch.jit.ScriptFunction.

        Returns:
           Protobuf schema for torch.jit.ScriptFunction.
        """
        return ScriptFunctionPB


class TopLevelTracedModuleWrapper(SyftSerializable):
    """
    Wrapper that serializes torch.jit.TopLevelTracedModule using protobuffers.
    """

    @staticmethod
    def bufferize(
        worker: AbstractWorker, script_module: torch.jit.TopLevelTracedModule
    ) -> TracedModulePB:
        """
        This method serializes a torch.jit.TopLevelTracedModule using TracedModulePB.

        Args:
            script_module (torch.jit.TopLevelTracedModule): input TopLevelTracedModule
            to be serialized.

        Returns:
            protobuf_script (TracedModulePB): serialized TopLevelTracedModule.
        """
        protobuf_script = ScriptModulePB()
        protobuf_script.obj = script_module.save_to_buffer()
        return protobuf_script

    @staticmethod
    def unbufferize(
        worker: AbstractWorker, protobuf_script: TracedModulePB
    ) -> torch.jit.TopLevelTracedModule:
        """
        This method deserializes TracedModulePB into torch.jit.TopLevelTracedModule.

        Args:
            protobuf_script (TracedModulePB): input serialized TracedModulePB.

        Returns:
            loaded_module (torch.jit.TopLevelTracedModule): deserialized TracedModulePB.
        """
        script_module_stream = io.BytesIO(protobuf_script.obj)
        loaded_module = torch.jit.load(script_module_stream)
        return loaded_module

    @staticmethod
    def get_protobuf_schema() -> type(TracedModulePB):
        """
        This method returns the protobuf schema used for torch.jit.TopLevelTracedModule.

        Returns:
           Protobuf schema for torch.jit.TopLevelTracedModule.
        """
        return TracedModulePB

    @staticmethod
    def get_original_class() -> type(torch.jit.TopLevelTracedModule):
        """
        This method returns the wrapped type.

        Returns:
            Wrapped type.
        """
        return torch.jit.TopLevelTracedModule


class TorchSizeWrapper(SyftSerializable):
    """
    Wrapper that serializes torch.Size using protobuffers.
    """

    @staticmethod
    def bufferize(worker: AbstractWorker, size: torch.Size) -> SizePB:
        """
        This method serializes torch.Size into SizePB.

        Args:
            size (torch.Size): input torch.Size to be serialized.

        Returns:
            protobuf_size: serialized torch.Size
        """
        protobuf_size = SizePB()
        protobuf_size.dims.extend(size)
        return protobuf_size

    @staticmethod
    def unbufferize(worker: AbstractWorker, protobuf_size: SizePB) -> torch.Size:
        """
        This method deserializes SizePB into torch.Size.

        Args:
            protobuf_size (SizePB): input SizePB to be deserialized.

        Returns:
            torch.Size: deserialized SizePB
        """
        return torch.Size(protobuf_size.dims)

    @staticmethod
    def get_original_class() -> type(torch.Size):
        """
        This method returns the wrapped type.

        Returns:
            Wrapped type.
        """
        return torch.Size

    @staticmethod
    def get_protobuf_schema() -> type(SizePB):
        """
        Returns the protobuf schema used for torch.Size.

        Returns:
            Protobuf schema for torch.Size.
        """
        return SizePB


class TorchMemFormatWrapper(SyftSerializable):
    """
    Wrapper that serializes torch.memory_format.
    """

    @staticmethod
    def bufferize(worker: AbstractWorker, mem_format: torch.memory_format) -> MemoryFormatPB:
        """
        This method serializes torch.memory_format into MemoryFormatPB.

         Args:
            size (torch.memory_format): input torch.memory_format to be serialized.

         Returns:
            message (MemoryFormatPB): serialized torch.memory_format
        """
        message = MemoryFormatPB()
        message.memory_format_type = str(mem_format).split(".")[-1]
        return message

    @staticmethod
    def unbufferize(
        worker: AbstractWorker, protobuf_mem_format: MemoryFormatPB
    ) -> torch.memory_format:
        """
        This method deserializes MemoryFormatPB into torch.memory_format.

        Args:
            protobuf_size (MemoryFormatPB): input MemoryFormatPB to be deserialized.

        Returns:
            torch.memory_format: deserialized MemoryFormatPB
        """
        return getattr(torch, protobuf_mem_format.memory_format_type)

    @staticmethod
    def get_original_class() -> type(torch.memory_format):
        return torch.memory_format

    @staticmethod
    def get_protobuf_schema() -> type(MemoryFormatPB):
        """
        Returns the protobuf schema used for torch.memory_format.

        Returns:
            Protobuf schema for torch.memory_format.
        """
        return MemoryFormatPB


class TorchDTypeWrapper(SyftSerializable):
    """
    Wrapper that serializes torch.dtype using protobuffers.
    """

    @staticmethod
    def bufferize(worker: AbstractWorker, torch_dtype: torch.dtype) -> TorchDTypePB:
        """
        This method serializes torch.dtype into TorchDTypePB.

        Args:
            torch_dtype (torch.dtype): input torch.dtype to be serialized.

        Returns:
            protobuf_size: serialized torch.dtype
        """
        protobuf_msg = TorchDTypePB()
        protobuf_msg.torch_type = str(torch_dtype)
        return protobuf_msg

    @staticmethod
    def unbufferize(worker: AbstractWorker, protobuf_dtype: TorchDTypePB) -> torch.dtype:
        """
        This method deserializes TorchDTypePB into torch.dtype.

        Args:
            protobuf_dtype (TorchDTypePB): input TorchDTypePB to be deserialized.

        Returns:
            torch.Size: deserialized TorchDTypePB
        """
        return pydoc.locate(protobuf_dtype.torch_type)

    @staticmethod
    def get_original_class() -> type(torch.dtype):
        """
        This method returns the wrapped type.

        Returns:
            Wrapped type.
        """
        return torch.dtype

    @staticmethod
    def get_protobuf_schema() -> type(TorchDTypePB):
        """
        Returns the protobuf schema used for torch.dtype.

        Returns:
            Protobuf schema for torch.dtype.
        """
        return TorchDTypePB
