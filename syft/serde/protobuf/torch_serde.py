"""
This file exists to provide one common place for all serialisation and bufferize_ and _unbufferize
for all tensors (Torch and Numpy).
"""
from collections import OrderedDict
import io
from tempfile import TemporaryFile
from typing import Tuple, List
import warnings

import numpy
import torch

import syft
from syft.generic.pointers.pointer_tensor import PointerTensor
from syft.generic.tensor import initialize_tensor
from syft.generic.tensor import AbstractTensor
from syft.workers.abstract import AbstractWorker
from syft.codes import TENSOR_SERIALIZATION

from syft.serde.torch.serde import TORCH_DTYPE_STR
from syft.serde.torch.serde import TORCH_STR_DTYPE
from syft.serde.torch.serde import torch_tensor_serializer
from syft.serde.torch.serde import torch_tensor_deserializer
from syft.serde.torch.serde import numpy_tensor_serializer
from syft.serde.torch.serde import numpy_tensor_deserializer

from syft_proto.types.syft.v1.shape_pb2 import Shape as ShapePB
from syft_proto.types.torch.v1.tensor_data_pb2 import TensorData as TensorDataPB
from syft_proto.types.torch.v1.tensor_pb2 import TorchTensor as TorchTensorPB


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
    """"Strategy to deserialize a binary input using Protobuf"""
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


# Bufferize/Unbufferize Torch Tensors


def _bufferize_torch_tensor(worker: AbstractWorker, tensor: torch.Tensor) -> bin:
    """
    This function converts a torch tensor into a serliaized torch tensor
    using pickle. We choose to use this because PyTorch has a custom and
    very fast PyTorch pickler.

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
                grad_chain = _bufferize_torch_tensor(worker, tensor.grad)
        else:
            grad_chain = _bufferize_torch_tensor(worker, tensor.grad)

    else:
        grad_chain = None

    chain = None
    if hasattr(tensor, "child"):
        chain = syft.serde.protobuf.serde._bufferize(worker, tensor.child)

    protobuf_tensor = TorchTensorPB()
    protobuf_tensor.id.CopyFrom(syft.serde.protobuf.serde.create_protobuf_id(tensor.id))
    protobuf_tensor.shape.dims.extend(tensor.shape)

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


def _unbufferize_torch_tensor(
    worker: AbstractWorker, protobuf_tensor: "TorchTensorPB"
) -> torch.Tensor:
    """
    This function converts a serialized torch tensor into a torch tensor
    using pickle.

    Args:
        tensor_tuple (bin): serialized obj of torch tensor. It's a tuple where
            the first value is the ID, the second vlaue is the binary for the
            PyTorch object, the third value is the chain of tensor abstractions,
            and the fourth object is the chain of gradients (.grad.grad, etc.)

    Returns:
        torch.Tensor: a torch tensor that was serialized
    """
    tensor_id = getattr(protobuf_tensor.id, protobuf_tensor.id.WhichOneof("id"))
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
        tensor.grad = _unbufferize_torch_tensor(worker, grad_chain)

    initialize_tensor(
        hook=syft.torch.hook, obj=tensor, owner=worker, id=tensor_id, init_args=[], init_kwargs={}
    )

    if protobuf_tensor.HasField("chain"):
        chain = protobuf_tensor.chain
        chain = syft.serde.protobuf.serde._unbufferize(worker, chain)
        tensor.child = chain
        tensor.is_wrapper = True

    tensor.tags = set(tags)
    tensor.description = description

    return tensor


# Maps a type to its bufferizer and unbufferizer functions
MAP_TORCH_PROTOBUF_TRANSLATORS = OrderedDict(
    {torch.Tensor: (_bufferize_torch_tensor, _unbufferize_torch_tensor)}
)
