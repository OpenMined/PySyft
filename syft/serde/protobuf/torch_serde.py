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

from syft_proto.generic.v1.tensor_pb2 import Tensor as TensorPB

# Torch dtypes to string (and back) mappers
TORCH_DTYPE_STR = {
    torch.uint8: "uint8",
    torch.int8: "int8",
    torch.int16: "int16",
    torch.int32: "int32",
    torch.int64: "int64",
    torch.float16: "float16",
    torch.float32: "float32",
    torch.float64: "float64",
    torch.complex32: "complex32",
    torch.complex64: "complex64",
    torch.complex128: "complex128",
    torch.bool: "bool",
    torch.qint8: "qint8",
    torch.quint8: "quint8",
    torch.qint32: "qint32",
    torch.bfloat16: "bfloat16",
}
TORCH_STR_DTYPE = {name: cls for cls, name in TORCH_DTYPE_STR.items()}


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
        TENSOR_SERIALIZATION.NUMPY: numpy_tensor_serializer,
    }
    if serializer not in deserializers:
        raise NotImplementedError(
            f"Cannot deserialize tensor serialized with '{serializer}' strategy"
        )
    deserializer = deserializers[serializer]
    return deserializer(worker, tensor_bin)


def numpy_tensor_serializer(worker: AbstractWorker, tensor: torch.Tensor) -> bin:
    """Strategy to serialize a tensor using numpy npy format.
    If tensor requires to calculate gradients, it will be detached.
    """
    if tensor.requires_grad:
        warnings.warn(
            "Torch to Numpy serializer can only be used with tensors that do not require grad. "
            "Detaching tensor to continue"
        )
        tensor = tensor.detach()

    np_tensor = tensor.numpy()
    outfile = TemporaryFile()
    numpy.save(outfile, np_tensor)
    # Simulate close and open by calling seek
    outfile.seek(0)
    return outfile.read()


def numpy_tensor_deserializer(worker: AbstractWorker, tensor_bin) -> torch.Tensor:
    """"Strategy to deserialize a binary input in npy format into a Torch tensor"""
    input_file = TemporaryFile()
    input_file.write(tensor_bin)
    # read data from file
    input_file.seek(0)
    return torch.from_numpy(numpy.load(input_file))


def torch_tensor_serializer(worker: AbstractWorker, tensor) -> bin:
    """Strategy to serialize a tensor using Torch saver"""
    binary_stream = io.BytesIO()
    torch.save(tensor, binary_stream)
    return binary_stream.getvalue()


def torch_tensor_deserializer(worker: AbstractWorker, tensor_bin) -> torch.Tensor:
    """"Strategy to deserialize a binary input using Torch load"""
    bin_tensor_stream = io.BytesIO(tensor_bin)
    return torch.load(bin_tensor_stream)


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

    tensor_bin = _serialize_tensor(worker, tensor)

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

    protobuf_tensor = TensorPB()

    protobuf_tensor.id.CopyFrom(syft.serde.protobuf.serde.create_protobuf_id(tensor.id))

    protobuf_tensor.shape.dims.extend(tensor.shape)
    protobuf_tensor.bin = tensor_bin

    if chain:
        protobuf_tensor.chain.CopyFrom(chain)
    if grad_chain:
        protobuf_tensor.grad_chain.CopyFrom(grad_chain)
    if tensor.description:
        protobuf_tensor.description = tensor.description

    protobuf_tensor.tags.extend(tensor.tags)
    protobuf_tensor.serializer = (
        TensorPB.Serializer.SERIALIZER_TORCH
    )  # TODO[karlhigley]: Fix this so the other serializer types also work

    return protobuf_tensor


def _unbufferize_torch_tensor(worker: AbstractWorker, protobuf_tensor: "TensorPB") -> torch.Tensor:
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
    tensor_bin = protobuf_tensor.bin
    tags = protobuf_tensor.tags
    description = protobuf_tensor.description
    serializer = (
        TENSOR_SERIALIZATION.TORCH
    )  # TODO[karlhigley]: Fix this so the other serializer types also work

    tensor = _deserialize_tensor(worker, serializer, tensor_bin)

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
