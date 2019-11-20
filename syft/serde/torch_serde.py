"""
This file exists to provide one common place for all serialisation and simplify_ and _detail
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
        TENSOR_SERIALIZATION.ALL: generic_tensor_serializer,
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
        TENSOR_SERIALIZATION.ALL: generic_tensor_deserializer,
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


def generic_tensor_serializer(worker: AbstractWorker, tensor: torch.Tensor) -> tuple:
    """Strategy to serialize a tensor to native python types.
    If tensor requires to calculate gradients, it will be detached.
    """
    if tensor.requires_grad:
        warnings.warn(
            "Torch to native serializer can only be used with tensors that do not require grad. "
            "Detaching tensor to continue"
        )
        tensor = tensor.detach()

    tensor_tuple = (tuple(tensor.size()), TORCH_DTYPE_STR[tensor.dtype], tensor.flatten().tolist())
    return syft.serde._simplify(worker, tensor_tuple)


def generic_tensor_deserializer(worker: AbstractWorker, tensor_tuple: tuple) -> torch.Tensor:
    """"Strategy to deserialize a simplified tensor into a Torch tensor"""

    size, dtype, data_arr = syft.serde._detail(worker, tensor_tuple)
    tensor = torch.tensor(data_arr, dtype=TORCH_STR_DTYPE[dtype]).reshape(size)
    return tensor


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


# Simplify/Detail Torch Tensors


def _simplify_torch_tensor(worker: AbstractWorker, tensor: torch.Tensor) -> bin:
    """
    This function converts a torch tensor into a serliaized torch tensor
    using pickle. We choose to use this because PyTorch has a custom and
    very fast PyTorch pickler.

    Args:
        tensor (torch.Tensor): an input tensor to be serialized

    Returns:
        tuple: serialized tuple of torch tensor. The first value is the
        id of the tensor and the second is the binary for the PyTorch
        object. The third is the chain of abstractions, and the fourth
        (optinally) is the chain of graident tensors (nested tuple)
    """

    tensor_bin = _serialize_tensor(worker, tensor)

    # note we need to do this explicitly because torch.save does not
    # seem to be including .grad by default

    if tensor.grad is not None:
        if hasattr(tensor, "child"):
            if isinstance(tensor.child, PointerTensor):
                grad_chain = None
            else:
                grad_chain = _simplify_torch_tensor(worker, tensor.grad)
        else:
            grad_chain = _simplify_torch_tensor(worker, tensor.grad)

    else:
        grad_chain = None

    chain = None

    # I think the pointer bug is is between here

    if hasattr(tensor, "child"):
        chain = syft.serde._simplify(worker, tensor.child)

    # and here... leaving a reerence here so i can find it later
    # TODO fix pointer bug

    tags = tensor.tags
    if tags is not None:
        tags = list(tags)
    return (
        tensor.id,
        tensor_bin,
        chain,
        grad_chain,
        tags,
        tensor.description,
        syft.serde._simplify(worker, worker.serializer),
    )


def _detail_torch_tensor(worker: AbstractWorker, tensor_tuple: tuple) -> torch.Tensor:
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

    tensor_id, tensor_bin, chain, grad_chain, tags, description, serializer = tensor_tuple

    tensor = _deserialize_tensor(worker, syft.serde._detail(worker, serializer), tensor_bin)

    # note we need to do this explicitly because torch.load does not
    # include .grad informatino
    if grad_chain is not None:
        tensor.grad = _detail_torch_tensor(worker, grad_chain)

    initialize_tensor(
        hook=syft.torch.hook, obj=tensor, owner=worker, id=tensor_id, init_args=[], init_kwargs={}
    )

    if tags is not None:

        tags = list(tags)

        for i in range(len(tags)):
            tag = tags[i]
            if isinstance(tag, bytes):
                tag = tag.decode("utf-8")
            tags[i] = tag
        tensor.tags = tags

    if description is not None:
        if isinstance(description, bytes):
            description = description.decode("utf-8")
        tensor.description = description

    if chain is not None:
        chain = syft.serde._detail(worker, chain)
        tensor.child = chain
        tensor.is_wrapper = True

    return tensor


# Simplify/Detail Parameters


def _simplify_torch_parameter(worker: AbstractWorker, param: torch.nn.Parameter) -> bin:
    """
    This function converts a torch Parameter into a serialized torch Parameter

    Args:
        param (torch.nn.Parameter): an input Parameter to be serialized

    Returns:
        tuple: serialized tuple of torch Parameter. The first value is the
        id of the Parameter and the second is the binary for the PyTorch
        tensor data attribute and last is the requires_grad attr.
    """

    tensor = param.data
    tensor_ser = syft.serde._simplify(worker, tensor)

    grad = param.grad

    if grad is not None and not (hasattr(grad, "child") and isinstance(grad.child, PointerTensor)):
        grad_ser = _simplify_torch_tensor(worker, grad)
    else:
        grad_ser = None

    return (param.id, tensor_ser, param.requires_grad, grad_ser)


def _detail_torch_parameter(worker: AbstractWorker, param_tuple: tuple) -> torch.nn.Parameter:
    """
    This function converts a serialized torch Parameter into a torch Parameter.

    Args:
        param_tuple (tuple): serialized obj of torch tensor. It's a tuple where
            the first value is the ID and the second value is the binary for the
            PyTorch data attribute et and third value is the requires_grad attr.

    Returns:
        torch.Parameter: a torch Parameter that was serialized
    """
    param_id, tensor_ser, requires_grad, grad_ser = param_tuple

    tensor = syft.serde._detail(worker, tensor_ser)

    if grad_ser is not None:
        grad = _detail_torch_tensor(worker, grad_ser)
        grad.garbage_collect_data = False
    elif hasattr(tensor, "child") and isinstance(tensor.child, PointerTensor):
        grad = tensor.attr("grad")
    else:
        grad = None

    param = torch.nn.Parameter(tensor, requires_grad)
    param.id = param_id
    param.grad = grad
    param.is_wrapper = isinstance(tensor, AbstractTensor) or tensor.is_wrapper

    return param


def _simplify_torch_device(worker: AbstractWorker, device: torch.device) -> Tuple[str]:
    return device.type


def _detail_torch_device(worker: AbstractWorker, device_type: str) -> torch.device:
    return torch.device(type=device_type)


def _simplify_torch_size(worker: AbstractWorker, shape: torch.Size) -> Tuple:
    return (list(shape),)


def _detail_torch_size(worker: AbstractWorker, shape: List[int]) -> torch.Size:
    return torch.Size(*shape)


def _simplify_script_module(worker: AbstractWorker, obj: torch.jit.ScriptModule) -> str:
    """Strategy to serialize a script module using Torch.jit"""
    return obj.save_to_buffer()


def _detail_script_module(worker: AbstractWorker, script_module_bin: str) -> torch.jit.ScriptModule:
    """"Strategy to deserialize a binary input using Torch load"""
    script_module_stream = io.BytesIO(script_module_bin)
    loaded_module = torch.jit.load(script_module_stream)
    return loaded_module


def _simplify_torch_size(worker: AbstractWorker, size: torch.Size) -> Tuple[int]:
    return tuple(size)


def _detail_torch_size(worker: AbstractWorker, size: Tuple[int]) -> torch.Size:
    return torch.Size(size)


# Maps a type to a tuple containing its simplifier and detailer function
# IMPORTANT: serialization constants for these objects need to be defined
# in `proto.json` file of https://github.com/OpenMined/proto
MAP_TORCH_SIMPLIFIERS_AND_DETAILERS = OrderedDict(
    {
        torch.device: (_simplify_torch_device, _detail_torch_device),
        torch.Size: (_simplify_torch_size, _detail_torch_size),
        torch.jit.ScriptModule: (_simplify_script_module, _detail_script_module),
        torch._C.Function: (_simplify_script_module, _detail_script_module),
        torch.jit.TopLevelTracedModule: (_simplify_script_module, _detail_script_module),
        torch.nn.Parameter: (_simplify_torch_parameter, _detail_torch_parameter),
        torch.Tensor: (_simplify_torch_tensor, _detail_torch_tensor),
        torch.Size: (_simplify_torch_size, _detail_torch_size),
    }
)
