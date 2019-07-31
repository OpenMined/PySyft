"""
This file exists to provide one common place for all serialisation and simplify_ and _detail
for all tensors (Torch and Numpy).
"""
from collections import OrderedDict
from tempfile import TemporaryFile
from typing import Tuple
import torch

import io
import numpy
import warnings

import syft

from syft.federated import TrainConfig

from syft.workers import AbstractWorker
from syft.workers import VirtualWorker

from syft.federated import Plan

from syft.exceptions import GetNotPermittedError
from syft.exceptions import ResponseSignatureError

from syft.frameworks.torch.tensors.decorators import LoggingTensor
from syft.frameworks.torch.tensors.interpreters import AdditiveSharingTensor
from syft.frameworks.torch.tensors.interpreters import MultiPointerTensor
from syft.frameworks.torch.tensors.interpreters.abstract import initialize_tensor
from syft.frameworks.torch import pointers


def _serialize_tensor(tensor) -> bin:
    """Serialize the tensor using as default Torch serialization strategy
    This function can be overridden to provide different tensor serialization strategies

    Args
        (torch.Tensor): an input tensor to be serialized

    Returns
        A serialized version of the input tensor

    """
    return torch_tensor_serializer(tensor)


def _deserialize_tensor(tensor_bin) -> torch.Tensor:
    """Deserialize the input tensor passed as parameter into a Torch tensor.
    This function can be overridden to provide different deserialization strategies

    Args
        tensor_bin: A binary representation of a tensor

    Returns
        a Torch tensor
    """
    return torch_tensor_deserializer(tensor_bin)


def numpy_tensor_serializer(tensor: torch.Tensor) -> bin:
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


def numpy_tensor_deserializer(tensor_bin) -> torch.Tensor:
    """"Strategy to deserialize a binary input in npy format into a Torch tensor"""
    input_file = TemporaryFile()
    input_file.write(tensor_bin)
    # read data from file
    input_file.seek(0)
    return torch.from_numpy(numpy.load(input_file))


def torch_tensor_serializer(tensor) -> bin:
    """Strategy to serialize a tensor using Torch saver"""
    binary_stream = io.BytesIO()
    torch.save(tensor, binary_stream)
    return binary_stream.getvalue()


def torch_tensor_deserializer(tensor_bin) -> torch.Tensor:
    """"Strategy to deserialize a binary input using Torch load"""
    bin_tensor_stream = io.BytesIO(tensor_bin)
    return torch.load(bin_tensor_stream)


# Simplify/Detail Torch Tensors


def _simplify_torch_tensor(tensor: torch.Tensor) -> bin:
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

    tensor_bin = _serialize_tensor(tensor)

    # note we need to do this expicitly because torch.save does not
    # seem to be including .grad by default

    if tensor.grad is not None:
        if hasattr(tensor, "child"):
            if isinstance(tensor.child, pointers.PointerTensor):
                grad_chain = None
            else:
                grad_chain = _simplify_torch_tensor(tensor.grad)
        else:
            grad_chain = _simplify_torch_tensor(tensor.grad)

    else:
        grad_chain = None

    chain = None

    # I think the pointer bug is is between here

    if hasattr(tensor, "child"):
        chain = syft.serde._simplify(tensor.child)

    # and here... leaving a reerence here so i can find it later
    # TODO fix pointer bug

    tags = tensor.tags
    if tags is not None:
        tags = list(tags)
    return (tensor.id, tensor_bin, chain, grad_chain, tags, tensor.description)


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

    tensor_id, tensor_bin, chain, grad_chain, tags, description = tensor_tuple

    tensor = _deserialize_tensor(tensor_bin)

    # note we need to do this explicitly because torch.load does not
    # include .grad informatino
    if grad_chain is not None:
        tensor.grad = _detail_torch_tensor(worker, grad_chain)

    initialize_tensor(
        hook_self=syft.torch.hook,
        cls=tensor,
        torch_tensor=True,
        owner=worker,
        id=tensor_id,
        init_args=[],
        kwargs={},
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


def _simplify_torch_parameter(param: torch.nn.Parameter) -> bin:
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

    tensor_ser = _simplify_torch_tensor(tensor)

    grad = param.grad

    if grad is not None and not (
        hasattr(grad, "child") and isinstance(grad.child, pointers.PointerTensor)
    ):
        grad_ser = _simplify_torch_tensor(grad)
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

    tensor = _detail_torch_tensor(worker, tensor_ser)

    if grad_ser is not None:
        grad = _detail_torch_tensor(worker, grad_ser)
        grad.garbage_collect_data = False
    elif hasattr(tensor, "child") and isinstance(tensor.child, pointers.PointerTensor):
        grad = tensor.attr("grad")
    else:
        grad = None

    param = torch.nn.Parameter(tensor, requires_grad)
    param.id = param_id
    param.grad = grad

    return param


#   Numpy array


def _simplify_ndarray(my_array: numpy.ndarray) -> Tuple[bin, Tuple, str]:
    """
    This function gets the byte representation of the array
        and stores the dtype and shape for reconstruction

    Args:
        my_array (numpy.ndarray): a numpy array

    Returns:
        list: a list holding the byte representation, shape and dtype of the array

    Examples:

        arr_representation = _simplify_ndarray(numpy.random.random([1000, 1000])))

    """
    arr_bytes = my_array.tobytes()
    arr_shape = my_array.shape
    arr_dtype = my_array.dtype.name

    return (arr_bytes, arr_shape, arr_dtype)


def _detail_ndarray(
    worker: AbstractWorker, arr_representation: Tuple[bin, Tuple, str]
) -> numpy.ndarray:
    """
    This function reconstruct a numpy array from it's byte data, the shape and the dtype
        by first loading the byte data with the appropiate dtype and then reshaping it into the
        original shape

    Args:
        worker: the worker doing the deserialization
        arr_representation (tuple): a tuple holding the byte representation, shape
        and dtype of the array

    Returns:
        numpy.ndarray: a numpy array

    Examples:
        arr = _detail_ndarray(arr_representation)

    """
    res = numpy.frombuffer(arr_representation[0], dtype=arr_representation[2]).reshape(
        arr_representation[1]
    )

    assert type(res) == numpy.ndarray

    return res


def _simplify_torch_device(device: torch.device) -> Tuple[str]:
    return device.type


def _detail_torch_device(worker: AbstractWorker, device_type: str) -> torch.device:
    return torch.device(type=device_type)


def _simplify_script_module(obj: torch.jit.ScriptModule) -> str:
    """Strategy to serialize a script module using Torch.jit"""
    return obj.save_to_buffer()


def _detail_script_module(worker: AbstractWorker, script_module_bin: str) -> torch.jit.ScriptModule:
    """"Strategy to deserialize a binary input using Torch load"""
    script_module_stream = io.BytesIO(script_module_bin)
    loaded_module = torch.jit.load(script_module_stream)
    return loaded_module


def _simplify_torch_size(size: torch.Size) -> Tuple[int]:
    return tuple(size)


def _detail_torch_size(worker: AbstractWorker, size: Tuple[int]) -> torch.Size:
    return torch.Size(size)


# Maps a type to a tuple containing its simplifier and detailer function
# IMPORTANT: keep this structure sorted A-Z (by type name)
MAP_TORCH_SIMPLIFIERS_AND_DETAILERS = OrderedDict(
    {
        numpy.ndarray: (_simplify_ndarray, _detail_ndarray),
        torch.device: (_simplify_torch_device, _detail_torch_device),
        torch.jit.ScriptModule: (_simplify_script_module, _detail_script_module),
        torch.jit.TopLevelTracedModule: (_simplify_script_module, _detail_script_module),
        torch.nn.Parameter: (_simplify_torch_parameter, _detail_torch_parameter),
        torch.Tensor: (_simplify_torch_tensor, _detail_torch_tensor),
        torch.Size: (_simplify_torch_size, _detail_torch_size),
    }
)
