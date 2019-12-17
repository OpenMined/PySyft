"""
This file exists to provide one common place for all serialisation and simplify_ and _detail
for all tensors (Torch and Numpy).
"""
from collections import OrderedDict
from tempfile import TemporaryFile

import numpy

import syft
from syft.frameworks.numpy.wrapped_numpy.wrapper import WrappedNumpy
from syft.generic.tensor import initialize_tensor
from syft.workers.abstract import AbstractWorker
from syft.codes import TENSOR_SERIALIZATION


def _serialize_tensor(worker: AbstractWorker, tensor) -> bin:
    """Serialize the tensor using as default Torch serialization strategy
    This function can be overridden to provide different tensor serialization strategies

    Args
        (torch.Tensor): an input tensor to be serialized

    Returns
        A serialized version of the input tensor

    """
    serializers = {
        TENSOR_SERIALIZATION.WRAPPEDNUMPY: wrapped_numpy_serializer,
    }
    if worker.serializer not in serializers:
        raise NotImplementedError(
            f"Tensor serialization strategy is not supported: {worker.serializer}"
        )
    serializer = serializers[worker.serializer]
    return serializer(worker, tensor)


def _deserialize_tensor(worker: AbstractWorker, serializer: str, tensor_bin) -> WrappedNumpy.WrappedNdarray:
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
        TENSOR_SERIALIZATION.WRAPPEDNUMPY: wrapped_numpy__deserializer,
    }
    if serializer not in deserializers:
        raise NotImplementedError(
            f"Cannot deserialize tensor serialized with '{serializer}' strategy"
        )
    deserializer = deserializers[serializer]
    return deserializer(worker, tensor_bin)


def wrapped_numpy_serializer(worker: AbstractWorker, tensor: WrappedNumpy.WrappedNdarray) -> bin:
    """Strategy to serialize a tensor using numpy npy format.
    """
    np_tensor = tensor._array
    outfile = TemporaryFile()
    numpy.save(outfile, np_tensor)
    # Simulate close and open by calling seek
    outfile.seek(0)
    return outfile.read()


def wrapped_numpy__deserializer(worker: AbstractWorker, tensor_bin) -> WrappedNumpy.WrappedNdarray:
    """"Strategy to deserialize a binary input in npy format into a WrappedNdarray tensor"""
    input_file = TemporaryFile()
    input_file.write(tensor_bin)
    # read data from file
    input_file.seek(0)
    return WrappedNumpy.WrappedNdarray(numpy.load(input_file, allow_pickle=True))


# Simplify/Detail WrappedNumpy Tensors


def _simplify_WrappedNumpy_tensor(worker: AbstractWorker, tensor: WrappedNumpy.WrappedNdarray) -> bin:
    """
    This function converts a WrappedNumpy tensor into a serialized WrappedNumpy tensor
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

    chain = None
    if hasattr(tensor, "child"):
        chain = syft.serde._simplify(worker, tensor.child)

    tags = tensor.tags
    if tags is not None:
        tags = list(tags)
    return (
        tensor.id,
        tensor_bin,
        chain,
        tags,
        tensor.description,
        syft.serde._simplify(worker, worker.serializer),
    )


def _detail_WrappedNumpy_tensor(worker: AbstractWorker, tensor_tuple: tuple) -> WrappedNumpy.WrappedNdarray:
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

    tensor_id, tensor_bin, chain, tags, description, serializer = tensor_tuple

    tensor = _deserialize_tensor(worker, syft.serde._detail(worker, serializer), tensor_bin)

    initialize_tensor(
        hook=syft.numpy.hook, obj=tensor, owner=worker, id=tensor_id, init_args=[], init_kwargs={}
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



# Maps a type to a tuple containing its simplifier and detailer function
# IMPORTANT: serialization constants for these objects need to be defined
# in `proto.json` file of https://github.com/OpenMined/proto
MAP_WRAPPEDNUMPY_SIMPLIFIERS_AND_DETAILERS = OrderedDict(
    {
        WrappedNumpy.WrappedNdarray: (_simplify_WrappedNumpy_tensor, _detail_WrappedNumpy_tensor),
    }
)