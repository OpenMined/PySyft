import io
import torch

from syft.workers.abstract import AbstractWorker

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


TORCH_MFORMAT_ID = {torch.channels_last: 1, torch.contiguous_format: 2, torch.preserve_format: 3}

TORCH_ID_MFORMAT = {i: cls for cls, i in TORCH_MFORMAT_ID.items()}


def torch_tensor_serializer(worker: AbstractWorker, tensor) -> bin:
    """Strategy to serialize a tensor using Torch saver"""
    binary_stream = io.BytesIO()
    torch.save(tensor, binary_stream)
    return binary_stream.getvalue()


def torch_tensor_deserializer(worker: AbstractWorker, tensor_bin) -> torch.Tensor:
    """Strategy to deserialize a binary input using Torch load"""
    bin_tensor_stream = io.BytesIO(tensor_bin)
    return torch.load(bin_tensor_stream)


def numpy_tensor_serializer(worker: AbstractWorker, tensor) -> bin:
    """Strategy to serialize a tensor using numpy conversion"""
    return tensor.detach().numpy()


def numpy_tensor_deserializer(worker: AbstractWorker, tensor) -> torch.Tensor:
    """Strategy to deserialize a binary input using numpy conversion """
    return torch.from_numpy(tensor)
