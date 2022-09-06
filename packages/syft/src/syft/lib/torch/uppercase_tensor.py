# stdlib
from typing import cast

# third party
import torch as th

# relative
from ...core.common.serde import _deserialize
from ...core.common.serde import _serialize
from ...core.common.serde import recursive_serde_register
from ...logger import warning
from .tensor_util import tensor_deserializer
from .tensor_util import tensor_serializer

torch_tensor_type = type(th.tensor([1, 2, 3]))


def serialize(obj: object) -> bytes:
    serialized_tensor = tensor_serializer(obj)
    requires_grad = getattr(obj, "requires_grad", False)
    device = getattr(obj, "device", th.device("cpu"))

    if requires_grad:
        grad = getattr(obj, "grad", None)
        if grad is not None:
            grad = tensor_serializer(grad)
    else:
        grad = None

    return cast(
        bytes,
        _serialize((serialized_tensor, device, requires_grad, grad), to_bytes=True),
    )


def deserialize(message: bytes) -> th.Tensor:
    (serialized_tensor, device, requires_grad, grad) = _deserialize(
        message, from_bytes=True
    )

    tensor = tensor_deserializer(serialized_tensor)

    if requires_grad:
        tensor.grad = tensor_deserializer(grad)

    tensor.requires_grad_(requires_grad)

    if device.type == "cuda" and th.cuda.is_available():
        cuda_index = device.index
        if th.cuda.device_count() < (cuda_index + 1):
            cuda_index = th.cuda.device_count() - 1
            warning(
                f"The requested CUDA index {device.index} is invalid."
                + f"Falling back to GPU index {cuda_index}.",
                print=True,
            )
        return tensor.cuda(cuda_index)

    if device.type == "cuda" and not th.cuda.is_available():
        warning("Cannot find any CUDA devices, falling back to CPU.", print=True)

    return tensor


recursive_serde_register(
    torch_tensor_type, serialize=serialize, deserialize=deserialize
)
