# third party
import torch as th
from torch.nn import Parameter

from ...core.common.serde import _serialize, _deserialize, recursive_serde_register
# relative
from .tensor_util import tensor_deserializer
from .tensor_util import tensor_serializer

torch_tensor = th.tensor([1.0, 2.0, 3.0])
torch_parameter_type = type(th.nn.parameter.Parameter(torch_tensor))


def serialize(obj: Parameter) -> bytes:
    tensor_data = getattr(obj, "data", None)
    if tensor_data is not None:
        tensor = tensor_serializer(tensor_data)
    else:
        tensor = None

    requires_grad = getattr(obj, "requires_grad", False)
    grad = getattr(obj, "grad", None)
    if grad is not None:
        grad = tensor_serializer(grad)

    # opacus monkey patches this onto the Parameter class
    grad_sample = getattr(obj, "grad_sample", None)
    if grad_sample is not None:
        grad_sample = tensor_serializer(grad_sample)

    return _serialize((tensor, requires_grad, grad, grad_sample), to_bytes=True)


def deserialize(message: bytes) -> Parameter:
    (tensor, requires_grad, grad, grad_sample) = _deserialize(message, from_bytes=True)

    data = tensor_deserializer(tensor)

    param = Parameter(data, requires_grad=requires_grad)

    if grad:
        param.grad = tensor_deserializer(grad)

    # opacus monkey patches this onto the Parameter class
    if grad_sample:
        param.grad_sample = tensor_deserializer(grad_sample)

    return param

recursive_serde_register(
    torch_parameter_type,
    serialize=serialize,
    deserialize=deserialize
)
