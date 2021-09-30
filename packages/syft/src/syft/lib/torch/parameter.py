# stdlib

# third party
import torch as th
from torch.nn import Parameter

# relative
from ...core.common.serde.serializable import serializable
from ...proto.lib.torch.parameter_pb2 import ParameterProto as Parameter_PB
from .tensor_util import tensor_deserializer
from .tensor_util import tensor_serializer

torch_tensor = th.tensor([1.0, 2.0, 3.0])
torch_parameter_type = type(th.nn.parameter.Parameter(torch_tensor))


def object2proto(obj: object) -> Parameter_PB:
    proto = Parameter_PB()
    tensor_data = getattr(obj, "data", None)
    if tensor_data is not None:
        proto.tensor = tensor_serializer(tensor_data)

    proto.requires_grad = getattr(obj, "requires_grad", False)
    grad = getattr(obj, "grad", None)
    if grad is not None:
        proto.grad = tensor_serializer(grad)

    # opacus monkey patches this onto the Parameter class
    grad_sample = getattr(obj, "grad_sample", None)
    if grad_sample is not None:
        proto.grad_sample = tensor_serializer(grad_sample)
    return proto


def proto2object(proto: Parameter_PB) -> Parameter:
    data = tensor_deserializer(proto.tensor)
    param = Parameter(data, requires_grad=proto.requires_grad)
    if proto.HasField("grad"):
        param.grad = tensor_deserializer(proto.grad)

    # opacus monkey patches this onto the Parameter class
    if proto.HasField("grad_sample"):
        param.grad_sample = tensor_deserializer(proto.grad_sample)
    return param


serializable(generate_wrapper=True)(
    wrapped_type=torch_parameter_type,
    import_path="torch.nn.parameter.Parameter",
    protobuf_scheme=Parameter_PB,
    type_object2proto=object2proto,
    type_proto2object=proto2object,
)
