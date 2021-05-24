# stdlib

# third party
import torch as th
from torch.nn import Parameter

# syft relative
from ...generate_wrapper import GenerateWrapper
from ...lib.torch.tensor_util import protobuf_tensor_deserializer
from ...lib.torch.tensor_util import protobuf_tensor_serializer
from ...proto.lib.torch.parameter_pb2 import ParameterProto as Parameter_PB

torch_tensor = th.tensor([1.0, 2.0, 3.0])
torch_parameter_type = type(th.nn.parameter.Parameter(torch_tensor))


def object2proto(obj: object) -> Parameter_PB:
    proto = Parameter_PB()
    tensor_data = getattr(obj, "data", None)
    if tensor_data is not None:
        proto.tensor.CopyFrom(protobuf_tensor_serializer(tensor_data))
    proto.requires_grad = getattr(obj, "requires_grad", False)
    grad = getattr(obj, "grad", None)
    if grad is not None:
        proto.grad.CopyFrom(protobuf_tensor_serializer(grad))

    # opacus monkey patches this onto the Parameter class
    grad_sample = getattr(obj, "grad_sample", None)
    if grad_sample is not None:
        proto.grad_sample.CopyFrom(protobuf_tensor_serializer(grad_sample))
    return proto


def proto2object(proto: Parameter_PB) -> Parameter:
    data = protobuf_tensor_deserializer(proto.tensor)
    param = Parameter(data, requires_grad=proto.requires_grad)
    if proto.HasField("grad"):
        param.grad = protobuf_tensor_deserializer(proto.grad)

    # opacus monkey patches this onto the Parameter class
    if proto.HasField("grad_sample"):
        param.grad_sample = protobuf_tensor_deserializer(proto.grad_sample)
    return param


GenerateWrapper(
    wrapped_type=torch_parameter_type,
    import_path="torch.nn.parameter.Parameter",
    protobuf_scheme=Parameter_PB,
    type_object2proto=object2proto,
    type_proto2object=proto2object,
)
