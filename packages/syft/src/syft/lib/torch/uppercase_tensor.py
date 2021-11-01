# stdlib

# third party
import torch as th

# relative
from ...core.common.serde.serializable import serializable
from ...logger import warning
from ...proto.lib.torch.device_pb2 import Device as Device_PB
from ...proto.lib.torch.tensor_pb2 import TensorProto as Tensor_PB
from .tensor_util import tensor_deserializer
from .tensor_util import tensor_serializer

torch_tensor_type = type(th.tensor([1, 2, 3]))


def object2proto(obj: object) -> Tensor_PB:
    proto = Tensor_PB()
    proto.tensor = tensor_serializer(obj)

    proto.requires_grad = getattr(obj, "requires_grad", False)
    proto.device.CopyFrom(
        Device_PB(
            type=obj.device.type,  # type: ignore
            index=obj.device.index,  # type: ignore
        )
    )

    if proto.requires_grad:
        grad = getattr(obj, "grad", None)
        if grad is not None:
            proto.grad = tensor_serializer(grad)

    return proto


def proto2object(proto: Tensor_PB) -> th.Tensor:
    tensor = tensor_deserializer(proto.tensor)
    if proto.requires_grad:
        tensor.grad = tensor_deserializer(proto.grad)

    tensor.requires_grad_(proto.requires_grad)

    if proto.device.type == "cuda" and th.cuda.is_available():
        cuda_index = proto.device.index
        if th.cuda.device_count() < (cuda_index + 1):
            cuda_index = th.cuda.device_count() - 1
            warning(
                f"The requested CUDA index {proto.device.index} is invalid."
                + f"Falling back to GPU index {cuda_index}.",
                print=True,
            )
        return tensor.cuda(cuda_index)

    if proto.device.type == "cuda" and not th.cuda.is_available():
        warning("Cannot find any CUDA devices, falling back to CPU.", print=True)

    return tensor


serializable(generate_wrapper=True)(
    wrapped_type=torch_tensor_type,
    import_path="torch.Tensor",
    protobuf_scheme=Tensor_PB,
    type_object2proto=object2proto,
    type_proto2object=proto2object,
)
