# stdlib

# third party
import torch as th

# relative
from ...generate_wrapper import GenerateWrapper
from ...lib.torch.tensor_util import tensor_deserializer
from ...lib.torch.tensor_util import tensor_serializer
from ...logger import warning
from ...proto.lib.torch.device_pb2 import Device as Device_PB
from ...proto.lib.torch.tensor_pb2 import TensorProto as Tensor_PB
from ...core.compression.compression_params import compression_params
from ...core.compression.compressed_tensor import CompressedTensor
torch_tensor_type = type(th.tensor([1, 2, 3]))


def object2proto(obj: object) -> Tensor_PB:
    if compression_params.tensor['compress']:
        compressed = CompressedTensor(obj)

        for compressor in compression_params.tensor['compressors']:
            if getattr(compressor, "grad_hist_store", False):
                if not hasattr(obj, "compressor_objs"):
                    obj.compressor_objs = dict()
                if compressor not in obj.compressor_objs:
                    obj.compressor_objs[compressor] = compressor()
                compressor = obj.compressor_objs[compressor]
            compressed.compress_more(compressor)

        return compressed._object2proto()
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


GenerateWrapper(
    wrapped_type=torch_tensor_type,
    import_path="torch.Tensor",
    protobuf_scheme=Tensor_PB,
    type_object2proto=object2proto,
    type_proto2object=proto2object,
)
