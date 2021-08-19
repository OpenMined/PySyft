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
from ...proto.core.tensor.tensor_pb2 import Tensor as CoreTensor_PB
from ...core.compression.compression_params import compression_params
from ...core.compression.compressed_tensor import CompressedTensor
from ...core.compression.util import named_compressors
torch_tensor_type = type(th.tensor([1, 2, 3]))


def object2proto(obj: object, use_compression:bool = True) -> CoreTensor_PB:
    compressed = CompressedTensor(obj, [])
    if use_compression and compression_params.tensor['compress']:
        for compressor in compression_params.tensor['compressors']:
            print(compressor)
            compressor = named_compressors[compressor]
            if getattr(compressor, "grad_hist_store", False):
                if not hasattr(obj, "compressor_objs"):
                    obj.compressor_objs = dict()
                if compressor not in obj.compressor_objs:
                    obj.compressor_objs[compressor] = compressor()
                compressor = obj.compressor_objs[compressor]
            compressed.compress_more(compressor)

    return compressed._object2proto()


def proto2object(proto: CoreTensor_PB) -> th.Tensor:
    return CompressedTensor._proto2object(proto)


GenerateWrapper(
    wrapped_type=torch_tensor_type,
    import_path="torch.Tensor",
    protobuf_scheme=CoreTensor_PB,
    type_object2proto=object2proto,
    type_proto2object=proto2object,
)
