# third party
import numpy as np
import torch

# syft relative
from ...generate_wrapper import GenerateWrapper
from ...lib.torch.tensor_util import protobuf_tensor_deserializer
from ...lib.torch.tensor_util import protobuf_tensor_serializer
from ...proto.lib.numpy.array_pb2 import NumpyProto

SUPPORTED_BOOL_TYPES = [np.bool_]
SUPPORTED_INT_TYPES = [
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
]

SUPPORTED_FLOAT_TYPES = [
    np.float16,
    np.float32,
    np.float64,
]

SUPPORTED_DTYPES = SUPPORTED_BOOL_TYPES + SUPPORTED_INT_TYPES + SUPPORTED_FLOAT_TYPES

DTYPE_REFACTOR = {
    np.dtype("uint16"): np.int16,
    np.dtype("uint32"): np.int32,
    np.dtype("uint64"): np.int64,
}


def object2proto(obj: np.ndarray) -> NumpyProto:
    original_dtype = obj.dtype
    if original_dtype not in SUPPORTED_DTYPES:
        raise NotImplementedError(f"{original_dtype} is not supported")

    if original_dtype in DTYPE_REFACTOR:
        # store as a signed int, the negative wrap around values convert back to the
        # same original unsigned values on the other side
        obj = obj.astype(DTYPE_REFACTOR[original_dtype])

    tensor = torch.from_numpy(obj).clone()
    tensor_proto = protobuf_tensor_serializer(tensor)
    dtype = original_dtype.name

    return NumpyProto(tensor=tensor_proto, dtype=dtype)


def proto2object(proto: NumpyProto) -> np.ndarray:
    tensor = protobuf_tensor_deserializer(proto.tensor)
    array = tensor.to("cpu").detach().numpy().copy()

    str_dtype = proto.dtype
    original_dtype = np.dtype(str_dtype)

    obj = array.astype(original_dtype)
    return obj


GenerateWrapper(
    wrapped_type=np.ndarray,
    import_path="numpy.ndarray",
    protobuf_scheme=NumpyProto,
    type_object2proto=object2proto,
    type_proto2object=proto2object,
)
