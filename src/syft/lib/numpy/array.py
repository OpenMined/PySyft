# third party
import numpy as np
import torch

# syft relative
from ...generate_wrapper import GenerateWrapper
from ...lib.torch.tensor_util import protobuf_tensor_deserializer
from ...lib.torch.tensor_util import protobuf_tensor_serializer
from ...proto.lib.numpy.array_pb2 import NumpyProto

ExceptionDtype = [
    np.object_,
    np.str_,
    np.unicode_,
    np.complex64,
    np.complex128,
]

DtypeRefactor = {
    np.dtype("uint16"): np.int16,
    np.dtype("uint32"): np.int32,
    np.dtype("uint64"): np.int64,
}


def object2proto(obj: np.ndarray) -> NumpyProto:
    if obj.dtype in ExceptionDtype:
        raise NotImplementedError(f"{obj.dtype} is not supported")

    dtype = obj.dtype.name
    if obj.dtype in DtypeRefactor:
        obj = obj.astype(DtypeRefactor[obj.dtype])

    tensor = torch.from_numpy(obj).clone()
    tensor_proto = protobuf_tensor_serializer(tensor)

    return NumpyProto(tensor=tensor_proto, dtype=dtype)


def proto2object(proto: NumpyProto) -> np.ndarray:
    tensor = protobuf_tensor_deserializer(proto.tensor)
    array = tensor.to("cpu").detach().numpy().copy()

    str_dtype = proto.dtype
    npdtype = np.dtype(str_dtype)

    obj = array.astype(npdtype)

    return obj


GenerateWrapper(
    wrapped_type=np.ndarray,
    import_path="numpy.ndarray",
    protobuf_scheme=NumpyProto,
    type_object2proto=object2proto,
    type_proto2object=proto2object,
)
