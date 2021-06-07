# third party
import numpy as np
import pyarrow as pa
import torch

# syft relative
from ...experimental_flags import flags
from ...generate_wrapper import GenerateWrapper
from ...lib.torch.tensor_util import protobuf_tensor_deserializer
from ...lib.torch.tensor_util import protobuf_tensor_serializer
from ...proto.lib.numpy.array_pb2 import NumpyProto
from ...proto.lib.numpy.array_pb2 import NumpyProtoArrow

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


def arrow_object2proto(obj: np.ndarray) -> NumpyProtoArrow:
    apache_arrow = pa.Tensor.from_numpy(obj=obj)
    sink = pa.BufferOutputStream()
    pa.ipc.write_tensor(apache_arrow, sink)
    buf = sink.getvalue().to_pybytes()
    proto = NumpyProtoArrow(data=buf)
    return proto


def arrow_proto2object(proto: NumpyProtoArrow) -> np.ndarray:
    reader = pa.BufferReader(proto.data)
    buf = reader.read_buffer()
    result = pa.ipc.read_tensor(buf)
    np_array = result.to_numpy()
    np_array.setflags(write=True)
    return np_array


def protobuf_object2proto(obj: np.ndarray) -> NumpyProto:
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


def protobuf_proto2object(proto: NumpyProto) -> np.ndarray:
    tensor = protobuf_tensor_deserializer(proto.tensor)
    array = tensor.to("cpu").detach().numpy().copy()
    str_dtype = proto.dtype
    original_dtype = np.dtype(str_dtype)
    obj = array.astype(original_dtype)
    return obj


def _generate_serde() -> None:
    if flags.APACHE_ARROW_TENSOR_SERDE:
        NumpyProtoArrow.schema2type = None
        GenerateWrapper(
            wrapped_type=np.ndarray,
            import_path="numpy.ndarray",
            protobuf_scheme=NumpyProtoArrow,
            type_object2proto=arrow_object2proto,
            type_proto2object=arrow_proto2object,
        )
    else:
        NumpyProto.schema2type = None
        GenerateWrapper(
            wrapped_type=np.ndarray,
            import_path="numpy.ndarray",
            protobuf_scheme=NumpyProto,
            type_object2proto=protobuf_object2proto,
            type_proto2object=protobuf_proto2object,
        )


_generate_serde()
flags._regenerate_numpy_serde = _generate_serde
