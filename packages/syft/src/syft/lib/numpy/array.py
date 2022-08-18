# stdlib
from typing import Tuple
from typing import Union

# third party
from capnp.lib.capnp import _DynamicStructBuilder
import numpy as np
import pyarrow as pa

# relative
from ...core.common.serde.capnp import CapnpModule
from ...core.common.serde.capnp import chunk_bytes
from ...core.common.serde.capnp import combine_bytes
from ...core.common.serde.capnp import get_capnp_schema
from ...experimental_flags import ApacheArrowCompression
from ...experimental_flags import flags
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


# TODO: move to sy.serialize interface, when protobuf for numpy is removed.
def capnp_serialize(obj: np.ndarray, to_bytes: bool = False) -> _DynamicStructBuilder:
    schema = get_capnp_schema(schema_file="array.capnp")
    array_struct: CapnpModule = schema.Array  # type: ignore
    array_msg = array_struct.new_message()
    metadata_schema = array_struct.TensorMetadata
    array_metadata = metadata_schema.new_message()

    obj_bytes, obj_decompressed_size = arrow_serialize(obj, get_bytes=True)
    chunk_bytes(obj_bytes, "array", array_msg)
    array_metadata.dtype = str(obj.dtype)
    array_metadata.decompressedSize = obj_decompressed_size

    array_msg.arrayMetadata = array_metadata

    if not to_bytes:
        return array_msg
    else:
        return array_msg.to_bytes_packed()


def capnp_deserialize(
    msg: Union[_DynamicStructBuilder, bytes], from_bytes: bool = False
) -> np.ndarray:
    array_msg: _DynamicStructBuilder
    if from_bytes:
        schema = get_capnp_schema(schema_file="array.capnp")
        array_struct: CapnpModule = schema.Array  # type: ignore
        # https://stackoverflow.com/questions/48458839/capnproto-maximum-filesize
        MAX_TRAVERSAL_LIMIT = 2**64 - 1
        # to pack or not to pack?
        # array_msg = array_struct.from_bytes(buf, traversal_limit_in_words=2 ** 64 - 1)
        array_msg = array_struct.from_bytes_packed(
            msg, traversal_limit_in_words=MAX_TRAVERSAL_LIMIT
        )
    else:
        array_msg = msg

    array_metadata = array_msg.arrayMetadata
    obj = arrow_deserialize(
        combine_bytes(array_msg.array),
        decompressed_size=array_metadata.decompressedSize,
        dtype=array_metadata.dtype,
    )

    return obj


def arrow_serialize(
    obj: np.ndarray, get_bytes: bool = False
) -> Union[Tuple[bytes, int], NumpyProto]:
    original_dtype = obj.dtype
    apache_arrow = pa.Tensor.from_numpy(obj=obj)
    sink = pa.BufferOutputStream()
    pa.ipc.write_tensor(apache_arrow, sink)
    buffer = sink.getvalue()
    if flags.APACHE_ARROW_COMPRESSION is ApacheArrowCompression.NONE:
        numpy_bytes = buffer.to_pybytes()
    else:
        numpy_bytes = pa.compress(
            buffer, asbytes=True, codec=flags.APACHE_ARROW_COMPRESSION.value
        )
    dtype = original_dtype.name

    if get_bytes:
        return numpy_bytes, buffer.size
    else:
        return NumpyProto(
            arrow_data=numpy_bytes, dtype=dtype, decompressed_size=buffer.size
        )


def arrow_deserialize(buf: bytes, decompressed_size: int, dtype: str) -> np.ndarray:
    original_dtype = np.dtype(dtype)
    if flags.APACHE_ARROW_COMPRESSION is ApacheArrowCompression.NONE:
        reader = pa.BufferReader(buf)
        buf = reader.read_buffer()
    else:
        buf = pa.decompress(
            buf,
            decompressed_size=decompressed_size,
            codec=flags.APACHE_ARROW_COMPRESSION.value,
        )

    result = pa.ipc.read_tensor(buf)
    np_array = result.to_numpy()
    np_array.setflags(write=True)
    return np_array.astype(original_dtype)
