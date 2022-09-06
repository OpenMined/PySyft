# stdlib
from typing import cast

# third party
import numpy as np
import pyarrow as pa

# relative
from ....experimental_flags import ApacheArrowCompression
from ....experimental_flags import flags
from .deserialize import _deserialize
from .serialize import _serialize


def arrow_serialize(obj: np.ndarray) -> bytes:
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

    return cast(bytes, _serialize((numpy_bytes, buffer.size, dtype), to_bytes=True))


def arrow_deserialize(buf: bytes) -> np.ndarray:
    (numpy_bytes, decompressed_size, dtype) = _deserialize(buf, from_bytes=True)
    original_dtype = np.dtype(dtype)
    if flags.APACHE_ARROW_COMPRESSION is ApacheArrowCompression.NONE:
        reader = pa.BufferReader(buf)
        numpy_bytes = reader.read_buffer()
    else:
        numpy_bytes = pa.decompress(
            numpy_bytes,
            decompressed_size=decompressed_size,
            codec=flags.APACHE_ARROW_COMPRESSION.value,
        )

    result = pa.ipc.read_tensor(numpy_bytes)
    np_array = result.to_numpy()
    np_array.setflags(write=True)
    return np_array.astype(original_dtype)
