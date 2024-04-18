# stdlib
from typing import cast

# third party
import numpy as np
import pyarrow as pa

# relative
from ..util.experimental_flags import ApacheArrowCompression
from ..util.experimental_flags import flags
from .deserialize import _deserialize
from .serialize import _serialize


def arrow_serialize(obj: np.ndarray) -> bytes:
    # inner function to make sure variables go out of scope after this
    def inner(obj: np.ndarray) -> tuple:
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
        return (numpy_bytes, buffer.size, dtype)

    m = inner(obj)
    return cast(bytes, _serialize(m, to_bytes=True))


def arrow_deserialize(
    numpy_bytes: bytes, decompressed_size: int, dtype: str
) -> np.ndarray:
    original_dtype = np.dtype(dtype)
    if flags.APACHE_ARROW_COMPRESSION is ApacheArrowCompression.NONE:
        reader = pa.BufferReader(numpy_bytes)
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


def numpyutf8toarray(input_index: np.ndarray) -> np.ndarray:
    """Decodes utf-8 encoded numpy array to string numpy array.

    Args:
        input_index (np.ndarray): utf-8 encoded array

    Returns:
        np.ndarray: decoded NumpyArray.
    """
    shape_length = int(input_index[-1])
    shape = tuple(input_index[-(shape_length + 1) : -1])  # noqa
    string_index = input_index[: -(shape_length + 1)]
    index_length = int(string_index[-1])
    index_array = string_index[-(index_length + 1) : -1]  # noqa
    string_array: np.ndarray = string_index[: -(index_length + 1)]
    output_bytes: bytes = string_array.astype(np.uint8).tobytes()
    output_list = []
    last_offset = 0
    for offset in index_array:
        chars = output_bytes[last_offset:offset]
        final_string = chars.decode("utf-8")
        last_offset = offset
        output_list.append(final_string)
    return np.array(output_list).reshape(shape)


def arraytonumpyutf8(string_list: str | np.ndarray) -> bytes:
    """Encodes string Numpyarray  to utf-8 encoded numpy array.

    Args:
        string_list (np.ndarray): NumpyArray to be encoded

    Returns:
        bytes: serialized utf-8 encoded int Numpy array
    """
    array_shape = np.array(string_list).shape
    string_list = np.array(string_list).flatten()
    bytes_list = []
    indexes = []
    offset = 0

    for item in string_list:
        name_bytes = item.encode("utf-8")
        offset += len(name_bytes)
        indexes.append(offset)
        bytes_list.append(name_bytes)

    np_bytes = np.frombuffer(b"".join(bytes_list), dtype=np.uint8)
    np_bytes = np_bytes.astype(np.uint64)
    np_indexes = np.array(indexes, dtype=np.uint64)
    index_length = np.array([len(np_indexes)], dtype=np.uint64)
    shape = np.array(array_shape, dtype=np.uint64)
    shape_length = np.array([len(shape)], dtype=np.uint64)
    output_array = np.concatenate(
        [np_bytes, np_indexes, index_length, shape, shape_length]
    )

    return cast(bytes, _serialize(output_array, to_bytes=True))


def numpy_serialize(obj: np.ndarray) -> bytes:
    if obj.dtype.type != np.str_:
        return arrow_serialize(obj)
    else:
        return arraytonumpyutf8(obj)


def numpy_deserialize(buf: bytes) -> np.ndarray:
    deser = _deserialize(buf, from_bytes=True)
    if isinstance(deser, tuple):
        return arrow_deserialize(*deser)
    elif isinstance(deser, np.ndarray):
        return numpyutf8toarray(deser)
    else:
        raise ValueError(f"Invalid type:{type(deser)} for numpy deserialization")
