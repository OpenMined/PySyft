# third party
import numpy as np
from numpy import frombuffer

# relative
from ..types.syft_object import SYFT_OBJECT_VERSION_1
from .arrow import numpy_deserialize
from .arrow import numpy_serialize
from .recursive import recursive_serde_register

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

recursive_serde_register(
    np.ndarray,
    serialize=numpy_serialize,
    deserialize=numpy_deserialize,
    canonical_name="numpy_ndarray",
    version=SYFT_OBJECT_VERSION_1,
)

recursive_serde_register(
    np._globals._NoValueType,
    canonical_name="numpy_no_value",
    version=SYFT_OBJECT_VERSION_1,
)
#  serialize=numpy_serialize, deserialize=numpy_deserialize


recursive_serde_register(
    np.bool_,
    serialize=lambda x: x.tobytes(),
    deserialize=lambda buffer: frombuffer(buffer, dtype=np.bool_)[0],
    canonical_name="numpy_bool",
    version=SYFT_OBJECT_VERSION_1,
)

recursive_serde_register(
    np.int8,
    serialize=lambda x: x.tobytes(),
    deserialize=lambda buffer: frombuffer(buffer, dtype=np.int8)[0],
    canonical_name="numpy_int8",
    version=SYFT_OBJECT_VERSION_1,
)

recursive_serde_register(
    np.int16,
    serialize=lambda x: x.tobytes(),
    deserialize=lambda buffer: frombuffer(buffer, dtype=np.int16)[0],
    canonical_name="numpy_int16",
    version=SYFT_OBJECT_VERSION_1,
)

recursive_serde_register(
    np.int32,
    serialize=lambda x: x.tobytes(),
    deserialize=lambda buffer: frombuffer(buffer, dtype=np.int32)[0],
    canonical_name="numpy_int32",
    version=SYFT_OBJECT_VERSION_1,
)

recursive_serde_register(
    np.int64,
    serialize=lambda x: x.tobytes(),
    deserialize=lambda buffer: frombuffer(buffer, dtype=np.int64)[0],
    canonical_name="numpy_int64",
    version=SYFT_OBJECT_VERSION_1,
)

recursive_serde_register(
    np.uint8,
    serialize=lambda x: x.tobytes(),
    deserialize=lambda buffer: frombuffer(buffer, dtype=np.uint8)[0],
    canonical_name="numpy_uint8",
    version=SYFT_OBJECT_VERSION_1,
)

recursive_serde_register(
    np.uint16,
    serialize=lambda x: x.tobytes(),
    deserialize=lambda buffer: frombuffer(buffer, dtype=np.uint16)[0],
    canonical_name="numpy_uint16",
    version=SYFT_OBJECT_VERSION_1,
)

recursive_serde_register(
    np.uint32,
    serialize=lambda x: x.tobytes(),
    deserialize=lambda buffer: frombuffer(buffer, dtype=np.uint32)[0],
    canonical_name="numpy_uint32",
    version=SYFT_OBJECT_VERSION_1,
)

recursive_serde_register(
    np.uint64,
    serialize=lambda x: x.tobytes(),
    deserialize=lambda buffer: frombuffer(buffer, dtype=np.uint64)[0],
    canonical_name="numpy_uint64",
    version=SYFT_OBJECT_VERSION_1,
)

recursive_serde_register(
    np.single,
    serialize=lambda x: x.tobytes(),
    deserialize=lambda buffer: frombuffer(buffer, dtype=np.single)[0],
    canonical_name="numpy_single",
    version=SYFT_OBJECT_VERSION_1,
)

recursive_serde_register(
    np.double,
    serialize=lambda x: x.tobytes(),
    deserialize=lambda buffer: frombuffer(buffer, dtype=np.double)[0],
    canonical_name="numpy_double",
    version=SYFT_OBJECT_VERSION_1,
)

recursive_serde_register(
    np.float16,
    serialize=lambda x: x.tobytes(),
    deserialize=lambda buffer: frombuffer(buffer, dtype=np.float16)[0],
    canonical_name="numpy_float16",
    version=SYFT_OBJECT_VERSION_1,
)

recursive_serde_register(
    np.float32,
    serialize=lambda x: x.tobytes(),
    deserialize=lambda buffer: frombuffer(buffer, dtype=np.float32)[0],
    canonical_name="numpy_float32",
    version=SYFT_OBJECT_VERSION_1,
)

recursive_serde_register(
    np.float64,
    serialize=lambda x: x.tobytes(),
    deserialize=lambda buffer: frombuffer(buffer, dtype=np.float64)[0],
    canonical_name="numpy_float64",
    version=SYFT_OBJECT_VERSION_1,
)

recursive_serde_register(
    np.number,
    serialize=lambda x: x.tobytes(),
    deserialize=lambda buffer: frombuffer(buffer, dtype=np.number)[0],
    canonical_name="numpy_number",
    version=SYFT_OBJECT_VERSION_1,
)

# TODO: There is an incorrect mapping in looping,which makes it not work.
# numpy_scalar_types = [
#     np.bool_,
#     np.int8,
#     np.int16,
#     np.int32,
#     np.int64,
#     np.uint8,
#     np.uint16,
#     np.uint32,
#     np.uint64,
#     np.half,
#     np.single,
#     np.double,
# ]

# for numpy_scalar_type in numpy_scalar_types:
#     recursive_serde_register(
#     numpy_scalar_type,
#     serialize=lambda x: x.tobytes(),
#     deserialize=lambda buffer: frombuffer(buffer, dtype=numpy_scalar_type),
# )

# how else do you import a relative file to execute it?
NOTHING = None
