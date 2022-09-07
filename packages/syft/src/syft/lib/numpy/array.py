# third party
import numpy as np

# relative
from ...core.common.serde import recursive_serde_register
from ...core.common.serde.arrow import arrow_deserialize
from ...core.common.serde.arrow import arrow_serialize

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
    np.ndarray, serialize=arrow_serialize, deserialize=arrow_deserialize
)
