# stdlib
from datetime import date
from datetime import datetime
from datetime import time
import functools
from importlib.util import find_spec
from io import BytesIO

# third party
from dateutil import parser
from nacl.signing import SigningKey
from nacl.signing import VerifyKey
import numpy as np
from pandas import DataFrame
from pandas import Series
from pandas._libs.tslibs.timestamps import Timestamp
import pyarrow as pa
import pyarrow.parquet as pq
import pydantic
from pydantic._internal._model_construction import ModelMetaclass
from pymongo.collection import Collection
from result import Err
from result import Ok
from result import Result

# relative
from ..types.dicttuple import DictTuple
from ..types.dicttuple import _Meta as _DictTupleMetaClass
from ..types.syft_metaclass import EmptyType
from ..types.syft_metaclass import PartialModelMetaclass
from .array import numpy_deserialize
from .array import numpy_serialize
from .deserialize import _deserialize as deserialize
from .recursive_primitives import _serialize_kv_pairs
from .recursive_primitives import deserialize_kv
from .recursive_primitives import deserialize_type
from .recursive_primitives import recursive_serde_register
from .recursive_primitives import recursive_serde_register_type
from .recursive_primitives import serialize_type
from .serialize import _serialize as serialize

recursive_serde_register(
    SigningKey,
    serialize=lambda x: bytes(x),
    deserialize=lambda x: SigningKey(x),
)

recursive_serde_register(
    VerifyKey,
    serialize=lambda x: bytes(x),
    deserialize=lambda x: VerifyKey(x),
)


# result Ok and Err
recursive_serde_register(Ok, serialize_attrs=["_value"])
recursive_serde_register(Err, serialize_attrs=["_value"])
recursive_serde_register(Result)

# exceptions
recursive_serde_register(cls=TypeError)

# mongo collection
recursive_serde_register_type(Collection)


def serialize_dataframe(df: DataFrame) -> bytes:
    table = pa.Table.from_pandas(df)
    sink = pa.BufferOutputStream()
    # 🟡 TODO 37: Should we warn about this?
    parquet_args = {
        "coerce_timestamps": "us",
        "allow_truncated_timestamps": True,
    }
    pq.write_table(table, sink, **parquet_args)
    buffer = sink.getvalue()
    numpy_bytes = buffer.to_pybytes()
    return numpy_bytes


def deserialize_dataframe(buf: bytes) -> DataFrame:
    reader = pa.BufferReader(buf)
    numpy_bytes = reader.read_buffer()
    result = pq.read_table(numpy_bytes)
    df = result.to_pandas()
    return df


# pandas
recursive_serde_register(
    DataFrame,
    serialize=serialize_dataframe,
    deserialize=deserialize_dataframe,
)


def deserialize_series(blob: bytes) -> Series:
    df: DataFrame = DataFrame.from_dict(deserialize(blob, from_bytes=True))
    return Series(df[df.columns[0]])


recursive_serde_register(
    Series,
    serialize=lambda x: serialize(DataFrame(x).to_dict(), to_bytes=True),
    deserialize=deserialize_series,
)

recursive_serde_register(
    datetime,
    serialize=lambda x: serialize(x.isoformat(), to_bytes=True),
    deserialize=lambda x: parser.isoparse(deserialize(x, from_bytes=True)),
)

recursive_serde_register(
    time,
    serialize=lambda x: serialize(x.isoformat(), to_bytes=True),
    deserialize=lambda x: parser.parse(deserialize(x, from_bytes=True)).time(),
)

recursive_serde_register(
    date,
    serialize=lambda x: serialize(x.isoformat(), to_bytes=True),
    deserialize=lambda x: parser.parse(deserialize(x, from_bytes=True)).date(),
)

recursive_serde_register(
    Timestamp,
    serialize=lambda x: serialize(x.value, to_bytes=True),
    deserialize=lambda x: Timestamp(deserialize(x, from_bytes=True)),
)


def _serialize_dicttuple(x: DictTuple) -> bytes:
    return _serialize_kv_pairs(size=len(x), kv_pairs=zip(x.keys(), x))


recursive_serde_register(
    _DictTupleMetaClass,
    serialize=serialize_type,
    deserialize=deserialize_type,
)
recursive_serde_register(
    DictTuple,
    serialize=_serialize_dicttuple,
    deserialize=functools.partial(deserialize_kv, DictTuple),
)


recursive_serde_register(
    EmptyType,
    serialize=serialize_type,
    deserialize=deserialize_type,
)


recursive_serde_register_type(ModelMetaclass)
recursive_serde_register_type(PartialModelMetaclass)


def serialize_bytes_io(io: BytesIO) -> bytes:
    io.seek(0)
    return serialize(io.read(), to_bytes=True)


recursive_serde_register(
    BytesIO,
    serialize=serialize_bytes_io,
    deserialize=lambda x: BytesIO(deserialize(x, from_bytes=True)),
)

try:
    # third party
    from IPython.display import Image

    recursive_serde_register(Image)

except Exception:  # nosec
    pass


try:
    # third party
    import torch
    from torch._C import _TensorMeta

    recursive_serde_register_type(_TensorMeta)
    recursive_serde_register_type(torch.Tensor)

    def torch_serialize(tensor: torch.Tensor) -> bytes:
        return numpy_serialize(tensor.numpy())

    def torch_deserialize(buffer: bytes) -> torch.tensor:
        np_array = numpy_deserialize(buffer)
        return torch.from_numpy(np_array)

    recursive_serde_register(
        torch.Tensor,
        serialize=torch_serialize,
        deserialize=lambda data: torch_deserialize(data),
    )

except Exception:  # nosec
    pass

# unsure why we have to register the object not the type but this works
recursive_serde_register(np.core._ufunc_config._unspecified())

recursive_serde_register(
    pydantic.EmailStr,
    serialize=lambda x: x.encode(),
    deserialize=lambda x: pydantic.EmailStr(x.decode()),
)


# how else do you import a relative file to execute it?
NOTHING = None

try:
    # Just register these serializers if the google.cloud.bigquery & db_dtypes module are available
    # third party
    from google.cloud.bigquery.job.query import QueryJob
    from google.cloud.bigquery.table import RowIterator

    # Checking db_dtypes availability this way to avoid unused ruff issues, but this package is used internally
    if not find_spec("db_dtypes"):
        raise ImportError("db_dtypes module not found")

    def convert_to_dataframe(obj: RowIterator) -> bytes:
        dataframe = obj.to_dataframe()
        return serialize_dataframe(dataframe)

    def convert_from_dataframe(blob: bytes) -> DataFrame:
        dataframe = deserialize_dataframe(blob)
        return dataframe

    recursive_serde_register(
        RowIterator, serialize=convert_to_dataframe, deserialize=convert_from_dataframe
    )

    recursive_serde_register(
        QueryJob,
        serialize=lambda obj: convert_to_dataframe(obj.result()),
        deserialize=convert_from_dataframe,
    )
except ImportError:
    pass
