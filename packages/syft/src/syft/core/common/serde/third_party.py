# stdlib

# stdlib
from datetime import date
from datetime import datetime
from datetime import time

# third party
from dateutil import parser
from nacl.signing import SigningKey
from nacl.signing import VerifyKey
from oblv.oblv_client import OblvClient
from pandas import DataFrame
from pandas import Series
from pandas._libs.tslibs.timestamps import Timestamp
import pyarrow as pa
import pyarrow.parquet as pq
import pydantic
from pymongo.collection import Collection
from result import Err
from result import Ok
from result import Result

# relative
from .deserialize import _deserialize as deserialize
from .recursive_primitives import recursive_serde_register
from .recursive_primitives import recursive_serde_register_type
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

# Oblivious Client serde
recursive_serde_register(
    OblvClient,
    serialize=lambda x: serialize([x.token, x.oblivious_user_id], to_bytes=True),
    deserialize=lambda x: OblvClient(*deserialize(x, from_bytes=True)),
)

# result Ok and Err
recursive_serde_register(Ok, attr_allowlist=["_value"])
recursive_serde_register(Err, attr_allowlist=["_value"])

recursive_serde_register_type(pydantic.main.ModelMetaclass)
recursive_serde_register_type(Result, attr_allowlist=["_value"])

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
    df = DataFrame.from_dict(deserialize(blob, from_bytes=True))
    return df[df.columns[0]]


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


# how else do you import a relative file to execute it?
NOTHING = None
