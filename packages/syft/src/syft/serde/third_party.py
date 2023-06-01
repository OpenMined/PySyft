# stdlib
from datetime import date
from datetime import datetime
from datetime import time
from io import BytesIO

# third party
from dateutil import parser
from jax import numpy as jnp
from jaxlib.xla_extension import ArrayImpl
from nacl.signing import SigningKey
from nacl.signing import VerifyKey
import networkx as nx
from networkx import DiGraph
import numpy as np
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
import zmq.green as zmq

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


# result Ok and Err
recursive_serde_register(Ok, serialize_attrs=["_value"])
recursive_serde_register(Err, serialize_attrs=["_value"])

recursive_serde_register_type(pydantic.main.ModelMetaclass)
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

# jax
recursive_serde_register(
    ArrayImpl,
    serialize=lambda x: serialize(np.array(x), to_bytes=True),
    deserialize=lambda x: jnp.array(deserialize(x, from_bytes=True)),
)


# unsure why we have to register the object not the type but this works
recursive_serde_register(np.core._ufunc_config._unspecified())

recursive_serde_register(
    pydantic.networks.EmailStr,
    serialize=lambda x: x.encode(),
    deserialize=lambda x: pydantic.networks.EmailStr(x.decode()),
)

recursive_serde_register(
    zmq._Socket,
    serialize_attrs=[
        "_shadow",
        "_monitor_socket",
        "_type_name",
    ],
)
recursive_serde_register(zmq._Context)

# how else do you import a relative file to execute it?
NOTHING = None


# TODO: debug serializing after updating a node
def serialize_networkx_graph(graph: DiGraph) -> bytes:
    graph_dict: dict = nx.node_link_data(graph)
    return serialize(graph_dict, to_bytes=True)


def deserialize_networkx_graph(buf: bytes) -> DiGraph:
    graph_dict: dict = deserialize(buf, from_bytes=True)
    return nx.node_link_graph(graph_dict)


recursive_serde_register(
    DiGraph,
    serialize=serialize_networkx_graph,
    deserialize=deserialize_networkx_graph,
)
