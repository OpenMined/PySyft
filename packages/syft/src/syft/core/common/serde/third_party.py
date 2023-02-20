# stdlib

# third party
from nacl.signing import SigningKey
from nacl.signing import VerifyKey
from oblv.oblv_client import OblvClient
from pandas import DataFrame
from pandas import Series
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


# pandas
recursive_serde_register(
    DataFrame,
    serialize=lambda x: serialize(x.to_dict(), to_bytes=True),
    deserialize=lambda x: DataFrame.from_dict(deserialize(x, from_bytes=True)),
)


def deserialize_series(blob: bytes) -> Series:
    df = DataFrame.from_dict(deserialize(blob, from_bytes=True))
    return df[df.columns[0]]


recursive_serde_register(
    Series,
    serialize=lambda x: serialize(DataFrame(x).to_dict(), to_bytes=True),
    deserialize=deserialize_series,
)

# how else do you import a relative file to execute it?
NOTHING = None
