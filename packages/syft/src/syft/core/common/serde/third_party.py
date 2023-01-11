# third party
from nacl.signing import SigningKey
from nacl.signing import VerifyKey
import pydantic
from result import Err
from result import Ok
from result import Result

# relative
from .recursive_primitives import recursive_serde_register
from .recursive_primitives import recursive_serde_register_type

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
recursive_serde_register(cls=Ok, attr_allowlist=["_value"])
recursive_serde_register(cls=Err, attr_allowlist=["_value"])

recursive_serde_register_type(pydantic.main.ModelMetaclass)
recursive_serde_register_type(Result)

# exceptions
recursive_serde_register(cls=TypeError)

# how else do you import a relative file to execute it?
NOTHING = None
