# third party
from nacl.signing import SigningKey
from nacl.signing import VerifyKey

# relative
from .recursive_primitives import recursive_serde_register

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

# how else do you import a relative file to execute it?
NOTHING = None
