# stdlib
import uuid

# third party
from nacl.signing import SigningKey
from nacl.signing import VerifyKey
import pytest

# syft absolute
import syft as sy
from syft import ReprMessage
from syft import serialize
from syft.core.common.message import SignedImmediateSyftMessageWithoutReply
from syft.core.common.serde.deserialize import PROTOBUF_START_MAGIC_HEADER_BYTES
from syft.core.common.uid import UID
from syft.core.io.address import Address
from syft.core.io.location import SpecificLocation
from syft.util import get_fully_qualified_name


def get_signing_key() -> SigningKey:
    # return a the signing key used to sign the get_signed_message_bytes fixture
    key = "e89ff2e651b42393b6ecb5956419088781309d953d72bd73a0968525a3a6a951"
    return SigningKey(bytes.fromhex(key))


def get_repr_message() -> ReprMessage:
    uid = UID(value=uuid.UUID(int=333779996850170035686993356951732753684))
    addr = Address(name="alices_domain", domain=SpecificLocation(id=uid, name="alice"))
    return ReprMessage(address=addr, msg_id=uid)


def get_verify_key() -> VerifyKey:
    # return the verification key derived from the get_signing_key signing key
    return get_signing_key().verify_key
