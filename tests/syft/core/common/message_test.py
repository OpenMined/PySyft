# external imports
import json
import uuid
import pytest
from nacl.signing import SigningKey, VerifyKey
from nacl.exceptions import BadSignatureError


# syft imports
import syft as sy
from syft.core.common.message import SignedMessage
from syft.core.common import ObjectWithID
from syft.core.common import UID
from syft.util import get_fully_qualified_name


def get_signing_key() -> SigningKey:
    # return a the signing key used to sign the get_signed_message_bytes fixture
    key = "e89ff2e651b42393b6ecb5956419088781309d953d72bd73a0968525a3a6a951"
    return SigningKey(bytes.fromhex(key))


def get_signed_message_bytes() -> bytes:
    # return a signed message fixture containing the uid from get_uid
    content = {
        "objType": "syft.core.common.object.ObjectWithID",
        "signature": (
            "SwyOFhFnAdqFMWDO8XDUGgsclW3wjMi52nOhz8HvfQ4jE+dz2hqrx04TT6o/oSny3JER8EBabM"
            + "dqT8j9aUamDA=="
        ),
        "verifyKey": "gf/M/DfEVS6KKh8iPTAQxO+IyDAB8H0zC9SXrS9Qjw8=",
        "data": (
            "eyJvYmpUeXBlIjogInN5ZnQuY29yZS5jb21tb24ub2JqZWN0Lk9iamVjdFdpdGhJRCIsICJjb2"
            + "50ZW50IjogIntcImlkXCI6IHtcInZhbHVlXCI6IFwiK3h1d1oxdTNURW0renVjQXF3b1ZGQT"
            + "09XCJ9fSJ9"
        ),
    }
    main = {
        "objType": "syft.core.common.message.SignedMessage",
        "content": json.dumps(content),
    }
    blob = bytes(json.dumps(main), "utf-8")
    return blob


def get_uid() -> ObjectWithID:
    # return a uid fixture
    uid = UID(value=uuid.UUID(int=333779996850170035686993356951732753684))
    obj = ObjectWithID(id=uid)
    return obj


def get_verify_key() -> VerifyKey:
    # return the verification key derived from the get_signing_key signing key
    return get_signing_key().verify_key


def test_create_signed_message() -> None:
    """Tests that SignedMessage can be created and serialized"""

    # we will be signing this serializable object
    obj = get_uid()
    signing_key = get_signing_key()
    signed_message = signing_key.sign(obj.serialize(to_binary=True))

    sig_msg = SignedMessage(
        obj_type=get_fully_qualified_name(obj=obj),
        signature=signed_message.signature,
        verify_key=bytes(signing_key.verify_key),
        message=signed_message.message,
    )

    assert sig_msg.obj_type == get_fully_qualified_name(obj=obj)
    assert str(sig_msg.obj_type) == "syft.core.common.object.ObjectWithID"
    assert type(sig_msg.signature) == bytes
    assert type(sig_msg.data) == bytes
    assert sig_msg.signature == signed_message.signature
    assert sig_msg.verify_key == bytes(get_verify_key())
    assert sig_msg.data == signed_message.message

    assert get_signed_message_bytes() == sig_msg.serialize(to_binary=True)


def test_deserialize_signed_message() -> None:
    """Tests that SignedMessage can be deserialized"""

    blob = get_signed_message_bytes()
    sig_msg = sy.deserialize(blob=blob, from_binary=True)

    obj = get_uid()
    signing_key = get_signing_key()
    signed_message = signing_key.sign(obj.serialize(to_binary=True))
    sig_msg_comp = SignedMessage(
        obj_type=get_fully_qualified_name(obj=obj),
        signature=signed_message.signature,
        verify_key=bytes(get_verify_key()),
        message=signed_message.message,
    )

    assert sig_msg.obj_type == sig_msg_comp.obj_type
    assert str(sig_msg.obj_type) == "syft.core.common.object.ObjectWithID"
    assert type(sig_msg.signature) == bytes
    assert type(sig_msg.verify_key) == bytes
    assert type(sig_msg.data) == bytes
    assert sig_msg.signature == sig_msg_comp.signature
    assert sig_msg.verify_key == bytes(get_verify_key())
    assert sig_msg.data == sig_msg_comp.data

    assert sig_msg.serialize(to_binary=True) == sig_msg_comp.serialize(to_binary=True)


def test_verify_message() -> None:
    """Tests that SignedMessage can be verified"""

    blob = get_signed_message_bytes()
    sig_msg = sy.deserialize(blob=blob, from_binary=True)

    veri_msg = sig_msg.inner_message()
    obj = get_uid()

    assert veri_msg == obj


def test_verify_message_fails_key() -> None:
    """Tests that SignedMessage cant be verified with the wrong verification key"""

    blob = get_signed_message_bytes()
    sig_msg = sy.deserialize(blob=blob, from_binary=True)

    signing_key = SigningKey.generate()
    sig_msg.verify_key = bytes(signing_key.verify_key)
    with pytest.raises(BadSignatureError):
        _ = sig_msg.inner_message()


def test_verify_message_fails_sig() -> None:
    """Tests that SignedMessage cant be verified with the wrong signature"""

    blob = get_signed_message_bytes()
    sig_msg = sy.deserialize(blob=blob, from_binary=True)

    sig_msg.signature += b"a"
    with pytest.raises(BadSignatureError):
        _ = sig_msg.inner_message()


def test_verify_message_fails_message() -> None:
    """Tests that SignedMessage cant be verified with the wrong message"""

    blob = get_signed_message_bytes()
    sig_msg = sy.deserialize(blob=blob, from_binary=True)

    sig_msg.data += b"a"
    with pytest.raises(BadSignatureError):
        _ = sig_msg.inner_message()


def test_verify_message_fails_empty() -> None:
    """Tests that SignedMessage cant be verified with empty sig"""

    blob = get_signed_message_bytes()
    sig_msg = sy.deserialize(blob=blob, from_binary=True)

    sig_msg.signature = b""
    with pytest.raises(BadSignatureError):
        _ = sig_msg.inner_message()


def test_decode_message() -> None:
    """Tests that SignedMessage data is not encrypted"""

    blob = get_signed_message_bytes()
    sig_msg = sy.deserialize(blob=blob, from_binary=True)

    nonveri_msg = sy.deserialize(blob=sig_msg.data, from_binary=True)
    obj = get_uid()

    assert nonveri_msg == obj


def test_get_inner_message() -> None:
    """Tests that SignedMessage verification can be ignored"""

    blob = get_signed_message_bytes()
    sig_msg = sy.deserialize(blob=blob, from_binary=True)

    sig_msg.signature += b"a"
    nonveri_msg = sig_msg.inner_message(allow_invalid=True)
    obj = get_uid()

    assert nonveri_msg == obj


def test_verify_message_fails_type() -> None:
    """Tests that SignedMessage cant be verified with the wrong storage type"""

    blob = get_signed_message_bytes()
    sig_msg = sy.deserialize(blob=blob, from_binary=True)

    sig_msg.obj_type = "BadClass"
    with pytest.raises(KeyError):
        _ = sig_msg.inner_message()

    sig_msg.obj_type = "syft.core.common.message.SyftMessage"
    with pytest.raises(TypeError):
        _ = sig_msg.inner_message()


def test_verify_message_is_not_valid() -> None:
    """Tests that SignedMessage is_valid returns False on invalid messages"""

    blob = get_signed_message_bytes()
    sig_msg = sy.deserialize(blob=blob, from_binary=True)

    sig_msg.signature = b""
    with pytest.raises(BadSignatureError):
        _ = sig_msg.inner_message()

    assert False is sig_msg.is_valid()


def test_verify_message_is_valid() -> None:
    """Tests that SignedMessage is_valid returns True on a valid messages"""

    blob = get_signed_message_bytes()
    sig_msg = sy.deserialize(blob=blob, from_binary=True)

    _ = sig_msg.inner_message()
    assert True is sig_msg.is_valid()
