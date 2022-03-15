# third party
from nacl.signing import SigningKey
from nacl.signing import VerifyKey
import pytest

# syft absolute
import syft as sy
from syft import ReprMessage
from syft import serialize
from syft.core.common.message import SignedImmediateSyftMessageWithoutReply
from syft.util import get_fully_qualified_name


def get_signing_key() -> SigningKey:
    # return a the signing key used to sign the get_signed_message_bytes fixture
    key = "e89ff2e651b42393b6ecb5956419088781309d953d72bd73a0968525a3a6a951"
    return SigningKey(bytes.fromhex(key))


def get_signed_message_bytes() -> bytes:
    # return a signed message fixture containing the uid from get_uid
    blob = (
        b"\n?syft.core.common.message.SignedImmediateSyftMessageWithoutReply"
        + b"\x12\xd5\x02\n\x12\n\x10o\xdeI!\x8eH@\xf7\x89xQ7\x8dWN\x8b\x12"
        + b"Lsyft.core.node.common.node_service.testing_services.repr_service.ReprMessage"
        + b"\x1a@\xfe\xc3\xc9\xe4\xb7a\xc1n\xa8t\xb9\xe6n\x0c\x89\xd4Om~c\xb4\xfe\xb5\x9e\xa5"
        + b"\x19\xdeD\x18\xa8\x82zd\x11\xd9bZ<\xa6\xf4\xcb\xf6v\xc9P\xeb\x91`N\x8b\x13%\xd1\xc41"
        + b'\xbe\x18\xa22\x81B\x8f\xc2\x04" \x81\xff\xcc\xfc7\xc4U.\x8a*\x1f"=0\x10\xc4\xef\x88\xc80'
        + b"\x01\xf0}3\x0b\xd4\x97\xad/P\x8f\x0f*\x8c\x01"
        + b"\nLsyft.core.node.common.node_service.testing_services.repr_service.ReprMessage"
        + b"\x12<\n\x12\n\x10o\xdeI!\x8eH@\xf7\x89xQ7\x8dWN\x8b\x12&\n\x05alice(\x012\x1b\n\x12\n\x10"
        + b'\x8b\x8cU\x94\xad@E\x95\x8f\x9a\x8c\x10#"\x12\xb7\x12\x05alice'
    )
    return blob


def get_repr_message_bytes() -> bytes:
    blob = (
        b"\nLsyft.core.node.common.node_service.testing_services.repr_service.ReprMessage\x12<\n"
        + b"\x12\n\x10o\xdeI!\x8eH@\xf7\x89xQ7\x8dWN\x8b\x12&\n\x05alice(\x012\x1b\n\x12\n\x10\x8b"
        + b'\x8cU\x94\xad@E\x95\x8f\x9a\x8c\x10#"\x12\xb7\x12\x05alice'
    )
    return blob


def get_repr_message() -> ReprMessage:
    # return a repr message fixture
    blob = get_repr_message_bytes()
    return sy.deserialize(blob=blob, from_bytes=True)


def get_verify_key() -> VerifyKey:
    # return the verification key derived from the get_signing_key signing key
    return get_signing_key().verify_key


def test_create_signed_message() -> None:
    """Tests that SignedMessage can be created and serialized"""

    # we will be signing this serializable object
    msg = get_repr_message()
    signing_key = get_signing_key()
    sig_msg = msg.sign(signing_key=signing_key)

    assert sig_msg.obj_type == get_fully_qualified_name(obj=msg)
    assert (
        str(sig_msg.obj_type)
        == "syft.core.node.common.node_service.testing_services.repr_service.ReprMessage"
    )
    assert type(sig_msg.signature) == bytes

    assert get_fully_qualified_name(obj=msg) in str(type(sig_msg.message))
    assert len(sig_msg.signature) > 0
    assert sig_msg.verify_key == get_verify_key()
    assert sig_msg.message == msg
    assert sig_msg.serialized_message == serialize(msg, to_bytes=True)


def test_deserialize_signed_message() -> None:
    """Tests that SignedMessage can be deserialized"""

    sig_msg_blob = get_signed_message_bytes()
    sig_msg = sy.deserialize(blob=sig_msg_blob, from_bytes=True)

    msg = get_repr_message()
    signing_key = get_signing_key()
    sig_msg_comp = msg.sign(signing_key=signing_key)

    assert sig_msg.obj_type == sig_msg_comp.obj_type
    assert (
        str(sig_msg.obj_type)
        == "syft.core.node.common.node_service.testing_services.repr_service.ReprMessage"
    )

    assert type(sig_msg.signature) == bytes
    assert type(sig_msg.verify_key) == VerifyKey
    assert sig_msg.signature == sig_msg_comp.signature
    assert sig_msg.verify_key == get_verify_key()
    assert sig_msg.message == msg
    assert sig_msg.message == sig_msg_comp.message
    assert sig_msg.serialized_message == sig_msg_comp.serialized_message


def test_serde_matches() -> None:
    """Tests that the nested serde is reversible at all levels"""

    # serial
    sig_msg_blob = get_signed_message_bytes()

    # deserial should be expected type
    sig_msg = sy.deserialize(blob=sig_msg_blob, from_bytes=True)
    assert type(sig_msg) == SignedImmediateSyftMessageWithoutReply

    # reserial should be same as original fixture
    comp_blob = serialize(sig_msg, to_bytes=True)
    assert type(comp_blob) == bytes
    assert comp_blob == sig_msg_blob

    # now try sub message
    msg = sig_msg.message
    assert type(msg) == ReprMessage

    # resign and the result should be the same
    signing_key = get_signing_key()
    sig_msg_comp = msg.sign(signing_key=signing_key)
    assert type(sig_msg_comp) == SignedImmediateSyftMessageWithoutReply
    assert type(sig_msg_comp) == type(sig_msg)

    # make sure they have the same message id for comparison
    sig_msg_comp._id = sig_msg.id
    # identical (except the auto generated UID for the envelope)
    assert sig_msg_comp == sig_msg


def test_verify_message() -> None:
    """Tests that SignedMessage can be verified"""

    blob = get_signed_message_bytes()
    sig_msg = sy.deserialize(blob=blob, from_bytes=True)

    veri_msg = sig_msg.message
    obj = get_repr_message()

    assert veri_msg == obj


def test_verify_message_fails_key() -> None:
    """Tests that SignedMessage cant be verified with the wrong verification key"""

    blob = get_signed_message_bytes()
    sig_msg = sy.deserialize(blob=blob, from_bytes=True)

    # everything is good
    assert sig_msg.is_valid is True

    # change verify_key
    signing_key = SigningKey.generate()
    sig_msg.verify_key = signing_key.verify_key

    # not so good
    assert sig_msg.is_valid is False


def test_verify_message_fails_sig() -> None:
    """Tests that SignedMessage cant be verified with the wrong signature"""

    blob = get_signed_message_bytes()
    sig_msg = sy.deserialize(blob=blob, from_bytes=True)

    # everything is good
    assert sig_msg.is_valid is True

    # change signature
    sig = list(sig_msg.signature)
    sig[-1] = 0  # change last byte
    sig_msg.signature = bytes(sig)

    # not so good
    assert sig_msg.is_valid is False


def test_verify_message_fails_message() -> None:
    """Tests that SignedMessage cant be verified with the wrong message"""

    blob = get_signed_message_bytes()
    sig_msg = sy.deserialize(blob=blob, from_bytes=True)

    # everything is good
    assert sig_msg.is_valid is True

    # change message
    sig_msg.serialized_message += b"a"

    # not so good
    assert sig_msg.is_valid is False


def test_verify_message_fails_empty() -> None:
    """Tests that SignedMessage cant be verified with empty sig"""

    blob = get_signed_message_bytes()
    sig_msg = sy.deserialize(blob=blob, from_bytes=True)

    # everything is good
    assert sig_msg.is_valid is True

    # change message
    sig_msg.signature = b""

    # not so good
    with pytest.raises(ValueError):
        assert sig_msg.is_valid is False


def test_decode_message() -> None:
    """Tests that SignedMessage serialized_message is not encrypted"""

    blob = get_signed_message_bytes()
    sig_msg = sy.deserialize(blob=blob, from_bytes=True)

    nonveri_msg = sy.deserialize(blob=sig_msg.serialized_message, from_bytes=True)
    obj = get_repr_message()

    assert nonveri_msg == obj


def test_get_message() -> None:
    """Tests that SignedMessage verification can be ignored"""

    blob = get_signed_message_bytes()
    sig_msg = sy.deserialize(blob=blob, from_bytes=True)

    sig_msg.signature += b"a"
    nonveri_msg = sig_msg.message
    obj = get_repr_message()

    assert nonveri_msg == obj
