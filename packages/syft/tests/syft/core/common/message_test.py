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


def get_signed_message_bytes() -> bytes:
    # return a signed message fixture containing the uid from get_uid
    return (
        b"\n\t"
        + PROTOBUF_START_MAGIC_HEADER_BYTES
        + b"\x12?syft.core.common.message.SignedImmediateSyftMessageWithoutReply"
        + b"\x1a\xe8\x02\n\x12\n\x10\xfb\x1b\xb0g[\xb7LI\xbe\xce\xe7\x00\xab\n\x15\x14"
        + b"\x12Lsyft.core.node.common.node_service.testing_services.repr_service."
        + b"ReprMessage\x1a@9\x1c\x9d\xd5\xb8\xed\x02=\xac`\xb4\x7f\xb8}j\x11G%\xd1\xaa"
        + b"\x1f\x07s\xaa\xc7z\xd2\xeb\x18\x18\xce\x1c\xfdS\x93\xb7\\\xdb\x89\xe1\x7f"
        + b"\xe2\x9d\xa4\xdd#\xc5\x1c\x1ci\xb3\x1c\xba\xfd\xeeD\x19\xbc\xea\xcdh\xea"
        + b'\xfc\x0c" \x81\xff\xcc\xfc7\xc4U.\x8a*\x1f"=0\x10\xc4\xef\x88\xc80\x01'
        + b"\xf0}3\x0b\xd4\x97\xad/P\x8f\x0f*\x9f\x01\n\tprotobuf:\x12Lsyft.core.node."
        + b"common.node_service.testing_services.repr_service.ReprMessage\x1aD\n\x12\n"
        + b"\x10\xfb\x1b\xb0g[\xb7LI\xbe\xce\xe7\x00\xab\n\x15\x14\x12.\n\ralices_domain"
        + b"(\x012\x1b\n\x12\n\x10\xfb\x1b\xb0g[\xb7LI\xbe\xce\xe7\x00\xab\n\x15\x14\x12"
        + b"\x05alice"
    )


def get_repr_message_bytes() -> bytes:
    return (
        b"\n\t"
        + PROTOBUF_START_MAGIC_HEADER_BYTES
        + b"\x12Lsyft.core.node.common.node_service.testing_services.repr_service."
        + b"ReprMessage\x1aD\n\x12\n\x10\xfb\x1b\xb0g[\xb7LI\xbe\xce\xe7\x00\xab\n\x15"
        + b"\x14\x12.\n\ralices_domain(\x012\x1b\n\x12\n\x10\xfb\x1b\xb0g[\xb7LI\xbe"
        + b"\xce\xe7\x00\xab\n\x15\x14\x12\x05alice"
    )


def get_repr_message() -> ReprMessage:
    uid = UID(value=uuid.UUID(int=333779996850170035686993356951732753684))
    addr = Address(name="alices_domain", domain=SpecificLocation(id=uid, name="alice"))
    return ReprMessage(address=addr, msg_id=uid)


def test_repr_message() -> None:
    msg = get_repr_message()
    msg_bytes = sy.serialize(msg, to_bytes=True)
    assert msg_bytes == get_repr_message_bytes()


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
