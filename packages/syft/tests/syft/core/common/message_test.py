# third party
from nacl.signing import SigningKey
from nacl.signing import VerifyKey

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
        + b"\x12\xad\x02\n\x12\n\x10\x8c3\x19,\xcd\xd3\xf3N\xe2\xb0\xc6\tU\xdf\x02u\x12"
        + b"6syft.core.node.common.service.repr_service.ReprMessage"
        + b"\x1a@@\x82\x13\xfaC\xfb=\x01H\x853\x1e\xceE+\xc6\xb5\rX\x16Z\xb8l\x02\x10"
        + b"\x8algj\xd6U\x11]\xe9R\x0ei\xd8\xca\xb9\x00=\xa1\xeeoEa\xe2C\xa0\x960\xf7A"
        + b'\xfad<(9\xe1\x8c\x93\xf1\x0b" \x81\xff\xcc\xfc7\xc4U.\x8a*\x1f"=0\x10\xc4'
        + b"\xef\x88\xc80\x01\xf0}3\x0b\xd4\x97\xad/P\x8f\x0f*{\n6"
        + b"syft.core.node.common.service.repr_service.ReprMessage\x12A\n\x12\n\x10"
        + b"\x8c3\x19,\xcd\xd3\xf3N\xe2\xb0\xc6\tU\xdf\x02u\x12+\n\x0bGoofy KirchH\x01R"
        + b"\x1a\n\x12\n\x10\xfb\x1b\xb0g[\xb7LI\xbe\xce\xe7\x00\xab\n\x15\x14\x12\x04Test"
    )
    return blob


def get_repr_message_bytes() -> bytes:
    blob = (
        b"\n6syft.core.node.common.service.repr_service.ReprMessage\x12A\n\x12\n\x10"
        + b"\x8c3\x19,\xcd\xd3\xf3N\xe2\xb0\xc6\tU\xdf\x02u\x12+\n\x0bGoofy KirchH\x01R"
        + b"\x1a\n\x12\n\x10\xfb\x1b\xb0g[\xb7LI\xbe\xce\xe7\x00\xab\n\x15\x14\x12\x04Test"
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
        == "syft.core.node.common.service.repr_service.ReprMessage"
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
        == "syft.core.node.common.service.repr_service.ReprMessage"
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
    sig_msg.signature += b"a"

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
