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
    key = "0b38fd089aaf9d3794c7845039718977770f40f5500a067935486625c475e7bd"
    return SigningKey(bytes.fromhex(key))


def get_signed_message_bytes() -> bytes:
    # return a signed message fixture containing the uid from get_uid
    blob = (
        b"\xfa\x02\x8b\x02\n\x12\n\x10\xbaU\x10\x0c`\xbcB\x06\x89\xa3\xba\x7f\n\xb4\xb8\x93"
        b"\x126syft.core.node.common.service.repr_service.ReprMessage\x1a@h;^#a\xbb\x8f\x9f"
        b"\xad\xde\xf0i\xb4\xfa\x10\xab\t\xe1qt\xb7\xbe\xd9\xba\x1di\x7f\xc5+\xbf(\xa6\x86"
        b'\xe96\xf5\xc9\x86\xad<\xb1\x18p\xe2T\xaf\xbd\xb6h\xdd\xafL\xd9\x0cg\xe5\xe3\xe5"h}'
        b'\xf5\x1e\x07" \xbd\\\xfe\x89\t\xa20\rp\xdf\xcb\xf6\x15\xfe\xa1U\r\x9a\xf1.]'
        b"\xfb4\x1cAH\xe9\x7f.\x17\xb6-*Y\xe2\x01V\n\x12\n\x10\xbaU\x10\x0c`\xbcB\x06\x89\xa3"
        b"\xba\x7f\n\xb4\xb8\x93\x12@\n\x12Zealous Montalcini8\x01B(\n\x12\n\x10P\xf2sP\x81"
        b"\xa5A\xea\xb5F\xe3J\x0e\xa0\x98\xa2\x12\x12Zealous Montalcini"
    )
    return blob


def get_repr_message_bytes() -> bytes:
    blob = (
        b"\xe2\x01V\n\x12\n\x10\xbaU\x10\x0c`\xbcB\x06\x89\xa3\xba\x7f\n\xb4\xb8\x93\x12@\n"
        b"\x12Zealous Montalcini8\x01B(\n\x12\n\x10P\xf2sP\x81\xa5A\xea\xb5F\xe3J\x0e\xa0"
        b"\x98\xa2\x12\x12Zealous Montalcini"
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
