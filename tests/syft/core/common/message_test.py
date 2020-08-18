# external imports
import json
from nacl.signing import SigningKey, VerifyKey


# syft imports
import syft as sy
from syft.core.common.message import SignedImmediateSyftMessageWithoutReply
from syft import ReprMessage
from syft.util import get_fully_qualified_name


def get_signing_key() -> SigningKey:
    # return a the signing key used to sign the get_signed_message_bytes fixture
    key = "e89ff2e651b42393b6ecb5956419088781309d953d72bd73a0968525a3a6a951"
    return SigningKey(bytes.fromhex(key))


def get_signed_message_bytes() -> bytes:
    # return a signed message fixture containing the uid from get_uid
    message = {
        "msgId": {"value": "OSCFpuBSSeGS6TrP0S7Dkw=="},
        "objType": "syft.core.node.common.service.repr_service.ReprMessage",
        "signature": (
            "642MAzjJBunVfkUw9oKr/px8hWzoBngd623f4fxH8tBAv1EFagvotgkvEAXkj4UN4rpbTMZfyv"
            + "4vXRBQfBryAg=="
        ),
        "verifyKey": "gf/M/DfEVS6KKh8iPTAQxO+IyDAB8H0zC9SXrS9Qjw8=",
        "message": (
            "eyJvYmpUeXBlIjogInN5ZnQuY29yZS5ub2RlLmNvbW1vbi5zZXJ2aWNlLnJlcHJfc2VydmljZS"
            + "5SZXByTWVzc2FnZSIsICJjb250ZW50IjogIntcIm1zZ0lkXCI6IHtcInZhbHVlXCI6IFwiT1"
            + "NDRnB1QlNTZUdTNlRyUDBTN0Rrdz09XCJ9fSJ9"
        ),
    }
    envelope = {
        "objType": "syft.core.common.message.SignedImmediateSyftMessageWithoutReply",
        "content": json.dumps(message),
    }
    blob = bytes(json.dumps(envelope), "utf-8")
    return blob


def get_repr_message_bytes() -> bytes:
    content = {"msgId": {"value": "OSCFpuBSSeGS6TrP0S7Dkw=="}}
    message = {
        "objType": "syft.core.node.common.service.repr_service.ReprMessage",
        "content": json.dumps(content),
    }

    blob = bytes(json.dumps(message), "utf-8")
    return blob


def get_repr_message() -> ReprMessage:
    # return a repr message fixture
    blob = get_repr_message_bytes()
    return sy.deserialize(blob=blob, from_json=True, from_binary=True)


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
    assert sig_msg.serialized_message == msg.serialize(to_binary=True)


def test_deserialize_signed_message() -> None:
    """Tests that SignedMessage can be deserialized"""

    sig_msg_blob = get_signed_message_bytes()
    sig_msg = sy.deserialize(blob=sig_msg_blob, from_binary=True)

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
    sig_msg = sy.deserialize(blob=sig_msg_blob, from_binary=True)
    assert type(sig_msg) == SignedImmediateSyftMessageWithoutReply

    # reserial should be same as original fixture
    comp_blob = sig_msg.serialize(to_binary=True)
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
    sig_msg = sy.deserialize(blob=blob, from_binary=True)

    veri_msg = sig_msg.message
    obj = get_repr_message()

    assert veri_msg == obj


def test_verify_message_fails_key() -> None:
    """Tests that SignedMessage cant be verified with the wrong verification key"""

    blob = get_signed_message_bytes()
    sig_msg = sy.deserialize(blob=blob, from_binary=True)

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
    sig_msg = sy.deserialize(blob=blob, from_binary=True)

    # everything is good
    assert sig_msg.is_valid is True

    # change signature
    sig_msg.signature += b"a"

    # not so good
    assert sig_msg.is_valid is False


def test_verify_message_fails_message() -> None:
    """Tests that SignedMessage cant be verified with the wrong message"""

    blob = get_signed_message_bytes()
    sig_msg = sy.deserialize(blob=blob, from_binary=True)

    # everything is good
    assert sig_msg.is_valid is True

    # change message
    sig_msg.serialized_message += b"a"

    # not so good
    assert sig_msg.is_valid is False


def test_verify_message_fails_empty() -> None:
    """Tests that SignedMessage cant be verified with empty sig"""

    blob = get_signed_message_bytes()
    sig_msg = sy.deserialize(blob=blob, from_binary=True)

    # everything is good
    assert sig_msg.is_valid is True

    # change message
    sig_msg.signature = b""

    # not so good
    assert sig_msg.is_valid is False


def test_decode_message() -> None:
    """Tests that SignedMessage serialized_message is not encrypted"""

    blob = get_signed_message_bytes()
    sig_msg = sy.deserialize(blob=blob, from_binary=True)

    nonveri_msg = sy.deserialize(blob=sig_msg.serialized_message, from_binary=True)
    obj = get_repr_message()

    assert nonveri_msg == obj


def test_get_message() -> None:
    """Tests that SignedMessage verification can be ignored"""

    blob = get_signed_message_bytes()
    sig_msg = sy.deserialize(blob=blob, from_binary=True)

    sig_msg.signature += b"a"
    nonveri_msg = sig_msg.message
    obj = get_repr_message()

    assert nonveri_msg == obj
