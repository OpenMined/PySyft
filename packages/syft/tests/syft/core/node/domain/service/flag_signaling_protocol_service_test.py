import pytest
from syft.core.io.address import Address
from syft.core.node.domain.service.flag_signaling_protocol_service import SetProtocolMessage, SetProtocolMessageReply
from syft import serialize, deserialize
from enum import Enum


@pytest.fixture()
def protocol():
    return Enum(
        "TestEnum",
        [
            ("Continue", 1),
            ("Close", 2),
            ("ClearStore", 3)
         ]
    )

@pytest.fixture()
def set_protocol_message(protocol):
    return SetProtocolMessage(
        flags=protocol,
        reply_to=Address(),
        address=Address()
    )

@pytest.fixture()
def set_protocol_message_reply(set_protocol_message):
    return SetProtocolMessageReply(
        response=True,
        address=set_protocol_message.reply_to
    )

def test_serde_set_protocol_message(set_protocol_message):
    set_protocol_message_clone = deserialize(serialize(set_protocol_message, to_bytes=True), from_bytes=True)

    assert set_protocol_message.reply_to == set_protocol_message_clone.reply_to
    assert set_protocol_message.id == set_protocol_message_clone.id
    assert set_protocol_message.address == set_protocol_message_clone.address

    original_values = set_protocol_message.flags.__members__.values()
    back_values = set_protocol_message_clone.flags.__members__.values()

    for value1, value2 in zip(original_values, back_values):
        assert value1.name == value2.name
        assert value1.value == value2.value

def test_serde_set_protocol_message_reply(set_protocol_message_reply):
    set_protocol_message_reply_clone = deserialize(serialize(set_protocol_message_reply,
                                                             to_bytes=True), from_bytes=True)

    assert set_protocol_message_reply.id == set_protocol_message_reply_clone.id
    assert set_protocol_message_reply.address == set_protocol_message_reply_clone.address
    assert set_protocol_message_reply.response == set_protocol_message_reply_clone.response

