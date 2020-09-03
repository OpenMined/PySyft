# syft absolute
import syft as sy
from syft.grid.services.signaling_service import SignalingAnswerMessage
from syft.grid.services.signaling_service import SignalingOfferMessage


def test_signaling_offer_message_serde() -> None:
    bob_vm = sy.VirtualMachine(name="Bob")

    msg = SignalingOfferMessage(address=bob_vm.address, payload="Test Payload")

    blob = msg.serialize()
    msg2 = sy.deserialize(blob=blob)

    assert msg.id == msg2.id
    assert msg.address == msg2.address
    assert msg.payload == msg2.payload
    assert msg2.payload == "Test Payload"
    assert msg == msg2


def test_signaling_answer_message_serde() -> None:
    bob_vm = sy.VirtualMachine(name="Bob")

    msg = SignalingAnswerMessage(address=bob_vm.address, payload="Test Payload")

    blob = msg.serialize()
    msg2 = sy.deserialize(blob=blob)

    assert msg.id == msg2.id
    assert msg.address == msg2.address
    assert msg.payload == msg2.payload
    assert msg2.payload == "Test Payload"
    assert msg == msg2
