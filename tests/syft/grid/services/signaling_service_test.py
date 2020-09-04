# syft absolute
import syft as sy
from syft.grid.services.signaling_service import SignalingAnswerMessage
from syft.grid.services.signaling_service import SignalingOfferMessage
from syft.grid.services.signaling_service import OfferPullRequestMessage
from syft.grid.services.signaling_service import AnswerPullRequestMessage

from syft.core.io.address import Address


def test_signaling_offer_message_serde() -> None:
    bob_vm = sy.VirtualMachine(name="Bob")
    target = Address(name="Alice")

    msg = SignalingOfferMessage(
        address=target,
        payload="Test Payload",
        target_metadata=bob_vm.get_metadata_for_client(),
        reply_to=bob_vm.address,
    )

    blob = msg.serialize()
    msg2 = sy.deserialize(blob=blob)

    assert msg.id == msg2.id
    assert msg.address == target
    assert msg.payload == msg2.payload
    assert msg2.payload == "Test Payload"
    assert msg2.target_metadata == bob_vm.get_metadata_for_client()
    assert msg == msg2


def test_signaling_answer_message_serde() -> None:
    bob_vm = sy.VirtualMachine(name="Bob")
    target = Address(name="Alice")

    msg = SignalingAnswerMessage(
        address=target,
        payload="Test Payload",
        target_metadata=bob_vm.get_metadata_for_client(),
        reply_to=bob_vm.address,
    )

    blob = msg.serialize()
    msg2 = sy.deserialize(blob=blob)

    assert msg.id == msg2.id
    assert msg.address == target
    assert msg.payload == msg2.payload
    assert msg2.payload == "Test Payload"
    assert msg2.target_metadata == bob_vm.get_metadata_for_client()
    assert msg == msg2


def test_signaling_answer_pull_request_message_serde() -> None:
    bob_vm = sy.VirtualMachine(name="Bob")
    target = Address(name="Alice")

    msg = AnswerPullRequestMessage(address=target, reply_to=bob_vm.address)

    blob = msg.serialize()
    msg2 = sy.deserialize(blob=blob)

    assert msg.id == msg2.id
    assert msg.address == target
    assert msg == msg2


def test_signaling_offer_pull_request_message_serde() -> None:
    bob_vm = sy.VirtualMachine(name="Bob")
    target = Address(name="Alice")

    msg = OfferPullRequestMessage(address=target, reply_to=bob_vm.address)

    blob = msg.serialize()
    msg2 = sy.deserialize(blob=blob)

    assert msg.id == msg2.id
    assert msg.address == target
    assert msg == msg2
