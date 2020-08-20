import syft as sy

from syft.grid.services.signaling_service import (
    SignalingService,
    SignalingAnswerMessage,
    SignalingOfferMessage,
)


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


def test_signaling_service() -> None:
    bob_vm = sy.VirtualMachine(name="Bob")
    bob_vm.immediate_services_without_reply.append(SignalingService)
    bob_vm._register_services()  # re-register all services including SignalingService
    bob_vm_client = bob_vm.get_root_client()

    bob_vm_client.send_immediate_msg_without_reply(
        msg=SignalingAnswerMessage(address=bob_vm.address, payload="Test Payload")
    )

    # TODO add some asserts here that verify something happened
