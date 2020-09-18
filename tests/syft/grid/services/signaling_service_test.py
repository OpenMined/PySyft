# stdlib
from typing import Tuple

# syft absolute
import syft as sy
from syft.core.io.address import Address
from syft.core.node.common.node import Node
from syft.grid.services.signaling_service import AnswerPullRequestMessage
from syft.grid.services.signaling_service import OfferPullRequestMessage
from syft.grid.services.signaling_service import PullSignalingService
from syft.grid.services.signaling_service import PushSignalingService
from syft.grid.services.signaling_service import SignalingAnswerMessage
from syft.grid.services.signaling_service import SignalingOfferMessage
from syft.grid.services.signaling_service import SignalingRequestsNotFound


def get_preset_nodes() -> Tuple[Node, Node, Node]:
    om_network = sy.Network(name="OpenMined")
    om_network.immediate_services_without_reply.append(PushSignalingService)
    om_network.immediate_services_with_reply.append(PullSignalingService)
    om_network._register_services()  # re-register all services including SignalingService
    bob_vm = sy.VirtualMachine(name="Bob")
    alice_vm = sy.VirtualMachine(name="Alice")
    return om_network, bob_vm, alice_vm


def test_signaling_offer_message_serde() -> None:
    bob_vm = sy.VirtualMachine(name="Bob")
    target = Address(name="Alice")

    msg = SignalingOfferMessage(
        address=target,
        payload="SDP",
        host_metadata=bob_vm.get_metadata_for_client(),
        target_peer=target,
        host_peer=bob_vm.address,
    )

    blob = msg.serialize()
    msg2 = sy.deserialize(blob=blob)

    assert msg.id == msg2.id
    assert msg.address == target
    assert msg.payload == msg2.payload
    assert msg2.payload == "SDP"
    assert msg2.host_metadata == bob_vm.get_metadata_for_client()
    assert msg == msg2


def test_signaling_answer_message_serde() -> None:
    bob_vm = sy.VirtualMachine(name="Bob")
    target = Address(name="Alice")

    msg = SignalingAnswerMessage(
        address=target,
        payload="SDP",
        host_metadata=bob_vm.get_metadata_for_client(),
        target_peer=target,
        host_peer=bob_vm.address,
    )

    blob = msg.serialize()
    msg2 = sy.deserialize(blob=blob)

    assert msg.id == msg2.id
    assert msg.address == target
    assert msg.payload == msg2.payload
    assert msg2.payload == "SDP"
    assert msg2.host_metadata == bob_vm.get_metadata_for_client()
    assert msg == msg2


def test_signaling_answer_pull_request_message_serde() -> None:
    bob_vm = sy.VirtualMachine(name="Bob")
    target = Address(name="Alice")

    msg = AnswerPullRequestMessage(
        address=target,
        target_peer=target,
        host_peer=bob_vm.address,
        reply_to=bob_vm.address,
    )

    blob = msg.serialize()
    msg2 = sy.deserialize(blob=blob)

    assert msg.id == msg2.id
    assert msg.address == target
    assert msg == msg2


def test_signaling_offer_pull_request_message_serde() -> None:
    bob_vm = sy.VirtualMachine(name="Bob")
    target = Address(name="Alice")

    msg = OfferPullRequestMessage(
        address=target,
        target_peer=target,
        host_peer=bob_vm.address,
        reply_to=bob_vm.address,
    )

    blob = msg.serialize()
    msg2 = sy.deserialize(blob=blob)

    assert msg.id == msg2.id
    assert msg.address == target
    assert msg == msg2


def test_push_offer_signaling_service() -> None:
    om_network, bob_vm, alice_vm = get_preset_nodes()
    om_network_client = om_network.get_root_client()

    offer_msg = SignalingOfferMessage(
        address=om_network.address,
        payload="SDP",
        host_metadata=alice_vm.get_metadata_for_client(),
        target_peer=alice_vm.address,
        host_peer=bob_vm.address,
    )
    om_network_client.send_immediate_msg_without_reply(msg=offer_msg)

    assert om_network.signaling_msgs.pop(offer_msg.id) == offer_msg


def test_push_answer_signaling_service() -> None:
    om_network, bob_vm, alice_vm = get_preset_nodes()
    om_network_client = om_network.get_root_client()

    ans_msg = SignalingAnswerMessage(
        address=om_network.address,
        payload="SDP",
        host_metadata=alice_vm.get_metadata_for_client(),
        target_peer=alice_vm.address,
        host_peer=bob_vm.address,
    )
    om_network_client.send_immediate_msg_without_reply(msg=ans_msg)

    assert om_network.signaling_msgs.pop(ans_msg.id) == ans_msg


def test_pull_offer_signaling_service() -> None:
    om_network, bob_vm, alice_vm = get_preset_nodes()
    om_network_client = om_network.get_root_client()

    offer_msg = SignalingOfferMessage(
        address=om_network.address,
        payload="SDP",
        host_metadata=alice_vm.get_metadata_for_client(),
        target_peer=alice_vm.address,
        host_peer=bob_vm.address,
    )
    om_network_client.send_immediate_msg_without_reply(msg=offer_msg)

    assert om_network.signaling_msgs.get(offer_msg.id) == offer_msg

    offer_pull_req = OfferPullRequestMessage(
        address=om_network.address,
        target_peer=bob_vm.address,
        host_peer=alice_vm.address,
        reply_to=om_network_client.address,
    )

    response = om_network_client.send_immediate_msg_with_reply(msg=offer_pull_req)

    assert response == offer_msg


def test_pull_answer_signaling_service() -> None:
    om_network, bob_vm, alice_vm = get_preset_nodes()
    om_network_client = om_network.get_root_client()

    answer_msg = SignalingAnswerMessage(
        address=om_network.address,
        payload="SDP",
        host_metadata=alice_vm.get_metadata_for_client(),
        target_peer=alice_vm.address,
        host_peer=bob_vm.address,
    )
    om_network_client.send_immediate_msg_without_reply(msg=answer_msg)

    assert om_network.signaling_msgs.get(answer_msg.id) == answer_msg

    ans_pull_req = AnswerPullRequestMessage(
        address=om_network.address,
        target_peer=bob_vm.address,
        host_peer=alice_vm.address,
        reply_to=om_network_client.address,
    )

    response = om_network_client.send_immediate_msg_with_reply(msg=ans_pull_req)

    assert response == answer_msg


def test_not_found_pull_offer_requests_signaling_service() -> None:
    om_network, bob_vm, alice_vm = get_preset_nodes()
    om_network_client = om_network.get_root_client()

    offer_pull_req = OfferPullRequestMessage(
        address=om_network.address,
        target_peer=bob_vm.address,
        host_peer=alice_vm.address,
        reply_to=om_network_client.address,
    )

    response = om_network_client.send_immediate_msg_with_reply(msg=offer_pull_req)

    assert isinstance(response, SignalingRequestsNotFound)


def test_not_found_pull_ans_requests_signaling_service() -> None:
    om_network, bob_vm, alice_vm = get_preset_nodes()
    om_network_client = om_network.get_root_client()

    ans_pull_req = AnswerPullRequestMessage(
        address=om_network.address,
        target_peer=bob_vm.address,
        host_peer=alice_vm.address,
        reply_to=om_network_client.address,
    )

    response = om_network_client.send_immediate_msg_with_reply(msg=ans_pull_req)

    assert isinstance(response, SignalingRequestsNotFound)
