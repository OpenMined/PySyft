# stdlib
import secrets
from typing import Tuple

# syft absolute
import syft as sy
from syft import serialize
from syft.core.common.message import SyftMessage
from syft.core.io.address import Address
from syft.core.node.common.node import Node
from syft.grid.services.signaling_service import AnswerPullRequestMessage
from syft.grid.services.signaling_service import InvalidLoopBackRequest
from syft.grid.services.signaling_service import OfferPullRequestMessage
from syft.grid.services.signaling_service import PullSignalingService
from syft.grid.services.signaling_service import PushSignalingService
from syft.grid.services.signaling_service import RegisterDuetPeerService
from syft.grid.services.signaling_service import RegisterNewPeerMessage
from syft.grid.services.signaling_service import SignalingAnswerMessage
from syft.grid.services.signaling_service import SignalingOfferMessage
from syft.grid.services.signaling_service import SignalingRequestsNotFound


def get_preset_nodes() -> Tuple[Node, Node, Node]:
    om_network = sy.Network(name="OpenMined")
    om_network.immediate_services_without_reply.append(PushSignalingService)
    om_network.immediate_services_with_reply.append(PullSignalingService)
    om_network.immediate_services_with_reply.append(RegisterDuetPeerService)
    om_network._register_services()  # re-register all services including SignalingService
    bob_vm = sy.VirtualMachine(name="Bob")
    alice_vm = sy.VirtualMachine(name="Alice")
    return om_network, bob_vm, alice_vm


def test_signaling_offer_message_serde(node: sy.VirtualMachine) -> None:
    target = Address(name="Alice")

    target_id = secrets.token_hex(nbytes=16)
    host_id = secrets.token_hex(nbytes=16)

    msg = SignalingOfferMessage(
        address=target,
        payload="SDP",
        host_metadata=node.get_metadata_for_client(),
        target_peer=target_id,
        host_peer=host_id,
    )

    blob = serialize(msg)
    msg2 = sy.deserialize(blob=blob)

    msg_metadata = node.get_metadata_for_client()
    msg2_metadata = msg2.host_metadata

    assert msg.id == msg2.id
    assert msg.address == target
    assert msg.payload == msg2.payload
    assert msg2.payload == "SDP"
    assert msg2.host_peer == host_id
    assert msg2.target_peer == target_id
    assert msg == msg2

    assert msg_metadata.name == msg2_metadata.name
    assert msg_metadata.node == msg2_metadata.node
    assert msg_metadata.id == msg2_metadata.id


def test_signaling_answer_message_serde(node: sy.VirtualMachine) -> None:
    target = Address(name="Alice")
    target_id = secrets.token_hex(nbytes=16)
    host_id = secrets.token_hex(nbytes=16)

    msg = SignalingAnswerMessage(
        address=target,
        payload="SDP",
        host_metadata=node.get_metadata_for_client(),
        target_peer=target_id,
        host_peer=host_id,
    )
    msg_metadata = node.get_metadata_for_client()

    blob = serialize(msg)
    msg2 = sy.deserialize(blob=blob)
    msg2_metadata = msg2.host_metadata

    assert msg.id == msg2.id
    assert msg.address == target
    assert msg.payload == msg2.payload
    assert msg2.payload == "SDP"
    assert msg2.host_peer == host_id
    assert msg2.target_peer == target_id
    assert msg == msg2

    assert msg_metadata.name == msg2_metadata.name
    assert msg_metadata.node == msg2_metadata.node
    assert msg_metadata.id == msg2_metadata.id


def test_signaling_answer_pull_request_message_serde(node: sy.VirtualMachine) -> None:
    target = Address(name="Alice")

    target_id = secrets.token_hex(nbytes=16)
    host_id = secrets.token_hex(nbytes=16)

    msg = AnswerPullRequestMessage(
        address=target,
        target_peer=target_id,
        host_peer=host_id,
        reply_to=node.address,
    )

    blob = serialize(msg)
    msg2 = sy.deserialize(blob=blob)

    assert msg.id == msg2.id
    assert msg.address == target
    assert msg == msg2
    assert msg2.host_peer == host_id
    assert msg2.target_peer == target_id


def test_signaling_offer_pull_request_message_serde(node: sy.VirtualMachine) -> None:
    target = Address(name="Alice")

    target_id = secrets.token_hex(nbytes=16)
    host_id = secrets.token_hex(nbytes=16)

    msg = OfferPullRequestMessage(
        address=target,
        target_peer=target_id,
        host_peer=host_id,
        reply_to=node.address,
    )

    blob = serialize(msg)
    msg2 = sy.deserialize(blob=blob)

    assert msg.id == msg2.id
    assert msg.address == target
    assert msg == msg2
    assert msg2.host_peer == host_id
    assert msg2.target_peer == target_id


def test_push_offer_signaling_service() -> None:
    om_network, bob_vm, alice_vm = get_preset_nodes()
    om_network_client = om_network.get_root_client()

    msg = RegisterNewPeerMessage(
        address=om_network.address, reply_to=om_network_client.address
    )

    target_id = om_network_client.send_immediate_msg_with_reply(msg=msg).peer_id
    host_id = om_network_client.send_immediate_msg_with_reply(msg=msg).peer_id

    offer_msg = SignalingOfferMessage(
        address=om_network.address,
        payload="SDP",
        host_metadata=alice_vm.get_metadata_for_client(),
        target_peer=target_id,
        host_peer=host_id,
    )
    om_network_client.send_immediate_msg_without_reply(msg=offer_msg)

    assert (
        om_network.signaling_msgs[target_id][SyftMessage].pop(offer_msg.id) == offer_msg
    )


def test_push_answer_signaling_service() -> None:
    om_network, bob_vm, alice_vm = get_preset_nodes()
    om_network_client = om_network.get_root_client()

    msg = RegisterNewPeerMessage(
        address=om_network.address, reply_to=om_network_client.address
    )

    target_id = om_network_client.send_immediate_msg_with_reply(msg=msg).peer_id
    host_id = om_network_client.send_immediate_msg_with_reply(msg=msg).peer_id

    offer_msg = SignalingAnswerMessage(
        address=om_network.address,
        payload="SDP",
        host_metadata=alice_vm.get_metadata_for_client(),
        target_peer=target_id,
        host_peer=host_id,
    )
    om_network_client.send_immediate_msg_without_reply(msg=offer_msg)

    assert (
        om_network.signaling_msgs[target_id][SyftMessage].pop(offer_msg.id) == offer_msg
    )


def test_pull_offer_signaling_service() -> None:
    om_network, bob_vm, alice_vm = get_preset_nodes()
    om_network_client = om_network.get_root_client()

    msg = RegisterNewPeerMessage(
        address=om_network.address, reply_to=om_network_client.address
    )

    target_id = om_network_client.send_immediate_msg_with_reply(msg=msg).peer_id
    host_id = om_network_client.send_immediate_msg_with_reply(msg=msg).peer_id

    offer_msg = SignalingOfferMessage(
        address=om_network.address,
        payload="SDP",
        host_metadata=alice_vm.get_metadata_for_client(),
        target_peer=target_id,
        host_peer=host_id,
    )
    om_network_client.send_immediate_msg_without_reply(msg=offer_msg)

    assert (
        om_network.signaling_msgs[target_id][SyftMessage].get(offer_msg.id) == offer_msg
    )

    offer_pull_req = OfferPullRequestMessage(
        address=om_network.address,
        target_peer=host_id,
        host_peer=target_id,
        reply_to=om_network_client.address,
    )

    response = om_network_client.send_immediate_msg_with_reply(msg=offer_pull_req)

    assert response == offer_msg


def test_pull_answer_signaling_service() -> None:
    om_network, bob_vm, alice_vm = get_preset_nodes()
    om_network_client = om_network.get_root_client()

    msg = RegisterNewPeerMessage(
        address=om_network.address, reply_to=om_network_client.address
    )

    target_id = om_network_client.send_immediate_msg_with_reply(msg=msg).peer_id
    host_id = om_network_client.send_immediate_msg_with_reply(msg=msg).peer_id

    answer_msg = SignalingAnswerMessage(
        address=om_network.address,
        payload="SDP",
        host_metadata=alice_vm.get_metadata_for_client(),
        target_peer=target_id,
        host_peer=host_id,
    )
    om_network_client.send_immediate_msg_without_reply(msg=answer_msg)

    assert (
        om_network.signaling_msgs[target_id][SyftMessage].get(answer_msg.id)
        == answer_msg
    )

    ans_pull_req = AnswerPullRequestMessage(
        address=om_network.address,
        target_peer=host_id,
        host_peer=target_id,
        reply_to=om_network_client.address,
    )

    response = om_network_client.send_immediate_msg_with_reply(msg=ans_pull_req)

    assert response == answer_msg


def test_loopback_offer_signaling_service() -> None:
    om_network, bob_vm, alice_vm = get_preset_nodes()
    om_network_client = om_network.get_root_client()

    msg = RegisterNewPeerMessage(
        address=om_network.address, reply_to=om_network_client.address
    )

    host_id = om_network_client.send_immediate_msg_with_reply(msg=msg).peer_id

    offer_msg = SignalingOfferMessage(
        address=om_network.address,
        payload="SDP",
        host_metadata=alice_vm.get_metadata_for_client(),
        target_peer=host_id,
        host_peer=host_id,
    )

    om_network_client.send_immediate_msg_without_reply(msg=offer_msg)

    # Do not enqueue loopback requests
    assert len(om_network.signaling_msgs[host_id][SyftMessage]) == 0

    offer_pull_req = OfferPullRequestMessage(
        address=om_network.address,
        target_peer=host_id,
        host_peer=host_id,
        reply_to=om_network_client.address,
    )

    response = om_network_client.send_immediate_msg_with_reply(msg=offer_pull_req)

    # Return InvalidLoopBack Message
    assert isinstance(response, InvalidLoopBackRequest)


def test_loopback_answer_signaling_service() -> None:
    om_network, bob_vm, alice_vm = get_preset_nodes()
    om_network_client = om_network.get_root_client()

    msg = RegisterNewPeerMessage(
        address=om_network.address, reply_to=om_network_client.address
    )

    host_id = om_network_client.send_immediate_msg_with_reply(msg=msg).peer_id

    metadata = alice_vm.get_metadata_for_client()
    answer_msg = SignalingAnswerMessage(
        address=om_network.address,
        payload="SDP",
        host_metadata=metadata,
        target_peer=host_id,
        host_peer=host_id,
    )
    om_network_client.send_immediate_msg_without_reply(msg=answer_msg)

    # Do not enqueue loopback requests
    assert len(om_network.signaling_msgs[host_id][SyftMessage]) == 0

    ans_pull_req = AnswerPullRequestMessage(
        address=om_network.address,
        target_peer=host_id,
        host_peer=host_id,
        reply_to=om_network_client.address,
    )

    response = om_network_client.send_immediate_msg_with_reply(msg=ans_pull_req)

    # Return InvalidLoopBackMessage
    assert isinstance(response, InvalidLoopBackRequest)


def test_not_found_pull_offer_requests_signaling_service() -> None:
    om_network, bob_vm, alice_vm = get_preset_nodes()
    om_network_client = om_network.get_root_client()

    msg = RegisterNewPeerMessage(
        address=om_network.address, reply_to=om_network_client.address
    )

    host_id = om_network_client.send_immediate_msg_with_reply(msg=msg).peer_id
    target_id = om_network_client.send_immediate_msg_with_reply(msg=msg).peer_id

    offer_pull_req = OfferPullRequestMessage(
        address=om_network.address,
        target_peer=target_id,
        host_peer=host_id,
        reply_to=om_network_client.address,
    )

    response = om_network_client.send_immediate_msg_with_reply(msg=offer_pull_req)

    assert isinstance(response, SignalingRequestsNotFound)


def test_not_found_pull_ans_requests_signaling_service() -> None:
    om_network, bob_vm, alice_vm = get_preset_nodes()
    om_network_client = om_network.get_root_client()

    msg = RegisterNewPeerMessage(
        address=om_network.address, reply_to=om_network_client.address
    )

    host_id = om_network_client.send_immediate_msg_with_reply(msg=msg).peer_id
    target_id = om_network_client.send_immediate_msg_with_reply(msg=msg).peer_id

    ans_pull_req = AnswerPullRequestMessage(
        address=om_network.address,
        target_peer=target_id,
        host_peer=host_id,
        reply_to=om_network_client.address,
    )

    response = om_network_client.send_immediate_msg_with_reply(msg=ans_pull_req)

    assert isinstance(response, SignalingRequestsNotFound)
