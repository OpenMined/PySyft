# stdlib
import secrets
from typing import Tuple

# syft absolute
import syft as sy
from syft.core.common.message import SyftMessage
from syft.core.io.address import Address
from syft.grid.services.vm_management_service import CreateVMService
from syft.core.node.device.service.vm_management_message import CreateVMMessage
from syft.core.node.device.service.vm_management_message import CreateVMResponseMessage


def test_create_vm_message_serde() -> None:
    bob_domain = sy.Domain(name="Bob")
    target_device = sy.Device(name="Common Device")
    
    settings = {"cpu": "xeon", "gpu": "Tesla", "memory": "32gb"}
    msg = CreateVMMessage(
            address=target_device, settings=settings, reply_to=bob_domain
    )

    blob = msg.serialize()
    msg2 = sy.deserialize(blob=blob)

    assert msg.id == msg2.id
    assert msg.address == target_device.address
    assert msg.settings == msg2.settings
    assert msg == msg2

def test_create_vm_message_serde() -> None:
    bob_domain = sy.Domain(name="Bob")
    target_device = sy.Device(name="Common Device")
    
    new_vm = sy.VirtualMachine(name="Custom Virtual Machine")

    msg = CreateVMResponseMessage(
            address=bob_domain,
            success=True,
            msg="Virtual Machine initialized successfully!",
            vm_address=new_vm.address,
        )

    blob = msg.serialize()
    msg2 = sy.deserialize(blob=blob)

    assert msg.id == msg2.id
    assert msg.address == bob_domain.address
    assert msg.vm_address == new_vm.address
    assert msg == msg2
    assert msg.msg == msg.msg
    assert msg.success == msg.success

'''
def test_signaling_answer_message_serde() -> None:
    bob_vm = sy.VirtualMachine(name="Bob")
    target = Address(name="Alice")
    target_id = secrets.token_hex(nbytes=16)
    host_id = secrets.token_hex(nbytes=16)

    msg = SignalingAnswerMessage(
        address=target,
        payload="SDP",
        host_metadata=bob_vm.get_metadata_for_client(),
        target_peer=target_id,
        host_peer=host_id,
    )
    msg_metadata = bob_vm.get_metadata_for_client()

    blob = msg.serialize()
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


def test_signaling_answer_pull_request_message_serde() -> None:
    bob_vm = sy.VirtualMachine(name="Bob")
    target = Address(name="Alice")

    target_id = secrets.token_hex(nbytes=16)
    host_id = secrets.token_hex(nbytes=16)

    msg = AnswerPullRequestMessage(
        address=target,
        target_peer=target_id,
        host_peer=host_id,
        reply_to=bob_vm.address,
    )

    blob = msg.serialize()
    msg2 = sy.deserialize(blob=blob)

    assert msg.id == msg2.id
    assert msg.address == target
    assert msg == msg2
    assert msg2.host_peer == host_id
    assert msg2.target_peer == target_id


def test_signaling_offer_pull_request_message_serde() -> None:
    bob_vm = sy.VirtualMachine(name="Bob")
    target = Address(name="Alice")

    target_id = secrets.token_hex(nbytes=16)
    host_id = secrets.token_hex(nbytes=16)

    msg = OfferPullRequestMessage(
        address=target,
        target_peer=target_id,
        host_peer=host_id,
        reply_to=bob_vm.address,
    )

    blob = msg.serialize()
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
'''
