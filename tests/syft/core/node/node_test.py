# third party
from nacl.signing import SigningKey
from nacl.signing import VerifyKey
import pytest

# syft absolute
import syft as sy
from syft.core.node.common.service.auth import AuthorizationException


def get_signing_key() -> SigningKey:
    # return a the signing key to use
    key = "e89ff2e651b42393b6ecb5956419088781309d953d72bd73a0968525a3a6a951"
    return SigningKey(bytes.fromhex(key))


def get_verify_key() -> VerifyKey:
    return get_signing_key().verify_key


def test_to_string() -> None:
    bob_vm = sy.VirtualMachine(name="Bob")

    assert str(bob_vm) == f"VirtualMachine: Bob: {bob_vm.id}"
    assert bob_vm.__repr__() == f"VirtualMachine: Bob: {bob_vm.id}"


def test_send_message_from_vm_client_to_vm() -> None:

    bob_vm = sy.VirtualMachine(name="Bob")
    bob_vm_client = bob_vm.get_client()

    assert bob_vm.device is None

    with pytest.raises(AuthorizationException):
        bob_vm_client.send_immediate_msg_without_reply(
            msg=sy.ReprMessage(address=bob_vm_client.address)
        )


def test_send_message_from_device_client_to_device() -> None:
    bob_phone = sy.Device(name="Bob's iPhone")
    bob_phone_client = bob_phone.get_client()

    with pytest.raises(AuthorizationException):
        bob_phone_client.send_immediate_msg_without_reply(
            msg=sy.ReprMessage(address=bob_phone_client.address)
        )


def test_register_vm_on_device_fails() -> None:

    bob_vm = sy.VirtualMachine(name="Bob")
    bob_vm_client = bob_vm.get_client()

    bob_phone = sy.Device(name="Bob's iPhone")
    bob_phone_client = bob_phone.get_client()

    with pytest.raises(AuthorizationException):
        bob_phone_client.register(client=bob_vm_client)

    assert bob_vm.device is None

    # TODO: prevent device being set when Authorization fails
    assert bob_vm_client.device is not None


def test_register_vm_on_device_succeeds() -> None:
    # Register a 🍰 with a 📱

    bob_vm = sy.VirtualMachine(name="Bob")
    bob_vm_client = bob_vm.get_client()
    bob_vm.root_verify_key = bob_vm_client.verify_key  # inject 📡🔑 as 📍🗝

    bob_phone = sy.Device(name="Bob's iPhone")
    bob_phone_client = bob_phone.get_client()
    bob_phone.root_verify_key = bob_phone_client.verify_key  # inject 📡🔑 as 📍🗝

    bob_phone_client.register(client=bob_vm_client)

    assert bob_vm.device is not None
    assert bob_vm_client.device is not None


def test_known_child_nodes() -> None:
    bob_vm = sy.VirtualMachine(name="Bob VM")
    bob_vm_client = bob_vm.get_client()
    bob_vm.root_verify_key = bob_vm_client.verify_key  # inject 📡🔑 as 📍🗝

    bob_vm_2 = sy.VirtualMachine(name="Bob VM 2")
    bob_vm_client_2 = bob_vm_2.get_client()
    bob_vm_2.root_verify_key = bob_vm_client_2.verify_key  # inject 📡🔑 as 📍🗝

    bob_phone = sy.Device(name="Bob's iPhone")
    bob_phone_client = bob_phone.get_client()
    bob_phone.root_verify_key = bob_phone_client.verify_key  # inject 📡🔑 as 📍🗝

    bob_phone_client.register(client=bob_vm_client)

    assert len(bob_phone.known_child_nodes) == 1
    assert bob_vm in bob_phone.known_child_nodes

    bob_phone_client.register(client=bob_vm_client_2)

    assert len(bob_phone.known_child_nodes) == 2
    assert bob_vm_2 in bob_phone.known_child_nodes


def test_send_message_from_device_client_to_vm() -> None:
    # Register a 🍰 with a 📱
    # Send ✉️ from 📱 ➡️ 🍰

    bob_vm = sy.VirtualMachine(name="Bob")
    bob_vm_client = bob_vm.get_client()
    bob_vm.root_verify_key = bob_vm_client.verify_key  # inject 📡🔑 as 📍🗝

    bob_phone = sy.Device(name="Bob's iPhone")
    bob_phone_client = bob_phone.get_client()
    bob_phone.root_verify_key = bob_phone_client.verify_key  # inject 📡🔑 as 📍🗝

    bob_phone_client.register(client=bob_vm_client)

    assert bob_vm.device is not None
    assert bob_vm_client.device is not None

    # switch keys
    bob_vm.root_verify_key = bob_phone_client.verify_key  # inject 📡🔑 as 📍🗝

    bob_phone_client.send_immediate_msg_without_reply(
        msg=sy.ReprMessage(address=bob_vm_client.address)
    )


@pytest.mark.asyncio
def test_send_message_from_domain_client_to_vm() -> None:
    # Register a 🍰 with a 📱
    # Register a 📱 with a 🏰
    # Send ✉️ from 🏰 ➡️ 🍰

    bob_vm = sy.VirtualMachine(name="Bob")
    bob_vm_client = bob_vm.get_client()
    bob_vm.root_verify_key = bob_vm_client.verify_key  # inject 📡🔑 as 📍🗝

    bob_phone = sy.Device(name="Bob's iPhone")
    bob_phone_client = bob_phone.get_client()
    bob_phone.root_verify_key = bob_phone_client.verify_key  # inject 📡🔑 as 📍🗝

    bob_phone_client.register(client=bob_vm_client)

    assert bob_vm.device is not None
    assert bob_vm_client.device is not None

    bob_domain = sy.Domain(name="Bob's Domain")
    bob_domain_client = bob_domain.get_client()
    bob_domain.root_verify_key = bob_domain_client.verify_key  # inject 📡🔑 as 📍🗝

    # switch keys
    bob_vm.root_verify_key = bob_domain_client.verify_key  # inject 📡🔑 as 📍🗝
    bob_domain_client.register(client=bob_phone_client)

    assert bob_phone.domain is not None
    assert bob_phone_client.domain is not None

    bob_domain_client.send_immediate_msg_without_reply(
        msg=sy.ReprMessage(address=bob_vm.address)
    )


@pytest.mark.asyncio
def test_send_message_from_network_client_to_vm() -> None:
    # Register a 🍰 with a 📱
    # Register a 📱 with a 🏰
    # Register a 🏰 with a 🔗
    # Send ✉️ from 🔗 ➡️ 🍰

    bob_vm = sy.VirtualMachine(name="Bob")
    bob_vm_client = bob_vm.get_client()
    bob_vm.root_verify_key = bob_vm_client.verify_key  # inject 📡🔑 as 📍🗝

    bob_phone = sy.Device(name="Bob's iPhone")
    bob_phone_client = bob_phone.get_client()
    bob_phone.root_verify_key = bob_phone_client.verify_key  # inject 📡🔑 as 📍🗝

    bob_phone_client.register(client=bob_vm_client)

    assert bob_vm.device is not None
    assert bob_vm_client.device is not None

    bob_domain = sy.Domain(name="Bob's Domain")
    bob_domain_client = bob_domain.get_client()
    bob_domain.root_verify_key = bob_domain_client.verify_key  # inject 📡🔑 as 📍🗝

    bob_domain_client.register(client=bob_phone_client)

    assert bob_phone.domain is not None
    assert bob_phone_client.domain is not None

    bob_network = sy.Network(name="Bob's Network")
    bob_network_client = bob_network.get_client()
    bob_network.root_verify_key = bob_network_client.verify_key  # inject 📡🔑 as 📍🗝

    # switch keys
    bob_vm.root_verify_key = bob_network_client.verify_key  # inject 📡🔑 as 📍🗝
    bob_network_client.register(client=bob_domain_client)

    assert bob_domain.network is not None
    assert bob_domain_client.network is not None

    bob_network_client.send_immediate_msg_without_reply(
        msg=sy.ReprMessage(address=bob_vm.address)
    )


@pytest.mark.asyncio
def test_autoapprove_requests_made_by_root_clients_5015() -> None:
    import torch

    alice = sy.VirtualMachine(name="alice")
    alice_client = alice.get_root_client()
    p = alice_client.torch.Tensor([1, 2, 3])
    t = p.get(request_block=True, name="Test")
    assert torch.equal(t, torch.Tensor([1, 2, 3]))

    alice_guest = alice.get_client()
    p = alice_guest.torch.Tensor([1, 2, 3])
    t = p.get(request_block=True, name="Test")
    assert t is None

    p = torch.Tensor([4, 5, 6])
    p.send(alice_client, searchable=True)
    p = alice_client.store[0]
    t = p.get(request_block=True, name="Test", delete_obj=False)
    assert torch.equal(t, torch.Tensor([4, 5, 6]))

    assert len(alice_guest.store) == 1
    p = alice_guest.store[0]
    t = p.get(request_block=True, name="Test", delete_obj=False)
    assert t is None
