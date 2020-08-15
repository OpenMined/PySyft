import pytest
from nacl.signing import SigningKey, VerifyKey

import syft as sy
from syft.core.node.common.service.auth import AuthorizationException


def get_signing_key() -> SigningKey:
    # return a the signing key to use
    key = "e89ff2e651b42393b6ecb5956419088781309d953d72bd73a0968525a3a6a951"
    return SigningKey(bytes.fromhex(key))


def get_verify_key() -> VerifyKey:
    return get_signing_key().verify_key


def test_send_message_from_vm_client_to_vm() -> None:

    bob_vm = sy.VirtualMachine(name="Bob")
    bob_vm_client = bob_vm.get_client()

    assert bob_vm.device is None

    with pytest.raises(AuthorizationException):
        bob_vm_client.send_immediate_msg_without_reply(
            msg=sy.ReprMessage(address=bob_vm_client)
        )


def test_send_message_from_device_client_to_device() -> None:
    bob_phone = sy.Device(name="Bob's iPhone")
    bob_phone_client = bob_phone.get_client()

    with pytest.raises(AuthorizationException):
        bob_phone_client.send_immediate_msg_without_reply(
            msg=sy.ReprMessage(address=bob_phone_client)
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
    if sy.VERBOSE:
        print(f"> {bob_vm.pprint} {bob_vm.keys}")
        print(f"> {bob_vm_client.pprint} {bob_vm_client.keys}")

    bob_phone = sy.Device(name="Bob's iPhone")
    bob_phone_client = bob_phone.get_client()
    bob_phone.root_verify_key = bob_phone_client.verify_key  # inject 📡🔑 as 📍🗝
    if sy.VERBOSE:
        print(f"> {bob_phone.pprint} {bob_phone.keys}")
        print(f"> {bob_phone_client.pprint} {bob_phone_client.keys}")

    bob_phone_client.register(client=bob_vm_client)

    assert bob_vm.device is not None
    assert bob_vm_client.device is not None


def test_send_message_from_device_client_to_vm() -> None:
    # Register a 🍰 with a 📱
    # Send ✉️ from 📱 ➡️ 🍰

    bob_vm = sy.VirtualMachine(name="Bob")
    bob_vm_client = bob_vm.get_client()
    bob_vm.root_verify_key = bob_vm_client.verify_key  # inject 📡🔑 as 📍🗝
    print(f"> {bob_vm.pprint} {bob_vm.keys}")
    print(f"> {bob_vm_client.pprint} {bob_vm_client.keys}")

    bob_phone = sy.Device(name="Bob's iPhone")
    bob_phone_client = bob_phone.get_client()
    bob_phone.root_verify_key = bob_phone_client.verify_key  # inject 📡🔑 as 📍🗝
    print(f"> {bob_phone.pprint} {bob_phone.keys}")
    print(f"> {bob_phone_client.pprint} {bob_phone_client.keys}")

    bob_phone_client.register(client=bob_vm_client)

    assert bob_vm.device is not None
    assert bob_vm_client.device is not None

    # switch keys
    bob_vm.root_verify_key = bob_phone_client.verify_key  # inject 📡🔑 as 📍🗝

    bob_phone_client.send_immediate_msg_without_reply(
        msg=sy.ReprMessage(address=bob_vm_client.address)
    )


def test_send_message_from_domain_client_to_vm() -> None:
    # Register a 🍰 with a 📱
    # Register a 📱 with a 🏰
    # Send ✉️ from 🏰 ➡️ 🍰

    bob_vm = sy.VirtualMachine(name="Bob")
    bob_vm_client = bob_vm.get_client()
    bob_vm.root_verify_key = bob_vm_client.verify_key  # inject 📡🔑 as 📍🗝
    print(f"> {bob_vm.pprint} {bob_vm.keys}")
    print(f"> {bob_vm_client.pprint} {bob_vm_client.keys}")

    bob_phone = sy.Device(name="Bob's iPhone")
    bob_phone_client = bob_phone.get_client()
    bob_phone.root_verify_key = bob_phone_client.verify_key  # inject 📡🔑 as 📍🗝
    print(f"> {bob_phone.pprint} {bob_phone.keys}")
    print(f"> {bob_phone_client.pprint} {bob_phone_client.keys}")

    bob_phone_client.register(client=bob_vm_client)

    assert bob_vm.device is not None
    assert bob_vm_client.device is not None

    bob_domain = sy.Domain(name="Bob's Domain")
    bob_domain_client = bob_domain.get_client()
    bob_domain.root_verify_key = bob_domain_client.verify_key  # inject 📡🔑 as 📍🗝

    # switch keys
    bob_vm.root_verify_key = bob_domain_client.verify_key  # inject 📡🔑 as 📍🗝
    bob_domain_client.register(client=bob_phone_client)

    bob_domain_client.send_immediate_msg_without_reply(
        msg=sy.ReprMessage(address=bob_vm)
    )


def test_send_message_from_network_client_to_vm() -> None:
    # Register a 🍰 with a 📱
    # Register a 📱 with a 🏰
    # Register a 🏰 with a 🔗
    # Send ✉️ from 🔗 ➡️ 🍰

    bob_vm = sy.VirtualMachine(name="Bob")
    bob_vm_client = bob_vm.get_client()
    bob_vm.root_verify_key = bob_vm_client.verify_key  # inject 📡🔑 as 📍🗝
    print(f"> {bob_vm.pprint} {bob_vm.keys}")
    print(f"> {bob_vm_client.pprint} {bob_vm_client.keys}")

    bob_phone = sy.Device(name="Bob's iPhone")
    bob_phone_client = bob_phone.get_client()
    bob_phone.root_verify_key = bob_phone_client.verify_key  # inject 📡🔑 as 📍🗝
    print(f"> {bob_phone.pprint} {bob_phone.keys}")
    print(f"> {bob_phone_client.pprint} {bob_phone_client.keys}")

    bob_phone_client.register(client=bob_vm_client)

    assert bob_vm.device is not None
    assert bob_vm_client.device is not None

    bob_domain = sy.Domain(name="Bob's Domain")
    bob_domain_client = bob_domain.get_client()
    bob_domain.root_verify_key = bob_domain_client.verify_key  # inject 📡🔑 as 📍🗝

    # # switch keys
    # # bob_vm.root_verify_key = bob_domain_client.verify_key  # inject 📡🔑 as 📍🗝
    bob_domain_client.register(client=bob_phone_client)

    assert bob_phone.domain is not None
    assert bob_phone_client.domain is not None

    bob_network = sy.Network(name="Bob's Network")
    bob_network_client = bob_network.get_client()
    bob_network.root_verify_key = bob_network_client.verify_key  # inject 📡🔑 as 📍🗝

    bob_network_client.register(client=bob_domain_client)

    assert bob_domain.network is not None
    assert bob_domain_client.network is not None

    # # switch keys
    bob_vm.root_verify_key = bob_network_client.verify_key  # inject 📡🔑 as 📍🗝

    bob_network_client.send_immediate_msg_without_reply(
        msg=sy.ReprMessage(address=bob_vm)
    )
