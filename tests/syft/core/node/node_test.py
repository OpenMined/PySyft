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

    bob_vm_client.send_immediate_msg_without_reply(
        msg=sy.ReprMessage(address=bob_vm_client)
    )


def test_send_message_from_device_client_to_device() -> None:
    bob_phone = sy.Device(name="Bob's iPhone")
    bob_phone_client = bob_phone.get_client()

    bob_phone_client.send_immediate_msg_without_reply(
        msg=sy.ReprMessage(address=bob_phone_client)
    )


def test_register_vm_on_device_fails() -> None:

    bob_vm = sy.VirtualMachine(name="Bob")
    bob_vm_client = bob_vm.get_client()

    signing_key = SigningKey.generate()
    bob_phone = sy.Device(name="Bob's iPhone")
    bob_phone_client = bob_phone.get_client()
    bob_phone_client.signing_key = signing_key
    bob_phone_client.verify_key = signing_key.verify_key

    with pytest.raises(AuthorizationException):
        bob_phone_client.register(client=bob_vm_client)

    assert bob_vm.device is None

    # TODO: prevent device being set when Authorization fails
    assert bob_vm_client.device is not None


def test_register_vm_on_device_succeeds() -> None:
    bob_vm = sy.VirtualMachine(
        name="Bob", signing_key=get_signing_key(), verify_key=get_verify_key()
    )

    bob_vm_client = bob_vm.get_client()

    bob_phone = sy.Device(name="Bob's iPhone", signing_key=get_signing_key())

    bob_phone_client = bob_phone.get_client()
    bob_phone_client.register(client=bob_vm_client)

    assert bob_vm.device is not None
    assert bob_vm_client.device is not None


def test_send_message_from_device_client_to_vm() -> None:
    bob_vm = sy.VirtualMachine(name="Bob")
    bob_vm_client = bob_vm.get_client()

    bob_phone = sy.Device(name="Bob's iPhone")
    bob_phone_client = bob_phone.get_client()

    bob_phone_client.register(client=bob_vm_client)

    assert bob_vm.device is not None

    assert bob_vm_client.device is not None

    bob_phone_client.send_immediate_msg_without_reply(
        msg=sy.ReprMessage(address=bob_vm_client)
    )


def test_send_message_from_domain_client_to_vm() -> None:

    bob_vm = sy.VirtualMachine(name="Bob")
    bob_vm_client = bob_vm.get_client()

    bob_phone = sy.Device(name="Bob's iPhone")
    bob_phone_client = bob_phone.get_client()

    bob_domain = sy.Domain(name="Bob's Domain")
    bob_domain_client = bob_domain.get_client()

    bob_phone_client.register(client=bob_vm_client)
    bob_domain_client.register(client=bob_phone_client)

    bob_domain_client.send_immediate_msg_without_reply(
        msg=sy.ReprMessage(address=bob_vm)
    )


def test_send_message_from_network_client_to_vm() -> None:

    bob_vm = sy.VirtualMachine(name="Bob")
    bob_vm_client = bob_vm.get_client()

    bob_phone = sy.Device(name="Bob's iPhone")
    bob_phone_client = bob_phone.get_client()

    bob_domain = sy.Domain(name="Bob's Domain")
    bob_domain_client = bob_domain.get_client()

    bob_network = sy.Network(name="Bob's Network")
    bob_network_client = bob_network.get_client()

    bob_phone_client.register(client=bob_vm_client)
    bob_domain_client.register(client=bob_phone_client)
    bob_network_client.register(client=bob_domain_client)

    bob_network_client.send_immediate_msg_without_reply(
        msg=sy.ReprMessage(address=bob_vm)
    )
