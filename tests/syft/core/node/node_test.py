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

    assert bob_vm.device is not None

    # TODO: prevent device being set when Authorization fails
    assert bob_vm_client.device is not None


def test_register_vm_on_device_succeeds() -> None:
    bob_vm = sy.VirtualMachine(name="Bob")  # generated signing key
    bob_vm_client = bob_vm.get_client()  # generated signing key

    # set the signing_key to set the root_verify_key
    bob_phone = sy.Device(name="Bob's iPhone")
    bob_phone.root_verify_key = get_verify_key()  # inject

    # give bob_phone_client the same credentials as the destination node root_verify_key
    bob_phone_client = bob_phone.get_client()  # generated signing key
    bob_phone_client.signing_key = get_signing_key()  # inject
    bob_phone_client.verify_key = get_verify_key()  # inject

    # bob_phone should trust messages signed by bob_vm_client
    assert bob_phone_client.verify_key == bob_phone.root_verify_key
    bob_phone_client.register(client=bob_vm_client)

    # WARNING this is hacked due to HeritageUpdateMessage
    assert bob_vm.device is not None
    assert bob_vm_client.device is not None


def test_send_message_from_device_client_to_vm() -> None:
    bob_vm = sy.VirtualMachine(name="Bob")
    bob_vm_client = bob_vm.get_client()

    bob_phone = sy.Device(name="Bob's iPhone")
    bob_phone.root_verify_key = get_verify_key()  # inject

    bob_phone_client = bob_phone.get_client()
    bob_phone_client.signing_key = get_signing_key()  # inject
    bob_phone_client.verify_key = get_verify_key()  # inject

    bob_phone_client.register(client=bob_vm_client)

    # WARNING this is hacked due to HeritageUpdateMessage
    assert bob_vm.device is not None
    assert bob_vm_client.device is not None

    # TODO: Fix this
    with pytest.raises(AttributeError) as excinfo:
        bob_phone_client.send_immediate_msg_without_reply(
            msg=sy.ReprMessage(address=bob_vm_client.address)
        )

    assert (
        "'StorableObject' object has no attribute 'send_signed_msg_with_reply'"
        in str(excinfo.value)
    )


def test_send_message_from_domain_client_to_vm() -> None:

    bob_vm = sy.VirtualMachine(name="Bob")
    bob_vm_client = bob_vm.get_client()

    bob_phone = sy.Device(name="Bob's iPhone")
    bob_phone.root_verify_key = get_verify_key()  # inject
    bob_phone_client = bob_phone.get_client()
    bob_phone_client.signing_key = get_signing_key()  # inject
    bob_phone_client.verify_key = get_verify_key()  # inject

    bob_domain = sy.Domain(name="Bob's Domain")
    bob_domain.root_verify_key = get_verify_key()  # inject
    bob_domain_client = bob_domain.get_client()
    bob_domain_client.signing_key = get_signing_key()  # inject
    bob_domain_client.verify_key = get_verify_key()  # inject

    bob_phone_client.register(client=bob_vm_client)
    bob_domain_client.register(client=bob_phone_client)

    # same issues as above, so disabling until fixed
    # bob_domain_client.send_immediate_msg_without_reply(
    #     msg=sy.ReprMessage(address=bob_vm)
    # )


# def test_send_message_from_network_client_to_vm() -> None:

#     bob_vm = sy.VirtualMachine(name="Bob")
#     bob_vm_client = bob_vm.get_client()

#     bob_phone = sy.Device(name="Bob's iPhone")
#     bob_phone_client = bob_phone.get_client()

#     bob_domain = sy.Domain(name="Bob's Domain")
#     bob_domain_client = bob_domain.get_client()

#     bob_network = sy.Network(name="Bob's Network")
#     bob_network_client = bob_network.get_client()

#     bob_phone_client.register(client=bob_vm_client)
#     bob_domain_client.register(client=bob_phone_client)
#     bob_network_client.register(client=bob_domain_client)

#     # same issues as above, so disabling until fixed
#     bob_network_client.send_immediate_msg_without_reply(
#         msg=sy.ReprMessage(address=bob_vm)
#     )
