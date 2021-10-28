# third party
from nacl.signing import SigningKey
from nacl.signing import VerifyKey
import pytest

# syft absolute
import syft as sy
from syft.core.node.common.node_service.auth import AuthorizationException


def get_signing_key() -> SigningKey:
    # return a the signing key to use
    key = "e89ff2e651b42393b6ecb5956419088781309d953d72bd73a0968525a3a6a951"
    return SigningKey(bytes.fromhex(key))


def get_verify_key() -> VerifyKey:
    return get_signing_key().verify_key


def test_to_string(node: sy.VirtualMachine) -> None:
    assert str(node) == f"VirtualMachine: Bob: {node.id}"
    assert node.__repr__() == f"VirtualMachine: Bob: {node.id}"


def test_send_message_from_vm_client_to_vm(
    node: sy.VirtualMachine, client: sy.VirtualMachineClient
) -> None:
    assert node.device is None

    with pytest.raises(AuthorizationException):
        client.send_immediate_msg_without_reply(
            msg=sy.ReprMessage(address=client.address)
        )


def test_send_message_from_device_client_to_device() -> None:
    bob_phone = sy.Device(name="Bob's iPhone")
    bob_phone_client = bob_phone.get_client()

    with pytest.raises(AuthorizationException):
        bob_phone_client.send_immediate_msg_without_reply(
            msg=sy.ReprMessage(address=bob_phone_client.address)
        )
