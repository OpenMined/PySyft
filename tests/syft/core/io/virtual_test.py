from typing import Callable
from typing import Union

import pytest
from nacl.signing import SigningKey
from nacl.signing import VerifyKey

from syft.core.common.uid import UID
from syft.core.common.message import SignedEventualSyftMessageWithoutReply
from syft.core.common.message import SignedImmediateSyftMessageWithReply
from syft.core.common.message import SignedImmediateSyftMessageWithoutReply
from syft.core.io.address import Address
from syft.core.io.location.specific import SpecificLocation
from syft.core.io.virtual import create_virtual_connection
from syft.core.io.virtual import VirtualClientConnection
from syft.core.io.virtual import VirtualServerConnection
from syft.core.node.common.node import Node


def _gen_address() -> Address:
    """Helper method to construct an Address"""
    return Address(
        network=SpecificLocation(id=UID()),
        domain=SpecificLocation(id=UID()),
        device=SpecificLocation(id=UID()),
        vm=SpecificLocation(id=UID()),
    )


def _gen_node() -> Node:
    return Node(
        network=SpecificLocation(id=UID()),
        domain=SpecificLocation(id=UID()),
        device=SpecificLocation(id=UID()),
        vm=SpecificLocation(id=UID()),
    )


def get_signing_key() -> SigningKey:
    # return a the signing key to use
    key = "e89ff2e651b42393b6ecb5956419088781309d953d72bd73a0968525a3a6a951"
    return SigningKey(bytes.fromhex(key))


def get_verify_key() -> VerifyKey:
    return get_signing_key().verify_key


def _gen_dummy_message(
    msg_class: Callable,
) -> Union[
    SignedEventualSyftMessageWithoutReply,
    SignedImmediateSyftMessageWithReply,
    SignedImmediateSyftMessageWithoutReply,
]:
    """
    Helper method to construct a dummy SignedMessage of the following:
    - SignedEventualSyftMessageWithoutReply
    - SignedImmediateSyftMessageWithReply
    - SignedImmediateSyftMessageWithoutReply
    """
    address = _gen_address()
    key = get_verify_key()

    return msg_class(
        address=address,
        obj_type="signed_message",
        signature=b"my_signature",
        verify_key=key,
        message=b"hello_world",
    )


# --------------------- HELPER FUNCTION ---------------------


def test_create_virtual_connection() -> None:
    """
    Test that create_virtual_connection function returns VirtualClientConnection.
    """
    client = create_virtual_connection(node=_gen_node())
    assert isinstance(client, VirtualClientConnection)


# --------------------- INITIALIZATION ---------------------


def test_virtual_server_connection_init() -> None:
    """
    Test that VirtualServerConnection uses the AbstractNode
    passed into its constructor.
    """
    node = _gen_node()
    server = VirtualServerConnection(node=node)

    assert server.node == node


def test_virtual_client_connection_init() -> None:
    """
    Test that VirtualClientConnection uses the VirtualServerConnection
    passed into its constructor.
    """
    node = _gen_node()
    server = VirtualServerConnection(node=node)
    client = VirtualClientConnection(server=server)
    assert client.server == server


# --------------------- CLASS METHODS ---------------------

# def test_virtual_server_connection_recv_immedaite_msg_with_reply() -> None:
#     """Test that VirtualServerConnection.recv_immedaite_msg_with_reply works."""
#     node = _gen_node()
#     server = VirtualServerConnection(node=node)
#     msg = _gen_dummy_message(SignedImmediateSyftMessageWithReply)
#     assert isinstance(
#         server.recv_immediate_msg_with_reply(msg=msg),
#         SignedEventualSyftMessageWithoutReply
#     )
#
#
# def test_virtual_server_connection_recv_immedaite_msg_without_reply() -> None:
#     """Test that VirtualServerConnection.recv_immedaite_msg_without_reply works."""
#     node = _gen_node()
#     server = VirtualServerConnection(node=node)
#     msg = _gen_dummy_message(SignedImmediateSyftMessageWithoutReply)
#     assert server.recv_immediate_msg_without_reply(msg=msg) is None


def test_virtual_server_connection_recv_eventual_msg_without_reply() -> None:
    """
    Test that VirtualServerConnection.recv_eventual_msg_with_reply raises NotImplementedError.
    """
    node = _gen_node()
    server = VirtualServerConnection(node=node)
    msg = _gen_dummy_message(SignedEventualSyftMessageWithoutReply)

    with pytest.raises(NotImplementedError):
        assert server.recv_eventual_msg_without_reply(msg=msg) is None
