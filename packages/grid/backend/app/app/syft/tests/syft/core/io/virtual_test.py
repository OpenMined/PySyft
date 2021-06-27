# syft absolute
from syft.core.common.message import SignedEventualSyftMessageWithoutReply
from syft.core.common.message import SignedImmediateSyftMessageWithReply
from syft.core.common.message import SignedImmediateSyftMessageWithoutReply
from syft.core.io.virtual import VirtualClientConnection
from syft.core.io.virtual import VirtualServerConnection
from syft.core.io.virtual import create_virtual_connection

# syft relative
from .utils_test import MockNode
from .utils_test import construct_dummy_message

# --------------------- HELPER FUNCTION ---------------------


def test_create_virtual_connection() -> None:
    """
    Test that create_virtual_connection function returns VirtualClientConnection.
    """
    client = create_virtual_connection(node=MockNode())
    assert isinstance(client, VirtualClientConnection)


# --------------------- INITIALIZATION ---------------------


def test_virtual_server_connection_init() -> None:
    """
    Test that VirtualServerConnection uses the AbstractNode
    passed into its constructor.
    """
    node = MockNode()
    server = VirtualServerConnection(node=node)

    assert server.node == node


def test_virtual_client_connection_init() -> None:
    """
    Test that VirtualClientConnection uses the VirtualServerConnection
    passed into its constructor.
    """
    node = MockNode()
    server = VirtualServerConnection(node=node)
    client = VirtualClientConnection(server=server)
    assert client.server == server


# --------------------- CLASS METHODS ---------------------


def test_virtual_server_connection_recv_immedaite_msg_with_reply() -> None:
    """Test that VirtualServerConnection.recv_immedaite_msg_with_reply works."""
    node = MockNode()
    server = VirtualServerConnection(node=node)
    msg = construct_dummy_message(SignedImmediateSyftMessageWithReply)

    assert isinstance(
        server.recv_immediate_msg_with_reply(msg=msg),
        SignedImmediateSyftMessageWithoutReply,
    )


def test_virtual_server_connection_recv_immedaite_msg_without_reply() -> None:
    """Test that VirtualServerConnection.recv_immedaite_msg_without_reply works."""
    node = MockNode()
    server = VirtualServerConnection(node=node)
    msg = construct_dummy_message(SignedImmediateSyftMessageWithoutReply)

    assert not server.recv_immediate_msg_without_reply(msg=msg)


def test_virtual_server_connection_recv_eventual_msg_without_reply() -> None:
    """
    Test that VirtualServerConnection.recv_eventual_msg_with_reply raises NotImplementedError.
    """
    node = MockNode()
    server = VirtualServerConnection(node=node)
    msg = construct_dummy_message(SignedEventualSyftMessageWithoutReply)

    assert server.recv_eventual_msg_without_reply(msg=msg) is None


def test_virtual_client_connection_send_immediate_msg_without_reply() -> None:
    """
    Test that VirtualClientConnection.send_immediate_msg_without_reply works.
    """
    node = MockNode()
    server = VirtualServerConnection(node=node)
    client = VirtualClientConnection(server=server)
    msg = construct_dummy_message(SignedImmediateSyftMessageWithoutReply)

    assert client.send_immediate_msg_without_reply(msg=msg) is None


def test_virtual_client_connection_send_immediate_msg_with_reply() -> None:
    """
    Test that VirtualClientConnection.send_immediate_msg_with_reply works.
    """
    node = MockNode()
    server = VirtualServerConnection(node=node)
    client = VirtualClientConnection(server=server)
    msg = construct_dummy_message(SignedImmediateSyftMessageWithReply)

    assert isinstance(
        client.send_immediate_msg_with_reply(msg=msg),
        SignedImmediateSyftMessageWithoutReply,
    )


def test_virtual_client_connection_send_eventual_msg_without_reply() -> None:
    """
    Test that VirtualClientConnection.send_eventual_msg_with_reply raises NotImplementedError.
    """
    node = MockNode()
    server = VirtualServerConnection(node=node)
    client = VirtualClientConnection(server=server)
    msg = construct_dummy_message(SignedEventualSyftMessageWithoutReply)

    assert client.send_eventual_msg_without_reply(msg=msg) is None
