# syft absolute
from syft.core.common.message import SignedEventualSyftMessageWithoutReply
from syft.core.common.message import SignedImmediateSyftMessageWithReply
from syft.core.common.message import SignedImmediateSyftMessageWithoutReply
from syft.core.io.location.specific import SpecificLocation
from syft.core.io.route import BroadcastRoute
from syft.core.io.route import Route
from syft.core.io.route import RouteSchema
from syft.core.io.route import SoloRoute
from syft.core.io.virtual import VirtualClientConnection
from syft.core.io.virtual import VirtualServerConnection
from syft.core.node.common.node import Node

# syft relative
from .utils_test import MockNode
from .utils_test import construct_dummy_message

# --------------------- INITIALIZATION ---------------------


def test_route_schema_init() -> None:
    """Test that RouteSchema will use the Location passed into its constructor"""
    destination = SpecificLocation()
    rschema = RouteSchema(destination)

    assert rschema.destination is not None
    assert rschema.destination._id == destination._id


def test_route_init() -> None:
    """Test that Route will use the RouteSchema passed into its constructor"""
    rschema = RouteSchema(SpecificLocation())
    route = Route(schema=rschema)

    assert route.schema is rschema  # Cannot use __eq__
    assert isinstance(route.stops, list)


def test_solo_route_init() -> None:
    """
    Test that SoloRoute will use the Location and
    ClientConnection/BidirectionalConnection passed into its constructor.
    """
    # Test SoloRoute with VirtualClientConnection (ClientConnection) in constructor
    destination = SpecificLocation()
    virtual_server = VirtualServerConnection(node=Node())
    virtual_client = VirtualClientConnection(server=virtual_server)
    h_solo = SoloRoute(destination=destination, connection=virtual_client)

    assert h_solo.schema.destination is destination
    assert h_solo.connection is virtual_client


def test_broadcast_route_init() -> None:
    """
    Test that BroadcastRoute will use the Location and
    ClientConnection passed into its constructor.
    """
    destination = SpecificLocation()
    virtual_server = VirtualServerConnection(node=Node())
    virtual_client = VirtualClientConnection(server=virtual_server)
    b_route = BroadcastRoute(destination=destination, connection=virtual_client)

    assert b_route.schema.destination is destination
    assert b_route.connection is virtual_client


# --------------------- CLASS & PROPERTY METHODS ---------------------


def test_route_icon_property_method() -> None:
    """Test Route.icon property method returns the correct icon."""
    route = Route(schema=RouteSchema(SpecificLocation()))
    assert route.icon == "🛣️ "


def test_route_pprint_property_method() -> None:
    """Test Route.pprint property method returns the correct icon."""
    route = Route(schema=RouteSchema(SpecificLocation()))
    assert route.pprint == "🛣️  (Route)"


def test_solo_route_send_immediate_msg_without_reply() -> None:
    """Test SoloRoute.send_immediate_msg_without_reply method works."""
    node = MockNode()
    destination = SpecificLocation()
    server = VirtualServerConnection(node=node)
    connection = VirtualClientConnection(server=server)

    h_solo = SoloRoute(destination=destination, connection=connection)
    msg = construct_dummy_message(SignedImmediateSyftMessageWithoutReply)

    assert h_solo.send_immediate_msg_without_reply(msg) is None


def test_solo_route_send_eventual_msg_without_reply() -> None:
    """Test SoloRoute.send_eventual_msg_without_reply method works."""
    node = MockNode()
    destination = SpecificLocation()
    server = VirtualServerConnection(node=node)
    connection = VirtualClientConnection(server=server)

    h_solo = SoloRoute(destination=destination, connection=connection)
    msg = construct_dummy_message(SignedEventualSyftMessageWithoutReply)

    assert not h_solo.send_eventual_msg_without_reply(msg)


def test_solo_route_send_immediate_msg_with_reply() -> None:
    """Test SoloRoute.send_immediate_msg_with_reply method works."""
    node = MockNode()
    destination = SpecificLocation()
    server = VirtualServerConnection(node=node)
    connection = VirtualClientConnection(server=server)

    h_solo = SoloRoute(destination=destination, connection=connection)
    msg = construct_dummy_message(SignedImmediateSyftMessageWithReply)
    ret = h_solo.send_immediate_msg_with_reply(msg)

    assert isinstance(
        ret,
        SignedImmediateSyftMessageWithoutReply,
    )
