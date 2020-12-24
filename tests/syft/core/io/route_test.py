from syft.core.node.common.node import Node
from syft.core.io.route import BroadcastRoute
from syft.core.io.route import Route
from syft.core.io.route import RouteSchema
from syft.core.io.route import SoloRoute
from syft.core.io.location.specific import SpecificLocation
from syft.grid.connections.http_connection import HTTPConnection
from syft.grid.connections.webrtc import WebRTCConnection


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

    assert route.schema is rschema  # Cannot use equality
    assert isinstance(route.stops, list)


def test_solo_route_init() -> None:
    """
    Test that SoloRoute will use the Location and
    ClientConnection/BidirectionalConnection passed into its constructor.
    """
    # Test SoloRoute with ClientConnection in constructor
    destination = SpecificLocation()
    connection = HTTPConnection(url="https://opengrid.openmined.org/")
    h_solo = SoloRoute(destination=destination, connection=connection)

    assert h_solo.schema.destination is destination
    assert h_solo.connection is connection

    # Test SoloRoute with WebRTCConnection in constructor
    connection = WebRTCConnection(node=Node())
    b_solo = SoloRoute(destination=destination, connection=connection)

    assert b_solo.schema.destination is destination
    assert b_solo.connection is connection


def test_broadcast_route_init() -> None:
    """
    Test that BroadcastRoute will use the Location and
    ClientConnection passed into its constructor.
    """
    destination = SpecificLocation()
    connection = HTTPConnection(url="https://opengrid.openmined.org/")
    b_route = BroadcastRoute(destination=destination, connection=connection)

    assert b_route.schema.destination is destination
    assert b_route.connection is connection
