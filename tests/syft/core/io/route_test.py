# std
from typing import Callable
from typing import Union

# third party
from nacl.bindings.crypto_sign import crypto_sign_keypair
from nacl.signing import VerifyKey

# syft
from syft.core.common.uid import UID
from syft.core.common.message import SignedEventualSyftMessageWithoutReply
from syft.core.common.message import SignedImmediateSyftMessageWithReply
from syft.core.common.message import SignedImmediateSyftMessageWithoutReply
from syft.core.io.address import Address
from syft.core.io.route import BroadcastRoute
from syft.core.io.route import Route
from syft.core.io.route import RouteSchema
from syft.core.io.route import SoloRoute
from syft.core.io.location.specific import SpecificLocation
from syft.core.node.common.node import Node
from syft.grid.connections.http_connection import HTTPConnection
from syft.grid.connections.webrtc import WebRTCConnection


# --------------------- TEST HELPERS ---------------------


def _construct_address() -> Address:
    """Helper method to construct an Address"""
    return Address(
        network=SpecificLocation(id=UID()),
        domain=SpecificLocation(id=UID()),
        device=SpecificLocation(id=UID()),
        vm=SpecificLocation(id=UID()),
    )


def _construct_dummy_message(
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
    address = _construct_address()
    key = VerifyKey(crypto_sign_keypair()[0])

    return msg_class(
        address=address,
        obj_type="signed_message",
        signature=b"my_signature",
        verify_key=key,
        message=b"hello_world",
    )


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
    destination = SpecificLocation()
    connection = HTTPConnection(url="https://opengrid.openmined.org/")
    h_solo = SoloRoute(destination=destination, connection=connection)
    msg = _construct_dummy_message(SignedImmediateSyftMessageWithoutReply)

    assert h_solo.send_immediate_msg_without_reply(msg) is None


def test_solo_route_send_eventual_msg_without_reply() -> None:
    """Test SoloRoute.send_eventual_msg_without_reply method works."""
    destination = SpecificLocation()
    connection = HTTPConnection(url="https://opengrid.openmined.org/")
    h_solo = SoloRoute(destination=destination, connection=connection)
    msg = _construct_dummy_message(SignedEventualSyftMessageWithoutReply)

    assert not h_solo.send_eventual_msg_without_reply(msg)


# def test_solo_route_send_immediate_msg_with_reply() -> None:
#     """Test SoloRoute.send_immediate_msg_with_reply method works."""
#     destination = SpecificLocation()
#     connection = HTTPConnection(url="https://opengrid.openmined.org/")
#     h_solo = SoloRoute(destination=destination, connection=connection)
#     msg = _construct_dummy_message(SignedImmediateSyftMessageWithReply)
#
#     assert isinstance(
#         h_solo.send_immediate_msg_with_reply(msg),
#         SignedImmediateSyftMessageWithoutReply,
#     )
