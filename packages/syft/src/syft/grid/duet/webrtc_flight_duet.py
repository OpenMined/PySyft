"""
PySyft Duet (WebRTC)

This class aims to implement the PySyft Duet concept by using WebRTC protocol as a
connection channel in order to allow two different users to establish a direct
connection with high-quality Real-time Communication using private addresses.

The most common example showing how it can be used is the notebook demo example:

Two different jupyter / collab notebooks in different machines using private addresses
behind routers, proxies and firewalls can connect using a full-duplex channel
to perform machine learning and data science tasks, working as a client
and server at the same time.

PS 1: You need a signaling server running somewhere.
If you don't know any public address running this service, or want to set up your own
signaling network you can use PyGrid's network app.

For local development you can run:
$ python src/syft/grid/example_nodes/network.py

PS 2: The PyGrid repo has a complimentary branch that matches the current PySyft release.
To use this feature you must use the correct PyGrid branch.
(https://github.com/OpenMined/PyGrid/)

You can get more details about all this process, in the syft/grid/connections/webrtc.py
source code.
"""

# stdlib
import asyncio
import threading
from typing import Optional
import inspect
from types import ModuleType
from typing import Any
from typing import Callable as CallableT
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union
import warnings

# third party
from nacl.signing import SigningKey

# syft relative
from ... import serialize
from ...core.io.route import SoloRoute
from ...core.node.common.metadata import Metadata
from ...core.node.domain.client import DomainClient
from ...core.node.domain.domain import Domain
from ...logger import error
from ...logger import traceback_and_raise
from ..connections.webrtc import WebRTCConnection
from ..duet.signaling_client import SignalingClient
from ..duet.webrtc_duet import Duet as WebRTCDuet
from ..services.signaling_service import AnswerPullRequestMessage
from ..services.signaling_service import InvalidLoopBackRequest
from ..services.signaling_service import OfferPullRequestMessage
from ..services.signaling_service import SignalingAnswerMessage
from ..services.signaling_service import SignalingOfferMessage
from ..flight.flight_client import FlightClientDuet
from ..flight.flight_server import FlightServerDuet
from ...core.node.common.action.save_object_action import SaveObjectAction
from ...core.common.message import SignedImmediateSyftMessageWithReply
from ...core.common.message import SignedImmediateSyftMessageWithoutReply
from ...core.common.message import ImmediateSyftMessageWithoutReply
from ...core.common.message import SignedImmediateSyftMessageWithoutReply
from ...core.common.message import ImmediateSyftMessageWithReply
from ...core.common.message import SyftMessage
from ...core.store.storeable_object import StorableObject
from ...core.node.common.action.call_do_exchange_action import CallDoExchangeAction
from ...core.node.common.action.get_object_action import GetObjectAction

class Duet(WebRTCDuet):
    def __init__(
        self,
        node: Domain,
        target_id: str,
        signaling_client: SignalingClient,
        offer: bool = True,
    ):
        super().__init__(
            node,
            target_id,
            signaling_client,
            offer
        )

        self.flight_enabled = True
        location = ('localhost', 8999)

        # If this peer will not start the signaling process
        if not offer:
            # scheme = "grpc+tcp"
            # host = "localhost"
            # port = 8999
            # self.flight_client = FlightClientDuet(f"{scheme}://{host}:{port}")
            self.flight_client = FlightClientDuet(location)
        else:
            # Push a WebRTC offer request to the address.
            flight_args = {
                    'scheme': 'grpc+tcp',
                    'tls': False,
                    'host': 'localhost',
                    'port': 8999,
                    'verify_client': False,
                    'root_certificates': None,
                    'auth_handler': None,
                    'location': location,
                }
            flight_server = FlightServerDuet(flight_args, node)
            threading.Thread(target=flight_server.serve).start()
            self.flight_server = flight_server


    def send_immediate_msg_without_reply(
        self,
        msg: Union[
            SignedImmediateSyftMessageWithoutReply, ImmediateSyftMessageWithoutReply
        ],
        route_index: int = 0,
    ) -> None:
        flight_success = False
        if isinstance(msg, SaveObjectAction) and self.flight_enabled:
            if "Tensor" in str(type(msg.obj.data)):
                print("USING FLIGHT")
                put_thread = threading.Thread(target=self.flight_client.put_object, args=(msg.obj.id, msg.obj.data))
                put_thread.start()
                # self.flight_client.put_object(msg.obj.id, msg.obj.data.numpy())
                storable = StorableObject(
                    id=msg.obj.id,
                    data=None,
                    tags=msg.obj.tags,
                    description=msg.obj.description,
                    search_permissions=msg.obj.search_permissions,
                )
                msg = CallDoExchangeAction(msg.obj.id, obj=storable, address=msg.address)
                flight_success = True
        WebRTCDuet.send_immediate_msg_without_reply(self, msg, route_index)
        

    def send_immediate_msg_with_reply(
        self,
        msg: Union[SignedImmediateSyftMessageWithReply, ImmediateSyftMessageWithReply],
        route_index: int = 0,
    ) -> SyftMessage:
        if isinstance(msg, GetObjectAction) and self.flight_enabled:
            msg.flight = True
            response = WebRTCDuet.send_immediate_msg_with_reply(self, msg, route_index)
            if response.flight_transfer:
                print("USING FLIGHT")
                data = self.flight_client.get_object(msg.id_at_location)
                response.data = data
            return response
        else:
            return WebRTCDuet.send_immediate_msg_with_reply(self, msg, route_index)