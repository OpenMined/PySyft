# future
from __future__ import annotations

# stdlib
import os
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

# third party
import ascii_magic
from nacl.signing import SigningKey
from nacl.signing import VerifyKey
from pydantic import BaseSettings

# relative
from ...lib.python import String
from ...logger import error
from ..common.message import SignedImmediateSyftMessageWithReply
from ..common.message import SignedMessage
from ..common.message import SyftMessage
from ..common.uid import UID
from ..io.location import Location
from ..io.location import SpecificLocation
from .common.node import Node
from .common.node_manager.association_request_manager import AssociationRequestManager
from .common.node_manager.node_manager import NodeManager
from .common.node_manager.node_route_manager import NodeRouteManager
from .common.node_manager.role_manager import RoleManager
from .common.node_manager.user_manager import UserManager
from .common.node_service.association_request.association_request_service import (
    AssociationRequestService,
)
from .common.node_service.association_request.association_request_service import (
    AssociationRequestWithoutReplyService,
)
from .common.node_service.network_search.network_search_service import (
    NetworkSearchService,
)
from .common.node_service.node_setup.node_setup_messages import (
    CreateInitialSetUpMessage,
)
from .common.node_service.node_setup.node_setup_service import NodeSetupService
from .common.node_service.peer_discovery.peer_discovery_service import (
    PeerDiscoveryService,
)
from .common.node_service.ping.ping_service import PingService
from .common.node_service.request_receiver.request_receiver_messages import (
    RequestMessage,
)
from .common.node_service.role_manager.role_manager_service import RoleManagerService
from .common.node_service.user_manager.user_manager_service import UserManagerService
from .common.node_service.vpn.vpn_service import VPNConnectService
from .common.node_service.vpn.vpn_service import VPNJoinSelfService
from .common.node_service.vpn.vpn_service import VPNJoinService
from .common.node_service.vpn.vpn_service import VPNRegisterService
from .common.node_service.vpn.vpn_service import VPNStatusService
from .domain import Domain
from .domain_client import DomainClient
from .network_client import NetworkClient
from .network_service import NetworkServiceClass


class Network(Node):

    network: SpecificLocation

    child_type = Domain
    client_type = NetworkClient
    child_type_client_type = DomainClient

    def __init__(
        self,
        name: Optional[str],
        network: SpecificLocation = SpecificLocation(),
        domain: Optional[Location] = None,
        device: Optional[Location] = None,
        vm: Optional[Location] = None,
        signing_key: Optional[SigningKey] = None,
        verify_key: Optional[VerifyKey] = None,
        root_key: Optional[VerifyKey] = None,
        db_engine: Any = None,
        settings: Optional[BaseSettings] = None,
    ):
        super().__init__(
            name=name,
            network=network,
            domain=domain,
            device=device,
            vm=vm,
            signing_key=signing_key,
            verify_key=verify_key,
            db_engine=db_engine,
            settings=settings,
        )

        # share settings with the FastAPI application level
        self.settings = settings

        # specific location with name
        self.network = SpecificLocation(name=self.name)
        self.root_key = root_key

        # Database Management Instances
        self.users = UserManager(db_engine)
        self.roles = RoleManager(db_engine)
        self.node = NodeManager(db_engine)
        self.node_route = NodeRouteManager(db_engine)
        self.association_requests = AssociationRequestManager(db_engine)

        # Grid Network Services
        self.immediate_services_with_reply.append(AssociationRequestService)
        self.immediate_services_with_reply.append(NodeSetupService)
        self.immediate_services_with_reply.append(RoleManagerService)
        self.immediate_services_with_reply.append(UserManagerService)
        self.immediate_services_with_reply.append(VPNConnectService)
        self.immediate_services_with_reply.append(VPNJoinService)
        self.immediate_services_with_reply.append(VPNRegisterService)
        self.immediate_services_with_reply.append(VPNStatusService)
        self.immediate_services_with_reply.append(VPNJoinSelfService)
        self.immediate_services_with_reply.append(PingService)
        self.immediate_services_with_reply.append(NetworkSearchService)
        self.immediate_services_with_reply.append(PeerDiscoveryService)

        # TODO: New Service registration process
        self.immediate_services_with_reply.append(NetworkServiceClass)

        self.immediate_services_without_reply.append(
            AssociationRequestWithoutReplyService
        )

        self.requests: List[RequestMessage] = list()
        # available_device_types = set()
        # TODO: add available compute types

        # default_device = None
        # TODO: add default compute type

        self._register_services()
        self.request_handlers: List[Dict[Union[str, String], Any]] = []
        self.handled_requests: Dict[Any, float] = {}

        self.post_init()

    def post_init(self) -> None:
        super().post_init()
        self.set_node_uid()

    def initial_setup(  # nosec
        self,
        signing_key: SigningKey,
        first_superuser_name: str = "Jane Doe",
        first_superuser_email: str = "info@openmined.org",
        first_superuser_password: str = "changethis",
        first_superuser_budget: float = 5.55,
        domain_name: str = "BigHospital",
    ) -> Network:
        Node.set_keys(node=self, signing_key=signing_key)

        # Build Syft Message
        msg: SignedImmediateSyftMessageWithReply = CreateInitialSetUpMessage(
            address=self.address,
            name=first_superuser_name,
            email=first_superuser_email,
            password=first_superuser_password,
            domain_name=domain_name,
            budget=first_superuser_budget,
            reply_to=self.address,
            signing_key=signing_key,
        ).sign(signing_key=self.signing_key)

        # Process syft message
        _ = self.recv_immediate_msg_with_reply(msg=msg).message

        return self

    def loud_print(self) -> None:
        try:
            install_path = os.path.abspath(
                os.path.join(os.path.realpath(__file__), "../../../img/")
            )
            ascii_magic.to_terminal(
                ascii_magic.from_image_file(
                    img_path=install_path + "/pygrid.png", columns=83
                )
            )

            print(
                r"""
                                                        |\ |  _ |_      _   _ |
                                                        | \| (- |_ \)/ (_) |  |(
    """
            )
        except Exception:
            print("NETOWRK NODE (print fail backup)")

    @property
    def icon(self) -> str:
        return "🔗"

    @property
    def id(self) -> UID:
        return self.network.id

    def message_is_for_me(self, msg: Union[SyftMessage, SignedMessage]) -> bool:
        # this needs to be defensive by checking network_id NOT network.id or it breaks
        try:
            return msg.address.network_id == self.id and msg.address.domain is None
        except Exception as e:
            error(f"Error checking if {msg.pprint} is for me on {self.pprint}. {e}")
            return False
