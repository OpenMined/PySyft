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

# relative
from ....lib.python import String
from ....logger import error
from ...common.message import SignedMessage
from ...common.message import SyftMessage
from ...common.uid import UID
from ...io.location import Location
from ...io.location import SpecificLocation
from ..common.node import Node
from ..common.node_manager.association_request_manager import AssociationRequestManager
from ..common.node_manager.group_manager import GroupManager
from ..common.node_manager.role_manager import RoleManager
from ..common.node_manager.user_manager import UserManager
from ..common.node_service.association_request.association_request_service import (
    AssociationRequestService,
)
from ..common.node_service.node_setup.node_setup_service import NodeSetupService
from ..common.node_service.request_receiver.request_receiver_messages import (
    RequestMessage,
)
from ..common.node_service.role_manager.role_manager_service import RoleManagerService
from ..common.node_service.user_manager.user_manager_service import UserManagerService
from ..domain.client import DomainClient
from ..domain.domain import Domain
from .client import NetworkClient


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
        db: Any = None,
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
            db=db,
        )

        # specific location with name
        self.network = SpecificLocation(name=self.name)
        self.root_key = root_key

        # Database Management Instances
        self.users = UserManager(db_engine)
        self.roles = RoleManager(db_engine)
        self.groups = GroupManager(db_engine)
        self.association_requests = AssociationRequestManager(db_engine)

        # Grid Network Services
        self.immediate_services_with_reply.append(AssociationRequestService)
        self.immediate_services_with_reply.append(NodeSetupService)
        self.immediate_services_with_reply.append(RoleManagerService)
        self.immediate_services_with_reply.append(UserManagerService)

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

    def loud_print(self) -> None:
        install_path = os.path.abspath(
            os.path.join(os.path.realpath(__file__), "../../../../img/")
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
