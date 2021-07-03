# stdlib
from typing import Optional
from typing import Union
from typing import Any

# third party
from nacl.signing import SigningKey
from nacl.signing import VerifyKey

# relative
from ....logger import error
from ...common.message import SignedMessage
from ...common.message import SyftMessage
from ...common.uid import UID
from ...io.location import Location
from ...io.location import SpecificLocation
from ..common.node import Node
from ..domain.client import DomainClient
from ..domain.domain import Domain
from .client import NetworkClient
from ..common.managers.setup_manager import SetupManager
from ..common.managers.role_manager import RoleManager
from ..common.managers.user_manager import UserManager
from ..common.managers.group_manager import GroupManager


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
        db_path: Optional[str] = None,
        db_engine: Any = None,
    ):
        super().__init__(
            name=name,
            network=network,
            domain=domain,
            device=device,
            vm=vm,
            signing_key=signing_key,
            verify_key=verify_key,
            db_path=db_path,
            db_engine=db_engine
        )

        # specific location with name
        self.network = SpecificLocation(name=self.name)
        self.root_key = root_key

        # # Database Management Instances
        self.users = UserManager(db_engine)
        self.roles = RoleManager(db_engine)
        self.groups = GroupManager(db_engine)
        self.setup = SetupManager(db_engine)
        # self.environments = EnvironmentManager(db_engine)
        # self.association_requests = AssociationRequestManager(db_engine)
        # self.data_requests = RequestManager(db_engine)
        # self.datasets = DatasetManager(db_engine)

        self._register_services()
        self.post_init()
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
