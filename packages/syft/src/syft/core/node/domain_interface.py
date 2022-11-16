# third party
from pydantic import BaseSettings

# relative
from .abstract.node_service_interface import NodeServiceInterface
from .common.node_manager.association_request_manager import AssociationRequestManager
from .common.node_manager.dataset_manager import DatasetManager
from .common.node_manager.role_manager import RoleManager
from .common.node_manager.setup_manager import SetupManager
from .common.node_manager.user_manager import UserManager


class DomainInterface(NodeServiceInterface):
    users: UserManager
    roles: RoleManager
    association_requests: AssociationRequestManager
    datasets: DatasetManager
    setup: SetupManager
    settings: BaseSettings
