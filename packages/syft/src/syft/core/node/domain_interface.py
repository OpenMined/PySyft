# third party
from pydantic import BaseSettings

# relative
from .abstract.node_service_interface import NodeServiceInterface
from .common.node_manager.association_request_manager import (
    NoSQLAssociationRequestManager,
)
from .common.node_manager.dataset_manager import DatasetManager
from .common.node_manager.role_manager import NewRoleManager
from .common.node_manager.setup_manager import SetupManager
from .common.node_manager.user_manager import NoSQLUserManager


class DomainInterface(NodeServiceInterface):
    users: NoSQLUserManager
    roles: NewRoleManager
    association_requests: NoSQLAssociationRequestManager
    datasets: DatasetManager
    setup: SetupManager
    settings: BaseSettings
