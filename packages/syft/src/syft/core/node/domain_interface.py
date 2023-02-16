# third party
from pydantic import BaseSettings

# relative
from .abstract.node_service_interface import NodeServiceInterface
from .common.node_manager.association_request_manager import (
    NoSQLAssociationRequestManager,
)
from .common.node_manager.dataset_manager import NoSQLDatasetManager
from .common.node_manager.oblv_key_manager import NoSQLOblvKeyManager
from .common.node_manager.role_manager import NewRoleManager
from .common.node_manager.setup_manager import NoSQLSetupManager
from .common.node_manager.task_manager import NoSQLTaskManager
from .common.node_manager.user_manager import NoSQLUserManager


class DomainInterface(NodeServiceInterface):
    users: NoSQLUserManager
    roles: NewRoleManager
    association_requests: NoSQLAssociationRequestManager
    datasets: NoSQLDatasetManager
    setup: NoSQLSetupManager
    settings: BaseSettings
    tasks: NoSQLTaskManager
    oblv_keys: NoSQLOblvKeyManager
