# relative
from ...adp.adversarial_accountant import AdversarialAccountant
from ..abstract.node_service_interface import NodeServiceInterface
from ..common.node_manager.association_request_manager import AssociationRequestManager
from ..common.node_manager.dataset_manager import DatasetManager
from ..common.node_manager.group_manager import GroupManager
from ..common.node_manager.role_manager import RoleManager
from ..common.node_manager.setup_manager import SetupManager
from ..common.node_manager.user_manager import UserManager


class DomainInterface(NodeServiceInterface):
    groups: GroupManager
    users: UserManager
    roles: RoleManager
    association_requests: AssociationRequestManager
    datasets: DatasetManager
    setup: SetupManager
    acc: AdversarialAccountant
