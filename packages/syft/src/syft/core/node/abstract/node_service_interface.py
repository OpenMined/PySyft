# relative
from ..abstract.node import AbstractNode
from ..common.node_manager.node_manager import NodeManager
from ..common.node_manager.node_route_manager import NodeRouteManager
from ..common.node_manager.role_manager import RoleManager
from ..common.node_manager.user_manager import UserManager
from ..common.node_service.node_credential.node_credentials import NodeCredentials


class NodeServiceInterface(AbstractNode):
    node: NodeManager
    node_route: NodeRouteManager
    users: UserManager
    roles: RoleManager

    def get_credentials(self) -> NodeCredentials:
        raise NotImplementedError
