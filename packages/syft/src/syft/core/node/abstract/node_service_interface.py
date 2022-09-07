# relative
from ..abstract.node import AbstractNode
from ..common.node_manager.node_manager import NodeManager
from ..common.node_manager.node_route_manager import NodeRouteManager
from ..common.node_manager.role_manager import NewRoleManager
from ..common.node_manager.user_manager import NoSQLUserManager
from ..common.node_service.node_credential.node_credentials import NodeCredentials


class NodeServiceInterface(AbstractNode):
    node: NodeManager
    node_route: NodeRouteManager
    users: NoSQLUserManager
    roles: NewRoleManager

    def get_credentials(self) -> NodeCredentials:
        raise NotImplementedError
