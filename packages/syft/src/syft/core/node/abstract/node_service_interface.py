# relative
from ..abstract.node import AbstractNode
from ..common.node_manager.node_manager import NodeManager
from ..common.node_manager.node_route_manager import NodeRouteManager


class NodeServiceInterface(AbstractNode):
    node: NodeManager
    node_route: NodeRouteManager
