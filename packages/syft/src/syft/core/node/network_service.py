# relative
from .network_msg_registry import NetworkMessageRegistry
from .node_service import NodeServiceClass


class NetworkServiceClass(NodeServiceClass):
    registry_type = NetworkMessageRegistry
