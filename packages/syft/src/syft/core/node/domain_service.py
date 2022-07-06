# relative
from .domain_msg_registry import DomainMessageRegistry
from .node_service import NodeServiceClass


class DomainServiceClass(NodeServiceClass):
    registry_type = DomainMessageRegistry
