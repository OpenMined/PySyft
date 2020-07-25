from . import service  # noqa: F401
from ..abstract.service.node_service import NodeService  # noqa: F401
from typing import Dict  # noqa: F401

message_service_mapping: Dict[str, NodeService] = {}
