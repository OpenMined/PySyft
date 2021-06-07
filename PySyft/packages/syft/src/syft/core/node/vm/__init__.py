# stdlib
from typing import Dict  # noqa: F401

# syft relative
from ..common.service.node_service import NodeService  # noqa: F401
from .client import VirtualMachineClient
from .vm import VirtualMachine

message_service_mapping: Dict[str, NodeService] = {}

__all__ = [
    "NodeService",
    "VirtualMachineClient",
    "VirtualMachine",
    "message_service_mapping",
]
