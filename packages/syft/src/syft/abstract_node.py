# stdlib
from collections.abc import Callable
from enum import Enum
from typing import TYPE_CHECKING

# relative
from .serde.serializable import serializable
from .types.uid import UID

if TYPE_CHECKING:
    # relative
    from .service.service import AbstractService


@serializable()
class NodeType(str, Enum):
    DOMAIN = "domain"
    NETWORK = "network"
    ENCLAVE = "enclave"
    GATEWAY = "gateway"

    def __str__(self) -> str:
        # Use values when transforming NodeType to str
        return self.value


@serializable()
class NodeSideType(str, Enum):
    LOW_SIDE = "low"
    HIGH_SIDE = "high"

    def __str__(self) -> str:
        return self.value


class AbstractNode:
    id: UID | None
    name: str | None
    node_type: NodeType | None
    node_side_type: NodeSideType | None
    in_memory_workers: bool

    def get_service(self, path_or_func: str | Callable) -> "AbstractService":
        raise NotImplementedError
