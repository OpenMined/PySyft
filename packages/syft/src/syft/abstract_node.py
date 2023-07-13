# stdlib
from enum import Enum
from typing import Callable
from typing import Optional
from typing import Union

# relative
from .serde.serializable import serializable
from .types.uid import UID


@serializable()
class NodeType(Enum):
    DOMAIN = "domain"
    NETWORK = "network"
    ENCLAVE = "enclave"
    GATEWAY = "gateway"

    def __str__(self) -> str:
        # Use values when transforming NodeType to str
        return self.value


class AbstractNode:
    id: Optional[UID]
    name: Optional[str]
    node_type: Optional[NodeType]

    def get_service(self, path_or_func: Union[str, Callable]) -> Callable:
        raise NotImplementedError
