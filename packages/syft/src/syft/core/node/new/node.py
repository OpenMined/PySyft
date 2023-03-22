# stdlib
from enum import Enum
from typing import Callable
from typing import Optional
from typing import Union

# relative
from .credentials import SyftSigningKey
from .serializable import serializable
from .uid import UID


@serializable(recursive_serde=True)
class NodeType(Enum):
    DOMAIN = "domain"
    NETWORK = "network"
    ENCLAVE = "enclave"


class NewNode:
    id: Optional[UID]
    name: Optional[str]
    signing_key: Optional[SyftSigningKey]
    node_type: Optional[NodeType]

    def get_service(self, path_or_func: Union[str, Callable]) -> Callable:
        raise NotImplementedError
