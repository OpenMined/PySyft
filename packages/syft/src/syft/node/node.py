# stdlib
from enum import Enum
from typing import Callable
from typing import Optional
from typing import Union

# relative
from ..serde.serializable import serializable
from ..types.uid import UID
from .credentials import SyftSigningKey


@serializable()
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
