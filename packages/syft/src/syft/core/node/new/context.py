# stdlib
from typing import Optional

# relative
from ..common.node_table.syft_object import SyftObject
from .credentials import SyftVerifyKey
from .node import NewNode


class NodeServiceContext(SyftObject):
    node: Optional[NewNode]


class AuthedServiceContext(NodeServiceContext):
    credentials: SyftVerifyKey
