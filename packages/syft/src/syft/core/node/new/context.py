# stdlib
from typing import Optional

# relative
from ..common.node_table.syft_object import Context
from ..common.node_table.syft_object import SYFT_OBJECT_VERSION_1
from ..common.node_table.syft_object import SyftObject
from .credentials import SyftVerifyKey
from .credentials import UserLoginCredentials
from .node import NewNode


class NodeServiceContext(Context, SyftObject):
    __canonical_name__ = "NodeServiceContext"
    __version__ = SYFT_OBJECT_VERSION_1
    node: Optional[NewNode]


class AuthedServiceContext(NodeServiceContext):
    __canonical_name__ = "AuthedServiceContext"
    __version__ = SYFT_OBJECT_VERSION_1

    credentials: SyftVerifyKey


class UnauthedServiceContext(NodeServiceContext):
    __canonical_name__ = "UnauthedServiceContext"
    __version__ = SYFT_OBJECT_VERSION_1

    login_credentials: UserLoginCredentials
    node: Optional[NewNode]
