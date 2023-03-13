# stdlib
from typing import Optional

# third party
from typing_extensions import Self

# relative
from .credentials import SyftVerifyKey
from .credentials import UserLoginCredentials
from .node import NewNode
from .syft_object import Context
from .syft_object import SYFT_OBJECT_VERSION_1
from .syft_object import SyftBaseObject
from .syft_object import SyftObject


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


class ChangeContext(SyftBaseObject):
    node: Optional[NewNode] = None
    approving_user_credentials: Optional[SyftVerifyKey]
    requesting_user_credentials: Optional[SyftVerifyKey]

    @staticmethod
    def from_service(context: AuthedServiceContext) -> Self:
        return ChangeContext(
            node=context.node, approving_user_credentials=context.credentials
        )
