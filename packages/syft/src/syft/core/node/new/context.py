# stdlib
from typing import List
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
from .uid import UID
from .user_roles import ROLE_TO_CAPABILITIES
from .user_roles import ServiceRole
from .user_roles import ServiceRoleCapability


class NodeServiceContext(Context, SyftObject):
    __canonical_name__ = "NodeServiceContext"
    __version__ = SYFT_OBJECT_VERSION_1
    id: Optional[UID]
    node: Optional[NewNode]


class AuthedServiceContext(NodeServiceContext):
    __canonical_name__ = "AuthedServiceContext"
    __version__ = SYFT_OBJECT_VERSION_1

    credentials: SyftVerifyKey
    role: ServiceRole = ServiceRole.NONE

    def capabilities(self) -> List[ServiceRoleCapability]:
        return ROLE_TO_CAPABILITIES.get(self.role, [])


class UnauthedServiceContext(NodeServiceContext):
    __canonical_name__ = "UnauthedServiceContext"
    __version__ = SYFT_OBJECT_VERSION_1

    login_credentials: UserLoginCredentials
    node: Optional[NewNode]
    role: ServiceRole = ServiceRole.NONE


class ChangeContext(SyftBaseObject):
    node: Optional[NewNode] = None
    approving_user_credentials: Optional[SyftVerifyKey]
    requesting_user_credentials: Optional[SyftVerifyKey]

    @staticmethod
    def from_service(context: AuthedServiceContext) -> Self:
        return ChangeContext(
            node=context.node, approving_user_credentials=context.credentials
        )
