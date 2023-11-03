# stdlib
from typing import Dict
from typing import List
from typing import Optional

# third party
from typing_extensions import Self

# relative
from ..abstract_node import AbstractNode
from ..node.credentials import SyftVerifyKey
from ..node.credentials import UserLoginCredentials
from ..types.syft_object import Context
from ..types.syft_object import SYFT_OBJECT_VERSION_1
from ..types.syft_object import SyftBaseObject
from ..types.syft_object import SyftObject
from ..types.uid import UID
from .user.user_roles import ROLE_TO_CAPABILITIES
from .user.user_roles import ServiceRole
from .user.user_roles import ServiceRoleCapability


class NodeServiceContext(Context, SyftObject):
    __canonical_name__ = "NodeServiceContext"
    __version__ = SYFT_OBJECT_VERSION_1
    id: Optional[UID]
    node: Optional[AbstractNode]


class AuthedServiceContext(NodeServiceContext):
    __canonical_name__ = "AuthedServiceContext"
    __version__ = SYFT_OBJECT_VERSION_1

    credentials: SyftVerifyKey
    role: ServiceRole = ServiceRole.NONE
    extra_kwargs: Dict = {}

    def capabilities(self) -> List[ServiceRoleCapability]:
        return ROLE_TO_CAPABILITIES.get(self.role, [])

    def with_credentials(self, credentials: SyftVerifyKey, role: ServiceRole):
        return AuthedServiceContext(credentials=credentials, role=role, node=self.node)

    def as_root_context(self):
        return AuthedServiceContext(
            credentials=self.node.verify_key, role=ServiceRole.ADMIN, node=self.node
        )


class UnauthedServiceContext(NodeServiceContext):
    __canonical_name__ = "UnauthedServiceContext"
    __version__ = SYFT_OBJECT_VERSION_1

    login_credentials: UserLoginCredentials
    node: Optional[AbstractNode]
    role: ServiceRole = ServiceRole.NONE


class ChangeContext(SyftBaseObject):
    __canonical_name__ = "ChangeContext"
    __version__ = SYFT_OBJECT_VERSION_1

    node: Optional[AbstractNode] = None
    approving_user_credentials: Optional[SyftVerifyKey]
    requesting_user_credentials: Optional[SyftVerifyKey]
    extra_kwargs: Dict = {}

    @staticmethod
    def from_service(context: AuthedServiceContext) -> Self:
        return ChangeContext(
            node=context.node,
            approving_user_credentials=context.credentials,
            extra_kwargs=context.extra_kwargs,
        )
