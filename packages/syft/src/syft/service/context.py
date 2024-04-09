# stdlib
from typing import Any

# third party
from typing_extensions import Self

# relative
from ..abstract_node import AbstractNode
from ..node.credentials import SyftVerifyKey
from ..node.credentials import UserLoginCredentials
from ..types.syft_object import Context
from ..types.syft_object import SYFT_OBJECT_VERSION_2
from ..types.syft_object import SyftBaseObject
from ..types.syft_object import SyftObject
from ..types.uid import UID
from .user.user_roles import ROLE_TO_CAPABILITIES
from .user.user_roles import ServiceRole
from .user.user_roles import ServiceRoleCapability


class NodeServiceContext(Context, SyftObject):
    __canonical_name__ = "NodeServiceContext"
    __version__ = SYFT_OBJECT_VERSION_2

    id: UID | None = None  # type: ignore[assignment]
    node: AbstractNode


class AuthedServiceContext(NodeServiceContext):
    __canonical_name__ = "AuthedServiceContext"
    __version__ = SYFT_OBJECT_VERSION_2

    credentials: SyftVerifyKey
    role: ServiceRole = ServiceRole.NONE
    job_id: UID | None = None
    extra_kwargs: dict = {}
    has_execute_permissions: bool = False

    @property
    def dev_mode(self) -> Any:
        return self.node.dev_mode  # type: ignore

    def capabilities(self) -> list[ServiceRoleCapability]:
        return ROLE_TO_CAPABILITIES.get(self.role, [])

    def with_credentials(self, credentials: SyftVerifyKey, role: ServiceRole) -> Self:
        return AuthedServiceContext(credentials=credentials, role=role, node=self.node)

    def as_root_context(self) -> Self:
        return AuthedServiceContext(
            credentials=self.node.verify_key, role=ServiceRole.ADMIN, node=self.node
        )

    @property
    def job(self):  # type: ignore
        # TODO: fix the mypy issue. The return type is Optional[Job].
        # but we can't import Job since it's a circular import
        if self.job_id is None:
            return None
        res = self.node.job_stash.get_by_uid(self.credentials, self.job_id)
        if res.is_err():
            return None
        else:
            return res.ok()


class UnauthedServiceContext(NodeServiceContext):
    __canonical_name__ = "UnauthedServiceContext"
    __version__ = SYFT_OBJECT_VERSION_2

    login_credentials: UserLoginCredentials
    node: AbstractNode
    role: ServiceRole = ServiceRole.NONE


class ChangeContext(SyftBaseObject):
    __canonical_name__ = "ChangeContext"
    __version__ = SYFT_OBJECT_VERSION_2

    node: AbstractNode
    approving_user_credentials: SyftVerifyKey | None = None
    requesting_user_credentials: SyftVerifyKey | None = None
    extra_kwargs: dict = {}

    @classmethod
    def from_service(cls, context: AuthedServiceContext) -> Self:
        return cls(
            node=context.node,
            approving_user_credentials=context.credentials,
            extra_kwargs=context.extra_kwargs,
        )

    def to_service_ctx(self) -> AuthedServiceContext:
        return AuthedServiceContext(
            node=self.node,
            credentials=self.approving_user_credentials,
            extra_kwargs=self.extra_kwargs,
        )
