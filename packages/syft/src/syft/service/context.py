# stdlib
from typing import Any

# third party
from typing_extensions import Self

# relative
from ..abstract_server import AbstractServer
from ..abstract_server import ServerSideType
from ..server.credentials import SyftVerifyKey
from ..server.credentials import UserLoginCredentials
from ..types.syft_object import Context
from ..types.syft_object import SYFT_OBJECT_VERSION_1
from ..types.syft_object import SyftBaseObject
from ..types.syft_object import SyftObject
from ..types.uid import UID
from .user.user_roles import ROLE_TO_CAPABILITIES
from .user.user_roles import ServiceRole
from .user.user_roles import ServiceRoleCapability


class ServerServiceContext(Context, SyftObject):
    __canonical_name__ = "ServerServiceContext"
    __version__ = SYFT_OBJECT_VERSION_1

    id: UID | None = None  # type: ignore[assignment]
    server: AbstractServer


class AuthedServiceContext(ServerServiceContext):
    __canonical_name__ = "AuthedServiceContext"
    __version__ = SYFT_OBJECT_VERSION_1

    credentials: SyftVerifyKey
    role: ServiceRole = ServiceRole.NONE
    job_id: UID | None = None
    extra_kwargs: dict = {}
    has_execute_permissions: bool = False
    is_blocking_api_call: bool = False
    client_warnings: list[str] = []

    @property
    def dev_mode(self) -> Any:
        return self.server.dev_mode  # type: ignore

    def add_warning(self, message: str) -> None:
        self.client_warnings.append(message)

    def capabilities(self) -> list[ServiceRoleCapability]:
        return ROLE_TO_CAPABILITIES.get(self.role, [])

    def with_credentials(self, credentials: SyftVerifyKey, role: ServiceRole) -> Self:
        return AuthedServiceContext(
            credentials=credentials, role=role, server=self.server
        )

    @property
    def is_l0_lowside(self) -> bool:
        """Returns True if this is a low side of a Level 0 deployment"""
        return self.server.server_side_type == ServerSideType.LOW_SIDE

    @property
    def server_allows_execution_for_ds(self) -> bool:
        """Returns True if this is a low side of a Level 0 deployment"""
        return not self.is_l0_lowside

    def as_root_context(self) -> Self:
        return AuthedServiceContext(
            credentials=self.server.verify_key,
            role=ServiceRole.ADMIN,
            server=self.server,
        )

    @property
    def job(self):  # type: ignore
        # TODO: fix the mypy issue. The return type is Optional[Job].
        # but we can't import Job since it's a circular import
        if self.job_id is None:
            return None
        return self.server.job_stash.get_by_uid(
            self.credentials, self.job_id
        ).ok()  # if this fails, it will return None


class UnauthedServiceContext(ServerServiceContext):
    __canonical_name__ = "UnauthedServiceContext"
    __version__ = SYFT_OBJECT_VERSION_1

    server: AbstractServer
    login_credentials: UserLoginCredentials | None = None
    role: ServiceRole = ServiceRole.NONE


class ChangeContext(SyftBaseObject):
    __canonical_name__ = "ChangeContext"
    __version__ = SYFT_OBJECT_VERSION_1

    server: AbstractServer
    approving_user_credentials: SyftVerifyKey | None = None
    requesting_user_credentials: SyftVerifyKey | None = None
    extra_kwargs: dict = {}

    @classmethod
    def from_service(cls, context: AuthedServiceContext) -> Self:
        return cls(
            server=context.server,
            approving_user_credentials=context.credentials,
            extra_kwargs=context.extra_kwargs,
        )

    def to_service_ctx(self) -> AuthedServiceContext:
        return AuthedServiceContext(
            server=self.server,
            credentials=self.approving_user_credentials,
            extra_kwargs=self.extra_kwargs,
        )
