# stdlib
import traceback
from types import TracebackType
from typing import List
from typing import Optional
import uuid

# relative
from ..service.context import AuthedServiceContext
from ..service.context import NodeServiceContext
from ..service.response import SyftError
from ..service.user.user_roles import ServiceRole

DEFAULT_PRIVATE_ERROR_MESSAGE = "Private error. Contact the node administrator."


class PySyftException(Exception):
    """Base class for all PySyft exceptions."""

    def __init__(
        self,
        message: str,
        roles: Optional[List[ServiceRole]] = None,
        context: Optional[NodeServiceContext] = None,
        traceback: Optional[TracebackType] = None,
        private_message: Optional[str] = None,
    ):
        super().__init__("PySyftException")  # do not expose message e.g. ExceptionInfo
        self.error_uuid = uuid.uuid4()
        self.message = message
        self.roles = roles if roles else [ServiceRole.ADMIN]
        self.context = context
        self.traceback = traceback
        self.private_message = private_message or DEFAULT_PRIVATE_ERROR_MESSAGE

        if ServiceRole.ADMIN not in self.roles:
            self.roles.append(ServiceRole.ADMIN)

    def handle(self) -> SyftError:
        output_message = (
            self.message
            if any(
                [role in [ServiceRole.NONE, ServiceRole.GUEST] for role in self.roles]
            )
            or (
                isinstance(self.context, AuthedServiceContext)
                and self.context.role in self.roles
            )
            else self.private_message
        )

        return SyftError(message=f"{output_message} [Ref: {self.error_uuid}]")

    def with_context(self, context: NodeServiceContext):
        self.context = context
        return self

    def with_traceback(self, __tb: TracebackType):
        self.traceback = traceback.format_tb(__tb)
        return self

    def get_uuid(self) -> str:
        return str(self.error_uuid)
