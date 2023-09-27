# stdlib
from traceback import format_tb
from types import TracebackType
from typing import List
from typing import Optional
import uuid

# relative
from ..service.context import AuthedServiceContext
from ..service.context import NodeServiceContext
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
        self.error_uuid = uuid.uuid4()
        self.message = message
        self.roles = roles or [ServiceRole.ADMIN]
        self.private_message = private_message or DEFAULT_PRIVATE_ERROR_MESSAGE

        # context determines if the message should be public or private.
        # If the context is unknown when the exception is raised, it can be set
        # via the with_context() method.
        self.context = context

        # We carry the traceback to be able to log it later. Traceback can be
        # re-set via the with_traceback() method.
        self.traceback = format_tb(traceback) if traceback else None

        if ServiceRole.ADMIN not in self.roles:
            self.roles.append(ServiceRole.ADMIN)

        # With message and roles set, we can now call handle() to get the
        # Exception message. We keep the message in self.message for potential
        # logging opportunities.
        output_message = self.handle()
        super().__init__(output_message)

    def handle(self) -> str:
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
        return f"{output_message} [Ref: {self.error_uuid}]"

    @property
    def uuid(self) -> str:
        return str(self.error_uuid)

    def with_context(self, context: NodeServiceContext) -> "PySyftException":
        self.context = context
        return self

    def with_traceback(self, tb: TracebackType) -> "PySyftException":
        self.traceback = format_tb(tb)
        return self
