# stdlib
from typing import List
from typing import Optional

# relative
from ..service.context import NodeServiceContext
from ..service.response import SyftError
from ..service.user.user_roles import ServiceRole


class PySyftException(Exception):
    """Base class for all PySyft exceptions."""

    def __init__(self, message: str, roles: Optional[List[ServiceRole]] = None):
        super().__init__(message)
        self.message = message
        self.roles = roles if roles else [ServiceRole.ADMIN]

    def raise_with_context(self, context: NodeServiceContext):
        self.context = context
        return self

    def handle(self) -> SyftError:
        # if self.context and self.context.role in self.roles:
        return SyftError(message=self.message)
        # else:
        #     return SyftError(message="Access denied to exception message.")
