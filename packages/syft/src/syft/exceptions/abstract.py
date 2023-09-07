# stdlib
from abc import ABC
from abc import abstractclassmethod
from enum import Enum
from typing import List

# relative
from ..service.context import NodeServiceContext
from ..service.response import SyftError
from ..service.user.user_roles import ServiceRole


class PySyftException(ABC, Exception):
    """Base class for all PySyft exceptions."""

    def __init__(self, message: str, context: NodeServiceContext):
        super().__init__(message)
        self.context = context
        self.message = message

    @abstractclassmethod
    def roles(self) -> List[Enum]:
        pass

    def handle(self) -> SyftError:
        if self.context.role == ServiceRole.ADMIN or self.context.role in self.roles():
            return SyftError(message=self.message)
        else:
            return SyftError(message=f"Access denied to exception message!")
