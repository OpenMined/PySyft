# stdlib
from abc import ABC
from abc import abstractclassmethod
from enum import Enum
from typing import List

# relative
from ..service.context import NodeServiceContext
from ..service.response import SyftError
from ..service.user.user_roles import ServiceRole
from .abstract import PySyftException


class UserAlreadyExistsException(PySyftException):
    """Base class for all PySyft exceptions."""

    def __init__(
        self, context: NodeServiceContext, message: str = "User already exists"
    ):
        super().__init__(message, context)

    def roles(self) -> List[Enum]:
        return [ServiceRole.DATA_SCIENTIST]
