from abc import ABC, abstractclassmethod
from enum import Enum
from typing import List
from ..service.response import SyftError
from ..service.context import NodeServiceContext
from .abstract import PySyftException
from ..service.user.user_roles import ServiceRole


class UserAlreadyExistsException(PySyftException):
    """Base class for all PySyft exceptions."""
    
    def __init__(self,context: NodeServiceContext, message: str = "User already exists"):
        super().__init__(message, context)
    
    def roles(self) -> List[Enum]:
        return [ServiceRole.DATA_SCIENTIST]