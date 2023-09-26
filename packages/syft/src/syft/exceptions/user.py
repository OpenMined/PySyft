# stdlib

# relative
from ..service.user.user_roles import ServiceRole
from .exception import PySyftException

UserAlreadyExistsException = PySyftException(
    message="User already exists", roles=[ServiceRole.ADMIN]
)

UserDoesNotExistException = PySyftException(
    message="User does not exist", roles=[ServiceRole.ADMIN]
)
