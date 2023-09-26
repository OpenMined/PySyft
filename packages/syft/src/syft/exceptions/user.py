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


def InvalidSearchParamsException(valid_search_params: str) -> PySyftException:
    return PySyftException(
        message=f"Invalid Search parameters. \
    Allowed params: {valid_search_params}",
        roles=[ServiceRole.ADMIN],
    )


def GenericSearchException(message: str) -> PySyftException:
    return PySyftException(
        message=message,
        roles=[ServiceRole.ADMIN],
    )
