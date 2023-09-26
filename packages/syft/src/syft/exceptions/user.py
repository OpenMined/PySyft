# stdlib

# relative
from ..service.user.user_roles import ServiceRole
from .exception import PySyftException

UserAlreadyExistsException = PySyftException(
    message="User already exists", roles=[ServiceRole.ADMIN]
)

AdminEnclaveLoginException = PySyftException(
    message="Admins are not allowed to login to Enclaves. \
    Kindly register a new data scientist account by your_client.register.",
    roles=[ServiceRole.ADMIN],
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


def UserDoesNotExistException(uid: str, email: str, err: any) -> PySyftException:
    if uid is not None:
        return PySyftException(
            message=f"No user exists for given id: {uid}", roles=[ServiceRole.ADMIN]
        )
    elif email is not None and err is not None:
        return PySyftException(
            message=f"No user exists with {email} and supplied password.",
            roles=[ServiceRole.ADMIN],
        )
    elif email is not None and err is None:
        return PySyftException(
            message=f"Failed to retrieve user with {email} with error: {err}",
            roles=[ServiceRole.ADMIN],
        )
    return PySyftException(message="User does not exist", roles=[ServiceRole.ADMIN])
