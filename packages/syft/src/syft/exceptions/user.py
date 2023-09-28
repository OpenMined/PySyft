# stdlib

# relative
from ..service.user.user_roles import ServiceRole
from .exception import PySyftException

UserAlreadyExistsException = PySyftException(
    message="User already exists", roles=[ServiceRole.ADMIN]
)

UserNotFoundException = PySyftException(
    message="User not found", roles=[ServiceRole.ADMIN]
)
AdminVerifyKeyException = PySyftException(
    message="Failed to get admin verify_key", roles=[ServiceRole.ADMIN]
)
RoleNotFoundException = PySyftException(
    message="Role not found", roles=[ServiceRole.ADMIN]
)


def NoUserWithVerifyKeyException(verify_key: str) -> PySyftException:
    return PySyftException(
        message=f"No User with verify_key: {verify_key}", roles=[ServiceRole.ADMIN]
    )


def NoUserWithEmailException(email: str) -> PySyftException:
    return PySyftException(
        message=f"No User with email: {email}", roles=[ServiceRole.ADMIN]
    )


def StashRetrievalException(message: str) -> PySyftException:
    return PySyftException(message=message, roles=[ServiceRole.ADMIN])
