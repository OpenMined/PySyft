# stdlib

# relative
from ..service.user.user_roles import ServiceRole
from .exception import PySyftException

AdminEnclaveLoginException = PySyftException(
    message="Admins are not allowed to login to Enclaves. \
    Kindly register a new data scientist account by your_client.register.",
    roles=[ServiceRole.ADMIN],
)

AdminVerifyKeyException = PySyftException(
    message="Failed to get admin verify_key", roles=[ServiceRole.ADMIN]
)


def DeleteUserPermissionsException(user_role: str, target_role: str) -> PySyftException:
    return PySyftException(
        message=f"As a {user_role} you have no permission to delete user with {target_role} permission",
        roles=[ServiceRole.ADMIN],
    )


def FailedToUpdateUserWithUIDException(uid: str, err: any) -> PySyftException:
    return PySyftException(
        message=f"Failed to update user with UID: {uid}. Error: {err}",
        roles=[ServiceRole.ADMIN],
    )


def GenericException(message: str) -> PySyftException:
    return PySyftException(
        message=message,
        roles=[ServiceRole.ADMIN],
    )


def InvalidSearchParamsException(valid_search_params: str) -> PySyftException:
    return PySyftException(
        message=f"Invalid Search parameters. \
    Allowed params: {valid_search_params}",
        roles=[ServiceRole.ADMIN],
    )


NoUserFoundException = PySyftException(
    message="User does not exist", roles=[ServiceRole.ADMIN]
)


def NoUserWithEmailException(email: str, err: any) -> PySyftException:
    if err is not None:
        return PySyftException(
            message=f"Failed to retrieve user with {email} with error: {err}",
            roles=[ServiceRole.ADMIN],
        )
    return PySyftException(
        message=f"No User with email: {email}", roles=[ServiceRole.ADMIN]
    )


def NoUserWithUIDException(uid: str) -> PySyftException:
    return PySyftException(
        message=f"No user exists for given id: {uid}", roles=[ServiceRole.ADMIN]
    )


def NoUserWithVerifyKeyException(verify_key: str) -> PySyftException:
    return PySyftException(
        message=f"No User with verify_key: {verify_key}", roles=[ServiceRole.ADMIN]
    )


def RegisterUserPermissionsException(domain: str) -> PySyftException:
    return PySyftException(
        message=f"You don't have permission to create an account "
        f"on the domain: {domain}. Please contact the Domain Owner.",
        roles=[ServiceRole.ADMIN],
    )


def RoleNotAllowedToEditRolesException(role: str) -> PySyftException:
    return PySyftException(
        message=f"{role} is not allowed to edit roles", roles=[ServiceRole.ADMIN]
    )


def RoleNotAllowedToEditSpecificRolesException(
    ctx_role: str, user_role: str, user_update_role: str
) -> PySyftException:
    if user_update_role is not None:
        return PySyftException(
            message=f"As a {ctx_role}, you are not allowed to edit {user_role} to {user_update_role}",
            roles=[ServiceRole.ADMIN],
        )
    return PySyftException(
        message=f"As a {ctx_role}, you are not allowed to edit {user_role}",
        roles=[ServiceRole.ADMIN],
    )


RoleNotFoundException = PySyftException(
    message="Role not found", roles=[ServiceRole.ADMIN]
)


def StashRetrievalException(message: str) -> PySyftException:
    return PySyftException(message=message, roles=[ServiceRole.ADMIN])


def UserWithEmailAlreadyExistsException(email: str) -> PySyftException:
    return PySyftException(
        message=f"User already exists with email: {email}", roles=[ServiceRole.ADMIN]
    )


# def UserDoesNotExistException(uid: str, email: str, err: any) -> PySyftException:
#     # if uid is not None:
#     #     return PySyftException(
#     #         message=f"No user exists for given id: {uid}", roles=[ServiceRole.ADMIN]
#     #     )
#     elif email is not None and err is not None:
#         return PySyftException(
#             message=f"No user exists with {email} and supplied password.",
#             roles=[ServiceRole.ADMIN],
#         )
#     elif email is not None and err is None:
#         return PySyftException(
#             message=f"Failed to retrieve user with {email} with error: {err}",
#             roles=[ServiceRole.ADMIN],
#         )
#     return PySyftException(message="User does not exist", roles=[ServiceRole.ADMIN])
