# relative
from ...types.errors import SyftException


class UserError(SyftException):
    public_message = "UserError. Please contact the admin."


class UserCreateError(UserError):
    public_message = "Failed to create user."


class UserDeleteError(UserError):
    public_message = "Failed to delete user."


class UserUpdateError(UserError):
    public_message = "Failed to update user."


class UserPasswordMismatchError(UserError):
    public_message = "Passwords do not match!"


class UserInvalidEmailError(UserError):
    public_message = "Invalid email address."


class UserSearchBadParamsError(UserError): ...


# public_message = (
#     f"Invalid Search parameters. Allowed params: "
#     f"{list(UserSearch.model_fields.keys())}"
# )


class UserPermissionError(UserError):
    public_message = "You are not permitted to perform this action."


class UserExchangeCredentials(UserError):
    public_message = "Invalid credential exchange. Please contact the admin."


class UserEnclaveAdminLoginError(UserError):
    public_message = (
        "Admins are not allowed to login to Enclaves.\n"
        "Kindly register a new data scientist account via `client.register`."
    )
