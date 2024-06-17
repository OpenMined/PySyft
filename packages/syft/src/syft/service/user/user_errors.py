# relative
from ...types.errors import SyftException


class UserError(SyftException):
    public_message = (
        "An error occurred with the user. Contact the admin for more information."
    )


class UserAlreadyExistsError(UserError): ...
