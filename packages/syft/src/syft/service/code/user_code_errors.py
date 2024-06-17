# relative
from ...types.errors import SyftException


class UserCodeError(SyftException): ...


class UserCodeNotApprovedError(UserCodeError):
    public_message = "The code has not been approved yet. Contact the administrator."


class UserCodeInvalidOutputPolicy(UserCodeError):
    public_message = "The output policy is invalid."


class UserCodeInvalidInputPolicy(UserCodeError):
    public_message = "The input policy is invalid."
