# relative
from .errors import SyftException


class NotFoundError(SyftException):
    public_message = "Item not found."


class SyftPermissionError(SyftException):
    public_message = "You don't have permission to perform this action."
