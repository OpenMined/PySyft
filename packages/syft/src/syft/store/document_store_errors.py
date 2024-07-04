# relative
from ..types.errors import SyftException


class NotFoundException(SyftException):
    public_message = "Item not found."


class TooManyItemsFoundException(SyftException):
    public_message = "Too many items found."


class StashException(SyftException):
    public_message = "There was an error retrieving data. Contact your admin."
