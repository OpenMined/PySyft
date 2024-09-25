# relative
from ..types.errors import SyftException


class NotFoundException(SyftException):
    public_message = "Item not found."


class TooManyItemsFoundException(SyftException):
    public_message = "Too many items found."


class StashException(SyftException):
    public_message = "There was an error retrieving data. Contact your admin."


class UniqueConstraintException(StashException):
    public_message = "Another item with the same unique constraint already exists."


class ObjectCRUDPermissionException(SyftException):
    public_message = "You do not have permission to perform this action."


class ObjectExecutionPermissionException(SyftException):
    public_message = "You do not have permission to execute this action."
