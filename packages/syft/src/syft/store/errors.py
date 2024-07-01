from src.syft.types.errors import SyftException


class NotFoundError(SyftException):
    public_message = "Item not found."


class StashError(SyftException):
    public_message = "There was an error retrieving data. Contact your admin."

