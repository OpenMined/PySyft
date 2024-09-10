# relative
from ..types.errors import SyftException


class SyftClientError(SyftException):
    public_message = "Unknown client error."


class SyftAPINotFoundException(SyftClientError): ...
