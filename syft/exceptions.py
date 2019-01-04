"""Specific Pysyft exceptions."""


class RemoteTensorFoundError(BaseException):
    """Exception raised for errors in the input.

    Attributes:
        expression -- input expression in which the error occurred
        message -- explanation of the error
    """

    def __init__(self, pointer):
        self.pointer = pointer


class WorkerNotFoundException(Exception):
    """Raised when a non-existent worker is requested."""

    pass


class CompressionNotFoundException(Exception):
    """Raised when a non existent compression/decompression scheme is requested."""

    pass
