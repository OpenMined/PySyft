"""Specific Pysyft exceptions."""


class PureTorchTensorFoundError(BaseException):
    """Exception raised for errors in the input.

        Attributes:
            expression -- input expression in which the error occurred
            message -- explanation of the error
        """

    def __init__(self, tensor):
        self.tensor = tensor


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


class CannotRequestTensorAttribute(Exception):
    """Raised when .get() is called on a pointer which points to an attribute of
    another tensor."""

    pass
