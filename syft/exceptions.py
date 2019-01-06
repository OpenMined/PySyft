"""Specific Pysyft exceptions."""


class AsyncOperationNotFinalizedError(RuntimeError):
    """Raised when a result of a async operation is not ready to be accessed."""

    pass


class CompressionNotFoundException(Exception):
    """Raised when a non existent compression/decompression scheme is requested."""

    pass


class ThreadTimeOutError(RuntimeError):
    """Raised when an async operation times out."""

    pass


class WorkerNotFoundException(Exception):
    """Raised when a non-existent worker is requested."""

    pass
