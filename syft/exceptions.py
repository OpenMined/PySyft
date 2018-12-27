"""Specific Pysyft exceptions."""


class WorkerNotFoundException(Exception):
    """Raised when a non-existent worker is requested."""

    pass


class CompressionNotFoundException(Exception):
    """Raised when a non existent compression/decompression scheme is requested."""

    pass
