"""Errors raised by the connections of every backend."""


class SyftConnectionError(Exception):
    """Base exception for connection errors."""


class ConnectionUnavailableError(SyftConnectionError):
    """The backend could not be reached."""
