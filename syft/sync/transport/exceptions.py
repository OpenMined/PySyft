"""Errors raised by every transport backend."""


class TransportError(Exception):
    """Base exception for transport errors."""


class TransportUnavailableError(TransportError):
    """The backend could not be reached."""
