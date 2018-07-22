# -*- coding: utf-8 -*-
"""
The class hierachy for exceptions is::

    Error
     +-- VersionMismatch
     +-- EncoderError
     |    +-- EncoderMissingError
     |    +-- EncodingError
     |    +-- DecodingError
     +-- CommunicationError
          +-- ProtocolError
          +-- StatusError
          +-- ErrorResponse
          +-- ConnectionError
          +-- TimeoutError

"""


class Error(Exception):
    """Base class for all exceptions in this module."""
    pass


class VersionMismatch(Error):
    """Raised when daemon version is not supported by this client version."""

    def __init__(self, current, minimum, maximum):
        self.current = current
        self.minimum = minimum
        self.maximum = maximum

        msg = "Unsupported daemon version '{}' (not in range: {} – {})".format(
            current, minimum, maximum
        )
        Error.__init__(self, msg)


###############
# encoding.py #
###############
class EncoderError(Error):
    """Base class for all encoding and decoding related errors."""

    def __init__(self, message, encoder_name):
        self.encoder_name = encoder_name

        Error.__init__(self, message)


class EncoderMissingError(EncoderError):
    """Raised when a requested encoder class does not actually exist."""

    def __init__(self, encoder_name):
        msg = "Unknown encoder: '{}'".format(encoder_name)
        EncoderError.__init__(self, msg, encoder_name)


class EncodingError(EncoderError):
    """Raised when encoding a Python object into a byte string has failed
    due to some problem with the input data."""

    def __init__(self, encoder_name, original):
        self.original = original

        msg = "Object encoding error: {}".format(original)
        EncoderError.__init__(self, msg, encoder_name)


class DecodingError(EncoderError):
    """Raised when decoding a byte string to a Python object has failed due to
    some problem with the input data."""

    def __init__(self, encoder_name, original):
        self.original = original

        msg = "Object decoding error: {}".format(original)
        EncoderError.__init__(self, msg, encoder_name)


###########
# http.py #
###########
class CommunicationError(Error):
    """Base class for all network communication related errors."""

    def __init__(self, original, _message=None):
        self.original = original

        if _message:
            msg = _message
        else:
            msg = "{}: {}".format(original.__class__.__name__, str(original))
        Error.__init__(self, msg)


class ProtocolError(CommunicationError):
    """Raised when parsing the response from the daemon has failed.

    This can most likely occur if the service on the remote end isn't in fact
    an IPFS daemon."""


class StatusError(CommunicationError):
    """Raised when the daemon responds with an error to our request."""


class ErrorResponse(StatusError):
    """Raised when the daemon has responded with an error message because the
    requested operation could not be carried out."""

    def __init__(self, message, original):
        StatusError.__init__(self, original, message)


class ConnectionError(CommunicationError):
    """Raised when connecting to the service has failed on the socket layer."""


class TimeoutError(CommunicationError):
    """Raised when the daemon didn't respond in time."""
