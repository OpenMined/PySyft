"""Specific PyGrid exceptions."""


class PyGridError(Exception):
    def __init__(self, message):
        super().__init__(message)


class AuthorizationError(PyGridError):
    def __init__(self, message=""):
        if not message:
            message = "User is not authorized for this operation!"
        super().__init__(message)


class RoleNotFoundError(PyGridError):
    def __init__(self):
        message = "Role ID not found!"
        super().__init__(message)


class UserNotFoundError(PyGridError):
    def __init__(self):
        message = "User not found!"
        super().__init__(message)


class EnvironmentNotFoundError(PyGridError):
    def __init__(self):
        message = "Environment not found!"
        super().__init__(message)


class SetupNotFoundError(PyGridError):
    def __init__(self):
        message = "Setup not found!"
        super().__init__(message)


class GroupNotFoundError(PyGridError):
    def __init__(self):
        message = "Group ID not found!"
        super().__init__(message)


class InvalidRequestKeyError(PyGridError):
    def __init__(self):
        message = "Invalid request key!"
        super().__init__(message)


class InvalidCredentialsError(PyGridError):
    def __init__(self):
        message = "Invalid credentials!"
        super().__init__(message)


class MissingRequestKeyError(PyGridError):
    def __init__(self, message=""):
        if not message:
            message = "Missing request key!"
        super().__init__(message)


class AssociationRequestError(PyGridError):
    def __init__(self):
        message = "Association Request ID not found!"
        super().__init__(message)


class AssociationError(PyGridError):
    def __init__(self):
        message = "Association ID not found!"


class RequestError(PyGridError):
    def __init__(self):
        message = "Request ID not found!"
        super().__init__(message)


class InvalidParameterValueError(PyGridError):
    def __init__(self, message=""):
        if not message:
            message = "Passed paramater value not valid!"
        super().__init__(message)
