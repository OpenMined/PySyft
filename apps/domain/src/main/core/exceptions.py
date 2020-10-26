"""Specific PyGrid exceptions."""


class PyGridError(Exception):
    def __init__(self, message):
        super().__init__(message)


class AuthorizationError(PyGridError):
    def __init__(self):
        message = "User is not authorized for this operation!"
        super().__init__(message)


class RoleNotFoundError(PyGridError):
    def __init__(self):
        message = "Role ID not found!"
        super().__init__(message)


class UserNotFoundError(PyGridError):
    def __init__(self):
        message = "User ID not found!"
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
    def __init__(self):
        message = "Missing request key!"
        super().__init__(message)
