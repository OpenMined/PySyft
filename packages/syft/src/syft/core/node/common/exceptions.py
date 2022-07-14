"""Specific PyGrid exceptions."""


class PyGridError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class PermissionsNotDefined(PyGridError):
    def __init__(self, message: str = "") -> None:
        if not message:
            message = "This message hasn't permissions defined, Please set up message permissions."
        super().__init__(message)


class BadPayloadException(PyGridError):
    def __init__(self, message: str = "") -> None:
        if not message:
            message = (
                "Missing mandatory message fields or wrong message format and type."
            )
        super().__init__(message)


class AuthorizationError(PyGridError):
    def __init__(self, message: str = "") -> None:
        if not message:
            message = "User is not authorized for this operation!"
        super().__init__(message)


class OwnerAlreadyExistsError(PyGridError):
    def __init__(self, message: str = "") -> None:
        if not message:
            message = "This PyGrid domain already has an owner!"
        super().__init__(message)


class RoleNotFoundError(PyGridError):
    def __init__(self) -> None:
        message = "Role ID not found!"
        super().__init__(message)


class ModelNotFoundError(PyGridError):
    def __init__(self) -> None:
        message = "Model ID not found!"
        super().__init__(message)


class CycleNotFoundError(PyGridError):
    def __init__(self) -> None:
        message = "Cycle not found!"
        super().__init__(message)


class PlanNotFoundError(PyGridError):
    def __init__(self) -> None:
        message = "Plan ID not found!"
        super().__init__(message)


class ProcessNotFoundError(PyGridError):
    def __init__(self) -> None:
        message = "Not found any process related with this cycle and worker ID."
        super().__init__(message)


class PlanInvalidError(PyGridError):
    def __init__(self) -> None:
        message = "Plan is not valid"
        super().__init__(message)


class PlanTranslationError(PyGridError):
    def __init__(self) -> None:
        message = "Failed to translate a Plan"
        super().__init__(message)


class WorkerNotFoundError(PyGridError):
    def __init__(self) -> None:
        message = "Worker ID not found!"
        super().__init__(message)


class ProtocolNotFoundError(PyGridError):
    def __init__(self) -> None:
        message = "Protocol ID not found!"
        super().__init__(message)


class FLProcessConflict(PyGridError):
    def __init__(self) -> None:
        message = "FL Process already exists."
        super().__init__(message)


class MaxCycleLimitExceededError(PyGridError):
    def __init__(self) -> None:
        message = "There are no cycles remaining"
        super().__init__(message)


class UserNotFoundError(PyGridError):
    def __init__(self) -> None:
        message = "User not found!"
        super().__init__(message)


class EnvironmentNotFoundError(PyGridError):
    def __init__(self) -> None:
        message = "Environment not found!"
        super().__init__(message)


class SetupNotFoundError(PyGridError):
    def __init__(self) -> None:
        message = "Setup not found!"
        super().__init__(message)


class GroupNotFoundError(PyGridError):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class InvalidRequestKeyError(PyGridError):
    def __init__(self) -> None:
        message = "Invalid request key!"
        super().__init__(message)


class InvalidCredentialsError(PyGridError):
    def __init__(self) -> None:
        message = "Invalid credentials!"
        super().__init__(message)


class MissingRequestKeyError(PyGridError):
    def __init__(self, message: str = "") -> None:
        if not message:
            message = "Missing request key!"
        super().__init__(message)


class AssociationRequestError(PyGridError):
    def __init__(self) -> None:
        message = "Association Request ID not found!"
        super().__init__(message)


class AssociationError(PyGridError):
    def __init__(self) -> None:
        message = "Association ID not found!"
        super().__init__(message)


class RequestError(PyGridError):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class DatasetNotFoundError(PyGridError):
    def __init__(self) -> None:
        message = "Dataset ID not found!"
        super().__init__(message)


class InvalidParameterValueError(PyGridError):
    def __init__(self, message: str = "") -> None:
        if not message:
            message = "Passed paramater value not valid!"
        super().__init__(message)


class AppInSleepyMode(PyGridError):
    def __init__(self, message: str = "") -> None:
        if not message:
            message = (
                "This app is in sleep mode. Please undergo the initial setup first"
            )
        super().__init__(message)


class DatasetUploadError(Exception):
    def __init__(self, message: str = "") -> None:
        if not message:
            message = "Failed to upload/send data to blob store."
        super().__init__(message)


class DatasetDownloadError(Exception):
    def __init__(self, message: str = "") -> None:
        if not message:
            message = "Failed to retrieve data from blob store."
        super().__init__(message)


class InvalidNodeCredentials(Exception):
    def __init__(self, message: str = "") -> None:
        if not message:
            message = "Invalid Credentials, verify_key does not match"
        super().__init__(message)
