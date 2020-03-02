"""Specific PyGrid exceptions."""


class PyGridError(Exception):
    def __init__(self, message):
        super().__init__(message)


class WorkerNotFoundError(PyGridError):
    def __init__(self):
        message = "Worker ID not found!"
        super().__init__(message)


class CycleNotFoundError(PyGridError):
    def __init__(self):
        message = "Cycle not found!"
        super().__init__(message)


class FLProcessNotFoundError(PyGridError):
    def __init__(self):
        message = "Federated Learning Process not found!"
        super().__init__(message)


class ProtocolNotFoundError(PyGridError):
    def __init__(self):
        message = "Protocol ID not found!"
        super().__init__(message)


class PlanNotFoundError(PyGridError):
    def __init__(self):
        message = "Plan ID not found!"
        super().__init__(message)


class ModelNotFoundError(PyGridError):
    def __init__(self):
        message = "Model ID not found!"
        super().__init__(message)


class ProcessFoundError(PyGridError):
    def __init__(self):
        message = "Not found any process related with this cycle and worker ID."
        super().__init__(message)


class ConfigsNotFoundError(PyGridError):
    def __init__(self):
        message = "Config ID not Found"
        super().__init__(message)


class CheckPointNotFound(PyGridError):
    def __init__(self):
        message = "Model's Checkpoint not found!"
        super().__init__(message)


class InvalidRequestKeyError(PyGridError):
    def __init__(self):
        message = "Invalid request key!"
        super().__init__(message)
