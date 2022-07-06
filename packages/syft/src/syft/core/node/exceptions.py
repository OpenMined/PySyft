class PyGridClientException(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class RequestAPIException(Exception):
    def __init__(self, message: str) -> None:
        message = "Something went wrong during this request: " + message
        super().__init__(message)
