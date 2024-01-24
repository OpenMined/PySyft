class OblvProxyConnectPCRError(Exception):
    def __init__(self, message: str = "") -> None:
        if not message:
            message = "Failed to connect to enclave. Unauthorized deployment provided."
        super().__init__(message)


class OblvEnclaveUnAuthorizedError(Exception):
    def __init__(self, message: str = "") -> None:
        if not message:
            message = "Domain unauthorized to perform this action in enclave"
        super().__init__(message)


class OblvEnclaveError(Exception):
    def __init__(self, message: str = "") -> None:
        if not message:
            message = "Failed to connect to the enclave"
        super().__init__(message)


class OblvError(Exception):
    def __init__(self, message: str = "") -> None:
        super().__init__(message)


class OblvUnAuthorizedError(Exception):
    def __init__(self, message: str = "") -> None:
        if not message:
            message = "User unauthorized to perform this action in enclave"
        super().__init__(message)


class OblvKeyAlreadyExistsError(Exception):
    def __init__(self, message: str = "") -> None:
        if not message:
            message = "Currently each domain node could have only one oblv public/private key pair"
        super().__init__(message)


class OblvLocalEnclaveError(Exception):
    def __init__(self, message: str = "") -> None:
        if not message:
            message = (
                "Failed to connect to locally deployed FastAPI based enclave services."
            )
        super().__init__(message)


class OblvKeyNotFoundError(Exception):
    def __init__(self, message: str = "") -> None:
        if not message:
            message = "Oblivious public key not found. Kindly request admin to create a new one"
        super().__init__(message)
