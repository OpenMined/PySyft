# stdlib
from enum import Enum
from typing import Literal

# syft absolute
from syft.service.context import AuthedServiceContext
from syft.service.user.user_roles import ServiceRole

class SyftErrorCodes(Enum):
    INVALID_WORKER_POOL = "invalid-worker-pool",
    PERMISSION_ERROR = "permission-error",
    USER_CODE_NOT_APPROVED = "user-code-not-approved"

SyftErrorVisibility = tuple[str, ServiceRole, str | None] # message, role, public_message

syft_errors: dict[SyftErrorCodes, SyftErrorVisibility] = {
    SyftErrorCodes.INVALID_WORKER_POOL: (
        "You tried to run a syft function attached to a worker pool in blocking mode,"
        " which is currently not supported. Run your function with `blocking=False` to run"
        " as a job on your worker poolInvalid worker",
        ServiceRole.ADMIN,
        None
    ),
    SyftErrorCodes.PERMISSION_ERROR: (
        "",
        ServiceRole.ADMIN,
        "You do not have permission to perform this operation.",
    ),
    SyftErrorCodes.USER_CODE_NOT_APPROVED: (
        "Your code has not been approved yet.",
        ServiceRole.DATA_SCIENTIST,
        None
    )
}


class SyftPublicException(Exception):
    def __init__(
        self,
        /,
        context: AuthedServiceContext,
        message: str,
        code: SyftErrorCodes | None = None,
        public_message: str | None = None,
        min_visible_role: ServiceRole = ServiceRole.NONE,
        *args,
        **kwargs,
    ) -> None:
        is_role_type_wrong = type(min_visible_role) is not ServiceRole
        is_not_visible_to_role = context.role.value < min_visible_role.value

        if is_role_type_wrong or is_not_visible_to_role or context.is_admin:
            message = public_message if public_message else "An error occurred. Contact the admin."
        else:
            message = message if not code else f"{message} [code: {code}]"

        super().__init__(message, *args, **kwargs)


class SyftException(Exception):
    def __init__(
        self,
        message: str,
        /,
        code: SyftErrorCodes | None = None,
        public_message: str | None = None,
        min_visible_role: ServiceRole | None = ServiceRole.ADMIN,
        *args,
        **kwargs,
    ) -> None:
        self.message = message
        self.code = code
        self.public_message = public_message
        self.min_visible_role = min_visible_role
        super().__init__(message, *args, **kwargs)


class WorkerError(SyftException):
    def __init__(
        self,
        /,
        code: SyftErrorCodes,
        message: str | None = None,
        *args,
        **kwargs,
    ) -> None:
        original_message, min_role, public_message = syft_errors[code]

        if message is None:
            message = original_message

        super().__init__(
            message=message,
            code=code,
            public_message=public_message,
            min_visible_role=min_role,
            *args,
            **kwargs,
        )

class PermissionError(SyftException):
    def __init__(
        self,
        /,
        message: str,
        public_message: str | None = None,
        *args,
        **kwargs,
    ) -> None:
        _, min_role, _= syft_errors["permission-error"]

        super().__init__(
            message,
            code="permission-error",
            public_message=public_message,
            min_visible_role=min_role,
            *args,
            **kwargs,
        )

class UserCodeError(SyftException):
    def __init__(
        self,
        /,
        message: str,
        code: SyftErrorCodes,
        public_message: str | None = None,
        *args,
        **kwargs,
    ) -> None:
        message, min_role, public_message = syft_errors[code]
        super().__init__(
            message,
            code=code,
            public_message=public_message,
            min_visible_role=min_role,
            *args,
            **kwargs,
        )

class InvalidWorkerPoolException(WorkerException):
    def __init__(self, message: str | None = None, *args, **kwargs):
        super().__init__(message=message, code="invalid-worker-pool", *args, **kwargs)

