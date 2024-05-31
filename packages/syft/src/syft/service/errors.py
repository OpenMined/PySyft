# stdlib
from typing import Literal

# syft absolute
from syft.service.user.user_roles import ServiceRole

SyftErrorCodes = Literal[
    'not-found',
    'not-permitted',
    'stash-error',
    'worker',
    'worker-invalid-pool',
    'usercode-not-approved'
]

SyftErrorVisibility = tuple[str, ServiceRole, str | None] # message, role, public_message

syft_errors: dict[SyftErrorCodes, SyftErrorVisibility] = {
    'not-permitted': (
        "You do not have permission to perform this operation.",
        ServiceRole.ADMIN,
        None
    ),
    'not-found': (
        "The requested object was not found.",
        ServiceRole.DATA_SCIENTIST,
        None
    ),
    'worker-invalid-pool': (
        (
            "You tried to run a syft function attached to a worker pool in blocking mode,"
            " which is currently not supported. Run your function with `blocking=False` to run"
            " as a job on your worker pool"
        ),
        ServiceRole.ADMIN,
        None
    ),
    'usercode-not-approved': (
        "Your code has not been approved yet.",
        ServiceRole.DATA_SCIENTIST,
        None
    ),

}

class SyftError(Exception):
    def __init__(
        self,
        message: str | None = None,
        code: SyftErrorCodes | None = None,
        public_message: str | None = None,
        min_visible_role: ServiceRole = ServiceRole.ADMIN,
    ) -> None:
        self.message = message
        self.code = code
        self.public_message = public_message
        self.min_visible_role = min_visible_role
        super().__init__(message)

    @property
    def public(self):
        return self.public_message if self.public_message else self.message

    def can_see_message(self, role: ServiceRole) -> bool:
        return role.value >= self.min_visible_role.value

