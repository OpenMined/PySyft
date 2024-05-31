# stdlib
from typing import Literal

from ..service.user.user_roles import ServiceRole

SyftErrorCodes = Literal[
    'not-found',
    'not-permitted',
    'database-error',
    'serde-serialization-error',
    'serde-deserialization-error',
    'stash-error',
    'worker-invalid-pool',
    'usercode-not-approved',
    'usercode-bad-input-policy',
    'usercode-bad-output-policy'
]

class SyftError(Exception):
    __match_args__ = ('code')

    def __init__(
        self,
        message: str,
        code: SyftErrorCodes,
        min_visible_role: ServiceRole = ServiceRole.ADMIN,
        public_message: str | None = None,
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

