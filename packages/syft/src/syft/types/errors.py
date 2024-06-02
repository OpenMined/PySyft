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
        public_message: str | None = None,
        private: bool | None = True,
        min_visible_role: ServiceRole = ServiceRole.ADMIN,
    ) -> None:
        self.message = message
        self.code = code
        self.private = private
        self.public_message = public_message
        self.min_visible_role = min_visible_role
        super().__init__(message)

    @property
    def public(self):
        if self.private:
            return self.public_message or "An error occurred. Contact your admininstrator for more information."
        return self.public_message or self.message

    def can_see_message(self, role: ServiceRole) -> bool:
        return role.value >= self.min_visible_role.value

