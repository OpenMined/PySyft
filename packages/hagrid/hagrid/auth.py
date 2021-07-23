# stdlib
from typing import Optional


class AuthCredentials:
    def __init__(
        self,
        username: str,
        key_path: Optional[str] = None,
        password: Optional[str] = None,
    ) -> None:
        self.username = username
        self.key_path = key_path
        self.password = password

    @property
    def uses_key(self) -> bool:
        if self.username and self.key_path:
            return True

    @property
    def uses_password(self) -> bool:
        if self.username and self.password:
            return True

    @property
    def valid(self) -> bool:
        return self.uses_key or self.uses_password
