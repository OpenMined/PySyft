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
        return bool(self.username and self.key_path)

    @property
    def uses_password(self) -> bool:
        return bool(self.username and self.password)

    @property
    def valid(self) -> bool:
        return bool(self.uses_key or self.uses_password)
