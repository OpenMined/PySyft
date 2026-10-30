from pathlib import Path
from syft.sync.connections.base_connection import ConnectionConfig


class GdriveConnectionConfig(ConnectionConfig):
    email: str
    token_path: Path | None = None

    @classmethod
    def from_token_path(cls, email: str, token_path: Path) -> "GdriveConnectionConfig":
        return cls(email=email, token_path=token_path)
