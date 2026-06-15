from pathlib import Path
from typing import ClassVar, Type
from syft_client.sync.connections.base_connection import ConnectionConfig
from syft_client.sync.connections.drive.gdrive_transport import GDriveConnection


class GdriveConnectionConfig(ConnectionConfig):
    connection_type: ClassVar[Type["GDriveConnection"]] = GDriveConnection
    email: str
    token_path: Path | None = None

    @classmethod
    def from_token_path(cls, email: str, token_path: Path) -> "GdriveConnectionConfig":
        return cls(email=email, token_path=token_path)
