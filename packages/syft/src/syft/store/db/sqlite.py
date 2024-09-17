# stdlib
from pathlib import Path
import tempfile
import uuid

# third party
from pydantic import Field
import sqlalchemy as sa

# relative
from ...serde.serializable import serializable
from ...server.credentials import SyftSigningKey
from ...server.credentials import SyftVerifyKey
from ...types.uid import UID
from .db import DBConfig
from .db import DBManager


@serializable(canonical_name="SQLiteDBConfig", version=1)
class SQLiteDBConfig(DBConfig):
    filename: str = Field(default_factory=lambda: f"{uuid.uuid4()}.db")
    path: Path = Field(default_factory=lambda: Path(tempfile.gettempdir()))

    @property
    def connection_string(self) -> str:
        """
        NOTE in-memory sqlite is not shared between connections, so:
        - using 2 workers (high/low) will not share a db
        - re-using a connection (e.g. for a Job worker) will not share a db
        """
        if self.path == Path("."):
            # Use in-memory database, only for unittests
            return "sqlite://"
        filepath = self.path / self.filename
        return f"sqlite:///{filepath.resolve()}"


class SQLiteDBManager(DBManager[SQLiteDBConfig]):
    def update_settings(self) -> None:
        connection = self.engine.connect()
        connection.execute(sa.text("PRAGMA journal_mode = WAL"))
        connection.execute(sa.text("PRAGMA busy_timeout = 5000"))
        connection.execute(sa.text("PRAGMA temp_store = 2"))
        connection.execute(sa.text("PRAGMA synchronous = 1"))

    @classmethod
    def random(
        cls,
        *,
        config: SQLiteDBConfig | None = None,
        server_uid: UID | None = None,
        root_verify_key: SyftVerifyKey | None = None,
    ) -> "SQLiteDBManager":
        root_verify_key = root_verify_key or SyftSigningKey.generate().verify_key
        server_uid = server_uid or UID()
        config = config or SQLiteDBConfig()
        return SQLiteDBManager(
            config=config,
            server_uid=server_uid,
            root_verify_key=root_verify_key,
        )
