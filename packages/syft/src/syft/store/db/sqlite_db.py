# stdlib
from pathlib import Path
import tempfile
import uuid

# third party
from pydantic import BaseModel
from pydantic import Field
import sqlalchemy as sa
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# relative
from ...serde.serializable import serializable
from ...server.credentials import SyftSigningKey
from ...server.credentials import SyftVerifyKey
from ...types.uid import UID
from .schema import Base


@serializable(canonical_name="DBConfig", version=1)
class DBConfig(BaseModel):
    reset: bool = False

    @property
    def connection_string(self) -> str:
        raise NotImplementedError("Subclasses must implement this method.")


@serializable(canonical_name="SQLiteDBConfig", version=1)
class SQLiteDBConfig(DBConfig):
    filename: str = Field(default_factory=lambda: f"{uuid.uuid4()}.db")
    path: Path = Field(default_factory=lambda: Path(tempfile.gettempdir()))

    @property
    def connection_string(self) -> str:
        filepath = self.path / self.filename
        return f"sqlite:///{filepath.resolve()}"


@serializable(canonical_name="PostgresDBConfig", version=1)
class PostgresDBConfig(DBConfig):
    host: str = "postgres"
    port: int = 5432
    user: str = "syft_postgres"
    password: str = "example"
    database: str = "syftdb_postgres"

    @property
    def connection_string(self) -> str:
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"


class DBManager:
    def __init__(
        self,
        config: SQLiteDBConfig,
        server_uid: UID,
        root_verify_key: SyftVerifyKey,
    ) -> None:
        self.config = config
        self.root_verify_key = root_verify_key
        self.server_uid = server_uid
        self.engine = create_engine(
            config.connection_string,
            # json_serializer=dumps,
            # json_deserializer=loads,
        )
        print(f"Connecting to {config.connection_string}")
        self.sessionmaker = sessionmaker(bind=self.engine)

        self.update_settings()

    def update_settings(self) -> None:
        pass

    def init_tables(self) -> None:
        pass

    def reset(self) -> None:
        pass


class SQLiteDBManager(DBManager):
    def update_settings(self) -> None:
        # TODO split SQLite / PostgresDBManager
        connection = self.engine.connect()

        if self.engine.dialect.name == "sqlite":
            connection.execute(sa.text("PRAGMA journal_mode = WAL"))
            connection.execute(sa.text("PRAGMA busy_timeout = 5000"))
            # TODO check
            connection.execute(sa.text("PRAGMA temp_store = 2"))
            connection.execute(sa.text("PRAGMA synchronous = 1"))

    def init_tables(self) -> None:
        if self.config.reset:
            # drop all tables that we know about
            Base.metadata.drop_all(bind=self.engine)
            self.config.reset = False
        Base.metadata.create_all(self.engine)

    def reset(self) -> None:
        Base.metadata.drop_all(bind=self.engine)
        Base.metadata.create_all(self.engine)

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
