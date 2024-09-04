# stdlib
from pathlib import Path
import tempfile
import threading
import uuid

# third party
from pydantic import BaseModel
from pydantic import Field
import sqlalchemy as sa
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.orm import sessionmaker

# relative
from ...server.credentials import SyftVerifyKey
from ...types.uid import UID
from .models import Base
from .utils import dumps
from .utils import loads


class DBConfig(BaseModel):
    reset: bool = False

    @property
    def connection_string(self) -> str:
        raise NotImplementedError("Subclasses must implement this method.")


class SQLiteDBConfig(DBConfig):
    filename: str = Field(default_factory=lambda: f"{uuid.uuid4()}.db")
    path: Path = Field(default_factory=lambda: Path(tempfile.gettempdir()))

    @property
    def connection_string(self) -> str:
        filepath = self.path / self.filename
        return f"sqlite:///{filepath.resolve()}"


class PostgresDBConfig(DBConfig):
    host: str = "localhost"
    port: int = 5432
    user: str = "postgres"
    password: str = "postgres"
    database: str = "postgres"

    @property
    def connection_string(self) -> str:
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"


class DBManager:
    def __init__(
        self,
        config: DBConfig,
        server_uid: UID,
        root_verify_key: SyftVerifyKey,
    ) -> None:
        self.config = config
        self.server_uid = server_uid
        self.root_verify_key = root_verify_key


class SQLiteDBManager(DBManager):
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
            config.connection_string, json_serializer=dumps, json_deserializer=loads
        )
        print(f"Connecting to {config.connection_string}")
        self.Session = sessionmaker(bind=self.engine)

        # TODO use AuthedServiceContext for session management instead of threading.local
        self.thread_local = threading.local()

        self.update_settings()

    def update_settings(self) -> None:
        connection = self.engine.connect()

        if self.engine.dialect.name == "sqlite":
            connection.execute(sa.text("PRAGMA journal_mode = WAL"))
            connection.execute(sa.text("PRAGMA busy_timeout = 5000"))
            connection.execute(sa.text("PRAGMA temp_store = 2"))
            connection.execute(sa.text("PRAGMA synchronous = 1"))

    def init_tables(self) -> None:
        if self.config.reset:
            # drop all tables that we know about
            Base.metadata.drop_all(bind=self.engine)
        Base.metadata.create_all(self.engine)

    def reset(self) -> None:
        Base.metadata.drop_all(bind=self.engine)
        Base.metadata.create_all(self.engine)

    # TODO remove
    def get_session_threading_local(self) -> Session:
        if not hasattr(self.thread_local, "session"):
            self.thread_local.session = self.Session()
        return self.thread_local.session

    # TODO remove
    @property
    def session(self) -> Session:
        return self.get_session_threading_local()
