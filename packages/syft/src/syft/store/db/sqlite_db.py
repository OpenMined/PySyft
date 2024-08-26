# stdlib
from pathlib import Path
import tempfile
import threading

# third party
from pydantic import BaseModel
from pydantic import Field
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
    pass


class SQLiteDBConfig(DBConfig):
    filename: str = "jsondb.sqlite"
    path: Path = Field(default_factory=tempfile.gettempdir)


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

        self.filepath = config.path / config.filename
        self.path = f"sqlite:///{self.filepath.resolve()}"
        self.engine = create_engine(
            self.path, json_serializer=dumps, json_deserializer=loads
        )
        self.Session = sessionmaker(bind=self.engine)
        # TODO use AuthedServiceContext for session management instead of threading.local
        self.thread_local = threading.local()

    def init_tables(self) -> None:
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
