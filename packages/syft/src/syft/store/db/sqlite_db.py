# stdlib
import threading

# third party
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.orm import sessionmaker

# relative
from ...types.uid import UID
from .models import Base
from .utils import dumps
from .utils import loads


class SQLiteDBManager:
    def __init__(self, server_uid: UID) -> None:
        self.server_uid = server_uid
        self.path = f"sqlite:////tmp/{str(server_uid)}.db"
        self.engine = create_engine(
            self.path, json_serializer=dumps, json_deserializer=loads
        )
        print(f"Connecting to {self.path}")
        self.Session = sessionmaker(bind=self.engine)
        self.thread_local = threading.local()

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
