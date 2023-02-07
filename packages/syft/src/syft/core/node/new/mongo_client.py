# stdlib
from threading import Lock
from typing import Dict
from typing import Optional

# third party
from pymongo.collection import Collection as _MongoCollection
from pymongo.database import Database as MongoDatabase
from pymongo.errors import ConnectionFailure
from pymongo.mongo_client import MongoClient as PyMongoClient
from typing_extensions import Self

# relative
from ...common.serde.serializable import serializable
from .base import SyftBaseModel
from .document_store import CollectionSettings


class ClientSettings(SyftBaseModel):
    hostname: str
    port: int
    username: str
    password: str
    tls: Optional[bool] = False


@serializable(recursive_serde=True)
class SingletonMeta(type):
    """This is a thread-safe implementation of Singleton."""

    _instances: Dict = {}

    _lock: Lock = Lock()

    def __call__(cls, *args, **kwargs):
        """Possible changes to the value of the `__init__` argument do not affect the returned instance."""
        with cls._lock:
            if cls not in cls._instances:
                instance = super().__call__(*args, **kwargs)
                cls._instances[cls] = instance
        return cls._instances[cls]


@serializable(recursive_serde=True)
class MongoClient(metaclass=SingletonMeta):
    client: PyMongoClient = None

    def __init__(self, settings: ClientSettings) -> None:
        self.connect(settings=settings)

    def connect(self, settings: ClientSettings):
        try:
            self.client = PyMongoClient(
                host=settings.hostname,
                port=settings.port,
                username=settings.username,
                password=settings.password,
                tls=settings.tls,
                uuidRepresentation="standard",
            )
        except ConnectionFailure as e:
            print("Failed to connect to mongo store !!!", e)
            raise e

    def with_db(self, db_name: str) -> MongoDatabase:
        return self.client[db_name]

    def with_collection(
        self, collection_settings: CollectionSettings
    ) -> _MongoCollection:
        db = self.with_db(db_name=collection_settings.db_name)
        return db.get_collection(name=collection_settings.name)

    @staticmethod
    def from_settings(settings: ClientSettings) -> Self:
        return MongoClient(settings=settings)


# 🟡 TODO 30: import from .env
MONGO_HOST = "localhost"
MONGO_PORT = "49165"
MONGO_USERNAME = "root"
MONGO_PASSWORD = "example"

MongoClientSettings = ClientSettings(
    hostname=MONGO_HOST,
    port=MONGO_PORT,
    username=MONGO_USERNAME,
    password=MONGO_PASSWORD,
)
