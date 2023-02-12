# stdlib
from threading import Lock
from typing import Dict

# third party
from pymongo.collection import Collection as MongoCollection
from pymongo.database import Database as MongoDatabase
from pymongo.errors import ConnectionFailure
from pymongo.mongo_client import MongoClient as PyMongoClient
from typing_extensions import Self

# relative
from ...common.serde.serializable import serializable
from .document_store import ClientConfig
from .document_store import PartitionSettings


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

    def __init__(self, config: ClientConfig) -> None:
        self.connect(config=config)

    def connect(self, config: ClientConfig):
        try:
            self.client = PyMongoClient(
                host=config.hostname,
                port=config.port,
                username=config.username,
                password=config.password,
                tls=config.tls,
                uuidRepresentation="standard",
            )
        except ConnectionFailure as e:
            print("Failed to connect to mongo store !!!", e)
            raise e

    def with_db(self, db_name: str) -> MongoDatabase:
        return self.client[db_name]

    def with_collection(
        self, collection_settings: PartitionSettings
    ) -> MongoCollection:
        db = self.with_db(db_name=collection_settings.db_name)
        return db.get_collection(name=collection_settings.name)

    @staticmethod
    def from_config(config: ClientConfig) -> Self:
        return MongoClient(config=config)
