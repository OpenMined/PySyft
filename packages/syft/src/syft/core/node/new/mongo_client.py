# stdlib
from threading import Lock
from typing import Dict
from typing import Optional
from typing import Type

# third party
from pymongo.collection import Collection as MongoCollection
from pymongo.database import Database as MongoDatabase
from pymongo.errors import ConnectionFailure
from pymongo.mongo_client import MongoClient as PyMongoClient
from typing_extensions import Self

# relative
from ...common.serde.serializable import serializable
from .document_store import PartitionSettings
from .document_store import StoreClientConfig


class MongoClientCache:
    __client_cache__: Dict[str, Type["MongoClient"]] = {}
    _lock: Lock = Lock()

    @classmethod
    def from_cache(cls, config: StoreClientConfig) -> Optional[PyMongoClient]:
        return cls.__client_cache__.get(hash(str(config)), None)

    @classmethod
    def set_cache(cls, config: StoreClientConfig, client: PyMongoClient) -> None:
        with cls._lock:
            cls.__client_cache__[hash(str(config))] = client


@serializable(recursive_serde=True)
class MongoClient:
    client: PyMongoClient = None

    def __init__(self, config: StoreClientConfig, cache: bool = True) -> None:
        if cache:
            self.client = MongoClientCache.from_cache(config=config)
        if not cache or self.client is None:
            self.connect(config=config)

    def connect(self, config: StoreClientConfig):
        self.client = PyMongoClient(
            host=config.hostname,
            port=config.port,
            username=config.username,
            password=config.password,
            tls=config.tls,
            uuidRepresentation="standard",
        )
        MongoClientCache.set_cache(config=config, client=self.client)
        try:
            # Check if mongo connection is still up
            self.client.admin.command("ping")
        except ConnectionFailure as e:
            print("Failed to connect to mongo store server !!!", e)
            raise e

    def with_db(self, db_name: str) -> MongoDatabase:
        return self.client[db_name]

    def with_collection(
        self, collection_settings: PartitionSettings
    ) -> MongoCollection:
        db = self.with_db(db_name=collection_settings.db_name)
        return db.get_collection(name=collection_settings.name)

    @staticmethod
    def from_config(config: StoreClientConfig, cache: bool = True) -> Self:
        return MongoClient(config=config, cache=cache)
