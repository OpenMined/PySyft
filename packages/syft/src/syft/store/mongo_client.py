# stdlib
from threading import Lock
from typing import Any
from typing import Dict
from typing import Optional
from typing import Type

# third party
from pymongo.collection import Collection as MongoCollection
from pymongo.database import Database as MongoDatabase
from pymongo.errors import ConnectionFailure
from pymongo.mongo_client import MongoClient as PyMongoClient
from result import Err
from result import Ok
from result import Result

# relative
from ..serde.serializable import serializable
from .document_store import PartitionSettings
from .document_store import StoreClientConfig
from .document_store import StoreConfig
from .mongo_codecs import SYFT_CODEC_OPTIONS


@serializable()
class MongoStoreClientConfig(StoreClientConfig):
    """
    Paramaters:
        `hostname`: optional string
            hostname or IP address or Unix domain socket path of a single mongod or mongos
            instance to connect to, or a mongodb URI, or a list of hostnames (but no more
            than one mongodb URI). If `host` is an IPv6 literal it must be enclosed in '['
            and ']' characters following the RFC2732 URL syntax (e.g. '[::1]' for localhost).
            Multihomed and round robin DNS addresses are **not** supported.
        `port` : optional int
            port number on which to connect
        `directConnection`: bool
            if ``True``, forces this client to connect directly to the specified MongoDB host
            as a standalone. If ``false``, the client connects to the entire replica set of which
            the given MongoDB host(s) is a part. If this is ``True`` and a mongodb+srv:// URI
            or a URI containing multiple seeds is provided, an exception will be raised.
        `maxPoolSize`: int. Default 100
            The maximum allowable number of concurrent connections to each connected server.
            Requests to a server will block if there are `maxPoolSize` outstanding connections
            to the requested server. Defaults to 100. Can be either 0 or None, in which case
            there is no limit on the number of concurrent connections.
        `minPoolSize` : int. Default 0
            The minimum required number of concurrent connections that the pool will maintain
            to each connected server. Default is 0.
        `maxIdleTimeMS`: int
            The maximum number of milliseconds that a connection can remain idle in the pool
            before being removed and replaced. Defaults to `None` (no limit).
        `appname`: string
            The name of the application that created this MongoClient instance. The server will
            log this value upon establishing each connection. It is also recorded in the slow
            query log and profile collections.
        `maxConnecting`: optional int
            The maximum number of connections that each pool can establish concurrently.
            Defaults to `2`.
        `timeoutMS`: (integer or None)
            Controls how long (in milliseconds) the driver will wait when executing an operation
            (including retry attempts) before raising a timeout error. ``0`` or ``None`` means
            no timeout.
        `socketTimeoutMS`: (integer or None)
            Controls how long (in milliseconds) the driver will wait for a response after sending
            an ordinary (non-monitoring) database operation before concluding that a network error
            has occurred. ``0`` or ``None`` means no timeout. Defaults to ``None`` (no timeout).
        `connectTimeoutMS`: (integer or None)
            Controls how long (in milliseconds) the driver will wait during server monitoring when
            connecting a new socket to a server before concluding the server is unavailable.
            ``0`` or ``None`` means no timeout. Defaults to ``20000`` (20 seconds).
        `serverSelectionTimeoutMS`: (integer)
            Controls how long (in milliseconds) the driver will wait to find an available, appropriate
            server to carry out a database operation; while it is waiting, multiple server monitoring
            operations may be carried out, each controlled by `connectTimeoutMS`.
            Defaults to ``120000`` (120 seconds).
        `waitQueueTimeoutMS`: (integer or None)
            How long (in milliseconds) a thread will wait for a socket from the pool if the pool
            has no free sockets. Defaults to ``None`` (no timeout).
        `heartbeatFrequencyMS`: (optional)
            The number of milliseconds between periodic server checks, or None to accept the default
            frequency of 10 seconds.
        # Auth
        username: str
            Database username
        password: str
            Database pass
        authSource: str
            The database to authenticate on.
            Defaults to the database specified in the URI, if provided, or to “admin”.
        tls: bool
            If True, create the connection to the server using transport layer security.
            Defaults to False.
        # Testing and connection reuse
        client: Optional[PyMongoClient]
            If provided, this client is reused. Default = None

    """

    # Connection
    hostname: Optional[str] = "127.0.0.1"
    port: Optional[int] = None
    directConnection: bool = False
    maxPoolSize: int = 200
    minPoolSize: int = 0
    maxIdleTimeMS: Optional[int] = None
    maxConnecting: int = 3
    timeoutMS: int = 0
    socketTimeoutMS: int = 0
    connectTimeoutMS: int = 20000
    serverSelectionTimeoutMS: int = 120000
    waitQueueTimeoutMS: Optional[int] = None
    heartbeatFrequencyMS: int = 10000
    appname: str = "pysyft"
    # Auth
    username: Optional[str] = None
    password: Optional[str] = None
    authSource: str = "admin"
    tls: Optional[bool] = False
    # Testing and connection reuse
    client: Any = None

    # this allows us to have one connection per `Node` object
    # in the MongoClientCache
    node_obj_python_id: Optional[int] = None


class MongoClientCache:
    __client_cache__: Dict[str, Type["MongoClient"]] = {}
    _lock: Lock = Lock()

    @classmethod
    def from_cache(cls, config: MongoStoreClientConfig) -> Optional[PyMongoClient]:
        return cls.__client_cache__.get(hash(str(config)), None)

    @classmethod
    def set_cache(cls, config: MongoStoreClientConfig, client: PyMongoClient) -> None:
        with cls._lock:
            cls.__client_cache__[hash(str(config))] = client


class MongoClient:
    client: PyMongoClient = None

    def __init__(self, config: MongoStoreClientConfig, cache: bool = True) -> None:
        self.config = config
        if config.client is not None:
            self.client = config.client
        elif cache:
            self.client = MongoClientCache.from_cache(config=config)

        if not cache or self.client is None:
            self.connect(config=config)

    def connect(self, config: MongoStoreClientConfig) -> Result[Ok, Err]:
        self.client = PyMongoClient(
            # Connection
            host=config.hostname,
            port=config.port,
            directConnection=config.directConnection,
            maxPoolSize=config.maxPoolSize,
            minPoolSize=config.minPoolSize,
            maxIdleTimeMS=config.maxIdleTimeMS,
            maxConnecting=config.maxConnecting,
            timeoutMS=config.timeoutMS,
            socketTimeoutMS=config.socketTimeoutMS,
            connectTimeoutMS=config.connectTimeoutMS,
            serverSelectionTimeoutMS=config.serverSelectionTimeoutMS,
            waitQueueTimeoutMS=config.waitQueueTimeoutMS,
            heartbeatFrequencyMS=config.heartbeatFrequencyMS,
            appname=config.appname,
            # Auth
            username=config.username,
            password=config.password,
            authSource=config.authSource,
            tls=config.tls,
            uuidRepresentation="standard",
        )
        MongoClientCache.set_cache(config=config, client=self.client)
        try:
            # Check if mongo connection is still up
            self.client.admin.command("ping")
        except ConnectionFailure as e:
            self.client = None
            return Err(str(e))

        return Ok()

    def with_db(self, db_name: str) -> Result[MongoDatabase, Err]:
        try:
            return Ok(self.client[db_name])
        except BaseException as e:
            return Err(str(e))

    def with_collection(
        self,
        collection_settings: PartitionSettings,
        store_config: StoreConfig,
        collection_name: Optional[str] = None,
    ) -> Result[MongoCollection, Err]:
        res = self.with_db(db_name=store_config.db_name)
        if res.is_err():
            return res
        db = res.ok()

        try:
            collection_name = (
                collection_name
                if collection_name is not None
                else collection_settings.name
            )
            collection = db.get_collection(
                name=collection_name, codec_options=SYFT_CODEC_OPTIONS
            )
        except BaseException as e:
            return Err(str(e))

        return Ok(collection)

    def with_collection_permissions(
        self, collection_settings: PartitionSettings, store_config: StoreConfig
    ) -> Result[MongoCollection, Err]:
        """
        For each collection, create a corresponding collection
        that store the permissions to the data in that collection
        """
        res = self.with_db(db_name=store_config.db_name)
        if res.is_err():
            return res
        db = res.ok()

        try:
            collection_permissions_name: str = collection_settings.name + "_permissions"
            collection_permissions = db.get_collection(
                name=collection_permissions_name, codec_options=SYFT_CODEC_OPTIONS
            )
        except BaseException as e:
            return Err(str(e))

        return Ok(collection_permissions)

    def close(self):
        self.client.close()
        MongoClientCache.__client_cache__.pop(hash(str(self.config)), None)
