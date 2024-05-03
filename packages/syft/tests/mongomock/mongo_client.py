# stdlib
import itertools
import warnings

# third party
from packaging import version

# relative
from . import ConfigurationError
from . import codec_options as mongomock_codec_options
from . import helpers
from . import read_preferences
from .database import Database
from .store import ServerStore

try:
    # third party
    from pymongo import ReadPreference
    from pymongo.uri_parser import parse_uri
    from pymongo.uri_parser import split_hosts

    _READ_PREFERENCE_PRIMARY = ReadPreference.PRIMARY
except ImportError:
    # relative
    from .helpers import parse_uri
    from .helpers import split_hosts

    _READ_PREFERENCE_PRIMARY = read_preferences.PRIMARY


def _convert_version_to_list(version_str):
    pieces = [int(part) for part in version_str.split(".")]
    return pieces + [0] * (4 - len(pieces))


class MongoClient(object):
    HOST = "localhost"
    PORT = 27017
    _CONNECTION_ID = itertools.count()

    def __init__(
        self,
        host=None,
        port=None,
        document_class=dict,
        tz_aware=False,
        connect=True,
        _store=None,
        read_preference=None,
        uuidRepresentation=None,
        type_registry=None,
        **kwargs,
    ):
        if host:
            self.host = host[0] if isinstance(host, (list, tuple)) else host
        else:
            self.host = self.HOST
        self.port = port or self.PORT

        self._tz_aware = tz_aware
        self._codec_options = mongomock_codec_options.CodecOptions(
            tz_aware=tz_aware,
            uuid_representation=uuidRepresentation,
            type_registry=type_registry,
        )
        self._database_accesses = {}
        self._store = _store or ServerStore()
        self._id = next(self._CONNECTION_ID)
        self._document_class = document_class
        if read_preference is not None:
            read_preferences.ensure_read_preference_type(
                "read_preference", read_preference
            )
        self._read_preference = read_preference or _READ_PREFERENCE_PRIMARY

        dbase = None

        if "://" in self.host:
            res = parse_uri(self.host, default_port=self.port, warn=True)
            self.host, self.port = res["nodelist"][0]
            dbase = res["database"]
        else:
            self.host, self.port = split_hosts(self.host, default_port=self.port)[0]

        self.__default_database_name = dbase
        # relative
        from . import SERVER_VERSION

        self._server_version = SERVER_VERSION

    def __getitem__(self, db_name):
        return self.get_database(db_name)

    def __getattr__(self, attr):
        return self[attr]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __repr__(self):
        return "mongomock.MongoClient('{0}', {1})".format(self.host, self.port)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.address == other.address
        return NotImplemented

    if helpers.PYMONGO_VERSION >= version.parse("3.12"):

        def __hash__(self):
            return hash(self.address)

    def close(self):
        pass

    @property
    def is_mongos(self):
        return True

    @property
    def is_primary(self):
        return True

    @property
    def address(self):
        return self.host, self.port

    @property
    def read_preference(self):
        return self._read_preference

    @property
    def codec_options(self):
        return self._codec_options

    def server_info(self):
        return {
            "version": self._server_version,
            "sysInfo": "Mock",
            "versionArray": _convert_version_to_list(self._server_version),
            "bits": 64,
            "debug": False,
            "maxBsonObjectSize": 16777216,
            "ok": 1,
        }

    if helpers.PYMONGO_VERSION < version.parse("4.0"):

        def database_names(self):
            warnings.warn(
                "database_names is deprecated. Use list_database_names instead."
            )
            return self.list_database_names()

    def list_database_names(self):
        return self._store.list_created_database_names()

    def drop_database(self, name_or_db):
        def drop_collections_for_db(_db):
            db_store = self._store[_db.name]
            for col_name in db_store.list_created_collection_names():
                _db.drop_collection(col_name)

        if isinstance(name_or_db, Database):
            db = next(db for db in self._database_accesses.values() if db is name_or_db)
            if db:
                drop_collections_for_db(db)

        elif name_or_db in self._store:
            db = self.get_database(name_or_db)
            drop_collections_for_db(db)

    def get_database(
        self,
        name=None,
        codec_options=None,
        read_preference=None,
        write_concern=None,
        read_concern=None,
    ):
        if name is None:
            db = self.get_default_database(
                codec_options=codec_options,
                read_preference=read_preference,
                write_concern=write_concern,
                read_concern=read_concern,
            )
        else:
            db = self._database_accesses.get(name)
        if db is None:
            db_store = self._store[name]
            db = self._database_accesses[name] = Database(
                self,
                name,
                read_preference=read_preference or self.read_preference,
                codec_options=codec_options or self._codec_options,
                _store=db_store,
                read_concern=read_concern,
            )
        return db

    def get_default_database(self, default=None, **kwargs):
        name = self.__default_database_name
        name = name if name is not None else default
        if name is None:
            raise ConfigurationError("No default database name defined or provided.")

        return self.get_database(name=name, **kwargs)

    def alive(self):
        """The original MongoConnection.alive method checks the status of the server.

        In our case as we mock the actual server, we should always return True.
        """
        return True

    def start_session(self, causal_consistency=True, default_transaction_options=None):
        """Start a logical session."""
        raise NotImplementedError("Mongomock does not support sessions yet")
