# stdlib
import warnings

# third party
from packaging import version

# relative
from . import CollectionInvalid
from . import InvalidName
from . import OperationFailure
from . import codec_options as mongomock_codec_options
from . import helpers
from . import read_preferences
from . import store
from .collection import Collection
from .filtering import filter_applies

try:
    # third party
    from pymongo import ReadPreference

    _READ_PREFERENCE_PRIMARY = ReadPreference.PRIMARY
except ImportError:
    _READ_PREFERENCE_PRIMARY = read_preferences.PRIMARY

try:
    # third party
    from pymongo.read_concern import ReadConcern
except ImportError:
    # relative
    from .read_concern import ReadConcern

_LIST_COLLECTION_FILTER_ALLOWED_OPERATORS = frozenset(["$regex", "$eq", "$ne"])


def _verify_list_collection_supported_op(keys):
    if set(keys) - _LIST_COLLECTION_FILTER_ALLOWED_OPERATORS:
        raise NotImplementedError(
            "list collection names filter operator {0} is not implemented yet in mongomock "
            "allowed operators are {1}".format(
                keys, _LIST_COLLECTION_FILTER_ALLOWED_OPERATORS
            )
        )


class Database(object):
    def __init__(
        self,
        client,
        name,
        _store,
        read_preference=None,
        codec_options=None,
        read_concern=None,
    ):
        self.name = name
        self._client = client
        self._collection_accesses = {}
        self._store = _store or store.DatabaseStore()
        self._read_preference = read_preference or _READ_PREFERENCE_PRIMARY
        mongomock_codec_options.is_supported(codec_options)
        self._codec_options = codec_options or mongomock_codec_options.CodecOptions()
        if read_concern and not isinstance(read_concern, ReadConcern):
            raise TypeError(
                "read_concern must be an instance of pymongo.read_concern.ReadConcern"
            )
        self._read_concern = read_concern or ReadConcern()

    def __getitem__(self, coll_name):
        return self.get_collection(coll_name)

    def __getattr__(self, attr):
        if attr.startswith("_"):
            raise AttributeError(
                "%s has no attribute '%s'. To access the %s collection, use database['%s']."
                % (self.__class__.__name__, attr, attr, attr)
            )
        return self[attr]

    def __repr__(self):
        return "Database({0}, '{1}')".format(self._client, self.name)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self._client == other._client and self.name == other.name
        return NotImplemented

    if helpers.PYMONGO_VERSION >= version.parse("3.12"):

        def __hash__(self):
            return hash((self._client, self.name))

    @property
    def client(self):
        return self._client

    @property
    def read_preference(self):
        return self._read_preference

    @property
    def codec_options(self):
        return self._codec_options

    @property
    def read_concern(self):
        return self._read_concern

    def _get_created_collections(self):
        return self._store.list_created_collection_names()

    if helpers.PYMONGO_VERSION < version.parse("4.0"):

        def collection_names(self, include_system_collections=True, session=None):
            warnings.warn(
                "collection_names is deprecated. Use list_collection_names instead."
            )
            if include_system_collections:
                return list(self._get_created_collections())
            return self.list_collection_names(session=session)

    def list_collections(self, filter=None, session=None, nameOnly=False):
        raise NotImplementedError(
            "list_collections is a valid method of Database but has not been implemented in "
            "mongomock yet."
        )

    def list_collection_names(self, filter=None, session=None):
        """filter: only name field type with eq,ne or regex operator

        session: not supported
        for supported operator please see _LIST_COLLECTION_FILTER_ALLOWED_OPERATORS
        """
        field_name = "name"

        if session:
            raise NotImplementedError("Mongomock does not handle sessions yet")

        if filter:
            if not filter.get("name"):
                raise NotImplementedError(
                    "list collection {0} might be valid but is not "
                    "implemented yet in mongomock".format(filter)
                )

            filter = (
                {field_name: {"$eq": filter.get(field_name)}}
                if isinstance(filter.get(field_name), str)
                else filter
            )

            _verify_list_collection_supported_op(filter.get(field_name).keys())

            return [
                name
                for name in list(self._store._collections)
                if filter_applies(filter, {field_name: name})
                and not name.startswith("system.")
            ]

        return [
            name
            for name in self._get_created_collections()
            if not name.startswith("system.")
        ]

    def get_collection(
        self,
        name,
        codec_options=None,
        read_preference=None,
        write_concern=None,
        read_concern=None,
    ):
        if read_preference is not None:
            read_preferences.ensure_read_preference_type(
                "read_preference", read_preference
            )
        mongomock_codec_options.is_supported(codec_options)
        try:
            return self._collection_accesses[name].with_options(
                codec_options=codec_options or self._codec_options,
                read_preference=read_preference or self.read_preference,
                read_concern=read_concern,
                write_concern=write_concern,
            )
        except KeyError:
            self._ensure_valid_collection_name(name)
            collection = self._collection_accesses[name] = Collection(
                self,
                name=name,
                read_concern=read_concern,
                write_concern=write_concern,
                read_preference=read_preference or self.read_preference,
                codec_options=codec_options or self._codec_options,
                _db_store=self._store,
            )
            return collection

    def drop_collection(self, name_or_collection, session=None):
        if session:
            raise NotImplementedError("Mongomock does not handle sessions yet")
        if isinstance(name_or_collection, Collection):
            name_or_collection._store.drop()
        else:
            self._store[name_or_collection].drop()

    def _ensure_valid_collection_name(self, name):
        # These are the same checks that are done in pymongo.
        if not isinstance(name, str):
            raise TypeError("name must be an instance of str")
        if not name or ".." in name:
            raise InvalidName("collection names cannot be empty")
        if name[0] == "." or name[-1] == ".":
            raise InvalidName("collection names must not start or end with '.'")
        if "$" in name:
            raise InvalidName("collection names must not contain '$'")
        if "\x00" in name:
            raise InvalidName("collection names must not contain the null character")

    def create_collection(self, name, **kwargs):
        self._ensure_valid_collection_name(name)
        if name in self.list_collection_names():
            raise CollectionInvalid("collection %s already exists" % name)

        if kwargs:
            raise NotImplementedError("Special options not supported")

        self._store.create_collection(name)
        return self[name]

    def rename_collection(self, name, new_name, dropTarget=False):
        """Changes the name of an existing collection."""
        self._ensure_valid_collection_name(new_name)

        # Reference for server implementation:
        # https://docs.mongodb.com/manual/reference/command/renameCollection/
        if not self._store[name].is_created:
            raise OperationFailure(
                'The collection "{0}" does not exist.'.format(name), 10026
            )
        if new_name in self._store:
            if dropTarget:
                self.drop_collection(new_name)
            else:
                raise OperationFailure(
                    'The target collection "{0}" already exists'.format(new_name), 10027
                )
        self._store.rename(name, new_name)
        return {"ok": 1}

    def dereference(self, dbref, session=None):
        if session:
            raise NotImplementedError("Mongomock does not handle sessions yet")

        if not hasattr(dbref, "collection") or not hasattr(dbref, "id"):
            raise TypeError("cannot dereference a %s" % type(dbref))
        if dbref.database is not None and dbref.database != self.name:
            raise ValueError(
                "trying to dereference a DBRef that points to "
                "another database (%r not %r)" % (dbref.database, self.name)
            )
        return self[dbref.collection].find_one({"_id": dbref.id})

    def command(self, command, **unused_kwargs):
        if isinstance(command, str):
            command = {command: 1}
        if "ping" in command:
            return {"ok": 1.0}
        # TODO(pascal): Differentiate NotImplementedError for valid commands
        # and OperationFailure if the command is not valid.
        raise NotImplementedError(
            "command is a valid Database method but is not implemented in Mongomock yet"
        )

    def with_options(
        self,
        codec_options=None,
        read_preference=None,
        write_concern=None,
        read_concern=None,
    ):
        mongomock_codec_options.is_supported(codec_options)

        if write_concern:
            raise NotImplementedError(
                "write_concern is a valid parameter for with_options but is not implemented yet in "
                "mongomock"
            )

        if read_preference is None or read_preference == self._read_preference:
            return self

        return Database(
            self._client,
            self.name,
            self._store,
            read_preference=read_preference or self._read_preference,
            codec_options=codec_options or self._codec_options,
            read_concern=read_concern or self._read_concern,
        )
