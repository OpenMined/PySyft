# stdlib
import os

try:
    # third party
    from pymongo.errors import PyMongoError
except ImportError:

    class PyMongoError(Exception):
        pass


try:
    # third party
    from pymongo.errors import OperationFailure
except ImportError:

    class OperationFailure(PyMongoError):
        def __init__(self, message, code=None, details=None):
            super(OperationFailure, self).__init__()
            self._message = message
            self._code = code
            self._details = details

        code = property(lambda self: self._code)
        details = property(lambda self: self._details)

        def __str__(self):
            return self._message


try:
    # third party
    from pymongo.errors import WriteError
except ImportError:

    class WriteError(OperationFailure):
        pass


try:
    # third party
    from pymongo.errors import DuplicateKeyError
except ImportError:

    class DuplicateKeyError(WriteError):
        pass


try:
    # third party
    from pymongo.errors import BulkWriteError
except ImportError:

    class BulkWriteError(OperationFailure):
        def __init__(self, results):
            super(BulkWriteError, self).__init__(
                "batch op errors occurred", 65, results
            )


try:
    # third party
    from pymongo.errors import CollectionInvalid
except ImportError:

    class CollectionInvalid(PyMongoError):
        pass


try:
    # third party
    from pymongo.errors import InvalidName
except ImportError:

    class InvalidName(PyMongoError):
        pass


try:
    # third party
    from pymongo.errors import InvalidOperation
except ImportError:

    class InvalidOperation(PyMongoError):
        pass


try:
    # third party
    from pymongo.errors import ConfigurationError
except ImportError:

    class ConfigurationError(PyMongoError):
        pass


try:
    # third party
    from pymongo.errors import InvalidURI
except ImportError:

    class InvalidURI(ConfigurationError):
        pass


from .helpers import ObjectId, utcnow  # noqa


__all__ = [
    "Database",
    "DuplicateKeyError",
    "Collection",
    "CollectionInvalid",
    "InvalidName",
    "MongoClient",
    "ObjectId",
    "OperationFailure",
    "WriteConcern",
    "ignore_feature",
    "patch",
    "warn_on_feature",
    "SERVER_VERSION",
]

# relative
from .collection import Collection
from .database import Database
from .mongo_client import MongoClient
from .not_implemented import ignore_feature
from .not_implemented import warn_on_feature
from .patch import patch
from .write_concern import WriteConcern

# The version of the server faked by mongomock. Callers may patch it before creating connections to
# update the behavior of mongomock.
# Keep the default version in sync with docker-compose.yml and travis.yml.
SERVER_VERSION = os.getenv("MONGODB", "5.0.5")
