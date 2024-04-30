# stdlib
import time

# relative
from .mongo_client import MongoClient

try:
    # stdlib
    from unittest import mock

    _IMPORT_MOCK_ERROR = None
except ImportError:
    try:
        # third party
        import mock

        _IMPORT_MOCK_ERROR = None
    except ImportError as error:
        _IMPORT_MOCK_ERROR = error

try:
    # third party
    import pymongo
    from pymongo.uri_parser import parse_uri
    from pymongo.uri_parser import split_hosts

    _IMPORT_PYMONGO_ERROR = None
except ImportError as error:
    # relative
    from .helpers import parse_uri
    from .helpers import split_hosts

    _IMPORT_PYMONGO_ERROR = error


def _parse_any_host(host, default_port=27017):
    if isinstance(host, tuple):
        return _parse_any_host(host[0], host[1])
    if "://" in host:
        return parse_uri(host, warn=True)["nodelist"]
    return split_hosts(host, default_port=default_port)


def patch(servers="localhost", on_new="error"):
    """Patch pymongo.MongoClient.

    This will patch the class MongoClient and use mongomock to mock MongoDB
    servers. It keeps a consistant state of servers across multiple clients so
    you can do:

    ```
    client = pymongo.MongoClient(host='localhost', port=27017)
    client.db.coll.insert_one({'name': 'Pascal'})

    other_client = pymongo.MongoClient('mongodb://localhost:27017')
    client.db.coll.find_one()
    ```

    The data is persisted as long as the patch lives.

    Args:
        on_new: Behavior when accessing a new server (not in servers):
            'create': mock a new empty server, accept any client connection.
            'error': raise a ValueError immediately when trying to access.
            'timeout': behave as pymongo when a server does not exist, raise an
                error after a timeout.
            'pymongo': use an actual pymongo client.
        servers: a list of server that are avaiable.
    """

    if _IMPORT_MOCK_ERROR:
        raise _IMPORT_MOCK_ERROR  # pylint: disable=raising-bad-type

    if _IMPORT_PYMONGO_ERROR:
        PyMongoClient = None
    else:
        PyMongoClient = pymongo.MongoClient

    persisted_clients = {}
    parsed_servers = set()
    for server in servers if isinstance(servers, (list, tuple)) else [servers]:
        parsed_servers.update(_parse_any_host(server))

    def _create_persistent_client(*args, **kwargs):
        if _IMPORT_PYMONGO_ERROR:
            raise _IMPORT_PYMONGO_ERROR  # pylint: disable=raising-bad-type

        client = MongoClient(*args, **kwargs)

        try:
            persisted_client = persisted_clients[client.address]
            client._store = persisted_client._store
            return client
        except KeyError:
            pass

        if client.address in parsed_servers or on_new == "create":
            persisted_clients[client.address] = client
            return client

        if on_new == "timeout":
            # TODO(pcorpet): Only wait when trying to access the server's data.
            time.sleep(kwargs.get("serverSelectionTimeoutMS", 30000))
            raise pymongo.errors.ServerSelectionTimeoutError(
                "%s:%d: [Errno 111] Connection refused" % client.address
            )

        if on_new == "pymongo":
            return PyMongoClient(*args, **kwargs)

        raise ValueError(
            "MongoDB server %s:%d does not exist.\n" % client.address
            + "%s" % parsed_servers
        )

    class _PersistentClient:
        def __new__(cls, *args, **kwargs):
            return _create_persistent_client(*args, **kwargs)

    return mock.patch("pymongo.MongoClient", _PersistentClient)
