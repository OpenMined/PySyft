# stdlib
from collections.abc import Callable

# third party
from unsync import Unfuture
from unsync import unsync

# syft absolute
from syft.client.datasite_client import DatasiteClient
from syft.orchestra import ServerHandle


def with_client(func, client: unsync | DatasiteClient | ServerHandle) -> Callable:
    if isinstance(client, ServerHandle):
        client = client.client

    def with_func():
        result = func(client)
        if isinstance(result, Unfuture):
            result = result.result()
        return result

    return with_func
