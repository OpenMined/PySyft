# stdlib
from typing import Union

# relative
from .....grid import GridURL
from ...abstract.node import AbstractNodeClient

ClientLike = Union[AbstractNodeClient, GridURL]


class NewRequestAPI:
    # def get_client(client: ClientLike) -> AbstractNodeClient:
    #     if isinstance(client, AbstractNodeClient):
    #         return client

    #     # relative
    #     from .....grid.client.client import connect

    #     return connect(url=client.with_path("/api/v1"), timeout=10)

    def get_client_url(self, client: ClientLike) -> GridURL:
        if isinstance(client, (str, GridURL)):
            return GridURL.from_url(client)
        else:
            return client.routes[0].connection.base_url  # type: ignore
