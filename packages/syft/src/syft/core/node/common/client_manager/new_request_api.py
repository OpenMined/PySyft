# stdlib
from typing import Union

# relative
from .....grid import GridURL
from ...abstract.node import AbstractNodeClient

ClientLike = Union[AbstractNodeClient, GridURL]


class NewRequestAPI:
    def __init__(self, client: AbstractNodeClient):
        self.client = client

    def get_client(self, client: ClientLike) -> AbstractNodeClient:
        if isinstance(client, AbstractNodeClient):
            client.signing_key = self.client.signing_key
            client.verify_key = self.client.signing_key.verify_key
            return client

        if isinstance(client, str):
            client = GridURL.from_url(client)

        # relative
        from .....grid.client.client import connect

        return connect(
            url=client.with_path("/api/v1"),
            timeout=10,
            user_key=self.client.signing_key,
        )

    def get_client_url(self, client: ClientLike) -> GridURL:
        if isinstance(client, (str, GridURL)):
            return GridURL.from_url(client)
        else:
            return client.routes[0].connection.base_url  # type: ignore
