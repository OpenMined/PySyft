# third party
from nacl.signing import SigningKey

# syft relative
from ...core.io.connection import ClientConnection
from ...core.io.route import SoloRoute
from ...core.node.network.client import NetworkClient


class GridNetworkClient(NetworkClient):
    def __init__(self, url: str, conn: ClientConnection, key: SigningKey):
        self.signing_key = key
        self.verify_key = self.signing_key.verify_key

        (
            spec_location,
            name,
            client_id,
        ) = NetworkClient.deserialize_client_metadata_from_node(
            metadata=conn._get_metadata(url)
        )
        route = SoloRoute(destination=spec_location, connection=conn)
        super().__init__(
            network=spec_location,
            name=name,
            routes=[route],
            signing_key=self.signing_key,
            verify_key=self.verify_key,
        )
