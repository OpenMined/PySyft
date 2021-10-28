# stdlib
from typing import Any
from typing import List

# third party
from pandas import DataFrame
from nacl.signing import SigningKey


# relative
from .....core.common.uid import UID
from .....core.io.address import Address
from .....grid.client.grid_connection import GridHTTPConnection
from .....grid.client.proxy_client import ProxyClient
from .....core.io.location import SpecificLocation
from .....core.io.route import SoloRoute
from ...abstract.node import AbstractNodeClient
from ..node_service.peer_discovery.peer_discovery_messages import (
    GetPeerInfoMessageWithReply,
)
from ..node_service.peer_discovery.peer_discovery_messages import (
    PeerDiscoveryMessageWithReply,
)
from .request_api import RequestAPI


class DomainRequestAPI(RequestAPI):
    def __init__(self, client: AbstractNodeClient):
        super().__init__(
            client=client,
            get_all_msg=PeerDiscoveryMessageWithReply,
            get_msg=GetPeerInfoMessageWithReply,
        )

    def all(self, pandas=True) -> List[Any]:
        response = self.perform_api_request_generic(
            syft_msg=PeerDiscoveryMessageWithReply, content={}
        )
        result = response.payload.kwargs  # type: ignore

        if result["status"] == "ok":
            data = result["data"]
            if pandas:
                data = DataFrame(data)

            return data
        return {"status": "error"}

    def get(self, node_uid: str) -> object:
        response = self.perform_api_request_generic(
            syft_msg=GetPeerInfoMessageWithReply, content={"uid": node_uid}
        )

        result = response.payload.kwargs.upcast() # type: ignore
        spec_location = SpecificLocation(UID.from_string(result['data']["id"]))
        addr = Address(name=result['data']['name'], domain=spec_location)

        conn = GridHTTPConnection(url=self.client.routes[0].connection.base_url)
        metadata = conn._get_metadata()  # type: ignore
        _user_key = SigningKey.generate()


        (
            spec_location,
            name,
            client_id,
        ) = ProxyClient.deserialize_client_metadata_from_node(metadata=metadata)

        # Create a new Solo Route using the selected connection type
        route = SoloRoute(destination=spec_location, connection=conn)

        
        proxy_client = ProxyClient(
                name=addr.name,
                routes=[route],
                signing_key=_user_key,
                domain=addr.domain
        )
        
        # Set Domain's proxy address
        proxy_client.proxy_address = proxy_address
        
        return proxy_client
    
    def __getitem__(self, node_uid: str) -> object:
        return self.get(node_uid=node_uid)
