# stdlib
from typing import Any
from typing import List

# third party
from pandas import DataFrame

# relative
from .....core.common.uid import UID
from .....core.io.location import SpecificLocation
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

        result = response.payload.kwargs  # type: ignore
        spec_location = SpecificLocation(UID.from_string(result["id"]))
    

    def __getitem__(self, node_uid: str) -> object:
        return self.get(node_uid=node_uid)
