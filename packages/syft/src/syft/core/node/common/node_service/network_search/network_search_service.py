# stdlib
from typing import List
from typing import Type

# third party
from nacl.signing import VerifyKey

# syft absolute
from syft.core.common.message import ImmediateSyftMessageWithReply
from syft.core.node.abstract.node import AbstractNode
from syft.core.node.common.node_service.auth import service_auth
from syft.core.node.common.node_service.node_service import (
    ImmediateNodeServiceWithReply,
)
from syft.grid.client.client import connect
from syft.grid.client.grid_connection import GridHTTPConnection

# relative
from .network_search_messages import NetworkSearchMessage
from .network_search_messages import NetworkSearchResponse


class NetworkSearchService(ImmediateNodeServiceWithReply):
    @staticmethod
    @service_auth(guests_welcome=True)
    def process(
        node: AbstractNode,
        msg: NetworkSearchMessage,
        verify_key: VerifyKey,
    ) -> NetworkSearchResponse:
        queries = set(msg.content.get("query", []))
        associations = node.association_requests.associations()

        def filter_domains(url: str) -> bool:
            domain = connect(
                url=url,  # Domain Address
                conn_type=GridHTTPConnection,  # HTTP Connection Protocol
            )

            for data in domain.store:
                if queries.issubset(set(data.tags)):
                    return True
            return False

        filtered_nodes = list(filter(lambda x: filter_domains(x.address), associations))

        match_nodes = [node.address for node in filtered_nodes]

        return NetworkSearchResponse(
            address=msg.reply_to, status_code=200, content={"match-nodes": match_nodes}
        )

    @staticmethod
    def message_handler_types() -> List[Type[ImmediateSyftMessageWithReply]]:
        return [NetworkSearchMessage]
