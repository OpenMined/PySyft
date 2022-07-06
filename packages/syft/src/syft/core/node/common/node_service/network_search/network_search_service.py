# stdlib
from typing import List
from typing import Set
from typing import Type

# third party
from nacl.signing import VerifyKey

# relative
from ......logger import error
from .....common.message import ImmediateSyftMessageWithReply
from .....common.uid import UID
from ....abstract.node import AbstractNode
from ....domain_client import DomainClient
from ..auth import service_auth
from ..node_service import ImmediateNodeServiceWithReply
from ..peer_discovery.peer_discovery_messages import node_id_to_peer_route_metadata
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
        queries = set(msg.content)

        # refresh any missing peer clients
        node.reload_peer_clients()  # type: ignore

        def query_client_store(client: DomainClient, tags: Set[str]) -> bool:
            # source is the domain in an association
            for data in client.store:
                if tags.issubset(set(data.tags)):
                    return True
            return False

        tested_nodes: Set[UID] = set()
        matching_nodes: Set[UID] = set()
        for node_id, clients in node.all_peer_clients().items():  # type: ignore
            # try each client / route one after the next
            for client in clients:
                try:
                    # only check a client once
                    if node_id not in tested_nodes and query_client_store(
                        client=client, tags=queries
                    ):
                        matching_nodes.add(node_id)
                except Exception as e:
                    error(f"Failed to query {node_id} with {client}. {e}")

        peer_routes = []
        for node_id in matching_nodes:
            node_row = node.node.first(node_uid=node_id.no_dash)  # type: ignore
            if node_row:
                peer_routes += node_id_to_peer_route_metadata(
                    node=node, node_row=node_row  # type: ignore
                )

        return NetworkSearchResponse(
            address=msg.reply_to,
            status_code=200,
            content={"status": "ok", "data": peer_routes},
        )

    @staticmethod
    def message_handler_types() -> List[Type[ImmediateSyftMessageWithReply]]:
        return [NetworkSearchMessage]
