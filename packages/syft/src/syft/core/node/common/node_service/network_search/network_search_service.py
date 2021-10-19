# stdlib
from typing import List
from typing import Type

# third party
from nacl.signing import VerifyKey

# relative
from ......grid.client.client import connect
from ......grid.client.grid_connection import GridHTTPConnection
from .....common.message import ImmediateSyftMessageWithReply
from ....abstract.node import AbstractNode
from ..auth import service_auth
from ..node_service import ImmediateNodeServiceWithReply
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
        associations = node.association_requests.associations()  # type: ignore

        print("querues", queries)
        print("associations", associations)

        def filter_domains(association: Any) -> bool:
            print("association", association)
            # domain = connect(
            #     url=url,  # Domain Address
            #     conn_type=GridHTTPConnection,  # HTTP Connection Protocol
            # )
            association.target
            print("association", association, type(association))
            print("does domain connect", association.target)

            for data in association.target.store:
                if queries.issubset(set(data.tags)):
                    return True
            return False

        filtered_nodes = list(filter(lambda x: filter_domains(x), associations))

        match_nodes = [node.address for node in filtered_nodes]

        return NetworkSearchResponse(
            address=msg.reply_to, status_code=200, content={"match-nodes": match_nodes}
        )

    @staticmethod
    def message_handler_types() -> List[Type[ImmediateSyftMessageWithReply]]:
        return [NetworkSearchMessage]
