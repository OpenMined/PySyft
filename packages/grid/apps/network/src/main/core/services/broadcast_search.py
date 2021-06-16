# stdlib
from datetime import datetime
import json
import secrets
from typing import List
from typing import Type
from typing import Union

# third party
from nacl.encoding import HexEncoder
from nacl.signing import SigningKey
from nacl.signing import VerifyKey
import requests
from syft.core.common.message import ImmediateSyftMessageWithReply

# syft relative
from syft.core.node.abstract.node import AbstractNode
from syft.core.node.common.service.auth import service_auth
from syft.core.node.common.service.node_service import ImmediateNodeServiceWithReply
from syft.core.node.domain.client import DomainClient
from syft.grid.client.client import connect
from syft.grid.client.grid_connection import GridHTTPConnection
from syft.grid.messages.network_search_message import NetworkSearchMessage
from syft.grid.messages.network_search_message import NetworkSearchResponse

# grid relative
from ..database.utils import model_to_json
from ..exceptions import AuthorizationError
from ..exceptions import InvalidParameterValueError
from ..exceptions import MissingRequestKeyError


class BroadcastSearchService(ImmediateNodeServiceWithReply):
    @staticmethod
    @service_auth(guests_welcome=True)
    def process(
        node: AbstractNode,
        msg: NetworkSearchMessage,
        verify_key: VerifyKey,
    ) -> NetworkSearchResponse:
        queries = msg.content.get("query", [])
        associations = node.association_requests.associations()

        match_nodes = []
        for node_association in associations:
            result = json.loads(
                requests.post(
                    node_association.address + "/search/domain-search",
                    json={"query": queries},
                ).text
            )
            if result["items"] > 0:
                result["address"] = node_association.address
                match_nodes.append(result)

        return NetworkSearchResponse(
            address=msg.reply_to, status_code=200, content=match_nodes
        )

    @staticmethod
    def message_handler_types() -> List[Type[ImmediateSyftMessageWithReply]]:
        return [NetworkSearchMessage]
