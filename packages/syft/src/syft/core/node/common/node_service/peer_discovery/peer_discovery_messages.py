# future
from __future__ import annotations

# stdlib
from typing import Any
from typing import Dict
from typing import Optional

# third party
from nacl.signing import VerifyKey
from typing_extensions import final

# relative
from .....common.serde.serializable import serializable
from ....abstract.node import AbstractNode
from ..generic_payload.messages import GenericPayloadMessage
from ..generic_payload.messages import GenericPayloadMessageWithReply
from ..generic_payload.messages import GenericPayloadReplyMessage


@serializable(recursive_serde=True)
@final
class PeerDiscoveryMessage(GenericPayloadMessage):
    ...


@serializable(recursive_serde=True)
@final
class PeerDiscoveryReplyMessage(GenericPayloadReplyMessage):
    ...


@serializable(recursive_serde=True)
@final
class PeerDiscoveryMessageWithReply(GenericPayloadMessageWithReply):
    message_type = PeerDiscoveryMessage
    message_reply_type = PeerDiscoveryReplyMessage

    def run(
            self, node: AbstractNode, verify_key: Optional[VerifyKey] = None
    ) -> Dict[str, Any]:
        try:
            # peer_route_clients: Dict[UID, Dict[str, Dict[str, Client]]] = {}
            peer_routes = []
            for peer in node.node.all():
                routes = node.node_route.query(node_id=peer.id)
                for route in routes:
                    peer_route = {}
                    peer_route["id"] = peer.node_uid
                    peer_route["name"] = peer.node_name
                    peer_route["host_or_ip"] = route.host_or_ip
                    peer_route["is_vpn"] = route.is_vpn
                    peer_routes.append(peer_route)
            return {"status": "ok", "data": peer_routes}
        except Exception as e:
            print(f"Failed to run {type(self)}", self.kwargs, e)
            return {"status": "error"}


@serializable(recursive_serde=True)
@final
class GetPeerInfoMessageWithReply(GenericPayloadMessageWithReply):
    message_type = PeerDiscoveryMessage
    message_reply_type = PeerDiscoveryReplyMessage

    def run(
        self, node: AbstractNode, verify_key: Optional[VerifyKey] = None
    ) -> Dict[str, Any]:
        try:
            peer = node.node.first(node_uid=str(self.kwargs['uid']))
            peer_route = {}
            if peer:
                route = node.node_route.first(node_id=peer.id)
                peer_route["id"] = peer.node_uid
                peer_route["name"] = peer.node_name
                peer_route["host_or_ip"] = route.host_or_ip
                peer_route["is_vpn"] = route.is_vpn
            return {'status': 'ok', 'data': peer_route} 
        except Exception as e:
            print(f"Failed to run {type(self)}", self.kwargs, e)
            return {"status": "error"}
