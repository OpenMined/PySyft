# stdlib
import logging
from typing import cast

# relative
from ...types.datetime import DateTime
from ..context import AuthedServiceContext
from ..response import SyftSuccess
from .network_service import NetworkService
from .node_peer import NodePeer
from .node_peer import NodePeerConnectionStatus


def peer_route_heathcheck(context: AuthedServiceContext) -> None:
    """
    Perform a health check on the peers in the network stash.

    Args:
        context (AuthedServiceContext): The authenticated service context.

    Returns:
        None
    """
    network_service = cast(NetworkService, context.node.get_service(NetworkService))
    network_stash = network_service.stash

    result = network_stash.get_all(context.node.verify_key)

    if result.is_err():
        logging.info(f"Failed to fetch peers from stash: {result.err()}")

    all_peers: list[NodePeer] = result.ok()

    for peer in all_peers:
        peer.pinged_timestamp = DateTime.now()
        try:
            peer_client = peer.client_with_context(context=context)
        except Exception as e:
            logging.error(f"Failed to create client for peer: {peer}: {e}")
            peer.ping_status = NodePeerConnectionStatus.TIMEOUT
            peer_client = None

        if peer_client is not None:
            peer_status = peer_client.api.services.network.find_peer(
                peer_id=context.node.id
            )
            peer.ping_status = (
                NodePeerConnectionStatus.ACTIVE
                if isinstance(peer_status, SyftSuccess)
                else NodePeerConnectionStatus.INACTIVE
            )
            peer.ping_status_message = peer_status.message

        result = network_stash.update_peer(
            credentials=context.node.verify_key, peer=peer
        )

        if result.is_err():
            logging.info(f"Failed to update peer in stash: {result.err()}")
