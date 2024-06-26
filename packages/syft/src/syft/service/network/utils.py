# stdlib
import logging
import threading
import time
from typing import cast

# relative
from ...serde.serializable import serializable
from ...types.datetime import DateTime
from ..context import AuthedServiceContext
from ..response import SyftError
from .network_service import NetworkService
from .network_service import NodePeerAssociationStatus
from .node_peer import NodePeer
from .node_peer import NodePeerConnectionStatus
from .node_peer import NodePeerUpdate

logger = logging.getLogger(__name__)


@serializable(without=["thread"])
class PeerHealthCheckTask:
    repeat_time = 10  # in seconds

    def __init__(self) -> None:
        self.thread: threading.Thread | None = None
        self.started_time = None
        self._stop = False

    def peer_route_heathcheck(self, context: AuthedServiceContext) -> SyftError | None:
        """
        Perform a health check on the peers in the network stash.
        - If peer is accessible, ping the peer.
        - Peer is connected to the network.

        Args:
            context (AuthedServiceContext): The authenticated service context.

        Returns:
            None
        """

        network_service = cast(NetworkService, context.node.get_service(NetworkService))
        network_stash = network_service.stash

        result = network_stash.get_all(context.node.verify_key)

        if result.is_err():
            logger.error(f"Failed to fetch peers from stash: {result.err()}")
            return SyftError(message=f"{result.err()}")

        all_peers: list[NodePeer] = result.ok()

        for peer in all_peers:
            peer_update = NodePeerUpdate(id=peer.id)
            peer_update.pinged_timestamp = DateTime.now()
            try:
                peer_client = peer.client_with_context(context=context)
                if peer_client.is_err():
                    logger.error(
                        f"Failed to create client for peer: {peer}: {peer_client.err()}"
                    )
                    peer_update.ping_status = NodePeerConnectionStatus.TIMEOUT
                    peer_client = None
            except Exception as e:
                logger.error(f"Failed to create client for peer: {peer}", exc_info=e)

                peer_update.ping_status = NodePeerConnectionStatus.TIMEOUT
                peer_client = None

            if peer_client is not None:
                peer_client = peer_client.ok()
                peer_status = peer_client.api.services.network.check_peer_association(
                    peer_id=context.node.id
                )
                peer_update.ping_status = (
                    NodePeerConnectionStatus.ACTIVE
                    if peer_status == NodePeerAssociationStatus.PEER_ASSOCIATED
                    else NodePeerConnectionStatus.INACTIVE
                )
                if isinstance(peer_status, SyftError):
                    peer_update.ping_status_message = (
                        f"Error `{peer_status.message}` when pinging peer '{peer.name}'"
                    )
                else:
                    peer_update.ping_status_message = (
                        f"Peer '{peer.name}''s ping status: "
                        f"{peer_update.ping_status.value.lower()}"
                    )

            result = network_stash.update(
                credentials=context.node.verify_key,
                peer_update=peer_update,
                has_permission=True,
            )

            if result.is_err():
                logger.error(f"Failed to update peer in stash: {result.err()}")

        return None

    def _run(self, context: AuthedServiceContext) -> None:
        self.started_time = DateTime.now()
        while True:
            if self._stop:
                break
            self.peer_route_heathcheck(context)
            time.sleep(self.repeat_time)

    def run(self, context: AuthedServiceContext) -> None:
        if self.thread is not None:
            logger.info(
                f"Peer health check task is already running in thread "
                f"{self.thread.name} with ID: {self.thread.ident}."
            )
        else:
            self.thread = threading.Thread(target=self._run, args=(context,))
            logger.info(
                f"Start running peers health check in thread "
                f"{self.thread.name} with ID: {self.thread.ident}."
            )
            self.thread.start()

    def stop(self) -> None:
        if self.thread:
            self._stop = True
            self.thread.join()
            self.thread = None
            self.started_time = None
        logger.info("Peer health check task stopped.")
