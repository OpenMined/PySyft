# stdlib
from enum import Enum
import itertools
import logging
import threading
import time
from typing import cast

# relative
from ...client.client import SyftClient
from ...serde.serializable import serializable
from ...types.datetime import DateTime
from ..context import AuthedServiceContext
from ..request.request import Request
from ..response import SyftError
from ..response import SyftSuccess
from ..user.user_roles import ServiceRole
from .network_service import NetworkService
from .network_service import ServerPeerAssociationStatus
from .server_peer import ServerPeer
from .server_peer import ServerPeerConnectionStatus
from .server_peer import ServerPeerUpdate

logger = logging.getLogger(__name__)


@serializable(without=["thread"], canonical_name="PeerHealthCheckTask", version=1)
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

        network_service = cast(
            NetworkService, context.server.get_service(NetworkService)
        )
        network_stash = network_service.stash

        result = network_stash.get_all(context.server.verify_key)

        if result.is_err():
            logger.error(f"Failed to fetch peers from stash: {result.err()}")
            return SyftError(message=f"{result.err()}")

        all_peers: list[ServerPeer] = result.ok()

        for peer in all_peers:
            peer_update = ServerPeerUpdate(id=peer.id)
            peer_update.pinged_timestamp = DateTime.now()
            try:
                peer_client = peer.client_with_context(context=context)
                if peer_client.is_err():
                    logger.error(
                        f"Failed to create client for peer: {peer}: {peer_client.err()}"
                    )
                    peer_update.ping_status = ServerPeerConnectionStatus.TIMEOUT
                    peer_client = None
            except Exception as e:
                logger.error(f"Failed to create client for peer: {peer}", exc_info=e)

                peer_update.ping_status = ServerPeerConnectionStatus.TIMEOUT
                peer_client = None

            if peer_client is not None:
                peer_client = peer_client.ok()
                peer_status = peer_client.api.services.network.check_peer_association(
                    peer_id=context.server.id
                )
                peer_update.ping_status = (
                    ServerPeerConnectionStatus.ACTIVE
                    if peer_status == ServerPeerAssociationStatus.PEER_ASSOCIATED
                    else ServerPeerConnectionStatus.INACTIVE
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
                credentials=context.server.verify_key,
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


def exchange_routes(
    clients: list[SyftClient], auto_approve: bool = False
) -> SyftSuccess | SyftError:
    """Exchange routes between a list of clients."""
    if auto_approve:
        # Check that all clients are admin clients
        for client in clients:
            if not client.user_role == ServiceRole.ADMIN:
                return SyftError(
                    message=f"Client {client} is not an admin client. "
                    "Only admin clients can auto-approve connection requests."
                )

    for client1, client2 in itertools.combinations(clients, 2):
        peer1 = ServerPeer.from_client(client1)
        peer2 = ServerPeer.from_client(client2)

        client1_connection_request = client1.api.services.network.add_peer(peer2)
        if isinstance(client1_connection_request, SyftError):
            return SyftError(
                message=f"Failed to add peer {peer2} to {client1}: {client1_connection_request}"
            )

        client2_connection_request = client2.api.services.network.add_peer(peer1)
        if isinstance(client2_connection_request, SyftError):
            return SyftError(
                message=f"Failed to add peer {peer1} to {client2}: {client2_connection_request}"
            )

        if auto_approve:
            if isinstance(client1_connection_request, Request):
                res1 = client1_connection_request.approve()
                if isinstance(res1, SyftError):
                    return SyftError(
                        message=f"Failed to approve connection request between {client1} and {client2}: {res1}"
                    )
            if isinstance(client2_connection_request, Request):
                res2 = client2_connection_request.approve()
                if isinstance(res2, SyftError):
                    return SyftError(
                        message=f"Failed to approve connection request between {client2} and {client1}: {res2}"
                    )
            logger.info(f"Exchanged routes between {client1} and {client2}")
        else:
            logger.info(f"Connection requests sent between {client1} and {client2}.")

    return SyftSuccess(message="Routes exchanged successfully.")


class NetworkTopology(Enum):
    STAR = "STAR"
    MESH = "MESH"
    HYBRID = "HYBRID"


def check_route_reachability(
    clients: list[SyftClient], topology: NetworkTopology = NetworkTopology.MESH
) -> SyftSuccess | SyftError:
    if topology == NetworkTopology.STAR:
        return SyftError(message="STAR topology is not supported yet")
    elif topology == NetworkTopology.MESH:
        return check_mesh_topology(clients)
    else:
        return SyftError(message=f"Invalid topology: {topology}")


def check_mesh_topology(clients: list[SyftClient]) -> SyftSuccess | SyftError:
    for client in clients:
        for other_client in clients:
            if client == other_client:
                continue
            result = client.api.services.network.ping_peer(
                verify_key=other_client.root_verify_key
            )
            if isinstance(result, SyftError):
                return SyftError(
                    message=f"{client.name}-<{client.id}> - cannot reach"
                    + f"{other_client.name}-<{other_client.id} - {result.message}"
                )
    return SyftSuccess(message="All clients are reachable")
