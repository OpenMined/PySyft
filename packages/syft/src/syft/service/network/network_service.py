# stdlib
from collections.abc import Callable
from enum import Enum
import logging
import secrets
from typing import Any
from typing import cast

# third party
from result import Result

# relative
from ...abstract_node import NodeType
from ...client.client import HTTPConnection
from ...client.client import PythonConnection
from ...client.client import SyftClient
from ...node.credentials import SyftVerifyKey
from ...node.worker_settings import WorkerSettings
from ...serde.serializable import serializable
from ...service.settings.settings import NodeSettings
from ...store.document_store import DocumentStore
from ...store.document_store import PartitionKey
from ...types.grid_url import GridURL
from ...types.transforms import TransformContext
from ...types.transforms import keep
from ...types.transforms import make_set_default
from ...types.transforms import transform
from ...types.transforms import transform_method
from ...types.uid import UID
from ...util.telemetry import instrument
from ...util.util import generate_token
from ...util.util import get_env
from ...util.util import prompt_warning_message
from ...util.util import str_to_bool
from ..context import AuthedServiceContext
from ..metadata.node_metadata import NodeMetadata
from ..request.request import Request
from ..request.request import RequestStatus
from ..request.request import SubmitRequest
from ..request.request_service import RequestService
from ..response import SyftError
from ..response import SyftInfo
from ..response import SyftSuccess
from ..service import AbstractService
from ..service import SERVICE_TO_TYPES
from ..service import TYPE_TO_SERVICE
from ..service import service_method
from ..user.user_roles import DATA_OWNER_ROLE_LEVEL
from ..user.user_roles import GUEST_ROLE_LEVEL
from ..warnings import CRUDWarning
from .association_request import AssociationRequestChange
from .network_stash import NetworkStash
from .node_peer import NodePeer
from .node_peer import NodePeerUpdate
from .reverse_tunnel_service import ReverseTunnelService
from .routes import HTTPNodeRoute
from .routes import NodeRoute
from .routes import NodeRouteType
from .routes import PythonNodeRoute

logger = logging.getLogger(__name__)

VerifyKeyPartitionKey = PartitionKey(key="verify_key", type_=SyftVerifyKey)
NodeTypePartitionKey = PartitionKey(key="node_type", type_=NodeType)
OrderByNamePartitionKey = PartitionKey(key="name", type_=str)

REVERSE_TUNNEL_ENABLED = "REVERSE_TUNNEL_ENABLED"


def reverse_tunnel_enabled() -> bool:
    return str_to_bool(get_env(REVERSE_TUNNEL_ENABLED, "false"))


@serializable()
class NodePeerAssociationStatus(Enum):
    PEER_ASSOCIATED = "PEER_ASSOCIATED"
    PEER_ASSOCIATION_PENDING = "PEER_ASSOCIATION_PENDING"
    PEER_NOT_FOUND = "PEER_NOT_FOUND"


@instrument
@serializable()
class NetworkService(AbstractService):
    store: DocumentStore
    stash: NetworkStash

    def __init__(self, store: DocumentStore) -> None:
        self.store = store
        self.stash = NetworkStash(store=store)
        if reverse_tunnel_enabled():
            self.rtunnel_service = ReverseTunnelService()

    @service_method(
        path="network.exchange_credentials_with",
        name="exchange_credentials_with",
        roles=GUEST_ROLE_LEVEL,
        warning=CRUDWarning(confirmation=True),
    )
    def exchange_credentials_with(
        self,
        context: AuthedServiceContext,
        self_node_route: NodeRoute,
        remote_node_route: NodeRoute,
        remote_node_verify_key: SyftVerifyKey,
        reverse_tunnel: bool = False,
    ) -> Request | SyftSuccess | SyftError:
        """
        Exchange Route With Another Node. If there is a pending association request, return it
        """

        # Step 1: Validate the Route
        self_node_peer = self_node_route.validate_with_context(context=context)

        if reverse_tunnel and not reverse_tunnel_enabled():
            return SyftError(message="Reverse tunneling is not enabled on this node.")
        elif reverse_tunnel:
            _rtunnel_route = self_node_peer.node_routes[-1]
            _rtunnel_route.rtunnel_token = generate_token()
            _rtunnel_route.host_or_ip = f"{self_node_peer.name}.syft.local"
            self_node_peer.node_routes[-1] = _rtunnel_route

        if isinstance(self_node_peer, SyftError):
            return self_node_peer

        # Step 2: Send the Node Peer to the remote node
        # Also give them their own to validate that it belongs to them
        # random challenge prevents replay attacks
        remote_client: SyftClient = remote_node_route.client_with_context(
            context=context
        )
        remote_node_peer = NodePeer.from_client(remote_client)

        # Step 3: Check remotely if the self node already exists as a peer
        # Update the peer if it exists, otherwise add it
        remote_self_node_peer = remote_client.api.services.network.get_peer_by_name(
            name=self_node_peer.name
        )

        association_request_approved = True
        if isinstance(remote_self_node_peer, NodePeer):
            updated_peer = NodePeerUpdate(
                id=self_node_peer.id, node_routes=self_node_peer.node_routes
            )
            result = remote_client.api.services.network.update_peer(
                peer_update=updated_peer
            )
            if isinstance(result, SyftError):
                logger.error(
                    f"Failed to update peer information on remote client. {result.message}"
                )
                return SyftError(
                    message=f"Failed to add peer information on remote client : {remote_client.id}"
                )

        # If  peer does not exist, ask the remote client to add this node
        # (represented by `self_node_peer`) as a peer
        if remote_self_node_peer is None:
            random_challenge = secrets.token_bytes(16)
            remote_res = remote_client.api.services.network.add_peer(
                peer=self_node_peer,
                challenge=random_challenge,
                self_node_route=remote_node_route,
                verify_key=remote_node_verify_key,
            )

            if isinstance(remote_res, SyftError):
                return SyftError(
                    message=f"Failed to add peer to remote client: {remote_client.id}. Error: {remote_res.message}"
                )

            association_request_approved = not isinstance(remote_res, Request)

        # Step 4: Save the remote peer for later
        result = self.stash.create_or_update_peer(
            context.node.verify_key,
            remote_node_peer,
        )
        if result.is_err():
            logging.error(
                f"Failed to save peer: {remote_node_peer}. Error: {result.err()}"
            )
            return SyftError(message="Failed to update route information.")

        # Step 5: Save config to enable reverse tunneling
        if reverse_tunnel and reverse_tunnel_enabled():
            self.set_reverse_tunnel_config(
                context=context,
                self_node_peer=self_node_peer,
                remote_node_peer=remote_node_peer,
            )

        return (
            SyftSuccess(message="Routes Exchanged")
            if association_request_approved
            else remote_res
        )

    @service_method(path="network.add_peer", name="add_peer", roles=GUEST_ROLE_LEVEL)
    def add_peer(
        self,
        context: AuthedServiceContext,
        peer: NodePeer,
    ) -> Request | SyftSuccess | SyftError:
        """Add a Network Node Peer. Called by a remote node to add
        itself as a peer for the current node.
        """

        # check if the peer already is a node peer
        existing_peer_res = self.stash.get_by_uid(context.node.verify_key, peer.id)
        if existing_peer_res.is_err():
            return SyftError(
                message=f"Failed to query peer from stash: {existing_peer_res.err()}"
            )

        if isinstance(existing_peer := existing_peer_res.ok(), NodePeer):
            msg = [
                f"The peer '{peer.name}' is already associated with '{context.node.name}'"
            ]

            if existing_peer != peer:
                result = self.stash.create_or_update_peer(
                    context.node.verify_key,
                    peer,
                )
                msg.append("Peer information change detected.")

                if result.is_err():
                    msg.append("Attempt to update peer information failed.")
                    return SyftError(message="\n".join(msg))

                msg.append("Peer information successfully updated.")
                return SyftSuccess(message="\n".join(msg))

            return SyftSuccess(message="\n".join(msg))

        # check if the peer already submitted an association request
        association_requests: list[Request] = self._get_association_requests_by_peer_id(
            context=context, peer_id=peer.id
        )
        if (
            association_requests
            and (association_request := association_requests[-1]).status
            == RequestStatus.PENDING
        ):
            return association_request
        # only create and submit a new request if there is no requests yet
        # or all previous requests have been rejected
        association_request_change = AssociationRequestChange(remote_peer=peer)
        submit_request = SubmitRequest(
            changes=[association_request_change],
            requesting_user_verify_key=context.credentials,
        )
        request_submit_method = context.node.get_service_method(RequestService.submit)
        request = request_submit_method(context, submit_request)
        if (
            isinstance(request, Request)
            and context.node.settings.association_request_auto_approval
        ):
            request_apply_method = context.node.get_service_method(RequestService.apply)
            return request_apply_method(context, uid=request.id)

        return request

    @service_method(
        path="network.challenge_nonce", name="challenge_nonce", roles=GUEST_ROLE_LEVEL
    )
    def challenge_nonce(
        self, context: AuthedServiceContext, challenge: bytes
    ) -> bytes | SyftError:
        """To check authenticity of the remote node"""

        # # Only the root user can ping the node to check its state
        # if context.node.verify_key != context.credentials:
        #     return SyftError(message=("Only the root user can access ping endpoint"))

        # this way they can match up who we are with who they think we are
        # Sending a signed messages for the peer to verify

        challenge_signature = context.node.signing_key.signing_key.sign(
            challenge
        ).signature

        return challenge_signature

    @service_method(path="network.ping", name="ping", roles=GUEST_ROLE_LEVEL)
    def ping(self, context: AuthedServiceContext) -> SyftSuccess:
        """To check liveness of the remote node"""
        return SyftSuccess(
            message=f"Reply from remote node:{context.node.name}-<{context.node.id}>"
        )

    @service_method(
        path="network.ping_peer",
        name="ping_peer",
        roles=GUEST_ROLE_LEVEL,
    )
    def ping_peer(
        self, context: AuthedServiceContext, verify_key: SyftVerifyKey
    ) -> SyftSuccess | SyftError:
        """Ping a remote node by its verify key"""

        remote_peer = self.stash.get_by_verify_key(
            credentials=context.node.verify_key, verify_key=verify_key
        )
        if remote_peer.is_err():
            return SyftError(
                message=f"Failed to query peer from stash. Err: {remote_peer}"
            )

        remote_peer = remote_peer.ok()
        if remote_peer is None:
            return SyftError(message=f"Peer not found with verify key: {verify_key}")
        try:
            remote_client = remote_peer.client_with_context(context=context)
            if remote_client.is_err():
                return SyftError(
                    message=f"Failed to create remote client for peer: {remote_peer}. Error: {remote_client.err()}"
                )
            remote_client = remote_client.ok()
            return remote_client.api.services.network.ping()
        except Exception as e:
            return SyftError(
                message=f"Cannot Ping Remote Peer: {remote_peer}. Error: {e}"
            )

    @service_method(
        path="network.check_peer_association",
        name="check_peer_association",
        roles=GUEST_ROLE_LEVEL,
    )
    def check_peer_association(
        self, context: AuthedServiceContext, peer_id: UID
    ) -> NodePeerAssociationStatus | SyftError:
        """Check if a peer exists in the network stash"""

        # get the node peer for the given sender peer_id
        peer = self.stash.get_by_uid(context.node.verify_key, peer_id)
        if err := peer.is_err():
            return SyftError(message=f"Failed to query peer from stash. Err: {err}")

        if isinstance(peer.ok(), NodePeer):
            return NodePeerAssociationStatus.PEER_ASSOCIATED

        if peer.ok() is None:  # peer is either pending or not found
            association_requests: list[Request] = (
                self._get_association_requests_by_peer_id(
                    context=context, peer_id=peer_id
                )
            )
            if (
                association_requests
                and association_requests[-1].status == RequestStatus.PENDING
            ):
                return NodePeerAssociationStatus.PEER_ASSOCIATION_PENDING

        return NodePeerAssociationStatus.PEER_NOT_FOUND

    @service_method(
        path="network.get_all_peers", name="get_all_peers", roles=GUEST_ROLE_LEVEL
    )
    def get_all_peers(
        self, context: AuthedServiceContext
    ) -> list[NodePeer] | SyftError:
        """Get all Peers"""

        result = self.stash.get_all(
            credentials=context.node.verify_key,
            order_by=OrderByNamePartitionKey,
        )
        if result.is_ok():
            peers = result.ok()
            return peers
        return SyftError(message=result.err())

    @service_method(
        path="network.get_peer_by_name", name="get_peer_by_name", roles=GUEST_ROLE_LEVEL
    )
    def get_peer_by_name(
        self, context: AuthedServiceContext, name: str
    ) -> NodePeer | None | SyftError:
        """Get Peer by Name"""

        result = self.stash.get_by_name(
            credentials=context.node.verify_key,
            name=name,
        )
        if result.is_ok():
            peer = result.ok()
            return peer
        return SyftError(message=str(result.err()))

    @service_method(
        path="network.get_peers_by_type",
        name="get_peers_by_type",
        roles=GUEST_ROLE_LEVEL,
    )
    def get_peers_by_type(
        self, context: AuthedServiceContext, node_type: NodeType
    ) -> list[NodePeer] | SyftError:
        result = self.stash.get_by_node_type(
            credentials=context.node.verify_key,
            node_type=node_type,
        )

        if result.is_err():
            return SyftError(message=str(result.err()))

        # Return peers or an empty list when result is None
        return result.ok() or []

    @service_method(
        path="network.update_peer",
        name="update_peer",
        roles=GUEST_ROLE_LEVEL,
    )
    def update_peer(
        self,
        context: AuthedServiceContext,
        peer_update: NodePeerUpdate,
    ) -> SyftSuccess | SyftError:
        # try setting all fields of NodePeerUpdate according to NodePeer

        result = self.stash.update(
            credentials=context.node.verify_key,
            peer_update=peer_update,
        )
        if result.is_err():
            return SyftError(
                message=f"Failed to update peer '{peer_update.name}'. Error: {result.err()}"
            )

        peer = result.ok()

        self.set_reverse_tunnel_config(context=context, remote_node_peer=peer)
        return SyftSuccess(
            message=f"Peer '{result.ok().name}' information successfully updated."
        )

    def set_reverse_tunnel_config(
        self,
        context: AuthedServiceContext,
        remote_node_peer: NodePeer,
        self_node_peer: NodePeer | None = None,
    ) -> None:
        node_type = cast(NodeType, context.node.node_type)
        if node_type.value == NodeType.GATEWAY.value:
            rtunnel_route = remote_node_peer.get_rtunnel_route()
            (
                self.rtunnel_service.set_server_config(remote_node_peer)
                if rtunnel_route
                else None
            )
        else:
            self_node_peer = (
                context.node.settings.to(NodePeer)
                if self_node_peer is None
                else self_node_peer
            )
            rtunnel_route = self_node_peer.get_rtunnel_route()
            (
                self.rtunnel_service.set_client_config(
                    self_node_peer=self_node_peer,
                    remote_node_route=remote_node_peer.pick_highest_priority_route(),
                )
                if rtunnel_route
                else None
            )

    @service_method(
        path="network.delete_peer_by_id",
        name="delete_peer_by_id",
        roles=DATA_OWNER_ROLE_LEVEL,
    )
    def delete_peer_by_id(
        self, context: AuthedServiceContext, uid: UID
    ) -> SyftSuccess | SyftError:
        """Delete Node Peer"""
        retrieve_result = self.stash.get_by_uid(context.credentials, uid)
        if err := retrieve_result.is_err():
            return SyftError(
                message=f"Failed to retrieve peer with UID {uid}: {retrieve_result.err()}."
            )
        peer_to_delete = cast(NodePeer, retrieve_result.ok())

        node_side_type = cast(NodeType, context.node.node_type)
        if node_side_type.value == NodeType.GATEWAY.value:
            rtunnel_route = peer_to_delete.get_rtunnel_route()
            (
                self.rtunnel_service.clear_server_config(peer_to_delete)
                if rtunnel_route
                else None
            )

        # TODO: Handle the case when peer is deleted from domain node

        result = self.stash.delete_by_uid(context.credentials, uid)
        if err := result.is_err():
            return SyftError(message=f"Failed to delete peer with UID {uid}: {err}.")
        # Delete all the association requests from this peer
        association_requests: list[Request] = self._get_association_requests_by_peer_id(
            context=context, peer_id=uid
        )
        for request in association_requests:
            request_delete_method = context.node.get_service_method(
                RequestService.delete_by_uid
            )
            res = request_delete_method(context, request.id)
            if isinstance(res, SyftError):
                return res
        # TODO: Notify the peer (either by email or by other form of notifications)
        # that it has been deleted from the network
        return SyftSuccess(message=f"Node Peer with id {uid} deleted.")

    @service_method(path="network.add_route_on_peer", name="add_route_on_peer")
    def add_route_on_peer(
        self,
        context: AuthedServiceContext,
        peer: NodePeer,
        route: NodeRoute,
    ) -> SyftSuccess | SyftError:
        """
        Add or update the route information on the remote peer.

        Args:
            context (AuthedServiceContext): The authentication context.
            peer (NodePeer): The peer representing the remote node.
            route (NodeRoute): The route to be added.

        Returns:
            SyftSuccess | SyftError: A success message if the route is verified,
                otherwise an error message.
        """
        # creates a client on the remote node based on the credentials
        # of the current node's client
        remote_client = peer.client_with_context(context=context)
        if remote_client.is_err():
            return SyftError(
                message=f"Failed to create remote client for peer: "
                f"{peer.id}. Error: {remote_client.err()}"
            )
        remote_client = remote_client.ok()
        # ask the remote node to add the route to the self node
        result = remote_client.api.services.network.add_route(
            peer_verify_key=context.credentials,
            route=route,
            called_by_peer=True,
        )
        return result

    @service_method(path="network.add_route", name="add_route", roles=GUEST_ROLE_LEVEL)
    def add_route(
        self,
        context: AuthedServiceContext,
        peer_verify_key: SyftVerifyKey,
        route: NodeRoute,
        called_by_peer: bool = False,
    ) -> SyftSuccess | SyftError:
        """
        Add a route to the peer. If the route already exists, update its priority.

        Args:
            context (AuthedServiceContext): The authentication context of the remote node.
            peer_verify_key (SyftVerifyKey): The verify key of the remote node peer.
            route (NodeRoute): The route to be added.
            called_by_peer (bool): The flag to indicate that it's called by a remote peer.

        Returns:
            SyftSuccess | SyftError
        """
        # verify if the peer is truly the one sending the request to add the route to itself
        if called_by_peer and peer_verify_key != context.credentials:
            return SyftError(
                message=(
                    f"The {type(peer_verify_key).__name__}: "
                    f"{peer_verify_key} does not match the signature of the message"
                )
            )
        # get the full peer object from the store to update its routes
        remote_node_peer: NodePeer | SyftError = (
            self._get_remote_node_peer_by_verify_key(context, peer_verify_key)
        )
        if isinstance(remote_node_peer, SyftError):
            return remote_node_peer
        # add and update the priority for the peer
        if route in remote_node_peer.node_routes:
            return SyftSuccess(
                message=f"The route already exists between '{context.node.name}' and "
                f"peer '{remote_node_peer.name}'."
            )

        remote_node_peer.update_route(route=route)
        # update the peer in the store with the updated routes
        peer_update = NodePeerUpdate(
            id=remote_node_peer.id, node_routes=remote_node_peer.node_routes
        )
        result = self.stash.update(
            credentials=context.node.verify_key,
            peer_update=peer_update,
        )
        if result.is_err():
            return SyftError(message=str(result.err()))
        return SyftSuccess(
            message=f"New route ({str(route)}) with id '{route.id}' "
            f"to peer {remote_node_peer.node_type.value} '{remote_node_peer.name}' "
            f"was added for {str(context.node.node_type)} '{context.node.name}'"
        )

    @service_method(path="network.delete_route_on_peer", name="delete_route_on_peer")
    def delete_route_on_peer(
        self,
        context: AuthedServiceContext,
        peer: NodePeer,
        route: NodeRoute,
    ) -> SyftSuccess | SyftError | SyftInfo:
        """
        Delete the route on the remote peer.

        Args:
            context (AuthedServiceContext): The authentication context for the service.
            peer (NodePeer): The peer for which the route will be deleted.
            route (NodeRoute): The route to be deleted.

        Returns:
            SyftSuccess: If the route is successfully deleted.
            SyftError: If there is an error deleting the route.
            SyftInfo: If there is only one route left for the peer and
                the admin chose not to remove it
        """
        # creates a client on the remote node based on the credentials
        # of the current node's client
        remote_client = peer.client_with_context(context=context)
        if remote_client.is_err():
            return SyftError(
                message=f"Failed to create remote client for peer: "
                f"{peer.id}. Error: {remote_client.err()}"
            )
        remote_client = remote_client.ok()
        # ask the remote node to delete the route to the self node,
        result = remote_client.api.services.network.delete_route(
            peer_verify_key=context.credentials,
            route=route,
            called_by_peer=True,
        )
        return result

    @service_method(
        path="network.delete_route", name="delete_route", roles=GUEST_ROLE_LEVEL
    )
    def delete_route(
        self,
        context: AuthedServiceContext,
        peer_verify_key: SyftVerifyKey,
        route: NodeRoute | None = None,
        called_by_peer: bool = False,
    ) -> SyftSuccess | SyftError | SyftInfo:
        """
        Delete a route for a given peer.
        If a peer has no routes left, there will be a prompt asking if the user want to remove it.
        If the answer is yes, it will be removed from the stash and will no longer be a peer.

        Args:
            context (AuthedServiceContext): The authentication context for the service.
            peer_verify_key (SyftVerifyKey): The verify key of the remote node peer.
            route (NodeRoute): The route to be deleted.
            called_by_peer (bool): The flag to indicate that it's called by a remote peer.

        Returns:
            SyftSuccess: If the route is successfully deleted.
            SyftError: If there is an error deleting the route.
            SyftInfo: If there is only one route left for the peer and
                the admin chose not to remove it
        """
        if called_by_peer and peer_verify_key != context.credentials:
            # verify if the peer is truly the one sending the request to delete the route to itself
            return SyftError(
                message=(
                    f"The {type(peer_verify_key).__name__}: "
                    f"{peer_verify_key} does not match the signature of the message"
                )
            )

        remote_node_peer: NodePeer | SyftError = (
            self._get_remote_node_peer_by_verify_key(
                context=context, peer_verify_key=peer_verify_key
            )
        )

        if len(remote_node_peer.node_routes) == 1:
            warning_message = (
                f"There is only one route left to peer "
                f"{remote_node_peer.node_type.value} '{remote_node_peer.name}'. "
                f"Removing this route will remove the peer for "
                f"{str(context.node.node_type)} '{context.node.name}'."
            )
            response: bool = prompt_warning_message(
                message=warning_message,
                confirm=False,
            )
            if not response:
                return SyftInfo(
                    message=f"The last route to {remote_node_peer.node_type.value} "
                    f"'{remote_node_peer.name}' with id "
                    f"'{remote_node_peer.node_routes[0].id}' was not deleted."
                )

        result = remote_node_peer.delete_route(route=route)
        return_message = (
            f"Route '{str(route)}' to peer "
            f"{remote_node_peer.node_type.value} '{remote_node_peer.name}' "
            f"was deleted for {str(context.node.node_type)} '{context.node.name}'."
        )
        if isinstance(result, SyftError):
            return result

        if len(remote_node_peer.node_routes) == 0:
            # remove the peer
            # TODO: should we do this as we are deleting the peer with a guest role level?
            result = self.stash.delete_by_uid(
                credentials=context.node.verify_key, uid=remote_node_peer.id
            )
            if isinstance(result, SyftError):
                return result
            return_message += (
                f" There is no routes left to connect to peer "
                f"{remote_node_peer.node_type.value} '{remote_node_peer.name}', so it is deleted for "
                f"{str(context.node.node_type)} '{context.node.name}'."
            )
        else:
            # update the peer with the route removed
            peer_update = NodePeerUpdate(
                id=remote_node_peer.id, node_routes=remote_node_peer.node_routes
            )
            result = self.stash.update(
                credentials=context.node.verify_key, peer_update=peer_update
            )
            if result.is_err():
                return SyftError(message=str(result.err()))

        return SyftSuccess(message=return_message)

    @service_method(
        path="network.update_route_priority_on_peer",
        name="update_route_priority_on_peer",
    )
    def update_route_priority_on_peer(
        self,
        context: AuthedServiceContext,
        peer: NodePeer,
        route: NodeRoute,
        priority: int | None = None,
    ) -> SyftSuccess | SyftError:
        """
        Update the route priority on the remote peer.

        Args:
            context (AuthedServiceContext): The authentication context.
            peer (NodePeer): The peer representing the remote node.
            route (NodeRoute): The route to be added.
            priority (int | None): The new priority value for the route. If not
                provided, it will be assigned the highest priority among all peers

        Returns:
            SyftSuccess | SyftError: A success message if the route is verified,
                otherwise an error message.
        """
        # creates a client on the remote node based on the credentials
        # of the current node's client
        remote_client = peer.client_with_context(context=context)
        if remote_client.is_err():
            return SyftError(
                message=f"Failed to create remote client for peer: "
                f"{peer.id}. Error: {remote_client.err()}"
            )
        remote_client = remote_client.ok()
        result = remote_client.api.services.network.update_route_priority(
            peer_verify_key=context.credentials,
            route=route,
            priority=priority,
            called_by_peer=True,
        )
        return result

    @service_method(
        path="network.update_route_priority",
        name="update_route_priority",
        roles=GUEST_ROLE_LEVEL,
    )
    def update_route_priority(
        self,
        context: AuthedServiceContext,
        peer_verify_key: SyftVerifyKey,
        route: NodeRoute,
        priority: int | None = None,
        called_by_peer: bool = False,
    ) -> SyftSuccess | SyftError:
        """
        Updates a route's priority for the given peer

        Args:
            context (AuthedServiceContext): The authentication context for the service.
            peer_verify_key (SyftVerifyKey): The verify key of the peer whose route priority needs to be updated.
            route (NodeRoute): The route for which the priority needs to be updated.
            priority (int | None): The new priority value for the route. If not
                provided, it will be assigned the highest priority among all peers

        Returns:
            SyftSuccess | SyftError: Successful / Error response
        """
        if called_by_peer and peer_verify_key != context.credentials:
            return SyftError(
                message=(
                    f"The {type(peer_verify_key).__name__}: "
                    f"{peer_verify_key} does not match the signature of the message"
                )
            )
        # get the full peer object from the store to update its routes
        remote_node_peer: NodePeer | SyftError = (
            self._get_remote_node_peer_by_verify_key(context, peer_verify_key)
        )
        if isinstance(remote_node_peer, SyftError):
            return remote_node_peer
        # update the route's priority for the peer
        updated_node_route: NodeRouteType | SyftError = (
            remote_node_peer.update_existed_route_priority(
                route=route, priority=priority
            )
        )
        if isinstance(updated_node_route, SyftError):
            return updated_node_route
        new_priority: int = updated_node_route.priority
        # update the peer in the store
        peer_update = NodePeerUpdate(
            id=remote_node_peer.id, node_routes=remote_node_peer.node_routes
        )
        result = self.stash.update(context.node.verify_key, peer_update)
        if result.is_err():
            return SyftError(message=str(result.err()))

        return SyftSuccess(
            message=f"Route {route.id}'s priority updated to "
            f"{new_priority} for peer {remote_node_peer.name}"
        )

    def _get_remote_node_peer_by_verify_key(
        self, context: AuthedServiceContext, peer_verify_key: SyftVerifyKey
    ) -> NodePeer | SyftError:
        """
        Helper function to get the full node peer object from t
        he stash using its verify key
        """
        remote_node_peer: Result[NodePeer | None, SyftError] = (
            self.stash.get_by_verify_key(
                credentials=context.node.verify_key,
                verify_key=peer_verify_key,
            )
        )
        if remote_node_peer.is_err():
            return SyftError(message=str(remote_node_peer.err()))
        remote_node_peer = remote_node_peer.ok()
        if remote_node_peer is None:
            return SyftError(
                message=f"Can't retrieve {remote_node_peer.name} from the store of peers (None)."
            )
        return remote_node_peer

    def _get_association_requests_by_peer_id(
        self, context: AuthedServiceContext, peer_id: UID
    ) -> list[Request]:
        """
        Get all the association requests from a peer. The association requests are sorted by request_time.
        """
        request_get_all_method: Callable = context.node.get_service_method(
            RequestService.get_all
        )
        all_requests: list[Request] = request_get_all_method(context)
        association_requests: list[Request] = [
            request
            for request in all_requests
            if any(
                isinstance(change, AssociationRequestChange)
                and change.remote_peer.id == peer_id
                for change in request.changes
            )
        ]

        return sorted(
            association_requests, key=lambda request: request.request_time.utc_timestamp
        )


TYPE_TO_SERVICE[NodePeer] = NetworkService
SERVICE_TO_TYPES[NetworkService].update({NodePeer})


def from_grid_url(context: TransformContext) -> TransformContext:
    if context.obj is not None and context.output is not None:
        url = context.obj.url.as_container_host()
        context.output["host_or_ip"] = url.host_or_ip
        context.output["protocol"] = url.protocol
        context.output["port"] = url.port
        context.output["private"] = False
        context.output["proxy_target_uid"] = context.obj.proxy_target_uid
        context.output["priority"] = 1
        context.output["rtunnel_token"] = context.obj.rtunnel_token

    return context


@transform(HTTPConnection, HTTPNodeRoute)
def http_connection_to_node_route() -> list[Callable]:
    return [from_grid_url]


def get_python_node_route(context: TransformContext) -> TransformContext:
    if context.output is not None and context.obj is not None:
        context.output["id"] = context.obj.node.id
        context.output["worker_settings"] = WorkerSettings.from_node(context.obj.node)
        context.output["proxy_target_uid"] = context.obj.proxy_target_uid
    return context


@transform(PythonConnection, PythonNodeRoute)
def python_connection_to_node_route() -> list[Callable]:
    return [get_python_node_route]


@transform_method(PythonNodeRoute, PythonConnection)
def node_route_to_python_connection(
    obj: Any, context: TransformContext | None = None
) -> list[Callable]:
    return PythonConnection(node=obj.node, proxy_target_uid=obj.proxy_target_uid)


@transform_method(HTTPNodeRoute, HTTPConnection)
def node_route_to_http_connection(
    obj: Any, context: TransformContext | None = None
) -> list[Callable]:
    url = GridURL(
        protocol=obj.protocol, host_or_ip=obj.host_or_ip, port=obj.port
    ).as_container_host()
    return HTTPConnection(
        url=url,
        proxy_target_uid=obj.proxy_target_uid,
        rtunnel_token=obj.rtunnel_token,
    )


@transform(NodeMetadata, NodePeer)
def metadata_to_peer() -> list[Callable]:
    return [
        keep(
            [
                "id",
                "name",
                "verify_key",
                "node_type",
            ]
        ),
        make_set_default("admin_email", ""),
    ]


@transform(NodeSettings, NodePeer)
def settings_to_peer() -> list[Callable]:
    return [
        keep(["id", "name", "verify_key", "node_type", "admin_email"]),
    ]
