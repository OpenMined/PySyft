# stdlib
from collections.abc import Callable
from enum import Enum
import logging
import secrets
from typing import Any

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
from ...store.document_store import BaseUIDStoreStash
from ...store.document_store import DocumentStore
from ...store.document_store import PartitionKey
from ...store.document_store import PartitionSettings
from ...store.document_store import QueryKeys
from ...types.grid_url import GridURL
from ...types.transforms import TransformContext
from ...types.transforms import keep
from ...types.transforms import make_set_default
from ...types.transforms import transform
from ...types.transforms import transform_method
from ...types.uid import UID
from ...util.telemetry import instrument
from ...util.util import prompt_warning_message
from ..context import AuthedServiceContext
from ..data_subject.data_subject import NamePartitionKey
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
from .node_peer import NodePeer
from .node_peer import NodePeerUpdate
from .routes import HTTPNodeRoute
from .routes import NodeRoute
from .routes import NodeRouteType
from .routes import PythonNodeRoute

logger = logging.getLogger(__name__)

VerifyKeyPartitionKey = PartitionKey(key="verify_key", type_=SyftVerifyKey)
NodeTypePartitionKey = PartitionKey(key="node_type", type_=NodeType)
OrderByNamePartitionKey = PartitionKey(key="name", type_=str)


@serializable()
class NodePeerAssociationStatus(Enum):
    PEER_ASSOCIATED = "PEER_ASSOCIATED"
    PEER_ASSOCIATION_PENDING = "PEER_ASSOCIATION_PENDING"
    PEER_NOT_FOUND = "PEER_NOT_FOUND"


@instrument
@serializable()
class NetworkStash(BaseUIDStoreStash):
    object_type = NodePeer
    settings: PartitionSettings = PartitionSettings(
        name=NodePeer.__canonical_name__, object_type=NodePeer
    )

    def __init__(self, store: DocumentStore) -> None:
        super().__init__(store=store)

    def get_by_name(
        self, credentials: SyftVerifyKey, name: str
    ) -> Result[NodePeer | None, str]:
        qks = QueryKeys(qks=[NamePartitionKey.with_obj(name)])
        return self.query_one(credentials=credentials, qks=qks)

    def update(
        self,
        credentials: SyftVerifyKey,
        peer_update: NodePeerUpdate,
        has_permission: bool = False,
    ) -> Result[NodePeer, str]:
        valid = self.check_type(peer_update, NodePeerUpdate)
        if valid.is_err():
            return SyftError(message=valid.err())
        return super().update(credentials, peer_update, has_permission=has_permission)

    def create_or_update_peer(
        self, credentials: SyftVerifyKey, peer: NodePeer
    ) -> Result[NodePeer, str]:
        """
        Update the selected peer and its route priorities if the peer already exists
        If the peer does not exist, simply adds it to the database.

        Args:
            credentials (SyftVerifyKey): The credentials used to authenticate the request.
            peer (NodePeer): The peer to be updated or added.

        Returns:
            Result[NodePeer, str]: The updated or added peer if the operation
            was successful, or an error message if the operation failed.
        """
        valid = self.check_type(peer, NodePeer)
        if valid.is_err():
            return SyftError(message=valid.err())

        existing = self.get_by_uid(credentials=credentials, uid=peer.id)
        if existing.is_ok() and existing.ok() is not None:
            existing_peer: NodePeer = existing.ok()
            existing_peer.update_routes(peer.node_routes)
            peer_update = NodePeerUpdate(
                id=peer.id, node_routes=existing_peer.node_routes
            )
            result = self.update(credentials, peer_update)
            return result
        else:
            result = self.set(credentials, peer)
            return result

    def get_by_verify_key(
        self, credentials: SyftVerifyKey, verify_key: SyftVerifyKey
    ) -> Result[NodePeer | None, SyftError]:
        qks = QueryKeys(qks=[VerifyKeyPartitionKey.with_obj(verify_key)])
        return self.query_one(credentials, qks)

    def get_by_node_type(
        self, credentials: SyftVerifyKey, node_type: NodeType
    ) -> Result[list[NodePeer], SyftError]:
        qks = QueryKeys(qks=[NodeTypePartitionKey.with_obj(node_type)])
        return self.query_all(
            credentials=credentials, qks=qks, order_by=OrderByNamePartitionKey
        )


@instrument
@serializable()
class NetworkService(AbstractService):
    store: DocumentStore
    stash: NetworkStash

    def __init__(self, store: DocumentStore) -> None:
        self.store = store
        self.stash = NetworkStash(store=store)

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
    ) -> Request | SyftSuccess | SyftError:
        """
        Exchange Route With Another Node. If there is a pending association request, return it
        """

        # Step 1: Validate the Route
        self_node_peer = self_node_route.validate_with_context(context=context)

        if isinstance(self_node_peer, SyftError):
            return self_node_peer

        # Step 2: Send the Node Peer to the remote node
        # Also give them their own to validate that it belongs to them
        # random challenge prevents replay attacks
        remote_client: SyftClient = remote_node_route.client_with_context(
            context=context
        )
        remote_node_peer = NodePeer.from_client(remote_client)

        # check locally if the remote node already exists as a peer
        existing_peer_result = self.stash.get_by_uid(
            context.node.verify_key, remote_node_peer.id
        )
        if (
            existing_peer_result.is_ok()
            and (existing_peer := existing_peer_result.ok()) is not None
        ):
            logger.info(
                f"{remote_node_peer.node_type} '{remote_node_peer.name}' already exist as a peer for "
                f"{self_node_peer.node_type} '{self_node_peer.name}'."
            )

            if existing_peer != remote_node_peer:
                result = self.stash.create_or_update_peer(
                    context.node.verify_key,
                    remote_node_peer,
                )
                if result.is_err():
                    return SyftError(
                        message=f"Failed to update peer: {remote_node_peer.name} information."
                    )
                logger.info(
                    f"{existing_peer.node_type} peer '{existing_peer.name}' information successfully updated."
                )

            # Also check remotely if the self node already exists as a peer
            remote_self_node_peer = remote_client.api.services.network.get_peer_by_name(
                name=self_node_peer.name
            )
            if isinstance(remote_self_node_peer, NodePeer):
                logger.info(
                    f"{self_node_peer.node_type} '{self_node_peer.name}' already exist "
                    f"as a peer for {remote_node_peer.node_type} '{remote_node_peer.name}'."
                )
                if remote_self_node_peer != self_node_peer:
                    updated_peer = NodePeerUpdate(
                        id=self_node_peer.id, node_routes=self_node_peer.node_routes
                    )
                    result = remote_client.api.services.network.update_peer(
                        peer_update=updated_peer
                    )
                    logger.info(
                        f"{self_node_peer.node_type} peer '{self_node_peer.name}' information change detected."
                    )
                    if isinstance(result, SyftError):
                        logger.error(
                            f"Attempt to remotely update {self_node_peer.node_type} peer "
                            f"'{self_node_peer.name}' information remotely failed. Error: {result.message}"
                        )
                        return SyftError(message="Failed to update peer information.")

                    logger.info(
                        f"{self_node_peer.node_type} peer '{self_node_peer.name}' "
                        f"information successfully updated."
                    )
                msg = (
                    f"Routes between {remote_node_peer.node_type} '{remote_node_peer.name}' and "
                    f"{self_node_peer.node_type} '{self_node_peer.name}' already exchanged."
                )
                return SyftSuccess(message="\n".join(msg))

        # If  peer does not exist, ask the remote client to add this node
        # (represented by `self_node_peer`) as a peer
        random_challenge = secrets.token_bytes(16)
        remote_res = remote_client.api.services.network.add_peer(
            peer=self_node_peer,
            challenge=random_challenge,
            self_node_route=remote_node_route,
            verify_key=remote_node_verify_key,
        )

        if isinstance(remote_res, SyftError):
            return remote_res

        association_request_approved = not isinstance(remote_res, Request)

        # save the remote peer for later
        result = self.stash.create_or_update_peer(
            context.node.verify_key,
            remote_node_peer,
        )
        if result.is_err():
            return SyftError(message="Failed to update route information.")

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
        challenge: bytes,
        self_node_route: NodeRoute,
        verify_key: SyftVerifyKey,
    ) -> Request | SyftSuccess | SyftError:
        """Add a Network Node Peer. Called by a remote node to add
        itself as a peer for the current node.
        """
        # Using the verify_key of the peer to verify the signature
        # It is also our single source of truth for the peer
        if peer.verify_key != context.credentials:
            return SyftError(
                message=(
                    f"The {type(peer).__name__}.verify_key: "
                    f"{peer.verify_key} does not match the signature of the message"
                )
            )

        if verify_key != context.node.verify_key:
            return SyftError(
                message="verify_key does not match the remote node's verify_key for add_peer"
            )

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
        association_request_change = AssociationRequestChange(
            self_node_route=self_node_route, challenge=challenge, remote_peer=peer
        )
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

    @service_method(path="network.ping", name="ping", roles=GUEST_ROLE_LEVEL)
    def ping(
        self, context: AuthedServiceContext, challenge: bytes
    ) -> bytes | SyftError:
        """To check alivesness/authenticity of a peer"""

        # # Only the root user can ping the node to check its state
        # if context.node.verify_key != context.credentials:
        #     return SyftError(message=("Only the root user can access ping endpoint"))

        # this way they can match up who we are with who they think we are
        # Sending a signed messages for the peer to verify

        challenge_signature = context.node.signing_key.signing_key.sign(
            challenge
        ).signature

        return challenge_signature

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
        return SyftSuccess(
            message=f"Peer '{result.ok().name}' information successfully updated."
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
        existed_route: NodeRoute | None = remote_node_peer.update_route(route)
        if existed_route:
            return SyftSuccess(
                message=f"The route already exists between '{context.node.name}' and "
                f"peer '{remote_node_peer.name}' with id '{existed_route.id}'."
            )
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
        route: NodeRoute | None = None,
        route_id: UID | None = None,
    ) -> SyftSuccess | SyftError | SyftInfo:
        """
        Delete the route on the remote peer.

        Args:
            context (AuthedServiceContext): The authentication context for the service.
            peer (NodePeer): The peer for which the route will be deleted.
            route (NodeRoute): The route to be deleted.
            route_id (UID): The UID of the route to be deleted.

        Returns:
            SyftSuccess: If the route is successfully deleted.
            SyftError: If there is an error deleting the route.
            SyftInfo: If there is only one route left for the peer and
                the admin chose not to remove it
        """
        if route is None and route_id is None:
            return SyftError(
                message="Either `route` or `route_id` arg must be provided"
            )

        if route and route_id and route.id != route_id:
            return SyftError(
                message=f"Both `route` and `route_id` are provided, but "
                f"route's id ({route.id}) and route_id ({route_id}) do not match"
            )

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
            route_id=route_id,
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
        route_id: UID | None = None,
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
            route_id (UID): The UID of the route to be deleted.
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

        if route:
            result = remote_node_peer.delete_route(route=route)
            return_message = (
                f"Route '{str(route)}' with id '{route.id}' to peer "
                f"{remote_node_peer.node_type.value} '{remote_node_peer.name}' "
                f"was deleted for {str(context.node.node_type)} '{context.node.name}'."
            )
        if route_id:
            result = remote_node_peer.delete_route(route_id=route_id)
            return_message = (
                f"Route with id '{route_id}' to peer "
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
                message=f"Can't retrive {remote_node_peer.name} from the store of peers (None)."
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
    return HTTPConnection(url=url, proxy_target_uid=obj.proxy_target_uid)


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
