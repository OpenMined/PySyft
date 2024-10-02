# stdlib
from collections.abc import Callable
from enum import Enum
import logging
import secrets
from typing import Any
from typing import cast

# relative
from ...abstract_server import ServerType
from ...client.client import HTTPConnection
from ...client.client import PythonConnection
from ...client.client import SyftClient
from ...serde.serializable import serializable
from ...server.credentials import SyftVerifyKey
from ...server.worker_settings import WorkerSettings
from ...service.settings.settings import ServerSettings
from ...store.db.db import DBManager
from ...store.db.stash import ObjectStash
from ...store.document_store_errors import NotFoundException
from ...store.document_store_errors import StashException
from ...types.errors import SyftException
from ...types.result import as_result
from ...types.server_url import ServerURL
from ...types.transforms import TransformContext
from ...types.transforms import keep
from ...types.transforms import make_set_default
from ...types.transforms import transform
from ...types.transforms import transform_method
from ...types.uid import UID
from ...util.util import generate_token
from ...util.util import get_env
from ...util.util import prompt_warning_message
from ...util.util import str_to_bool
from ..context import AuthedServiceContext
from ..metadata.server_metadata import ServerMetadata
from ..request.request import Request
from ..request.request import RequestStatus
from ..request.request import SubmitRequest
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
from .reverse_tunnel_service import ReverseTunnelService
from .routes import HTTPServerRoute
from .routes import PythonServerRoute
from .routes import ServerRoute
from .routes import ServerRouteType
from .server_peer import ServerPeer
from .server_peer import ServerPeerUpdate

logger = logging.getLogger(__name__)

REVERSE_TUNNEL_ENABLED = "REVERSE_TUNNEL_ENABLED"


def reverse_tunnel_enabled() -> bool:
    return str_to_bool(get_env(REVERSE_TUNNEL_ENABLED, "false"))


@serializable(canonical_name="ServerPeerAssociationStatus", version=1)
class ServerPeerAssociationStatus(Enum):
    PEER_ASSOCIATED = "PEER_ASSOCIATED"
    PEER_ASSOCIATION_PENDING = "PEER_ASSOCIATION_PENDING"
    PEER_NOT_FOUND = "PEER_NOT_FOUND"


@serializable(canonical_name="NetworkSQLStash", version=1)
class NetworkStash(ObjectStash[ServerPeer]):
    @as_result(StashException, NotFoundException)
    def get_by_name(self, credentials: SyftVerifyKey, name: str) -> ServerPeer:
        try:
            return self.get_one(
                credentials=credentials,
                filters={"name": name},
            ).unwrap()
        except NotFoundException as e:
            raise NotFoundException.from_exception(
                e, public_message=f"ServerPeer with {name} not found"
            )

    @as_result(StashException)
    def create_or_update_peer(
        self, credentials: SyftVerifyKey, peer: ServerPeer
    ) -> ServerPeer:
        """
        Update the selected peer and its route priorities if the peer already exists
        If the peer does not exist, simply adds it to the database.

        Args:
            credentials (SyftVerifyKey): The credentials used to authenticate the request.
            peer (ServerPeer): The peer to be updated or added.

        Returns:
            ServerPeer: The updated or added peer if the operation was successful.
                Raises an exception if the operation failed.
        """
        self.check_type(peer, ServerPeer).unwrap()

        try:
            existing_peer: ServerPeer = self.get_by_uid(
                credentials=credentials, uid=peer.id
            ).unwrap()
        except SyftException:
            return self.set(credentials, obj=peer).unwrap()

        existing_peer.update_routes(peer.server_routes)
        peer_update = ServerPeerUpdate(
            id=peer.id, server_routes=existing_peer.server_routes
        )
        return self.update(credentials, peer_update).unwrap()

    @as_result(StashException, NotFoundException)
    def get_by_verify_key(
        self, credentials: SyftVerifyKey, verify_key: SyftVerifyKey
    ) -> ServerPeer:
        return self.get_one(
            credentials=credentials,
            filters={"verify_key": verify_key},
        ).unwrap()

    @as_result(StashException)
    def get_by_server_type(
        self, credentials: SyftVerifyKey, server_type: ServerType
    ) -> list[ServerPeer]:
        return self.get_all(
            credentials=credentials,
            filters={"server_type": server_type},
        ).unwrap()


@serializable(canonical_name="NetworkService", version=1)
class NetworkService(AbstractService):
    stash: NetworkStash

    def __init__(self, store: DBManager) -> None:
        self.stash = NetworkStash(store=store)
        if reverse_tunnel_enabled():
            self.rtunnel_service = ReverseTunnelService()

    @service_method(
        path="network.exchange_credentials_with",
        name="exchange_credentials_with",
        roles=GUEST_ROLE_LEVEL,
        warning=CRUDWarning(confirmation=True),
        unwrap_on_success=False,
    )
    def exchange_credentials_with(
        self,
        context: AuthedServiceContext,
        self_server_route: ServerRoute,
        remote_server_route: ServerRoute,
        remote_server_verify_key: SyftVerifyKey,
        reverse_tunnel: bool = False,
    ) -> Request | SyftSuccess:
        """
        Exchange Route With Another Server. If there is a pending association request, return it
        """

        # Step 1: Validate the Route
        self_server_peer = self_server_route.validate_with_context(
            context=context
        ).unwrap()

        if reverse_tunnel and not reverse_tunnel_enabled():
            raise SyftException(
                public_message="Reverse tunneling is not enabled on this server."
            )

        elif reverse_tunnel:
            _rtunnel_route = self_server_peer.server_routes[-1]
            _rtunnel_route.rtunnel_token = generate_token()
            _rtunnel_route.host_or_ip = f"{self_server_peer.name}.syft.local"
            self_server_peer.server_routes[-1] = _rtunnel_route

        # Step 2: Send the Server Peer to the remote server
        # Also give them their own to validate that it belongs to them
        # random challenge prevents replay attacks
        remote_client: SyftClient = remote_server_route.client_with_context(
            context=context
        )
        remote_server_peer = ServerPeer.from_client(remote_client)

        # Step 3: Check remotely if the self server already exists as a peer
        # Update the peer if it exists, otherwise add it
        try:
            remote_self_server_peer = (
                remote_client.api.services.network.get_peer_by_name(
                    name=self_server_peer.name
                )
            )
        except SyftException:
            remote_self_server_peer = None

        association_request_approved = True
        if isinstance(remote_self_server_peer, ServerPeer):
            updated_peer = ServerPeerUpdate(
                id=self_server_peer.id, server_routes=self_server_peer.server_routes
            )
            try:
                remote_client.api.services.network.update_peer(peer_update=updated_peer)
            except Exception as e:
                logger.error(f"Failed to update peer information on remote client. {e}")
                raise SyftException.from_exception(
                    e,
                    public_message=f"Failed to add peer information on remote client : {remote_client.id}",
                    private_message=f"Error: {e}",
                )

        # If peer does not exist, ask the remote client to add this server
        # (represented by `self_server_peer`) as a peer
        if remote_self_server_peer is None:
            random_challenge = secrets.token_bytes(16)
            remote_res = remote_client.api.services.network.add_peer(
                peer=self_server_peer,
                challenge=random_challenge,
                self_server_route=remote_server_route,
                verify_key=remote_server_verify_key,
            )
            association_request_approved = not isinstance(remote_res, Request)

        # Step 4: Save the remote peer for later
        self.stash.create_or_update_peer(
            context.server.verify_key,
            remote_server_peer,
        ).unwrap(public_message=f"Failed to save peer: {remote_server_peer}.")
        # Step 5: Save config to enable reverse tunneling
        if reverse_tunnel and reverse_tunnel_enabled():
            self.set_reverse_tunnel_config(
                context=context,
                self_server_peer=self_server_peer,
                remote_server_peer=remote_server_peer,
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
        peer: ServerPeer,
        challenge: bytes,
        self_server_route: ServerRoute,
        verify_key: SyftVerifyKey,
    ) -> Request | SyftSuccess:
        """Add a Network Server Peer. Called by a remote server to add
        itself as a peer for the current server.
        """
        # Using the verify_key of the peer to verify the signature
        # It is also our single source of truth for the peer
        if peer.verify_key != context.credentials:
            raise SyftException(
                public_message=(
                    f"The {type(peer).__name__}.verify_key: "
                    f"{peer.verify_key} does not match the signature of the message"
                )
            )

        if verify_key != context.server.verify_key:
            raise SyftException(
                public_message="verify_key does not match the remote server's verify_key for add_peer"
            )

        # check if the peer already is a server peer
        existing_peer_res = self.stash.get_by_uid(context.server.verify_key, peer.id)

        if existing_peer_res.is_ok() and isinstance(
            existing_peer := existing_peer_res.ok(), ServerPeer
        ):
            msg = [
                f"The peer '{peer.name}' is already associated with '{context.server.name}'"
            ]

            if existing_peer != peer:
                msg.append("Peer information change detected.")
                self.stash.create_or_update_peer(
                    context.server.verify_key,
                    peer,
                ).unwrap(
                    public_message="\n".join(
                        msg + ["Attempt to update peer information failed."]
                    )
                )

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
            self_server_route=self_server_route, challenge=challenge, remote_peer=peer
        )
        submit_request = SubmitRequest(
            changes=[association_request_change],
            requesting_user_verify_key=context.credentials,
        )
        request = context.server.services.request.submit(context, submit_request)
        if (
            isinstance(request, Request)
            and context.server.settings.association_request_auto_approval
        ):
            return context.server.services.request.apply(context, uid=request.id)

        return request

    @service_method(path="network.ping", name="ping", roles=GUEST_ROLE_LEVEL)
    def ping(self, context: AuthedServiceContext, challenge: bytes) -> bytes:
        """To check alivesness/authenticity of a peer"""

        # # Only the root user can ping the server to check its state
        # if context.server.verify_key != context.credentials:
        #     return SyftError(message=("Only the root user can access ping endpoint"))

        # this way they can match up who we are with who they think we are
        # Sending a signed messages for the peer to verify

        challenge_signature = context.server.signing_key.signing_key.sign(
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
    ) -> ServerPeerAssociationStatus:
        """Check if a peer exists in the network stash"""

        # get the server peer for the given sender peer_id
        try:
            self.stash.get_by_uid(context.server.verify_key, peer_id).unwrap()
            return ServerPeerAssociationStatus.PEER_ASSOCIATED
        except SyftException:
            association_requests: list[Request] = (
                self._get_association_requests_by_peer_id(
                    context=context, peer_id=peer_id
                )
            )
            if (
                association_requests
                and association_requests[-1].status == RequestStatus.PENDING
            ):
                return ServerPeerAssociationStatus.PEER_ASSOCIATION_PENDING

        return ServerPeerAssociationStatus.PEER_NOT_FOUND

    @service_method(
        path="network.get_all_peers", name="get_all_peers", roles=GUEST_ROLE_LEVEL
    )
    def get_all_peers(self, context: AuthedServiceContext) -> list[ServerPeer]:
        """Get all Peers"""
        return self.stash.get_all(
            credentials=context.server.verify_key,
            order_by="name",
            sort_order="asc",
        ).unwrap()

    @service_method(
        path="network.get_peer_by_name", name="get_peer_by_name", roles=GUEST_ROLE_LEVEL
    )
    def get_peer_by_name(self, context: AuthedServiceContext, name: str) -> ServerPeer:
        """Get Peer by Name"""
        return self.stash.get_by_name(
            credentials=context.server.verify_key,
            name=name,
        ).unwrap()

    @service_method(
        path="network.get_peers_by_type",
        name="get_peers_by_type",
        roles=GUEST_ROLE_LEVEL,
    )
    def get_peers_by_type(
        self, context: AuthedServiceContext, server_type: ServerType
    ) -> list[ServerPeer]:
        return self.stash.get_by_server_type(
            credentials=context.server.verify_key,
            server_type=server_type,
        ).unwrap()

    @service_method(
        path="network.update_peer",
        name="update_peer",
        roles=GUEST_ROLE_LEVEL,
    )
    def update_peer(
        self,
        context: AuthedServiceContext,
        peer_update: ServerPeerUpdate,
    ) -> SyftSuccess:
        # try setting all fields of ServerPeerUpdate according to ServerPeer

        peer = self.stash.update(
            credentials=context.server.verify_key,
            obj=peer_update,
        ).unwrap()

        self.set_reverse_tunnel_config(context=context, remote_server_peer=peer)
        return SyftSuccess(
            message=f"Peer '{peer.name}' information successfully updated."
        )

    def set_reverse_tunnel_config(
        self,
        context: AuthedServiceContext,
        remote_server_peer: ServerPeer,
        self_server_peer: ServerPeer | None = None,
    ) -> None:
        server_type = cast(ServerType, context.server.server_type)
        if server_type.value == ServerType.GATEWAY.value:
            rtunnel_route = remote_server_peer.get_rtunnel_route()
            (
                self.rtunnel_service.set_server_config(remote_server_peer)
                if rtunnel_route
                else None
            )
        else:
            self_server_peer = (
                context.server.settings.to(ServerPeer)
                if self_server_peer is None
                else self_server_peer
            )
            rtunnel_route = self_server_peer.get_rtunnel_route()
            (
                self.rtunnel_service.set_client_config(
                    self_server_peer=self_server_peer,
                    remote_server_route=remote_server_peer.pick_highest_priority_route(),
                )
                if rtunnel_route
                else None
            )

    @service_method(
        path="network.delete_peer_by_id",
        name="delete_peer_by_id",
        roles=DATA_OWNER_ROLE_LEVEL,
        unwrap_on_success=False,
    )
    def delete_peer_by_id(self, context: AuthedServiceContext, uid: UID) -> SyftSuccess:
        """Delete Server Peer"""
        peer_to_delete = self.stash.get_by_uid(context.credentials, uid).unwrap()

        server_side_type = cast(ServerType, context.server.server_type)
        if server_side_type.value == ServerType.GATEWAY.value:
            rtunnel_route = peer_to_delete.get_rtunnel_route()
            (
                self.rtunnel_service.clear_server_config(peer_to_delete)
                if rtunnel_route
                else None
            )

        # TODO: Handle the case when peer is deleted from datasite server

        self.stash.delete_by_uid(context.credentials, uid).unwrap()

        # Delete all the association requests from this peer
        association_requests: list[Request] = self._get_association_requests_by_peer_id(
            context=context, peer_id=uid
        )
        for request in association_requests:
            context.server.services.request.delete_by_uid(context, request.id)
        # TODO: Notify the peer (either by email or by other form of notifications)
        # that it has been deleted from the network
        return SyftSuccess(message=f"Server Peer with id {uid} deleted.")

    @service_method(
        path="network.add_route_on_peer",
        name="add_route_on_peer",
        unwrap_on_success=False,
    )
    def add_route_on_peer(
        self,
        context: AuthedServiceContext,
        peer: ServerPeer,
        route: ServerRoute,
    ) -> SyftSuccess:
        """
        Add or update the route information on the remote peer.

        Args:
            context (AuthedServiceContext): The authentication context.
            peer (ServerPeer): The peer representing the remote server.
            route (ServerRoute): The route to be added.

        Returns:
            SyftSuccess: A success message if the route is verified,
                otherwise an error message.
        """
        # creates a client on the remote server based on the credentials
        # of the current server's client
        remote_client = peer.client_with_context(context=context).unwrap()
        # ask the remote server to add the route to the self server
        result = remote_client.api.services.network.add_route(
            peer_verify_key=context.credentials,
            route=route,
            called_by_peer=True,
        )

        return SyftSuccess(
            message="Route information added to remote peer", value=result
        )

    @service_method(
        path="network.add_route",
        name="add_route",
        roles=GUEST_ROLE_LEVEL,
        unwrap_on_success=False,
    )
    def add_route(
        self,
        context: AuthedServiceContext,
        peer_verify_key: SyftVerifyKey,
        route: ServerRoute,
        called_by_peer: bool = False,
    ) -> SyftSuccess:
        """
        Add a route to the peer. If the route already exists, update its priority.

        Args:
            context (AuthedServiceContext): The authentication context of the remote server.
            peer_verify_key (SyftVerifyKey): The verify key of the remote server peer.
            route (ServerRoute): The route to be added.
            called_by_peer (bool): The flag to indicate that it's called by a remote peer.

        Returns:
            SyftSuccess
        """
        # verify if the peer is truly the one sending the request to add the route to itself
        if called_by_peer and peer_verify_key != context.credentials:
            raise SyftException(
                public_message=(
                    f"The {type(peer_verify_key).__name__}: "
                    f"{peer_verify_key} does not match the signature of the message"
                )
            )
        # get the full peer object from the store to update its routes
        remote_server_peer: ServerPeer = (
            self._get_remote_server_peer_by_verify_key(context, peer_verify_key)
        ).unwrap()
        # add and update the priority for the peer
        if route in remote_server_peer.server_routes:
            return SyftSuccess(
                message=f"The route already exists between '{context.server.name}' and "
                f"peer '{remote_server_peer.name}'."
            )

        remote_server_peer.update_route(route=route)

        # update the peer in the store with the updated routes
        peer_update = ServerPeerUpdate(
            id=remote_server_peer.id, server_routes=remote_server_peer.server_routes
        )
        self.stash.update(
            credentials=context.server.verify_key,
            obj=peer_update,
        ).unwrap()

        return SyftSuccess(
            message=f"New route ({str(route)}) with id '{route.id}' "
            f"to peer {remote_server_peer.server_type.value} '{remote_server_peer.name}' "
            f"was added for {str(context.server.server_type)} '{context.server.name}'"
        )

    @service_method(
        path="network.delete_route_on_peer",
        name="delete_route_on_peer",
        unwrap_on_success=False,
    )
    def delete_route_on_peer(
        self,
        context: AuthedServiceContext,
        peer: ServerPeer,
        route: ServerRoute,
    ) -> SyftSuccess | SyftInfo:
        """
        Delete the route on the remote peer.

        Args:
            context (AuthedServiceContext): The authentication context for the service.
            peer (ServerPeer): The peer for which the route will be deleted.
            route (ServerRoute): The route to be deleted.

        Returns:
            SyftSuccess: If the route is successfully deleted.
            SyftInfo: If there is only one route left for the peer and
                the admin chose not to remove it
        """
        # creates a client on the remote server based on the credentials
        # of the current server's client
        remote_client = peer.client_with_context(context=context).unwrap()
        # ask the remote server to delete the route to the self server
        return remote_client.api.services.network.delete_route(
            peer_verify_key=context.credentials,
            route=route,
            called_by_peer=True,
        )

    @service_method(
        path="network.delete_route",
        name="delete_route",
        roles=GUEST_ROLE_LEVEL,
        unwrap_on_success=False,
    )
    def delete_route(
        self,
        context: AuthedServiceContext,
        peer_verify_key: SyftVerifyKey,
        route: ServerRoute | None = None,
        called_by_peer: bool = False,
    ) -> SyftSuccess | SyftInfo:
        """
        Delete a route for a given peer.
        If a peer has no routes left, there will be a prompt asking if the user want to remove it.
        If the answer is yes, it will be removed from the stash and will no longer be a peer.

        Args:
            context (AuthedServiceContext): The authentication context for the service.
            peer_verify_key (SyftVerifyKey): The verify key of the remote server peer.
            route (ServerRoute): The route to be deleted.
            called_by_peer (bool): The flag to indicate that it's called by a remote peer.

        Returns:
            SyftSuccess: If the route is successfully deleted.
            SyftInfo: If there is only one route left for the peer and
                the admin chose not to remove it
        """
        if called_by_peer and peer_verify_key != context.credentials:
            # verify if the peer is truly the one sending the request to delete the route to itself
            raise SyftException(
                public_message=(
                    f"The {type(peer_verify_key).__name__}: "
                    f"{peer_verify_key} does not match the signature of the message"
                )
            )
        remote_server_peer: ServerPeer = self._get_remote_server_peer_by_verify_key(
            context=context, peer_verify_key=peer_verify_key
        ).unwrap()

        if len(remote_server_peer.server_routes) == 1:
            warning_message = (
                f"There is only one route left to peer "
                f"{remote_server_peer.server_type.value} '{remote_server_peer.name}'. "
                f"Removing this route will remove the peer for "
                f"{str(context.server.server_type)} '{context.server.name}'."
            )
            response: bool = prompt_warning_message(
                message=warning_message,
                confirm=False,
            )
            if not response:
                return SyftInfo(
                    message=f"The last route to {remote_server_peer.server_type.value} "
                    f"'{remote_server_peer.name}' with id "
                    f"'{remote_server_peer.server_routes[0].id}' was not deleted."
                )

        if route is not None:
            remote_server_peer.delete_route(route)

        return_message = (
            f"Route '{str(route)}' to peer "
            f"{remote_server_peer.server_type.value} '{remote_server_peer.name}' "
            f"was deleted for {str(context.server.server_type)} '{context.server.name}'."
        )

        if len(remote_server_peer.server_routes) == 0:
            # remove the peer
            # TODO: should we do this as we are deleting the peer with a guest role level?
            self.stash.delete_by_uid(
                credentials=context.server.verify_key, uid=remote_server_peer.id
            ).unwrap()
            return_message += (
                f" There is no routes left to connect to peer "
                f"{remote_server_peer.server_type.value} '{remote_server_peer.name}', so it is deleted for "
                f"{str(context.server.server_type)} '{context.server.name}'."
            )
        else:
            # update the peer with the route removed
            peer_update = ServerPeerUpdate(
                id=remote_server_peer.id, server_routes=remote_server_peer.server_routes
            )
            self.stash.update(
                credentials=context.server.verify_key, obj=peer_update
            ).unwrap()

        return SyftSuccess(message=return_message)

    @service_method(
        path="network.update_route_priority_on_peer",
        name="update_route_priority_on_peer",
        unwrap_on_success=False,
    )
    def update_route_priority_on_peer(
        self,
        context: AuthedServiceContext,
        peer: ServerPeer,
        route: ServerRoute,
        priority: int | None = None,
    ) -> SyftSuccess:
        """
        Update the route priority on the remote peer.

        Args:
            context (AuthedServiceContext): The authentication context.
            peer (ServerPeer): The peer representing the remote server.
            route (ServerRoute): The route to be added.
            priority (int | None): The new priority value for the route. If not
                provided, it will be assigned the highest priority among all peers

        Returns:
            SyftSuccess: A success message if the route is verified,
                otherwise an error message.
        """
        # creates a client on the remote server based on the credentials
        # of the current server's client
        remote_client = peer.client_with_context(context=context).unwrap()
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
        unwrap_on_success=False,
    )
    def update_route_priority(
        self,
        context: AuthedServiceContext,
        peer_verify_key: SyftVerifyKey,
        route: ServerRoute,
        priority: int | None = None,
        called_by_peer: bool = False,
    ) -> SyftSuccess:
        """
        Updates a route's priority for the given peer

        Args:
            context (AuthedServiceContext): The authentication context for the service.
            peer_verify_key (SyftVerifyKey): The verify key of the peer whose route priority needs to be updated.
            route (ServerRoute): The route for which the priority needs to be updated.
            priority (int | None): The new priority value for the route. If not
                provided, it will be assigned the highest priority among all peers

        Returns:
            SyftSuccess : Successful response
        """
        if called_by_peer and peer_verify_key != context.credentials:
            raise SyftException(
                public_message=(
                    f"The {type(peer_verify_key).__name__}: "
                    f"{peer_verify_key} does not match the signature of the message"
                )
            )
        # get the full peer object from the store to update its routes
        remote_server_peer: ServerPeer = (
            self._get_remote_server_peer_by_verify_key(context, peer_verify_key)
        ).unwrap()
        # update the route's priority for the peer
        updated_server_route: ServerRouteType = (
            remote_server_peer.update_existed_route_priority(
                route=route, priority=priority
            ).unwrap()
        )
        new_priority: int = updated_server_route.priority
        # update the peer in the store
        peer_update = ServerPeerUpdate(
            id=remote_server_peer.id, server_routes=remote_server_peer.server_routes
        )
        self.stash.update(context.server.verify_key, peer_update).unwrap()
        return SyftSuccess(
            message=f"Route {route.id}'s priority updated to "
            f"{new_priority} for peer {remote_server_peer.name}"
        )

    @as_result(SyftException)
    def _get_remote_server_peer_by_verify_key(
        self, context: AuthedServiceContext, peer_verify_key: SyftVerifyKey
    ) -> ServerPeer:
        """
        Helper function to get the full server peer object from t
        he stash using its verify key
        """
        return self.stash.get_by_verify_key(
            credentials=context.server.verify_key,
            verify_key=peer_verify_key,
        ).unwrap()

    def _get_association_requests_by_peer_id(
        self, context: AuthedServiceContext, peer_id: UID
    ) -> list[Request]:
        """
        Get all the association requests from a peer. The association requests are sorted by request_time.
        """
        all_requests: list[Request] = context.server.services.request.get_all(context)
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


TYPE_TO_SERVICE[ServerPeer] = NetworkService
SERVICE_TO_TYPES[NetworkService].update({ServerPeer})


def from_server_url(context: TransformContext) -> TransformContext:
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


@transform(HTTPConnection, HTTPServerRoute)
def http_connection_to_server_route() -> list[Callable]:
    return [from_server_url]


def get_python_server_route(context: TransformContext) -> TransformContext:
    if context.output is not None and context.obj is not None:
        context.output["id"] = context.obj.server.id
        context.output["worker_settings"] = WorkerSettings.from_server(
            context.obj.server
        )
        context.output["proxy_target_uid"] = context.obj.proxy_target_uid
    return context


@transform(PythonConnection, PythonServerRoute)
def python_connection_to_server_route() -> list[Callable]:
    return [get_python_server_route]


@transform_method(PythonServerRoute, PythonConnection)
def server_route_to_python_connection(
    obj: Any, context: TransformContext | None = None
) -> list[Callable]:
    return PythonConnection(server=obj.server, proxy_target_uid=obj.proxy_target_uid)


@transform_method(HTTPServerRoute, HTTPConnection)
def server_route_to_http_connection(
    obj: Any, context: TransformContext | None = None
) -> list[Callable]:
    url = ServerURL(
        protocol=obj.protocol, host_or_ip=obj.host_or_ip, port=obj.port
    ).as_container_host()
    return HTTPConnection(
        url=url,
        proxy_target_uid=obj.proxy_target_uid,
        rtunnel_token=obj.rtunnel_token,
    )


@transform(ServerMetadata, ServerPeer)
def metadata_to_peer() -> list[Callable]:
    return [
        keep(
            [
                "id",
                "name",
                "verify_key",
                "server_type",
            ]
        ),
        make_set_default("admin_email", ""),
    ]


@transform(ServerSettings, ServerPeer)
def settings_to_peer() -> list[Callable]:
    return [
        keep(["id", "name", "verify_key", "server_type", "admin_email"]),
    ]
